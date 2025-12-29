{-# LANGUAGE OverloadedStrings #-}

module Gemma.Regression.MultiTokenPromptSpec (spec) where

import Test.Hspec
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import qualified Data.Aeson as A
import qualified Data.ByteString.Lazy as BL
import Data.Maybe (fromJust)

import Gemma.Model
import Gemma.SafeTensors
import Gemma.KVCache (KVCache)

-- Reference data from PyTorch
data PromptReference = PromptReference
  { promptTokens :: [Int]
  , topToken :: Int
  , topLogit :: Float
  , top10Tokens :: [Int]
  , top10Logits :: [Float]
  } deriving (Show)

instance A.FromJSON PromptReference where
  parseJSON = A.withObject "PromptReference" $ \v -> PromptReference
    <$> v A..: "prompt_tokens"
    <*> v A..: "top_token"
    <*> v A..: "top_logit"
    <*> v A..: "top_10_tokens"
    <*> v A..: "top_10_logits"

spec :: Spec
spec = describe "Multi-Token Prompt (TDD PyTorch Comparison)" $ do

  it "Q4 model predicts correct token after multi-token prompt (sequential processing)" $ do
    -- Load PyTorch reference
    refData <- BL.readFile "/tmp/prompt_reference.json"
    let ref = fromJust $ A.decode refData :: PromptReference

    putStrLn "\n=== PyTorch Reference ==="
    putStrLn $ "Prompt tokens (" ++ show (length (promptTokens ref)) ++ "): " ++ show (promptTokens ref)
    putStrLn $ "Top predicted token: " ++ show (topToken ref) ++ " (logit=" ++ show (topLogit ref) ++ ")"
    putStrLn $ "Top 10 tokens: " ++ show (top10Tokens ref)

    -- Load Q4 model
    putStrLn "\n=== Loading Haskell Q4 Model ==="
    let modelPath = "../models/gemma3-1b-q4.safetensors"
    model <- loadGemmaModel modelPath gemma3_1BConfig

    -- Process prompt tokens sequentially (as in autoregressiveGenerateStreaming)
    putStrLn $ "\n=== Processing " ++ show (length (promptTokens ref)) ++ " prompt tokens sequentially ==="
    logits <- processPromptTokens model (promptTokens ref) Nothing

    -- Get top prediction
    let maxIdx = V.maxIndex logits
        maxLogit = logits V.! maxIdx

    putStrLn "\n=== Haskell Prediction ==="
    putStrLn $ "Top predicted token: " ++ show maxIdx ++ " (logit=" ++ show maxLogit ++ ")"

    -- Get top 10 for comparison
    let logitsList = V.toList logits
        indexedLogits = zip [0..] logitsList
        haskellTop10 = map fst $ take 10 $ reverse $
          sortBy (\(_, a) (_, b) -> compare a b) indexedLogits

    putStrLn $ "Haskell top 10 tokens: " ++ show haskellTop10

    putStrLn "\n=== Comparison ==="
    if maxIdx == topToken ref
      then putStrLn $ "✅ MATCH! Haskell predicts same token as PyTorch: " ++ show maxIdx
      else do
        putStrLn $ "❌ MISMATCH!"
        putStrLn $ "  PyTorch: " ++ show (topToken ref) ++ " (logit=" ++ show (topLogit ref) ++ ")"
        putStrLn $ "  Haskell: " ++ show maxIdx ++ " (logit=" ++ show maxLogit ++ ")"
        putStrLn $ "\nPyTorch top 10: " ++ show (top10Tokens ref)
        putStrLn $ "Haskell top 10: " ++ show haskellTop10

    -- Test assertion
    maxIdx `shouldBe` topToken ref

-- Process prompt tokens sequentially using runGemmaInferenceCached
-- Returns the logits from the LAST token only
processPromptTokens :: GemmaModel dtype -> [Int] -> Maybe KVCache -> IO (Vector Float)
processPromptTokens model [] _cache = error "Empty prompt"
processPromptTokens model [lastToken] cache = do
  putStrLn $ "Processing last prompt token: " ++ show lastToken
  (logits, _finalCache) <- runGemmaInferenceCached model (V.singleton lastToken) cache
  return logits
processPromptTokens model (token:rest) cache = do
  putStrLn $ "Processing prompt token: " ++ show token
  (_logits, newCache) <- runGemmaInferenceCached model (V.singleton token) cache
  processPromptTokens model rest (Just newCache)

sortBy :: (a -> a -> Ordering) -> [a] -> [a]
sortBy cmp = foldr insert []
  where
    insert x [] = [x]
    insert x (y:ys) = case cmp x y of
      GT -> y : insert x ys
      _  -> x : y : ys
