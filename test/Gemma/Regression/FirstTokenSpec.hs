{-# LANGUAGE OverloadedStrings #-}

module Gemma.Regression.FirstTokenSpec (spec) where

import Test.Hspec
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import Graphics.WebGPU.Dawn.ContT (evalContT)
import Data.List (sortBy)
import Data.Ord (Down(..), comparing)
import Data.Aeson (FromJSON(..), withObject, (.:))
import Control.Monad (forM_, foldM)

import Gemma.Model
import Gemma.Tokenizer
import Gemma.TestCache (loadOrGenerateCache)

data FirstTokenReference = FirstTokenReference
  { ftrPrompt :: String
  , ftrChatPrompt :: String
  , ftrInputTokens :: [Int]
  , ftrTop10Tokens :: [Int]
  , ftrTop10Logits :: [Float]
  , ftrExpectedFirstToken :: Int
  , ftrExpectedFirstLogit :: Float
  , ftrExpectedFirstRank :: Int
  , ftrVocabSize :: Int
  } deriving (Show)

instance FromJSON FirstTokenReference where
  parseJSON = withObject "FirstTokenReference" $ \v -> FirstTokenReference
    <$> v .: "prompt"
    <*> v .: "chat_prompt"
    <*> v .: "input_tokens"
    <*> v .: "top_10_tokens"
    <*> v .: "top_10_logits"
    <*> v .: "expected_first_token"
    <*> v .: "expected_first_logit"
    <*> v .: "expected_first_rank"
    <*> v .: "vocab_size"

spec :: Spec
spec = do
  describe "First Token Generation (TDD PyTorch Comparison)" $ do

    it "first token logits match PyTorch top-10" $ do
      -- Load PyTorch reference
      ref <- loadOrGenerateCache
        "test/Gemma/Regression/FirstTokenSpec_pytorch.py"
        "test/Gemma/Regression/FirstTokenSpec_reference.json" :: IO FirstTokenReference

      putStrLn $ "\n=== First Token Logits Comparison ==="
      putStrLn $ "Prompt: " ++ ftrPrompt ref
      putStrLn $ "Input tokens: " ++ show (ftrInputTokens ref)
      putStrLn $ "\nPyTorch Top 10:"
      forM_ (zip [1..] $ zip (ftrTop10Tokens ref) (ftrTop10Logits ref)) $ \(i, (tok, logit)) -> do
        putStrLn $ "  " ++ show i ++ ". Token " ++ show tok ++ " (logit=" ++ show logit ++ ")"

      -- Load Haskell model
      let modelPath = "../models/gemma3-1b-official-instruct/model.safetensors"
          tokenizerPath = "../models/gemma3-1b-official-instruct/tokenizer.model"
          config = gemma3_1BConfig  -- Use default Gemma 3 1B config

      putStrLn "\n📦 Loading model..."
      model <- loadGemmaModel modelPath config

      -- Run inference on prompt tokens (all 17 tokens)
      let inputTokens = V.fromList $ map fromIntegral $ ftrInputTokens ref

      putStrLn $ "\n🚀 Running Haskell inference on " ++ show (V.length inputTokens) ++ " prompt tokens..."

      -- Process tokens one-by-one with KV cache (like autoregressive generation)
      -- Start with no cache, then accumulate cache for each token
      (logits, _) <- foldM
        (\(_, mCache) tok -> do
          let singleToken = V.singleton tok
          (logits', cache') <- runGemmaInferenceCached model singleToken mCache
          pure (logits', Just cache')
        )
        (V.empty, Nothing)  -- Initial: no logits, no cache
        (V.toList inputTokens)

      putStrLn $ "✅ Got logits vector of length: " ++ show (V.length logits)

      -- Get top 10 from Haskell
      let logitsList = V.toList logits
          indexed = zip [0..] logitsList
          sorted = sortBy (comparing (Down . snd)) indexed
          top10 = take 10 sorted

      putStrLn $ "\nHaskell Top 10:"
      forM_ (zip [1..] top10) $ \(i, (tok, logit)) -> do
        putStrLn $ "  " ++ show i ++ ". Token " ++ show tok ++ " (logit=" ++ show logit ++ ")"

      -- Check if PyTorch's top token matches
      let pyTop1 = head $ ftrTop10Tokens ref
          pyTop1Logit = head $ ftrTop10Logits ref
          hsTop1 = fst $ head top10
          hsTop1Logit = snd $ head top10

      putStrLn $ "\n📊 Comparison:"
      putStrLn $ "  PyTorch top-1: Token " ++ show pyTop1 ++ " (logit=" ++ show pyTop1Logit ++ ")"
      putStrLn $ "  Haskell top-1: Token " ++ show hsTop1 ++ " (logit=" ++ show hsTop1Logit ++ ")"

      if hsTop1 == pyTop1
        then putStrLn "  ✅ MATCH! Haskell predicts same token as PyTorch"
        else do
          putStrLn "  ❌ MISMATCH! Haskell predicts different token"
          -- Check if PyTorch's top token is in Haskell's top 10
          let hsTop10Tokens = map fst top10
          if pyTop1 `elem` hsTop10Tokens
            then do
              let rank = length (takeWhile (/= pyTop1) hsTop10Tokens) + 1
              putStrLn $ "  PyTorch's top token is rank " ++ show rank ++ " in Haskell"
            else putStrLn $ "  PyTorch's top token NOT in Haskell's top 10!"

      -- Test: Top-1 token should match
      hsTop1 `shouldBe` pyTop1
