{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE OverloadedStrings #-}
-- TDD test for FP32 single token inference
-- Compares with PyTorch reference from FP32InferenceSpec_singleToken.py

module Gemma.Regression.FP32InferenceSpec (spec) where

import Test.Hspec
import qualified Data.Vector.Storable as V
import Data.List (sortBy)
import Data.Ord (Down(..))
import Gemma.Model (loadGemmaModel, runGemmaInferenceCached, gemma3_1BConfig, GemmaConfig(..), GemmaModel(..))
import Gemma.TestCache (loadOrGenerateCache)
import qualified Data.Aeson as JSON
import Data.Aeson ((.:))

data FP32Reference = FP32Reference
  { tokenId :: Int
  , top5Tokens :: [Int]
  , top5Logits :: [Float]
  } deriving (Show)

instance JSON.FromJSON FP32Reference where
  parseJSON = JSON.withObject "FP32Reference" $ \v -> FP32Reference
    <$> v .: "token_id"
    <*> v .: "top5_tokens"
    <*> v .: "top5_logits"

spec :: Spec
spec = describe "FP32 Single Token Inference" $ do
  it "generates correct logits for token 6974 matching PyTorch" $ do
    -- Load PyTorch reference (auto-regenerate if stale)
    ref <- loadOrGenerateCache
      "test/Gemma/Regression/FP32InferenceSpec_singleToken.py"
      "test/Gemma/Regression/FP32InferenceSpec_singleToken.json"

    putStrLn $ "\n=== FP32 Single Token Test ==="
    putStrLn $ "Token ID: " ++ show (tokenId ref)
    putStrLn $ "PyTorch expects top token: " ++ show (head $ top5Tokens ref)
    putStrLn $ "PyTorch top 5 logits: " ++ show (top5Logits ref)

    -- Load Gemma 3 FP32 model (original model WITHOUT QK-norm)
    -- NOTE: Using the ORIGINAL model that Q4 was derived from, NOT the official FP32 one!
    let config = gemma3_1BConfig {
          gcUseFP16 = False,
          gcUseQKNorm = False,  -- Original model doesn't have QK-norm
          gcUsePostAttnNorm = False,  -- Disable all Gemma 3 specific features
          gcUsePostFFNNorm = False
        }
    putStrLn $ "\nLoading Gemma 3 FP32 model..."
    putStrLn $ "  Path: ../models/gemma3-1b.safetensors (ORIGINAL model, not official)"
    putStrLn $ "  Config: gcUseFP16=" ++ show (gcUseFP16 config) ++ ", gcUseQKNorm=" ++ show (gcUseQKNorm config)
    model <- loadGemmaModel "../models/gemma3-1b.safetensors" config

    -- Run inference on token 6974
    putStrLn $ "Running inference on token " ++ show (tokenId ref) ++ "..."
    let tokenVec = V.singleton (tokenId ref)

    -- DEBUG: Check embedding lookup
    putStrLn $ "\nDEBUG: Checking embedding..."
    let embTable = gmEmbeddings model
        vocabSize = gcVocabSize (gmConfig model)
        hiddenDim = gcHiddenDim (gmConfig model)
        tokenIdx = tokenId ref
        embOffset = tokenIdx * hiddenDim
        tokenEmb = V.slice embOffset hiddenDim embTable
    putStrLn $ "  Embedding (first 10): " ++ show (V.toList $ V.take 10 tokenEmb)

    (logits, _) <- runGemmaInferenceCached model tokenVec Nothing

    -- Check for NaN
    let hasNaN = V.any isNaN logits
    putStrLn $ "Has NaN: " ++ show hasNaN
    hasNaN `shouldBe` False

    -- Get top 5 tokens from Haskell
    let logitsList = V.toList logits
        indexed = zip [0..] logitsList
        sorted = sortBy (comparing (Down . snd)) indexed
        top5 = take 5 sorted
        (topToken, topLogit) = head top5

    putStrLn $ "\nHaskell top 5:"
    mapM_ (\(idx, val) -> putStrLn $ "  Token " ++ show idx ++ ": " ++ show val) top5

    putStrLn $ "\nPyTorch top 5:"
    mapM_ (\(idx, val) -> putStrLn $ "  Token " ++ show idx ++ ": " ++ show val)
      (zip (top5Tokens ref) (top5Logits ref))

    -- Check specific logits
    let logit68 = logits V.! 68
        logit524 = logits V.! 524
    putStrLn $ "\nSpecific logits:"
    putStrLn $ "  Token 68:  Haskell=" ++ show logit68 ++ ", PyTorch=-18.89"
    putStrLn $ "  Token 524: Haskell=" ++ show logit524 ++ ", PyTorch=48.38"

    -- Main assertion: top token should match
    topToken `shouldBe` (head $ top5Tokens ref)

    -- Tolerance check for logit value (±0.1)
    abs (topLogit - head (top5Logits ref)) `shouldSatisfy` (< 0.1)

comparing :: Ord b => (a -> b) -> a -> a -> Ordering
comparing f x y = compare (f x) (f y)
