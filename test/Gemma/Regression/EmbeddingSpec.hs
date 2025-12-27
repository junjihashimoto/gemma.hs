{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

module Gemma.Regression.EmbeddingSpec (spec) where

import Test.Hspec
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import Data.Aeson (FromJSON(..), withObject, (.:))
import Control.Monad (forM_)

import Gemma.Model
import Gemma.Layers.Embedding (runEmbedding)
import Gemma.TestCache (loadOrGenerateCache)
import Graphics.WebGPU.Dawn.ContT (evalContT)

-- Layer output statistics from PyTorch
data LayerStats = LayerStats
  { lsMean :: Float
  , lsStd :: Float
  , lsMin :: Float
  , lsMax :: Float
  , lsFirst10 :: [Float]
  , lsLast10 :: [Float]
  } deriving (Show)

instance FromJSON LayerStats where
  parseJSON = withObject "LayerStats" $ \v -> LayerStats
    <$> v .: "mean"
    <*> v .: "std"
    <*> v .: "min"
    <*> v .: "max"
    <*> v .: "first_10"
    <*> v .: "last_10"

data LayerOutputs = LayerOutputs
  { loEmbeddings :: LayerStats
  , loLayer0 :: LayerStats
  , loLayer1 :: LayerStats
  } deriving (Show)

instance FromJSON LayerOutputs where
  parseJSON = withObject "LayerOutputs" $ \v -> LayerOutputs
    <$> v .: "embeddings"
    <*> v .: "layer_0"
    <*> v .: "layer_1"

data LayerByLayerReference = LayerByLayerReference
  { lbrPrompt :: String
  , lbrInputTokens :: [Int]
  , lbrNumLayers :: Int
  , lbrLayerOutputs :: LayerOutputs
  } deriving (Show)

instance FromJSON LayerByLayerReference where
  parseJSON = withObject "LayerByLayerReference" $ \v -> LayerByLayerReference
    <$> v .: "prompt"
    <*> v .: "input_tokens"
    <*> v .: "num_layers"
    <*> v .: "layer_outputs"

-- Compare vector statistics with reference
compareStats :: String -> Vector Float -> LayerStats -> IO ()
compareStats name vec ref = do
  let mean = V.sum vec / fromIntegral (V.length vec)
      variance = V.sum (V.map (\x -> (x - mean) * (x - mean)) vec) / fromIntegral (V.length vec)
      std = sqrt variance
      minVal = V.minimum vec
      maxVal = V.maximum vec
      first10 = V.toList $ V.take 10 vec
      last10 = V.toList $ V.drop (V.length vec - 10) vec

  putStrLn $ "\n📊 " ++ name
  putStrLn "  PyTorch vs Haskell:"
  putStrLn $ "    Mean:  " ++ show (lsMean ref) ++ " vs " ++ show mean
  putStrLn $ "    Std:   " ++ show (lsStd ref) ++ " vs " ++ show std
  putStrLn $ "    Min:   " ++ show (lsMin ref) ++ " vs " ++ show minVal
  putStrLn $ "    Max:   " ++ show (lsMax ref) ++ " vs " ++ show maxVal

  putStrLn "  First 10 values:"
  putStrLn $ "    PyTorch: " ++ show (lsFirst10 ref)
  putStrLn $ "    Haskell: " ++ show first10

  -- Check if they're close (within 1% tolerance)
  let meanDiff = abs (mean - lsMean ref)
      meanTolerance = abs (lsMean ref) * 0.01  -- 1% tolerance
      meansMatch = meanDiff < meanTolerance || meanDiff < 0.01  -- absolute tolerance for near-zero means

  if meansMatch
    then putStrLn "  ✅ Statistics MATCH (within 1% tolerance)"
    else putStrLn $ "  ❌ Statistics DIVERGE! Mean diff: " ++ show meanDiff ++ " (tolerance: " ++ show meanTolerance ++ ")"

spec :: Spec
spec = do
  describe "Embedding Layer Comparison" $ do

    it "embeddings match PyTorch (last token after scaling)" $ do
      -- Load PyTorch reference
      ref <- loadOrGenerateCache
        "test/Gemma/Regression/FirstTokenSpec_layerByLayer.py"
        "test/Gemma/Regression/FirstTokenSpec_layerByLayer.json" :: IO LayerByLayerReference

      putStrLn $ "\n=== Embedding Layer Comparison ==="
      putStrLn $ "Prompt: " ++ lbrPrompt ref
      putStrLn $ "Input tokens: " ++ show (lbrInputTokens ref)
      putStrLn $ "Number of layers: " ++ show (lbrNumLayers ref)

      -- Load Haskell model
      let modelPath = "../models/gemma3-1b-official-instruct/model.safetensors"
          config = gemma3_1BConfig

      putStrLn "\n📦 Loading model..."
      model <- loadGemmaModel modelPath config

      -- Process all 17 tokens to get embeddings for the last token
      let inputTokens = V.fromList $ map fromIntegral $ lbrInputTokens ref
          lastTokenId = V.singleton (V.last inputTokens)

      putStrLn $ "\n🚀 Getting embedding for last token (token " ++ show (V.last inputTokens) ++ ")..."

      -- Get embedding using the model's embedding function
      -- We need to extract the embedding AFTER the sqrt(hidden_dim) scaling
      -- Let's run a single-token inference and capture the embedding
      let GemmaModel{..} = model
          GemmaConfig{..} = gmConfig

      -- Run embedding with scaling (like in runGemmaInferenceCached)
      rawEmbedding <- evalContT $ runEmbedding lastTokenId gmEmbeddings gcVocabSize gcHiddenDim

      -- Apply same scaling as in Model.hs (Gemma 3 requirement)
      let embeddingScale = sqrt (fromIntegral gcHiddenDim :: Double)
          scaledEmbedding = V.map (\x -> realToFrac (realToFrac x * embeddingScale :: Double)) rawEmbedding

      putStrLn $ "✅ Got scaled embedding of length: " ++ show (V.length scaledEmbedding)

      -- Compare with PyTorch
      compareStats "Embeddings (last token, after scaling)" scaledEmbedding (loEmbeddings $ lbrLayerOutputs ref)

      -- Check that first 10 values match (within tolerance)
      let hsFirst10 = V.toList $ V.take 10 scaledEmbedding
          pyFirst10 = lsFirst10 $ loEmbeddings $ lbrLayerOutputs ref
          maxDiff = maximum $ zipWith (\a b -> abs (a - b)) hsFirst10 pyFirst10
          tolerance = 0.01  -- 1% tolerance

      putStrLn $ "\n📏 Max difference in first 10 values: " ++ show maxDiff

      maxDiff `shouldSatisfy` (< tolerance)
