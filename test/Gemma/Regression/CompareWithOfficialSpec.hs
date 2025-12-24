{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

module Gemma.Regression.CompareWithOfficialSpec (spec) where

import Test.Hspec
import qualified Data.Vector.Storable as V
import qualified Data.Aeson as A
import Data.Aeson ((.=), (.:))
import Data.Aeson.Types (Parser)
import qualified Data.ByteString.Lazy as BL
import Data.Maybe (fromMaybe)
import System.IO.Unsafe (unsafePerformIO)

import Gemma.Model
import Gemma.Tokenizer
import Gemma.SafeTensors

-- Reference data from official model
data OfficialReference = OfficialReference
  { refTokens :: [Int]
  , refTopToken :: Int
  , refTopLogit :: Double
  , refLayerOutputs :: [LayerOutput]
  , refConfig :: ModelConfig
  } deriving (Show)

data LayerOutput = LayerOutput
  { layerName :: String
  , layerMean :: Double
  , layerStd :: Double
  , layerMin :: Double
  , layerMax :: Double
  , layerFirst10 :: [Double]
  , layerLast10 :: [Double]
  } deriving (Show)

data ModelConfig = ModelConfig
  { cfgNumLayers :: Int
  , cfgHiddenSize :: Int
  , cfgNumAttentionHeads :: Int
  , cfgNumKeyValueHeads :: Int
  , cfgHeadDim :: Int
  , cfgIntermediateSize :: Int
  , cfgRopeTheta :: Double
  } deriving (Show)

instance A.FromJSON OfficialReference where
  parseJSON = A.withObject "OfficialReference" $ \v -> do
    refTokens <- v A..: "tokens"
    predictions <- v A..: "predictions"
    let topPred = head predictions
    refTopToken <- topPred A..: "token"
    refTopLogit <- topPred A..: "logit"

    layerOutputsRaw <- v A..: "layer_outputs"
    refLayerOutputs <- mapM parseLayerOutput layerOutputsRaw

    config <- v A..: "config"
    refConfig <- parseModelConfig config

    return OfficialReference{..}

parseLayerOutput :: A.Value -> Parser LayerOutput
parseLayerOutput = A.withObject "LayerOutput" $ \v -> do
  layer <- v A..: "layer"
  let layerName = case layer of
        A.String s -> show s
        A.Number n -> "Layer " ++ show (floor n :: Int)
        _ -> "Unknown"
  layerMean <- v A..: "mean"
  layerStd <- v A..: "std"
  layerMin <- v A..: "min"
  layerMax <- v A..: "max"
  layerFirst10 <- v A..: "first_10"
  layerLast10 <- v A..: "last_10"
  return LayerOutput{..}

parseModelConfig :: A.Value -> Parser ModelConfig
parseModelConfig = A.withObject "ModelConfig" $ \v -> do
  cfgNumLayers <- v A..: "num_layers"
  cfgHiddenSize <- v A..: "hidden_size"
  cfgNumAttentionHeads <- v A..: "num_attention_heads"
  cfgNumKeyValueHeads <- v A..: "num_key_value_heads"
  cfgHeadDim <- v A..: "head_dim"
  cfgIntermediateSize <- v A..: "intermediate_size"
  cfgRopeTheta <- v A..: "rope_theta"
  return ModelConfig{..}

-- Load reference data
loadOfficialReference :: IO OfficialReference
loadOfficialReference = do
  let refPath = "test/Gemma/Regression/OfficialReference_prompt17.json"
  putStrLn $ "\n📖 Loading official reference: " ++ refPath
  content <- BL.readFile refPath
  case A.eitherDecode content of
    Left err -> error $ "Failed to parse reference: " ++ err
    Right ref -> do
      putStrLn $ "✅ Loaded reference with " ++ show (length (refLayerOutputs ref)) ++ " layers"
      return ref

-- Calculate vector statistics
vectorStats :: V.Vector Float -> (Double, Double, Double, Double)
vectorStats v =
  let vals = V.toList v
      n = fromIntegral (V.length v)
      mean = sum (map realToFrac vals) / n
      variance = sum (map (\x -> (realToFrac x - mean) ** 2) vals) / n
      std = sqrt variance
      vmin = realToFrac (V.minimum v)
      vmax = realToFrac (V.maximum v)
  in (mean, std, vmin, vmax)

-- Compare two floating point numbers with tolerance
approxEqual :: Double -> Double -> Double -> Bool
approxEqual tolerance expected actual =
  abs (expected - actual) < tolerance

-- Compare vector stats with official reference
compareStats :: String -> LayerOutput -> V.Vector Float -> IO Bool
compareStats layerName ref output = do
  let (mean, std, vmin, vmax) = vectorStats output
      first10 = map realToFrac $ V.toList $ V.take 10 output
      last10 = map realToFrac $ V.toList $ V.drop (V.length output - 10) output

  putStrLn $ "\n  " ++ layerName ++ ":"
  putStrLn $ "    Mean:  expected=" ++ show (layerMean ref) ++ " actual=" ++ show mean
  putStrLn $ "    Std:   expected=" ++ show (layerStd ref) ++ " actual=" ++ show std
  putStrLn $ "    Min:   expected=" ++ show (layerMin ref) ++ " actual=" ++ show vmin
  putStrLn $ "    Max:   expected=" ++ show (layerMax ref) ++ " actual=" ++ show vmax

  -- Check if stats match within tolerance
  let tolerance = 0.01  -- 1% tolerance
      meanMatch = approxEqual (tolerance * abs (layerMean ref)) (layerMean ref) mean
      stdMatch = approxEqual (tolerance * abs (layerStd ref)) (layerStd ref) std

  if meanMatch && stdMatch
    then do
      putStrLn "    Status: ✅ MATCH"
      return True
    else do
      putStrLn "    Status: ❌ MISMATCH"
      putStrLn "    First 10 (expected):"
      putStrLn $ "      " ++ show (layerFirst10 ref)
      putStrLn "    First 10 (actual):"
      putStrLn $ "      " ++ show first10
      return False

spec :: Spec
spec = describe "Compare with Official Gemma 3 Reference" $ do

  it "matches official model config" $ do
    putStrLn "\n=== Comparing Model Configuration ==="
    ref <- loadOfficialReference

    let cfg = gemma3_1BConfig
        config = refConfig ref

    putStrLn $ "\nOfficial config:"
    putStrLn $ "  Layers: " ++ show (cfgNumLayers config)
    putStrLn $ "  Hidden size: " ++ show (cfgHiddenSize config)
    putStrLn $ "  Attention heads: " ++ show (cfgNumAttentionHeads config)
    putStrLn $ "  KV heads: " ++ show (cfgNumKeyValueHeads config)
    putStrLn $ "  Head dim: " ++ show (cfgHeadDim config)
    putStrLn $ "  FFN size: " ++ show (cfgIntermediateSize config)
    putStrLn $ "  RoPE base: " ++ show (cfgRopeTheta config)

    putStrLn $ "\nOur config:"
    putStrLn $ "  Layers: " ++ show (gcNumLayers cfg)
    putStrLn $ "  Hidden size: " ++ show (gcHiddenDim cfg)
    putStrLn $ "  Attention heads: " ++ show (gcNumHeads cfg)
    putStrLn $ "  KV heads: " ++ show (gcNumKVHeads cfg)
    putStrLn $ "  Head dim: " ++ show (gcHeadDim cfg)
    putStrLn $ "  FFN size: " ++ show (gcFFNDim cfg)
    putStrLn $ "  RoPE base: " ++ show (gcRopeBase cfg)

    gcNumLayers cfg `shouldBe` cfgNumLayers config
    gcHiddenDim cfg `shouldBe` cfgHiddenSize config
    gcNumHeads cfg `shouldBe` cfgNumAttentionHeads config
    gcNumKVHeads cfg `shouldBe` cfgNumKeyValueHeads config
    gcHeadDim cfg `shouldBe` cfgHeadDim config
    gcFFNDim cfg `shouldBe` cfgIntermediateSize config
    floor (gcRopeBase cfg) `shouldBe` floor (cfgRopeTheta config)

  it "FP32: Embedding output matches official model" $ do
    putStrLn "\n=== Comparing Embedding Layer ==="
    ref <- loadOfficialReference

    let tokens = refTokens ref
        embeddingRef = head (refLayerOutputs ref)

    putStrLn $ "Tokens: " ++ show tokens
    putStrLn $ "Testing last token: " ++ show (last tokens)

    -- Load model
    st <- loadSafeTensors "../models/gemma3-1b-official-instruct/model.safetensors"

    -- Get embedding for last token
    embedWeight <- getTensor st "model.embed_tokens.weight"
    let lastToken = last tokens
        rawEmbedding = V.slice (lastToken * 1152) 1152 embedWeight

    -- Gemma applies embedding normalization: multiply by sqrt(hidden_dim)
    -- This matches what transformers does in hidden_states[0]
    let hiddenDim = 1152
        normScale = sqrt (fromIntegral hiddenDim :: Double)
        normalizedEmbedding = V.map (\x -> realToFrac (realToFrac x * normScale :: Double)) rawEmbedding

    -- Compare with reference
    match <- compareStats "Embedding (normalized)" embeddingRef normalizedEmbedding
    match `shouldBe` True

  xit "FP32: Layer 0 output matches official model" $ do
    putStrLn "\n=== Comparing Layer 0 ==="
    ref <- loadOfficialReference

    putStrLn "⚠️  Full layer comparison not yet implemented"
    putStrLn "Next step: Implement full transformer block and compare outputs"

    pending
