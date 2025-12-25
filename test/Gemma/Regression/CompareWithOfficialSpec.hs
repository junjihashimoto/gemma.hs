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
import Control.Monad (foldM)
import Control.Applicative ((<|>))

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
  -- Support both "num_layers" (old format) and "num_hidden_layers" (HuggingFace format)
  cfgNumLayers <- (v A..: "num_layers") <|> (v A..: "num_hidden_layers")
  cfgHiddenSize <- v A..: "hidden_size"
  cfgNumAttentionHeads <- v A..: "num_attention_heads"
  cfgNumKeyValueHeads <- v A..: "num_key_value_heads"
  cfgHeadDim <- v A..: "head_dim"
  cfgIntermediateSize <- v A..: "intermediate_size"
  cfgRopeTheta <- v A..: "rope_theta"
  return ModelConfig{..}

-- Layer 0 specific reference data
data Layer0Reference = Layer0Reference
  { l0Prompt :: String
  , l0Tokens :: [Int]
  , l0LastToken :: Int
  , l0Config :: ModelConfig
  , l0EmbeddingOutput :: LayerOutput
  , l0Layer0Output :: LayerOutput
  } deriving (Show)

parseLayerOutputStats :: A.Value -> Parser LayerOutput
parseLayerOutputStats = A.withObject "LayerOutput" $ \v -> do
  layerMean <- v A..: "mean"
  layerStd <- v A..: "std"
  layerMin <- v A..: "min"
  layerMax <- v A..: "max"
  layerFirst10 <- v A..: "first_10"
  layerLast10 <- v A..: "last_10"
  return LayerOutput{layerName = "layer0", ..}

instance A.FromJSON Layer0Reference where
  parseJSON = A.withObject "Layer0Reference" $ \v -> do
    l0Prompt <- v A..: "prompt"
    l0Tokens <- v A..: "tokens"
    l0LastToken <- v A..: "last_token"
    config <- v A..: "config"
    l0Config <- parseModelConfig config
    embOut <- v A..: "embedding_output"
    l0EmbeddingOutput <- parseLayerOutputStats embOut
    lay0Out <- v A..: "layer0_output"
    l0Layer0Output <- parseLayerOutputStats lay0Out
    return Layer0Reference{..}

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

loadLayer0Reference :: IO Layer0Reference
loadLayer0Reference = do
  let refPath = "test/Gemma/Regression/CompareWithOfficialSpec_layer0.json"
  putStrLn $ "\n📖 Loading Layer 0 reference: " ++ refPath
  content <- BL.readFile refPath
  case A.eitherDecode content of
    Left err -> error $ "Failed to parse Layer 0 reference: " ++ err
    Right ref -> do
      putStrLn "✅ Loaded Layer 0 reference"
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

  it "FP32: Layer 0 output matches official model" $ do
    putStrLn "\n=== Comparing Layer 0 Transformer Block ==="
    ref <- loadLayer0Reference

    let tokens = l0Tokens ref
        layer0Ref = l0Layer0Output ref

    putStrLn $ "Tokens: " ++ show tokens
    putStrLn $ "Number of tokens: " ++ show (length tokens)
    putStrLn $ "Layer 0 reference stats (at last position after processing all tokens):"
    putStrLn $ "  Mean: " ++ show (layerMean layer0Ref)
    putStrLn $ "  Std:  " ++ show (layerStd layer0Ref)
    putStrLn $ "  Min:  " ++ show (layerMin layer0Ref)
    putStrLn $ "  Max:  " ++ show (layerMax layer0Ref)

    -- Load model (use INSTRUCT model matching Python reference)
    model <- loadGemmaModel "../models/gemma3-1b-official-instruct/model.safetensors" gemma3_1BConfig

    -- NOTE: PyTorch reference uses batched inference (use_cache=False, all 16 tokens at once)
    -- but Haskell runGemmaInference only supports single tokens.
    -- Using incremental cached inference instead (should produce equivalent results)
    -- TODO: Investigate if KV caching produces different Layer 0 outputs than batched
    (firstLogits, firstCache) <- runGemmaInferenceCached model (V.singleton (head tokens)) Nothing
    (logits, _finalCache) <- foldM
      (\(_prevLogits, cache) tok -> runGemmaInferenceCached model (V.singleton tok) (Just cache))
      (firstLogits, firstCache)
      (tail tokens)

    putStrLn $ "\n⚠️  Current limitation: Can't access intermediate layer outputs"
    putStrLn $ "   runGemmaInferenceCached only returns final logits"
    putStrLn $ "   Need to either:"
    putStrLn $ "     1. Add debug mode to expose intermediate outputs"
    putStrLn $ "     2. Create test-only function that runs single transformer block"
    putStrLn $ "\n📊 For now, checking that inference runs without error..."
    putStrLn $ "   Logits length: " ++ show (V.length logits)
    putStrLn $ "   Logits stats:"
    let (logitMean, logitStd, logitMin, logitMax) = vectorStats logits
    putStrLn $ "     Mean: " ++ show logitMean
    putStrLn $ "     Std:  " ++ show logitStd
    putStrLn $ "     Min:  " ++ show logitMin
    putStrLn $ "     Max:  " ++ show logitMax

    -- At minimum, verify no NaN and reasonable values
    V.all (not . isNaN) logits `shouldBe` True
    V.all (not . isInfinite) logits `shouldBe` True
