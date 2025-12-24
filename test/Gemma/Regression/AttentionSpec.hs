{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}

{-|
Module: Gemma.Regression.AttentionSpec
Description: Test attention layer against PyTorch reference (TDD)

This module compares each stage of the attention mechanism against PyTorch
reference outputs to identify discrepancies.

Following TDD principles from CLAUDE.md:
- Perfect one-to-one correspondence between Haskell tests and PyTorch expectations
- Expected values created for each layer (not ad-hoc)
- No implicit data conversion between CPU and GPU
-}

module Gemma.Regression.AttentionSpec (spec) where

import Test.Hspec
import qualified Data.Vector.Storable as V
import qualified Data.Text as T
import qualified Data.Aeson as JSON
import qualified Data.ByteString.Lazy as BL
import Data.Aeson ((.:), (.:?))
import qualified Data.Aeson.Types as JSON
import qualified Data.Aeson.KeyMap as KM
import qualified Data.Aeson.Key as K
import Gemma.SafeTensors
import Text.Printf
import Graphics.WebGPU.Dawn.ContT (evalContT)
import Gemma.Layers.RMSNorm (runRMSNormVariant)

-- | Test stage from PyTorch reference
data TestStage = TestStage
  { tsDescription :: String
  , tsShape :: [Int]
  , tsFirst20 :: Maybe [Float]
  , tsAllValues :: [Float]
  } deriving (Show)

-- | PyTorch reference data
data PyTorchReference = PyTorchReference
  { prModelPath :: String
  , prModelDtype :: String
  , prLayerIdx :: Int
  , prTokenId :: Int
  , prConfig :: JSON.Object
  , prHasQKNorm :: Bool
  , prStages :: [(String, TestStage)]
  } deriving (Show)

instance JSON.FromJSON TestStage where
  parseJSON = JSON.withObject "TestStage" $ \v -> TestStage
    <$> v .: "description"
    <*> v .: "shape"
    <*> v .:? "first_20"
    <*> v .: "all_values"

instance JSON.FromJSON PyTorchReference where
  parseJSON = JSON.withObject "PyTorchReference" $ \v -> do
    modelPath <- v .: "model_path"
    modelDtype <- v .: "model_dtype"
    layerIdx <- v .: "layer_idx"
    tokenId <- v .: "token_id"
    config <- v .: "config"
    hasQKNorm <- v .: "has_qk_norm"
    stagesObj <- v .: "stages" :: JSON.Parser JSON.Object
    stagesList <- mapM (\(k, val) -> do
        tc <- JSON.parseJSON val
        return (T.unpack $ K.toText k, tc)) (KM.toList stagesObj)
    return $ PyTorchReference modelPath modelDtype layerIdx tokenId config hasQKNorm stagesList

-- | Get value from JSON config
getConfigInt :: JSON.Object -> T.Text -> Int
getConfigInt obj key = case KM.lookup (K.fromText key) obj of
  Just (JSON.Number n) -> round n
  _ -> error $ "Config key not found or not a number: " ++ T.unpack key

getConfigFloat :: JSON.Object -> T.Text -> Float
getConfigFloat obj key = case KM.lookup (K.fromText key) obj of
  Just (JSON.Number n) -> realToFrac n
  _ -> error $ "Config key not found or not a number: " ++ T.unpack key

getConfigBool :: JSON.Object -> T.Text -> Bool
getConfigBool obj key = case KM.lookup (K.fromText key) obj of
  Just (JSON.Bool b) -> b
  _ -> error $ "Config key not found or not a boolean: " ++ T.unpack key

-- | Compare two vectors with tolerance
compareVectors :: Float -> V.Vector Float -> [Float] -> (Bool, String)
compareVectors tolerance haskellVec pytorchList =
  let haskellList = V.toList haskellVec
      len = length pytorchList
  in if length haskellList /= len
     then (False, "Length mismatch: " ++ show (length haskellList) ++ " vs " ++ show len)
     else
       let differences = zipWith (\h p -> abs (h - p)) haskellList pytorchList
           maxDiff = maximum differences
           avgDiff = sum differences / fromIntegral len
           numMismatches = length $ filter (> tolerance) differences
           summary = printf "Max diff: %.10f, Avg diff: %.10f, Mismatches: %d" maxDiff avgDiff numMismatches
       in if numMismatches == 0
          then (True, "PASS - " ++ summary)
          else
            let firstMismatch = head $ filter (\(_, _, diff) -> diff > tolerance)
                                       $ zipWith3 (\i h p -> (i, (h, p), abs (h - p)))
                                                  ([0..] :: [Int]) haskellList pytorchList
                (idx, (hVal, pVal), diff) = firstMismatch
                detail = printf "\nFirst mismatch at index %d: Haskell=%.10f, PyTorch=%.10f, Diff=%.10f"
                               idx hVal pVal diff
            in (False, "FAIL - " ++ summary ++ detail)

spec :: Spec
spec = describe "Attention Layer PyTorch Comparison" $ do
  it "Stage 1: Input embedding matches PyTorch" $ do
    -- Load PyTorch reference
    referenceJson <- BL.readFile "scripts/attention_reference_fp32.json"
    case JSON.eitherDecode referenceJson of
      Left err -> expectationFailure $ "Failed to parse reference JSON: " ++ err
      Right (ref :: PyTorchReference) -> do
        -- Load Haskell SafeTensors
        st <- loadSafeTensors "../models/gemma3-1b.safetensors"

        -- Extract config
        let config = prConfig ref
            hiddenDim = getConfigInt config "hidden_dim"
            tokenId = prTokenId ref
            tolerance = 1e-6 :: Float  -- FP32 should be very accurate

        -- Stage 1: Input embedding
        embedTable <- getTensor st "model.embed_tokens.weight"
        let inputEmbedding = V.slice (tokenId * hiddenDim) hiddenDim embedTable

        case lookup "1_input_embedding" (prStages ref) of
          Nothing -> expectationFailure "Stage 1_input_embedding not found in reference"
          Just stage -> do
            let pytorchValues = tsAllValues stage
                (pass, msg) = compareVectors tolerance inputEmbedding pytorchValues
            if not pass
              then expectationFailure $ "Stage 1 (Input Embedding): " ++ msg
              else pass `shouldBe` True

  it "Stage 2: Pre-attention RMSNorm matches PyTorch" $ do
    -- Load PyTorch reference
    referenceJson <- BL.readFile "scripts/attention_reference_fp32.json"
    case JSON.eitherDecode referenceJson of
      Left err -> expectationFailure $ "Failed to parse reference JSON: " ++ err
      Right (ref :: PyTorchReference) -> do
        -- Load Haskell SafeTensors
        st <- loadSafeTensors "../models/gemma3-1b.safetensors"

        -- Extract config
        let config = prConfig ref
            hiddenDim = getConfigInt config "hidden_dim"
            tokenId = prTokenId ref
            layerIdx = prLayerIdx ref
            zeroCentered = getConfigBool config "zero_centered"
            tolerance = 2e-5 :: Float  -- Slightly relaxed for GPU floating-point precision

        -- Get input embedding
        embedTable <- getTensor st "model.embed_tokens.weight"
        let inputEmbedding = V.slice (tokenId * hiddenDim) hiddenDim embedTable

        -- Get norm weights
        attnNormWeights <- getTensor st (T.pack $ "model.layers." ++ show layerIdx ++ ".input_layernorm.weight")

        -- Run RMSNorm using high-level API with zero-centered mode (Gemma 3)
        rmsNormOut <- evalContT $ runRMSNormVariant zeroCentered inputEmbedding attnNormWeights

        case lookup "2_attn_norm_out" (prStages ref) of
          Nothing -> expectationFailure "Stage 2_attn_norm_out not found in reference"
          Just stage -> do
            let pytorchValues = tsAllValues stage
                (pass, msg) = compareVectors tolerance rmsNormOut pytorchValues
            if not pass
              then expectationFailure $ "Stage 2 (Pre-attention RMSNorm): " ++ msg
              else pass `shouldBe` True

  -- TODO: Stage 3 (Q/K/V Projections) - Need to find high-level API or use low-level GPU API
  -- For now, we'll test RMSNorm first and see if that's where the bug is
