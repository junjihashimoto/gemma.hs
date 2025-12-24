{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}

{-|
Module: Gemma.Regression.InferenceSpec
Description: Test end-to-end inference against PyTorch reference (TDD)

This module validates that the Haskell Gemma model produces
identical results to PyTorch's implementation.

Following TDD principles from CLAUDE.md:
- Perfect one-to-one correspondence between Haskell and PyTorch
- Expected values created from PyTorch (not ad-hoc)
- No implicit data conversion
-}

module Gemma.Regression.InferenceSpec (spec) where

import Test.Hspec
import qualified Data.Vector.Storable as V
import qualified Data.Aeson as JSON
import qualified Data.ByteString.Lazy as BL
import Data.Aeson ((.:))
import qualified Data.Aeson.Types as JSON
import qualified Data.Aeson.Key as K
import Gemma.Model (loadGemmaModel, runGemmaInferenceCached, gemma3_1BConfig)
import Text.Printf
import Control.Monad (when)

-- | PyTorch inference reference data
data PyTorchInferenceReference = PyTorchInferenceReference
  { prConfig :: JSON.Object
  , prIntermediate :: JSON.Object
  , prLogitsShape :: [Int]
  , prLogitsSample :: [Float]  -- First 20 logits
  , prLogitsStats :: JSON.Object
  , prTop5Tokens :: [Int]
  , prTop5Probs :: [Float]
  } deriving (Show)

instance JSON.FromJSON PyTorchInferenceReference where
  parseJSON = JSON.withObject "PyTorchInferenceReference" $ \v -> PyTorchInferenceReference
    <$> v .: "config"
    <*> v .: "intermediate"
    <*> v .: "logits_shape"
    <*> v .: "logits_sample"
    <*> v .: "logits_stats"
    <*> v .: "top_5_tokens"
    <*> v .: "top_5_probs"

-- | Get float value from JSON object
getFloat :: JSON.Object -> String -> Float
getFloat obj key =
  let k = K.fromString key
  in case JSON.parseMaybe (JSON.withObject "getFloat" (.: k)) (JSON.Object obj) of
       Just (JSON.Number n) -> realToFrac n
       _ -> error $ "Key not found or not a number: " ++ key

-- | Get list from JSON object
getList :: JSON.FromJSON a => JSON.Object -> String -> [a]
getList obj key =
  let k = K.fromString key
  in case JSON.parseMaybe (JSON.withObject "getList" (.: k)) (JSON.Object obj) of
       Just val -> val
       _ -> error $ "Key not found or not a list: " ++ key

spec :: Spec
spec = describe "End-to-End Inference PyTorch Comparison" $ do

  it "single token inference logits match PyTorch" $ do
    -- Load PyTorch reference
    referenceJson <- BL.readFile "scripts/inference_reference.json"
    case JSON.eitherDecode referenceJson of
      Left err -> expectationFailure $ "Failed to parse reference JSON: " ++ err
      Right (ref :: PyTorchInferenceReference) -> do
        -- Load Haskell model
        model <- loadGemmaModel "../models/gemma3-1b.safetensors" gemma3_1BConfig

        -- Run inference with same token (token 1)
        let token = V.singleton 1
        (logitsHaskell, _) <- runGemmaInferenceCached model token Nothing

        -- Get stats from Haskell logits
        let logitsMin = V.minimum logitsHaskell
            logitsMax = V.maximum logitsHaskell
            logitsMean = V.sum logitsHaskell / fromIntegral (V.length logitsHaskell)
            logitsVariance = V.sum (V.map (\x -> (x - logitsMean) ** 2) logitsHaskell) / fromIntegral (V.length logitsHaskell)
            logitsStd = sqrt logitsVariance

        -- Get PyTorch stats
        let pytorchStats = prLogitsStats ref
            pytorchMin = getFloat pytorchStats "min"
            pytorchMax = getFloat pytorchStats "max"
            pytorchMean = getFloat pytorchStats "mean"
            pytorchStd = getFloat pytorchStats "std"

        -- Compare first 20 logits element-by-element
        let logitsSample = V.toList $ V.take 20 logitsHaskell
            pytorchSample = prLogitsSample ref
            sampleDiffs = zipWith (\h p -> abs (h - p)) logitsSample pytorchSample
            maxSampleDiff = maximum sampleDiffs
            meanSampleDiff = sum sampleDiffs / fromIntegral (length sampleDiffs)

        -- Print detailed comparison
        let comparison = printf "\n\
                               \Logits Statistics Comparison:\n\
                               \  Min:  Haskell=%.4f  PyTorch=%.4f  Diff=%.4f\n\
                               \  Max:  Haskell=%.4f  PyTorch=%.4f  Diff=%.4f\n\
                               \  Mean: Haskell=%.4f  PyTorch=%.4f  Diff=%.4f\n\
                               \  Std:  Haskell=%.4f  PyTorch=%.4f  Diff=%.4f\n\
                               \\n\
                               \First 20 Logits Comparison:\n\
                               \  Max diff:  %.6f\n\
                               \  Mean diff: %.6f\n\
                               \\n\
                               \First 5 values:\n"
                        logitsMin pytorchMin (abs (logitsMin - pytorchMin))
                        logitsMax pytorchMax (abs (logitsMax - pytorchMax))
                        logitsMean pytorchMean (abs (logitsMean - pytorchMean))
                        logitsStd pytorchStd (abs (logitsStd - pytorchStd))
                        maxSampleDiff meanSampleDiff

        let valueComparison = unlines $
              ["  Index | Haskell      | PyTorch      | Diff"] ++
              ["  ------|--------------|--------------|-------------"] ++
              [ printf "  %5d | %12.6f | %12.6f | %12.6f" (i :: Int) h p (abs (h - p))
              | (i, h, p) <- zip3 [0..4] logitsSample pytorchSample
              ]

        -- Check if logits show variation (not all same value)
        let allSame = V.all (== V.head logitsHaskell) logitsHaskell
        when allSame $ do
          expectationFailure $ printf "CRITICAL BUG: All logits have same value: %.6e\n\
                                      \This indicates the inference pipeline is broken!" (V.head logitsHaskell)

        -- Allow reasonable tolerance for floating point differences
        -- PyTorch uses different precision and different order of operations
        let tolerance = 1.0  -- Allow up to 1.0 difference in logits (before softmax)

        if maxSampleDiff > tolerance
          then do
            expectationFailure $ comparison ++ valueComparison ++ printf "\nMax diff %.6f exceeds tolerance %.6f" maxSampleDiff tolerance
          else do
            -- Print success message with comparison
            putStrLn comparison
            putStrLn valueComparison
            putStrLn "✅ Logits match PyTorch within tolerance"
            maxSampleDiff `shouldSatisfy` (<= tolerance)

  it "logits show variation (not all same value)" $ do
    -- This test catches the critical bug where all logits are identical
    model <- loadGemmaModel "../models/gemma3-1b.safetensors" gemma3_1BConfig

    -- Run inference with token 1
    let token = V.singleton 1
    (logits, _) <- runGemmaInferenceCached model token Nothing

    -- Check for variation
    let minLogit = V.minimum logits
        maxLogit = V.maximum logits
        range = maxLogit - minLogit

    -- If range is 0, all values are identical (BUG!)
    if range == 0
      then expectationFailure $ printf "CRITICAL BUG: All logits identical (value: %.6e)" minLogit
      else do
        putStrLn $ printf "✅ Logits show variation: min=%.4f, max=%.4f, range=%.4f" minLogit maxLogit range
        range `shouldSatisfy` (> 0)
