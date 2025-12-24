{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}

{-|
Module: Gemma.Regression.Q4Spec
Description: Test Q4 quantization against PyTorch reference (TDD)

This module validates that the Haskell Q4 implementation produces
identical results to PyTorch's Q4 quantization.

Following TDD principles from CLAUDE.md:
- Perfect one-to-one correspondence between Haskell tests and PyTorch expectations
- Expected values created for each layer (not ad-hoc)
- No implicit data conversion between CPU and GPU
-}

module Gemma.Regression.Q4Spec (spec) where

import Test.Hspec
import qualified Data.Vector.Storable as V
import qualified Data.Aeson as JSON
import qualified Data.ByteString.Lazy as BL
import Data.Aeson ((.:))
import qualified Data.Aeson.Types as JSON
import qualified Data.Aeson.Key as K
import Gemma.Quantization.Q4 (quantizeQ4, dequantizeQ4)
import Gemma.SafeTensors
import Text.Printf
import Data.Word (Word32)

-- | PyTorch Q4 reference data
data PyTorchQ4Reference = PyTorchQ4Reference
  { prModelPath :: String
  , prLayerName :: String
  , prOriginalShape :: [Int]
  , prBlockSize :: Int
  , prNumElements :: Int
  , prNumPacked :: Int
  , prNumScales :: Int
  , prOriginalStats :: JSON.Object
  , prQuantizationError :: JSON.Object
  , prOriginalSample :: [Float]
  , prQuantizedSample :: [Int]  -- Word8 in Python
  , prScalesSample :: [Float]
  , prPackedSample :: [Word32]
  , prDequantizedSample :: [Float]
  } deriving (Show)

instance JSON.FromJSON PyTorchQ4Reference where
  parseJSON = JSON.withObject "PyTorchQ4Reference" $ \v -> PyTorchQ4Reference
    <$> v .: "model_path"
    <*> v .: "layer_name"
    <*> v .: "original_shape"
    <*> v .: "block_size"
    <*> v .: "num_elements"
    <*> v .: "num_packed"
    <*> v .: "num_scales"
    <*> v .: "original_stats"
    <*> v .: "quantization_error"
    <*> v .: "original_sample"
    <*> v .: "quantized_sample"
    <*> v .: "scales_sample"
    <*> v .: "packed_sample"
    <*> v .: "dequantized_sample"

-- | Get float value from JSON object
getFloat :: JSON.Object -> String -> Float
getFloat obj key =
  let k = K.fromString key
  in case JSON.parseMaybe (JSON.withObject "getFloat" (.: k)) (JSON.Object obj) of
       Just (JSON.Number n) -> realToFrac n
       _ -> error $ "Key not found or not a number: " ++ key

spec :: Spec
spec = describe "Q4 Quantization PyTorch Comparison" $ do

  it "Q4 quantized values match PyTorch (first 256 elements)" $ do
    -- Load PyTorch reference
    referenceJson <- BL.readFile "scripts/q4_reference.json"
    case JSON.eitherDecode referenceJson of
      Left err -> expectationFailure $ "Failed to parse reference JSON: " ++ err
      Right (ref :: PyTorchQ4Reference) -> do
        -- Load Haskell SafeTensors
        st <- loadSafeTensors "../models/gemma3-1b.safetensors"

        -- Get the same layer
        weights <- getTensor st "model.layers.0.self_attn.q_proj.weight"

        let [outSize, inSize] = prOriginalShape ref
            totalElements = outSize * inSize

        -- Take first 256 elements for comparison
        let sampleSize = 256
            weightsSample = V.take sampleSize weights
            pytorchOriginal = prOriginalSample ref

        -- First verify we loaded the same data
        let maxDiffLoad = maximum $ zipWith (\h p -> abs (h - p))
                                             (V.toList weightsSample)
                                             pytorchOriginal

        maxDiffLoad `shouldSatisfy` (< 1e-6)  -- Should be identical

        -- Quantize with Haskell (using full weights to get proper dimensions)
        let (packedHaskell, scalesHaskell) = quantizeQ4 weights outSize inSize

        -- Take first 32 packed words (256 elements / 8 = 32 words)
        let packedSample = V.toList $ V.take 32 packedHaskell
            pytorchPacked = prPackedSample ref

        -- Compare packed values
        let packedMismatches = length $ filter id $ zipWith (/=) packedSample pytorchPacked
            packedTotal = min (length packedSample) (length pytorchPacked)

        -- Compare scales (first 8 scales cover 256 elements)
        let scalesSample = V.toList $ V.take 8 scalesHaskell
            pytorchScales = prScalesSample ref
            scalesMaxDiff = maximum $ zipWith (\h p -> abs (h - p)) scalesSample pytorchScales

        -- Report results
        if packedMismatches > 0 || scalesMaxDiff > 1e-5
          then do
            let msg = printf "Q4 Quantization Mismatch:\n\
                             \  Packed mismatches: %d / %d\n\
                             \  Scales max diff: %.10f\n\
                             \  First 5 packed (Haskell): %s\n\
                             \  First 5 packed (PyTorch): %s\n\
                             \  First 5 scales (Haskell): %s\n\
                             \  First 5 scales (PyTorch): %s"
                      packedMismatches packedTotal scalesMaxDiff
                      (show $ take 5 packedSample)
                      (show $ take 5 pytorchPacked)
                      (show $ take 5 scalesSample)
                      (show $ take 5 pytorchScales)
            expectationFailure msg
          else
            (packedMismatches == 0 && scalesMaxDiff < 1e-5) `shouldBe` True

  it "Q4 dequantization matches PyTorch" $ do
    -- Load PyTorch reference
    referenceJson <- BL.readFile "scripts/q4_reference.json"
    case JSON.eitherDecode referenceJson of
      Left err -> expectationFailure $ "Failed to parse reference JSON: " ++ err
      Right (ref :: PyTorchQ4Reference) -> do
        -- Load Haskell SafeTensors
        st <- loadSafeTensors "../models/gemma3-1b.safetensors"

        -- Get the same layer
        weights <- getTensor st "model.layers.0.self_attn.q_proj.weight"

        let [outSize, inSize] = prOriginalShape ref

        -- Quantize then dequantize
        let (packed, scales) = quantizeQ4 weights outSize inSize
            dequantized = dequantizeQ4 packed scales

        -- Compare first 256 elements
        let sampleSize = 256
            dequantizedSample = V.toList $ V.take sampleSize dequantized
            pytorchDequantized = prDequantizedSample ref

        -- Compute difference
        let diffs = zipWith (\h p -> abs (h - p)) dequantizedSample pytorchDequantized
            maxDiff = maximum diffs
            meanDiff = sum diffs / fromIntegral (length diffs)

        -- Allow small tolerance for floating point precision
        let tolerance = 1e-5

        if maxDiff > tolerance
          then do
            let msg = printf "Q4 Dequantization Mismatch:\n\
                             \  Max diff: %.10f\n\
                             \  Mean diff: %.10f\n\
                             \  First mismatch: Haskell=%.6f, PyTorch=%.6f"
                      maxDiff meanDiff (head dequantizedSample) (head pytorchDequantized)
            expectationFailure msg
          else
            maxDiff `shouldSatisfy` (<= tolerance)

  it "Q4 quantization error is within expected bounds" $ do
    -- Load PyTorch reference
    referenceJson <- BL.readFile "scripts/q4_reference.json"
    case JSON.eitherDecode referenceJson of
      Left err -> expectationFailure $ "Failed to parse reference JSON: " ++ err
      Right (ref :: PyTorchQ4Reference) -> do
        -- Load Haskell SafeTensors
        st <- loadSafeTensors "../models/gemma3-1b.safetensors"

        -- Get the same layer
        weights <- getTensor st "model.layers.0.self_attn.q_proj.weight"

        let [outSize, inSize] = prOriginalShape ref
            sampleSize = 256

        -- Take first 256 elements to avoid numerical overflow
        let weightsSample = V.take sampleSize weights
            pytorchOriginal = take sampleSize $ prOriginalSample ref

        -- Quantize then dequantize
        let (packed, scales) = quantizeQ4 weights outSize inSize
            dequantized = dequantizeQ4 packed scales
            dequantizedSample = V.take sampleSize dequantized

        -- Compute error metrics on sample
        let diffs = V.zipWith (\orig deq -> abs (orig - deq)) weightsSample dequantizedSample
            maxDiff = V.maximum diffs
            meanDiff = V.sum diffs / fromIntegral (V.length diffs)
            absSum = V.sum (V.map abs weightsSample)
            relError = meanDiff / ((absSum / fromIntegral (V.length weightsSample)) + 1e-8)

        -- Q4 quantization should have reasonable error bounds
        -- Based on PyTorch reference: max_diff ~0.023, mean_diff ~0.002, rel_error ~11%
        let maxDiffBound = 0.03     -- Allow up to 3% max error
            meanDiffBound = 0.005   -- Allow up to 0.5% mean error
            relErrorBound = 0.15    -- Allow up to 15% relative error

        if maxDiff > maxDiffBound || meanDiff > meanDiffBound || relError > relErrorBound
          then do
            let msg = printf "Q4 Quantization Error Out of Bounds:\n\
                             \  Max diff: %.6f (bound: %.6f)\n\
                             \  Mean diff: %.6f (bound: %.6f)\n\
                             \  Relative error: %.4f%% (bound: %.4f%%)"
                      maxDiff maxDiffBound
                      meanDiff meanDiffBound
                      (relError * 100) (relErrorBound * 100)
            expectationFailure msg
          else
            (maxDiff <= maxDiffBound) `shouldBe` True
