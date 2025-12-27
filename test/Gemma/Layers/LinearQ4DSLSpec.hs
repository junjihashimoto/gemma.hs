{-# LANGUAGE OverloadedStrings #-}

module Gemma.Layers.LinearQ4DSLSpec (spec) where

import Test.Hspec
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import Graphics.WebGPU.Dawn.ContT (evalContT)

-- Import Q4 functions
import Gemma.Quantization.Q4
import Gemma.Layers.LinearQ4DSL

-- Helper: CPU implementation of matrix-vector multiplication (y = W @ x)
cpuLinear :: Vector Float -> Vector Float -> Int -> Int -> Vector Float
cpuLinear weight input outSize inSize =
  V.generate outSize $ \row ->
    V.sum $ V.zipWith (*)
      (V.slice (row * inSize) inSize weight)
      input

-- Helper: Calculate mean absolute error
meanAbsError :: Vector Float -> Vector Float -> Float
meanAbsError v1 v2 =
  let diffs = V.zipWith (\a b -> abs (a - b)) v1 v2
  in V.sum diffs / fromIntegral (V.length diffs)

-- Helper: Calculate max absolute error
maxAbsError :: Vector Float -> Vector Float -> Float
maxAbsError v1 v2 =
  V.maximum $ V.zipWith (\a b -> abs (a - b)) v1 v2

spec :: Spec
spec = do
  describe "LinearQ4DSL (DSL-based Q4 GPU Linear Layer)" $ do

    -- Test 1: DSL Q4 GPU matches CPU quantized result
    it "DSL Q4 GPU linear matches CPU quantized linear (small matrix)" $ do
      -- Test: [4x64] @ [64] = [4]
      let outSize = 4
          inSize = 64  -- Must be multiple of 32 for Q4
          weights = V.generate (outSize * inSize) (\i -> fromIntegral (i `mod` 20) - 10.0) :: Vector Float
          input = V.generate inSize (\i -> fromIntegral (i `mod` 10) / 10.0) :: Vector Float

      -- CPU quantization
      let (packedWeights, scales) = quantizeQ4 weights outSize inSize

      -- CPU dequantized linear (baseline)
      let dequantizedWeights = dequantizeQ4 packedWeights scales
          cpuOutput = cpuLinear dequantizedWeights input outSize inSize

      -- DEBUG: Print info
      putStrLn $ "=== LinearQ4DSL Test 1 (Small Matrix) ==="
      putStrLn $ "Weight shape: [" ++ show outSize ++ " x " ++ show inSize ++ "]"
      putStrLn $ "Input sum: " ++ show (V.sum input)
      putStrLn $ "CPU output: " ++ show (V.toList cpuOutput)
      putStrLn $ "First 4 scales: " ++ show (V.toList $ V.take 4 scales)

      -- GPU Q4 DSL linear
      gpuOutput <- evalContT $ runLinearQ4DSL packedWeights scales input outSize inSize

      -- DEBUG: Print GPU output
      putStrLn $ "GPU DSL output: " ++ show (V.toList gpuOutput)
      putStrLn $ "Mean error: " ++ show (meanAbsError cpuOutput gpuOutput)
      putStrLn $ "Max error: " ++ show (maxAbsError cpuOutput gpuOutput)

      -- Should match closely (small numerical differences OK)
      let error = meanAbsError cpuOutput gpuOutput
      error `shouldSatisfy` (< 0.01)

    -- Test 2: Larger matrix
    it "DSL Q4 GPU linear matches CPU quantized linear (medium matrix)" $ do
      -- Test: [64x128] @ [128] = [64]
      let outSize = 64
          inSize = 128  -- 4 blocks per row
          weights = V.generate (outSize * inSize) (\i -> sin (fromIntegral i / 100.0)) :: Vector Float
          input = V.generate inSize (\i -> cos (fromIntegral i / 50.0)) :: Vector Float

      -- CPU quantization
      let (packedWeights, scales) = quantizeQ4 weights outSize inSize

      -- CPU dequantized linear (baseline)
      let dequantizedWeights = dequantizeQ4 packedWeights scales
          cpuOutput = cpuLinear dequantizedWeights input outSize inSize

      putStrLn $ "=== LinearQ4DSL Test 2 (Medium Matrix) ==="
      putStrLn $ "Weight shape: [" ++ show outSize ++ " x " ++ show inSize ++ "]"
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuOutput)

      -- GPU Q4 DSL linear
      gpuOutput <- evalContT $ runLinearQ4DSL packedWeights scales input outSize inSize

      putStrLn $ "GPU DSL output sum: " ++ show (V.sum gpuOutput)
      putStrLn $ "Mean error: " ++ show (meanAbsError cpuOutput gpuOutput)
      putStrLn $ "Max error: " ++ show (maxAbsError cpuOutput gpuOutput)

      -- Should match closely
      let error = meanAbsError cpuOutput gpuOutput
      error `shouldSatisfy` (< 0.01)

    -- Test 3: Large matrix
    it "DSL Q4 GPU linear matches CPU quantized linear (large matrix)" $ do
      -- Test: [256x256] @ [256] = [256]
      let outSize = 256
          inSize = 256  -- 8 blocks per row
          weights = V.generate (outSize * inSize) (\i -> fromIntegral (i `mod` 100) / 100.0 - 0.5) :: Vector Float
          input = V.generate inSize (\i -> fromIntegral (i `mod` 50) / 50.0) :: Vector Float

      -- CPU quantization
      let (packedWeights, scales) = quantizeQ4 weights outSize inSize

      -- CPU dequantized linear (baseline)
      let dequantizedWeights = dequantizeQ4 packedWeights scales
          cpuOutput = cpuLinear dequantizedWeights input outSize inSize

      putStrLn $ "=== LinearQ4DSL Test 3 (Large Matrix) ==="
      putStrLn $ "Weight shape: [" ++ show outSize ++ " x " ++ show inSize ++ "]"
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuOutput)

      -- GPU Q4 DSL linear
      gpuOutput <- evalContT $ runLinearQ4DSL packedWeights scales input outSize inSize

      putStrLn $ "GPU DSL output sum: " ++ show (V.sum gpuOutput)
      putStrLn $ "Mean error: " ++ show (meanAbsError cpuOutput gpuOutput)
      putStrLn $ "Max error: " ++ show (maxAbsError cpuOutput gpuOutput)

      -- Should match closely
      let error = meanAbsError cpuOutput gpuOutput
      error `shouldSatisfy` (< 0.01)

    -- Test 4: Q4 quantization error vs FP32
    it "DSL Q4 linear has acceptable error vs FP32 linear" $ do
      -- Test: [32x64] @ [64] = [32]
      let outSize = 32
          inSize = 64
          -- Typical neural network weights: ~N(0, 0.1)
          weights = V.generate (outSize * inSize) (\i -> sin (fromIntegral i) * 0.1) :: Vector Float
          input = V.generate inSize (\i -> cos (fromIntegral i) * 0.5) :: Vector Float

      -- FP32 baseline
      let fp32Output = cpuLinear weights input outSize inSize

      -- Q4 quantized
      let (packedWeights, scales) = quantizeQ4 weights outSize inSize
      q4Output <- evalContT $ runLinearQ4DSL packedWeights scales input outSize inSize

      putStrLn $ "=== LinearQ4DSL Test 4 (Q4 vs FP32) ==="
      putStrLn $ "FP32 output sum: " ++ show (V.sum fp32Output)
      putStrLn $ "Q4 output sum: " ++ show (V.sum q4Output)

      -- Calculate relative error
      let absError = meanAbsError fp32Output q4Output
          maxVal = V.maximum (V.map abs fp32Output)
          relativeError = absError / maxVal

      putStrLn $ "Absolute error: " ++ show absError
      putStrLn $ "Relative error: " ++ show (relativeError * 100) ++ "%"

      -- Q4 should have <10% error for typical weights
      relativeError `shouldSatisfy` (< 0.10)
