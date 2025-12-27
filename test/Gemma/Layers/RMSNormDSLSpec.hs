{-# LANGUAGE OverloadedStrings #-}

module Gemma.Layers.RMSNormDSLSpec (spec) where

import Test.Hspec
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import Graphics.WebGPU.Dawn.ContT (evalContT)

import Gemma.Layers.RMSNormDSL

-- | CPU implementation of RMSNorm
--
-- Formula: RMSNorm(x) = (x / RMS(x)) * weight
-- where RMS(x) = sqrt(mean(x²) + ε)
--
-- For zero-centered mode (Gemma 3):
--   RMSNorm(x) = (x / RMS(x)) * (1 + weight)
cpuRMSNorm :: Vector Float -> Vector Float -> Bool -> Vector Float
cpuRMSNorm input weight zeroCentered =
  let -- Step 1: Compute mean of squares
      squares = V.map (\x -> x * x) input
      meanSq = V.sum squares / fromIntegral (V.length input)

      -- Step 2: Compute RMS with epsilon for stability
      epsilon = 1e-6
      rms = sqrt (meanSq + epsilon)

      -- Step 3: Normalize and apply weight
      normalized = if zeroCentered
        then V.zipWith (\x w -> (x / rms) * (1.0 + w)) input weight
        else V.zipWith (\x w -> (x / rms) * w) input weight
  in normalized

-- | Calculate mean absolute error
meanAbsError :: Vector Float -> Vector Float -> Float
meanAbsError v1 v2 =
  let diffs = V.zipWith (\a b -> abs (a - b)) v1 v2
  in V.sum diffs / fromIntegral (V.length diffs)

-- | Calculate max absolute error
maxAbsError :: Vector Float -> Vector Float -> Float
maxAbsError v1 v2 =
  V.maximum $ V.zipWith (\a b -> abs (a - b)) v1 v2

spec :: Spec
spec = do
  describe "RMSNormDSL (DSL-based GPU RMSNorm)" $ do

    -- Test 1: Small vector with FP32
    it "DSL-based GPU RMSNorm matches CPU RMSNorm (small, FP32)" $ do
      let hiddenSize = 256
          input = V.generate hiddenSize (\i -> sin (fromIntegral i / 10.0)) :: Vector Float
          weight = V.generate hiddenSize (\i -> 0.1 + cos (fromIntegral i / 20.0)) :: Vector Float
          zeroCentered = False

      -- CPU baseline
      let cpuOutput = cpuRMSNorm input weight zeroCentered

      putStrLn $ "=== RMSNormDSL Test 1 (Small, FP32) ==="
      putStrLn $ "Hidden size: " ++ show hiddenSize
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuOutput)

      -- GPU DSL-based RMSNorm
      gpuOutput <- evalContT $ runRMSNormDSL input weight hiddenSize zeroCentered

      putStrLn $ "GPU DSL output sum: " ++ show (V.sum gpuOutput)
      putStrLn $ "Mean error: " ++ show (meanAbsError cpuOutput gpuOutput)
      putStrLn $ "Max error: " ++ show (maxAbsError cpuOutput gpuOutput)

      -- FP32 should be very accurate
      let err = meanAbsError cpuOutput gpuOutput
      err `shouldSatisfy` (< 1e-5)

    -- Test 2: Medium vector with FP32
    it "DSL-based GPU RMSNorm matches CPU RMSNorm (medium, FP32)" $ do
      let hiddenSize = 1024
          input = V.generate hiddenSize (\i -> fromIntegral (i `mod` 100) / 100.0) :: Vector Float
          weight = V.generate hiddenSize (\i -> 0.5 + fromIntegral (i `mod` 50) / 100.0) :: Vector Float
          zeroCentered = False

      let cpuOutput = cpuRMSNorm input weight zeroCentered

      putStrLn $ "=== RMSNormDSL Test 2 (Medium, FP32) ==="
      putStrLn $ "Hidden size: " ++ show hiddenSize
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuOutput)

      gpuOutput <- evalContT $ runRMSNormDSL input weight hiddenSize zeroCentered

      putStrLn $ "GPU DSL output sum: " ++ show (V.sum gpuOutput)
      putStrLn $ "Mean error: " ++ show (meanAbsError cpuOutput gpuOutput)
      putStrLn $ "Max error: " ++ show (maxAbsError cpuOutput gpuOutput)

      let err = meanAbsError cpuOutput gpuOutput
      err `shouldSatisfy` (< 1e-5)

    -- Test 3: FP16 precision test
    it "DSL-based GPU RMSNorm with FP16 matches CPU (medium)" $ do
      let hiddenSize = 1024
          input = V.generate hiddenSize (\i -> sin (fromIntegral i / 50.0)) :: Vector Float
          weight = V.generate hiddenSize (\i -> 0.3 + cos (fromIntegral i / 30.0)) :: Vector Float
          zeroCentered = False

      let cpuOutput = cpuRMSNorm input weight zeroCentered

      putStrLn $ "=== RMSNormDSL Test 3 (FP16) ==="
      putStrLn $ "Hidden size: " ++ show hiddenSize
      putStrLn $ "CPU (FP32) output sum: " ++ show (V.sum cpuOutput)

      -- GPU with FP16
      gpuOutput <- evalContT $ runRMSNormDSLWithPrecision input weight hiddenSize zeroCentered True False

      putStrLn $ "GPU FP16 output sum: " ++ show (V.sum gpuOutput)
      putStrLn $ "Mean error: " ++ show (meanAbsError cpuOutput gpuOutput)
      putStrLn $ "Max error: " ++ show (maxAbsError cpuOutput gpuOutput)

      -- FP16 loses some precision but should still be very good
      let err = meanAbsError cpuOutput gpuOutput
      err `shouldSatisfy` (< 5e-3)  -- 0.5% error tolerance for FP16

    -- Test 4: Vec4 SIMD optimization (FP32)
    it "DSL-based GPU RMSNorm with Vec4 SIMD matches CPU (medium)" $ do
      let hiddenSize = 1024
          input = V.generate hiddenSize (\i -> fromIntegral (i `mod` 75) / 75.0) :: Vector Float
          weight = V.generate hiddenSize (\i -> 0.2 + fromIntegral (i `mod` 60) / 120.0) :: Vector Float
          zeroCentered = False

      let cpuOutput = cpuRMSNorm input weight zeroCentered

      putStrLn $ "=== RMSNormDSL Test 4 (Vec4, FP32) ==="
      putStrLn $ "Hidden size: " ++ show hiddenSize
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuOutput)

      -- GPU with Vec4 SIMD
      gpuOutput <- evalContT $ runRMSNormDSLWithPrecision input weight hiddenSize zeroCentered False True

      putStrLn $ "GPU Vec4 output sum: " ++ show (V.sum gpuOutput)
      putStrLn $ "Mean error: " ++ show (meanAbsError cpuOutput gpuOutput)
      putStrLn $ "Max error: " ++ show (maxAbsError cpuOutput gpuOutput)

      let err = meanAbsError cpuOutput gpuOutput
      err `shouldSatisfy` (< 1e-5)

    -- Test 5: Vec4 + FP16 (full optimization)
    it "DSL-based GPU RMSNorm with Vec4+FP16 matches CPU (medium)" $ do
      let hiddenSize = 2048
          input = V.generate hiddenSize (\i -> sin (fromIntegral i / 100.0)) :: Vector Float
          weight = V.generate hiddenSize (\i -> 0.4 + cos (fromIntegral i / 80.0)) :: Vector Float
          zeroCentered = False

      let cpuOutput = cpuRMSNorm input weight zeroCentered

      putStrLn $ "=== RMSNormDSL Test 5 (Vec4+FP16) ==="
      putStrLn $ "Hidden size: " ++ show hiddenSize
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuOutput)

      -- GPU with Vec4 + FP16 (full optimization)
      gpuOutput <- evalContT $ runRMSNormDSLWithPrecision input weight hiddenSize zeroCentered True True

      putStrLn $ "GPU Vec4+FP16 output sum: " ++ show (V.sum gpuOutput)
      putStrLn $ "Mean error: " ++ show (meanAbsError cpuOutput gpuOutput)
      putStrLn $ "Max error: " ++ show (maxAbsError cpuOutput gpuOutput)

      let err = meanAbsError cpuOutput gpuOutput
      err `shouldSatisfy` (< 5e-3)  -- 0.5% tolerance for FP16

    -- Test 6: Zero-centered mode (Gemma 3)
    it "DSL-based GPU RMSNorm with zero-centered mode matches CPU" $ do
      let hiddenSize = 1024
          input = V.generate hiddenSize (\i -> fromIntegral (i `mod` 90) / 90.0) :: Vector Float
          weight = V.generate hiddenSize (\i -> -0.1 + fromIntegral (i `mod` 40) / 200.0) :: Vector Float
          zeroCentered = True  -- Gemma 3 mode

      let cpuOutput = cpuRMSNorm input weight zeroCentered

      putStrLn $ "=== RMSNormDSL Test 6 (Zero-Centered) ==="
      putStrLn $ "Hidden size: " ++ show hiddenSize
      putStrLn $ "Zero-centered mode: True (Gemma 3)"
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuOutput)

      gpuOutput <- evalContT $ runRMSNormDSL input weight hiddenSize zeroCentered

      putStrLn $ "GPU DSL output sum: " ++ show (V.sum gpuOutput)
      putStrLn $ "Mean error: " ++ show (meanAbsError cpuOutput gpuOutput)
      putStrLn $ "Max error: " ++ show (maxAbsError cpuOutput gpuOutput)

      let err = meanAbsError cpuOutput gpuOutput
      err `shouldSatisfy` (< 1e-5)

    -- Test 7: Large vector (realistic transformer size)
    it "DSL-based GPU RMSNorm matches CPU (large, realistic)" $ do
      let hiddenSize = 2048  -- Typical for Gemma models
          input = V.generate hiddenSize (\i -> sin (fromIntegral i / 200.0) * 0.5) :: Vector Float
          weight = V.generate hiddenSize (\i -> 1.0 + cos (fromIntegral i / 150.0) * 0.1) :: Vector Float
          zeroCentered = False

      let cpuOutput = cpuRMSNorm input weight zeroCentered

      putStrLn $ "=== RMSNormDSL Test 7 (Large, Realistic) ==="
      putStrLn $ "Hidden size: " ++ show hiddenSize
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuOutput)

      -- Test with full optimization (Vec4 + FP16)
      gpuOutput <- evalContT $ runRMSNormDSLWithPrecision input weight hiddenSize zeroCentered True True

      putStrLn $ "GPU optimized output sum: " ++ show (V.sum gpuOutput)
      putStrLn $ "Mean error: " ++ show (meanAbsError cpuOutput gpuOutput)
      putStrLn $ "Max error: " ++ show (maxAbsError cpuOutput gpuOutput)

      let err = meanAbsError cpuOutput gpuOutput
      err `shouldSatisfy` (< 5e-3)
