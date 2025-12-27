{-# LANGUAGE OverloadedStrings #-}

module Gemma.Layers.RoPEDSLSpec (spec) where

import Test.Hspec
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import Graphics.WebGPU.Dawn.ContT (evalContT)

import Gemma.Layers.RoPEDSL

-- | CPU implementation of RoPE for reference
--
-- Formula:
--   For each pair (2i, 2i+1):
--     freq = 1.0 / base^(i/headDim)
--     theta = position * freq
--     output[2i] = input[2i] * cos(theta) - input[2i+1] * sin(theta)
--     output[2i+1] = input[2i] * sin(theta) + input[2i+1] * cos(theta)
cpuRoPE :: Vector Float -> Int -> Int -> Int -> Float -> Vector Float
cpuRoPE input numHeads headDim position ropeBase =
  let size = numHeads * headDim
      pos = fromIntegral position

      applyRotation :: Int -> Float
      applyRotation idx =
        let headIdx = idx `div` headDim
            localIdx = idx `mod` headDim
            pairIdx = localIdx `div` 2
            isFirst = localIdx `mod` 2 == 0

            baseIdx = headIdx * headDim + pairIdx * 2
            x = input V.! baseIdx
            y = input V.! (baseIdx + 1)

            -- freq = 1.0 / base^(i/headDim)
            freqExp = fromIntegral (pairIdx * 2) / fromIntegral headDim
            freq = 1.0 / (ropeBase ** freqExp)
            theta = pos * freq

            cosTheta = cos theta
            sinTheta = sin theta

            x' = x * cosTheta - y * sinTheta
            y' = x * sinTheta + y * cosTheta

        in if isFirst then x' else y'

  in V.generate size applyRotation

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
  describe "RoPEDSL (DSL-based GPU RoPE)" $ do

    -- Test 1: Single head, position 0 (identity)
    it "DSL-based GPU RoPE at position 0 is identity (single head, FP32)" $ do
      let numHeads = 1
          headDim = 128
          position = 0
          ropeBase = 10000.0
          input = V.generate (numHeads * headDim) (\i -> sin (fromIntegral i / 10.0)) :: Vector Float

      -- At position 0, rotation should be identity
      let cpuOutput = cpuRoPE input numHeads headDim position ropeBase

      putStrLn $ "=== RoPEDSL Test 1 (Position 0, FP32) ==="
      putStrLn $ "Num heads: " ++ show numHeads ++ ", Head dim: " ++ show headDim
      putStrLn $ "Position: " ++ show position
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuOutput)

      -- GPU DSL-based RoPE
      gpuOutput <- evalContT $ runRoPEDSL input numHeads headDim position ropeBase

      putStrLn $ "GPU DSL output sum: " ++ show (V.sum gpuOutput)
      putStrLn $ "Mean error: " ++ show (meanAbsError cpuOutput gpuOutput)
      putStrLn $ "Max error: " ++ show (maxAbsError cpuOutput gpuOutput)

      -- Position 0 should be nearly identity (very small angle)
      let err = meanAbsError cpuOutput gpuOutput
      err `shouldSatisfy` (< 1e-4)

    -- Test 2: Single head, non-zero position
    it "DSL-based GPU RoPE matches CPU (single head, position 1, FP32)" $ do
      let numHeads = 1
          headDim = 128
          position = 1
          ropeBase = 10000.0
          input = V.generate (numHeads * headDim) (\i -> sin (fromIntegral i / 10.0)) :: Vector Float

      let cpuOutput = cpuRoPE input numHeads headDim position ropeBase

      putStrLn $ "=== RoPEDSL Test 2 (Position 1, FP32) ==="
      putStrLn $ "Num heads: " ++ show numHeads ++ ", Head dim: " ++ show headDim
      putStrLn $ "Position: " ++ show position
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuOutput)

      gpuOutput <- evalContT $ runRoPEDSL input numHeads headDim position ropeBase

      putStrLn $ "GPU DSL output sum: " ++ show (V.sum gpuOutput)
      putStrLn $ "Mean error: " ++ show (meanAbsError cpuOutput gpuOutput)
      putStrLn $ "Max error: " ++ show (maxAbsError cpuOutput gpuOutput)

      let err = meanAbsError cpuOutput gpuOutput
      err `shouldSatisfy` (< 1e-4)

    -- Test 3: Multiple heads, position 5
    it "DSL-based GPU RoPE matches CPU (multi-head, FP32)" $ do
      let numHeads = 8
          headDim = 64
          position = 5
          ropeBase = 10000.0
          input = V.generate (numHeads * headDim) (\i -> cos (fromIntegral i / 20.0)) :: Vector Float

      let cpuOutput = cpuRoPE input numHeads headDim position ropeBase

      putStrLn $ "=== RoPEDSL Test 3 (Multi-head, FP32) ==="
      putStrLn $ "Num heads: " ++ show numHeads ++ ", Head dim: " ++ show headDim
      putStrLn $ "Position: " ++ show position
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuOutput)

      gpuOutput <- evalContT $ runRoPEDSL input numHeads headDim position ropeBase

      putStrLn $ "GPU DSL output sum: " ++ show (V.sum gpuOutput)
      putStrLn $ "Mean error: " ++ show (meanAbsError cpuOutput gpuOutput)
      putStrLn $ "Max error: " ++ show (maxAbsError cpuOutput gpuOutput)

      let err = meanAbsError cpuOutput gpuOutput
      err `shouldSatisfy` (< 1e-4)

    -- Test 4: FP16 precision test
    it "DSL-based GPU RoPE with FP16 matches CPU (multi-head)" $ do
      let numHeads = 8
          headDim = 64
          position = 10
          ropeBase = 10000.0
          input = V.generate (numHeads * headDim) (\i -> sin (fromIntegral i / 30.0)) :: Vector Float

      let cpuOutput = cpuRoPE input numHeads headDim position ropeBase

      putStrLn $ "=== RoPEDSL Test 4 (FP16) ==="
      putStrLn $ "Num heads: " ++ show numHeads ++ ", Head dim: " ++ show headDim
      putStrLn $ "Position: " ++ show position
      putStrLn $ "CPU (FP32) output sum: " ++ show (V.sum cpuOutput)

      -- GPU with FP16
      gpuOutput <- evalContT $ runRoPEDSLWithPrecision True input numHeads headDim position ropeBase

      putStrLn $ "GPU FP16 output sum: " ++ show (V.sum gpuOutput)
      putStrLn $ "Mean error: " ++ show (meanAbsError cpuOutput gpuOutput)
      putStrLn $ "Max error: " ++ show (maxAbsError cpuOutput gpuOutput)

      -- FP16 loses some precision
      let err = meanAbsError cpuOutput gpuOutput
      err `shouldSatisfy` (< 5e-3)  -- 0.5% error tolerance for FP16

    -- Test 5: Large position (long context)
    it "DSL-based GPU RoPE matches CPU (large position, FP32)" $ do
      let numHeads = 4
          headDim = 128
          position = 1000  -- Long context
          ropeBase = 10000.0
          input = V.generate (numHeads * headDim) (\i -> fromIntegral (i `mod` 100) / 50.0 - 1.0) :: Vector Float

      let cpuOutput = cpuRoPE input numHeads headDim position ropeBase

      putStrLn $ "=== RoPEDSL Test 5 (Large Position) ==="
      putStrLn $ "Num heads: " ++ show numHeads ++ ", Head dim: " ++ show headDim
      putStrLn $ "Position: " ++ show position ++ " (long context)"
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuOutput)

      gpuOutput <- evalContT $ runRoPEDSL input numHeads headDim position ropeBase

      putStrLn $ "GPU DSL output sum: " ++ show (V.sum gpuOutput)
      putStrLn $ "Mean error: " ++ show (meanAbsError cpuOutput gpuOutput)
      putStrLn $ "Max error: " ++ show (maxAbsError cpuOutput gpuOutput)

      let err = meanAbsError cpuOutput gpuOutput
      err `shouldSatisfy` (< 1e-4)

    -- Test 6: Extended RoPE base (1M for very long context)
    it "DSL-based GPU RoPE with extended base matches CPU (FP32)" $ do
      let numHeads = 4
          headDim = 128
          position = 500
          ropeBase = 1000000.0  -- Extended base for long context
          input = V.generate (numHeads * headDim) (\i -> cos (fromIntegral i / 40.0)) :: Vector Float

      let cpuOutput = cpuRoPE input numHeads headDim position ropeBase

      putStrLn $ "=== RoPEDSL Test 6 (Extended Base) ==="
      putStrLn $ "Num heads: " ++ show numHeads ++ ", Head dim: " ++ show headDim
      putStrLn $ "Position: " ++ show position
      putStrLn $ "RoPE base: " ++ show ropeBase
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuOutput)

      gpuOutput <- evalContT $ runRoPEDSL input numHeads headDim position ropeBase

      putStrLn $ "GPU DSL output sum: " ++ show (V.sum gpuOutput)
      putStrLn $ "Mean error: " ++ show (meanAbsError cpuOutput gpuOutput)
      putStrLn $ "Max error: " ++ show (maxAbsError cpuOutput gpuOutput)

      let err = meanAbsError cpuOutput gpuOutput
      err `shouldSatisfy` (< 1e-4)
