{-# LANGUAGE OverloadedStrings #-}

module Gemma.Layers.LinearQ4Spec (spec) where

import Test.Hspec
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import Graphics.WebGPU.Dawn.ContT (evalContT)

-- Import Q4 functions
import Gemma.Quantization.Q4
import Gemma.Layers.LinearQ4

-- Helper: Calculate mean absolute error
meanAbsError :: Vector Float -> Vector Float -> Float
meanAbsError v1 v2 =
  let diffs = V.zipWith (\a b -> abs (a - b)) v1 v2
  in V.sum diffs / fromIntegral (V.length diffs)

spec :: Spec
spec = do
  describe "Q4 Linear Layer (GPU Shader)" $ do

    -- Test 1: Q4 GPU matches CPU quantized result
    describe "Q4 GPU vs CPU Quantization" $ do
      it "Q4 GPU linear matches CPU quantized linear (small matrix)" $ do
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

        -- DEBUG: Print CPU output
        putStrLn $ "=== DEBUG Q4 Test 1 ==="
        putStrLn $ "Input sum should be: " ++ show (V.sum input)
        putStrLn $ "First 10 inputs: " ++ show (V.toList $ V.take 10 input)
        putStrLn $ "CPU output: " ++ show (V.toList cpuOutput)
        putStrLn $ "First 10 dequantized weights: " ++ show (V.toList $ V.take 10 dequantizedWeights)
        putStrLn $ "First 2 scales: " ++ show (V.toList $ V.take 2 scales)
        putStrLn $ "First 2 packed: " ++ show (V.toList $ V.take 2 packedWeights)

        -- GPU Q4 linear
        gpuOutput <- evalContT $ runLinearQ4GPU packedWeights scales input outSize inSize

        -- DEBUG: Print GPU output
        putStrLn $ "GPU output: " ++ show (V.toList gpuOutput)
        putStrLn $ "Error: " ++ show (meanAbsError cpuOutput gpuOutput)

        -- Should match closely (small numerical differences OK)
        let error = meanAbsError cpuOutput gpuOutput
        error `shouldSatisfy` (< 0.01)

      it "Q4 GPU linear matches CPU quantized linear (larger matrix)" $ do
        -- Test: [128x128] @ [128] = [128]
        let outSize = 128
            inSize = 128  -- 4 blocks per row
            weights = V.generate (outSize * inSize) (\i -> sin (fromIntegral i / 100.0)) :: Vector Float
            input = V.generate inSize (\i -> cos (fromIntegral i / 50.0)) :: Vector Float

        -- CPU quantization
        let (packedWeights, scales) = quantizeQ4 weights outSize inSize

        -- CPU dequantized linear (baseline)
        let dequantizedWeights = dequantizeQ4 packedWeights scales
            cpuOutput = cpuLinear dequantizedWeights input outSize inSize

        -- GPU Q4 linear
        gpuOutput <- evalContT $ runLinearQ4GPU packedWeights scales input outSize inSize

        -- Should match closely
        let error = meanAbsError cpuOutput gpuOutput
        error `shouldSatisfy` (< 0.01)

    -- Test 2: Q4 quantization error is acceptable
    describe "Q4 Quantization Error" $ do
      it "Q4 linear has <5% error vs FP32 linear (typical weights)" $ do
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
        q4Output <- evalContT $ runLinearQ4GPU packedWeights scales input outSize inSize

        -- Calculate relative error
        let absError = meanAbsError fp32Output q4Output
            maxVal = V.maximum (V.map abs fp32Output)
            relativeError = absError / maxVal

        -- Should have <5% relative error
        relativeError `shouldSatisfy` (< 0.05)

    -- Test 3: Q4 shader correctness on edge cases
    describe "Q4 Edge Cases" $ do
      it "handles zero weights correctly" $ do
        let outSize = 8
            inSize = 32
            weights = V.replicate (outSize * inSize) 0.0 :: Vector Float
            input = V.generate inSize (\i -> fromIntegral i) :: Vector Float

        let (packedWeights, scales) = quantizeQ4 weights outSize inSize
        gpuOutput <- evalContT $ runLinearQ4GPU packedWeights scales input outSize inSize

        -- All outputs should be zero
        V.all (< 0.01) (V.map abs gpuOutput) `shouldBe` True

      it "handles zero input correctly" $ do
        let outSize = 8
            inSize = 32
            weights = V.generate (outSize * inSize) (\i -> fromIntegral i) :: Vector Float
            input = V.replicate inSize 0.0 :: Vector Float

        let (packedWeights, scales) = quantizeQ4 weights outSize inSize
        gpuOutput <- evalContT $ runLinearQ4GPU packedWeights scales input outSize inSize

        -- All outputs should be zero
        V.all (< 0.01) (V.map abs gpuOutput) `shouldBe` True

      it "handles large values without overflow" $ do
        let outSize = 16
            inSize = 64
            weights = V.replicate (outSize * inSize) 10.0 :: Vector Float
            input = V.replicate inSize 5.0 :: Vector Float

        let (packedWeights, scales) = quantizeQ4 weights outSize inSize
        gpuOutput <- evalContT $ runLinearQ4GPU packedWeights scales input outSize inSize

        -- Check no NaN/Inf
        V.all (\x -> not (isNaN x) && not (isInfinite x)) gpuOutput `shouldBe` True

    -- Test 4: Different matrix sizes
    describe "Q4 Matrix Sizes" $ do
      it "handles 1x32 matrix (single row)" $ do
        pending
        -- let outSize = 1
        --     inSize = 32
        --     weights = V.generate inSize (\i -> fromIntegral i) :: Vector Float
        --     input = V.generate inSize (\i -> fromIntegral (i `mod` 5)) :: Vector Float
        -- let (packedWeights, scales) = quantizeQ4 weights outSize inSize
        -- gpuOutput <- evalContT $ runLinearQ4GPU packedWeights scales input outSize inSize
        -- V.length gpuOutput `shouldBe` outSize

      it "handles 256x128 matrix (larger)" $ do
        pending
        -- let outSize = 256
        --     inSize = 128
        --     weights = V.generate (outSize * inSize) (\i -> sin (fromIntegral i / 50.0)) :: Vector Float
        --     input = V.generate inSize (\i -> fromIntegral i / 100.0) :: Vector Float
        -- let (packedWeights, scales) = quantizeQ4 weights outSize inSize
        -- gpuOutput <- evalContT $ runLinearQ4GPU packedWeights scales input outSize inSize
        -- V.length gpuOutput `shouldBe` outSize

-- Helper: CPU matrix-vector multiplication
cpuLinear :: Vector Float -> Vector Float -> Int -> Int -> Vector Float
cpuLinear weights input outSize inSize =
  V.generate outSize $ \row ->
    let rowStart = row * inSize
        rowWeights = V.slice rowStart inSize weights
    in V.sum $ V.zipWith (*) rowWeights input
