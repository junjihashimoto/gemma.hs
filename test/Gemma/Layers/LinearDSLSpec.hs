{-# LANGUAGE OverloadedStrings #-}

module Gemma.Layers.LinearDSLSpec (spec) where

import Test.Hspec
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import Graphics.WebGPU.Dawn.ContT (evalContT)
import Data.Word (Word32)
import Data.Bits ((.&.), (.|.), shiftL)

import Gemma.Layers.LinearDSL
import Gemma.Utils.Half (vectorFloatToHalf, vectorHalfToFloat)

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

-- Helper: Quantize weights to Q4 format (for testing)
-- Quantization: nibble = clamp(round(weight / scale + 7.5), 0, 15)
-- Dequantization: weight = (nibble - 7.5) * scale
quantizeToQ4 :: Vector Float -> Int -> Q4Weights
quantizeToQ4 weights blockSize =
  let numWeights = V.length weights
      numBlocks = (numWeights + blockSize - 1) `div` blockSize
      nibblesPerU32 = 8

      -- Calculate scale for each block (max absolute value in block)
      scales = V.generate numBlocks $ \blockIdx ->
        let blockStart = blockIdx * blockSize
            blockEnd = min (blockStart + blockSize) numWeights
            blockWeights = V.slice blockStart (blockEnd - blockStart) weights
            maxAbs = V.maximum (V.map abs blockWeights)
        in if maxAbs < 1e-7 then 0.0 else maxAbs / 7.5

      -- Quantize each weight to 4-bit nibble
      nibbles = V.generate numWeights $ \i ->
        let blockIdx = i `div` blockSize
            scale = scales V.! blockIdx
            weight = weights V.! i
            normalized = if scale < 1e-7 then 7.5 else weight / scale + 7.5
            clamped = max 0 (min 15 (round normalized :: Int))
        in fromIntegral clamped :: Word32

      -- Pack 8 nibbles into each Word32
      numPacked = (numWeights + nibblesPerU32 - 1) `div` nibblesPerU32
      packed = V.generate numPacked $ \wordIdx ->
        let base = wordIdx * nibblesPerU32
            packNibble offset =
              if base + offset < numWeights
                then (nibbles V.! (base + offset)) `shiftL` (offset * 4)
                else 0
        in foldr (.|.) 0 [packNibble i | i <- [0..nibblesPerU32-1]]

  in Q4Weights packed scales

spec :: Spec
spec = do
  describe "LinearDSL (DSL-based GPU Linear Layer)" $ do

    -- Test 1: Small matrix to verify correctness
    it "DSL-based GPU linear matches CPU linear (small matrix)" $ do
      -- Test: [4x8] @ [8] = [4]
      let outSize = 4
          inSize = 8
          weights = V.fromList [1,2,3,4,5,6,7,8,  -- row 0
                                9,10,11,12,13,14,15,16,  -- row 1
                                1,1,1,1,1,1,1,1,  -- row 2
                                0,0,0,0,1,1,1,1] :: Vector Float  -- row 3
          input = V.fromList [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0] :: Vector Float

      -- CPU baseline
      let cpuOutput = cpuLinear weights input outSize inSize

      -- DEBUG: Print expected output
      putStrLn $ "=== LinearDSL Test 1 (Small Matrix) ==="
      putStrLn $ "Weight shape: [" ++ show outSize ++ " x " ++ show inSize ++ "]"
      putStrLn $ "Input: " ++ show (V.toList input)
      putStrLn $ "CPU output: " ++ show (V.toList cpuOutput)

      -- GPU DSL-based linear
      gpuOutput <- evalContT $ runLinearDSL weights input outSize inSize

      -- DEBUG: Print GPU output
      putStrLn $ "GPU DSL output: " ++ show (V.toList gpuOutput)
      putStrLn $ "Mean error: " ++ show (meanAbsError cpuOutput gpuOutput)
      putStrLn $ "Max error: " ++ show (maxAbsError cpuOutput gpuOutput)

      -- Should match exactly (or very close due to floating point)
      let error = meanAbsError cpuOutput gpuOutput
      error `shouldSatisfy` (< 1e-5)

    -- Test 2: Larger matrix
    it "DSL-based GPU linear matches CPU linear (medium matrix)" $ do
      -- Test: [64x128] @ [128] = [64]
      let outSize = 64
          inSize = 128
          weights = V.generate (outSize * inSize) (\i -> sin (fromIntegral i / 100.0)) :: Vector Float
          input = V.generate inSize (\i -> cos (fromIntegral i / 50.0)) :: Vector Float

      -- CPU baseline
      let cpuOutput = cpuLinear weights input outSize inSize

      putStrLn $ "=== LinearDSL Test 2 (Medium Matrix) ==="
      putStrLn $ "Weight shape: [" ++ show outSize ++ " x " ++ show inSize ++ "]"
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuOutput)

      -- GPU DSL-based linear
      gpuOutput <- evalContT $ runLinearDSL weights input outSize inSize

      putStrLn $ "GPU DSL output sum: " ++ show (V.sum gpuOutput)
      putStrLn $ "Mean error: " ++ show (meanAbsError cpuOutput gpuOutput)
      putStrLn $ "Max error: " ++ show (maxAbsError cpuOutput gpuOutput)

      -- Should match closely
      let error = meanAbsError cpuOutput gpuOutput
      error `shouldSatisfy` (< 1e-4)

    -- Test 3: Large matrix (stress test)
    it "DSL-based GPU linear matches CPU linear (large matrix)" $ do
      -- Test: [256x512] @ [512] = [256]
      let outSize = 256
          inSize = 512
          weights = V.generate (outSize * inSize) (\i -> fromIntegral (i `mod` 100) / 100.0) :: Vector Float
          input = V.generate inSize (\i -> fromIntegral (i `mod` 50) / 50.0) :: Vector Float

      -- CPU baseline
      let cpuOutput = cpuLinear weights input outSize inSize

      putStrLn $ "=== LinearDSL Test 3 (Large Matrix) ==="
      putStrLn $ "Weight shape: [" ++ show outSize ++ " x " ++ show inSize ++ "]"
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuOutput)

      -- GPU DSL-based linear
      gpuOutput <- evalContT $ runLinearDSL weights input outSize inSize

      putStrLn $ "GPU DSL output sum: " ++ show (V.sum gpuOutput)
      putStrLn $ "Mean error: " ++ show (meanAbsError cpuOutput gpuOutput)
      putStrLn $ "Max error: " ++ show (maxAbsError cpuOutput gpuOutput)

      -- Should match closely
      let error = meanAbsError cpuOutput gpuOutput
      error `shouldSatisfy` (< 1e-4)

    -- Test 4: FP16 precision (uses f16 for weights/input, accumulates in f32)
    it "DSL-based GPU linear with FP16 matches CPU (medium matrix)" $ do
      -- Test: [32x64] @ [64] = [32]
      let outSize = 32
          inSize = 64
          weights = V.generate (outSize * inSize) (\i -> sin (fromIntegral i / 50.0)) :: Vector Float
          input = V.generate inSize (\i -> cos (fromIntegral i / 25.0)) :: Vector Float

      -- CPU baseline (FP32)
      let cpuOutput = cpuLinear weights input outSize inSize

      putStrLn $ "=== LinearDSL Test 4 (FP16 Precision) ==="
      putStrLn $ "Weight shape: [" ++ show outSize ++ " x " ++ show inSize ++ "]"
      putStrLn $ "CPU (FP32) output sum: " ++ show (V.sum cpuOutput)

      -- GPU FP16-based linear
      gpuOutput <- evalContT $ runLinearDSLWithPrecision weights input outSize inSize True False

      putStrLn $ "GPU (FP16) output sum: " ++ show (V.sum gpuOutput)
      putStrLn $ "Mean error: " ++ show (meanAbsError cpuOutput gpuOutput)
      putStrLn $ "Max error: " ++ show (maxAbsError cpuOutput gpuOutput)

      -- FP16 has lower precision, so allow larger error
      let err = meanAbsError cpuOutput gpuOutput
      err `shouldSatisfy` (< 0.01)  -- ~1% error acceptable for FP16

    -- Test 5: Vec4 SIMD with FP32 (4x speedup)
    it "DSL-based GPU linear with Vec4 SIMD (FP32) matches CPU" $ do
      -- Test: [64x128] @ [128] = [64]
      -- inSize=128 is divisible by 4, perfect for Vec4
      let outSize = 64
          inSize = 128
          weights = V.generate (outSize * inSize) (\i -> sin (fromIntegral i / 100.0)) :: Vector Float
          input = V.generate inSize (\i -> cos (fromIntegral i / 50.0)) :: Vector Float

      -- CPU baseline
      let cpuOutput = cpuLinear weights input outSize inSize

      putStrLn $ "=== LinearDSL Test 5 (Vec4 FP32) ==="
      putStrLn $ "Weight shape: [" ++ show outSize ++ " x " ++ show inSize ++ "]"
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuOutput)

      -- GPU Vec4-based linear (FP32)
      gpuOutput <- evalContT $ runLinearDSLWithPrecision weights input outSize inSize False True

      putStrLn $ "GPU Vec4 output sum: " ++ show (V.sum gpuOutput)
      putStrLn $ "Mean error: " ++ show (meanAbsError cpuOutput gpuOutput)
      putStrLn $ "Max error: " ++ show (maxAbsError cpuOutput gpuOutput)

      -- Should match closely (same precision as scalar FP32)
      let error = meanAbsError cpuOutput gpuOutput
      error `shouldSatisfy` (< 1e-4)

    -- Test 6: Vec4 SIMD with FP16 (2x memory bandwidth + 4x SIMD)
    it "DSL-based GPU linear with Vec4 SIMD (FP16) matches CPU" $ do
      -- Test: [32x64] @ [64] = [32]
      -- inSize=64 is divisible by 4, perfect for Vec4
      let outSize = 32
          inSize = 64
          weights = V.generate (outSize * inSize) (\i -> sin (fromIntegral i / 50.0)) :: Vector Float
          input = V.generate inSize (\i -> cos (fromIntegral i / 25.0)) :: Vector Float

      -- CPU baseline (FP32)
      let cpuOutput = cpuLinear weights input outSize inSize

      putStrLn $ "=== LinearDSL Test 6 (Vec4 FP16) ==="
      putStrLn $ "Weight shape: [" ++ show outSize ++ " x " ++ show inSize ++ "]"
      putStrLn $ "CPU (FP32) output sum: " ++ show (V.sum cpuOutput)

      -- GPU Vec4 + FP16-based linear
      gpuOutput <- evalContT $ runLinearDSLWithPrecision weights input outSize inSize True True

      putStrLn $ "GPU Vec4+FP16 output sum: " ++ show (V.sum gpuOutput)
      putStrLn $ "Mean error: " ++ show (meanAbsError cpuOutput gpuOutput)
      putStrLn $ "Max error: " ++ show (maxAbsError cpuOutput gpuOutput)

      -- FP16 has lower precision, so allow larger error
      let err = meanAbsError cpuOutput gpuOutput
      err `shouldSatisfy` (< 0.01)  -- ~1% error acceptable for FP16

    -- Test 7: MMA (Subgroup Matrix Multiply-Accumulate) - Hardware-accelerated
    --
    -- NOTE: Based on WebGPU spec (github.com/gpuweb/gpuweb/proposals/subgroup-matrix.md):
    --   - Subgroup matrices use 8×8 tiles (Metal: simdgroup_float8x8)
    --   - Designed for matrix-matrix multiplication
    --   - Single vector [inSize, 1] wastes 7/8 of tile → inefficient
    --
    -- RECOMMENDATION: Use MMA for:
    --   ✅ Batched linear layers: [batch, outSize] = W @ [batch, inSize]^T (batch ≥ 8)
    --   ✅ Attention layers: Q @ K^T, Attn @ V (natural matrix-matrix ops)
    --   ❌ Single vector linear: Use Vec4 SIMD instead (~4x faster for batch=1)
    --
    -- This test is pending until we implement batched linear layer support
    xit "DSL-based GPU linear with MMA (subgroup matrices) matches CPU" $ do
      -- Test: [64x128] @ [128] = [64]
      -- Dimensions must be multiples of 8 for MMA 8x8 tiles
      let outSize = 64
          inSize = 128
          weights = V.generate (outSize * inSize) (\i -> sin (fromIntegral i / 100.0)) :: Vector Float
          input = V.generate inSize (\i -> cos (fromIntegral i / 50.0)) :: Vector Float

      -- CPU baseline (FP32)
      let cpuOutput = cpuLinear weights input outSize inSize

      putStrLn $ "=== LinearDSL Test 7 (MMA Subgroup Matrices) ==="
      putStrLn $ "Weight shape: [" ++ show outSize ++ " x " ++ show inSize ++ "]"
      putStrLn $ "CPU (FP32) output sum: " ++ show (V.sum cpuOutput)
      putStrLn $ "Note: Requires chromium_experimental_subgroup_matrix support"

      -- GPU MMA-based linear
      gpuOutput <- evalContT $ runLinearDSLWithMMA weights input outSize inSize

      putStrLn $ "GPU MMA output sum: " ++ show (V.sum gpuOutput)
      putStrLn $ "Mean error: " ++ show (meanAbsError cpuOutput gpuOutput)
      putStrLn $ "Max error: " ++ show (maxAbsError cpuOutput gpuOutput)

      -- MMA uses FP16 internally, so allow FP16-level error
      let err = meanAbsError cpuOutput gpuOutput
      err `shouldSatisfy` (< 0.01)  -- ~1% error acceptable for FP16

    -- ═══════════════════════════════════════════════════════════════
    -- UNIFIED CONFIG INTERFACE (Phase 2)
    -- ═══════════════════════════════════════════════════════════════

    -- Test 8: Unified config with FP32 precision
    it "Unified config: FP32 precision matches CPU" $ do
      -- Test: [32x64] @ [64] = [32]
      let outSize = 32
          inSize = 64
          weights = V.generate (outSize * inSize) (\i -> sin (fromIntegral i / 50.0)) :: Vector Float
          input = V.generate inSize (\i -> cos (fromIntegral i / 25.0)) :: Vector Float
          config = defaultLinearConfig { lcPrecision = FP32 }

      -- CPU baseline
      let cpuOutput = cpuLinear weights input outSize inSize

      putStrLn $ "\n=== LinearDSL Test 8 (Unified Config: FP32) ==="
      putStrLn $ "Weight shape: [" ++ show outSize ++ " x " ++ show inSize ++ "]"
      putStrLn $ "Config: " ++ show config
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuOutput)

      -- GPU with unified config
      gpuOutput <- evalContT $ runLinearWithConfig config (Left weights) input outSize inSize

      putStrLn $ "GPU (unified config) output sum: " ++ show (V.sum gpuOutput)
      putStrLn $ "Mean error: " ++ show (meanAbsError cpuOutput gpuOutput)
      putStrLn $ "Max error: " ++ show (maxAbsError cpuOutput gpuOutput)

      -- Should match closely (same as Test 1-3)
      let error = meanAbsError cpuOutput gpuOutput
      error `shouldSatisfy` (< 1e-4)

    -- Test 9: Unified config with FP16 precision
    it "Unified config: FP16 precision matches CPU" $ do
      -- Test: [32x64] @ [64] = [32]
      let outSize = 32
          inSize = 64
          weights = V.generate (outSize * inSize) (\i -> sin (fromIntegral i / 50.0)) :: Vector Float
          input = V.generate inSize (\i -> cos (fromIntegral i / 25.0)) :: Vector Float
          config = defaultLinearConfig { lcPrecision = FP16 }

      -- CPU baseline
      let cpuOutput = cpuLinear weights input outSize inSize

      putStrLn $ "\n=== LinearDSL Test 9 (Unified Config: FP16) ==="
      putStrLn $ "Weight shape: [" ++ show outSize ++ " x " ++ show inSize ++ "]"
      putStrLn $ "Config: " ++ show config
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuOutput)

      -- GPU with unified config
      gpuOutput <- evalContT $ runLinearWithConfig config (Left weights) input outSize inSize

      putStrLn $ "GPU (unified config FP16) output sum: " ++ show (V.sum gpuOutput)
      putStrLn $ "Mean error: " ++ show (meanAbsError cpuOutput gpuOutput)
      putStrLn $ "Max error: " ++ show (maxAbsError cpuOutput gpuOutput)

      -- FP16 has lower precision (same as Test 4)
      let err = meanAbsError cpuOutput gpuOutput
      err `shouldSatisfy` (< 0.01)  -- ~1% error acceptable for FP16

    -- Test 10: Unified config with Q4 precision
    it "Unified config: Q4 precision matches CPU (with quantization error)" $ do
      -- Test: [32x64] @ [64] = [32]
      -- Note: inSize must be multiple of blockSize (32)
      let outSize = 32
          inSize = 64
          blockSize = 32
          weights = V.generate (outSize * inSize) (\i -> sin (fromIntegral i / 50.0) * 0.5) :: Vector Float
          input = V.generate inSize (\i -> cos (fromIntegral i / 25.0)) :: Vector Float
          config = defaultLinearConfig { lcPrecision = Q4, lcQ4BlockSize = blockSize }

      -- Quantize weights to Q4
      let q4Weights = quantizeToQ4 weights blockSize

      -- CPU baseline (using FP32 weights)
      let cpuOutput = cpuLinear weights input outSize inSize

      putStrLn $ "\n=== LinearDSL Test 10 (Unified Config: Q4) ==="
      putStrLn $ "Weight shape: [" ++ show outSize ++ " x " ++ show inSize ++ "]"
      putStrLn $ "Config: " ++ show config
      putStrLn $ "Block size: " ++ show blockSize
      putStrLn $ "CPU (FP32) output sum: " ++ show (V.sum cpuOutput)

      -- GPU with unified config (Q4)
      gpuOutput <- evalContT $ runLinearWithConfig config (Right q4Weights) input outSize inSize

      putStrLn $ "GPU (unified config Q4) output sum: " ++ show (V.sum gpuOutput)
      putStrLn $ "Mean error: " ++ show (meanAbsError cpuOutput gpuOutput)
      putStrLn $ "Max error: " ++ show (maxAbsError cpuOutput gpuOutput)

      -- Q4 has quantization error: 4-bit precision means ~6% error
      -- (each weight stored in 4 bits = 16 levels, range -7.5 to +7.5)
      let err = meanAbsError cpuOutput gpuOutput
      err `shouldSatisfy` (< 0.1)  -- Allow up to 10% error for Q4 quantization
