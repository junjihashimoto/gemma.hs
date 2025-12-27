{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

{-|
Module: Gemma.Layers.RMSNormDSL
Description: DSL-based Root Mean Square Layer Normalization

RMSNorm is a simpler alternative to LayerNorm that normalizes using only
the root mean square (RMS) of the inputs.

Formula:
  RMSNorm(x) = (x / RMS(x)) * weight
  where RMS(x) = sqrt(mean(x²) + ε)

For Gemma 3 (zero-centered):
  RMSNorm(x) = (x / RMS(x)) * (1 + weight)

This DSL implementation provides:
  - Type-safe workgroup reduction for RMS computation
  - FP16 support for 2x memory bandwidth
  - Vec4 SIMD for 4x additional speedup
  - Zero-centered mode for Gemma 3 compatibility

Reference: https://arxiv.org/abs/1910.07467
-}

module Gemma.Layers.RMSNormDSL
  ( -- * Main RMSNorm Functions
    runRMSNormDSL
  , runRMSNormDSLWithPrecision
    -- * Component Kernels
  , rmsNormKernelDSL
  , rmsNormKernelFP16
  , rmsNormKernelFP32
  ) where

import Graphics.WebGPU.Dawn.ContT
import Graphics.WebGPU.Dawn (WGPUFeatureName(FeatureShaderF16))
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import WGSL.DSL hiding ((<), (>), (<=), (>=), (==), (/=), (&&), (||), not)
import qualified WGSL.DSL as DSL
import WGSL.Execute (executeShaderNamed)
import Graphics.WebGPU.Dawn.Types (AnyTensor(..), Half)
import Gemma.Utils.Half (vectorFloatToHalf, vectorHalfToFloat)
import Prelude

-- | RMSNorm kernel with DSL
--
-- Computes: output = (input / RMS(input)) * weight
-- where RMS(input) = sqrt(mean(input²) + ε)
--
-- Uses workgroup shared memory for efficient parallel reduction.
--
-- Parameters:
-- - hiddenSize: Size of the input vector
-- - useFP16: When True, uses FP16 for 2x memory bandwidth
-- - zeroCentered: When True, uses (1 + weight) for Gemma 3
-- - useVec4: When True, uses vec4 SIMD for 4x speedup
rmsNormKernelDSL :: Int -> Bool -> Bool -> Bool -> ShaderM ()
rmsNormKernelDSL hiddenSize useFP16 zeroCentered useVec4 = do
  if useFP16
    then rmsNormKernelFP16 hiddenSize zeroCentered useVec4
    else rmsNormKernelFP32 hiddenSize zeroCentered useVec4

-- | FP16 version of RMSNorm kernel
rmsNormKernelFP16 :: Int -> Bool -> Bool -> ShaderM ()
rmsNormKernelFP16 hiddenSize zeroCentered useVec4 = do
  -- Declare buffers with FP16 storage
  input <- declareInputBuffer "input" (TArray hiddenSize TF16)
  weight <- declareInputBuffer "weight" (TArray hiddenSize TF16)
  output <- declareOutputBuffer "output" (TArray hiddenSize TF16)

  -- Declare shared memory for parallel reduction
  sharedSum <- sharedNamed "shared_sum" (TArray 256 TF32)  -- Use FP32 for accumulation

  lid <- localId
  let tid = vecX lid

  -- Phase 1: Compute sum of squares with parallel reduction
  -- Accumulate in FP32 for precision
  sumSq <- var TF32 (litF32 0.0)

  if useVec4 && (hiddenSize `mod` 4 == 0)
    then do
      -- Vec4 SIMD path: process 4 elements at a time
      let vec4Iters = hiddenSize `div` 4
      loop (litI32 0) (litI32 vec4Iters) (litI32 1) $ \i -> do
        let cond = (i32 tid + i * litI32 256) DSL.< litI32 vec4Iters
        if_ cond
          (do
            let idx = (i32 tid + i * litI32 256) * litI32 4

            -- Load 4 elements and convert to FP32
            v0 <- readBuffer input idx
            v1 <- readBuffer input (idx + litI32 1)
            v2 <- readBuffer input (idx + litI32 2)
            v3 <- readBuffer input (idx + litI32 3)

            let v0_f32 = F16ToF32 v0
            let v1_f32 = F16ToF32 v1
            let v2_f32 = F16ToF32 v2
            let v3_f32 = F16ToF32 v3

            -- Compute squares and accumulate
            currentSum <- readPtr sumSq
            sumSq <== currentSum + (v0_f32 * v0_f32) + (v1_f32 * v1_f32) +
                                    (v2_f32 * v2_f32) + (v3_f32 * v3_f32)
          )
          (return ())

    else do
      -- Scalar path
      loop (litI32 0) (litI32 ((hiddenSize + 255) `div` 256)) (litI32 1) $ \i -> do
        let idx = i32 tid + i * litI32 256
        let cond = idx DSL.< litI32 hiddenSize
        if_ cond
          (do
            val <- readBuffer input idx
            let val_f32 = F16ToF32 val
            currentSum <- readPtr sumSq
            sumSq <== currentSum + (val_f32 * val_f32)
          )
          (return ())

  -- Store partial sum to shared memory
  sum_val <- readPtr sumSq
  writeWorkgroup sharedSum (i32 tid) sum_val

  barrier

  -- Parallel reduction in shared memory
  -- Reduce 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
  let reductionLoop stride = do
        let cond = i32 tid DSL.< litI32 stride
        if_ cond
          (do
            a <- readWorkgroup sharedSum (i32 tid)
            b <- readWorkgroup sharedSum (i32 tid + litI32 stride)
            writeWorkgroup sharedSum (i32 tid) (a + b)
          )
          (return ())
        barrier

  reductionLoop 128
  reductionLoop 64
  reductionLoop 32
  reductionLoop 16
  reductionLoop 8
  reductionLoop 4
  reductionLoop 2
  reductionLoop 1

  -- Thread 0 computes RMS and stores back to shared memory
  if_ (i32 tid DSL.== litI32 0)
    (do
      totalSum <- readWorkgroup sharedSum (litI32 0)
      let meanSq = totalSum / litF32 (fromIntegral hiddenSize)
      let rms = sqrt' (meanSq + litF32 1e-6)
      writeWorkgroup sharedSum (litI32 0) rms
    )
    (return ())

  barrier

  -- Phase 2: Normalize and apply weights
  rms <- readWorkgroup sharedSum (litI32 0)

  if useVec4 && (hiddenSize `mod` 4 == 0)
    then do
      -- Vec4 SIMD path
      let vec4Iters = hiddenSize `div` 4
      loop (litI32 0) (litI32 vec4Iters) (litI32 1) $ \i -> do
        let cond = (i32 tid + i * litI32 256) DSL.< litI32 vec4Iters
        if_ cond
          (do
            let idx = (i32 tid + i * litI32 256) * litI32 4

            -- Load input and weight
            in0 <- readBuffer input idx
            in1 <- readBuffer input (idx + litI32 1)
            in2 <- readBuffer input (idx + litI32 2)
            in3 <- readBuffer input (idx + litI32 3)

            w0 <- readBuffer weight idx
            w1 <- readBuffer weight (idx + litI32 1)
            w2 <- readBuffer weight (idx + litI32 2)
            w3 <- readBuffer weight (idx + litI32 3)

            -- Convert to FP32 for computation
            let in0_f32 = F16ToF32 in0
            let in1_f32 = F16ToF32 in1
            let in2_f32 = F16ToF32 in2
            let in3_f32 = F16ToF32 in3

            let w0_f32 = F16ToF32 w0
            let w1_f32 = F16ToF32 w1
            let w2_f32 = F16ToF32 w2
            let w3_f32 = F16ToF32 w3

            -- Normalize and apply weight
            let norm0_f32 = if zeroCentered
                  then (in0_f32 / rms) * (litF32 1.0 + w0_f32)
                  else (in0_f32 / rms) * w0_f32
            let norm1_f32 = if zeroCentered
                  then (in1_f32 / rms) * (litF32 1.0 + w1_f32)
                  else (in1_f32 / rms) * w1_f32
            let norm2_f32 = if zeroCentered
                  then (in2_f32 / rms) * (litF32 1.0 + w2_f32)
                  else (in2_f32 / rms) * w2_f32
            let norm3_f32 = if zeroCentered
                  then (in3_f32 / rms) * (litF32 1.0 + w3_f32)
                  else (in3_f32 / rms) * w3_f32

            -- Convert back to FP16 and write
            writeBuffer output idx (F32ToF16 norm0_f32)
            writeBuffer output (idx + litI32 1) (F32ToF16 norm1_f32)
            writeBuffer output (idx + litI32 2) (F32ToF16 norm2_f32)
            writeBuffer output (idx + litI32 3) (F32ToF16 norm3_f32)
          )
          (return ())

    else do
      -- Scalar path
      loop (litI32 0) (litI32 ((hiddenSize + 255) `div` 256)) (litI32 1) $ \i -> do
        let idx = i32 tid + i * litI32 256
        let cond = idx DSL.< litI32 hiddenSize
        if_ cond
          (do
            inVal <- readBuffer input idx
            wVal <- readBuffer weight idx

            let in_f32 = F16ToF32 inVal
            let w_f32 = F16ToF32 wVal

            let norm_f32 = if zeroCentered
                  then (in_f32 / rms) * (litF32 1.0 + w_f32)
                  else (in_f32 / rms) * w_f32

            writeBuffer output idx (F32ToF16 norm_f32)
          )
          (return ())

-- | FP32 version of RMSNorm kernel (similar structure)
rmsNormKernelFP32 :: Int -> Bool -> Bool -> ShaderM ()
rmsNormKernelFP32 hiddenSize zeroCentered useVec4 = do
  -- Declare buffers with FP32 storage
  input <- declareInputBuffer "input" (TArray hiddenSize TF32)
  weight <- declareInputBuffer "weight" (TArray hiddenSize TF32)
  output <- declareOutputBuffer "output" (TArray hiddenSize TF32)

  -- Declare shared memory for parallel reduction
  sharedSum <- sharedNamed "shared_sum" (TArray 256 TF32)

  lid <- localId
  let tid = vecX lid

  -- Phase 1: Compute sum of squares
  sumSq <- var TF32 (litF32 0.0)

  if useVec4 && (hiddenSize `mod` 4 == 0)
    then do
      let vec4Iters = hiddenSize `div` 4
      loop (litI32 0) (litI32 vec4Iters) (litI32 1) $ \i -> do
        let cond = (i32 tid + i * litI32 256) DSL.< litI32 vec4Iters
        if_ cond
          (do
            let idx = (i32 tid + i * litI32 256) * litI32 4
            v0 <- readBuffer input idx
            v1 <- readBuffer input (idx + litI32 1)
            v2 <- readBuffer input (idx + litI32 2)
            v3 <- readBuffer input (idx + litI32 3)

            currentSum <- readPtr sumSq
            sumSq <== currentSum + (v0 * v0) + (v1 * v1) + (v2 * v2) + (v3 * v3)
          )
          (return ())
    else do
      loop (litI32 0) (litI32 ((hiddenSize + 255) `div` 256)) (litI32 1) $ \i -> do
        let idx = i32 tid + i * litI32 256
        let cond = idx DSL.< litI32 hiddenSize
        if_ cond
          (do
            val <- readBuffer input idx
            currentSum <- readPtr sumSq
            sumSq <== currentSum + (val * val)
          )
          (return ())

  sum_val <- readPtr sumSq
  writeWorkgroup sharedSum (i32 tid) sum_val
  barrier

  -- Reduction (same as FP16 version)
  let reductionLoop stride = do
        let cond = i32 tid DSL.< litI32 stride
        if_ cond
          (do
            a <- readWorkgroup sharedSum (i32 tid)
            b <- readWorkgroup sharedSum (i32 tid + litI32 stride)
            writeWorkgroup sharedSum (i32 tid) (a + b)
          )
          (return ())
        barrier

  reductionLoop 128
  reductionLoop 64
  reductionLoop 32
  reductionLoop 16
  reductionLoop 8
  reductionLoop 4
  reductionLoop 2
  reductionLoop 1

  if_ (i32 tid DSL.== litI32 0)
    (do
      totalSum <- readWorkgroup sharedSum (litI32 0)
      let meanSq = totalSum / litF32 (fromIntegral hiddenSize)
      let rms = sqrt' (meanSq + litF32 1e-6)
      writeWorkgroup sharedSum (litI32 0) rms
    )
    (return ())

  barrier

  -- Phase 2: Normalize and apply weights
  rms <- readWorkgroup sharedSum (litI32 0)

  if useVec4 && (hiddenSize `mod` 4 == 0)
    then do
      let vec4Iters = hiddenSize `div` 4
      loop (litI32 0) (litI32 vec4Iters) (litI32 1) $ \i -> do
        let cond = (i32 tid + i * litI32 256) DSL.< litI32 vec4Iters
        if_ cond
          (do
            let idx = (i32 tid + i * litI32 256) * litI32 4

            in0 <- readBuffer input idx
            in1 <- readBuffer input (idx + litI32 1)
            in2 <- readBuffer input (idx + litI32 2)
            in3 <- readBuffer input (idx + litI32 3)

            w0 <- readBuffer weight idx
            w1 <- readBuffer weight (idx + litI32 1)
            w2 <- readBuffer weight (idx + litI32 2)
            w3 <- readBuffer weight (idx + litI32 3)

            let norm0 = if zeroCentered
                  then (in0 / rms) * (litF32 1.0 + w0)
                  else (in0 / rms) * w0
            let norm1 = if zeroCentered
                  then (in1 / rms) * (litF32 1.0 + w1)
                  else (in1 / rms) * w1
            let norm2 = if zeroCentered
                  then (in2 / rms) * (litF32 1.0 + w2)
                  else (in2 / rms) * w2
            let norm3 = if zeroCentered
                  then (in3 / rms) * (litF32 1.0 + w3)
                  else (in3 / rms) * w3

            writeBuffer output idx norm0
            writeBuffer output (idx + litI32 1) norm1
            writeBuffer output (idx + litI32 2) norm2
            writeBuffer output (idx + litI32 3) norm3
          )
          (return ())
    else do
      loop (litI32 0) (litI32 ((hiddenSize + 255) `div` 256)) (litI32 1) $ \i -> do
        let idx = i32 tid + i * litI32 256
        let cond = idx DSL.< litI32 hiddenSize
        if_ cond
          (do
            inVal <- readBuffer input idx
            wVal <- readBuffer weight idx

            let norm = if zeroCentered
                  then (inVal / rms) * (litF32 1.0 + wVal)
                  else (inVal / rms) * wVal

            writeBuffer output idx norm
          )
          (return ())

-- | Run RMSNorm with DSL (FP32, no Vec4)
runRMSNormDSL :: Vector Float -> Vector Float -> Int -> Bool
              -> ContT r IO (Vector Float)
runRMSNormDSL input weight hiddenSize zeroCentered =
  runRMSNormDSLWithPrecision input weight hiddenSize zeroCentered False False

-- | Run RMSNorm with DSL and configurable precision/optimizations
runRMSNormDSLWithPrecision :: Vector Float -> Vector Float -> Int -> Bool
                           -> Bool -> Bool  -- useFP16, useVec4
                           -> ContT r IO (Vector Float)
runRMSNormDSLWithPrecision input weight hiddenSize zeroCentered useFP16 useVec4 = do
  -- Validate inputs
  if V.length input /= hiddenSize
    then error $ "RMSNormDSL: input size mismatch: " ++ show (V.length input) ++ " vs " ++ show hiddenSize
    else pure ()
  if V.length weight /= hiddenSize
    then error $ "RMSNormDSL: weight size mismatch: " ++ show (V.length weight) ++ " vs " ++ show hiddenSize
    else pure ()

  -- Create GPU context
  let features = if useFP16 then [FeatureShaderF16] else []
  ctx <- createContextWithFeatures [] features

  let shape = Shape [hiddenSize]

  if useFP16
    then do
      -- FP16 path
      let inputHalf = vectorFloatToHalf input
          weightHalf = vectorFloatToHalf weight

      inputTensor <- createTensorWithData ctx shape inputHalf
      weightTensor <- createTensorWithData ctx shape weightHalf
      outputTensor <- createTensor ctx shape F16

      let shader = (buildShaderWithAutoBinding (256, 1, 1) $
                     rmsNormKernelFP16 hiddenSize zeroCentered useVec4)
                   { moduleExtensions = ["f16"] }

      liftIO $ executeShaderNamed ctx shader
        [ ("input", AnyTensor inputTensor)
        , ("weight", AnyTensor weightTensor)
        , ("output", AnyTensor outputTensor)
        ]
        (WorkgroupSize 1 1 1)

      outputHalf <- liftIO $ fromGPU ctx outputTensor hiddenSize
      let outputFloat = vectorHalfToFloat outputHalf
      pure outputFloat

    else do
      -- FP32 path
      inputTensor <- createTensorWithData ctx shape input
      weightTensor <- createTensorWithData ctx shape weight
      outputTensor <- createTensor ctx shape F32

      let shader = buildShaderWithAutoBinding (256, 1, 1) $
                   rmsNormKernelFP32 hiddenSize zeroCentered useVec4

      liftIO $ executeShaderNamed ctx shader
        [ ("input", AnyTensor inputTensor)
        , ("weight", AnyTensor weightTensor)
        , ("output", AnyTensor outputTensor)
        ]
        (WorkgroupSize 1 1 1)

      outputFloat <- liftIO $ fromGPU ctx outputTensor hiddenSize
      pure outputFloat
