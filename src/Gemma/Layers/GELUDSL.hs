{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

{-|
Module: Gemma.Layers.GELUDSL
Description: DSL-based GELU (Gaussian Error Linear Unit) activation

GELU is a smooth, non-monotonic activation function defined as:
  GELU(x) = x * Φ(x)
where Φ is the CDF of the standard normal distribution.

Approximation used (as in the original BERT/GPT papers):
  GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

This DSL implementation provides:
  - Type-safe GELU computation
  - FP16 support for 2x memory bandwidth
  - Vec4 SIMD for 4x additional speedup
  - Clamping to prevent tanh overflow

Reference: https://arxiv.org/abs/1606.08415
-}

module Gemma.Layers.GELUDSL
  ( -- * Main GELU Functions
    runGELUDSL
  , runGELUDSLWithPrecision
    -- * Component Kernels
  , geluKernelDSL
  , geluKernelFP16
  , geluKernelFP32
  ) where

import Graphics.WebGPU.Dawn.ContT
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import WGSL.DSL hiding ((<), (>), (<=), (>=), (==), (/=), (&&), (||), not)
import qualified WGSL.DSL as DSL
import WGSL.Execute (executeShaderNamed)
import Gemma.Utils.Half (vectorFloatToHalf, vectorHalfToFloat)
import Prelude

-- | GELU kernel with DSL
--
-- Computes: output = GELU(input)
-- where GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
--
-- Parameters:
-- - size: Number of elements in the input vector
-- - useFP16: When True, uses FP16 for 2x memory bandwidth
-- - useVec4: When True, uses vec4 SIMD for 4x speedup
geluKernelDSL :: Int -> Bool -> Bool -> ShaderM ()
geluKernelDSL size useFP16 useVec4 = do
  if useFP16
    then geluKernelFP16 size useVec4
    else geluKernelFP32 size useVec4

-- | FP16 version of GELU kernel
geluKernelFP16 :: Int -> Bool -> ShaderM ()
geluKernelFP16 size useVec4 = do
  -- Declare buffers with FP16 storage
  input <- declareInputBuffer "input" (TArray size TF16)
  output <- declareOutputBuffer "output" (TArray size TF16)

  gid <- globalId
  let idx = U32ToI32 (vecX gid)

  -- Constants
  let sqrt2OverPi = litF16 0.7978845608  -- √(2/π)
      coeff = litF16 0.044715
      half = litF16 0.5
      one = litF16 1.0
      clampMin = litF16 (-10.0)
      clampMax = litF16 10.0

  if useVec4 && (size `mod` 4 == 0)
    then do
      -- Vec4 SIMD path: process 4 elements at a time
      let vec4Iters = size `div` 4
      let cond = idx DSL.< litI32 vec4Iters
      if_ cond
        (do
          let baseIdx = idx * litI32 4

          -- Load 4 elements
          x0 <- readBuffer input baseIdx
          x1 <- readBuffer input (baseIdx + litI32 1)
          x2 <- readBuffer input (baseIdx + litI32 2)
          x3 <- readBuffer input (baseIdx + litI32 3)

          -- Compute GELU for each element
          -- x_cubed = x * x * x
          let xCubed0 = x0 * x0 * x0
              xCubed1 = x1 * x1 * x1
              xCubed2 = x2 * x2 * x2
              xCubed3 = x3 * x3 * x3

          -- inner = sqrt(2/pi) * (x + 0.044715 * x^3)
          let inner0 = sqrt2OverPi * (x0 + coeff * xCubed0)
              inner1 = sqrt2OverPi * (x1 + coeff * xCubed1)
              inner2 = sqrt2OverPi * (x2 + coeff * xCubed2)
              inner3 = sqrt2OverPi * (x3 + coeff * xCubed3)

          -- Clamp to prevent tanh overflow
          let clamped0 = clamp' inner0 clampMin clampMax
              clamped1 = clamp' inner1 clampMin clampMax
              clamped2 = clamp' inner2 clampMin clampMax
              clamped3 = clamp' inner3 clampMin clampMax

          -- Compute GELU: 0.5 * x * (1 + tanh(clamped))
          let result0 = half * x0 * (one + tanh' clamped0)
              result1 = half * x1 * (one + tanh' clamped1)
              result2 = half * x2 * (one + tanh' clamped2)
              result3 = half * x3 * (one + tanh' clamped3)

          -- Write output
          writeBuffer output baseIdx result0
          writeBuffer output (baseIdx + litI32 1) result1
          writeBuffer output (baseIdx + litI32 2) result2
          writeBuffer output (baseIdx + litI32 3) result3
        )
        (return ())
    else do
      -- Scalar path
      let cond = idx DSL.< litI32 size
      if_ cond
        (do
          x <- readBuffer input idx

          -- Compute GELU
          let xCubed = x * x * x
              inner = sqrt2OverPi * (x + coeff * xCubed)
              clamped = clamp' inner clampMin clampMax
              result = half * x * (one + tanh' clamped)

          writeBuffer output idx result
        )
        (return ())

-- | FP32 version of GELU kernel
geluKernelFP32 :: Int -> Bool -> ShaderM ()
geluKernelFP32 size useVec4 = do
  -- Declare buffers with FP32 storage
  input <- declareInputBuffer "input" (TArray size TF32)
  output <- declareOutputBuffer "output" (TArray size TF32)

  gid <- globalId
  let idx = U32ToI32 (vecX gid)

  -- Constants
  let sqrt2OverPi = litF32 0.7978845608  -- √(2/π)
      coeff = litF32 0.044715
      half = litF32 0.5
      one = litF32 1.0
      clampMin = litF32 (-10.0)
      clampMax = litF32 10.0

  if useVec4 && (size `mod` 4 == 0)
    then do
      -- Vec4 SIMD path: process 4 elements at a time
      let vec4Iters = size `div` 4
      let cond = idx DSL.< litI32 vec4Iters
      if_ cond
        (do
          let baseIdx = idx * litI32 4

          -- Load 4 elements
          x0 <- readBuffer input baseIdx
          x1 <- readBuffer input (baseIdx + litI32 1)
          x2 <- readBuffer input (baseIdx + litI32 2)
          x3 <- readBuffer input (baseIdx + litI32 3)

          -- Compute GELU for each element
          let xCubed0 = x0 * x0 * x0
              xCubed1 = x1 * x1 * x1
              xCubed2 = x2 * x2 * x2
              xCubed3 = x3 * x3 * x3

          let inner0 = sqrt2OverPi * (x0 + coeff * xCubed0)
              inner1 = sqrt2OverPi * (x1 + coeff * xCubed1)
              inner2 = sqrt2OverPi * (x2 + coeff * xCubed2)
              inner3 = sqrt2OverPi * (x3 + coeff * xCubed3)

          let clamped0 = clamp' inner0 clampMin clampMax
              clamped1 = clamp' inner1 clampMin clampMax
              clamped2 = clamp' inner2 clampMin clampMax
              clamped3 = clamp' inner3 clampMin clampMax

          let result0 = half * x0 * (one + tanh' clamped0)
              result1 = half * x1 * (one + tanh' clamped1)
              result2 = half * x2 * (one + tanh' clamped2)
              result3 = half * x3 * (one + tanh' clamped3)

          writeBuffer output baseIdx result0
          writeBuffer output (baseIdx + litI32 1) result1
          writeBuffer output (baseIdx + litI32 2) result2
          writeBuffer output (baseIdx + litI32 3) result3
        )
        (return ())
    else do
      -- Scalar path
      let cond = idx DSL.< litI32 size
      if_ cond
        (do
          x <- readBuffer input idx

          let xCubed = x * x * x
              inner = sqrt2OverPi * (x + coeff * xCubed)
              clamped = clamp' inner clampMin clampMax
              result = half * x * (one + tanh' clamped)

          writeBuffer output idx result
        )
        (return ())

-- | Run GELU with DSL (FP32, no Vec4)
runGELUDSL :: Vector Float -> ContT r IO (Vector Float)
runGELUDSL input =
  runGELUDSLWithPrecision False False input

-- | Run GELU with DSL and configurable precision/optimizations
runGELUDSLWithPrecision :: Bool  -- ^ Use FP16?
                        -> Bool  -- ^ Use Vec4?
                        -> Vector Float  -- ^ Input vector
                        -> ContT r IO (Vector Float)
runGELUDSLWithPrecision useFP16 useVec4 input = do
  let size = V.length input

  -- Validate inputs
  if useVec4 && (size `mod` 4 /= 0)
    then error $ "GELUDSL: Vec4 mode requires size to be multiple of 4, got: " ++ show size
    else pure ()

  -- Create GPU context with features
  let features = if useFP16 then [FeatureShaderF16] else []
  ctx <- createContextWithFeatures [] features

  let shape = Shape [size]

  if useFP16
    then do
      -- FP16 path
      let inputHalf = vectorFloatToHalf input

      inputTensor <- createTensorWithData ctx shape inputHalf
      outputTensor <- createTensor ctx shape F16

      let shader = (buildShaderWithAutoBinding (256, 1, 1) $
                     geluKernelFP16 size useVec4)
                   { moduleExtensions = ["f16"] }

      let numWorkgroups = if useVec4
                          then ((size `div` 4) + 255) `div` 256
                          else (size + 255) `div` 256

      liftIO $ executeShaderNamed ctx shader
        [ ("input", AnyTensor inputTensor)
        , ("output", AnyTensor outputTensor)
        ]
        (WorkgroupSize numWorkgroups 1 1)

      outputHalf <- liftIO $ fromGPU ctx outputTensor size
      let outputFloat = vectorHalfToFloat outputHalf
      pure outputFloat

    else do
      -- FP32 path
      inputTensor <- createTensorWithData ctx shape input
      outputTensor <- createTensor ctx shape F32

      let shader = buildShaderWithAutoBinding (256, 1, 1) $
                   geluKernelFP32 size useVec4

      let numWorkgroups = if useVec4
                          then ((size `div` 4) + 255) `div` 256
                          else (size + 255) `div` 256

      liftIO $ executeShaderNamed ctx shader
        [ ("input", AnyTensor inputTensor)
        , ("output", AnyTensor outputTensor)
        ]
        (WorkgroupSize numWorkgroups 1 1)

      outputFloat <- liftIO $ fromGPU ctx outputTensor size
      pure outputFloat
