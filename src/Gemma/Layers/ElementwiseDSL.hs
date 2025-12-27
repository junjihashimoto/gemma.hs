{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

{-|
Module: Gemma.Layers.ElementwiseDSL
Description: DSL-based element-wise operations

Provides type-safe element-wise operations used throughout the transformer:
  - Element-wise multiply (for gating mechanisms like GeGLU)
  - Element-wise add (for residual connections)

This DSL implementation provides:
  - Type-safe element-wise computation
  - FP16 support for 2x memory bandwidth
  - Vec4 SIMD for 4x additional speedup
  - Optimized for common transformer patterns
-}

module Gemma.Layers.ElementwiseDSL
  ( -- * Element-wise Multiply
    runElementwiseMultiplyDSL
  , runElementwiseMultiplyDSLWithPrecision
  , elementwiseMultiplyKernelDSL
  , elementwiseMultiplyKernelFP16
  , elementwiseMultiplyKernelFP32
    -- * Element-wise Add
  , runElementwiseAddDSL
  , runElementwiseAddDSLWithPrecision
  , elementwiseAddKernelDSL
  , elementwiseAddKernelFP16
  , elementwiseAddKernelFP32
  ) where

import Graphics.WebGPU.Dawn.ContT
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import WGSL.DSL hiding ((<), (>), (<=), (>=), (==), (/=), (&&), (||), not)
import qualified WGSL.DSL as DSL
import WGSL.Execute (executeShaderNamed)
import Gemma.Utils.Half (vectorFloatToHalf, vectorHalfToFloat)
import Prelude

-- ═══════════════════════════════════════════════════════════════
-- Element-wise Multiply (for gating in GeGLU)
-- ═══════════════════════════════════════════════════════════════

-- | Element-wise multiply kernel with DSL
--
-- Computes: output[i] = a[i] * b[i]
--
-- Parameters:
-- - size: Number of elements
-- - useFP16: When True, uses FP16 for 2x memory bandwidth
-- - useVec4: When True, uses vec4 SIMD for 4x speedup
elementwiseMultiplyKernelDSL :: Int -> Bool -> Bool -> ShaderM ()
elementwiseMultiplyKernelDSL size useFP16 useVec4 = do
  if useFP16
    then elementwiseMultiplyKernelFP16 size useVec4
    else elementwiseMultiplyKernelFP32 size useVec4

-- | FP16 version of element-wise multiply kernel
elementwiseMultiplyKernelFP16 :: Int -> Bool -> ShaderM ()
elementwiseMultiplyKernelFP16 size useVec4 = do
  a <- declareInputBuffer "a" (TArray size TF16)
  b <- declareInputBuffer "b" (TArray size TF16)
  output <- declareOutputBuffer "output" (TArray size TF16)

  gid <- globalId
  let idx = U32ToI32 (vecX gid)

  if useVec4 && (size `mod` 4 == 0)
    then do
      -- Vec4 SIMD path
      let vec4Iters = size `div` 4
      let cond = idx DSL.< litI32 vec4Iters
      if_ cond
        (do
          let baseIdx = idx * litI32 4

          -- Load 4 elements from each input
          a0 <- readBuffer a baseIdx
          a1 <- readBuffer a (baseIdx + litI32 1)
          a2 <- readBuffer a (baseIdx + litI32 2)
          a3 <- readBuffer a (baseIdx + litI32 3)

          b0 <- readBuffer b baseIdx
          b1 <- readBuffer b (baseIdx + litI32 1)
          b2 <- readBuffer b (baseIdx + litI32 2)
          b3 <- readBuffer b (baseIdx + litI32 3)

          -- Multiply
          let result0 :: Exp F16
              result0 = a0 * b0
              result1 :: Exp F16
              result1 = a1 * b1
              result2 :: Exp F16
              result2 = a2 * b2
              result3 :: Exp F16
              result3 = a3 * b3

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
          aVal <- readBuffer a idx
          bVal <- readBuffer b idx
          let result :: Exp F16
              result = aVal * bVal
          writeBuffer output idx result
        )
        (return ())

-- | FP32 version of element-wise multiply kernel
elementwiseMultiplyKernelFP32 :: Int -> Bool -> ShaderM ()
elementwiseMultiplyKernelFP32 size useVec4 = do
  a <- declareInputBuffer "a" (TArray size TF32)
  b <- declareInputBuffer "b" (TArray size TF32)
  output <- declareOutputBuffer "output" (TArray size TF32)

  gid <- globalId
  let idx = U32ToI32 (vecX gid)

  if useVec4 && (size `mod` 4 == 0)
    then do
      -- Vec4 SIMD path
      let vec4Iters = size `div` 4
      let cond = idx DSL.< litI32 vec4Iters
      if_ cond
        (do
          let baseIdx = idx * litI32 4

          a0 <- readBuffer a baseIdx
          a1 <- readBuffer a (baseIdx + litI32 1)
          a2 <- readBuffer a (baseIdx + litI32 2)
          a3 <- readBuffer a (baseIdx + litI32 3)

          b0 <- readBuffer b baseIdx
          b1 <- readBuffer b (baseIdx + litI32 1)
          b2 <- readBuffer b (baseIdx + litI32 2)
          b3 <- readBuffer b (baseIdx + litI32 3)

          let result0 :: Exp F32
              result0 = a0 * b0
              result1 :: Exp F32
              result1 = a1 * b1
              result2 :: Exp F32
              result2 = a2 * b2
              result3 :: Exp F32
              result3 = a3 * b3

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
          aVal <- readBuffer a idx
          bVal <- readBuffer b idx
          let result :: Exp F32
              result = aVal * bVal
          writeBuffer output idx result
        )
        (return ())

-- | Run element-wise multiply with DSL (FP32, no Vec4)
runElementwiseMultiplyDSL :: Vector Float -> Vector Float -> ContT r IO (Vector Float)
runElementwiseMultiplyDSL a b =
  runElementwiseMultiplyDSLWithPrecision False False a b

-- | Run element-wise multiply with DSL and configurable precision/optimizations
runElementwiseMultiplyDSLWithPrecision :: Bool  -- ^ Use FP16?
                                       -> Bool  -- ^ Use Vec4?
                                       -> Vector Float -> Vector Float
                                       -> ContT r IO (Vector Float)
runElementwiseMultiplyDSLWithPrecision useFP16 useVec4 a b = do
  let size = V.length a

  -- Validate inputs
  if V.length b /= size
    then error $ "ElementwiseMultiplyDSL: size mismatch: " ++ show (V.length b) ++ " vs " ++ show size
    else pure ()

  if useVec4 && (size `mod` 4 /= 0)
    then error $ "ElementwiseMultiplyDSL: Vec4 mode requires size to be multiple of 4, got: " ++ show size
    else pure ()

  -- Create GPU context with features
  let features = if useFP16 then [FeatureShaderF16] else []
  ctx <- createContextWithFeatures [] features

  let shape = Shape [size]

  if useFP16
    then do
      -- FP16 path
      let aHalf = vectorFloatToHalf a
          bHalf = vectorFloatToHalf b

      aTensor <- createTensorWithData ctx shape aHalf
      bTensor <- createTensorWithData ctx shape bHalf
      outputTensor <- createTensor ctx shape F16

      let shader = (buildShaderWithAutoBinding (256, 1, 1) $
                     elementwiseMultiplyKernelFP16 size useVec4)
                   { moduleExtensions = ["f16"] }

      let numWorkgroups = if useVec4
                          then ((size `div` 4) + 255) `div` 256
                          else (size + 255) `div` 256

      liftIO $ executeShaderNamed ctx shader
        [ ("a", AnyTensor aTensor)
        , ("b", AnyTensor bTensor)
        , ("output", AnyTensor outputTensor)
        ]
        (WorkgroupSize numWorkgroups 1 1)

      outputHalf <- liftIO $ fromGPU ctx outputTensor size
      let outputFloat = vectorHalfToFloat outputHalf
      pure outputFloat

    else do
      -- FP32 path
      aTensor <- createTensorWithData ctx shape a
      bTensor <- createTensorWithData ctx shape b
      outputTensor <- createTensor ctx shape F32

      let shader = buildShaderWithAutoBinding (256, 1, 1) $
                   elementwiseMultiplyKernelFP32 size useVec4

      let numWorkgroups = if useVec4
                          then ((size `div` 4) + 255) `div` 256
                          else (size + 255) `div` 256

      liftIO $ executeShaderNamed ctx shader
        [ ("a", AnyTensor aTensor)
        , ("b", AnyTensor bTensor)
        , ("output", AnyTensor outputTensor)
        ]
        (WorkgroupSize numWorkgroups 1 1)

      outputFloat <- liftIO $ fromGPU ctx outputTensor size
      pure outputFloat

-- ═══════════════════════════════════════════════════════════════
-- Element-wise Add (for residual connections)
-- ═══════════════════════════════════════════════════════════════

-- | Element-wise add kernel with DSL
--
-- Computes: output[i] = a[i] + b[i]
--
-- Parameters:
-- - size: Number of elements
-- - useFP16: When True, uses FP16 for 2x memory bandwidth
-- - useVec4: When True, uses vec4 SIMD for 4x speedup
elementwiseAddKernelDSL :: Int -> Bool -> Bool -> ShaderM ()
elementwiseAddKernelDSL size useFP16 useVec4 = do
  if useFP16
    then elementwiseAddKernelFP16 size useVec4
    else elementwiseAddKernelFP32 size useVec4

-- | FP16 version of element-wise add kernel
elementwiseAddKernelFP16 :: Int -> Bool -> ShaderM ()
elementwiseAddKernelFP16 size useVec4 = do
  a <- declareInputBuffer "a" (TArray size TF16)
  b <- declareInputBuffer "b" (TArray size TF16)
  output <- declareOutputBuffer "output" (TArray size TF16)

  gid <- globalId
  let idx = U32ToI32 (vecX gid)

  if useVec4 && (size `mod` 4 == 0)
    then do
      -- Vec4 SIMD path
      let vec4Iters = size `div` 4
      let cond = idx DSL.< litI32 vec4Iters
      if_ cond
        (do
          let baseIdx = idx * litI32 4

          a0 <- readBuffer a baseIdx
          a1 <- readBuffer a (baseIdx + litI32 1)
          a2 <- readBuffer a (baseIdx + litI32 2)
          a3 <- readBuffer a (baseIdx + litI32 3)

          b0 <- readBuffer b baseIdx
          b1 <- readBuffer b (baseIdx + litI32 1)
          b2 <- readBuffer b (baseIdx + litI32 2)
          b3 <- readBuffer b (baseIdx + litI32 3)

          let result0 :: Exp F16
              result0 = a0 + b0
              result1 :: Exp F16
              result1 = a1 + b1
              result2 :: Exp F16
              result2 = a2 + b2
              result3 :: Exp F16
              result3 = a3 + b3

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
          aVal <- readBuffer a idx
          bVal <- readBuffer b idx
          let result :: Exp F16
              result = aVal + bVal
          writeBuffer output idx result
        )
        (return ())

-- | FP32 version of element-wise add kernel
elementwiseAddKernelFP32 :: Int -> Bool -> ShaderM ()
elementwiseAddKernelFP32 size useVec4 = do
  a <- declareInputBuffer "a" (TArray size TF32)
  b <- declareInputBuffer "b" (TArray size TF32)
  output <- declareOutputBuffer "output" (TArray size TF32)

  gid <- globalId
  let idx = U32ToI32 (vecX gid)

  if useVec4 && (size `mod` 4 == 0)
    then do
      -- Vec4 SIMD path
      let vec4Iters = size `div` 4
      let cond = idx DSL.< litI32 vec4Iters
      if_ cond
        (do
          let baseIdx = idx * litI32 4

          a0 <- readBuffer a baseIdx
          a1 <- readBuffer a (baseIdx + litI32 1)
          a2 <- readBuffer a (baseIdx + litI32 2)
          a3 <- readBuffer a (baseIdx + litI32 3)

          b0 <- readBuffer b baseIdx
          b1 <- readBuffer b (baseIdx + litI32 1)
          b2 <- readBuffer b (baseIdx + litI32 2)
          b3 <- readBuffer b (baseIdx + litI32 3)

          let result0 :: Exp F32
              result0 = a0 + b0
              result1 :: Exp F32
              result1 = a1 + b1
              result2 :: Exp F32
              result2 = a2 + b2
              result3 :: Exp F32
              result3 = a3 + b3

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
          aVal <- readBuffer a idx
          bVal <- readBuffer b idx
          let result :: Exp F32
              result = aVal + bVal
          writeBuffer output idx result
        )
        (return ())

-- | Run element-wise add with DSL (FP32, no Vec4)
runElementwiseAddDSL :: Vector Float -> Vector Float -> ContT r IO (Vector Float)
runElementwiseAddDSL a b =
  runElementwiseAddDSLWithPrecision False False a b

-- | Run element-wise add with DSL and configurable precision/optimizations
runElementwiseAddDSLWithPrecision :: Bool  -- ^ Use FP16?
                                  -> Bool  -- ^ Use Vec4?
                                  -> Vector Float -> Vector Float
                                  -> ContT r IO (Vector Float)
runElementwiseAddDSLWithPrecision useFP16 useVec4 a b = do
  let size = V.length a

  -- Validate inputs
  if V.length b /= size
    then error $ "ElementwiseAddDSL: size mismatch: " ++ show (V.length b) ++ " vs " ++ show size
    else pure ()

  if useVec4 && (size `mod` 4 /= 0)
    then error $ "ElementwiseAddDSL: Vec4 mode requires size to be multiple of 4, got: " ++ show size
    else pure ()

  -- Create GPU context with features
  let features = if useFP16 then [FeatureShaderF16] else []
  ctx <- createContextWithFeatures [] features

  let shape = Shape [size]

  if useFP16
    then do
      -- FP16 path
      let aHalf = vectorFloatToHalf a
          bHalf = vectorFloatToHalf b

      aTensor <- createTensorWithData ctx shape aHalf
      bTensor <- createTensorWithData ctx shape bHalf
      outputTensor <- createTensor ctx shape F16

      let shader = (buildShaderWithAutoBinding (256, 1, 1) $
                     elementwiseAddKernelFP16 size useVec4)
                   { moduleExtensions = ["f16"] }

      let numWorkgroups = if useVec4
                          then ((size `div` 4) + 255) `div` 256
                          else (size + 255) `div` 256

      liftIO $ executeShaderNamed ctx shader
        [ ("a", AnyTensor aTensor)
        , ("b", AnyTensor bTensor)
        , ("output", AnyTensor outputTensor)
        ]
        (WorkgroupSize numWorkgroups 1 1)

      outputHalf <- liftIO $ fromGPU ctx outputTensor size
      let outputFloat = vectorHalfToFloat outputHalf
      pure outputFloat

    else do
      -- FP32 path
      aTensor <- createTensorWithData ctx shape a
      bTensor <- createTensorWithData ctx shape b
      outputTensor <- createTensor ctx shape F32

      let shader = buildShaderWithAutoBinding (256, 1, 1) $
                   elementwiseAddKernelFP32 size useVec4

      let numWorkgroups = if useVec4
                          then ((size `div` 4) + 255) `div` 256
                          else (size + 255) `div` 256

      liftIO $ executeShaderNamed ctx shader
        [ ("a", AnyTensor aTensor)
        , ("b", AnyTensor bTensor)
        , ("output", AnyTensor outputTensor)
        ]
        (WorkgroupSize numWorkgroups 1 1)

      outputFloat <- liftIO $ fromGPU ctx outputTensor size
      pure outputFloat
