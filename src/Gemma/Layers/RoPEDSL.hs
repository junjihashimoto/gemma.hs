{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

{-|
Module: Gemma.Layers.RoPEDSL
Description: DSL-based Rotary Position Embedding

RoPE encodes position information by applying rotation matrices to pairs of dimensions.
For each pair (i, i+1), we rotate by an angle θ = position * freq:

  [x_i']     [cos(θ)  -sin(θ)] [x_i  ]
  [x_i+1'] = [sin(θ)   cos(θ)] [x_i+1]

where freq = 1.0 / base^(i/dim)

This DSL implementation provides:
  - Type-safe rotation computation
  - FP16 support for 2x memory bandwidth
  - Multi-head support
  - Flexible position handling

Reference: https://arxiv.org/abs/2104.09864
-}

module Gemma.Layers.RoPEDSL
  ( -- * Main RoPE Functions
    runRoPEDSL
  , runRoPEDSLWithPrecision
    -- * Component Kernels
  , ropeKernelDSL
  , ropeKernelFP16
  , ropeKernelFP32
  ) where

import Graphics.WebGPU.Dawn.ContT
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import WGSL.DSL hiding ((<), (>), (<=), (>=), (==), (/=), (&&), (||), not)
import qualified WGSL.DSL as DSL
import WGSL.Execute (executeShaderNamed)
import Gemma.Utils.Half (vectorFloatToHalf, vectorHalfToFloat)
import Prelude

-- | RoPE kernel with DSL
--
-- Computes: output = RoPE(input, position)
-- For each pair (2i, 2i+1):
--   freq = 1.0 / base^(i/headDim)
--   theta = position * freq
--   output[2i] = input[2i] * cos(theta) - input[2i+1] * sin(theta)
--   output[2i+1] = input[2i] * sin(theta) + input[2i+1] * cos(theta)
--
-- One thread per head, looping through dimension pairs.
--
-- Parameters:
-- - numHeads: Number of attention heads
-- - headDim: Dimension of each head (must be even)
-- - position: Sequence position (integer)
-- - ropeBase: RoPE base frequency (10000.0 standard, 1000000.0 for long context)
-- - useFP16: When True, uses FP16 for 2x memory bandwidth
ropeKernelDSL :: Int -> Int -> Int -> Float -> Bool -> ShaderM ()
ropeKernelDSL numHeads headDim position ropeBase useFP16 = do
  if useFP16
    then ropeKernelFP16 numHeads headDim position ropeBase
    else ropeKernelFP32 numHeads headDim position ropeBase

-- | FP16 RoPE kernel
ropeKernelFP16 :: Int -> Int -> Int -> Float -> ShaderM ()
ropeKernelFP16 numHeads headDim position ropeBase = do
  -- Input/output buffers
  input <- declareInputBuffer "input" (TArray (numHeads * headDim) TF16)
  output <- declareOutputBuffer "output" (TArray (numHeads * headDim) TF16)

  -- Get thread ID (one per head)
  gid <- globalId
  let headIdx = U32ToI32 (vecX gid)

  -- Bounds check
  if_ (headIdx DSL.>= litI32 (fromIntegral numHeads))
    (return ())
    (return ())

  -- Position and head offset
  let pos = litF16 (realToFrac position)
      headOffset = headIdx * litI32 (fromIntegral headDim)
      base = litF16 (realToFrac ropeBase)

  -- Gemma 3 uses "split-half" RoPE strategy:
  -- For each position in the first half, compute rotation with corresponding second half position
  -- output[i] = input[i] * cos(theta_i) - input[i + half] * sin(theta_i)
  -- output[i + half] = input[i] * sin(theta_i) + input[i + half] * cos(theta_i)

  let halfDim = litI32 (fromIntegral (headDim `div` 2))

  -- Loop through first half dimensions (step by 1)
  loop (litI32 0) halfDim (litI32 1) $ \i -> do
    let idx = headOffset + i
        idxPlusHalf = idx + halfDim

    -- Read values from first and second half
    x <- readBuffer input idx
    y <- readBuffer input idxPlusHalf

    -- Compute frequency: freq = 1.0 / base^((2*i)/headDim)
    -- Note: i goes 0,1,2,... so we need 2*i to get the pair indices 0,2,4,...
    let freqExp = (I32ToF16 i * litF16 2.0) / litF16 (fromIntegral headDim)
        freq = litF16 1.0 / pow' base freqExp

    -- Compute rotation angle
    let theta = pos * freq

    -- Compute sin and cos
    let cosTheta = cos' theta
        sinTheta = sin' theta

    -- Apply split-half rotation (matching PyTorch Gemma 3)
    let x' = x * cosTheta - y * sinTheta
        y' = x * sinTheta + y * cosTheta

    -- Write output
    writeBuffer output idx x'
    writeBuffer output idxPlusHalf y'

-- | FP32 RoPE kernel
ropeKernelFP32 :: Int -> Int -> Int -> Float -> ShaderM ()
ropeKernelFP32 numHeads headDim position ropeBase = do
  -- Input/output buffers
  input <- declareInputBuffer "input" (TArray (numHeads * headDim) TF32)
  output <- declareOutputBuffer "output" (TArray (numHeads * headDim) TF32)

  -- Get thread ID (one per head)
  gid <- globalId
  let headIdx = U32ToI32 (vecX gid)

  -- Bounds check
  if_ (headIdx DSL.>= litI32 (fromIntegral numHeads))
    (return ())
    (return ())

  -- Position and head offset
  let pos = litF32 (realToFrac position)
      headOffset = headIdx * litI32 (fromIntegral headDim)
      base = litF32 ropeBase

  -- Gemma 3 uses "split-half" RoPE strategy:
  -- For each position in the first half, compute rotation with corresponding second half position
  -- output[i] = input[i] * cos(theta_i) - input[i + half] * sin(theta_i)
  -- output[i + half] = input[i] * sin(theta_i) + input[i + half] * cos(theta_i)

  let halfDim = litI32 (fromIntegral (headDim `div` 2))

  -- Loop through first half dimensions (step by 1)
  loop (litI32 0) halfDim (litI32 1) $ \i -> do
    let idx = headOffset + i
        idxPlusHalf = idx + halfDim

    -- Read values from first and second half
    x <- readBuffer input idx
    y <- readBuffer input idxPlusHalf

    -- Compute frequency: freq = 1.0 / base^((2*i)/headDim)
    -- Note: i goes 0,1,2,... so we need 2*i to get the pair indices 0,2,4,...
    let freqExp = (I32ToF32 i * litF32 2.0) / litF32 (fromIntegral headDim)
        freq = litF32 1.0 / pow' base freqExp

    -- Compute rotation angle
    let theta = pos * freq

    -- Compute sin and cos
    let cosTheta = cos' theta
        sinTheta = sin' theta

    -- Apply split-half rotation (matching PyTorch Gemma 3)
    let x' = x * cosTheta - y * sinTheta
        y' = x * sinTheta + y * cosTheta

    -- Write output
    writeBuffer output idx x'
    writeBuffer output idxPlusHalf y'

-- | Run RoPE with DSL (FP32 by default)
runRoPEDSL :: Vector Float  -- ^ Input [numHeads * headDim]
           -> Int           -- ^ Number of heads
           -> Int           -- ^ Head dimension
           -> Int           -- ^ Position
           -> Float         -- ^ RoPE base
           -> ContT r IO (Vector Float)
runRoPEDSL = runRoPEDSLWithPrecision False

-- | Run RoPE with DSL and specified precision
runRoPEDSLWithPrecision
  :: Bool          -- ^ Use FP16?
  -> Vector Float  -- ^ Input [numHeads * headDim]
  -> Int           -- ^ Number of heads
  -> Int           -- ^ Head dimension
  -> Int           -- ^ Position
  -> Float         -- ^ RoPE base
  -> ContT r IO (Vector Float)
runRoPEDSLWithPrecision useFP16 input numHeads headDim position ropeBase = do
  let size = numHeads * headDim

  -- Validate inputs
  if V.length input /= size
    then error $ "RoPEDSL: input size mismatch: " ++ show (V.length input) ++ " vs " ++ show size
    else pure ()

  if headDim `mod` 2 /= 0
    then error $ "RoPEDSL: head_dim must be even, got: " ++ show headDim
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
                     ropeKernelFP16 numHeads headDim position ropeBase)
                   { moduleExtensions = ["f16"] }

      let numWorkgroups = (numHeads + 255) `div` 256

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
                   ropeKernelFP32 numHeads headDim position ropeBase

      let numWorkgroups = (numHeads + 255) `div` 256

      liftIO $ executeShaderNamed ctx shader
        [ ("input", AnyTensor inputTensor)
        , ("output", AnyTensor outputTensor)
        ]
        (WorkgroupSize numWorkgroups 1 1)

      outputFloat <- liftIO $ fromGPU ctx outputTensor size
      pure outputFloat
