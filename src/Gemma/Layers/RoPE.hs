{-# LANGUAGE OverloadedStrings #-}

{-|
Module: Gemma.Layers.RoPE
Description: Rotary Positional Embeddings (RoPE)

RoPE encodes position information by applying rotation matrices to pairs of dimensions.
For each pair (i, i+1), we rotate by an angle θ = position * base^(-2i/dim):

  [x_i']     [cos(θ)  -sin(θ)] [x_i  ]
  [x_i+1'] = [sin(θ)   cos(θ)] [x_i+1]

This applies to Q and K projections in attention to encode relative positions.

Reference: RoFormer - Enhanced Transformer with Rotary Position Embedding
           https://arxiv.org/abs/2104.09864
-}

module Gemma.Layers.RoPE
  ( runRoPE
  , runRoPEWithContext
  , ropeShader
  ) where

import Graphics.WebGPU.Dawn.ContT
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)

-- | WGSL compute shader for RoPE
--
-- Applies rotary positional embeddings to input tensor.
-- Each pair of dimensions (2i, 2i+1) is rotated by position-dependent angle.
ropeShader :: Int -> Int -> Float -> String
ropeShader headDim position baseFreq = unlines
  [ "// RoPE (Rotary Positional Embeddings)"
  , "// Applies rotation to pairs of dimensions"
  , ""
  , "@group(0) @binding(0) var<storage, read_write> input: array<f32>;"
  , "@group(0) @binding(1) var<storage, read_write> output: array<f32>;"
  , ""
  , "const HEAD_DIM: u32 = " ++ show headDim ++ "u;"
  , "const POSITION: f32 = " ++ show position ++ ".0;"
  , "const BASE: f32 = " ++ show baseFreq ++ ";"
  , "const PI: f32 = 3.14159265359;"
  , ""
  , "@compute @workgroup_size(256)"
  , "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {"
  , "  let pair_idx = gid.x;"
  , "  let num_pairs = HEAD_DIM / 2u;"
  , "  "
  , "  if (pair_idx < num_pairs) {"
  , "    let i = pair_idx * 2u;"
  , "    "
  , "    // Compute rotation frequency for this pair"
  , "    // freq = base^(-2i/dim)"
  , "    let exponent = -2.0 * f32(pair_idx) / f32(HEAD_DIM);"
  , "    let freq = pow(BASE, exponent);"
  , "    "
  , "    // Compute rotation angle"
  , "    let theta = POSITION * freq;"
  , "    "
  , "    // Compute sin and cos"
  , "    let cos_theta = cos(theta);"
  , "    let sin_theta = sin(theta);"
  , "    "
  , "    // Read pair of values"
  , "    let x0 = input[i];"
  , "    let x1 = input[i + 1u];"
  , "    "
  , "    // Apply 2D rotation matrix"
  , "    // [x0']   [cos  -sin] [x0]"
  , "    // [x1'] = [sin   cos] [x1]"
  , "    output[i]     = x0 * cos_theta - x1 * sin_theta;"
  , "    output[i + 1u] = x0 * sin_theta + x1 * cos_theta;"
  , "  }"
  , "}"
  ]

-- | WGSL compute shader for RoPE applied to multiple heads
--
-- Same as ropeShader but handles multiple heads in a single kernel launch
ropeShaderMultiHead :: Int -> Int -> Int -> Float -> String
ropeShaderMultiHead headDim numHeads position baseFreq = unlines
  [ "// RoPE (Rotary Positional Embeddings) for multiple heads"
  , "// Applies rotation to pairs of dimensions across all heads"
  , ""
  , "@group(0) @binding(0) var<storage, read_write> input: array<f32>;"
  , "@group(0) @binding(1) var<storage, read_write> output: array<f32>;"
  , ""
  , "const HEAD_DIM: u32 = " ++ show headDim ++ "u;"
  , "const NUM_HEADS: u32 = " ++ show numHeads ++ "u;"
  , "const POSITION: f32 = " ++ show position ++ ".0;"
  , "const BASE: f32 = " ++ show baseFreq ++ ";"
  , "const PI: f32 = 3.14159265359;"
  , ""
  , "@compute @workgroup_size(256)"
  , "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {"
  , "  let global_pair_idx = gid.x;"
  , "  let total_pairs = (HEAD_DIM * NUM_HEADS) / 2u;"
  , "  "
  , "  if (global_pair_idx < total_pairs) {"
  , "    // Determine which head and which pair within that head"
  , "    let pairs_per_head = HEAD_DIM / 2u;"
  , "    let head_idx = global_pair_idx / pairs_per_head;"
  , "    let local_pair_idx = global_pair_idx % pairs_per_head;"
  , "    "
  , "    // Global index within the flattened array"
  , "    let i = global_pair_idx * 2u;"
  , "    "
  , "    // Compute rotation frequency for this pair"
  , "    let exponent = -2.0 * f32(local_pair_idx) / f32(HEAD_DIM);"
  , "    let freq = pow(BASE, exponent);"
  , "    "
  , "    // Compute rotation angle"
  , "    let theta = POSITION * freq;"
  , "    "
  , "    // Compute sin and cos"
  , "    let cos_theta = cos(theta);"
  , "    let sin_theta = sin(theta);"
  , "    "
  , "    // Read pair of values"
  , "    let x0 = input[i];"
  , "    let x1 = input[i + 1u];"
  , "    "
  , "    // Apply 2D rotation matrix"
  , "    output[i]     = x0 * cos_theta - x1 * sin_theta;"
  , "    output[i + 1u] = x0 * sin_theta + x1 * cos_theta;"
  , "  }"
  , "}"
  ]

-- | Run RoPE on GPU
--
-- Takes:
-- - input: input vector [head_dim]
-- - position: sequence position (integer)
-- - headDim: dimension of attention head (must be even)
-- - baseFreq: RoPE base frequency (e.g., 10000.0 or 1000000.0)
--
-- Returns: rotated vector [head_dim]
runRoPE :: Vector Float -> Int -> Int -> Float -> ContT r IO (Vector Float)
runRoPE input position headDim baseFreq = do
  -- Validate inputs
  if V.length input /= headDim
    then error $ "RoPE: input size mismatch: " ++ show (V.length input) ++ " vs " ++ show headDim
    else pure ()

  if headDim `mod` 2 /= 0
    then error $ "RoPE: head_dim must be even, got: " ++ show headDim
    else pure ()

  -- Create GPU context
  ctx <- createContext

  -- Create tensors
  let shape = Shape [headDim]
  inputTensor <- createTensorWithData ctx shape input
  outputTensor <- createTensor ctx shape F32

  -- Compile shader
  let shaderCode = ropeShader headDim position baseFreq
  code <- createKernelCode shaderCode

  -- Create kernel - one thread per pair
  let numPairs = headDim `div` 2
      numWorkgroups = (numPairs + 255) `div` 256
  kernel <- createKernel ctx code [inputTensor, outputTensor]
            (WorkgroupSize numWorkgroups 1 1)

  -- Dispatch kernel
  liftIO $ dispatchKernel ctx kernel

  -- Read result from GPU
  result <- liftIO $ fromGPU ctx outputTensor headDim

  pure result

-- | Run RoPE with given context for multiple heads
-- This version takes concatenated heads and applies RoPE to each head independently
runRoPEWithContext :: Context -> Vector Float -> Int -> Int -> Int -> Float -> ContT r IO (Vector Float)
runRoPEWithContext ctx input position headDim numHeads baseFreq = do
  let totalSize = numHeads * headDim

  -- Validate inputs
  if V.length input /= totalSize
    then error $ "RoPE: input size mismatch: " ++ show (V.length input) ++ " vs " ++ show totalSize
    else pure ()

  if headDim `mod` 2 /= 0
    then error $ "RoPE: head_dim must be even, got: " ++ show headDim
    else pure ()

  -- Create tensors
  let shape = Shape [totalSize]
  inputTensor <- createTensorWithData ctx shape input
  outputTensor <- createTensor ctx shape F32

  -- Compile shader
  let shaderCode = ropeShaderMultiHead headDim numHeads position baseFreq
  code <- createKernelCode shaderCode

  -- Create kernel - one thread per element pair across all heads
  let numPairs = totalSize `div` 2
      numWorkgroups = (numPairs + 255) `div` 256
  kernel <- createKernel ctx code [inputTensor, outputTensor]
            (WorkgroupSize numWorkgroups 1 1)

  -- Dispatch kernel
  liftIO $ dispatchKernel ctx kernel

  -- Read result from GPU
  result <- liftIO $ fromGPU ctx outputTensor totalSize

  pure result
