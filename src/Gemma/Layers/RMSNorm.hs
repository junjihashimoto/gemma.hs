{-# LANGUAGE OverloadedStrings #-}

{-|
Module: Gemma.Layers.RMSNorm
Description: Root Mean Square Layer Normalization

RMSNorm is a simpler alternative to LayerNorm that normalizes using only
the root mean square (RMS) of the inputs.

Formula:
  RMSNorm(x) = (x / RMS(x)) * weight
  where RMS(x) = sqrt(mean(x²) + ε)

Reference: https://arxiv.org/abs/1910.07467
-}

module Gemma.Layers.RMSNorm
  ( runRMSNorm
  , runRMSNormVariant
  , runRMSNormWithContext
  , runRMSNormGPU
  , runRMSNormPreloadedGPU
  , runRMSNormLinearFusedGPU
  , runRMSNormLinearFusedPreloadedGPU
  , rmsNormShader
  , rmsNormLinearFusedShader
  ) where

import Graphics.WebGPU.Dawn.ContT
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)

-- | WGSL compute shader for RMSNorm
--
-- This shader computes RMSNorm in two phases:
-- 1. Phase 1: Compute RMS (root mean square) using parallel reduction in shared memory
-- 2. Phase 2: Normalize and apply weights in parallel
--
-- Uses workgroup shared memory for efficient parallel reduction.
--
-- When zeroCentered = True, uses (1 + weight) for Gemma 3 compatibility
-- When zeroCentered = False, uses weight directly for Gemma 1/2
-- useFP16 = True uses FP16 for 2x memory bandwidth
-- useVec4 = True uses vec4 SIMD for 4x additional speedup
rmsNormShader :: Bool -> Int -> Bool -> Bool -> String
rmsNormShader useFP16 hiddenSize zeroCentered useVec4 =
  let floatType = if useFP16 then "f16" else "f32"
      floatLit = if useFP16 then "h" else ""
      enableDirective = if useFP16 then ["enable f16;", ""] else []
      vec4Iters = hiddenSize `div` 4
      remainder = hiddenSize `mod` 4
  in unlines $
  [ "// RMSNorm compute shader with parallel workgroup reduction"
  , "// Input:  x (hidden_size,)"
  , "// Weight: w (hidden_size,)"
  , "// Output: y (hidden_size,)"
  ] ++ enableDirective ++
  [ "@group(0) @binding(0) var<storage, read_write> input: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(1) var<storage, read_write> weight: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(2) var<storage, read_write> output: array<" ++ floatType ++ ">;"
  , ""
  , "const HIDDEN_SIZE: u32 = " ++ show hiddenSize ++ "u;"
  , "const EPSILON: " ++ floatType ++ " = 1e-6" ++ floatLit ++ ";"
  , "const WORKGROUP_SIZE: u32 = 256u;"
  , ""
  , "var<workgroup> shared_sum: array<" ++ floatType ++ ", WORKGROUP_SIZE>;"
  , ""
  , "@compute @workgroup_size(256)"
  , "fn main(@builtin(local_invocation_id) lid: vec3<u32>,"
  , "        @builtin(workgroup_id) wid: vec3<u32>) {"
  , ""
  , "  let tid = lid.x;"
  , "  "
  , "  // Phase 1: Compute partial sum of squares"
  , "  var sum_sq: " ++ floatType ++ " = 0.0" ++ floatLit ++ ";"
  ] ++ (if useVec4 && vec4Iters > 0 then
      [ "  // Vectorized loop: process 4 elements at a time"
      , "  for (var i: u32 = tid; i < " ++ show vec4Iters ++ "u; i = i + WORKGROUP_SIZE) {"
      , "    let idx = i * 4u;"
      , "    let vals = vec4<" ++ floatType ++ ">("
      , "      input[idx], input[idx + 1u], input[idx + 2u], input[idx + 3u]"
      , "    );"
      , "    let sq = vals * vals;"
      , "    sum_sq = sum_sq + sq.x + sq.y + sq.z + sq.w;"
      , "  }"
      ] else []) ++
  (if useVec4 && remainder > 0 then
      [ "  // Handle remainder elements (scalar)"
      , "  for (var i: u32 = tid + " ++ show (vec4Iters * 4) ++ "u; i < HIDDEN_SIZE; i = i + WORKGROUP_SIZE) {"
      , "    let val = input[i];"
      , "    sum_sq = sum_sq + (val * val);"
      , "  }"
      ] else if not useVec4 then
      [ "  for (var i: u32 = tid; i < HIDDEN_SIZE; i = i + WORKGROUP_SIZE) {"
      , "    let val = input[i];"
      , "    sum_sq = sum_sq + (val * val);"
      , "  }"
      ] else []) ++
  [ "  shared_sum[tid] = sum_sq;"
  , "  workgroupBarrier();"
  , "  "
  , "  // Parallel reduction in shared memory"
  , "  for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {"
  , "    if (tid < stride) {"
  , "      shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];"
  , "    }"
  , "    workgroupBarrier();"
  , "  }"
  , "  "
  , "  // Thread 0 computes RMS and stores in shared memory"
  , "  if (tid == 0u) {"
  , "    let mean_sq = shared_sum[0] / " ++ floatType ++ "(HIDDEN_SIZE);"
  , "    let rms = sqrt(mean_sq + EPSILON);"
  , "    shared_sum[0] = rms;  // Store RMS for phase 2"
  , "  }"
  , "  workgroupBarrier();"
  , "  "
  , "  // Phase 2: Normalize and apply weights in parallel"
  , "  let rms = shared_sum[0];"
  ] ++ (if useVec4 && vec4Iters > 0 then
      [ "  // Vectorized loop: process 4 elements at a time"
      , "  for (var i: u32 = tid; i < " ++ show vec4Iters ++ "u; i = i + WORKGROUP_SIZE) {"
      , "    let idx = i * 4u;"
      , "    let in_vec = vec4<" ++ floatType ++ ">("
      , "      input[idx], input[idx + 1u], input[idx + 2u], input[idx + 3u]"
      , "    );"
      , "    let w_vec = vec4<" ++ floatType ++ ">("
      , "      weight[idx], weight[idx + 1u], weight[idx + 2u], weight[idx + 3u]"
      , "    );"
      , if zeroCentered
        then "    let norm_vec = (in_vec / rms) * (vec4<" ++ floatType ++ ">(1.0" ++ floatLit ++ ") + w_vec);"
        else "    let norm_vec = (in_vec / rms) * w_vec;"
      , "    output[idx] = norm_vec.x;"
      , "    output[idx + 1u] = norm_vec.y;"
      , "    output[idx + 2u] = norm_vec.z;"
      , "    output[idx + 3u] = norm_vec.w;"
      , "  }"
      ] else []) ++
  (if useVec4 && remainder > 0 then
      [ "  // Handle remainder elements (scalar)"
      , "  for (var i: u32 = tid + " ++ show (vec4Iters * 4) ++ "u; i < HIDDEN_SIZE; i = i + WORKGROUP_SIZE) {"
      , if zeroCentered
        then "    output[i] = (input[i] / rms) * (1.0" ++ floatLit ++ " + weight[i]);"
        else "    output[i] = (input[i] / rms) * weight[i];"
      , "  }"
      ] else if not useVec4 then
      [ "  for (var i: u32 = tid; i < HIDDEN_SIZE; i = i + WORKGROUP_SIZE) {"
      , if zeroCentered
        then "    output[i] = (input[i] / rms) * (1.0" ++ floatLit ++ " + weight[i]);"
        else "    output[i] = (input[i] / rms) * weight[i];"
      , "  }"
      ] else []) ++
  [ "}"
  ]

-- | Run RMSNorm on GPU (standard variant - for backward compatibility)
--
-- Takes input vector and weight vector, returns normalized output.
-- All computation happens on GPU.
-- Uses standard RMSNorm: output = (input / rms) * weight
runRMSNorm :: Vector Float -> Vector Float -> ContT r IO (Vector Float)
runRMSNorm = runRMSNormVariant False

-- | Run RMSNorm on GPU with configurable zero-centered mode
--
-- Takes input vector and weight vector, returns normalized output.
-- All computation happens on GPU.
--
-- When zeroCentered=True (Gemma 3): output = (input / rms) * (1 + weight)
-- When zeroCentered=False (Gemma 1/2): output = (input / rms) * weight
runRMSNormVariant :: Bool -> Vector Float -> Vector Float -> ContT r IO (Vector Float)
runRMSNormVariant zeroCentered input weight = do
  let hiddenSize = V.length input

  -- Validate inputs
  if V.length weight /= hiddenSize
    then error $ "RMSNorm: weight length mismatch: " ++ show (V.length weight) ++ " vs " ++ show hiddenSize
    else pure ()

  -- Create GPU context
  ctx <- createContext

  -- Create tensors
  let shape = Shape [hiddenSize]
  inputTensor <- createTensorWithData ctx shape input
  weightTensor <- createTensorWithData ctx shape weight
  outputTensor <- createTensor ctx shape F32

  -- Compile shader with appropriate mode
  let shaderCode = rmsNormShader False hiddenSize zeroCentered False  -- FP32, configurable zero-centered, no vec4 for now
  code <- createKernelCode shaderCode

  -- Create kernel with single workgroup (all 256 threads work together)
  -- Note: We only need 1 workgroup since all threads cooperate via shared memory
  kernel <- createKernel ctx code [inputTensor, weightTensor, outputTensor]
            (WorkgroupSize 1 1 1)

  -- Dispatch kernel
  liftIO $ dispatchKernelAsync ctx kernel

  -- Read result from GPU
  result <- liftIO $ fromGPU ctx outputTensor hiddenSize

  pure result

-- | Run RMSNorm with given context (for use in larger pipelines)
-- Uses standard (non-zero-centered) RMSNorm for backward compatibility
runRMSNormWithContext :: Context -> Vector Float -> Vector Float -> ContT r IO (Vector Float)
runRMSNormWithContext ctx input weight = do
  let hiddenSize = V.length input

  -- Validate inputs
  if V.length weight /= hiddenSize
    then error $ "RMSNorm: weight length mismatch: " ++ show (V.length weight) ++ " vs " ++ show hiddenSize
    else pure ()

  -- Create tensors
  let shape = Shape [hiddenSize]
  inputTensor <- createTensorWithData ctx shape input
  weightTensor <- createTensorWithData ctx shape weight
  outputTensor <- createTensor ctx shape F32

  -- Compile shader with standard (non-zero-centered) mode for backward compatibility
  let shaderCode = rmsNormShader False hiddenSize False False  -- FP32, standard mode, no vec4 for now
  code <- createKernelCode shaderCode

  -- Create kernel with single workgroup
  kernel <- createKernel ctx code [inputTensor, weightTensor, outputTensor]
            (WorkgroupSize 1 1 1)

  -- Dispatch kernel
  liftIO $ dispatchKernelAsync ctx kernel

  -- Read result from GPU
  result <- liftIO $ fromGPU ctx outputTensor hiddenSize

  pure result

-- | GPU-resident RMSNorm - keeps tensors on GPU (NO CPU transfers!)
--
-- This version accepts GPU tensors and returns GPU tensor, eliminating
-- the CPU↔GPU transfer overhead. Use this for building GPU-resident pipelines.
--
-- Parameters:
-- - ctx: Reusable GPU context
-- - inputTensor: Input tensor already on GPU [hiddenSize]
-- - weight: Weight vector (uploaded once)
-- - hiddenSize: Hidden dimension
-- - zeroCentered: Use zero-centered weights (Gemma 3 style)
--
-- Returns: Output tensor on GPU (NO download)
runRMSNormGPU :: Context
              -> Tensor dtype
              -> Vector Float
              -> Int
              -> Bool
              -> ContT r IO (Tensor dtype)
runRMSNormGPU ctx inputTensor weight hiddenSize zeroCentered = do
  -- Validate inputs
  if V.length weight /= hiddenSize
    then error $ "RMSNormGPU: weight length mismatch: " ++ show (V.length weight) ++ " vs " ++ show hiddenSize
    else pure ()

  -- Create weight tensor and output tensor (input already on GPU!)
  let shape = Shape [hiddenSize]
  weightTensor <- createTensorWithData ctx shape weight
  outputTensor <- createTensor ctx shape F32

  -- Compile shader
  let shaderCode = rmsNormShader False hiddenSize zeroCentered False  -- FP32, no vec4 for now
  code <- createKernelCode shaderCode

  -- Create kernel with single workgroup
  kernel <- createKernel ctx code [inputTensor, weightTensor, outputTensor]
            (WorkgroupSize 1 1 1)

  -- Dispatch kernel
  liftIO $ dispatchKernelAsync ctx kernel

  -- Return GPU tensor (NO download!)
  pure outputTensor

-- | Fused RMSNorm + Linear (GPU-resident)
--
-- Combines RMSNorm and Linear into a single kernel dispatch, eliminating
-- one GPU synchronization point. This is a critical optimization.
--
-- Pipeline: input → RMSNorm(input) → Linear(normalized) → output
-- All in ONE kernel launch instead of TWO!
runRMSNormLinearFusedGPU :: Context
                         -> Tensor dtype  -- Input tensor [hiddenDim]
                         -> Vector Float  -- RMSNorm weights [hiddenDim]
                         -> Vector Float  -- Linear weights [outDim, hiddenDim]
                         -> Int           -- hiddenDim
                         -> Int           -- outDim
                         -> Bool          -- zeroCentered
                         -> ContT r IO (Tensor dtype)
runRMSNormLinearFusedGPU ctx inputTensor normWeights linearWeights hiddenDim outDim zeroCentered = do
  -- Create weight tensors
  let normShape = Shape [hiddenDim]
      linearShape = Shape [outDim, hiddenDim]
      outShape = Shape [outDim]
  
  normWeightsTensor <- createTensorWithData ctx normShape normWeights
  linearWeightsTensor <- createTensorWithData ctx linearShape linearWeights
  outputTensor <- createTensor ctx outShape F32

  -- Fused shader that does RMSNorm + Linear in one pass
  let shaderCode = rmsNormLinearFusedShader False hiddenDim outDim zeroCentered  -- FP32 for now
  code <- createKernelCode shaderCode

  -- One kernel launch for both operations!
  let numWorkgroups = (outDim + 255) `div` 256
  kernel <- createKernel ctx code [inputTensor, normWeightsTensor, linearWeightsTensor, outputTensor]
            (WorkgroupSize numWorkgroups 1 1)
  liftIO $ dispatchKernelAsync ctx kernel

  pure outputTensor

-- | Fused RMSNorm + Linear with PRE-UPLOADED weights AND PRE-COMPILED shader!
--
-- Takes GPU tensors for weights AND pre-compiled KernelCode.
-- Zero uploads, zero shader compilations - maximum performance!
runRMSNormLinearFusedPreloadedGPU :: Context
                                  -> Tensor dtype  -- Input tensor [hiddenDim]
                                  -> Tensor dtype  -- RMSNorm weights on GPU [hiddenDim]
                                  -> Tensor dtype  -- Linear weights on GPU [outDim, hiddenDim]
                                  -> KernelCode    -- PRE-COMPILED shader!
                                  -> Int           -- outDim
                                  -> ContT r IO (Tensor dtype)
runRMSNormLinearFusedPreloadedGPU ctx inputTensor normWeightsTensor linearWeightsTensor code outDim = do
  let outShape = Shape [outDim]
  outputTensor <- createTensor ctx outShape F32

  let numWorkgroups = (outDim + 255) `div` 256
  kernel <- createKernel ctx code [inputTensor, normWeightsTensor, linearWeightsTensor, outputTensor]
            (WorkgroupSize numWorkgroups 1 1)
  liftIO $ dispatchKernelAsync ctx kernel

  pure outputTensor

-- Fused RMSNorm + Linear shader - Parameterized FP16/FP32
rmsNormLinearFusedShader :: Bool -> Int -> Int -> Bool -> String
rmsNormLinearFusedShader useFP16 hiddenDim outDim zeroCentered =
  let floatType = if useFP16 then "f16" else "f32"
      floatLit = if useFP16 then "h" else ""
      enableDirective = if useFP16 then ["enable f16;", ""] else []
  in unlines $
  [ "// Fused RMSNorm + Linear"
  , "// Eliminates one kernel dispatch by combining operations"
  ] ++ enableDirective ++
  [ "@group(0) @binding(0) var<storage, read_write> input: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(1) var<storage, read_write> norm_weight: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(2) var<storage, read_write> linear_weight: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(3) var<storage, read_write> output: array<" ++ floatType ++ ">;"
  , ""
  , "const HIDDEN_DIM: u32 = " ++ show hiddenDim ++ "u;"
  , "const OUT_DIM: u32 = " ++ show outDim ++ "u;"
  , "const EPSILON: " ++ floatType ++ " = 1e-6" ++ floatLit ++ ";"
  , ""
  , "@compute @workgroup_size(256)"
  , "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {"
  , "  let out_idx = gid.x;"
  , "  if (out_idx >= OUT_DIM) { return; }"
  , ""
  , "  // Step 1: Compute RMS (shared across all threads)"
  , "  var sum_sq: " ++ floatType ++ " = 0.0" ++ floatLit ++ ";"
  , "  for (var i: u32 = 0u; i < HIDDEN_DIM; i++) {"
  , "    let val = input[i];"
  , "    sum_sq += val * val;"
  , "  }"
  , "  let mean_sq = sum_sq / " ++ floatType ++ "(HIDDEN_DIM);"
  , "  let rms = sqrt(mean_sq + EPSILON);"
  , ""
  , "  // Step 2: Compute Linear(RMSNorm(input))"
  , "  var sum: " ++ floatType ++ " = 0.0" ++ floatLit ++ ";"
  , "  for (var i: u32 = 0u; i < HIDDEN_DIM; i++) {"
  , if zeroCentered
     then "    let normalized = (input[i] / rms) * (1.0" ++ floatLit ++ " + norm_weight[i]);"
     else "    let normalized = (input[i] / rms) * norm_weight[i];"
  , "    let weight_idx = out_idx * HIDDEN_DIM + i;"
  , "    sum += normalized * linear_weight[weight_idx];"
  , "  }"
  , "  output[out_idx] = sum;"
  , "}"
  ]

-- | GPU-resident RMSNorm with PRE-UPLOADED weights and PRE-COMPILED shader
--
-- This version accepts:
-- - inputTensor: Input tensor on GPU
-- - weightTensor: Pre-uploaded weight tensor on GPU
-- - code: Pre-compiled kernel code
-- - hiddenSize: Hidden dimension
--
-- Returns: Output tensor on GPU (NO download!)
-- | RMSNorm with pre-allocated output buffer (for kernel reuse optimization)
runRMSNormPreloadedGPU :: Context
                       -> Tensor dtype     -- Input tensor on GPU
                       -> Tensor dtype     -- Weight tensor on GPU (pre-uploaded)
                       -> Tensor dtype     -- Output buffer (pre-allocated, REUSED!)
                       -> KernelCode       -- Pre-compiled shader
                       -> Int              -- hiddenSize
                       -> ContT r IO ()
runRMSNormPreloadedGPU ctx inputTensor weightTensor outputTensor code hiddenSize = do
  -- Create kernel with pre-compiled code and pre-allocated output
  -- WebGPU will cache this bind group since tensor pointers are stable!
  kernel <- createKernel ctx code [inputTensor, weightTensor, outputTensor]
            (WorkgroupSize 1 1 1)  -- Single workgroup for RMS computation

  -- Dispatch asynchronously
  liftIO $ dispatchKernelAsync ctx kernel
