{-# LANGUAGE OverloadedStrings #-}

{-|
Module: Gemma.Layers.Linear
Description: Linear (fully-connected) layer with matrix-vector multiplication

Linear layers perform the operation: y = W @ x
where:
  - W is a weight matrix of shape [out_size, in_size]
  - x is an input vector of shape [in_size]
  - y is an output vector of shape [out_size]

This is the fundamental building block for Q/K/V projections and MLPs.
-}

module Gemma.Layers.Linear
  ( runLinear
  , runLinearWithContext
  , runLinearGPU
  , runLinearPreloadedGPU
  , linearShader
  ) where

import Graphics.WebGPU.Dawn.ContT
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)

-- | WGSL compute shader for matrix-vector multiplication
--
-- Computes y = W @ x where:
-- - W is [out_size, in_size] stored in row-major order
-- - x is [in_size]
-- - y is [out_size]
--
-- Each thread computes one output element.
--
-- useFP16: When True, uses FP16 for 2x memory bandwidth improvement
-- useVec4: When True, uses vec4 SIMD for 4x additional speedup
linearShader :: Int -> Int -> Bool -> Bool -> String
linearShader outSize inSize useFP16 useVec4 =
  let floatType = if useFP16 then "f16" else "f32"
      floatLit = if useFP16 then "h" else ""
      enableDirective = if useFP16 then ["enable f16;", ""] else []
      vec4Iters = inSize `div` 4
      remainder = inSize `mod` 4
  in unlines $
  [ "// Linear layer: matrix-vector multiplication"
  , "// W: [out_size, in_size] in row-major order"
  , "// x: [in_size]"
  , "// y: [out_size]"
  , if useVec4 then "// Using vec4 SIMD optimization (4x speedup)" else ""
  ] ++ enableDirective ++
  [ "@group(0) @binding(0) var<storage, read_write> weight: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(1) var<storage, read_write> input: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(2) var<storage, read_write> output: array<" ++ floatType ++ ">;"
  , ""
  , "const OUT_SIZE: u32 = " ++ show outSize ++ "u;"
  , "const IN_SIZE: u32 = " ++ show inSize ++ "u;"
  , ""
  , "@compute @workgroup_size(256)"
  , "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {"
  , "  let row = gid.x;"
  , "  "
  , "  if (row < OUT_SIZE) {"
  , "    // CRITICAL: Always use FP32 for accumulation to prevent overflow"
  , "    // Even when storage is FP16, we must accumulate in FP32"
  , "    var sum: f32 = 0.0;"
  , "    "
  ] ++ (if useVec4 && vec4Iters > 0 then
      [ "    // Vectorized loop: process 4 elements at a time"
      , "    for (var i: u32 = 0u; i < " ++ show vec4Iters ++ "u; i = i + 1u) {"
      , "      let idx = i * 4u;"
      , "      let w_base = row * IN_SIZE + idx;"
      , "      "
      , "      // Load 4 input elements as vec4 (FP16 storage)"
      , "      let in_vec = vec4<" ++ floatType ++ ">("
      , "        input[idx], input[idx + 1u], input[idx + 2u], input[idx + 3u]"
      , "      );"
      , "      "
      , "      // Load 4 weight elements as vec4 (FP16 storage)"
      , "      let w_vec = vec4<" ++ floatType ++ ">("
      , "        weight[w_base], weight[w_base + 1u], weight[w_base + 2u], weight[w_base + 3u]"
      , "      );"
      , "      "
      , "      // Multiply and accumulate in FP32"
      , "      let prod = in_vec * w_vec;"
      , "      sum = sum + f32(prod.x) + f32(prod.y) + f32(prod.z) + f32(prod.w);"
      , "    }"
      , "    "
      ] else []) ++
  (if useVec4 && remainder > 0 then
      [ "    // Handle remainder elements (scalar)"
      , "    for (var i: u32 = " ++ show (vec4Iters * 4) ++ "u; i < IN_SIZE; i = i + 1u) {"
      , "      let weight_idx = row * IN_SIZE + i;"
      , "      // Cast to FP32 before multiply and accumulate"
      , "      sum = sum + f32(weight[weight_idx]) * f32(input[i]);"
      , "    }"
      ] else if not useVec4 then
      [ "    // Compute dot product of row with input vector"
      , "    for (var i: u32 = 0u; i < IN_SIZE; i = i + 1u) {"
      , "      let weight_idx = row * IN_SIZE + i;"
      , "      // Cast to FP32 before multiply and accumulate"
      , "      sum = sum + f32(weight[weight_idx]) * f32(input[i]);"
      , "    }"
      ] else []) ++
  [ "    "
  , "    // Cast back to storage type when writing output"
  , if useFP16
    then "    output[row] = " ++ floatType ++ "(sum);"
    else "    output[row] = sum;"
  , "  }"
  , "}"
  ]

-- | Run Linear layer (matrix-vector multiplication) on GPU
--
-- Takes:
-- - weight: flattened weight matrix [out_size * in_size] in row-major order
-- - input: input vector [in_size]
-- - outSize: number of output features
-- - inSize: number of input features
--
-- Returns: output vector [out_size]
runLinear :: Vector Float -> Vector Float -> Int -> Int -> ContT r IO (Vector Float)
runLinear weight input outSize inSize = do
  -- Validate inputs
  if V.length weight /= outSize * inSize
    then error $ "Linear: weight size mismatch: " ++ show (V.length weight) ++ " vs " ++ show (outSize * inSize)
    else pure ()

  if V.length input /= inSize
    then error $ "Linear: input size mismatch: " ++ show (V.length input) ++ " vs " ++ show inSize
    else pure ()

  -- Create GPU context
  ctx <- createContext

  -- Create tensors
  let weightShape = Shape [outSize * inSize]
      inputShape = Shape [inSize]
      outputShape = Shape [outSize]

  weightTensor <- createTensorWithData ctx weightShape weight
  inputTensor <- createTensorWithData ctx inputShape input
  outputTensor <- createTensor ctx outputShape F32

  -- Compile shader (use FP32, no vec4 for now)
  let shaderCode = linearShader outSize inSize False False
  code <- createKernelCode shaderCode

  -- Create kernel with enough workgroups to cover all output rows
  let numWorkgroups = (outSize + 255) `div` 256  -- Ceiling division
  kernel <- createKernel ctx code [weightTensor, inputTensor, outputTensor]
            (WorkgroupSize numWorkgroups 1 1)

  -- Dispatch kernel
  liftIO $ dispatchKernelAsync ctx kernel

  -- Read result from GPU
  result <- liftIO $ fromGPU ctx outputTensor outSize

  pure result

-- | Run Linear with given context (for use in larger pipelines)
runLinearWithContext :: Context -> Vector Float -> Vector Float -> Int -> Int -> ContT r IO (Vector Float)
runLinearWithContext ctx weight input outSize inSize = do
  -- Validate inputs
  if V.length weight /= outSize * inSize
    then error $ "Linear: weight size mismatch: " ++ show (V.length weight) ++ " vs " ++ show (outSize * inSize)
    else pure ()

  if V.length input /= inSize
    then error $ "Linear: input size mismatch: " ++ show (V.length input) ++ " vs " ++ show inSize
    else pure ()

  -- Create tensors
  let weightShape = Shape [outSize * inSize]
      inputShape = Shape [inSize]
      outputShape = Shape [outSize]

  weightTensor <- createTensorWithData ctx weightShape weight
  inputTensor <- createTensorWithData ctx inputShape input
  outputTensor <- createTensor ctx outputShape F32

  -- Compile shader (use FP32, no vec4 for now)
  let shaderCode = linearShader outSize inSize False False
  code <- createKernelCode shaderCode

  -- Create kernel with enough workgroups to cover all output rows
  let numWorkgroups = (outSize + 255) `div` 256
  kernel <- createKernel ctx code [weightTensor, inputTensor, outputTensor]
            (WorkgroupSize numWorkgroups 1 1)

  -- Dispatch kernel
  liftIO $ dispatchKernelAsync ctx kernel

  -- Read result from GPU
  result <- liftIO $ fromGPU ctx outputTensor outSize

  pure result

-- | GPU-resident Linear layer - keeps tensors on GPU (NO CPU transfers!)
--
-- Performs matrix-vector multiplication: y = W @ x
-- Input tensor stays on GPU, eliminating transfer overhead.
--
-- Parameters:
-- - ctx: Reusable GPU context
-- - weight: Weight matrix [outSize, inSize] (uploaded once)
-- - inputTensor: Input tensor already on GPU [inSize]
-- - outSize: Output dimension
-- - inSize: Input dimension
--
-- Returns: Output tensor on GPU (NO download)
runLinearGPU :: Context
             -> Vector Float
             -> Tensor dtype
             -> Int
             -> Int
             -> ContT r IO (Tensor dtype)
runLinearGPU ctx weight inputTensor outSize inSize = do
  -- Validate
  if V.length weight /= outSize * inSize
    then error $ "LinearGPU: weight size mismatch"
    else pure ()

  -- Create weight and output tensors (input already on GPU!)
  let weightShape = Shape [outSize, inSize]
      outputShape = Shape [outSize]
  weightTensor <- createTensorWithData ctx weightShape weight
  outputTensor <- createTensor ctx outputShape F32

  -- Compile shader (use FP32, no vec4 for now)
  let shaderCode = linearShader outSize inSize False False
  code <- createKernelCode shaderCode

  -- Create and dispatch kernel
  let numWorkgroups = (outSize + 255) `div` 256
  kernel <- createKernel ctx code [weightTensor, inputTensor, outputTensor]
            (WorkgroupSize numWorkgroups 1 1)
  liftIO $ dispatchKernelAsync ctx kernel

  -- Return GPU tensor (NO download!)
  pure outputTensor

-- | GPU-resident Linear with PRE-UPLOADED weights AND PRE-COMPILED shader!
--
-- Takes GPU tensor for weights AND pre-compiled KernelCode.
-- Zero uploads, zero shader compilations - maximum performance!
runLinearPreloadedGPU :: Context
                      -> Tensor dtype  -- Weight tensor on GPU [outSize, inSize]
                      -> Tensor dtype  -- Input tensor on GPU [inSize]
                      -> Tensor dtype  -- Output buffer (pre-allocated, REUSED!)
                      -> KernelCode      -- PRE-COMPILED shader!
                      -> Int             -- outSize
                      -> ContT r IO ()
runLinearPreloadedGPU ctx weightTensor inputTensor outputTensor code outSize = do
  let numWorkgroups = (outSize + 255) `div` 256
  kernel <- createKernel ctx code [weightTensor, inputTensor, outputTensor]
            (WorkgroupSize numWorkgroups 1 1)
  liftIO $ dispatchKernelAsync ctx kernel
