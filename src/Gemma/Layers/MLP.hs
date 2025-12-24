{-# LANGUAGE OverloadedStrings #-}

{-|
Module: Gemma.Layers.MLP
Description: Multi-Layer Perceptron with GeGLU activation

GeGLU (Gated Linear Unit with GELU) is defined as:
  GeGLU(x, W, V) = GELU(x @ W) ⊙ (x @ V)

Where:
  - GELU is the Gaussian Error Linear Unit activation
  - ⊙ is element-wise multiplication
  - W and V are weight matrices

The full MLP in Gemma is:
  MLP(x) = GeGLU(x, W_gate, W_up) @ W_down

Reference: GLU Variants Improve Transformer
           https://arxiv.org/abs/2002.05202
-}

module Gemma.Layers.MLP
  ( runGELU
  , runGeGLU
  , runGeGLUWithContext
  , runGELUGPU
  , runElementwiseMultiplyGPU
  , runGELUMultiplyFusedGPU
  , runRMSNormGateUpFusedGPU
  , runRMSNormGateUpFusedPreloadedGPU
  , geluShader
  , rmsNormGateUpFusedShader
  , geluMultiplyFusedShader
  , runElementwiseAddWithContext
  , runResidualAddGPU
  , residualAddShader
  , ffnOutputFusedShader
  , runFFNOutputFusedPreloadedGPU
  ) where

import Graphics.WebGPU.Dawn.ContT
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)

-- | WGSL shader for GELU activation
--
-- GELU(x) = x * Φ(x) where Φ is the CDF of standard normal distribution
-- Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
geluShader :: Bool -> Int -> String
geluShader useFP16 size =
  let floatType = if useFP16 then "f16" else "f32"
      floatLit = if useFP16 then "h" else ""
      enableDirective = if useFP16 then ["enable f16;", ""] else []
  in unlines $
  [ "// GELU (Gaussian Error Linear Unit) activation"
  ] ++ enableDirective ++
  [ "@group(0) @binding(0) var<storage, read_write> input: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(1) var<storage, read_write> output: array<" ++ floatType ++ ">;"
  , ""
  , "const SIZE: u32 = " ++ show size ++ "u;"
  , "const SQRT_2_OVER_PI: " ++ floatType ++ " = 0.7978845608" ++ floatLit ++ ";"
  , "const COEFF: " ++ floatType ++ " = 0.044715" ++ floatLit ++ ";"
  , ""
  , "@compute @workgroup_size(256)"
  , "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {"
  , "  let i = gid.x;"
  , "  "
  , "  if (i < SIZE) {"
  , "    let x = input[i];"
  , "    "
  , "    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"
  , "    // Clamp inner value to prevent tanh overflow (tanh saturates around ±10)"
  , "    let x_cubed = x * x * x;"
  , "    let inner = SQRT_2_OVER_PI * (x + COEFF * x_cubed);"
  , "    let clamped_inner = clamp(inner, -10.0" ++ floatLit ++ ", 10.0" ++ floatLit ++ ");"
  , "    let tanh_val = tanh(clamped_inner);"
  , "    "
  , "    output[i] = 0.5" ++ floatLit ++ " * x * (1.0" ++ floatLit ++ " + tanh_val);"
  , "  }"
  , "}"
  ]

-- | Run GELU activation on GPU
--
-- Takes: input vector
-- Returns: GELU(input)
runGELU :: Vector Float -> ContT r IO (Vector Float)
runGELU input = do
  -- Create GPU context
  ctx <- createContext
  runGELUWithContext ctx input

-- | Run GELU activation with given context
runGELUWithContext :: Context -> Vector Float -> ContT r IO (Vector Float)
runGELUWithContext ctx input = do
  let size = V.length input

  -- Create tensors
  let shape = Shape [size]
  inputTensor <- createTensorWithData ctx shape input
  outputTensor <- createTensor ctx shape F32

  -- Compile shader
  let shaderCode = geluShader False size  -- FP32 for now
  code <- createKernelCode shaderCode

  -- Create kernel
  let numWorkgroups = (size + 255) `div` 256
  kernel <- createKernel ctx code [inputTensor, outputTensor]
            (WorkgroupSize numWorkgroups 1 1)

  -- Dispatch kernel asynchronously (GPU can pipeline!)
  liftIO $ dispatchKernelAsync ctx kernel

  -- Read result from GPU (waits for completion automatically)
  result <- liftIO $ fromGPU ctx outputTensor size

  pure result

-- | WGSL shader for element-wise addition
--
-- Adds two vectors element-wise: c[i] = a[i] + b[i]
elementwiseAddShader :: Bool -> Int -> String
elementwiseAddShader useFP16 size =
  let floatType = if useFP16 then "f16" else "f32"
      enableDirective = if useFP16 then ["enable f16;", ""] else []
  in unlines $
  [ "// Element-wise addition (for residual connections)"
  ] ++ enableDirective ++
  [ "@group(0) @binding(0) var<storage, read_write> a: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(1) var<storage, read_write> b: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(2) var<storage, read_write> output: array<" ++ floatType ++ ">;"
  , ""
  , "const SIZE: u32 = " ++ show size ++ "u;"
  , ""
  , "@compute @workgroup_size(256)"
  , "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {"
  , "  let i = gid.x;"
  , "  "
  , "  if (i < SIZE) {"
  , "    output[i] = a[i] + b[i];"
  , "  }"
  , "}"
  ]

-- | WGSL shader for element-wise multiplication
--
-- Multiplies two vectors element-wise: c[i] = a[i] * b[i]
elementwiseMultiplyShader :: Bool -> Int -> String
elementwiseMultiplyShader useFP16 size =
  let floatType = if useFP16 then "f16" else "f32"
      enableDirective = if useFP16 then ["enable f16;", ""] else []
  in unlines $
  [ "// Element-wise multiplication"
  ] ++ enableDirective ++
  [ "@group(0) @binding(0) var<storage, read_write> a: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(1) var<storage, read_write> b: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(2) var<storage, read_write> output: array<" ++ floatType ++ ">;"
  , ""
  , "const SIZE: u32 = " ++ show size ++ "u;"
  , ""
  , "@compute @workgroup_size(256)"
  , "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {"
  , "  let i = gid.x;"
  , "  "
  , "  if (i < SIZE) {"
  , "    output[i] = a[i] * b[i];"
  , "  }"
  , "}"
  ]

-- | Helper: Element-wise add two vectors with given context
runElementwiseAddWithContext :: Context -> Vector Float -> Vector Float -> ContT r IO (Vector Float)
runElementwiseAddWithContext ctx a b = do
  let size = V.length a

  if V.length b /= size
    then error $ "ElementwiseAdd: size mismatch: " ++ show (V.length b) ++ " vs " ++ show size
    else pure ()

  let shape = Shape [size]

  aTensor <- createTensorWithData ctx shape a
  bTensor <- createTensorWithData ctx shape b
  outputTensor <- createTensor ctx shape F32

  let shaderCode = elementwiseAddShader False size  -- FP32 for now
  code <- createKernelCode shaderCode

  let numWorkgroups = (size + 255) `div` 256
  kernel <- createKernel ctx code [aTensor, bTensor, outputTensor]
            (WorkgroupSize numWorkgroups 1 1)

  liftIO $ dispatchKernelAsync ctx kernel
  result <- liftIO $ fromGPU ctx outputTensor size
  pure result

-- | Helper: Element-wise multiply two vectors with given context
runElementwiseMultiplyWithContext :: Context -> Vector Float -> Vector Float -> ContT r IO (Vector Float)
runElementwiseMultiplyWithContext ctx a b = do
  let size = V.length a

  if V.length b /= size
    then error $ "ElementwiseMultiply: size mismatch: " ++ show (V.length b) ++ " vs " ++ show size
    else pure ()

  let shape = Shape [size]

  aTensor <- createTensorWithData ctx shape a
  bTensor <- createTensorWithData ctx shape b
  outputTensor <- createTensor ctx shape F32

  let shaderCode = elementwiseMultiplyShader False size  -- FP32 for now
  code <- createKernelCode shaderCode

  let numWorkgroups = (size + 255) `div` 256
  kernel <- createKernel ctx code [aTensor, bTensor, outputTensor]
            (WorkgroupSize numWorkgroups 1 1)

  liftIO $ dispatchKernelAsync ctx kernel
  result <- liftIO $ fromGPU ctx outputTensor size
  pure result

-- | Run full MLP with GeGLU on GPU
--
-- GeGLU MLP computes:
-- 1. Gate projection: gate = x @ W_gate
-- 2. Up projection: up = x @ W_up
-- 3. GELU activation on gate: gelu_gate = GELU(gate)
-- 4. Element-wise multiply: intermediate = gelu_gate ⊙ up
-- 5. Down projection: output = intermediate @ W_down
--
-- This implements the full feedforward network used in Gemma.
runGeGLU :: Vector Float  -- Input [hidden_dim]
         -> Vector Float  -- W_gate [ffn_dim * hidden_dim] row-major
         -> Vector Float  -- W_up [ffn_dim * hidden_dim] row-major
         -> Vector Float  -- W_down [hidden_dim * ffn_dim] row-major
         -> Int           -- hidden_dim
         -> Int           -- ffn_dim
         -> ContT r IO (Vector Float)  -- Output [hidden_dim]
runGeGLU input wGate wUp wDown hiddenDim ffnDim = do
  -- Create single shared context for all operations
  ctx <- createContext
  runGeGLUWithContext ctx input wGate wUp wDown hiddenDim ffnDim

-- | Run GeGLU with given context (for use in larger pipelines)
runGeGLUWithContext :: Context
                    -> Vector Float  -- Input [hidden_dim]
                    -> Vector Float  -- W_gate [ffn_dim * hidden_dim] row-major
                    -> Vector Float  -- W_up [ffn_dim * hidden_dim] row-major
                    -> Vector Float  -- W_down [hidden_dim * ffn_dim] row-major
                    -> Int           -- hidden_dim
                    -> Int           -- ffn_dim
                    -> ContT r IO (Vector Float)  -- Output [hidden_dim]
runGeGLUWithContext ctx input wGate wUp wDown hiddenDim ffnDim = do
  -- Step 1: Gate projection (x @ W_gate^T)
  gate <- linearProjectionWithContext ctx input wGate ffnDim hiddenDim

  -- Step 2: Up projection (x @ W_up^T)
  up <- linearProjectionWithContext ctx input wUp ffnDim hiddenDim

  -- Step 3: GELU(gate)
  geluGate <- runGELUWithContext ctx gate

  -- Step 4: Element-wise multiply
  intermediate <- runElementwiseMultiplyWithContext ctx geluGate up

  -- Step 5: Down projection (intermediate @ W_down^T)
  output <- linearProjectionWithContext ctx intermediate wDown hiddenDim ffnDim

  pure output

-- Helper: Linear projection with given context
linearProjectionWithContext :: Context -> Vector Float -> Vector Float -> Int -> Int -> ContT r IO (Vector Float)
linearProjectionWithContext ctx input weight outSize inSize = do
  -- Validate
  if V.length input /= inSize
    then error $ "LinearProjection: input size mismatch"
    else pure ()
  if V.length weight /= outSize * inSize
    then error $ "LinearProjection: weight size mismatch"
    else pure ()

  let weightShape = Shape [outSize * inSize]
      inputShape = Shape [inSize]
      outputShape = Shape [outSize]

  weightTensor <- createTensorWithData ctx weightShape weight
  inputTensor <- createTensorWithData ctx inputShape input
  outputTensor <- createTensor ctx outputShape F32

  -- Use same shader as Linear layer
  let shaderCode = linearShader outSize inSize False  -- FP32 for now
  code <- createKernelCode shaderCode

  let numWorkgroups = (outSize + 255) `div` 256
  kernel <- createKernel ctx code [weightTensor, inputTensor, outputTensor]
            (WorkgroupSize numWorkgroups 1 1)

  liftIO $ dispatchKernel ctx kernel
  result <- liftIO $ fromGPU ctx outputTensor outSize
  pure result

-- Linear shader (copied from Linear module) - Parameterized FP16/FP32
linearShader :: Int -> Int -> Bool -> String
linearShader outSize inSize useFP16 =
  let floatType = if useFP16 then "f16" else "f32"
      floatLit = if useFP16 then "h" else ""
      enableDirective = if useFP16 then ["enable f16;", ""] else []
  in unlines $
  [ "// Linear layer: matrix-vector multiplication"
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
  , "    var sum: " ++ floatType ++ " = 0.0" ++ floatLit ++ ";"
  , "    "
  , "    for (var i: u32 = 0u; i < IN_SIZE; i = i + 1u) {"
  , "      let weight_idx = row * IN_SIZE + i;"
  , "      sum = sum + weight[weight_idx] * input[i];"
  , "    }"
  , "    "
  , "    output[row] = sum;"
  , "  }"
  , "}"
  ]

-- | GPU-resident GELU activation - keeps tensor on GPU
runGELUGPU :: Context -> Tensor dtype -> Int -> ContT r IO (Tensor dtype)
runGELUGPU ctx inputTensor size = do
  let shape = Shape [size]
  outputTensor <- createTensor ctx shape F32

  let shaderCode = geluShader False size  -- FP32 for now
  code <- createKernelCode shaderCode
  
  let numWorkgroups = (size + 255) `div` 256
  kernel <- createKernel ctx code [inputTensor, outputTensor]
            (WorkgroupSize numWorkgroups 1 1)
  liftIO $ dispatchKernelAsync ctx kernel

  pure outputTensor

-- | GPU-resident elementwise multiply - keeps tensors on GPU
runElementwiseMultiplyGPU :: Context -> Tensor dtype -> Tensor dtype -> Int -> ContT r IO (Tensor dtype)
runElementwiseMultiplyGPU ctx aTensor bTensor size = do
  let shape = Shape [size]
  outputTensor <- createTensor ctx shape F32

  let shaderCode = elementwiseMultiplyShader False size  -- FP32 for now
  code <- createKernelCode shaderCode

  let numWorkgroups = (size + 255) `div` 256
  kernel <- createKernel ctx code [aTensor, bTensor, outputTensor]
            (WorkgroupSize numWorkgroups 1 1)
  liftIO $ dispatchKernelAsync ctx kernel

  pure outputTensor

-- Shader for elementwise multiplication

-- | Fused GELU + ElementwiseMultiply with PRE-COMPILED shader!
--
-- Combines GELU(a) * b into a single kernel, eliminating one dispatch.
-- Takes pre-compiled KernelCode - zero shader compilation overhead!
runGELUMultiplyFusedGPU :: Context -> Tensor dtype -> Tensor dtype -> Tensor dtype -> KernelCode -> Int -> ContT r IO ()
                                             -- gate    up       output↑
runGELUMultiplyFusedGPU ctx aTensor bTensor outputTensor code size = do
  let numWorkgroups = (size + 255) `div` 256
  kernel <- createKernel ctx code [aTensor, bTensor, outputTensor]
            (WorkgroupSize numWorkgroups 1 1)
  liftIO $ dispatchKernelAsync ctx kernel

-- Fused GELU + Multiply shader
geluMultiplyFusedShader :: Bool -> Int -> Bool -> String
geluMultiplyFusedShader useFP16 size useVec4 =
  let floatType = if useFP16 then "f16" else "f32"
      floatLit = if useFP16 then "h" else ""
      enableDirective = if useFP16 then ["enable f16;", ""] else []
      vec4Iters = size `div` 4
      remainder = size `mod` 4
  in unlines $
  [ "// Fused GELU + Multiply: out = GELU(a) * b"
  ] ++ enableDirective ++
  [ "@group(0) @binding(0) var<storage, read_write> a: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(1) var<storage, read_write> b: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(2) var<storage, read_write> output: array<" ++ floatType ++ ">;"
  , ""
  , "const SIZE: u32 = " ++ show size ++ "u;"
  , "const SQRT_2_OVER_PI: " ++ floatType ++ " = 0.7978845608" ++ floatLit ++ ";"
  , "const COEFF: " ++ floatType ++ " = 0.044715" ++ floatLit ++ ";"
  , ""
  , "@compute @workgroup_size(256)"
  , "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {"
  , "  let idx = gid.x;"
  ] ++ (if useVec4 && vec4Iters > 0 then
      [ "  // Vectorized loop: process 4 elements at a time"
      , "  if (idx < " ++ show vec4Iters ++ "u) {"
      , "    let base_idx = idx * 4u;"
      , "    let x_vec = vec4<" ++ floatType ++ ">(a[base_idx], a[base_idx + 1u], a[base_idx + 2u], a[base_idx + 3u]);"
      , "    let b_vec = vec4<" ++ floatType ++ ">(b[base_idx], b[base_idx + 1u], b[base_idx + 2u], b[base_idx + 3u]);"
      , "    "
      , "    // GELU on vec4"
      , "    let x_cubed = x_vec * x_vec * x_vec;"
      , "    let tanh_arg = SQRT_2_OVER_PI * (x_vec + COEFF * x_cubed);"
      , "    let tanh_arg_clamped = clamp(tanh_arg, vec4<" ++ floatType ++ ">(-10.0" ++ floatLit ++ "), vec4<" ++ floatType ++ ">(10.0" ++ floatLit ++ "));"
      , "    let gelu_vec = x_vec * 0.5" ++ floatLit ++ " * (vec4<" ++ floatType ++ ">(1.0" ++ floatLit ++ ") + tanh(tanh_arg_clamped));"
      , "    "
      , "    // Multiply by b_vec"
      , "    let result = gelu_vec * b_vec;"
      , "    output[base_idx] = result.x;"
      , "    output[base_idx + 1u] = result.y;"
      , "    output[base_idx + 2u] = result.z;"
      , "    output[base_idx + 3u] = result.w;"
      , "  }"
      ] else []) ++
  (if useVec4 && remainder > 0 then
      [ "  // Handle remainder elements (scalar)"
      , "  if (idx >= " ++ show vec4Iters ++ "u && idx < SIZE) {"
      , "    let x = a[idx];"
      , "    let x_cubed = x * x * x;"
      , "    let tanh_arg = SQRT_2_OVER_PI * (x + COEFF * x_cubed);"
      , "    let tanh_arg_clamped = clamp(tanh_arg, -10.0" ++ floatLit ++ ", 10.0" ++ floatLit ++ ");"
      , "    let gelu_val = x * 0.5" ++ floatLit ++ " * (1.0" ++ floatLit ++ " + tanh(tanh_arg_clamped));"
      , "    output[idx] = gelu_val * b[idx];"
      , "  }"
      ] else if not useVec4 then
      [ "  if (idx < SIZE) {"
      , "    let x = a[idx];"
      , "    let x_cubed = x * x * x;"
      , "    let tanh_arg = SQRT_2_OVER_PI * (x + COEFF * x_cubed);"
      , "    let tanh_arg_clamped = clamp(tanh_arg, -10.0" ++ floatLit ++ ", 10.0" ++ floatLit ++ ");"
      , "    let gelu_val = x * 0.5" ++ floatLit ++ " * (1.0" ++ floatLit ++ " + tanh(tanh_arg_clamped));"
      , "    output[idx] = gelu_val * b[idx];"
      , "  }"
      ] else []) ++
  [ "}"
  ]

-- | Triple-fused: RMSNorm + Gate + Up projections
--
-- This is the key optimization for GeGLU!
-- Fuses: normalized = RMSNorm(input), gate = Linear_gate(normalized), up = Linear_up(normalized)
-- Returns: (gate_tensor, up_tensor)
-- 
-- Reduces 3 dispatches to 1!
runRMSNormGateUpFusedGPU :: Context
                         -> Tensor dtype  -- Input [hiddenDim]
                         -> Vector Float    -- RMSNorm weights [hiddenDim]
                         -> Vector Float    -- Gate weights [ffnDim, hiddenDim]
                         -> Vector Float    -- Up weights [ffnDim, hiddenDim]
                         -> Int             -- hiddenDim
                         -> Int             -- ffnDim
                         -> Bool            -- zeroCentered
                         -> ContT r IO (Tensor dtype, Tensor dtype)  -- (gate, up)
runRMSNormGateUpFusedGPU ctx inputTensor normWeights gateWeights upWeights hiddenDim ffnDim zeroCentered = do
  let normShape = Shape [hiddenDim]
      weightShape = Shape [ffnDim, hiddenDim]
      outShape = Shape [ffnDim]
  
  normTensor <- createTensorWithData ctx normShape normWeights
  gateTensor <- createTensorWithData ctx weightShape gateWeights
  upTensor <- createTensorWithData ctx weightShape upWeights
  gateOut <- createTensor ctx outShape F32
  upOut <- createTensor ctx outShape F32

  let shaderCode = rmsNormGateUpFusedShader False hiddenDim ffnDim zeroCentered False  -- FP32, no vec4 for now
  code <- createKernelCode shaderCode

  let numWorkgroups = (ffnDim + 255) `div` 256
  kernel <- createKernel ctx code [inputTensor, normTensor, gateTensor, upTensor, gateOut, upOut]
            (WorkgroupSize numWorkgroups 1 1)
  liftIO $ dispatchKernelAsync ctx kernel

  pure (gateOut, upOut)

-- | Triple-fused with PRE-UPLOADED weights AND PRE-COMPILED shader!
--
-- Takes GPU tensors for weights AND pre-compiled KernelCode.
-- Zero uploads, zero shader compilations per token - maximum performance!
runRMSNormGateUpFusedPreloadedGPU :: Context
                                  -> Tensor dtype  -- Input [hiddenDim]
                                  -> Tensor dtype  -- RMSNorm weights on GPU [hiddenDim]
                                  -> Tensor dtype  -- Gate weights on GPU [ffnDim, hiddenDim]
                                  -> Tensor dtype  -- Up weights on GPU [ffnDim, hiddenDim]
                                  -> Tensor dtype  -- Gate output buffer (pre-allocated, REUSED!)
                                  -> Tensor dtype  -- Up output buffer (pre-allocated, REUSED!)
                                  -> KernelCode      -- PRE-COMPILED shader!
                                  -> Int             -- ffnDim
                                  -> ContT r IO ()
runRMSNormGateUpFusedPreloadedGPU ctx inputTensor normTensor gateTensor upTensor gateOut upOut code ffnDim = do
  let numWorkgroups = (ffnDim + 255) `div` 256
  kernel <- createKernel ctx code [inputTensor, normTensor, gateTensor, upTensor, gateOut, upOut]
            (WorkgroupSize numWorkgroups 1 1)
  liftIO $ dispatchKernelAsync ctx kernel

-- Shader for triple fusion
rmsNormGateUpFusedShader :: Bool -> Int -> Int -> Bool -> Bool -> String
rmsNormGateUpFusedShader useFP16 hiddenDim ffnDim zeroCentered useVec4 =
  let floatType = if useFP16 then "f16" else "f32"
      floatLit = if useFP16 then "h" else ""
      enableDirective = if useFP16 then ["enable f16;", ""] else []
      vec4Iters = hiddenDim `div` 4
      remainder = hiddenDim `mod` 4
  in unlines $
  [ "// Triple-fused: RMSNorm + Gate + Up"
  ] ++ enableDirective ++
  [ "@group(0) @binding(0) var<storage, read_write> input: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(1) var<storage, read_write> norm_weight: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(2) var<storage, read_write> gate_weight: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(3) var<storage, read_write> up_weight: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(4) var<storage, read_write> gate_out: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(5) var<storage, read_write> up_out: array<" ++ floatType ++ ">;"
  , ""
  , "const HIDDEN_DIM: u32 = " ++ show hiddenDim ++ "u;"
  , "const FFN_DIM: u32 = " ++ show ffnDim ++ "u;"
  , "const EPSILON: " ++ floatType ++ " = 1e-6" ++ floatLit ++ ";"
  , ""
  , "@compute @workgroup_size(256)"
  , "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {"
  , "  let out_idx = gid.x;"
  , "  if (out_idx >= FFN_DIM) { return; }"
  , ""
  , "  // Step 1: Compute RMS once"
  , "  var sum_sq: " ++ floatType ++ " = 0.0" ++ floatLit ++ ";"
  ] ++ (if useVec4 && vec4Iters > 0 then
      [ "  // Vectorized RMS computation"
      , "  for (var i: u32 = 0u; i < " ++ show vec4Iters ++ "u; i++) {"
      , "    let idx = i * 4u;"
      , "    let vals = vec4<" ++ floatType ++ ">("
      , "      input[idx], input[idx + 1u], input[idx + 2u], input[idx + 3u]"
      , "    );"
      , "    let sq = vals * vals;"
      , "    sum_sq += sq.x + sq.y + sq.z + sq.w;"
      , "  }"
      ] else []) ++
  (if useVec4 && remainder > 0 then
      [ "  // RMS remainder"
      , "  for (var i: u32 = " ++ show (vec4Iters * 4) ++ "u; i < HIDDEN_DIM; i++) {"
      , "    let val = input[i];"
      , "    sum_sq += val * val;"
      , "  }"
      ] else if not useVec4 then
      [ "  for (var i: u32 = 0u; i < HIDDEN_DIM; i++) {"
      , "    let val = input[i];"
      , "    sum_sq += val * val;"
      , "  }"
      ] else []) ++
  [ "  let rms = sqrt(sum_sq / " ++ floatType ++ "(HIDDEN_DIM) + EPSILON);"
  , ""
  , "  // Step 2: Compute both Gate and Up projections from normalized input"
  , "  var gate_sum: " ++ floatType ++ " = 0.0" ++ floatLit ++ ";"
  , "  var up_sum: " ++ floatType ++ " = 0.0" ++ floatLit ++ ";"
  ] ++ (if useVec4 && vec4Iters > 0 then
      [ "  // Vectorized Gate and Up projections"
      , "  for (var i: u32 = 0u; i < " ++ show vec4Iters ++ "u; i++) {"
      , "    let idx = i * 4u;"
      , "    let w_base = out_idx * HIDDEN_DIM + idx;"
      , "    "
      , "    // Load input vector"
      , "    let in_vec = vec4<" ++ floatType ++ ">("
      , "      input[idx], input[idx + 1u], input[idx + 2u], input[idx + 3u]"
      , "    );"
      , "    "
      , "    // Load norm weights"
      , "    let norm_vec = vec4<" ++ floatType ++ ">("
      , "      norm_weight[idx], norm_weight[idx + 1u], norm_weight[idx + 2u], norm_weight[idx + 3u]"
      , "    );"
      , "    "
      , if zeroCentered
         then "    // Normalize with zero-centered weights\n" ++
              "    let normalized = (in_vec / rms) * (vec4<" ++ floatType ++ ">(1.0" ++ floatLit ++ ") + norm_vec);"
         else "    // Normalize\n" ++
              "    let normalized = (in_vec / rms) * norm_vec;"
      , "    "
      , "    // Load gate and up weights"
      , "    let gate_vec = vec4<" ++ floatType ++ ">("
      , "      gate_weight[w_base], gate_weight[w_base + 1u], gate_weight[w_base + 2u], gate_weight[w_base + 3u]"
      , "    );"
      , "    let up_vec = vec4<" ++ floatType ++ ">("
      , "      up_weight[w_base], up_weight[w_base + 1u], up_weight[w_base + 2u], up_weight[w_base + 3u]"
      , "    );"
      , "    "
      , "    // Compute products"
      , "    let gate_prod = normalized * gate_vec;"
      , "    let up_prod = normalized * up_vec;"
      , "    "
      , "    gate_sum += gate_prod.x + gate_prod.y + gate_prod.z + gate_prod.w;"
      , "    up_sum += up_prod.x + up_prod.y + up_prod.z + up_prod.w;"
      , "  }"
      ] else []) ++
  (if useVec4 && remainder > 0 then
      [ "  // Gate/Up remainder"
      , "  for (var i: u32 = " ++ show (vec4Iters * 4) ++ "u; i < HIDDEN_DIM; i++) {"
      , if zeroCentered
         then "    let normalized = (input[i] / rms) * (1.0" ++ floatLit ++ " + norm_weight[i]);"
         else "    let normalized = (input[i] / rms) * norm_weight[i];"
      , "    gate_sum += normalized * gate_weight[out_idx * HIDDEN_DIM + i];"
      , "    up_sum += normalized * up_weight[out_idx * HIDDEN_DIM + i];"
      , "  }"
      ] else if not useVec4 then
      [ "  for (var i: u32 = 0u; i < HIDDEN_DIM; i++) {"
      , if zeroCentered
         then "    let normalized = (input[i] / rms) * (1.0" ++ floatLit ++ " + norm_weight[i]);"
         else "    let normalized = (input[i] / rms) * norm_weight[i];"
      , "    gate_sum += normalized * gate_weight[out_idx * HIDDEN_DIM + i];"
      , "    up_sum += normalized * up_weight[out_idx * HIDDEN_DIM + i];"
      , "  }"
      ] else []) ++
  [ "  gate_out[out_idx] = gate_sum;"
  , "  up_out[out_idx] = up_sum;"
  , "}"
  ]

-- | GPU-resident residual addition (elementwise add)
--
-- Computes: output = a + b (used for residual connections)
-- | Residual addition with pre-allocated output buffer
runResidualAddGPU :: Context
                  -> Tensor dtype  -- Input A (GPU)
                  -> Tensor dtype  -- Input B (GPU)
                  -> Tensor dtype  -- Output buffer (pre-allocated, REUSED!)
                  -> KernelCode    -- Pre-compiled shader
                  -> Int           -- size
                  -> ContT r IO ()
runResidualAddGPU ctx aTensor bTensor outputTensor code size = do
  -- Dispatch kernel with pre-allocated output buffer
  let numWorkgroups = (size + 255) `div` 256
  kernel <- createKernel ctx code [aTensor, bTensor, outputTensor]
            (WorkgroupSize numWorkgroups 1 1)
  liftIO $ dispatchKernelAsync ctx kernel

-- | WGSL shader for residual addition
residualAddShader :: Bool -> Int -> Bool -> String
residualAddShader useFP16 size useVec4 =
  let floatType = if useFP16 then "f16" else "f32"
      enableDirective = if useFP16 then ["enable f16;", ""] else []
      vec4Iters = size `div` 4
      remainder = size `mod` 4
  in unlines $
  [ "// Residual Add: output = a + b"
  ] ++ enableDirective ++
  [ "@group(0) @binding(0) var<storage, read_write> a: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(1) var<storage, read_write> b: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(2) var<storage, read_write> output: array<" ++ floatType ++ ">;"
  , ""
  , "const SIZE: u32 = " ++ show size ++ "u;"
  , ""
  , "@compute @workgroup_size(256)"
  , "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {"
  , "  let idx = gid.x;"
  ] ++ (if useVec4 && vec4Iters > 0 then
      [ "  // Vectorized loop: process 4 elements at a time"
      , "  if (idx < " ++ show vec4Iters ++ "u) {"
      , "    let base_idx = idx * 4u;"
      , "    let a_vec = vec4<" ++ floatType ++ ">(a[base_idx], a[base_idx + 1u], a[base_idx + 2u], a[base_idx + 3u]);"
      , "    let b_vec = vec4<" ++ floatType ++ ">(b[base_idx], b[base_idx + 1u], b[base_idx + 2u], b[base_idx + 3u]);"
      , "    let result = a_vec + b_vec;"
      , "    output[base_idx] = result.x;"
      , "    output[base_idx + 1u] = result.y;"
      , "    output[base_idx + 2u] = result.z;"
      , "    output[base_idx + 3u] = result.w;"
      , "  }"
      ] else []) ++
  (if useVec4 && remainder > 0 then
      [ "  // Handle remainder elements (scalar)"
      , "  if (idx >= " ++ show vec4Iters ++ "u && idx < SIZE) {"
      , "    output[idx] = a[idx] + b[idx];"
      , "  }"
      ] else if not useVec4 then
      [ "  if (idx < SIZE) {"
      , "    output[idx] = a[idx] + b[idx];"
      , "  }"
      ] else []) ++
  [ "}"
  ]

-- ═══════════════════════════════════════════════════════════════════════
-- Phase 3.1: FFN Output Fusion (LinearDown + Residual + RMSNorm)
-- ═══════════════════════════════════════════════════════════════════════

-- | MEGA-FUSION: LinearDown + Residual + RMSNorm
--
-- This shader combines 3 separate operations:
-- 1. Linear Down projection (FFN intermediate → hidden dim)
-- 2. Residual Add (with pre-FFN residual)
-- 3. Post-FFN RMSNorm (optional, Gemma 3 only)
--
-- Benefits:
-- - Eliminates 2 kernel dispatches (~60-100μs saved per layer)
-- - Eliminates 2 intermediate tensor transfers
-- - Improved cache locality
-- - Total savings: 2 dispatches × 26 layers = 52 dispatches per forward pass
--
-- Input:
--   - input: FFN activation output [ffnDim]
--   - down_weight: Down projection weights [hiddenDim, ffnDim]
--   - residual: Pre-FFN residual [hiddenDim]
--   - norm_weight: RMSNorm weights [hiddenDim]
--
-- Output:
--   - output: Normalized residual [hiddenDim]
ffnOutputFusedShader :: Bool    -- useFP16
                     -> Bool    -- useVec4
                     -> Int     -- hiddenDim
                     -> Int     -- ffnDim
                     -> Bool    -- zeroCentered (RMSNorm weights)
                     -> String
ffnOutputFusedShader useFP16 useVec4 hiddenDim ffnDim zeroCentered =
  let floatType = if useFP16 then "f16" else "f32"
      floatLit = if useFP16 then "h" else ""
      enableDirective = if useFP16 then ["enable f16;", ""] else []
      vec4Iters = ffnDim `div` 4
      remainder = ffnDim `mod` 4
  in unlines $
  [ "// MEGA-FUSION: LinearDown + Residual + RMSNorm"
  , "// Phase 3.1: FFN Output Fusion"
  , "//"
  , "// Combines 3 operations into 1 kernel dispatch:"
  , "//   1. Linear Down projection"
  , "//   2. Residual Add"
  , "//   3. Post-FFN RMSNorm"
  , "//"
  , "// Savings: 3 dispatches → 1 dispatch = 2 dispatches saved per layer"
  , "//          2 × 26 layers = 52 dispatches saved per forward pass"
  ] ++ enableDirective ++
  [ ""
  , "// Inputs"
  , "@group(0) @binding(0) var<storage, read_write> input: array<" ++ floatType ++ ">;        // [ffnDim]"
  , "@group(0) @binding(1) var<storage, read_write> down_weight: array<" ++ floatType ++ ">;  // [hiddenDim, ffnDim]"
  , "@group(0) @binding(2) var<storage, read_write> residual: array<" ++ floatType ++ ">;     // [hiddenDim]"
  , "@group(0) @binding(3) var<storage, read_write> norm_weight: array<" ++ floatType ++ ">;  // [hiddenDim]"
  , "@group(0) @binding(4) var<storage, read_write> output: array<" ++ floatType ++ ">;       // [hiddenDim]"
  , ""
  , "// Constants"
  , "const HIDDEN_DIM: u32 = " ++ show hiddenDim ++ "u;"
  , "const FFN_DIM: u32 = " ++ show ffnDim ++ "u;"
  , "const EPSILON: " ++ floatType ++ " = 1e-6" ++ floatLit ++ ";"
  , "const WORKGROUP_SIZE: u32 = 256u;"
  , ""
  , "// Shared memory for RMS computation"
  , "var<workgroup> shared_rms: array<" ++ floatType ++ ", WORKGROUP_SIZE>;"
  , "var<workgroup> rms_value: " ++ floatType ++ ";"
  , ""
  , "@compute @workgroup_size(256)"
  , "fn main(@builtin(local_invocation_id) lid: vec3<u32>,"
  , "        @builtin(global_invocation_id) gid: vec3<u32>) {"
  , ""
  , "  let tid = lid.x;"
  , "  let output_idx = gid.x;"
  , ""
  , "  // ═══════════════════════════════════════════════════════════"
  , "  // STEP 1: Linear Down Projection (each thread computes one output)"
  , "  // ═══════════════════════════════════════════════════════════"
  , "  var down_sum: " ++ floatType ++ " = 0.0" ++ floatLit ++ ";"
  , ""
  , "  if (output_idx < HIDDEN_DIM) {"
  ] ++
  (if useVec4 && vec4Iters > 0 then
      [ "    // Vectorized Down projection"
      , "    for (var i: u32 = 0u; i < " ++ show vec4Iters ++ "u; i++) {"
      , "      let idx = i * 4u;"
      , "      let w_base = output_idx * FFN_DIM + idx;"
      , "      let in_vec = vec4<" ++ floatType ++ ">(input[idx], input[idx+1u], input[idx+2u], input[idx+3u]);"
      , "      let w_vec = vec4<" ++ floatType ++ ">(down_weight[w_base], down_weight[w_base+1u], down_weight[w_base+2u], down_weight[w_base+3u]);"
      , "      let prod = in_vec * w_vec;"
      , "      down_sum += prod.x + prod.y + prod.z + prod.w;"
      , "    }"
      ]
    else []) ++
  (if useVec4 && remainder > 0 || not useVec4 then
      [ "    // Scalar Down projection (remainder or full scalar)"
      , "    let start_idx = " ++ (if useVec4 then show (vec4Iters * 4) ++ "u" else "0u") ++ ";"
      , "    for (var i: u32 = start_idx; i < FFN_DIM; i++) {"
      , "      let w_idx = output_idx * FFN_DIM + i;"
      , "      down_sum += input[i] * down_weight[w_idx];"
      , "    }"
      ]
    else []) ++
  [ "  }"
  , ""
  , "  // ═══════════════════════════════════════════════════════════"
  , "  // STEP 2: Residual Add"
  , "  // ═══════════════════════════════════════════════════════════"
  , "  var with_residual: " ++ floatType ++ " = 0.0" ++ floatLit ++ ";"
  , "  if (output_idx < HIDDEN_DIM) {"
  , "    with_residual = down_sum + residual[output_idx];"
  , "  }"
  , ""
  , "  // ═══════════════════════════════════════════════════════════"
  , "  // STEP 3: RMSNorm (all threads cooperate)"
  , "  // ═══════════════════════════════════════════════════════════"
  , ""
  , "  // Step 3a: Compute sum of squares (parallel reduction)"
  , "  var sum_sq: " ++ floatType ++ " = 0.0" ++ floatLit ++ ";"
  , "  if (output_idx < HIDDEN_DIM) {"
  , "    sum_sq = with_residual * with_residual;"
  , "  }"
  , ""
  , "  shared_rms[tid] = sum_sq;"
  , "  workgroupBarrier();"
  , ""
  , "  // Parallel reduction"
  , "  for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {"
  , "    if (tid < stride) {"
  , "      shared_rms[tid] = shared_rms[tid] + shared_rms[tid + stride];"
  , "    }"
  , "    workgroupBarrier();"
  , "  }"
  , ""
  , "  // Thread 0 computes and broadcasts RMS"
  , "  if (tid == 0u) {"
  , "    let mean_sq = shared_rms[0] / " ++ floatType ++ "(HIDDEN_DIM);"
  , "    rms_value = sqrt(mean_sq + EPSILON);"
  , "  }"
  , "  workgroupBarrier();"
  , "  let rms = rms_value;"
  , ""
  , "  // Step 3b: Apply RMSNorm"
  , "  if (output_idx < HIDDEN_DIM) {"
  , if zeroCentered
     then "    output[output_idx] = (with_residual / rms) * (1.0" ++ floatLit ++ " + norm_weight[output_idx]);"
     else "    output[output_idx] = (with_residual / rms) * norm_weight[output_idx];"
  , "  }"
  , "}"
  ]

-- | Run FFN output fusion with pre-uploaded weights (GPU-resident)
--
-- Replaces 3 separate dispatches:
--   1. runLinearPreloadedGPU (down projection)
--   2. runResidualAddGPU
--   3. runRMSNormPreloadedGPU (post-FFN norm)
--
-- With a single fused dispatch, saving ~60-100μs per layer.
runFFNOutputFusedPreloadedGPU :: Context
                              -> Tensor dtype  -- Input FFN activation [ffnDim]
                              -> Tensor dtype  -- Down weight tensor [hiddenDim, ffnDim]
                              -> Tensor dtype  -- Residual tensor [hiddenDim]
                              -> Tensor dtype  -- Norm weight tensor [hiddenDim]
                              -> Tensor dtype  -- Output buffer (pre-allocated, REUSED!)
                              -> KernelCode  -- Pre-compiled fused shader
                              -> Int         -- hiddenDim
                              -> ContT r IO ()
runFFNOutputFusedPreloadedGPU ctx inputTensor downWeightTensor residualTensor normWeightTensor outputTensor code hiddenDim = do
  -- Dispatch fused kernel
  -- All inputs are GPU-resident, output buffer is reused
  -- WebGPU caches bind groups automatically for stable tensor pointers
  let numWorkgroups = (hiddenDim + 255) `div` 256
  kernel <- createKernel ctx code
            [inputTensor, downWeightTensor, residualTensor, normWeightTensor, outputTensor]
            (WorkgroupSize numWorkgroups 1 1)
  liftIO $ dispatchKernelAsync ctx kernel
