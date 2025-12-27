{-# LANGUAGE OverloadedStrings #-}

{-|
Module: Gemma.Layers.AttentionGPU
Description: GPU-resident attention with KV-cache for Gemma 3

Fully GPU-resident attention implementation:
- All operations stay on GPU
- Pre-uploaded weights
- Pre-compiled shaders
- No CPU transfers
-}

module Gemma.Layers.AttentionGPU
  ( runRoPEGPU
  , ropeShader
  , runQKVProjectionsGPU
  , qkvProjectionShader
  , runQKNormGPU
  , qkNormShader
  , runAttentionScoresGPU
  , attentionScoresShader
  , runAttentionOutputGPU
  , attentionOutputShader
  , runOutputProjectionGPU
  , outputProjectionShader
  , runAppendKVCacheGPU
  , appendKVCacheShader
  , attentionPostFusedShader
  , runAttentionPostFusedPreloadedGPU
  , attentionCoreFusedShader
  , runAttentionCoreFusedPreloadedGPU
  ) where

import Graphics.WebGPU.Dawn.ContT
import Graphics.WebGPU.Dawn.Types (KernelCode, Tensor, Context, Shape(..), NumType(..))
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import System.Environment (lookupEnv)

-- | GPU-resident RoPE (Rotary Position Embedding) with pre-allocated output
--
-- Applies rotary position encoding to Q or K tensors
runRoPEGPU :: Context
           -> Tensor dtype  -- Input Q or K [size]
           -> Tensor dtype  -- Output buffer (pre-allocated, REUSED!)
           -> KernelCode    -- Pre-compiled RoPE shader
           -> Int           -- position
           -> Float         -- ropeBase
           -> Int           -- headDim
           -> Int           -- size (numHeads * headDim)
           -> ContT r IO ()
runRoPEGPU ctx inputTensor outputTensor code position ropeBase headDim size = do
  -- Create position tensor (small, unavoidable)
  let posShape = Shape [1]
      posData = V.singleton (fromIntegral position :: Float)
  posTensor <- createTensorWithData ctx posShape posData

  -- Dispatch kernel with pre-allocated output buffer
  -- WebGPU will cache bind group since outputTensor pointer is stable!
  let numHeads = size `div` headDim
      numWorkgroups = (numHeads + 255) `div` 256
  kernel <- createKernel ctx code [inputTensor, posTensor, outputTensor]
            (WorkgroupSize numWorkgroups 1 1)
  liftIO $ dispatchKernelAsync ctx kernel

-- | WGSL shader for RoPE
ropeShader :: Bool -> Int -> Int -> Float -> String
ropeShader useFP16 numHeads headDim ropeBase =
  let floatType = if useFP16 then "f16" else "f32"
      floatLit = if useFP16 then "h" else ""
      enableDirective = if useFP16 then ["enable f16;", ""] else []
  in unlines $
  [ "// Rotary Position Embedding (RoPE)"
  ] ++ enableDirective ++
  [ "@group(0) @binding(0) var<storage, read_write> input: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(1) var<storage, read_write> position: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(2) var<storage, read_write> output: array<" ++ floatType ++ ">;"
  , ""
  , "const NUM_HEADS: u32 = " ++ show numHeads ++ "u;"
  , "const HEAD_DIM: u32 = " ++ show headDim ++ "u;"
  , "const ROPE_BASE: " ++ floatType ++ " = " ++ show ropeBase ++ floatLit ++ ";"
  , "const PI: " ++ floatType ++ " = 3.14159265359" ++ floatLit ++ ";"
  , ""
  , "@compute @workgroup_size(256)"
  , "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {"
  , "  let head_idx = gid.x;"
  , "  if (head_idx >= NUM_HEADS) { return; }"
  , ""
  , "  let pos = position[0];"
  , "  let head_offset = head_idx * HEAD_DIM;"
  , "  let half_dim = HEAD_DIM / 2u;"
  , ""
  , "  // Gemma 3 uses split-half RoPE: pair input[i] with input[i + half_dim]"
  , "  for (var i: u32 = 0u; i < half_dim; i = i + 1u) {"
  , "    let idx = head_offset + i;"
  , "    let idx_plus_half = idx + half_dim;"
  , "    let x = input[idx];"
  , "    let y = input[idx_plus_half];"
  , ""
  , "    // Calculate rotation angle"
  , "    // Note: i goes 0,1,2,... so multiply by 2 to get pair indices 0,2,4,..."
  , "    let freq_exp = " ++ floatType ++ "(i * 2u) / " ++ floatType ++ "(HEAD_DIM);"
  , "    let freq = 1.0" ++ floatLit ++ " / pow(ROPE_BASE, freq_exp);"
  , "    let theta = pos * freq;"
  , ""
  , "    // Apply split-half rotation matrix (matching PyTorch Gemma 3)"
  , "    let cos_theta = cos(theta);"
  , "    let sin_theta = sin(theta);"
  , ""
  , "    output[idx] = x * cos_theta - y * sin_theta;"
  , "    output[idx_plus_half] = x * sin_theta + y * cos_theta;"
  , "  }"
  , "}"
  ]

-- | GPU-resident Q/K/V projections with pre-allocated outputs
--
-- Applies linear projections to hidden states to get Q, K, V
runQKVProjectionsGPU :: Context
                     -> Tensor dtype  -- Input [hiddenDim]
                     -> Tensor dtype  -- Q weights on GPU [qSize, hiddenDim]
                     -> Tensor dtype  -- K weights on GPU [kvSize, hiddenDim]
                     -> Tensor dtype  -- V weights on GPU [kvSize, hiddenDim]
                     -> Tensor dtype  -- Q output buffer (pre-allocated, REUSED!)
                     -> Tensor dtype  -- K output buffer (pre-allocated, REUSED!)
                     -> Tensor dtype  -- V output buffer (pre-allocated, REUSED!)
                     -> KernelCode    -- Pre-compiled shader
                     -> Int           -- hiddenDim
                     -> Int           -- qSize (numHeads * headDim)
                     -> Int           -- kvSize (numKVHeads * headDim)
                     -> ContT r IO ()
runQKVProjectionsGPU ctx inputTensor qWeights kWeights vWeights qTensor kTensor vTensor code hiddenDim qSize kvSize = do
  -- Dispatch kernel with pre-allocated output buffers
  -- WebGPU will cache bind group since all tensor pointers are stable!
  let maxSize = max qSize kvSize
      numWorkgroups = (maxSize + 255) `div` 256
  kernel <- createKernel ctx code [inputTensor, qWeights, kWeights, vWeights, qTensor, kTensor, vTensor]
            (WorkgroupSize numWorkgroups 1 1)
  liftIO $ dispatchKernelAsync ctx kernel

-- | WGSL shader for Q/K/V projections
qkvProjectionShader :: Bool -> Int -> Int -> Int -> String
qkvProjectionShader useFP16 hiddenDim qSize kvSize =
  let floatType = if useFP16 then "f16" else "f32"
      floatLit = if useFP16 then "h" else ""
      enableDirective = if useFP16 then ["enable f16;", ""] else []
  in unlines $
  [ "// Q/K/V Projections (3 matrix multiplications in parallel) - FORCE RECOMPILE v4"
  , "// MIXED PRECISION: FP32 input + " ++ floatType ++ " weights → FP32 output"
  ] ++ enableDirective ++
  [ "@group(0) @binding(0) var<storage, read_write> input: array<f32>;"  -- ALWAYS FP32 input
  , "@group(0) @binding(1) var<storage, read_write> q_weights: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(2) var<storage, read_write> k_weights: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(3) var<storage, read_write> v_weights: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(4) var<storage, read_write> q_out: array<f32>;"  -- ALWAYS FP32 output
  , "@group(0) @binding(5) var<storage, read_write> k_out: array<f32>;"  -- ALWAYS FP32 output
  , "@group(0) @binding(6) var<storage, read_write> v_out: array<f32>;"  -- ALWAYS FP32 output
  , ""
  , "const HIDDEN_DIM: u32 = " ++ show hiddenDim ++ "u;"
  , "const Q_SIZE: u32 = " ++ show qSize ++ "u;"
  , "const KV_SIZE: u32 = " ++ show kvSize ++ "u;"
  , ""
  , "@compute @workgroup_size(256)"
  , "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {"
  , "  let idx = gid.x;"
  , ""
  , "  // Compute Q projection (FP32 accumulation for accuracy)"
  , "  if (idx < Q_SIZE) {"
  , "    var sum: f32 = 0.0;"
  , "    for (var i: u32 = 0u; i < HIDDEN_DIM; i = i + 1u) {"
  , "      sum = sum + f32(q_weights[idx * HIDDEN_DIM + i]) * input[i];"
  , "    }"
  , "    q_out[idx] = sum;"
  , "  }"
  , ""
  , "  // Compute K projection (FP32 accumulation for accuracy)"
  , "  if (idx < KV_SIZE) {"
  , "    var sum: f32 = 0.0;"
  , "    for (var i: u32 = 0u; i < HIDDEN_DIM; i = i + 1u) {"
  , "      sum = sum + f32(k_weights[idx * HIDDEN_DIM + i]) * input[i];"
  , "    }"
  , "    k_out[idx] = sum;"
  , "  }"
  , ""
  , "  // Compute V projection (FP32 accumulation for accuracy)"
  , "  if (idx < KV_SIZE) {"
  , "    var sum: f32 = 0.0;"
  , "    for (var i: u32 = 0u; i < HIDDEN_DIM; i = i + 1u) {"
  , "      sum = sum + f32(v_weights[idx * HIDDEN_DIM + i]) * input[i];"
  , "    }"
  , "    v_out[idx] = sum;"
  , "  }"
  , "}"
  ]

-- | GPU-resident QK-Norm (Gemma 3 specific)
--
-- Applies RMSNorm to Q and K separately
-- | QK-Norm with pre-allocated output buffer
runQKNormGPU :: Context
             -> Tensor dtype  -- Q or K input [size]
             -> Tensor dtype  -- Norm weights on GPU [headDim]
             -> Tensor dtype  -- Output buffer (pre-allocated, REUSED!)
             -> KernelCode    -- Pre-compiled shader
             -> Int           -- numHeads
             -> Int           -- headDim
             -> ContT r IO ()
runQKNormGPU ctx inputTensor normWeights outputTensor code numHeads headDim = do
  -- Dispatch kernel with pre-allocated output buffer
  let numWorkgroups = (numHeads + 255) `div` 256
  kernel <- createKernel ctx code [inputTensor, normWeights, outputTensor]
            (WorkgroupSize numWorkgroups 1 1)
  liftIO $ dispatchKernelAsync ctx kernel

-- | WGSL shader for QK-Norm
-- zeroCentered = True uses (1 + weight) for Gemma 3 compatibility
qkNormShader :: Bool -> Int -> Int -> Bool -> String
qkNormShader useFP16 numHeads headDim zeroCentered =
  let floatType = if useFP16 then "f16" else "f32"
      floatLit = if useFP16 then "h" else ""
      enableDirective = if useFP16 then ["enable f16;", ""] else []
  in unlines $
  [ "// QK-Norm: RMSNorm applied per head"
  ] ++ enableDirective ++
  [ "@group(0) @binding(0) var<storage, read_write> input: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(1) var<storage, read_write> weights: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(2) var<storage, read_write> output: array<" ++ floatType ++ ">;"
  , ""
  , "const NUM_HEADS: u32 = " ++ show numHeads ++ "u;"
  , "const HEAD_DIM: u32 = " ++ show headDim ++ "u;"
  , "const EPSILON: " ++ floatType ++ " = 1e-6" ++ floatLit ++ ";"
  , ""
  , "@compute @workgroup_size(256)"
  , "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {"
  , "  let head_idx = gid.x;"
  , "  if (head_idx >= NUM_HEADS) { return; }"
  , ""
  , "  let head_offset = head_idx * HEAD_DIM;"
  , ""
  , "  // Compute RMS for this head"
  , "  var sum_sq: " ++ floatType ++ " = 0.0" ++ floatLit ++ ";"
  , "  for (var i: u32 = 0u; i < HEAD_DIM; i = i + 1u) {"
  , "    let val = input[head_offset + i];"
  , "    sum_sq = sum_sq + val * val;"
  , "  }"
  , "  let rms = sqrt(sum_sq / " ++ floatType ++ "(HEAD_DIM) + EPSILON);"
  , ""
  , "  // Normalize and apply weights"
  , "  for (var i: u32 = 0u; i < HEAD_DIM; i = i + 1u) {"
  , "    let idx = head_offset + i;"
  , if zeroCentered
     then "    output[idx] = (input[idx] / rms) * (1.0" ++ floatLit ++ " + weights[i]);  // Zero-centered (Gemma 3)"
     else "    output[idx] = (input[idx] / rms) * weights[i];                            // Standard"
  , "  }"
  , "}"
  ]

-- | GPU-resident attention scores (Q @ K^T / sqrt(d) + softmax)
--
-- Computes attention scores for single query against cached K
-- | Attention scores with pre-allocated output buffer
runAttentionScoresGPU :: Context
                      -> Tensor dtype  -- Q [numHeads, headDim]
                      -> Tensor dtype  -- Cached K [cacheLen, numKVHeads, headDim]
                      -> Tensor dtype  -- Scores output buffer (pre-allocated, REUSED!)
                      -> KernelCode    -- Pre-compiled shader
                      -> Int           -- numHeads
                      -> Int           -- numKVHeads
                      -> Int           -- headDim
                      -> Int           -- cacheLen
                      -> Maybe Int     -- windowSize
                      -> Int           -- maxCacheLen (for pre-compiled shader)
                      -> ContT r IO ()
runAttentionScoresGPU ctx qTensor kCacheTensor scoresTensor code numHeads numKVHeads headDim cacheLen windowSize maxCacheLen = do
  -- BUG FIX: Use cacheLen directly (it's already effectiveLen = min(cacheLen', windowSize))
  -- The caller (TransformerBlock) already computed the window, don't re-apply it!
  let windowVal = fromIntegral cacheLen :: Float
      windowShape = Shape [1]
      windowData = V.singleton windowVal
  -- DEBUG: Print window value (ALWAYS for debugging Bug #2 fix)
  liftIO $ putStrLn $ "  🐛 BUG#2 FIX ACTIVE: windowVal=" ++ show windowVal ++ " (was using windowSize=" ++ show windowSize ++ " before fix)"
  windowTensor <- createTensorWithData ctx windowShape windowData

  -- Dispatch kernel with pre-allocated scores buffer
  let numWorkgroups = (numHeads + 255) `div` 256
  kernel <- createKernel ctx code [qTensor, kCacheTensor, windowTensor, scoresTensor]
            (WorkgroupSize numWorkgroups 1 1)
  liftIO $ dispatchKernelAsync ctx kernel

-- | WGSL shader for attention scores
attentionScoresShader :: Bool -> Int -> Int -> Int -> Int -> String
attentionScoresShader useFP16 numHeads numKVHeads headDim maxCacheLen =
  let floatType = if useFP16 then "f16" else "f32"
      floatLit = if useFP16 then "h" else ""
      enableDirective = if useFP16 then ["enable f16;", ""] else []
  in unlines $
  [ "// Attention Scores: Q @ K^T / sqrt(d) + softmax"
  ] ++ enableDirective ++
  [ "@group(0) @binding(0) var<storage, read_write> q: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(1) var<storage, read_write> k_cache: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(2) var<storage, read_write> window_size: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(3) var<storage, read_write> scores: array<" ++ floatType ++ ">;"
  , ""
  , "const NUM_HEADS: u32 = " ++ show numHeads ++ "u;"
  , "const NUM_KV_HEADS: u32 = " ++ show numKVHeads ++ "u;"
  , "const HEAD_DIM: u32 = " ++ show headDim ++ "u;"
  , "const MAX_CACHE_LEN: u32 = " ++ show maxCacheLen ++ "u;"
  , "const SCALE: " ++ floatType ++ " = " ++ show (1.0 / sqrt (fromIntegral headDim :: Float)) ++ floatLit ++ ";"
  , ""
  , "@compute @workgroup_size(256)"
  , "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {"
  , "  let head_idx = gid.x;"
  , "  if (head_idx >= NUM_HEADS) { return; }"
  , ""
  , "  // GQA: Map query head to KV head"
  , "  let kv_head_idx = head_idx / (NUM_HEADS / NUM_KV_HEADS);"
  , "  let q_offset = head_idx * HEAD_DIM;"
  , ""
  , "  let window = u32(window_size[0]);"
  , ""
  , "  // Compute Q @ K^T for all cached positions"
  , "  var max_score: " ++ floatType ++ " = -1e9" ++ floatLit ++ ";"
  , "  for (var pos: u32 = 0u; pos < window; pos = pos + 1u) {"
  , "    let k_offset = pos * NUM_KV_HEADS * HEAD_DIM + kv_head_idx * HEAD_DIM;"
  , "    "
  , "    var dot: " ++ floatType ++ " = 0.0" ++ floatLit ++ ";"
  , "    for (var i: u32 = 0u; i < HEAD_DIM; i = i + 1u) {"
  , "      dot = dot + q[q_offset + i] * k_cache[k_offset + i];"
  , "    }"
  , "    "
  , "    let score = dot * SCALE;"
  , "    scores[head_idx * MAX_CACHE_LEN + pos] = score;"
  , "    max_score = max(max_score, score);"
  , "  }"
  , ""
  , "  // Softmax: exp(x - max) / sum"
  , "  var sum_exp: " ++ floatType ++ " = 0.0" ++ floatLit ++ ";"
  , "  for (var pos: u32 = 0u; pos < window; pos = pos + 1u) {"
  , "    let idx = head_idx * MAX_CACHE_LEN + pos;"
  , "    let exp_val = exp(scores[idx] - max_score);"
  , "    scores[idx] = exp_val;"
  , "    sum_exp = sum_exp + exp_val;"
  , "  }"
  , "  "
  , "  // Normalize"
  , "  for (var pos: u32 = 0u; pos < window; pos = pos + 1u) {"
  , "    let idx = head_idx * MAX_CACHE_LEN + pos;"
  , "    scores[idx] = scores[idx] / sum_exp;"
  , "  }"
  , "}"
  ]

-- | GPU-resident attention output (scores @ V)
--
-- Computes weighted sum of values using attention scores
-- | Attention output with pre-allocated buffer
runAttentionOutputGPU :: Context
                      -> Tensor dtype  -- Attention scores [numHeads, cacheLen]
                      -> Tensor dtype  -- Cached V [cacheLen, numKVHeads, headDim]
                      -> Tensor dtype  -- Output buffer (pre-allocated, REUSED!)
                      -> KernelCode    -- Pre-compiled shader
                      -> Int           -- numHeads
                      -> Int           -- numKVHeads
                      -> Int           -- headDim
                      -> Int           -- cacheLen (actual window size, NOT max!)
                      -> ContT r IO ()
runAttentionOutputGPU ctx scoresTensor vCacheTensor outputTensor code numHeads numKVHeads headDim cacheLen = do
  -- Create window size tensor (actual window, not max!)
  let windowVal = fromIntegral cacheLen :: Float
      windowShape = Shape [1]
      windowData = V.singleton windowVal
  windowTensor <- createTensorWithData ctx windowShape windowData

  -- Dispatch kernel with pre-allocated output buffer
  let numWorkgroups = (numHeads + 255) `div` 256
  kernel <- createKernel ctx code [scoresTensor, vCacheTensor, windowTensor, outputTensor]
            (WorkgroupSize numWorkgroups 1 1)
  liftIO $ dispatchKernelAsync ctx kernel

-- | WGSL shader for attention output
attentionOutputShader :: Bool -> Int -> Int -> Int -> Int -> String
attentionOutputShader useFP16 numHeads numKVHeads headDim maxCacheLen =
  let floatType = if useFP16 then "f16" else "f32"
      floatLit = if useFP16 then "h" else ""
      enableDirective = if useFP16 then ["enable f16;", ""] else []
  in unlines $
  [ "// Attention Output: scores @ V"
  ] ++ enableDirective ++
  [ "@group(0) @binding(0) var<storage, read_write> scores: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(1) var<storage, read_write> v_cache: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(2) var<storage, read_write> window_size: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(3) var<storage, read_write> output: array<" ++ floatType ++ ">;"
  , ""
  , "const NUM_HEADS: u32 = " ++ show numHeads ++ "u;"
  , "const NUM_KV_HEADS: u32 = " ++ show numKVHeads ++ "u;"
  , "const HEAD_DIM: u32 = " ++ show headDim ++ "u;"
  , "const MAX_CACHE_LEN: u32 = " ++ show maxCacheLen ++ "u;"
  , ""
  , "@compute @workgroup_size(256)"
  , "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {"
  , "  let head_idx = gid.x;"
  , "  if (head_idx >= NUM_HEADS) { return; }"
  , ""
  , "  let window = u32(window_size[0]);"
  , ""
  , "  // GQA: Map query head to KV head"
  , "  let kv_head_idx = head_idx / (NUM_HEADS / NUM_KV_HEADS);"
  , "  let out_offset = head_idx * HEAD_DIM;"
  , "  let scores_offset = head_idx * MAX_CACHE_LEN;"
  , ""
  , "  // Compute weighted sum: output = sum(scores[i] * V[i])"
  , "  for (var d: u32 = 0u; d < HEAD_DIM; d = d + 1u) {"
  , "    var sum: " ++ floatType ++ " = 0.0" ++ floatLit ++ ";"
  , "    for (var pos: u32 = 0u; pos < window; pos = pos + 1u) {"
  , "      let score = scores[scores_offset + pos];"
  , "      let v_offset = pos * NUM_KV_HEADS * HEAD_DIM + kv_head_idx * HEAD_DIM;"
  , "      sum = sum + score * v_cache[v_offset + d];"
  , "    }"
  , "    output[out_offset + d] = sum;"
  , "  }"
  , "}"
  ]

-- | GPU-resident output projection
--
-- Projects attention output back to hidden dimension
-- | Output projection with pre-allocated buffer
runOutputProjectionGPU :: Context
                       -> Tensor dtype  -- Attention output [numHeads * headDim]
                       -> Tensor dtype  -- Output weights on GPU [hiddenDim, numHeads * headDim]
                       -> Tensor dtype  -- Output buffer (pre-allocated, REUSED!)
                       -> KernelCode    -- Pre-compiled shader
                       -> Int           -- hiddenDim
                       -> Int           -- inputSize (numHeads * headDim)
                       -> ContT r IO ()
runOutputProjectionGPU ctx inputTensor weightsTensor outputTensor code hiddenDim inputSize = do
  -- Dispatch kernel with pre-allocated output buffer
  let numWorkgroups = (hiddenDim + 255) `div` 256
  kernel <- createKernel ctx code [inputTensor, weightsTensor, outputTensor]
            (WorkgroupSize numWorkgroups 1 1)
  liftIO $ dispatchKernelAsync ctx kernel

-- | WGSL shader for output projection
outputProjectionShader :: Bool -> Int -> Int -> String
outputProjectionShader useFP16 hiddenDim inputSize =
  let floatType = if useFP16 then "f16" else "f32"
      floatLit = if useFP16 then "h" else ""
      enableDirective = if useFP16 then ["enable f16;", ""] else []
  in unlines $
  [ "// Output Projection: Linear layer"
  ] ++ enableDirective ++
  [ "@group(0) @binding(0) var<storage, read_write> input: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(1) var<storage, read_write> weights: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(2) var<storage, read_write> output: array<" ++ floatType ++ ">;"
  , ""
  , "const HIDDEN_DIM: u32 = " ++ show hiddenDim ++ "u;"
  , "const INPUT_SIZE: u32 = " ++ show inputSize ++ "u;"
  , ""
  , "@compute @workgroup_size(256)"
  , "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {"
  , "  let idx = gid.x;"
  , "  if (idx >= HIDDEN_DIM) { return; }"
  , ""
  , "  var sum: " ++ floatType ++ " = 0.0" ++ floatLit ++ ";"
  , "  for (var i: u32 = 0u; i < INPUT_SIZE; i = i + 1u) {"
  , "    sum = sum + weights[idx * INPUT_SIZE + i] * input[i];"
  , "  }"
  , "  output[idx] = sum;"
  , "}"
  ]

-- | GPU-resident KV cache update
--
-- Appends new K/V values to the cache directly on GPU
-- Cache layout: [maxSeqLen, numKVHeads, headDim]
runAppendKVCacheGPU :: Context
                    -> Tensor dtype  -- K cache [maxSeqLen * numKVHeads * headDim]
                    -> Tensor dtype  -- V cache [maxSeqLen * numKVHeads * headDim]
                    -> Tensor dtype  -- New K [numKVHeads * headDim]
                    -> Tensor dtype  -- New V [numKVHeads * headDim]
                    -> KernelCode    -- Pre-compiled shader
                    -> Int           -- Current position (where to append)
                    -> Int           -- numKVHeads
                    -> Int           -- headDim
                    -> ContT r IO ()
runAppendKVCacheGPU ctx kCacheTensor vCacheTensor kNewTensor vNewTensor code position numKVHeads headDim = do
  -- Create position tensor
  let posVal = fromIntegral position :: Float
      posShape = Shape [1]
      posData = V.singleton posVal
  posTensor <- createTensorWithData ctx posShape posData

  -- Dispatch kernel - one thread per KV element
  let kvSize = numKVHeads * headDim
      numWorkgroups = (kvSize + 255) `div` 256
  kernel <- createKernel ctx code [kCacheTensor, vCacheTensor, kNewTensor, vNewTensor, posTensor]
            (WorkgroupSize numWorkgroups 1 1)
  liftIO $ dispatchKernelAsync ctx kernel

-- | WGSL shader for appending to KV cache
appendKVCacheShader :: Bool -> Int -> Int -> Int -> String
appendKVCacheShader useFP16 maxSeqLen numKVHeads headDim =
  let floatType = if useFP16 then "f16" else "f32"
      enableDirective = if useFP16 then ["enable f16;", ""] else []
  in unlines $
  [ "// Append new K/V to cache at position"
  ] ++ enableDirective ++
  [ "@group(0) @binding(0) var<storage, read_write> k_cache: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(1) var<storage, read_write> v_cache: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(2) var<storage, read_write> k_new: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(3) var<storage, read_write> v_new: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(4) var<storage, read_write> position: array<" ++ floatType ++ ">;"
  , ""
  , "const MAX_SEQ_LEN: u32 = " ++ show maxSeqLen ++ "u;"
  , "const NUM_KV_HEADS: u32 = " ++ show numKVHeads ++ "u;"
  , "const HEAD_DIM: u32 = " ++ show headDim ++ "u;"
  , ""
  , "@compute @workgroup_size(256)"
  , "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {"
  , "  let idx = gid.x;"
  , "  let kv_size = NUM_KV_HEADS * HEAD_DIM;"
  , "  if (idx >= kv_size) { return; }"
  , ""
  , "  let pos = u32(position[0]);"
  , "  "
  , "  // Cache layout: [maxSeqLen, numKVHeads, headDim]"
  , "  // New tensor layout: [numKVHeads, headDim]"
  , "  let cache_offset = pos * kv_size + idx;"
  , "  "
  , "  k_cache[cache_offset] = k_new[idx];"
  , "  v_cache[cache_offset] = v_new[idx];"
  , "}"
  ]

-- ═══════════════════════════════════════════════════════════════════════
-- Phase 4.1: Attention Core Fusion (Scores + Output)
-- ═══════════════════════════════════════════════════════════════════════

-- | MEGA-FUSION: Attention Scores + Softmax + Weighted Sum
--
-- Fuses the two core attention operations:
-- 1. Attention Scores: Q @ K^T / sqrt(d) + softmax
-- 2. Attention Output: scores @ V
--
-- Key optimization: Scores stay in workgroup shared memory!
-- - NO global memory writes for intermediate scores
-- - NO global memory reads to fetch them back
-- - Saves ~2× memory bandwidth + dispatch overhead
--
-- Expected speedup: 10-15% (2 dispatches → 1, zero score memory traffic)
attentionCoreFusedShader :: Bool -> Int -> Int -> Int -> Int -> Maybe Int -> Int -> String
attentionCoreFusedShader useFP16 numHeads numKVHeads headDim cacheLen windowSize maxCacheLen =
  let floatType = if useFP16 then "f16" else "f32"
      floatLit = if useFP16 then "h" else ""
      enableDirective = if useFP16 then ["enable f16;", ""] else []
      window = case windowSize of
        Just w -> w
        Nothing -> cacheLen
  in unlines $
  [ "// PHASE 4.1: Attention Core Fusion (Scores + Output)"
  , "// Fuses: Q @ K^T + Softmax + Scores @ V"
  , "// Scores stay in workgroup shared memory (ZERO global writes!)"
  ] ++ enableDirective ++
  [ "@group(0) @binding(0) var<storage, read_write> q: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(1) var<storage, read_write> k_cache: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(2) var<storage, read_write> v_cache: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(3) var<storage, read_write> output: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(4) var<storage, read_write> window_size: array<" ++ floatType ++ ">;"  -- FIX: Dynamic cache length
  , ""
  , "const NUM_HEADS: u32 = " ++ show numHeads ++ "u;"
  , "const NUM_KV_HEADS: u32 = " ++ show numKVHeads ++ "u;"
  , "const HEAD_DIM: u32 = " ++ show headDim ++ "u;"
  , "const MAX_CACHE_LEN: u32 = " ++ show maxCacheLen ++ "u;"  -- FIX: Renamed to MAX_CACHE_LEN for clarity
  , "const SCALE: " ++ floatType ++ " = " ++ show (1.0 / sqrt (fromIntegral headDim :: Float)) ++ floatLit ++ ";"
  , ""
  , "// Workgroup shared memory for scores (avoids global memory!)"
  , "var<workgroup> scores: array<" ++ floatType ++ ", " ++ show maxCacheLen ++ ">;"  -- FIX: Use MAX_CACHE_LEN for array size
  , ""
  , "@compute @workgroup_size(256)"
  , "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {"
  , "  let head_idx = gid.x;"
  , "  if (head_idx >= NUM_HEADS) { return; }"
  , ""
  , "  // GQA: Map query head to KV head"
  , "  let kv_head_idx = head_idx / (NUM_HEADS / NUM_KV_HEADS);"
  , "  let q_offset = head_idx * HEAD_DIM;"
  , ""
  , "  let window = u32(window_size[0]);"  -- FIX: Read dynamic window size
  , ""
  , "  // STEP 1: Compute Q @ K^T (attention scores)"
  , "  var max_score: " ++ floatType ++ " = -1e9" ++ floatLit ++ ";"
  , "  for (var pos: u32 = 0u; pos < window; pos++) {"  -- FIX: Use dynamic window
  , "    let k_offset = pos * NUM_KV_HEADS * HEAD_DIM + kv_head_idx * HEAD_DIM;"
  , "    var dot: " ++ floatType ++ " = 0.0" ++ floatLit ++ ";"
  , "    for (var i: u32 = 0u; i < HEAD_DIM; i++) {"
  , "      dot += q[q_offset + i] * k_cache[k_offset + i];"
  , "    }"
  , "    let score = dot * SCALE;"
  , "    scores[pos] = score;  // Workgroup shared memory!"
  , "    max_score = max(max_score, score);"
  , "  }"
  , ""
  , "  // STEP 2: Softmax normalization (exp and normalize)"
  , "  var sum_exp: " ++ floatType ++ " = 0.0" ++ floatLit ++ ";"
  , "  for (var pos: u32 = 0u; pos < window; pos++) {"  -- FIX: Use dynamic window
  , "    let exp_val = exp(scores[pos] - max_score);"
  , "    scores[pos] = exp_val;  // Update in-place!"
  , "    sum_exp += exp_val;"
  , "  }"
  , "  for (var pos: u32 = 0u; pos < window; pos++) {"  -- FIX: Use dynamic window
  , "    scores[pos] /= sum_exp;  // Normalize in-place!"
  , "  }"
  , ""
  , "  // STEP 3: Compute weighted sum (scores @ V)"
  , "  let out_offset = head_idx * HEAD_DIM;"
  , "  for (var d: u32 = 0u; d < HEAD_DIM; d++) {"
  , "    var sum: " ++ floatType ++ " = 0.0" ++ floatLit ++ ";"
  , "    for (var pos: u32 = 0u; pos < window; pos++) {"  -- FIX: Use dynamic window
  , "      let v_offset = pos * NUM_KV_HEADS * HEAD_DIM + kv_head_idx * HEAD_DIM;"
  , "      sum += scores[pos] * v_cache[v_offset + d];  // Read from shared memory!"
  , "    }"
  , "    output[out_offset + d] = sum;"
  , "  }"
  , "}"
  ]

-- | Run attention core fusion with pre-uploaded K/V cache
--
-- Replaces 2 separate dispatches:
--   1. runAttentionScoresGPU
--   2. runAttentionOutputGPU
--
-- With a single fused dispatch, saving memory bandwidth + dispatch overhead.
runAttentionCoreFusedPreloadedGPU :: Context
                                  -> Tensor dtype  -- Q [numHeads, headDim]
                                  -> Tensor dtype  -- K cache [cacheLen, numKVHeads, headDim]
                                  -> Tensor dtype  -- V cache [cacheLen, numKVHeads, headDim]
                                  -> Tensor dtype  -- Output buffer (pre-allocated, REUSED!)
                                  -> KernelCode    -- Pre-compiled fused shader
                                  -> Int           -- numHeads
                                  -> Int           -- numKVHeads
                                  -> Int           -- headDim
                                  -> Int           -- cacheLen (effective window)
                                  -> Maybe Int     -- windowSize
                                  -> Int           -- maxCacheLen
                                  -> ContT r IO ()
runAttentionCoreFusedPreloadedGPU ctx qTensor kCacheTensor vCacheTensor outputTensor code numHeads numKVHeads headDim cacheLen windowSize maxCacheLen = do
  -- BUG FIX: Use cacheLen directly (it's already effectiveLen = min(cacheLen', windowSize))
  -- The caller (TransformerBlock) already computed the window, don't re-apply it!
  let windowVal = fromIntegral cacheLen :: Float
      windowShape = Shape [1]
      windowData = V.singleton windowVal
  windowTensor <- createTensorWithData ctx windowShape windowData

  let numWorkgroups = (numHeads + 255) `div` 256
  kernel <- createKernel ctx code [qTensor, kCacheTensor, vCacheTensor, outputTensor, windowTensor]  -- FIX: Added windowTensor
            (WorkgroupSize numWorkgroups 1 1)
  liftIO $ dispatchKernelAsync ctx kernel

-- ═══════════════════════════════════════════════════════════════════════
-- Phase 3.2: Attention Postprocessing Fusion
-- ═══════════════════════════════════════════════════════════════════════

-- | MEGA-FUSION: Output Projection + Residual + RMSNorm
--
-- Combines 3 attention postprocessing operations:
-- 1. Output Projection (attention output → hidden dim)
-- 2. Residual Add (with pre-attention input)
-- 3. Post-Attention RMSNorm (Gemma 3 only)
--
-- Benefits:
-- - Eliminates 2 kernel dispatches per layer
-- - Total savings: 2 × 26 layers = 52 dispatches per forward pass
attentionPostFusedShader :: Bool -> Bool -> Int -> Int -> Bool -> String
attentionPostFusedShader useFP16 useVec4 hiddenDim qSize zeroCentered =
  let floatType = if useFP16 then "f16" else "f32"
      floatLit = if useFP16 then "h" else ""
      enableDirective = if useFP16 then ["enable f16;", ""] else []
      vec4Iters = qSize `div` 4
      remainder = qSize `mod` 4
  in unlines $
  [ "// MEGA-FUSION: Attention Postprocessing"
  , "// Phase 3.2: Output Projection + Residual + RMSNorm"
  ] ++ enableDirective ++
  [ ""
  , "@group(0) @binding(0) var<storage, read_write> input: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(1) var<storage, read_write> out_weight: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(2) var<storage, read_write> residual: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(3) var<storage, read_write> norm_weight: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(4) var<storage, read_write> output: array<" ++ floatType ++ ">;"
  , ""
  , "const HIDDEN_DIM: u32 = " ++ show hiddenDim ++ "u;"
  , "const Q_SIZE: u32 = " ++ show qSize ++ "u;"
  , "const EPSILON: " ++ floatType ++ " = 1e-6" ++ floatLit ++ ";"
  , "const WORKGROUP_SIZE: u32 = 256u;"
  , ""
  , "var<workgroup> shared_rms: array<" ++ floatType ++ ", WORKGROUP_SIZE>;"
  , "var<workgroup> rms_value: " ++ floatType ++ ";"
  , ""
  , "@compute @workgroup_size(256)"
  , "fn main(@builtin(local_invocation_id) lid: vec3<u32>,"
  , "        @builtin(global_invocation_id) gid: vec3<u32>) {"
  , "  let tid = lid.x;"
  , "  let output_idx = gid.x;"
  , ""
  , "  // STEP 1: Output Projection"
  , "  var proj_sum: " ++ floatType ++ " = 0.0" ++ floatLit ++ ";"
  , "  if (output_idx < HIDDEN_DIM) {"
  ] ++
  (if useVec4 && vec4Iters > 0 then
      [ "    for (var i: u32 = 0u; i < " ++ show vec4Iters ++ "u; i++) {"
      , "      let idx = i * 4u;"
      , "      let w_base = output_idx * Q_SIZE + idx;"
      , "      let in_vec = vec4<" ++ floatType ++ ">(input[idx], input[idx+1u], input[idx+2u], input[idx+3u]);"
      , "      let w_vec = vec4<" ++ floatType ++ ">(out_weight[w_base], out_weight[w_base+1u], out_weight[w_base+2u], out_weight[w_base+3u]);"
      , "      let prod = in_vec * w_vec;"
      , "      proj_sum += prod.x + prod.y + prod.z + prod.w;"
      , "    }"
      ]
    else []) ++
  (if useVec4 && remainder > 0 || not useVec4 then
      [ "    let start_idx = " ++ (if useVec4 then show (vec4Iters * 4) ++ "u" else "0u") ++ ";"
      , "    for (var i: u32 = start_idx; i < Q_SIZE; i++) {"
      , "      let w_idx = output_idx * Q_SIZE + i;"
      , "      proj_sum += input[i] * out_weight[w_idx];"
      , "    }"
      ]
    else []) ++
  [ "  }"
  , ""
  , "  // STEP 2: Residual Add"
  , "  var with_residual: " ++ floatType ++ " = 0.0" ++ floatLit ++ ";"
  , "  if (output_idx < HIDDEN_DIM) {"
  , "    with_residual = proj_sum + residual[output_idx];"
  , "  }"
  , ""
  , "  // STEP 3: RMSNorm"
  , "  var sum_sq: " ++ floatType ++ " = 0.0" ++ floatLit ++ ";"
  , "  if (output_idx < HIDDEN_DIM) {"
  , "    sum_sq = with_residual * with_residual;"
  , "  }"
  , "  shared_rms[tid] = sum_sq;"
  , "  workgroupBarrier();"
  , ""
  , "  for (var stride: u32 = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {"
  , "    if (tid < stride) {"
  , "      shared_rms[tid] = shared_rms[tid] + shared_rms[tid + stride];"
  , "    }"
  , "    workgroupBarrier();"
  , "  }"
  , ""
  , "  if (tid == 0u) {"
  , "    let mean_sq = shared_rms[0] / " ++ floatType ++ "(HIDDEN_DIM);"
  , "    rms_value = sqrt(mean_sq + EPSILON);"
  , "  }"
  , "  workgroupBarrier();"
  , "  let rms = rms_value;"
  , ""
  , "  if (output_idx < HIDDEN_DIM) {"
  , if zeroCentered
     then "    output[output_idx] = (with_residual / rms) * (1.0" ++ floatLit ++ " + norm_weight[output_idx]);"
     else "    output[output_idx] = (with_residual / rms) * norm_weight[output_idx];"
  , "  }"
  , "}"
  ]

-- | Run attention postprocessing fusion with pre-uploaded weights
--
-- Replaces 3 separate dispatches:
--   1. runOutputProjectionGPU
--   2. runResidualAddGPU
--   3. runRMSNormPreloadedGPU (post-attention norm)
--
-- With a single fused dispatch, saving ~60-100μs per layer.
runAttentionPostFusedPreloadedGPU :: Context
                                  -> Tensor dtype  -- Attention output [qSize]
                                  -> Tensor dtype  -- Output projection weight [hiddenDim, qSize]
                                  -> Tensor dtype  -- Residual (pre-attention input) [hiddenDim]
                                  -> Tensor dtype  -- Norm weight tensor [hiddenDim]
                                  -> Tensor dtype  -- Output buffer (pre-allocated, REUSED!)
                                  -> KernelCode    -- Pre-compiled fused shader
                                  -> Int           -- hiddenDim
                                  -> ContT r IO ()
runAttentionPostFusedPreloadedGPU ctx attnOutTensor outProjWeightTensor residualTensor normWeightTensor outputTensor code hiddenDim = do
  let numWorkgroups = (hiddenDim + 255) `div` 256
  kernel <- createKernel ctx code
            [attnOutTensor, outProjWeightTensor, residualTensor, normWeightTensor, outputTensor]
            (WorkgroupSize numWorkgroups 1 1)
  liftIO $ dispatchKernelAsync ctx kernel
