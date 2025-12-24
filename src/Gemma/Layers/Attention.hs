{-# LANGUAGE OverloadedStrings #-}

{-|
Module: Gemma.Layers.Attention
Description: Scaled Dot-Product Attention

Attention computes:
  1. Scores = Q @ K^T / sqrt(head_dim)
  2. Attention_weights = softmax(Scores, dim=-1)
  3. Output = Attention_weights @ V

Where:
  - Q: Query matrix [seq_len, head_dim]
  - K: Key matrix [seq_len, head_dim]
  - V: Value matrix [seq_len, head_dim]

Reference: Attention Is All You Need
           https://arxiv.org/abs/1706.03762
-}

module Gemma.Layers.Attention
  ( runAttention
  , runAttentionWithContext
  , attentionScoresShader
  , attentionOutputShader
  ) where

import Graphics.WebGPU.Dawn.ContT
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)

-- | WGSL shader to compute attention scores: Q @ K^T / sqrt(head_dim) and apply softmax
--
-- Each thread computes one row of the attention matrix.
-- If windowSize is Just n, applies sliding window masking (local attention).
-- If windowSize is Nothing, uses full attention (global).
attentionScoresShader :: Int -> Int -> Maybe Int -> String
attentionScoresShader seqLen headDim windowSize = unlines
  [ "// Attention Scores: Q @ K^T / sqrt(head_dim) + softmax"
  , ""
  , "@group(0) @binding(0) var<storage, read_write> q: array<f32>;"
  , "@group(0) @binding(1) var<storage, read_write> k: array<f32>;"
  , "@group(0) @binding(2) var<storage, read_write> scores: array<f32>;"
  , ""
  , "const SEQ_LEN: u32 = " ++ show seqLen ++ "u;"
  , "const HEAD_DIM: u32 = " ++ show headDim ++ "u;"
  , "const SCALE: f32 = " ++ show (1.0 / sqrt (fromIntegral headDim :: Float)) ++ ";"
  , case windowSize of
      Just w -> "const WINDOW_SIZE: u32 = " ++ show w ++ "u;  // Sliding window for local attention"
      Nothing -> "// Full attention (no window masking)"
  , ""
  , "@compute @workgroup_size(256)"
  , "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {"
  , "  let row = gid.x;"
  , "  "
  , "  if (row < SEQ_LEN) {"
  , "    // Compute Q[row] @ K^T (dot product with each K row)"
  , "    var max_val: f32 = -1e10;"
  , "    "
  , "    // First pass: compute scores and find max for numerical stability"
  , "    for (var col: u32 = 0u; col < SEQ_LEN; col = col + 1u) {"
  , case windowSize of
      Just _ -> unlines
        [ "      // Sliding window masking: only attend to positions within window"
        , "      let is_in_window = (col <= row) && ((row - col) < WINDOW_SIZE);"
        , "      if (!is_in_window) {"
        , "        scores[row * SEQ_LEN + col] = -1e10;  // Mask with large negative value"
        , "        continue;"
        , "      }"
        ]
      Nothing -> ""
  , "      var dot: f32 = 0.0;"
  , "      for (var i: u32 = 0u; i < HEAD_DIM; i = i + 1u) {"
  , "        dot = dot + q[row * HEAD_DIM + i] * k[col * HEAD_DIM + i];"
  , "      }"
  , "      let scaled = dot * SCALE;"
  , "      scores[row * SEQ_LEN + col] = scaled;"
  , "      max_val = max(max_val, scaled);"
  , "    }"
  , "    "
  , "    // Second pass: exp and sum for softmax"
  , "    var sum: f32 = 0.0;"
  , "    for (var col: u32 = 0u; col < SEQ_LEN; col = col + 1u) {"
  , "      let idx = row * SEQ_LEN + col;"
  , "      let exp_val = exp(scores[idx] - max_val);"
  , "      scores[idx] = exp_val;"
  , "      sum = sum + exp_val;"
  , "    }"
  , "    "
  , "    // Third pass: normalize"
  , "    for (var col: u32 = 0u; col < SEQ_LEN; col = col + 1u) {"
  , "      let idx = row * SEQ_LEN + col;"
  , "      scores[idx] = scores[idx] / sum;"
  , "    }"
  , "  }"
  , "}"
  ]

-- | WGSL shader to compute attention output: Scores @ V
--
-- Each thread computes one element of the output.
attentionOutputShader :: Int -> Int -> String
attentionOutputShader seqLen headDim = unlines
  [ "// Attention Output: Scores @ V"
  , ""
  , "@group(0) @binding(0) var<storage, read_write> scores: array<f32>;"
  , "@group(0) @binding(1) var<storage, read_write> v: array<f32>;"
  , "@group(0) @binding(2) var<storage, read_write> output: array<f32>;"
  , ""
  , "const SEQ_LEN: u32 = " ++ show seqLen ++ "u;"
  , "const HEAD_DIM: u32 = " ++ show headDim ++ "u;"
  , ""
  , "@compute @workgroup_size(256)"
  , "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {"
  , "  let row = gid.x;"
  , "  "
  , "  if (row < SEQ_LEN) {"
  , "    // Compute one row of output: scores[row] @ V"
  , "    for (var col: u32 = 0u; col < HEAD_DIM; col = col + 1u) {"
  , "      var sum: f32 = 0.0;"
  , "      for (var i: u32 = 0u; i < SEQ_LEN; i = i + 1u) {"
  , "        sum = sum + scores[row * SEQ_LEN + i] * v[i * HEAD_DIM + col];"
  , "      }"
  , "      output[row * HEAD_DIM + col] = sum;"
  , "    }"
  , "  }"
  , "}"
  ]

-- | Run Attention on GPU
--
-- Takes:
-- - q: Query matrix [seq_len * head_dim] in row-major order
-- - k: Key matrix [seq_len * head_dim] in row-major order
-- - v: Value matrix [seq_len * head_dim] in row-major order
-- - seqLen: sequence length
-- - headDim: dimension of attention head
-- - windowSize: optional sliding window size (Just n for local, Nothing for global)
--
-- Returns: output matrix [seq_len * head_dim] in row-major order
runAttention :: Vector Float -> Vector Float -> Vector Float -> Int -> Int -> Maybe Int -> ContT r IO (Vector Float)
runAttention q k v seqLen headDim windowSize = do
  -- Validate inputs
  let expectedSize = seqLen * headDim
  if V.length q /= expectedSize
    then error $ "Attention: Q size mismatch: " ++ show (V.length q) ++ " vs " ++ show expectedSize
    else pure ()

  if V.length k /= expectedSize
    then error $ "Attention: K size mismatch: " ++ show (V.length k) ++ " vs " ++ show expectedSize
    else pure ()

  if V.length v /= expectedSize
    then error $ "Attention: V size mismatch: " ++ show (V.length v) ++ " vs " ++ show expectedSize
    else pure ()

  -- Create GPU context
  ctx <- createContext

  -- Create tensors
  let qkvShape = Shape [seqLen * headDim]
      scoresShape = Shape [seqLen * seqLen]

  qTensor <- createTensorWithData ctx qkvShape q
  kTensor <- createTensorWithData ctx qkvShape k
  vTensor <- createTensorWithData ctx qkvShape v
  scoresTensor <- createTensor ctx scoresShape F32
  outputTensor <- createTensor ctx qkvShape F32

  -- Step 1: Compute attention scores with softmax
  let scoresShaderCode = attentionScoresShader seqLen headDim windowSize
  scoresCode <- createKernelCode scoresShaderCode

  let numWorkgroups = (seqLen + 255) `div` 256
  scoresKernel <- createKernel ctx scoresCode [qTensor, kTensor, scoresTensor]
                  (WorkgroupSize numWorkgroups 1 1)

  liftIO $ dispatchKernel ctx scoresKernel

  -- Step 2: Compute output = scores @ V
  let outputShaderCode = attentionOutputShader seqLen headDim
  outputCode <- createKernelCode outputShaderCode

  outputKernel <- createKernel ctx outputCode [scoresTensor, vTensor, outputTensor]
                  (WorkgroupSize numWorkgroups 1 1)

  liftIO $ dispatchKernel ctx outputKernel

  -- Read result from GPU
  result <- liftIO $ fromGPU ctx outputTensor expectedSize

  pure result

-- | Run Attention with given context (for use in larger pipelines)
runAttentionWithContext :: Context -> Vector Float -> Vector Float -> Vector Float -> Int -> Int -> Maybe Int -> ContT r IO (Vector Float)
runAttentionWithContext ctx q k v seqLen headDim windowSize = do
  -- Validate inputs
  let expectedSize = seqLen * headDim
  if V.length q /= expectedSize
    then error $ "Attention: Q size mismatch: " ++ show (V.length q) ++ " vs " ++ show expectedSize
    else pure ()

  if V.length k /= expectedSize
    then error $ "Attention: K size mismatch: " ++ show (V.length k) ++ " vs " ++ show expectedSize
    else pure ()

  if V.length v /= expectedSize
    then error $ "Attention: V size mismatch: " ++ show (V.length v) ++ " vs " ++ show expectedSize
    else pure ()

  -- Create tensors
  let qkvShape = Shape [seqLen * headDim]
      scoresShape = Shape [seqLen * seqLen]

  qTensor <- createTensorWithData ctx qkvShape q
  kTensor <- createTensorWithData ctx qkvShape k
  vTensor <- createTensorWithData ctx qkvShape v
  scoresTensor <- createTensor ctx scoresShape F32
  outputTensor <- createTensor ctx qkvShape F32

  -- Step 1: Compute attention scores with softmax
  let scoresShaderCode = attentionScoresShader seqLen headDim windowSize
  scoresCode <- createKernelCode scoresShaderCode

  let numWorkgroups = (seqLen + 255) `div` 256
  scoresKernel <- createKernel ctx scoresCode [qTensor, kTensor, scoresTensor]
                  (WorkgroupSize numWorkgroups 1 1)

  liftIO $ dispatchKernel ctx scoresKernel

  -- Step 2: Compute output = scores @ V
  let outputShaderCode = attentionOutputShader seqLen headDim
  outputCode <- createKernelCode outputShaderCode

  outputKernel <- createKernel ctx outputCode [scoresTensor, vTensor, outputTensor]
                  (WorkgroupSize numWorkgroups 1 1)

  liftIO $ dispatchKernel ctx outputKernel

  -- Read result from GPU
  result <- liftIO $ fromGPU ctx outputTensor expectedSize

  pure result
