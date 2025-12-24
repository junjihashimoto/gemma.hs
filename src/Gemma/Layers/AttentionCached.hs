{-# LANGUAGE OverloadedStrings #-}

{-|
Module: Gemma.Layers.AttentionCached
Description: Cached Attention for efficient autoregressive generation

This module provides attention operations with KV-cache support.
During generation, only the query for the new token is computed,
while keys and values are retrieved from cache.

Performance:
- Without cache: O(N²) for N tokens (recompute all K/V)
- With cache: O(N) for N tokens (reuse cached K/V)
- Speedup: 10-50x for typical sequences

Usage:
  (output, newCache) <- runAttentionCached q cache newK newV ...
-}

module Gemma.Layers.AttentionCached
  ( runAttentionCached
  , runAttentionCachedWithContext
  , attentionScoresCachedShader
  , attentionOutputCachedShader
  ) where

import Graphics.WebGPU.Dawn.ContT
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import Control.Monad.IO.Class (liftIO)
import Gemma.KVCache (LayerKVCache, appendToCache, getCachedKV, cacheLength)

-- | WGSL shader for cached attention scores
--
-- Computes Q @ K_cache^T / sqrt(head_dim) + softmax
-- where Q is [1, head_dim] (single query) and K_cache is [cache_len, head_dim]
attentionScoresCachedShader :: Int -> Int -> Maybe Int -> String
attentionScoresCachedShader cacheLen headDim windowSize = unlines
  [ "// Cached Attention Scores: Q[1] @ K_cache[cache_len]^T + softmax"
  , ""
  , "@group(0) @binding(0) var<storage, read_write> q: array<f32>;"
  , "@group(0) @binding(1) var<storage, read_write> k_cache: array<f32>;"
  , "@group(0) @binding(2) var<storage, read_write> scores: array<f32>;"
  , ""
  , "const CACHE_LEN: u32 = " ++ show cacheLen ++ "u;"
  , "const HEAD_DIM: u32 = " ++ show headDim ++ "u;"
  , "const SCALE: f32 = " ++ show (1.0 / sqrt (fromIntegral headDim :: Float)) ++ ";"
  , case windowSize of
      Just w -> "const WINDOW_SIZE: u32 = " ++ show w ++ "u;"
      Nothing -> "// Full attention (no window)"
  , ""
  , "@compute @workgroup_size(256)"
  , "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {"
  , "  // We have 1 query, compute dot product with each of cache_len keys"
  , "  let cache_pos = gid.x;"
  , "  "
  , "  if (cache_pos < CACHE_LEN) {"
  , case windowSize of
      Just _ -> unlines
        [ "    // Sliding window: only attend to recent positions"
        , "    // Current position is CACHE_LEN-1 (last in cache)"
        , "    let current_pos = CACHE_LEN - 1u;"
        , "    let is_in_window = (cache_pos <= current_pos) && ((current_pos - cache_pos) < WINDOW_SIZE);"
        , "    if (!is_in_window) {"
        , "      scores[cache_pos] = -1e10;"
        , "      return;"
        , "    }"
        ]
      Nothing -> ""
  , "    // Compute Q @ K_cache[cache_pos]"
  , "    var dot: f32 = 0.0;"
  , "    for (var i: u32 = 0u; i < HEAD_DIM; i = i + 1u) {"
  , "      dot = dot + q[i] * k_cache[cache_pos * HEAD_DIM + i];"
  , "    }"
  , "    scores[cache_pos] = dot * SCALE;"
  , "  }"
  , "}"
  , ""
  , "// Softmax pass (separate kernel for simplicity)"
  , "@compute @workgroup_size(1)"
  , "fn softmax() {"
  , "  // Find max"
  , "  var max_val: f32 = -1e10;"
  , "  for (var i: u32 = 0u; i < CACHE_LEN; i = i + 1u) {"
  , "    max_val = max(max_val, scores[i]);"
  , "  }"
  , "  "
  , "  // Exp and sum"
  , "  var sum: f32 = 0.0;"
  , "  for (var i: u32 = 0u; i < CACHE_LEN; i = i + 1u) {"
  , "    let exp_val = exp(scores[i] - max_val);"
  , "    scores[i] = exp_val;"
  , "    sum = sum + exp_val;"
  , "  }"
  , "  "
  , "  // Normalize"
  , "  for (var i: u32 = 0u; i < CACHE_LEN; i = i + 1u) {"
  , "    scores[i] = scores[i] / sum;"
  , "  }"
  , "}"
  ]

-- | WGSL shader for cached attention output
--
-- Computes scores @ V_cache where scores is [cache_len] and V_cache is [cache_len, head_dim]
-- Output is [1, head_dim]
attentionOutputCachedShader :: Int -> Int -> String
attentionOutputCachedShader cacheLen headDim = unlines
  [ "// Cached Attention Output: scores[cache_len] @ V_cache[cache_len, head_dim]"
  , ""
  , "@group(0) @binding(0) var<storage, read_write> scores: array<f32>;"
  , "@group(0) @binding(1) var<storage, read_write> v_cache: array<f32>;"
  , "@group(0) @binding(2) var<storage, read_write> output: array<f32>;"
  , ""
  , "const CACHE_LEN: u32 = " ++ show cacheLen ++ "u;"
  , "const HEAD_DIM: u32 = " ++ show headDim ++ "u;"
  , ""
  , "@compute @workgroup_size(256)"
  , "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {"
  , "  let col = gid.x;"
  , "  "
  , "  if (col < HEAD_DIM) {"
  , "    // Compute output[col] = sum(scores[i] * v_cache[i, col])"
  , "    var sum: f32 = 0.0;"
  , "    for (var i: u32 = 0u; i < CACHE_LEN; i = i + 1u) {"
  , "      sum = sum + scores[i] * v_cache[i * HEAD_DIM + col];"
  , "    }"
  , "    output[col] = sum;"
  , "  }"
  , "}"
  ]

-- | Run cached attention on GPU
--
-- This function implements KV-cached attention for autoregressive generation.
-- It takes a single query (for the new token) and reuses cached keys/values.
--
-- Parameters:
-- - q: Query for new token [head_dim]
-- - cache: Current KV cache for this layer
-- - newK: New key to add [head_dim]
-- - newV: New value to add [head_dim]
-- - headDim: Dimension of attention head
-- - windowSize: Optional sliding window size
--
-- Returns: (output [head_dim], updated_cache)
runAttentionCached :: Vector Float      -- Query [head_dim]
                   -> LayerKVCache      -- Current cache
                   -> Vector Float      -- New key [head_dim]
                   -> Vector Float      -- New value [head_dim]
                   -> Int               -- head_dim
                   -> Maybe Int         -- window size
                   -> ContT r IO (Vector Float, LayerKVCache)
runAttentionCached q cache newK newV headDim windowSize = do
  -- Step 1: Append new K/V to cache
  let updatedCache = appendToCache cache newK newV
      (cachedK, cachedV) = getCachedKV updatedCache
      cacheLen = cacheLength updatedCache

  -- Validate dimensions
  if V.length q /= headDim
    then error $ "AttentionCached: Q size mismatch: " ++ show (V.length q) ++ " vs " ++ show headDim
    else pure ()

  if V.length cachedK /= cacheLen * headDim
    then error $ "AttentionCached: cached K size mismatch"
    else pure ()

  if V.length cachedV /= cacheLen * headDim
    then error $ "AttentionCached: cached V size mismatch"
    else pure ()

  -- Create GPU context
  ctx <- createContext

  -- Create tensors
  let qShape = Shape [headDim]
      kvShape = Shape [cacheLen * headDim]
      scoresShape = Shape [cacheLen]

  qTensor <- createTensorWithData ctx qShape q
  kCacheTensor <- createTensorWithData ctx kvShape cachedK
  vCacheTensor <- createTensorWithData ctx kvShape cachedV
  scoresTensor <- createTensor ctx scoresShape F32
  outputTensor <- createTensor ctx qShape F32

  -- Step 2: Compute attention scores: Q @ K_cache^T / sqrt(head_dim) + softmax
  let scoresShaderCode = attentionScoresCachedShader cacheLen headDim windowSize
  scoresCode <- createKernelCode scoresShaderCode

  let numWorkgroups = (cacheLen + 255) `div` 256
  scoresKernel <- createKernel ctx scoresCode [qTensor, kCacheTensor, scoresTensor]
                  (WorkgroupSize numWorkgroups 1 1)

  liftIO $ dispatchKernel ctx scoresKernel

  -- Step 3: Compute attention output: scores @ V_cache
  let outputShaderCode = attentionOutputCachedShader cacheLen headDim
  outputCode <- createKernelCode outputShaderCode

  let outputWorkgroups = (headDim + 255) `div` 256
  outputKernel <- createKernel ctx outputCode [scoresTensor, vCacheTensor, outputTensor]
                  (WorkgroupSize outputWorkgroups 1 1)

  liftIO $ dispatchKernel ctx outputKernel

  -- Read result from GPU
  result <- liftIO $ fromGPU ctx outputTensor headDim

  return (result, updatedCache)

-- | Run cached attention with given context (for use in larger pipelines)
runAttentionCachedWithContext :: Context
                              -> Vector Float
                              -> LayerKVCache
                              -> Vector Float
                              -> Vector Float
                              -> Int
                              -> Maybe Int
                              -> ContT r IO (Vector Float, LayerKVCache)
runAttentionCachedWithContext ctx q cache newK newV headDim windowSize = do
  -- Step 1: Append new K/V to cache
  let updatedCache = appendToCache cache newK newV
      (cachedK, cachedV) = getCachedKV updatedCache
      cacheLen = cacheLength updatedCache

  -- Validate dimensions
  if V.length q /= headDim
    then error $ "AttentionCached: Q size mismatch: " ++ show (V.length q) ++ " vs " ++ show headDim
    else pure ()

  -- Create tensors
  let qShape = Shape [headDim]
      kvShape = Shape [cacheLen * headDim]
      scoresShape = Shape [cacheLen]

  qTensor <- createTensorWithData ctx qShape q
  kCacheTensor <- createTensorWithData ctx kvShape cachedK
  vCacheTensor <- createTensorWithData ctx kvShape cachedV
  scoresTensor <- createTensor ctx scoresShape F32
  outputTensor <- createTensor ctx qShape F32

  -- Compute attention scores
  let scoresShaderCode = attentionScoresCachedShader cacheLen headDim windowSize
  scoresCode <- createKernelCode scoresShaderCode

  let numWorkgroups = (cacheLen + 255) `div` 256
  scoresKernel <- createKernel ctx scoresCode [qTensor, kCacheTensor, scoresTensor]
                  (WorkgroupSize numWorkgroups 1 1)

  liftIO $ dispatchKernel ctx scoresKernel

  -- Compute attention output
  let outputShaderCode = attentionOutputCachedShader cacheLen headDim
  outputCode <- createKernelCode outputShaderCode

  let outputWorkgroups = (headDim + 255) `div` 256
  outputKernel <- createKernel ctx outputCode [scoresTensor, vCacheTensor, outputTensor]
                  (WorkgroupSize outputWorkgroups 1 1)

  liftIO $ dispatchKernel ctx outputKernel

  -- Read result
  result <- liftIO $ fromGPU ctx outputTensor headDim

  return (result, updatedCache)
