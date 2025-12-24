{-# LANGUAGE RecordWildCards #-}

{-|
Module: Gemma.KVCache
Description: KV-Cache for efficient autoregressive generation

KV-cache stores previously computed Key and Value tensors to avoid
recomputing them during autoregressive generation.

Performance impact:
- Without cache: O(N²) computation (recompute all previous tokens)
- With cache: O(N) computation (constant time per token)
- Speedup: 10-50x for typical sequences

Memory cost:
- Per layer: max_seq_len * num_kv_heads * head_dim * 4 bytes * 2 (K+V)
- Gemma 3 1B (2048 max_seq_len): ~109 MB for 26 layers
-}

module Gemma.KVCache
  ( -- * Cache Types
    LayerKVCache(..)
  , KVCache(..)

    -- * Cache Operations
  , initKVCache
  , initLayerCache
  , appendToCache
  , getCachedKV
  , cacheLength
  , resetCache

    -- * Utilities
  , maxCacheSize
  ) where

import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import qualified Data.Vector as BV

-- | KV-cache for a single transformer layer
--
-- Stores Keys and Values for all previously processed tokens.
-- Pre-allocated to max_seq_len for efficient updates.
data LayerKVCache = LayerKVCache
  { -- | Cached keys [max_seq_len, num_kv_heads * head_dim]
    -- Only first cacheLen positions are valid
    cacheKeys   :: !(Vector Float)

    -- | Cached values [max_seq_len, num_kv_heads * head_dim]
    -- Only first cacheLen positions are valid
  , cacheValues :: !(Vector Float)

    -- | Current cache length (number of tokens cached)
  , cacheLen    :: !Int

    -- | Maximum cache capacity
  , cacheMaxLen :: !Int

    -- | KV dimension (num_kv_heads * head_dim)
  , cacheKVDim  :: !Int
  } deriving (Show, Eq)

-- | Full model KV-cache (one LayerKVCache per transformer layer)
newtype KVCache = KVCache
  { kvLayers :: BV.Vector LayerKVCache
  } deriving (Show, Eq)

-- | Initialize an empty KV-cache for the entire model
--
-- Parameters:
-- - numLayers: Number of transformer layers
-- - numKVHeads: Number of key/value heads (for GQA)
-- - headDim: Dimension per head
-- - maxSeqLen: Maximum sequence length to cache
--
-- Returns: Empty cache ready for use
initKVCache :: Int -> Int -> Int -> Int -> KVCache
initKVCache numLayers numKVHeads headDim maxSeqLen = KVCache
  { kvLayers = BV.generate numLayers $ \_ ->
      initLayerCache numKVHeads headDim maxSeqLen
  }

-- | Initialize an empty cache for a single layer
initLayerCache :: Int -> Int -> Int -> LayerKVCache
initLayerCache numKVHeads headDim maxSeqLen =
  let kvDim = numKVHeads * headDim
      totalSize = maxSeqLen * kvDim
  in LayerKVCache
    { cacheKeys   = V.replicate totalSize 0
    , cacheValues = V.replicate totalSize 0
    , cacheLen    = 0
    , cacheMaxLen = maxSeqLen
    , cacheKVDim  = kvDim
    }

-- | Append new Key and Value to the cache
--
-- Parameters:
-- - cache: Current layer cache
-- - newKey: New key tensor [num_kv_heads * head_dim]
-- - newValue: New value tensor [num_kv_heads * head_dim]
--
-- Returns: Updated cache with new K/V appended
--
-- Throws error if cache is full
appendToCache :: LayerKVCache -> Vector Float -> Vector Float -> LayerKVCache
appendToCache cache@LayerKVCache{..} newKey newValue
  | cacheLen >= cacheMaxLen =
      error $ "KV-cache is full! max_len=" ++ show cacheMaxLen
  | V.length newKey /= cacheKVDim =
      error $ "Key dimension mismatch: expected " ++ show cacheKVDim
           ++ ", got " ++ show (V.length newKey)
  | V.length newValue /= cacheKVDim =
      error $ "Value dimension mismatch: expected " ++ show cacheKVDim
           ++ ", got " ++ show (V.length newValue)
  | otherwise =
      let pos = cacheLen
          offset = pos * cacheKVDim

          -- Update keys: copy newKey to position 'pos'
          newKeys = cacheKeys V.// zip [offset .. offset + cacheKVDim - 1]
                                       (V.toList newKey)

          -- Update values: copy newValue to position 'pos'
          newValues = cacheValues V.// zip [offset .. offset + cacheKVDim - 1]
                                           (V.toList newValue)

      in cache
        { cacheKeys = newKeys
        , cacheValues = newValues
        , cacheLen = pos + 1
        }

-- | Get the cached Keys and Values up to current length
--
-- Returns: (keys, values) where both are [cacheLen, num_kv_heads * head_dim]
getCachedKV :: LayerKVCache -> (Vector Float, Vector Float)
getCachedKV LayerKVCache{..} =
  let totalLen = cacheLen * cacheKVDim
      keys = V.take totalLen cacheKeys
      values = V.take totalLen cacheValues
  in (keys, values)

-- | Get current cache length (number of tokens cached)
cacheLength :: LayerKVCache -> Int
cacheLength = cacheLen

-- | Reset cache to empty (keeps allocated buffers)
resetCache :: LayerKVCache -> LayerKVCache
resetCache cache = cache { cacheLen = 0 }

-- | Get maximum cache size
maxCacheSize :: LayerKVCache -> Int
maxCacheSize = cacheMaxLen
