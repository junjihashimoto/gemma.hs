{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{-|
Module: Gemma.Layers.TransformerBlock
Description: Complete Transformer Layer with Attention and MLP

A single transformer block consists of:
1. Pre-attention RMSNorm
2. Multi-head Self-Attention with RoPE
3. Residual connection 1
4. Pre-MLP RMSNorm
5. GeGLU MLP
6. Residual connection 2

Formula:
  x_norm1 = RMSNorm(x, attn_norm_weights)
  q, k, v = Linear(x_norm1) for each projection
  q_rot = RoPE(q, position)
  k_rot = RoPE(k, position)
  attn_out = Attention(q_rot, k_rot, v)
  attn_out = Linear(attn_out, out_proj_weights)
  x = x + attn_out  # Residual 1

  x_norm2 = RMSNorm(x, ffn_norm_weights)
  mlp_out = GeGLU(x_norm2, gate_weights, up_weights, down_weights)
  x = x + mlp_out  # Residual 2

  return x
-}

module Gemma.Layers.TransformerBlock
  ( TransformerLayer(..)
  , TransformerWeights(..)
  , LayerQ4Offsets(..)
  , runTransformerBlock
  , runTransformerBlockCached
  , expandKVHeads  -- Export for testing
  ) where

import Graphics.WebGPU.Dawn.ContT
import Graphics.WebGPU.Dawn.Types (KernelCode)
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import Control.Monad.IO.Class (liftIO)
import System.Environment (lookupEnv)

import Gemma.Layers.RMSNorm (runRMSNormWithContext)
import Gemma.Layers.Linear (runLinearWithContext)
import Gemma.Layers.LinearQ4 (runLinearQ4)
import Gemma.Layers.RoPE (runRoPEWithContext)
import Gemma.Layers.Attention (runAttentionWithContext)
import Gemma.Layers.AttentionCached (runAttentionCachedWithContext)
import Gemma.Layers.MLP (runGeGLUWithContext, runElementwiseAddWithContext)
import Gemma.KVCache (LayerKVCache)
import Data.Word (Word32)

-- | Debug print helper (checks DEBUG env var)
debugPrint :: String -> IO ()
debugPrint msg = do
  debug <- lookupEnv "DEBUG"
  case debug of
    Just "1" -> putStrLn msg
    Just "true" -> putStrLn msg
    _ -> pure ()

-- | Expand K/V heads for Grouped Query Attention (GQA)
--
-- In GQA, we have fewer K/V heads than Q heads. To use standard attention,
-- we need to replicate each K/V head to match the Q head count.
--
-- For example: If Q has 4 heads and K/V have 2 heads, we replicate:
--   K/V head 0 -> Q heads 0, 1
--   K/V head 1 -> Q heads 2, 3
expandKVHeads :: Vector Float -> Int -> Int -> Int -> Vector Float
expandKVHeads kv numQHeads numKVHeads headDim =
  let headsPerKV = numQHeads `div` numKVHeads
      expandedList = concatMap (\kvHead ->
        let kvOffset = kvHead * headDim
            headData = V.toList $ V.slice kvOffset headDim kv
        in concat $ replicate headsPerKV headData
        ) [0 .. numKVHeads - 1]
  in V.fromList expandedList

-- | Q4 weight offsets for consolidated tensors
-- All 7 weights (Q, K, V, Out, Gate, Up, Down) are concatenated into 2 large tensors:
-- - One for all packed weights (U32)
-- - One for all scales (F32)
-- This reduces buffer count from 14 to 2 per layer (364 → 52 total for 26 layers)
data LayerQ4Offsets = LayerQ4Offsets
  { q4QPackedOffset :: Int       -- Offset to Q packed weights in consolidated packed tensor
  , q4QPackedSize :: Int         -- Size of Q packed weights
  , q4QScalesOffset :: Int       -- Offset to Q scales in consolidated scales tensor
  , q4QScalesSize :: Int         -- Size of Q scales
  , q4KPackedOffset :: Int       -- Offset to K packed weights
  , q4KPackedSize :: Int         -- Size of K packed weights
  , q4KScalesOffset :: Int       -- Offset to K scales
  , q4KScalesSize :: Int         -- Size of K scales
  , q4VPackedOffset :: Int       -- Offset to V packed weights
  , q4VPackedSize :: Int         -- Size of V packed weights
  , q4VScalesOffset :: Int       -- Offset to V scales
  , q4VScalesSize :: Int         -- Size of V scales
  , q4OutPackedOffset :: Int     -- Offset to Out packed weights
  , q4OutPackedSize :: Int       -- Size of Out packed weights
  , q4OutScalesOffset :: Int     -- Offset to Out scales
  , q4OutScalesSize :: Int       -- Size of Out scales
  , q4GatePackedOffset :: Int    -- Offset to Gate packed weights
  , q4GatePackedSize :: Int      -- Size of Gate packed weights
  , q4GateScalesOffset :: Int    -- Offset to Gate scales
  , q4GateScalesSize :: Int      -- Size of Gate scales
  , q4UpPackedOffset :: Int      -- Offset to Up packed weights
  , q4UpPackedSize :: Int        -- Size of Up packed weights
  , q4UpScalesOffset :: Int      -- Offset to Up scales
  , q4UpScalesSize :: Int        -- Size of Up scales
  , q4DownPackedOffset :: Int    -- Offset to Down packed weights
  , q4DownPackedSize :: Int      -- Size of Down packed weights
  , q4DownScalesOffset :: Int    -- Offset to Down scales
  , q4DownScalesSize :: Int      -- Size of Down scales
  } deriving (Show, Eq)

-- | Projection weights for transformer layers (FP32 or Q4 quantized)
-- Normalization weights are always FP32 regardless of model type
data TransformerWeights
  = FP32Weights
      { wAttnQWeights :: Vector Float      -- [num_heads * head_dim, hidden_dim]
      , wAttnKWeights :: Vector Float      -- [num_kv_heads * head_dim, hidden_dim]
      , wAttnVWeights :: Vector Float      -- [num_kv_heads * head_dim, hidden_dim]
      , wAttnOutWeights :: Vector Float    -- [hidden_dim, num_heads * head_dim]
      , wFFNGateWeights :: Vector Float    -- [ffn_dim, hidden_dim]
      , wFFNUpWeights :: Vector Float      -- [ffn_dim, hidden_dim]
      , wFFNDownWeights :: Vector Float    -- [hidden_dim, ffn_dim]
      }
  | Q4Weights
      { wAttnQWeightsQ4 :: Vector Word32   -- Q4_0 Q projection
      , wAttnKWeightsQ4 :: Vector Word32   -- Q4_0 K projection
      , wAttnVWeightsQ4 :: Vector Word32   -- Q4_0 V projection
      , wAttnOutWeightsQ4 :: Vector Word32 -- Q4_0 output projection
      , wFFNGateWeightsQ4 :: Vector Word32 -- Q4_0 gate projection
      , wFFNUpWeightsQ4 :: Vector Word32   -- Q4_0 up projection
      , wFFNDownWeightsQ4 :: Vector Word32 -- Q4_0 down projection
      }
  deriving (Show, Eq)

-- | Transformer layer weights
-- CPU vectors for reference + GPU tensors for fast inference
data TransformerLayer dtype = TransformerLayer
  { -- Normalization weights (always FP32)
    tlAttnNormWeights :: Vector Float          -- [hidden_dim] - Pre-attention norm
  , tlQNormWeights :: Maybe (Vector Float)     -- [num_heads * head_dim] - QK-Norm for Q (Gemma 3)
  , tlKNormWeights :: Maybe (Vector Float)     -- [num_kv_heads * head_dim] - QK-Norm for K (Gemma 3)
  , tlPostAttnNormWeights :: Maybe (Vector Float)  -- [hidden_dim] - Post-attention norm (Gemma 3)
  , tlFFNNormWeights :: Vector Float           -- [hidden_dim] - Pre-FFN norm
  , tlPostFFNNormWeights :: Maybe (Vector Float)   -- [hidden_dim] - Post-FFN norm (Gemma 3)

  -- Projection weights (FP32 or Q4)
  , tlWeights :: TransformerWeights
  -- GPU tensors for Attention (uploaded once for performance!)
  , tlAttnNormTensor :: Tensor dtype  -- Pre-attention RMSNorm on GPU
  , tlPostAttnNormTensor :: Maybe (Tensor dtype)  -- Post-attention norm on GPU (Gemma 3)
  , tlAttnQTensor :: Tensor dtype  -- Q projection weights on GPU
  , tlAttnKTensor :: Tensor dtype  -- K projection weights on GPU
  , tlAttnVTensor :: Tensor dtype  -- V projection weights on GPU
  , tlQNormTensor :: Maybe (Tensor dtype)  -- QK-Norm for Q on GPU (Gemma 3)
  , tlKNormTensor :: Maybe (Tensor dtype)  -- QK-Norm for K on GPU (Gemma 3)
  , tlAttnOutTensor :: Tensor dtype  -- Output projection on GPU
  -- GPU tensors for FFN (uploaded once for performance!)
  , tlFFNNormTensor :: Tensor dtype  -- Pre-FFN RMSNorm on GPU (pre_feedforward_layernorm in Gemma 3)
  , tlFFNGateTensor :: Tensor dtype  -- Gate projection on GPU
  , tlFFNUpTensor :: Tensor dtype  -- Up projection on GPU
  , tlFFNDownTensor :: Tensor dtype  -- Down projection on GPU
  , tlPostFFNNormTensor :: Maybe (Tensor dtype)  -- Post-FFN norm on GPU (Gemma 3)
  -- Pre-compiled shaders for Attention
  , tlRMSNormAttnShader :: KernelCode          -- Pre-attention RMSNorm shader
  , tlQKVProjectionShader :: KernelCode        -- Q/K/V projection shader (3-in-1!)
  , tlQNormShader :: KernelCode                -- QK-Norm shader (Gemma 3)
  , tlRoPEShader :: KernelCode                 -- RoPE shader
  , tlAttentionScoresShader :: KernelCode      -- Attention scores shader (Q@K^T + softmax)
  , tlAttentionOutputShader :: KernelCode      -- Attention output shader (scores@V)
  , tlOutputProjectionShader :: KernelCode     -- Output projection shader
  , tlResidualAddShader :: KernelCode          -- Residual addition shader
  , tlPostAttnNormShader :: Maybe KernelCode   -- Post-attention RMSNorm shader (Gemma 3)
  -- Pre-compiled shaders for FFN
  , tlRMSNormGateUpShader :: KernelCode        -- Triple-fused shader
  , tlGELUMultiplyShader :: KernelCode         -- GELU+Multiply fused shader
  , tlLinearDownShader :: KernelCode           -- Linear down projection shader
  , tlPostFFNNormShader :: Maybe KernelCode    -- Post-FFN RMSNorm shader (Gemma 3)
  , tlFFNOutputFusedShader :: Maybe KernelCode -- Phase 3.1: LinearDown+Residual+Norm mega-fusion
  , tlAttentionPostFusedShader :: Maybe KernelCode -- Phase 3.2: OutProj+Residual+Norm mega-fusion
  , tlAttentionCoreFusedShader :: Maybe KernelCode -- Phase 4.1: Scores+Softmax+Output mega-fusion
  -- GPU-resident KV cache
  , tlKVCacheK :: Tensor dtype  -- K cache [maxSeqLen * numKVHeads * headDim]
  , tlKVCacheV :: Tensor dtype  -- V cache [maxSeqLen * numKVHeads * headDim]
  , tlAppendCacheShader :: KernelCode          -- Cache append shader
  -- Persistent intermediate buffers (pre-allocated, reused across tokens)
  , tlXNorm1Buffer :: Tensor dtype  -- After pre-attention RMSNorm [hiddenDim]
  , tlQBuffer :: Tensor dtype  -- Q projection output [qSize]
  , tlKBuffer :: Tensor dtype  -- K projection output [kvSize]
  , tlVBuffer :: Tensor dtype  -- V projection output [kvSize]
  , tlQNormBuffer :: Tensor dtype  -- Q after QK-Norm [qSize]
  , tlKNormBuffer :: Tensor dtype  -- K after QK-Norm [kvSize]
  , tlQRopeBuffer :: Tensor dtype  -- Q after RoPE [qSize]
  , tlKRopeBuffer :: Tensor dtype  -- K after RoPE [kvSize]
  , tlScoresBuffer :: Tensor dtype  -- Attention scores [numHeads * maxSeqLen]
  , tlAttnOutBuffer :: Tensor dtype  -- Attention output [qSize]
  , tlAttnProjBuffer :: Tensor dtype  -- After output projection [hiddenDim]
  , tlPostAttnNormBuffer :: Tensor dtype  -- After post-attention norm [hiddenDim] (Gemma 3)
  , tlAfterAttnBuffer :: Tensor dtype  -- After attn residual [hiddenDim]
  , tlGateBuffer :: Tensor dtype  -- FFN gate output [ffnDim]
  , tlUpBuffer :: Tensor dtype  -- FFN up output [ffnDim]
  , tlGeluUpBuffer :: Tensor dtype  -- After GELU*up [ffnDim]
  , tlDownBuffer :: Tensor dtype  -- FFN down output [hiddenDim]
  , tlPostFFNNormBuffer :: Tensor dtype  -- After post-FFN norm [hiddenDim] (Gemma 3)
  , tlOutputBuffer :: Tensor dtype  -- Final output [hiddenDim]
  -- Q4 quantized weight tensors (consolidated to reduce buffer count)
  , tlQ4PackedWeights :: Maybe (Tensor dtype)  -- All 7 packed weights concatenated
  , tlQ4ScalesWeights :: Maybe (Tensor dtype)  -- All 7 scales concatenated
  , tlQ4Offsets :: Maybe LayerQ4Offsets        -- Offsets into consolidated tensors
  -- Q4 shaders (optional, for Q4 models)
  , tlRMSNormLinearQ4Shader :: Maybe KernelCode     -- RMSNorm + Q4 Linear
  , tlRMSNormGateUpQ4Shader :: Maybe KernelCode     -- RMSNorm + Q4 Gate + Q4 Up
  , tlQKVProjectionQ4Shader :: Maybe KernelCode     -- Q4 QKV projection
  , tlOutputProjectionQ4Shader :: Maybe KernelCode  -- Q4 output projection (qSize → hiddenDim)
  , tlDownProjectionQ4Shader :: Maybe KernelCode    -- Q4 down projection (ffnDim → hiddenDim)
  }

-- | Run a complete transformer block
--
-- Parameters:
-- - input: hidden states [hidden_dim]
-- - layer: transformer layer weights
-- - position: current sequence position for RoPE
-- - numHeads: number of attention heads (e.g., 16)
-- - numKVHeads: number of key/value heads (e.g., 8 for GQA)
-- - headDim: dimension per head (e.g., 128)
-- - hiddenDim: hidden dimension (e.g., 2048)
-- - ffnDim: feedforward dimension (e.g., 5504)
-- - windowSize: optional sliding window size (Just n for local, Nothing for global)
-- - ropeBase: RoPE base frequency (10000.0 for local, 1000000.0 for global)
--
-- Returns: output hidden states [hidden_dim]
runTransformerBlock :: Vector Float       -- Input [hidden_dim]
                    -> TransformerLayer dtype   -- Layer weights
                    -> Int                -- position
                    -> Int                -- numHeads
                    -> Int                -- numKVHeads
                    -> Int                -- headDim
                    -> Int                -- hiddenDim
                    -> Int                -- ffnDim
                    -> Maybe Int          -- windowSize
                    -> Float              -- ropeBase
                    -> ContT r IO (Vector Float)  -- Output [hidden_dim]
runTransformerBlock input layer position numHeads numKVHeads headDim hiddenDim ffnDim windowSize ropeBase = do
  -- Create shared GPU context for all operations
  ctx <- createContext

  -- Step 1: Pre-attention RMSNorm
  xNorm1 <- runRMSNormWithContext ctx input (tlAttnNormWeights layer)

  -- Step 2: Q, K, V projections (FP32 or Q4)
  let qSize = numHeads * headDim
      kvSize = numKVHeads * headDim

  (q, k, v) <- case tlWeights layer of
    FP32Weights{..} -> do
      q <- runLinearWithContext ctx wAttnQWeights xNorm1 qSize hiddenDim
      k <- runLinearWithContext ctx wAttnKWeights xNorm1 kvSize hiddenDim
      v <- runLinearWithContext ctx wAttnVWeights xNorm1 kvSize hiddenDim
      pure (q, k, v)
    Q4Weights{..} -> do
      q <- runLinearQ4 wAttnQWeightsQ4 xNorm1 qSize hiddenDim False 30.0
      k <- runLinearQ4 wAttnKWeightsQ4 xNorm1 kvSize hiddenDim False 30.0
      v <- runLinearQ4 wAttnVWeightsQ4 xNorm1 kvSize hiddenDim False 30.0
      pure (q, k, v)

  -- Step 2.5: Optional QK-Norm (Gemma 3)
  -- Apply RMSNorm to Q and K after projection, before RoPE
  qNorm <- case tlQNormWeights layer of
    Just weights -> runRMSNormWithContext ctx q weights
    Nothing -> pure q
  kNorm <- case tlKNormWeights layer of
    Just weights -> runRMSNormWithContext ctx k weights
    Nothing -> pure k

  -- Step 3: Apply RoPE to Q and K
  qRot <- runRoPEWithContext ctx qNorm position headDim numHeads ropeBase
  kRot <- runRoPEWithContext ctx kNorm position headDim numKVHeads ropeBase

  -- Step 4: Expand K/V heads for GQA (Grouped Query Attention)
  -- If numQHeads > numKVHeads, replicate each K/V head to match Q heads
  let kExpanded = if numHeads > numKVHeads
                  then expandKVHeads kRot numHeads numKVHeads headDim
                  else kRot
      vExpanded = if numHeads > numKVHeads
                  then expandKVHeads v numHeads numKVHeads headDim
                  else v

  -- Step 5: Multi-head Attention
  -- Note: For simplicity, treating all heads as one large attention
  -- TODO: Handle multi-head properly with head splitting/concatenation
  attnOut <- runAttentionWithContext ctx qRot kExpanded vExpanded 1 (numHeads * headDim) windowSize

  -- Step 6: Output projection (FP32 or Q4)
  attnOutProj <- case tlWeights layer of
    FP32Weights{..} -> runLinearWithContext ctx wAttnOutWeights attnOut hiddenDim qSize
    Q4Weights{..} -> runLinearQ4 wAttnOutWeightsQ4 attnOut hiddenDim qSize False 30.0

  -- Step 6.5: Optional post-attention normalization (Gemma 3)
  attnNormed <- case tlPostAttnNormWeights layer of
    Just weights -> runRMSNormWithContext ctx attnOutProj weights
    Nothing -> pure attnOutProj

  -- Step 7: Residual connection 1
  x <- runElementwiseAddWithContext ctx input attnNormed

  -- Step 8: Pre-MLP RMSNorm
  xNorm2 <- runRMSNormWithContext ctx x (tlFFNNormWeights layer)

  -- Step 9: GeGLU MLP (FP32 or Q4)
  mlpOut <- case tlWeights layer of
    FP32Weights{..} -> runGeGLUWithContext ctx xNorm2
                         wFFNGateWeights
                         wFFNUpWeights
                         wFFNDownWeights
                         hiddenDim
                         ffnDim
    Q4Weights{..} -> do
      -- Q4 GeGLU: gate, up, down projections
      gate <- runLinearQ4 wFFNGateWeightsQ4 xNorm2 ffnDim hiddenDim False 30.0
      up <- runLinearQ4 wFFNUpWeightsQ4 xNorm2 ffnDim hiddenDim False 30.0
      -- GELU activation and element-wise multiply
      let geluUp = V.zipWith (\g u -> gelu g * u) gate up
      -- Down projection
      runLinearQ4 wFFNDownWeightsQ4 geluUp hiddenDim ffnDim False 30.0

  -- Step 9.5: Optional post-FFN normalization (Gemma 3)
  mlpNormed <- case tlPostFFNNormWeights layer of
    Just weights -> runRMSNormWithContext ctx mlpOut weights
    Nothing -> pure mlpOut

  -- Step 10: Residual connection 2
  output <- runElementwiseAddWithContext ctx x mlpNormed

  pure output
  where
    -- GELU activation function
    gelu :: Float -> Float
    gelu x = 0.5 * x * (1.0 + tanh (sqrt (2.0 / pi) * (x + 0.044715 * x * x * x)))

-- | Run a transformer block with KV-cache for autoregressive generation
--
-- This is optimized for single-token inference during generation.
-- The cache stores previously computed K/V tensors to avoid recomputation.
--
-- Parameters: Same as runTransformerBlock, plus:
-- - cache: Current KV cache for this layer
--
-- Returns: (output hidden states, updated cache)
runTransformerBlockCached :: Vector Float       -- Input [hidden_dim] (single token)
                          -> TransformerLayer dtype   -- Layer weights
                          -> LayerKVCache       -- Current cache
                          -> Int                -- position
                          -> Int                -- numHeads
                          -> Int                -- numKVHeads
                          -> Int                -- headDim
                          -> Int                -- hiddenDim
                          -> Int                -- ffnDim
                          -> Maybe Int          -- windowSize
                          -> Float              -- ropeBase
                          -> ContT r IO (Vector Float, LayerKVCache)
runTransformerBlockCached input layer cache position numHeads numKVHeads headDim hiddenDim ffnDim windowSize ropeBase = do
  -- Create shared GPU context
  ctx <- createContext

  -- Step 1: Pre-attention RMSNorm
  xNorm1 <- runRMSNormWithContext ctx input (tlAttnNormWeights layer)

  -- DEBUG: Compare with PyTorch's pre-attention RMSNorm output
  liftIO $ debugPrint $ "  DEBUG xNorm1 (pre-attn RMSNorm) first 10: " ++ show (V.take 10 xNorm1)

  -- Step 2: Q, K, V projections (only for current token, FP32 or Q4)
  let qSize = numHeads * headDim
      kvSize = numKVHeads * headDim

  (q, k, v) <- case tlWeights layer of
    FP32Weights{..} -> do
      q <- runLinearWithContext ctx wAttnQWeights xNorm1 qSize hiddenDim
      k <- runLinearWithContext ctx wAttnKWeights xNorm1 kvSize hiddenDim
      v <- runLinearWithContext ctx wAttnVWeights xNorm1 kvSize hiddenDim
      pure (q, k, v)
    Q4Weights{..} -> do
      q <- runLinearQ4 wAttnQWeightsQ4 xNorm1 qSize hiddenDim False 30.0
      k <- runLinearQ4 wAttnKWeightsQ4 xNorm1 kvSize hiddenDim False 30.0
      v <- runLinearQ4 wAttnVWeightsQ4 xNorm1 kvSize hiddenDim False 30.0
      pure (q, k, v)

  -- Step 2.5: Optional QK-Norm (Gemma 3)
  qNorm <- case tlQNormWeights layer of
    Just weights -> runRMSNormWithContext ctx q weights
    Nothing -> pure q
  kNorm <- case tlKNormWeights layer of
    Just weights -> runRMSNormWithContext ctx k weights
    Nothing -> pure k

  -- Step 3: Apply RoPE to Q and K
  qRot <- runRoPEWithContext ctx qNorm position headDim numHeads ropeBase
  kRot <- runRoPEWithContext ctx kNorm position headDim numKVHeads ropeBase

  -- Step 4: For GQA, expand K/V to match Q's head count before caching
  -- Cache stores expanded K/V [numHeads * headDim] to simplify attention computation
  -- TODO: For memory efficiency, could store unexpanded and expand during attention
  let kExpanded = if numHeads > numKVHeads
                  then expandKVHeads kRot numHeads numKVHeads headDim
                  else kRot
      vExpanded = if numHeads > numKVHeads
                  then expandKVHeads v numHeads numKVHeads headDim
                  else v

  -- Step 5: Cached Multi-head Attention
  -- This appends new K/V to cache and computes attention with all cached K/V
  -- Cache stores expanded K/V with dimension [numHeads * headDim]
  -- Q also has dimension [numHeads * headDim]
  (attnOut, updatedCache) <- runAttentionCachedWithContext ctx
                               qRot
                               cache
                               kExpanded  -- Expanded to match Q
                               vExpanded  -- Expanded to match Q
                               (numHeads * headDim)  -- Full dimension after expansion
                               windowSize

  -- Step 6: Output projection (FP32 or Q4)
  attnOutProj <- case tlWeights layer of
    FP32Weights{..} -> runLinearWithContext ctx wAttnOutWeights attnOut hiddenDim qSize
    Q4Weights{..} -> runLinearQ4 wAttnOutWeightsQ4 attnOut hiddenDim qSize False 30.0

  -- Step 6.5: Optional post-attention normalization (Gemma 3)
  attnNormed <- case tlPostAttnNormWeights layer of
    Just weights -> runRMSNormWithContext ctx attnOutProj weights
    Nothing -> pure attnOutProj

  -- Step 7: Residual connection 1
  x <- runElementwiseAddWithContext ctx input attnNormed

  -- Step 8: Pre-MLP RMSNorm
  xNorm2 <- runRMSNormWithContext ctx x (tlFFNNormWeights layer)

  -- Step 9: GeGLU MLP (FP32 or Q4)
  mlpOut <- case tlWeights layer of
    FP32Weights{..} -> runGeGLUWithContext ctx xNorm2
                         wFFNGateWeights
                         wFFNUpWeights
                         wFFNDownWeights
                         hiddenDim
                         ffnDim
    Q4Weights{..} -> do
      -- Q4 GeGLU: gate, up, down projections
      gate <- runLinearQ4 wFFNGateWeightsQ4 xNorm2 ffnDim hiddenDim False 30.0
      up <- runLinearQ4 wFFNUpWeightsQ4 xNorm2 ffnDim hiddenDim False 30.0
      -- GELU activation and element-wise multiply
      let geluUp = V.zipWith (\g u -> gelu g * u) gate up
      -- Down projection
      runLinearQ4 wFFNDownWeightsQ4 geluUp hiddenDim ffnDim False 30.0

  -- Step 9.5: Optional post-FFN normalization (Gemma 3)
  mlpNormed <- case tlPostFFNNormWeights layer of
    Just weights -> runRMSNormWithContext ctx mlpOut weights
    Nothing -> pure mlpOut

  -- Step 10: Residual connection 2
  output <- runElementwiseAddWithContext ctx x mlpNormed

  pure (output, updatedCache)
  where
    -- GELU activation function
    gelu :: Float -> Float
    gelu x = 0.5 * x * (1.0 + tanh (sqrt (2.0 / pi) * (x + 0.044715 * x * x * x)))
