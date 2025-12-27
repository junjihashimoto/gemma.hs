{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

{-|
Module: Gemma.Model
Description: Complete Gemma 3 (1B) Model

This module implements the full Gemma 3 transformer model with:
- Token embeddings
- 24 transformer layers
- Final RMSNorm
- LM head for token prediction

All weights are loaded to GPU memory once and reused across inference calls.
-}

module Gemma.Model
  ( GemmaConfig(..)
  , GemmaModel(..)
  , gemma2_2BConfig
  , gemma1BConfig
  , gemma3_1BConfig
  , tinyGemmaConfig
  , loadGemmaModel
  , runGemmaInference
  , runGemmaInferenceCached
  , runGemmaInferenceCachedWithLayer0  -- For TDD testing
  ) where

import Graphics.WebGPU.Dawn.ContT
import Graphics.WebGPU.Dawn.Types (Context, KernelCode)
import qualified Graphics.WebGPU.Dawn.Context as Ctx
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import Control.Monad (forM, foldM, when)
import Data.Int (Int32)
import qualified Data.Text as T
import Data.Text (Text)
import System.Environment (lookupEnv)

import Gemma.SafeTensors
import Gemma.Layers.Embedding (runEmbedding, runEmbeddingGPU, embeddingShader)
import Gemma.Layers.Scale (runScaleVectorGPU)
import Gemma.Layers.RMSNorm (runRMSNormWithContext, runRMSNormGPU, runRMSNormPreloadedGPU, runRMSNormLinearFusedGPU, runRMSNormLinearFusedPreloadedGPU, rmsNormLinearFusedShader, rmsNormShader)
import Gemma.Layers.LinearDSL (runLinearWithContext, runLinearGPU, linearShader)  -- Phase 4: Using DSL-based Linear instead of string-based Linear
import Gemma.Layers.MLP (rmsNormGateUpFusedShader, geluMultiplyFusedShader, residualAddShader, ffnOutputFusedShader)
import Gemma.Layers.AttentionGPU (qkvProjectionShader, qkNormShader, ropeShader, attentionScoresShader, attentionOutputShader, outputProjectionShader, appendKVCacheShader, attentionPostFusedShader, attentionCoreFusedShader)
import Gemma.Layers.LinearQ4Fused (qkvProjectionQ4Shader, outputProjectionQ4Shader, rmsNormGateUpQ4Shader, rmsNormLinearQ4Shader,
                                    outputProjectionQ4ConsolidatedShader, rmsNormGateUpQ4ConsolidatedShader, rmsNormLinearQ4ConsolidatedShader)
import qualified Graphics.WebGPU.Dawn.Kernel as K
import Gemma.Layers.TransformerBlock (TransformerLayer(..), LayerQ4Offsets(..), runTransformerBlock, runTransformerBlockCached)
import Gemma.Layers.TransformerBlockGPU (runTransformerBlockCachedGPU)
import Gemma.KVCache (KVCache(..), initKVCache, cacheLength)
import qualified Data.Vector as BV
import qualified Graphics.WebGPU.Dawn.Tensor as T
import Graphics.WebGPU.Dawn.Types (Tensor, Shape(..), NumType(..))
import qualified Data.Word as Word
import Data.Bits (shiftR, shiftL, (.&.), (.|.))
import Unsafe.Coerce (unsafeCoerce)

-- | Helper function to conditionally print debug messages
debugPrint :: String -> IO ()
debugPrint msg = do
  debug <- lookupEnv "DEBUG"
  case debug of
    Just "1" -> putStrLn msg
    Just "true" -> putStrLn msg
    _ -> return ()

-- | Convert Float32 to Float16 (IEEE 754 half precision)
-- This is the inverse of halfToFloat from SafeTensors.hs
floatToHalf :: Float -> Word.Word16
floatToHalf f =
  let w32 = unsafeCoerce f :: Word.Word32
      sign = (w32 `shiftR` 16) .&. 0x8000
      exp32 = (w32 `shiftR` 23) .&. 0xFF
      mantissa32 = w32 .&. 0x7FFFFF

      -- Handle special cases
      (exp16, mantissa16)
        | exp32 == 0xFF = (0x1F, if mantissa32 == 0 then 0 else 1) -- Inf or NaN
        | exp32 == 0 = (0, 0) -- Zero or denorm → flush to zero
        | otherwise =
            let exp_adj = fromIntegral exp32 - 127 + 15 :: Int
            in if exp_adj <= 0
               then (0, 0) -- Underflow → flush to zero
               else if exp_adj >= 0x1F
                    then (0x1F, 0) -- Overflow → Inf
                    else (fromIntegral exp_adj, mantissa32 `shiftR` 13)
  in fromIntegral $ sign .|. (exp16 `shiftL` 10) .|. (mantissa16 .&. 0x3FF)

-- | Upload FP16 tensor from Float32 data
-- Converts each Float to FP16 (Word16) and uploads with F16 type
createTensorFP16 :: Context -> Shape -> Vector Float -> IO (Tensor dtype)
createTensorFP16 ctx shape floatData = do
  let fp16Data = V.map floatToHalf floatData :: Vector Word.Word16
  T.createTensorWithDataPacked ctx shape Graphics.WebGPU.Dawn.Types.F16 fp16Data

-- | Gemma model configuration
data GemmaConfig = GemmaConfig
  { gcVocabSize :: Int      -- ^ Vocabulary size (e.g., 256000)
  , gcHiddenDim :: Int      -- ^ Hidden dimension (e.g., 2048)
  , gcNumLayers :: Int      -- ^ Number of transformer layers (e.g., 24)
  , gcNumHeads :: Int       -- ^ Number of attention heads (e.g., 16)
  , gcNumKVHeads :: Int     -- ^ Number of key/value heads for GQA (e.g., 8)
  , gcHeadDim :: Int        -- ^ Dimension per head (e.g., 128)
  , gcFFNDim :: Int         -- ^ Feedforward dimension (e.g., 5504)
  , gcRopeBase :: Float     -- ^ RoPE base frequency (e.g., 10000.0)
  -- Performance optimizations
  , gcUseFP16 :: Bool                -- ^ Use FP16 for 2x memory bandwidth (default: False)
  , gcUseVec4 :: Bool                -- ^ Use vec4 SIMD for 4x additional speedup (default: False)
  , gcUseFusion :: Bool              -- ^ Phase 3: Use kernel fusion for dispatch reduction (default: False)
  -- Gemma 3 specific features
  , gcUseQKNorm :: Bool              -- ^ Use Query-Key normalization
  , gcUsePostAttnNorm :: Bool        -- ^ Use post-attention normalization
  , gcUsePostFFNNorm :: Bool         -- ^ Use post-FFN normalization
  , gcUseSlidingWindow :: Bool       -- ^ Use sliding window attention
  , gcSlidingWindowSize :: Int       -- ^ Window size for local attention
  , gcLocalRopeScaling :: Float      -- ^ RoPE scaling for local layers
  , gcGlobalRopeScaling :: Float     -- ^ RoPE scaling for global layers
  , gcQueryHeadDimNormalize :: Bool  -- ^ Use query head dim normalization
  , gcUseZeroCenteredRMSNorm :: Bool -- ^ Use zero-centered RMSNorm weights (1 + weight)
  } deriving (Show, Eq)

-- | Default configuration for Gemma 1B
gemma1BConfig :: GemmaConfig
gemma1BConfig = GemmaConfig
  { gcVocabSize = 256000
  , gcHiddenDim = 2048
  , gcNumLayers = 24
  , gcNumHeads = 16
  , gcNumKVHeads = 8
  , gcHeadDim = 128
  , gcFFNDim = 5504
  , gcRopeBase = 10000.0
  -- Performance optimizations
  , gcUseFP16 = False  -- Set to True for 2x speedup (12 TPS → ~24 TPS)
  , gcUseVec4 = False  -- Set to True for 4x additional speedup (requires FP16)
  , gcUseFusion = False -- Set to True for Phase 3 kernel fusion (1.5-2x additional speedup)
  -- Gemma 1 doesn't use these features
  , gcUseQKNorm = False
  , gcUsePostAttnNorm = False
  , gcUsePostFFNNorm = False
  , gcUseSlidingWindow = False
  , gcSlidingWindowSize = 0
  , gcLocalRopeScaling = 1.0
  , gcGlobalRopeScaling = 1.0
  , gcQueryHeadDimNormalize = False
  , gcUseZeroCenteredRMSNorm = False
  }

-- | Configuration for Gemma 2 2B
gemma2_2BConfig :: GemmaConfig
gemma2_2BConfig = GemmaConfig
  { gcVocabSize = 256000
  , gcHiddenDim = 2304      -- hidden_size
  , gcNumLayers = 26        -- num_hidden_layers
  , gcNumHeads = 8          -- num_attention_heads
  , gcNumKVHeads = 4        -- num_key_value_heads
  , gcHeadDim = 256         -- head_dim
  , gcFFNDim = 9216         -- intermediate_size
  , gcRopeBase = 10000.0    -- rope_theta
  -- Performance optimizations
  , gcUseFP16 = False  -- Set to True for 2x speedup
  , gcUseVec4 = False  -- Set to True for 4x additional speedup (requires FP16)
  , gcUseFusion = False -- Set to True for Phase 3 kernel fusion (1.5-2x additional speedup)
  -- Gemma 2 features (similar to Gemma 3 but may differ)
  , gcUseQKNorm = False
  , gcUsePostAttnNorm = False
  , gcUsePostFFNNorm = False
  , gcUseSlidingWindow = False
  , gcSlidingWindowSize = 0
  , gcLocalRopeScaling = 1.0
  , gcGlobalRopeScaling = 1.0
  , gcQueryHeadDimNormalize = False
  , gcUseZeroCenteredRMSNorm = False
  }

-- | Configuration for tiny synthetic Gemma (for testing)
tinyGemmaConfig :: GemmaConfig
tinyGemmaConfig = GemmaConfig
  { gcVocabSize = 1000
  , gcHiddenDim = 128
  , gcNumLayers = 2
  , gcNumHeads = 4
  , gcNumKVHeads = 2
  , gcHeadDim = 32
  , gcFFNDim = 384
  , gcRopeBase = 10000.0
  -- Performance optimizations
  , gcUseFP16 = False  -- Set to True for 2x speedup
  , gcUseVec4 = False  -- Set to True for 4x additional speedup (requires FP16)
  , gcUseFusion = False -- Set to True for Phase 3 kernel fusion (1.5-2x additional speedup)
  -- Tiny model uses Gemma 1 architecture
  , gcUseQKNorm = False
  , gcUsePostAttnNorm = False
  , gcUsePostFFNNorm = False
  , gcUseSlidingWindow = False
  , gcSlidingWindowSize = 0
  , gcLocalRopeScaling = 1.0
  , gcGlobalRopeScaling = 1.0
  , gcQueryHeadDimNormalize = False
  , gcUseZeroCenteredRMSNorm = False
  }

-- | Configuration for Gemma 3 1B
gemma3_1BConfig :: GemmaConfig
gemma3_1BConfig = GemmaConfig
  { gcVocabSize = 262144        -- vocabulary_size
  , gcHiddenDim = 1152          -- hidden_dim
  , gcNumLayers = 26            -- num_layers
  , gcNumHeads = 4              -- num_query_heads
  , gcNumKVHeads = 1            -- num_key_value_heads (extreme GQA 4:1)
  , gcHeadDim = 256             -- head_dim
  , gcFFNDim = 6912             -- intermediate_dim
  , gcRopeBase = 1000000.0      -- rope_theta (official Gemma 3: 1000000, not 10000!)
  -- Performance optimizations
  , gcUseFP16 = False  -- PRECISION ALIGNMENT: Match PyTorch FP32 reference (CLAUDE.md line 15)
  , gcUseVec4 = False  -- Requires FP16
  , gcUseFusion = False  -- DISABLED: Test unfused path first to find Bug #2
  -- Gemma 3 features enabled
  , gcUseQKNorm = True          -- use_query_key_norm
  , gcUsePostAttnNorm = True    -- use_post_attention_norm
  , gcUsePostFFNNorm = True     -- use_post_ffw_norm
  , gcUseSlidingWindow = True   -- use_sliding_window_attention
  , gcSlidingWindowSize = 512   -- sliding_window_size
  , gcLocalRopeScaling = 1.0    -- local_rope_scaling_factor
  , gcGlobalRopeScaling = 1.0   -- global_rope_scaling_factor
  , gcQueryHeadDimNormalize = True  -- query_head_dim_normalize
  , gcUseZeroCenteredRMSNorm = True -- Gemma 3 uses ZERO-CENTERED RMSNorm: (x/rms) * (1+weight), NOT standard (x/rms)*weight
  }

-- | Complete Gemma model with all weights
--
-- ALL weights AND shaders are pre-uploaded/compiled during model loading!
-- This eliminates repeated weight uploads and shader compilations.
data GemmaModel dtype = GemmaModel
  { gmConfig :: GemmaConfig
  , gmContext :: Context                  -- ^ Reusable GPU context
  , gmEmbeddings :: Vector Float          -- ^ Token embeddings [vocab_size, hidden_dim]
  , gmLayers :: [TransformerLayer dtype]        -- ^ Transformer layers (24 for Gemma 1B)
  , gmFinalNormWeights :: Vector Float    -- ^ Final RMSNorm weights [hidden_dim]
  , gmLMHeadWeights :: Vector Float       -- ^ LM head weights [vocab_size, hidden_dim]
  -- GPU-resident weights (uploaded once!)
  , gmEmbeddingTensor :: Tensor dtype  -- ^ Embedding table on GPU
  , gmFinalNormTensor :: Tensor dtype  -- ^ Final RMSNorm weights on GPU
  , gmLMHeadTensor :: Tensor dtype  -- ^ LM head weights on GPU
  -- Pre-compiled shaders (compiled once!)
  , gmEmbeddingShader :: KernelCode       -- ^ Embedding shader
  , gmFinalRMSNormLinearShader :: KernelCode  -- ^ Final RMSNorm+Linear fused shader
  }

-- | Load Gemma model from SafeTensors file
--
-- Loads all model weights from a .safetensors file and creates a GemmaModel.
-- The model follows the HuggingFace transformers naming convention:
--
-- - model.embed_tokens.weight
-- - model.layers.{i}.input_layernorm.weight
-- - model.layers.{i}.self_attn.q_proj.weight
-- - model.layers.{i}.self_attn.k_proj.weight
-- - model.layers.{i}.self_attn.v_proj.weight
-- - model.layers.{i}.self_attn.o_proj.weight
-- - model.layers.{i}.post_attention_layernorm.weight
-- - model.layers.{i}.mlp.gate_proj.weight
-- - model.layers.{i}.mlp.up_proj.weight
-- - model.layers.{i}.mlp.down_proj.weight
-- - model.norm.weight
-- - lm_head.weight
loadGemmaModel :: FilePath -> GemmaConfig -> IO (GemmaModel dtype)
loadGemmaModel modelPath config = do
  -- Load safetensors file
  st <- loadSafeTensors modelPath

  -- Load embeddings
  embedData <- getTensor st "model.embed_tokens.weight"

  -- Load all transformer layers
  layers <- forM [0 .. gcNumLayers config - 1] $ \i -> do
    let prefix = T.pack $ "model.layers." ++ show i

    -- Required weights (all Gemma versions)
    attnNorm <- getTensor st (prefix <> ".input_layernorm.weight")

    -- Load Q4 quantized weights if available
    -- We load BOTH packed Q4 (for GPU) and dequantized FP32 (for CPU fallback/debugging)
    let qName = prefix <> ".self_attn.q_proj.weight"
        kName = prefix <> ".self_attn.k_proj.weight"
        vName = prefix <> ".self_attn.v_proj.weight"
        outName = prefix <> ".self_attn.o_proj.weight"

    (qWeights, mbQQ4) <- if hasQ4Weight st qName
                         then do
                           dequant <- loadQ4WeightDequantized st qName
                           (packed, scales) <- loadQ4Weight st qName
                           pure (dequant, Just (packed, scales))
                         else do
                           fp32 <- getTensor st qName
                           pure (fp32, Nothing)

    (kWeights, mbKQ4) <- if hasQ4Weight st kName
                         then do
                           dequant <- loadQ4WeightDequantized st kName
                           (packed, scales) <- loadQ4Weight st kName
                           pure (dequant, Just (packed, scales))
                         else do
                           fp32 <- getTensor st kName
                           pure (fp32, Nothing)

    (vWeights, mbVQ4) <- if hasQ4Weight st vName
                         then do
                           dequant <- loadQ4WeightDequantized st vName
                           (packed, scales) <- loadQ4Weight st vName
                           pure (dequant, Just (packed, scales))
                         else do
                           fp32 <- getTensor st vName
                           pure (fp32, Nothing)

    (outWeights, mbOutQ4) <- if hasQ4Weight st outName
                             then do
                               dequant <- loadQ4WeightDequantized st outName
                               (packed, scales) <- loadQ4Weight st outName
                               pure (dequant, Just (packed, scales))
                             else do
                               fp32 <- getTensor st outName
                               pure (fp32, Nothing)

    -- Pre-feedforward norm (Gemma 3: pre_feedforward_layernorm, older: post_attention_layernorm)
    ffnNorm <- if hasTensor st (prefix <> ".pre_feedforward_layernorm.weight")
               then getTensor st (prefix <> ".pre_feedforward_layernorm.weight")
               else getTensor st (prefix <> ".post_attention_layernorm.weight")  -- Fallback for older models

    let gateName = prefix <> ".mlp.gate_proj.weight"
        upName = prefix <> ".mlp.up_proj.weight"
        downName = prefix <> ".mlp.down_proj.weight"

    (gateWeights, mbGateQ4) <- if hasQ4Weight st gateName
                               then do
                                 dequant <- loadQ4WeightDequantized st gateName
                                 (packed, scales) <- loadQ4Weight st gateName
                                 pure (dequant, Just (packed, scales))
                               else do
                                 fp32 <- getTensor st gateName
                                 pure (fp32, Nothing)

    (upWeights, mbUpQ4) <- if hasQ4Weight st upName
                           then do
                             dequant <- loadQ4WeightDequantized st upName
                             (packed, scales) <- loadQ4Weight st upName
                             pure (dequant, Just (packed, scales))
                           else do
                             fp32 <- getTensor st upName
                             pure (fp32, Nothing)

    (downWeights, mbDownQ4) <- if hasQ4Weight st downName
                               then do
                                 dequant <- loadQ4WeightDequantized st downName
                                 (packed, scales) <- loadQ4Weight st downName
                                 pure (dequant, Just (packed, scales))
                               else do
                                 fp32 <- getTensor st downName
                                 pure (fp32, Nothing)

    -- Optional weights (Gemma 3 only)
    -- QK-Norm weights for query and key normalization
    let qNormName = prefix <> ".self_attn.q_norm.weight"
        kNormName = prefix <> ".self_attn.k_norm.weight"
    qNormWeights <- if hasTensor st qNormName
                    then Just <$> getTensor st qNormName
                    else pure Nothing
    kNormWeights <- if hasTensor st kNormName
                    then Just <$> getTensor st kNormName
                    else pure Nothing

    -- Post-attention normalization (Gemma 3)
    -- CORRECTED: post_attention_norm.weight is the ACTUAL post-attn norm (zero-centered)
    -- post_attention_layernorm.weight is used as pre-FFN norm (see line 330)
    -- These are DIFFERENT weights with different purposes!
    let postAttnNormName = prefix <> ".post_attention_norm.weight"  -- Primary: zero-centered norm
        postAttnNormNameOld = prefix <> ".post_attention_layernorm.weight"  -- Fallback
    postAttnNormWeights <- if gcUsePostAttnNorm config
                           then if hasTensor st postAttnNormName
                                then Just <$> getTensor st postAttnNormName  -- Prefer _norm over _layernorm!
                                else if hasTensor st postAttnNormNameOld
                                     then Just <$> getTensor st postAttnNormNameOld
                                     else pure Nothing
                           else pure Nothing

    -- Post-FFN normalization (Gemma 3)
    -- Only load if explicitly enabled in config
    let postFFNNormName = prefix <> ".post_feedforward_layernorm.weight"
    postFFNNormWeights <- if gcUsePostFFNNorm config
                          then if hasTensor st postFFNNormName
                               then Just <$> getTensor st postFFNNormName
                               else pure Nothing
                          else pure Nothing

    -- Create placeholder GPU tensors (will be uploaded after context creation)
    -- Return both FP32 weights and Q4 packed data (if available)
    pure $ (attnNorm, qWeights, kWeights, vWeights, qNormWeights, kNormWeights,
            outWeights, postAttnNormWeights, ffnNorm, gateWeights, upWeights,
            downWeights, postFFNNormWeights,
            mbQQ4, mbKQ4, mbVQ4, mbOutQ4, mbGateQ4, mbUpQ4, mbDownQ4)

  -- Load final norm and LM head
  finalNorm <- getTensor st "model.norm.weight"
  -- Official models use tied embeddings (no separate lm_head)
  lmHead <- if hasTensor st "lm_head.weight"
            then getTensor st "lm_head.weight"
            else getTensor st "model.embed_tokens.weight"  -- Use tied embeddings

  -- Create a persistent GPU context for the model (CRITICAL for performance!)
  -- This context will be reused across all forward passes instead of
  -- creating/destroying a new one for each token (which is extremely expensive)
  ctx <- Ctx.createContext

  -- Upload all weights to GPU ONCE during loading (massive performance win!)
  -- This eliminates repeated weight uploads on every forward pass
  -- NOTE: We use raw Tensor functions (not ContT) to keep tensors alive permanently
  let hiddenDim = gcHiddenDim config
      ffnDim = gcFFNDim config
      vocabSize = gcVocabSize config
      numHeads = gcNumHeads config
      numKVHeads = gcNumKVHeads config
      headDim = gcHeadDim config
      ropeBase = gcRopeBase config

  -- Pre-compile ALL shaders ONCE (massive performance win!)
  -- This eliminates shader recompilation on every forward pass
  let zeroCentered = gcUseZeroCenteredRMSNorm config
      useFP16 = gcUseFP16 config
      useVec4 = gcUseVec4 config
      useFusion = gcUseFusion config

  -- Compile embedding shader (using FP32 until FP16 bug is fixed)
  embeddingShader' <- K.createKernelCode $ embeddingShader False 1 hiddenDim  -- seqLen=1 for cached inference

  -- Compile final RMSNorm+Linear shader
  -- ALWAYS use FP32 for final output (logits) regardless of gcUseFP16
  -- This prevents FP16/FP32 tensor type mismatch bug (logits need full precision)
  finalRMSNormLinearShader <- K.createKernelCode $ rmsNormLinearFusedShader False hiddenDim vocabSize zeroCentered

  -- === VALIDATE EMBEDDING DATA BEFORE UPLOAD ===
  liftIO $ do
    let sample = V.take 10 embedData
        sampleForToken2 = V.slice (2 * hiddenDim) 10 embedData  -- Token 2 embedding (first 10 dims)
    putStrLn $ "🔍 EMBEDDING DATA CHECK (before GPU upload):"
    putStrLn $ "   Total size: " ++ show (V.length embedData)
    putStrLn $ "   First 10 values: " ++ show (V.toList sample)
    putStrLn $ "   Token ID 2 embedding (first 10 dims): " ++ show (V.toList sampleForToken2)
    putStrLn $ "   Min: " ++ show (V.minimum embedData) ++ ", Max: " ++ show (V.maximum embedData)
    let allZeros = V.all (== 0.0) embedData
    if allZeros
      then putStrLn "   ❌ ERROR: Embedding data is ALL ZEROS before upload!"
      else putStrLn "   ✅ Embedding data has non-zero values"

  -- Upload embedding table (permanent!)
  let embedShape = Shape [vocabSize * hiddenDim]

  liftIO $ putStrLn $ "🔍 Uploading embedding table to GPU: " ++ show (V.length embedData) ++ " elements"
  liftIO $ putStrLn $ "   useFP16=" ++ show useFP16

  -- FIXME: FP16 embedding upload appears to be broken, using FP32 for now
  embeddingTensor <- T.createTensorWithData ctx embedShape embedData
  liftIO $ putStrLn "✅ Embedding table uploaded to GPU (FP32)"

  -- Upload final norm and LM head weights (permanent - no auto-cleanup!)
  let normShape = Shape [hiddenDim]
      lmHeadShape = Shape [vocabSize, hiddenDim]
  finalNormTensor <- T.createTensorWithData ctx normShape finalNorm
  lmHeadTensor <- T.createTensorWithData ctx lmHeadShape lmHead

  -- Compile shared shaders for all layers (same dimensions)
  -- Attention shaders
  let qSize = numHeads * headDim
      kvSize = numKVHeads * headDim
      maxCacheLen = 2048  -- Max sequence length
  rmsNormAttnShader <- K.createKernelCode $ rmsNormShader useFP16 hiddenDim zeroCentered useVec4
  qkvProjShader <- K.createKernelCode $ qkvProjectionShader useFP16 hiddenDim qSize kvSize
  qNormShader' <- K.createKernelCode $ qkNormShader useFP16 numHeads headDim zeroCentered
  kNormShader' <- K.createKernelCode $ qkNormShader useFP16 numKVHeads headDim zeroCentered
  ropeShader' <- K.createKernelCode $ ropeShader useFP16 numHeads headDim ropeBase
  attnScoresShader <- K.createKernelCode $ attentionScoresShader useFP16 numHeads numKVHeads headDim maxCacheLen
  attnOutputShader <- K.createKernelCode $ attentionOutputShader useFP16 numHeads numKVHeads headDim maxCacheLen
  outProjShader <- K.createKernelCode $ outputProjectionShader useFP16 hiddenDim qSize
  residualShader <- K.createKernelCode $ residualAddShader useFP16 hiddenDim useVec4
  postAttnNormShader <- K.createKernelCode $ rmsNormShader useFP16 hiddenDim zeroCentered useVec4

  -- FFN shaders
  -- DIMENSION VALIDATION: Prevent shader dimension mismatches
  let validateFFNDims = do
        -- Gate/Up projection: hiddenDim (in) → ffnDim (out)
        when (hiddenDim <= 0 || ffnDim <= 0) $
          error $ "Invalid FFN dimensions: hiddenDim=" ++ show hiddenDim ++ ", ffnDim=" ++ show ffnDim
        -- Down projection: ffnDim (in) → hiddenDim (out)
        when (ffnDim <= 0 || hiddenDim <= 0) $
          error $ "Invalid down projection dimensions: ffnDim=" ++ show ffnDim ++ ", hiddenDim=" ++ show hiddenDim
  validateFFNDims

  rmsNormGateUpShader <- K.createKernelCode $ rmsNormGateUpFusedShader useFP16 hiddenDim ffnDim zeroCentered useVec4
  geluMultShader <- K.createKernelCode $ geluMultiplyFusedShader useFP16 ffnDim useVec4
  linearDownShader <- K.createKernelCode $ linearShader hiddenDim ffnDim useFP16 useVec4
  postFFNNormShader <- K.createKernelCode $ rmsNormShader useFP16 hiddenDim zeroCentered useVec4

  -- Phase 3.1: FFN Output Fusion shader (LinearDown + Residual + PostNorm)
  ffnOutputFusedShader' <- if useFusion
    then Just <$> K.createKernelCode (ffnOutputFusedShader useFP16 useVec4 hiddenDim ffnDim zeroCentered)
    else pure Nothing

  -- Phase 3.2: Attention Postprocessing Fusion shader (OutProj + Residual + PostAttnNorm)
  attentionPostFusedShader' <- if useFusion
    then Just <$> K.createKernelCode (attentionPostFusedShader useFP16 useVec4 hiddenDim qSize zeroCentered)
    else pure Nothing

  -- Phase 4.1: Attention Core Fusion shader (Scores + Softmax + Output)
  attentionCoreFusedShader' <- if useFusion
    then Just <$> K.createKernelCode (attentionCoreFusedShader useFP16 numHeads numKVHeads headDim maxCacheLen Nothing maxCacheLen)
    else pure Nothing

  -- KV cache shader (shared across all layers)
  appendCacheShader <- K.createKernelCode $ appendKVCacheShader useFP16 maxCacheLen numKVHeads headDim

  -- Upload each layer's weights to GPU (permanent!)
  gpuLayers <- forM (zip [0..] layers) $ \(layerIdx :: Int, (attnNorm, qWeights, kWeights, vWeights, qNormWeights, kNormWeights,
                                outWeights, postAttnNormWeights, ffnNorm, gateWeights, upWeights,
                                downWeights, postFFNNormWeights,
                                mbQQ4, mbKQ4, mbVQ4, mbOutQ4, mbGateQ4, mbUpQ4, mbDownQ4)) -> do
    -- Attention weight shapes
    let attnNormShape = Shape [hiddenDim]
        qShape = Shape [qSize, hiddenDim]
        kShape = Shape [kvSize, hiddenDim]
        vShape = Shape [kvSize, hiddenDim]
        qNormShape = Shape [headDim]
        outShape = Shape [hiddenDim, qSize]

    -- Upload attention weights
    attnNormT <- T.createTensorWithData ctx attnNormShape attnNorm
    qT <- T.createTensorWithData ctx qShape qWeights
    kT <- T.createTensorWithData ctx kShape kWeights
    vT <- T.createTensorWithData ctx vShape vWeights
    outT <- T.createTensorWithData ctx outShape outWeights

    -- Optional QK-Norm (Gemma 3)
    qNormT <- case qNormWeights of
      Just weights -> Just <$> T.createTensorWithData ctx qNormShape weights
      Nothing -> pure Nothing
    kNormT <- case kNormWeights of
      Just weights -> Just <$> T.createTensorWithData ctx qNormShape weights
      Nothing -> pure Nothing

    -- FFN weight shapes
    let ffnNormShape = Shape [hiddenDim]
        gateShape = Shape [ffnDim, hiddenDim]
        upShape = Shape [ffnDim, hiddenDim]
        downShape = Shape [hiddenDim, ffnDim]

    -- Upload FFN weights
    ffnNormT <- T.createTensorWithData ctx ffnNormShape ffnNorm

    -- DEBUG: Check weight size before upload
    debug2 <- lookupEnv "DEBUG"
    when (debug2 == Just "1" && layerIdx == 0) $ do
      putStrLn $ "\n  DEBUG LAYER 0 GATE UPLOAD INFO:"
      putStrLn $ "    Shape: " ++ show gateShape
      putStrLn $ "    Vector length: " ++ show (V.length gateWeights)
      putStrLn $ "    Expected elements: " ++ show (ffnDim * hiddenDim)

    gateT <- T.createTensorWithData ctx gateShape gateWeights
    upT <- T.createTensorWithData ctx upShape upWeights
    downT <- T.createTensorWithData ctx downShape downWeights

    -- Optional post-attention norm (Gemma 3)
    postAttnNormT <- case postAttnNormWeights of
      Just weights -> Just <$> T.createTensorWithData ctx ffnNormShape weights
      Nothing -> pure Nothing

    -- Optional post-FFN norm (Gemma 3)
    postFFNNormT <- case postFFNNormWeights of
      Just weights -> Just <$> T.createTensorWithData ctx ffnNormShape weights
      Nothing -> pure Nothing

    -- Create GPU-resident KV cache tensors (initialized to zeros)
    -- Shape: [maxSeqLen * numKVHeads * headDim]
    let cacheSize = maxCacheLen * kvSize
        emptyCacheData = V.replicate cacheSize (0.0 :: Float)
        cacheShape = Shape [cacheSize]
    kCacheT <- T.createTensorWithData ctx cacheShape emptyCacheData
    vCacheT <- T.createTensorWithData ctx cacheShape emptyCacheData

    -- Pre-allocate persistent intermediate buffers (REUSED across all tokens!)
    -- These buffers eliminate tensor allocation overhead during inference
    -- Buffer type must match useFP16 flag (CLAUDE.md: precision alignment)
    let numType = if useFP16 then Graphics.WebGPU.Dawn.Types.F16 else Graphics.WebGPU.Dawn.Types.F32
    xNorm1Buf <- T.createTensor ctx (Shape [hiddenDim]) numType
    qBuf <- T.createTensor ctx (Shape [qSize]) numType
    kBuf <- T.createTensor ctx (Shape [kvSize]) numType
    vBuf <- T.createTensor ctx (Shape [kvSize]) numType
    qNormBuf <- T.createTensor ctx (Shape [qSize]) numType
    kNormBuf <- T.createTensor ctx (Shape [kvSize]) numType
    qRopeBuf <- T.createTensor ctx (Shape [qSize]) numType
    kRopeBuf <- T.createTensor ctx (Shape [kvSize]) numType
    scoresBuf <- T.createTensor ctx (Shape [numHeads * maxCacheLen]) numType
    attnOutBuf <- T.createTensor ctx (Shape [qSize]) numType
    attnProjBuf <- T.createTensor ctx (Shape [hiddenDim]) numType
    postAttnNormBuf <- T.createTensor ctx (Shape [hiddenDim]) numType
    afterAttnBuf <- T.createTensor ctx (Shape [hiddenDim]) numType
    gateBuf <- T.createTensor ctx (Shape [ffnDim]) numType
    upBuf <- T.createTensor ctx (Shape [ffnDim]) numType
    geluUpBuf <- T.createTensor ctx (Shape [ffnDim]) numType
    downBuf <- T.createTensor ctx (Shape [hiddenDim]) numType
    postFFNNormBuf <- T.createTensor ctx (Shape [hiddenDim]) numType
    outputBuf <- T.createTensor ctx (Shape [hiddenDim]) numType

    -- ========== Q4 GPU CONSOLIDATION ==========
    -- If Q4 weights are available, consolidate them into single tensors and upload to GPU
    -- This enables 4× memory bandwidth reduction (0.5 bytes/weight vs 2 bytes FP16)
    (mbQ4Packed, mbQ4Scales, mbQ4Offsets, mbQ4Shaders) <- case (mbQQ4, mbKQ4, mbVQ4, mbOutQ4, mbGateQ4, mbUpQ4, mbDownQ4) of
      (Just (qP, qS), Just (kP, kS), Just (vP, vS), Just (outP, outS), Just (gateP, gateS), Just (upP, upS), Just (downP, downS)) -> do
        -- All 7 weights are Q4 - consolidate them!
        let qPackedLen = V.length qP
            qScalesLen = V.length qS
            kPackedLen = V.length kP
            kScalesLen = V.length kS
            vPackedLen = V.length vP
            vScalesLen = V.length vS
            outPackedLen = V.length outP
            outScalesLen = V.length outS
            gatePackedLen = V.length gateP
            gateScalesLen = V.length gateS
            upPackedLen = V.length upP
            upScalesLen = V.length upS
            downPackedLen = V.length downP
            downScalesLen = V.length downS

            -- Concatenate all packed weights
            allPacked = qP V.++ kP V.++ vP V.++ outP V.++ gateP V.++ upP V.++ downP
            -- Concatenate all scales
            allScales = qS V.++ kS V.++ vS V.++ outS V.++ gateS V.++ upS V.++ downS

            -- Calculate offsets
            offsets = LayerQ4Offsets
              { q4QPackedOffset = 0
              , q4QPackedSize = qPackedLen
              , q4QScalesOffset = 0
              , q4QScalesSize = qScalesLen
              , q4KPackedOffset = qPackedLen
              , q4KPackedSize = kPackedLen
              , q4KScalesOffset = qScalesLen
              , q4KScalesSize = kScalesLen
              , q4VPackedOffset = qPackedLen + kPackedLen
              , q4VPackedSize = vPackedLen
              , q4VScalesOffset = qScalesLen + kScalesLen
              , q4VScalesSize = vScalesLen
              , q4OutPackedOffset = qPackedLen + kPackedLen + vPackedLen
              , q4OutPackedSize = outPackedLen
              , q4OutScalesOffset = qScalesLen + kScalesLen + vScalesLen
              , q4OutScalesSize = outScalesLen
              , q4GatePackedOffset = qPackedLen + kPackedLen + vPackedLen + outPackedLen
              , q4GatePackedSize = gatePackedLen
              , q4GateScalesOffset = qScalesLen + kScalesLen + vScalesLen + outScalesLen
              , q4GateScalesSize = gateScalesLen
              , q4UpPackedOffset = qPackedLen + kPackedLen + vPackedLen + outPackedLen + gatePackedLen
              , q4UpPackedSize = upPackedLen
              , q4UpScalesOffset = qScalesLen + kScalesLen + vScalesLen + outScalesLen + gateScalesLen
              , q4UpScalesSize = upScalesLen
              , q4DownPackedOffset = qPackedLen + kPackedLen + vPackedLen + outPackedLen + gatePackedLen + upPackedLen
              , q4DownPackedSize = downPackedLen
              , q4DownScalesOffset = qScalesLen + kScalesLen + vScalesLen + outScalesLen + gateScalesLen + upScalesLen
              , q4DownScalesSize = downScalesLen
              }

        -- Upload consolidated tensors to GPU
        packedTensor <- T.createTensorWithDataPacked ctx (Shape [V.length allPacked]) U4 allPacked
        scalesTensor <- T.createTensorWithData ctx (Shape [V.length allScales]) allScales

        -- Create Q4 CONSOLIDATED shaders (use offsets into consolidated buffers)
        -- DIMENSION VALIDATION: Prevent shader dimension mismatches
        let validateShaderDims shaderName inputDim outputDim expectedIn expectedOut =
              when (inputDim /= expectedIn || outputDim /= expectedOut) $
                error $ "DIMENSION MISMATCH in " ++ shaderName ++ ": " ++
                        "expected (" ++ show expectedIn ++ " → " ++ show expectedOut ++ "), " ++
                        "got (" ++ show inputDim ++ " → " ++ show outputDim ++ ")"

        -- Validate QKV projection: hiddenDim → (qSize + 2*kvSize)
        validateShaderDims "qkvProjectionQ4" hiddenDim (qSize + 2 * kvSize) hiddenDim (qSize + 2 * kvSize)

        -- Validate output projection: qSize → hiddenDim
        validateShaderDims "outputProjectionQ4" qSize hiddenDim qSize hiddenDim

        -- Validate down projection: ffnDim → hiddenDim (THIS WAS THE BUG!)
        validateShaderDims "downProjectionQ4" ffnDim hiddenDim ffnDim hiddenDim

        -- Validate gate/up projection: hiddenDim → 2*ffnDim
        validateShaderDims "gateUpProjectionQ4" hiddenDim (2 * ffnDim) hiddenDim (2 * ffnDim)

        -- Validate norm+linear: hiddenDim → hiddenDim
        validateShaderDims "normLinearQ4" hiddenDim hiddenDim hiddenDim hiddenDim

        let qkvQ4Shader = K.createKernelCode $ qkvProjectionQ4Shader useFP16 hiddenDim qSize kvSize
            outQ4Shader = K.createKernelCode $ outputProjectionQ4ConsolidatedShader useFP16 qSize hiddenDim
            downQ4Shader = K.createKernelCode $ outputProjectionQ4ConsolidatedShader useFP16 ffnDim hiddenDim  -- ffnDim (in) → hiddenDim (out)
            gateUpQ4Shader = K.createKernelCode $ rmsNormGateUpQ4ConsolidatedShader useFP16 hiddenDim ffnDim zeroCentered
            normLinearQ4Shader = K.createKernelCode $ rmsNormLinearQ4ConsolidatedShader useFP16 hiddenDim hiddenDim zeroCentered

        qkvQ4Code <- qkvQ4Shader
        outQ4Code <- outQ4Shader
        downQ4Code <- downQ4Shader
        gateUpQ4Code <- gateUpQ4Shader
        normLinearQ4Code <- normLinearQ4Shader

        pure (Just packedTensor, Just scalesTensor, Just offsets,
              (Just normLinearQ4Code, Just gateUpQ4Code, Just qkvQ4Code, Just outQ4Code, Just downQ4Code))
      _ -> pure (Nothing, Nothing, Nothing, (Nothing, Nothing, Nothing, Nothing, Nothing))

    let (mbNormLinearQ4Shader, mbGateUpQ4Shader, mbQKVQ4Shader, mbOutQ4Shader, mbDownQ4Shader) = mbQ4Shaders

    pure $ TransformerLayer
      { -- CPU vectors
        tlAttnNormWeights = attnNorm
      , tlAttnQWeights = qWeights
      , tlAttnKWeights = kWeights
      , tlAttnVWeights = vWeights
      , tlQNormWeights = qNormWeights
      , tlKNormWeights = kNormWeights
      , tlAttnOutWeights = outWeights
      , tlPostAttnNormWeights = postAttnNormWeights
      , tlFFNNormWeights = ffnNorm
      , tlFFNGateWeights = gateWeights
      , tlFFNUpWeights = upWeights
      , tlFFNDownWeights = downWeights
      , tlPostFFNNormWeights = postFFNNormWeights
      -- GPU tensors for Attention
      , tlAttnNormTensor = attnNormT
      , tlAttnQTensor = qT
      , tlAttnKTensor = kT
      , tlAttnVTensor = vT
      , tlQNormTensor = qNormT
      , tlKNormTensor = kNormT
      , tlAttnOutTensor = outT
      , tlPostAttnNormTensor = postAttnNormT
      -- GPU tensors for FFN
      , tlFFNNormTensor = ffnNormT
      , tlFFNGateTensor = gateT
      , tlFFNUpTensor = upT
      , tlFFNDownTensor = downT
      , tlPostFFNNormTensor = postFFNNormT
      -- Shared pre-compiled shaders for Attention
      , tlRMSNormAttnShader = rmsNormAttnShader
      , tlQKVProjectionShader = qkvProjShader
      , tlQNormShader = qNormShader'
      , tlRoPEShader = ropeShader'
      , tlAttentionScoresShader = attnScoresShader
      , tlAttentionOutputShader = attnOutputShader
      , tlOutputProjectionShader = outProjShader
      , tlResidualAddShader = residualShader
      , tlPostAttnNormShader = Just postAttnNormShader
      -- Shared pre-compiled shaders for FFN
      , tlRMSNormGateUpShader = rmsNormGateUpShader
      , tlGELUMultiplyShader = geluMultShader
      , tlLinearDownShader = linearDownShader
      , tlPostFFNNormShader = Just postFFNNormShader
      , tlFFNOutputFusedShader = ffnOutputFusedShader'  -- Phase 3.1: FFN fusion
      , tlAttentionPostFusedShader = attentionPostFusedShader'  -- Phase 3.2: Attention fusion
      , tlAttentionCoreFusedShader = attentionCoreFusedShader'  -- Phase 4.1: Attention Core fusion
      -- Q4 quantization support (unused for FP16 models)
      , tlQ4PackedWeights = mbQ4Packed
      , tlQ4ScalesWeights = mbQ4Scales
      , tlQ4Offsets = mbQ4Offsets
      , tlRMSNormLinearQ4Shader = mbNormLinearQ4Shader
      , tlRMSNormGateUpQ4Shader = mbGateUpQ4Shader
      , tlQKVProjectionQ4Shader = mbQKVQ4Shader
      , tlOutputProjectionQ4Shader = mbOutQ4Shader
      , tlDownProjectionQ4Shader = mbDownQ4Shader
      -- GPU-resident KV cache
      , tlKVCacheK = kCacheT
      , tlKVCacheV = vCacheT
      , tlAppendCacheShader = appendCacheShader
      -- Persistent intermediate buffers (pre-allocated, reused)
      , tlXNorm1Buffer = xNorm1Buf
      , tlQBuffer = qBuf
      , tlKBuffer = kBuf
      , tlVBuffer = vBuf
      , tlQNormBuffer = qNormBuf
      , tlKNormBuffer = kNormBuf
      , tlQRopeBuffer = qRopeBuf
      , tlKRopeBuffer = kRopeBuf
      , tlScoresBuffer = scoresBuf
      , tlAttnOutBuffer = attnOutBuf
      , tlAttnProjBuffer = attnProjBuf
      , tlPostAttnNormBuffer = postAttnNormBuf
      , tlAfterAttnBuffer = afterAttnBuf
      , tlGateBuffer = gateBuf
      , tlUpBuffer = upBuf
      , tlGeluUpBuffer = geluUpBuf
      , tlDownBuffer = downBuf
      , tlPostFFNNormBuffer = postFFNNormBuf
      , tlOutputBuffer = outputBuf
      }

  pure $ GemmaModel
    { gmConfig = config
    , gmContext = ctx
    , gmEmbeddings = embedData
    , gmLayers = gpuLayers
    , gmFinalNormWeights = finalNorm
    , gmLMHeadWeights = lmHead
    , gmEmbeddingTensor = embeddingTensor
    , gmFinalNormTensor = finalNormTensor
    , gmLMHeadTensor = lmHeadTensor
    , gmEmbeddingShader = embeddingShader'
    , gmFinalRMSNormLinearShader = finalRMSNormLinearShader
    }

-- | Run inference on Gemma model
--
-- Takes token IDs and returns logits for next token prediction.
--
-- Pipeline:
-- 1. Embed tokens
-- 2. Run through all 24 transformer layers
-- 3. Apply final RMSNorm
-- 4. Project to vocabulary (LM head)
--
-- Returns: logits [seq_len * vocab_size] (flattened)
-- For autoregressive generation, use logits for last position only.
runGemmaInference :: GemmaModel dtype -> Vector Int -> IO (Vector Float)
runGemmaInference model@GemmaModel{..} tokenIds = evalContT $ do
  let GemmaConfig{..} = gmConfig
      seqLen = V.length tokenIds

  -- Step 1: Embed tokens [seq_len * hidden_dim]
  rawHidden <- runEmbedding tokenIds gmEmbeddings gcVocabSize gcHiddenDim

  -- Step 1.5: Apply embedding normalization (Gemma 3 requirement!)
  -- Gemma 3 multiplies embeddings by sqrt(hidden_dim) ≈ 33.94 for 1152
  let embeddingScale = sqrt (fromIntegral gcHiddenDim :: Double)
      hidden = V.map (\x -> realToFrac (realToFrac x * embeddingScale :: Double)) rawHidden

  -- For now, we only support single-token inference (seq_len = 1)
  -- Multi-token support requires batching or sequential processing
  if seqLen /= 1
    then error $ "Only single-token inference supported, got seq_len=" ++ show seqLen
    else pure ()

  -- Extract the single token's embedding [hidden_dim]
  let hiddenSingle = hidden

  -- Step 2: Run through all transformer layers
  -- Each layer takes [hidden_dim] and returns [hidden_dim]
  -- For Gemma 3:
  --   - Global layers (indices 3, 7, 11, 15, 19, 23) use full attention + high RoPE base
  --   - Local layers use sliding window attention + low RoPE base
  hiddenAfterLayers <- foldM
    (\h (layerIdx, layer) -> do
      let isGlobalLayer = gcUseSlidingWindow && (layerIdx `mod` 4 == 3)
          windowSize = if gcUseSlidingWindow && not isGlobalLayer
                      then Just gcSlidingWindowSize
                      else Nothing
          -- Dual RoPE frequencies: global layers use higher base (1M), local use lower (10k)
          ropeBase = if gcUseSlidingWindow
                     then if isGlobalLayer
                          then gcRopeBase * gcGlobalRopeScaling
                          else gcRopeBase * gcLocalRopeScaling
                     else gcRopeBase
      runTransformerBlock h layer layerIdx
        gcNumHeads gcNumKVHeads gcHeadDim gcHiddenDim gcFFNDim windowSize ropeBase
    )
    hiddenSingle
    (zip [0..] gmLayers)

  -- Step 3: Final RMSNorm
  ctx <- createContext
  hiddenNorm <- runRMSNormWithContext ctx hiddenAfterLayers gmFinalNormWeights

  -- Step 4: LM head projection [hidden_dim] -> [vocab_size]
  logits <- runLinearWithContext ctx gmLMHeadWeights hiddenNorm gcVocabSize gcHiddenDim

  pure logits

-- | Run inference with KV-cache for autoregressive generation
--
-- This function processes a single token and reuses cached K/V tensors.
-- Much faster than runGemmaInference for sequential generation.
--
-- Parameters:
-- - model: The Gemma model
-- - tokenId: Single token to process [1]
-- - cache: Optional cache from previous step (Nothing for first token)
--
-- Returns: (logits, updated_cache)
runGemmaInferenceCached :: GemmaModel dtype
                        -> Vector Int        -- Single token [1]
                        -> Maybe KVCache     -- Previous cache
                        -> IO (Vector Float, KVCache)
runGemmaInferenceCached model@GemmaModel{..} tokenIds mCache = evalContT $ do
  let GemmaConfig{..} = gmConfig
      seqLen = V.length tokenIds

  -- Validate single-token input
  if seqLen /= 1
    then error $ "Cached inference requires single token, got " ++ show seqLen
    else pure ()

  -- Initialize cache if needed
  -- Note: GPU-resident attention uses actual KV heads (GQA), not expanded heads
  let cache = case mCache of
        Just c -> c
        Nothing -> initKVCache gcNumLayers gcNumKVHeads gcHeadDim 2048  -- max_seq_len=2048

      position = case mCache of
        Nothing -> 0  -- First token
        Just c -> cacheLength (kvLayers c BV.! 0)  -- Position = cache length

  -- DEBUG: Print position and cache state
  liftIO $ putStrLn $ "🔍 Processing token at position " ++ show position ++ ", tokenId=" ++ show (V.head tokenIds)

  -- Step 1: Embed token using GPU-resident table and pre-compiled shader!
  -- No context creation, no table upload, no shader compilation!
  -- Using FP32 for embeddings until FP16 bug is fixed
  rawEmbedding <- runEmbeddingGPU gmContext tokenIds gmEmbeddingTensor gmEmbeddingShader False gcHiddenDim

  -- DEBUG: Check RAW embedding BEFORE scaling
  liftIO $ do
    debug <- lookupEnv "DEBUG"
    when (debug == Just "1" || debug == Just "true") $ do
      waitAll gmContext
      debugRawEmbed <- T.fromGPU gmContext rawEmbedding gcHiddenDim
      debugPrint $ "DEBUG RAW embedding BEFORE scaling (first 10): " ++ show (V.take 10 debugRawEmbed)
      debugPrint $ "DEBUG RAW embedding stats: min=" ++ show (V.minimum debugRawEmbed :: Float) ++ " max=" ++ show (V.maximum debugRawEmbed :: Float)

  -- Step 1.5: Apply embedding normalization (Gemma 3 requirement!)
  -- Gemma 3 multiplies embeddings by sqrt(hidden_dim) ≈ 33.94 for 1152
  -- This is critical - without it, all predictions are wrong!
  -- TEMPORARY: Using CPU scaling until GPU shader is debugged
  let embeddingScale = sqrt (fromIntegral gcHiddenDim :: Float)
  liftIO $ waitAll gmContext
  rawEmbedCPU <- liftIO $ T.fromGPU gmContext rawEmbedding gcHiddenDim
  let scaledEmbed = V.map (* embeddingScale) rawEmbedCPU
  hiddenTensor <- createTensorWithData gmContext (Shape [gcHiddenDim]) scaledEmbed

  -- DEBUG: Check embedding output (ALWAYS for debugging)
  liftIO $ do
    waitAll gmContext
    debugEmbed <- T.fromGPU gmContext hiddenTensor gcHiddenDim :: IO (V.Vector Float)
    let embMean :: Float
        embMean = V.sum debugEmbed / fromIntegral (V.length debugEmbed)
        embStd :: Float
        embStd = sqrt $ V.sum (V.map (\x -> (x - embMean) * (x - embMean)) debugEmbed) / fromIntegral (V.length debugEmbed)
    putStrLn $ "  📊 Embedding: mean=" ++ show embMean ++ ", std=" ++ show embStd ++ ", first5=" ++ show (V.take 5 debugEmbed)

  -- Step 2: Run through all transformer layers (FULLY GPU-RESIDENT with attention!)
  -- All intermediate tensors stay on GPU - NO downloads until the end!
  (finalHiddenTensor, updatedCacheLayers) <- foldM
    (\(hTensor, cacheLayers) (layerIdx, layer) -> do
      let layerCache = kvLayers cacheLayers BV.! layerIdx
          -- Determine if this is a local or global layer (Gemma 3 specific)
          -- Window size should be min(cacheLen, slidingWindowSize) to avoid reading uninitialized cache
          currentCacheLen = cacheLength layerCache + 1  -- Will be this after appending current token
          windowSize = if gcUseSlidingWindow
                       then Just (min currentCacheLen gcSlidingWindowSize)
                       else Nothing
          ropeBase = gcRopeBase

      -- Run FULLY GPU-resident transformer block with attention!
      (hOutTensor, updatedLayerCache) <- runTransformerBlockCachedGPU
        gmContext
        hTensor
        layer
        layerCache
        position
        gcNumHeads
        gcNumKVHeads
        gcHeadDim
        gcHiddenDim
        gcFFNDim
        windowSize
        ropeBase
        gcUseZeroCenteredRMSNorm

      -- DEBUG: Check layer 0 output
      liftIO $ when (layerIdx == 0) $ do
        -- ALWAYS print layer 0 for debugging
        waitAll gmContext
        debugLayer <- T.fromGPU gmContext hOutTensor gcHiddenDim :: IO (V.Vector Float)
        let layerMean = V.sum debugLayer / fromIntegral (V.length debugLayer)
            layerStd = sqrt $ V.sum (V.map (\x -> (x - layerMean) * (x - layerMean)) debugLayer) / fromIntegral (V.length debugLayer) :: Float
        putStrLn $ "  📊 Layer 0: mean=" ++ show layerMean ++ ", std=" ++ show layerStd ++ ", first10=" ++ show (V.take 10 debugLayer)

      -- Update cache for this layer
      let newCacheLayers = kvLayers cacheLayers BV.// [(layerIdx, updatedLayerCache)]
      return (hOutTensor, KVCache newCacheLayers)
    )
    (hiddenTensor, cache)
    (zip [0..] gmLayers)

  -- DEBUG: Check final hidden state before LM head
  liftIO $ do
    debug <- lookupEnv "DEBUG"
    when (debug == Just "1" || debug == Just "true") $ do
      waitAll gmContext
      debugFinal <- T.fromGPU gmContext finalHiddenTensor gcHiddenDim
      debugPrint $ "DEBUG before final RMSNorm+LM head (first 10): " ++ show (V.take 10 debugFinal)
      debugPrint $ "DEBUG before final RMSNorm+LM head stats: min=" ++ show (V.minimum debugFinal :: Float) ++ " max=" ++ show (V.maximum debugFinal :: Float)

  -- Step 3: FUSED Final RMSNorm + LM head with PRE-UPLOADED weights AND PRE-COMPILED shader!
  -- Zero weight uploads, zero shader compilations - MAXIMUM PERFORMANCE!
  logitsTensor <- runRMSNormLinearFusedPreloadedGPU gmContext finalHiddenTensor gmFinalNormTensor gmLMHeadTensor gmFinalRMSNormLinearShader gcVocabSize

  -- Step 4: Wait for ALL async kernels to complete (GPU pipelining!)
  liftIO $ waitAll gmContext

  -- Step 5: Download final logits (ONLY download at the very end!)
  logits <- liftIO $ T.fromGPU gmContext logitsTensor gcVocabSize

  return (logits, updatedCacheLayers)


-- | Version of runGemmaInferenceCached that also returns Layer 0 output for TDD testing
runGemmaInferenceCachedWithLayer0 :: GemmaModel dtype
                                  -> V.Vector Int
                                  -> Maybe KVCache
                                  -> IO (V.Vector Float, KVCache, V.Vector Float)
runGemmaInferenceCachedWithLayer0 model@GemmaModel{..} tokenIds mCache = evalContT $ do
  let GemmaConfig{..} = gmConfig
      seqLen = V.length tokenIds

  -- Validate single token
  if seqLen /= 1
    then error $ "Cached inference requires single token, got " ++ show seqLen
    else pure ()

  -- Initialize cache if needed
  let cache = case mCache of
        Just c -> c
        Nothing -> initKVCache gcNumLayers gcNumKVHeads gcHeadDim 2048

      position = case mCache of
        Nothing -> 0
        Just c -> cacheLength (kvLayers c BV.! 0)

  -- Step 1: Embed token
  rawEmbedding <- runEmbeddingGPU gmContext tokenIds gmEmbeddingTensor gmEmbeddingShader False gcHiddenDim

  -- DEBUG: Check raw embedding BEFORE scaling
  liftIO $ do
    debug <- lookupEnv "DEBUG"
    when (debug == Just "1" || debug == Just "true") $ do
      waitAll gmContext
      debugRawEmbed <- T.fromGPU gmContext rawEmbedding gcHiddenDim
      let rawMean = V.sum debugRawEmbed / fromIntegral (V.length debugRawEmbed) :: Float
          rawStd = sqrt $ V.sum (V.map (\x -> (x - rawMean) * (x - rawMean)) debugRawEmbed) / fromIntegral (V.length debugRawEmbed) :: Float
      debugPrint $ "DEBUG [Layer0WithCache] RAW embedding BEFORE scaling: mean=" ++ show rawMean ++ ", std=" ++ show rawStd ++ ", first_10=" ++ show (V.take 10 debugRawEmbed)

  -- Step 1.5: Apply embedding normalization
  let embeddingScale = sqrt (fromIntegral gcHiddenDim :: Float)
  liftIO $ waitAll gmContext
  rawEmbedCPU <- liftIO $ T.fromGPU gmContext rawEmbedding gcHiddenDim
  let scaledEmbed = V.map (* embeddingScale) rawEmbedCPU

  -- DEBUG: Check AFTER scaling
  liftIO $ do
    debug <- lookupEnv "DEBUG"
    when (debug == Just "1" || debug == Just "true") $ do
      let scaledMean = V.sum scaledEmbed / fromIntegral (V.length scaledEmbed) :: Float
          scaledStd = sqrt $ V.sum (V.map (\x -> (x - scaledMean) * (x - scaledMean)) scaledEmbed) / fromIntegral (V.length scaledEmbed) :: Float
      debugPrint $ "DEBUG [Layer0WithCache] AFTER scaling by sqrt(" ++ show gcHiddenDim ++ ")=" ++ show embeddingScale ++ ": mean=" ++ show scaledMean ++ ", std=" ++ show scaledStd ++ ", first_10=" ++ show (V.take 10 scaledEmbed)

  hiddenTensor <- createTensorWithData gmContext (Shape [gcHiddenDim]) scaledEmbed

  -- Step 2: Run through transformer layers and capture Layer 0 output
  (finalHiddenTensor, updatedCacheLayers, layer0Output) <- foldM
    (\(hTensor, cacheLayers, mLayer0) (layerIdx, layer) -> do
      let layerCache = kvLayers cacheLayers BV.! layerIdx
          currentCacheLen = cacheLength layerCache + 1
          windowSize = if gcUseSlidingWindow
                       then Just (min currentCacheLen gcSlidingWindowSize)
                       else Nothing
          ropeBase = gcRopeBase

      -- Run transformer block
      (hOutTensor, updatedLayerCache) <- runTransformerBlockCachedGPU
        gmContext
        hTensor
        layer
        layerCache
        position
        gcNumHeads
        gcNumKVHeads
        gcHeadDim
        gcHiddenDim
        gcFFNDim
        windowSize
        ropeBase
        gcUseZeroCenteredRMSNorm

      -- Capture Layer 0 output for TDD testing
      layer0Vec <- if layerIdx == 0
        then do
          liftIO $ waitAll gmContext
          liftIO $ T.fromGPU gmContext hOutTensor gcHiddenDim
        else return mLayer0

      -- Update cache for this layer
      let newCacheLayers = kvLayers cacheLayers BV.// [(layerIdx, updatedLayerCache)]
      return (hOutTensor, KVCache newCacheLayers, layer0Vec)
    )
    (hiddenTensor, cache, V.empty)
    (zip [0..] gmLayers)

  -- Step 3: Final RMSNorm + LM head
  logitsTensor <- runRMSNormLinearFusedPreloadedGPU gmContext finalHiddenTensor gmFinalNormTensor gmLMHeadTensor gmFinalRMSNormLinearShader gcVocabSize

  -- Step 4: Wait and download logits
  liftIO $ waitAll gmContext
  logits <- liftIO $ T.fromGPU gmContext logitsTensor gcVocabSize

  return (logits, updatedCacheLayers, layer0Output)
