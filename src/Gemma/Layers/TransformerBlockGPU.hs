{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

{-|
Module: Gemma.Layers.TransformerBlockGPU
Description: FULLY GPU-resident transformer block (ZERO CPU transfers!)

This module implements a FULLY GPU-resident transformer block that:
1. Keeps ALL intermediate tensors on GPU
2. Only uploads weights once during model loading
3. KV cache lives permanently on GPU (updated in-place with GPU kernels)
4. Never downloads intermediate results
5. Eliminates ALL CPU↔GPU transfers during inference

Expected speedup: 10-100× over CPU-based cache version
-}

module Gemma.Layers.TransformerBlockGPU
  ( runTransformerBlockCachedGPU
  ) where

import Graphics.WebGPU.Dawn.ContT
import Graphics.WebGPU.Dawn.Types (Context, Tensor)
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import Control.Monad (foldM, when)
import System.Environment (lookupEnv)
import Data.Maybe (isJust)

import Gemma.Layers.RMSNorm (runRMSNormPreloadedGPU)
import Gemma.Layers.Linear (runLinearPreloadedGPU)
import Gemma.Layers.MLP (runGELUMultiplyFusedGPU, runRMSNormGateUpFusedPreloadedGPU, runResidualAddGPU, runFFNOutputFusedPreloadedGPU)
import Gemma.Layers.LinearQ4Fused (runRMSNormGateUpQ4GPUPreloaded, runQKVProjectionQ4GPUPreloaded, runOutputProjectionQ4GPUPreloaded,
                                    runOutputProjectionQ4GPUConsolidated, runRMSNormGateUpQ4GPUConsolidated, runRMSNormLinearQ4GPUConsolidated)
import Gemma.Layers.TransformerBlock (TransformerLayer(..), LayerQ4Offsets(..))
import Gemma.Layers.AttentionGPU
import qualified Gemma.Layers.AttentionGPU as A
import Gemma.KVCache (LayerKVCache(..), appendToCache, cacheLen)
import qualified Graphics.WebGPU.Dawn.Tensor as T

-- | Helper function to conditionally print debug messages
debugPrint :: String -> IO ()
debugPrint msg = do
  debug <- lookupEnv "DEBUG"
  case debug of
    Just "1" -> putStrLn msg
    Just "true" -> putStrLn msg
    _ -> return ()

-- | FULLY GPU-resident transformer block for cached inference
--
-- This version keeps ALL tensors on GPU with ZERO CPU transfers:
-- 1. Input tensor stays on GPU
-- 2. All intermediate activations stay on GPU
-- 3. KV cache permanently lives on GPU (updated in-place)
-- 4. Output tensor stays on GPU
-- 5. NO CPU transfers whatsoever during inference!
--
-- Pipeline (100% GPU):
--   Input (GPU)
--   → Pre-Attention RMSNorm (GPU)
--   → Q/K/V Projections (GPU)
--   → QK-Norm (GPU, Gemma 3 only)
--   → RoPE (GPU)
--   → Append K/V to GPU cache in-place (GPU)
--   → Attention Scores using GPU cache (GPU)
--   → Attention Output using GPU cache (GPU)
--   → Output Projection (GPU)
--   → Post-Attention RMSNorm (GPU, Gemma 3 only) ← NEW!
--   → Residual (GPU)
--   → Pre-Feedforward RMSNorm (GPU)
--   → FFN Gate + Up projections (GPU)
--   → GELU + Multiply (GPU)
--   → FFN Down projection (GPU)
--   → Post-FFN RMSNorm (GPU, Gemma 3 only) ← NEW!
--   → Residual + Output (GPU)
--
runTransformerBlockCachedGPU :: Context
                             -> Tensor dtype  -- Input hidden states (GPU)
                             -> TransformerLayer dtype    -- Layer weights
                             -> LayerKVCache        -- KV-cache
                             -> Int                 -- position
                             -> Int                 -- numHeads
                             -> Int                 -- numKVHeads
                             -> Int                 -- headDim
                             -> Int                 -- hiddenDim
                             -> Int                 -- ffnDim
                             -> Maybe Int           -- windowSize
                             -> Float               -- ropeBase
                             -> Bool                -- zeroCentered
                             -> ContT r IO (Tensor dtype, LayerKVCache)   -- (Output, updated cache)
runTransformerBlockCachedGPU ctx inputTensor layer cache position numHeads numKVHeads headDim hiddenDim ffnDim windowSize ropeBase zeroCentered = do
  let TransformerLayer{..} = layer
      qSize = numHeads * headDim
      kvSize = numKVHeads * headDim

  -- DEBUG: Check input tensor BEFORE batch (only if DEBUG enabled)
  liftIO $ do
    debug <- lookupEnv "DEBUG"
    when (debug == Just "1" || debug == Just "true") $ do
      waitAll ctx
      debugInput <- T.fromGPU ctx inputTensor hiddenDim :: IO (V.Vector Float)
      debugPrint $ "  DEBUG TransformerBlock INPUT (first 10): " ++ show (V.take 10 debugInput)
      debugPrint $ "  DEBUG TransformerBlock INPUT all zeros? " ++ show (V.all (== 0.0) (debugInput :: V.Vector Float))

  -- Step 1: Pre-attention RMSNorm (GPU) - writes to pre-allocated buffer
  liftIO $ debugPrint "  DEBUG: Before RMSNorm"
  runRMSNormPreloadedGPU ctx inputTensor tlAttnNormTensor tlXNorm1Buffer tlRMSNormAttnShader hiddenDim
  let xNorm1 = tlXNorm1Buffer  -- Use the pre-allocated buffer

  -- DEBUG: Check RMSNorm output
  liftIO $ do
    debug <- lookupEnv "DEBUG"
    when (debug == Just "1" || debug == Just "true") $ do
      waitAll ctx
      debugNorm <- T.fromGPU ctx xNorm1 hiddenDim :: IO (V.Vector Float)
      debugPrint $ "  DEBUG after RMSNorm (first 10): " ++ show (V.take 10 debugNorm)
      debugPrint $ "  DEBUG after RMSNorm stats: min=" ++ show (V.minimum debugNorm :: Float) ++ " max=" ++ show (V.maximum debugNorm :: Float)

  -- Step 2: Q/K/V projections (GPU) - writes to pre-allocated buffers
  -- Use Q4 path if Q4 weights and shader are available (consolidated version)
  case (tlQ4PackedWeights, tlQ4ScalesWeights, tlQ4Offsets, tlQKVProjectionQ4Shader) of
    (Just allPacked, Just allScales, Just offsets, Just q4Shader) -> do
      -- Q4 PATH: Use consolidated Q4 QKV projection
      runQKVProjectionQ4GPUPreloaded ctx xNorm1
                                     allPacked allScales
                                     (q4QPackedOffset offsets) (q4KPackedOffset offsets) (q4VPackedOffset offsets)
                                     (q4QScalesOffset offsets) (q4KScalesOffset offsets) (q4VScalesOffset offsets)
                                     tlQBuffer tlKBuffer tlVBuffer
                                     q4Shader qSize kvSize
    _ -> do
      -- FP16 PATH: Use standard QKV projection
      liftIO $ do
        debug <- lookupEnv "DEBUG"
        when (debug == Just "1" || debug == Just "true") $ do
          debugPrint "  DEBUG: Using FP16 QKV projection path"
          debugPrint $ "  DEBUG: qSize=" ++ show qSize ++ ", kvSize=" ++ show kvSize ++ ", hiddenDim=" ++ show hiddenDim
      runQKVProjectionsGPU ctx xNorm1
                           tlAttnQTensor
                           tlAttnKTensor
                           tlAttnVTensor
                           tlQBuffer tlKBuffer tlVBuffer  -- Pre-allocated outputs
                           tlQKVProjectionShader
                           hiddenDim qSize kvSize
  let qTensor = tlQBuffer   -- Use pre-allocated buffers
      kTensor = tlKBuffer
      vTensor = tlVBuffer

  -- DEBUG: Check Q/K/V projection output
  liftIO $ do
    debug <- lookupEnv "DEBUG"
    when (debug == Just "1" || debug == Just "true") $ do
      waitAll ctx
      debugQ <- T.fromGPU ctx qTensor qSize :: IO (V.Vector Float)
      debugK <- T.fromGPU ctx kTensor kvSize :: IO (V.Vector Float)
      debugV <- T.fromGPU ctx vTensor kvSize :: IO (V.Vector Float)
      debugPrint $ "  DEBUG after Q proj (first 10): " ++ show (V.take 10 debugQ)
      debugPrint $ "  DEBUG after K proj (first 10): " ++ show (V.take 10 debugK)
      debugPrint $ "  DEBUG after V proj (first 10): " ++ show (V.take 10 debugV)
      debugPrint $ "  DEBUG Q stats: min=" ++ show (V.minimum debugQ :: Float) ++ " max=" ++ show (V.maximum debugQ :: Float)

  -- Step 3: Optional QK-Norm (Gemma 3) (GPU) - writes to pre-allocated buffers
  case tlQNormTensor of
    Just normTensor -> runQKNormGPU ctx qTensor normTensor tlQNormBuffer tlQNormShader numHeads headDim
    Nothing -> pure ()
  case tlKNormTensor of
    Just normTensor -> runQKNormGPU ctx kTensor normTensor tlKNormBuffer tlQNormShader numKVHeads headDim
    Nothing -> pure ()
  -- Use appropriate buffer depending on whether QKNorm was applied
  let qNorm = if isJust tlQNormTensor then tlQNormBuffer else qTensor
      kNorm = if isJust tlKNormTensor then tlKNormBuffer else kTensor

  -- DEBUG: Check QKNorm output
  liftIO $ do
    debug <- lookupEnv "DEBUG"
    when (debug == Just "1" || debug == Just "true") $ do
      when (isJust tlQNormTensor) $ do
        waitAll ctx
        debugQNorm <- T.fromGPU ctx qNorm qSize :: IO (V.Vector Float)
        debugKNorm <- T.fromGPU ctx kNorm kvSize :: IO (V.Vector Float)
        debugPrint $ "  DEBUG after QKNorm (first 10): " ++ show (V.take 10 debugQNorm)
        debugPrint $ "  DEBUG QNorm stats: min=" ++ show (V.minimum debugQNorm :: Float) ++ " max=" ++ show (V.maximum debugQNorm :: Float)

  -- Step 4: Apply RoPE (GPU) - writes to pre-allocated buffers
  runRoPEGPU ctx qNorm tlQRopeBuffer tlRoPEShader position ropeBase headDim qSize
  runRoPEGPU ctx kNorm tlKRopeBuffer tlRoPEShader position ropeBase headDim kvSize
  let qRope = tlQRopeBuffer  -- Use pre-allocated buffers
      kRope = tlKRopeBuffer

  -- DEBUG: Check RoPE output
  liftIO $ do
    debug <- lookupEnv "DEBUG"
    when (debug == Just "1" || debug == Just "true") $ do
      waitAll ctx
      debugQRope <- T.fromGPU ctx qRope qSize :: IO (V.Vector Float)
      debugKRope <- T.fromGPU ctx kRope kvSize :: IO (V.Vector Float)
      debugPrint $ "  DEBUG after RoPE Q (first 10): " ++ show (V.take 10 debugQRope)
      debugPrint $ "  DEBUG after RoPE Q stats: min=" ++ show (V.minimum debugQRope :: Float) ++ " max=" ++ show (V.maximum debugQRope :: Float)
      debugPrint $ "  DEBUG after RoPE K stats: min=" ++ show (V.minimum debugKRope :: Float) ++ " max=" ++ show (V.maximum debugKRope :: Float)

  -- Step 5: GPU-resident KV cache update (NO CPU TRANSFERS!)
  -- Append new K/V directly to GPU cache tensors in-place
  liftIO $ do
    debug <- lookupEnv "DEBUG"
    when (debug == Just "1" || debug == Just "true") $ do
      debugPrint "  DEBUG: About to append to KV cache"
  runAppendKVCacheGPU ctx tlKVCacheK tlKVCacheV kRope vTensor tlAppendCacheShader
                      position numKVHeads headDim
  liftIO $ do
    debug <- lookupEnv "DEBUG"
    when (debug == Just "1" || debug == Just "true") $ do
      debugPrint "  DEBUG: KV cache append completed"
      -- Check K cache immediately after append
      waitAll ctx
      debugKCache <- T.fromGPU ctx tlKVCacheK (numKVHeads * headDim) :: IO (V.Vector Float)
      debugVCache <- T.fromGPU ctx tlKVCacheV (numKVHeads * headDim) :: IO (V.Vector Float)
      debugPrint $ "  DEBUG K cache at pos " ++ show position ++ " (first 10): " ++ show (V.take 10 debugKCache)
      debugPrint $ "  DEBUG V cache at pos " ++ show position ++ " (first 10): " ++ show (V.take 10 debugVCache)

  -- Update cache length (CPU-side tracking only, no data transfer)
  let updatedCache = cache { cacheLen = cacheLen cache + 1 }
      cacheLen' = cacheLen updatedCache
      effectiveLen = case windowSize of
        Just w -> min cacheLen' w
        Nothing -> cacheLen'

  -- DEBUG: Print effectiveLen
  liftIO $ do
    debug <- lookupEnv "DEBUG"
    when (debug == Just "1" || debug == Just "true") $ do
      debugPrint $ "  DEBUG: cacheLen before=" ++ show (cacheLen cache) ++ ", after=" ++ show cacheLen' ++ ", effectiveLen=" ++ show effectiveLen

  -- Step 6-7: Attention Core (Scores + Output)
  -- Phase 4.1: Use fused kernel if available (2 dispatches → 1)
  let maxCacheLen = 2048
  case tlAttentionCoreFusedShader of
    Just fusedShader -> do
      -- FUSED PATH: Scores + Output in ONE dispatch
      -- Scores stay in workgroup shared memory - ZERO global memory writes!
      A.runAttentionCoreFusedPreloadedGPU ctx qRope tlKVCacheK tlKVCacheV tlAttnOutBuffer fusedShader
                                          numHeads numKVHeads headDim effectiveLen windowSize maxCacheLen
    Nothing -> do
      -- UNFUSED PATH: 2 separate dispatches (fallback)
      -- Step 6: Attention scores (GPU) - writes to pre-allocated buffer
      liftIO $ do
        debug <- lookupEnv "DEBUG"
        when (debug == Just "1" || debug == Just "true") $ do
          debugPrint "  DEBUG: About to compute attention scores"
      runAttentionScoresGPU ctx qRope tlKVCacheK tlScoresBuffer tlAttentionScoresShader
                             numHeads numKVHeads headDim effectiveLen windowSize maxCacheLen
      liftIO $ do
        debug <- lookupEnv "DEBUG"
        when (debug == Just "1" || debug == Just "true") $ do
          debugPrint "  DEBUG: Attention scores completed"
          -- Check attention scores immediately
          waitAll ctx
          debugScores <- T.fromGPU ctx tlScoresBuffer (numHeads * maxCacheLen) :: IO (V.Vector Float)
          debugPrint $ "  DEBUG scores (first 10): " ++ show (V.take 10 debugScores)
      let scoresTensor = tlScoresBuffer

      -- Step 7: Attention output (GPU) - writes to pre-allocated buffer
      liftIO $ do
        debug <- lookupEnv "DEBUG"
        when (debug == Just "1" || debug == Just "true") $ do
          debugPrint $ "  DEBUG: About to compute attention output with effectiveLen=" ++ show effectiveLen
      runAttentionOutputGPU ctx scoresTensor tlKVCacheV tlAttnOutBuffer tlAttentionOutputShader
                             numHeads numKVHeads headDim effectiveLen
      liftIO $ do
        debug <- lookupEnv "DEBUG"
        when (debug == Just "1" || debug == Just "true") $ do
          debugPrint "  DEBUG: Attention output completed"
          -- IMMEDIATELY check the attention output before it gets overwritten
          waitAll ctx
          debugAttnOutImmediate <- T.fromGPU ctx tlAttnOutBuffer qSize :: IO (V.Vector Float)
          debugPrint $ "  DEBUG IMMEDIATE attnOut (first 10): " ++ show (V.take 10 debugAttnOutImmediate)
          debugPrint $ "  DEBUG IMMEDIATE attnOut stats: min=" ++ show (V.minimum debugAttnOutImmediate :: Float) ++ " max=" ++ show (V.maximum debugAttnOutImmediate :: Float)
      pure ()

  let attnOutTensor = tlAttnOutBuffer


  -- Step 8-9: Attention Postprocessing (OutProj + Residual + Post-attn Norm)
  -- Phase 3.2: Use fused kernel if available (3 dispatches → 1)
  case (tlAttentionPostFusedShader, tlPostAttnNormTensor) of
    (Just fusedShader, Just normTensor) -> do
      -- FUSED PATH: OutProj + Residual + PostAttnNorm in ONE dispatch
      A.runAttentionPostFusedPreloadedGPU ctx attnOutTensor tlAttnOutTensor inputTensor normTensor
                                          tlAfterAttnBuffer fusedShader hiddenDim
    _ -> do
      -- UNFUSED PATH: 3 separate dispatches (fallback)
      -- Step 8: Output projection
      case (tlQ4PackedWeights, tlQ4ScalesWeights, tlQ4Offsets, tlOutputProjectionQ4Shader) of
        (Just allPacked, Just allScales, Just offsets, Just q4Shader) -> do
          -- Q4 PATH: Use consolidated Q4 output projection
          runOutputProjectionQ4GPUConsolidated ctx attnOutTensor allPacked allScales
                                               (q4OutPackedOffset offsets) (q4OutScalesOffset offsets)
                                               tlAttnProjBuffer q4Shader hiddenDim qSize
        _ -> do
          -- FP16 PATH: Use standard output projection
          runOutputProjectionGPU ctx attnOutTensor tlAttnOutTensor tlAttnProjBuffer tlOutputProjectionShader
                                 hiddenDim qSize
      let attnProjTensor = tlAttnProjBuffer

      -- Step 8.5: Post-attention norm (Gemma 3)
      case (tlPostAttnNormTensor, tlPostAttnNormShader) of
        (Just normTensor, Just normShader) ->
          runRMSNormPreloadedGPU ctx attnProjTensor normTensor tlPostAttnNormBuffer normShader hiddenDim
        _ -> pure ()
      let attnNormedTensor = case tlPostAttnNormTensor of
            Just _ -> tlPostAttnNormBuffer
            Nothing -> attnProjTensor

      -- DEBUG: Check post-attention norm output
      liftIO $ do
        debug <- lookupEnv "DEBUG"
        when (debug == Just "1" || debug == Just "true") $ do
          waitAll ctx
          debugAttnProj <- T.fromGPU ctx attnProjTensor hiddenDim :: IO (V.Vector Float)
          debugAttnNormed <- T.fromGPU ctx attnNormedTensor hiddenDim :: IO (V.Vector Float)
          debugPrint $ "  DEBUG attn_proj (before norm) (first 10): " ++ show (V.take 10 debugAttnProj)
          debugPrint $ "  DEBUG attn_normed (after norm) (first 10): " ++ show (V.take 10 debugAttnNormed)
          debugPrint $ "  DEBUG has post-attn norm? " ++ show (isJust tlPostAttnNormTensor)

      -- Step 9: Residual connection
      runResidualAddGPU ctx inputTensor attnNormedTensor tlAfterAttnBuffer tlResidualAddShader hiddenDim

  let afterAttnTensor = tlAfterAttnBuffer

  -- DEBUG: Check after attention residual
  liftIO $ do
    debug <- lookupEnv "DEBUG"
    when (debug == Just "1" || debug == Just "true") $ do
      waitAll ctx
      debugAfterAttn <- T.fromGPU ctx afterAttnTensor hiddenDim :: IO (V.Vector Float)
      debugPrint $ "  DEBUG AFTER ATTN RESIDUAL (first 10): " ++ show (V.take 10 debugAfterAttn)

  -- DEBUG: Check inputs BEFORE fused kernel
  liftIO $ do
    debug <- lookupEnv "DEBUG"
    when (debug == Just "1" || debug == Just "true") $ do
      waitAll ctx
      debugAfterAttnCheck <- T.fromGPU ctx afterAttnTensor hiddenDim :: IO (V.Vector Float)
      debugFFNNorm <- T.fromGPU ctx tlFFNNormTensor hiddenDim :: IO (V.Vector Float)
      debugFFNGate <- T.fromGPU ctx tlFFNGateTensor (ffnDim * hiddenDim) :: IO (V.Vector Float)
      debugPrint $ "  DEBUG BEFORE FUSED KERNEL: afterAttn (first 10): " ++ show (V.take 10 debugAfterAttnCheck)
      debugPrint $ "  DEBUG BEFORE FUSED KERNEL: afterAttn stats: min=" ++ show (V.minimum debugAfterAttnCheck :: Float) ++ " max=" ++ show (V.maximum debugAfterAttnCheck :: Float) ++ " has_nan=" ++ show (V.any isNaN debugAfterAttnCheck)
      debugPrint $ "  DEBUG BEFORE FUSED KERNEL: ffnNorm (first 10): " ++ show (V.take 10 debugFFNNorm)
      debugPrint $ "  DEBUG BEFORE FUSED KERNEL: ffnGate (first 10 weights): " ++ show (V.take 10 debugFFNGate)
      debugPrint $ "  DEBUG BEFORE FUSED KERNEL: ffnGate stats: min=" ++ show (V.minimum debugFFNGate :: Float) ++ " max=" ++ show (V.maximum debugFFNGate :: Float) ++ " has_nan=" ++ show (V.any isNaN debugFFNGate)
      debugPrint $ "  DEBUG BEFORE FUSED KERNEL: ffnGate has Infinity? " ++ show (V.any isInfinite debugFFNGate)

  -- Step 10: FFN (GPU) - TRIPLE-FUSED: RMSNorm + Gate + Up - writes to pre-allocated buffers
  case (tlQ4PackedWeights, tlQ4ScalesWeights, tlQ4Offsets, tlRMSNormGateUpQ4Shader) of
    (Just allPacked, Just allScales, Just offsets, Just q4Shader) -> do
      -- Q4 PATH: Use consolidated Q4 RMSNorm + Gate + Up
      runRMSNormGateUpQ4GPUConsolidated ctx afterAttnTensor tlFFNNormTensor
                                        allPacked allScales
                                        (q4GatePackedOffset offsets) (q4GateScalesOffset offsets)
                                        (q4UpPackedOffset offsets) (q4UpScalesOffset offsets)
                                        tlGateBuffer tlUpBuffer
                                        q4Shader ffnDim
    _ -> do
      -- FP16 PATH: Use standard RMSNorm + Gate + Up fusion
      runRMSNormGateUpFusedPreloadedGPU ctx afterAttnTensor
                                        tlFFNNormTensor
                                        tlFFNGateTensor
                                        tlFFNUpTensor
                                        tlGateBuffer tlUpBuffer
                                        tlRMSNormGateUpShader
                                        ffnDim
  let gateTensor = tlGateBuffer
      upTensor = tlUpBuffer


  -- Step 11: FUSED: GELU(gate) * up (GPU) - writes to pre-allocated buffer
  runGELUMultiplyFusedGPU ctx gateTensor upTensor tlGeluUpBuffer tlGELUMultiplyShader ffnDim
  let geluUpTensor = tlGeluUpBuffer

  -- DEBUG: Check GELU output BEFORE down projection
  liftIO $ do
    debug <- lookupEnv "DEBUG"
    when (debug == Just "1" || debug == Just "true") $ do
      waitAll ctx
      debugGate <- T.fromGPU ctx gateTensor ffnDim :: IO (V.Vector Float)
      debugUp <- T.fromGPU ctx upTensor ffnDim :: IO (V.Vector Float)
      debugGeluUp <- T.fromGPU ctx geluUpTensor ffnDim :: IO (V.Vector Float)
      debugPrint $ "  DEBUG BEFORE DOWN PROJ: gate (first 10): " ++ show (V.take 10 debugGate)
      debugPrint $ "  DEBUG BEFORE DOWN PROJ: up (first 10): " ++ show (V.take 10 debugUp)
      debugPrint $ "  DEBUG BEFORE DOWN PROJ: geluUp (first 10): " ++ show (V.take 10 debugGeluUp)
      debugPrint $ "  DEBUG BEFORE DOWN PROJ: geluUp stats: min=" ++ show (V.minimum debugGeluUp :: Float) ++ " max=" ++ show (V.maximum debugGeluUp :: Float)
      debugPrint $ "  DEBUG BEFORE DOWN PROJ: geluUp has NaN? " ++ show (V.any isNaN debugGeluUp)

  -- Step 12-13: FFN Output (Down projection + Residual + Post-norm)
  -- Phase 3.1: Use fused kernel if available (3 dispatches → 1)
  case (tlFFNOutputFusedShader, tlPostFFNNormTensor) of
    (Just fusedShader, Just normTensor) -> do
      -- FUSED PATH: LinearDown + Residual + PostNorm in ONE dispatch
      runFFNOutputFusedPreloadedGPU ctx geluUpTensor tlFFNDownTensor afterAttnTensor normTensor
                                    tlOutputBuffer fusedShader hiddenDim
    _ -> do
      -- UNFUSED PATH: 3 separate dispatches (fallback)
      -- Step 12: Down projection (GPU)
      case (tlQ4PackedWeights, tlQ4ScalesWeights, tlQ4Offsets, tlDownProjectionQ4Shader) of
        (Just allPacked, Just allScales, Just offsets, Just q4Shader) -> do
          -- Q4 PATH: Use consolidated Q4 down projection (plain linear, no RMSNorm)
          -- Uses separate down shader with correct dimensions: ffnDim (input) → hiddenDim (output)
          runOutputProjectionQ4GPUConsolidated ctx geluUpTensor allPacked allScales
                                               (q4DownPackedOffset offsets) (q4DownScalesOffset offsets)
                                               tlDownBuffer q4Shader hiddenDim ffnDim
        _ -> do
          -- FP16 PATH: Use standard down projection
          runLinearPreloadedGPU ctx tlFFNDownTensor geluUpTensor tlDownBuffer tlLinearDownShader hiddenDim
      let downTensor = tlDownBuffer

      -- Step 12.5: Post-FFN norm (Gemma 3) (GPU) - writes to pre-allocated buffer
      case (tlPostFFNNormTensor, tlPostFFNNormShader) of
        (Just normTensor, Just normShader) ->
          runRMSNormPreloadedGPU ctx downTensor normTensor tlPostFFNNormBuffer normShader hiddenDim
        _ -> pure ()
      -- Use normalized output if available, otherwise use down projection output
      let ffnNormedTensor = case tlPostFFNNormTensor of
            Just _ -> tlPostFFNNormBuffer
            Nothing -> downTensor

      -- Step 13: Final residual (GPU) - writes to pre-allocated buffer
      runResidualAddGPU ctx afterAttnTensor ffnNormedTensor tlOutputBuffer tlResidualAddShader hiddenDim

  let outputTensor = tlOutputBuffer

  -- DEBUG: Check intermediate tensors (only if DEBUG enabled)
  liftIO $ do
    debug <- lookupEnv "DEBUG"
    when (debug == Just "1" || debug == Just "true") $ do
      waitAll ctx  -- Just wait, don't call endBatch (we're not in batch mode)
      debugAttnOut <- T.fromGPU ctx attnOutTensor qSize :: IO (V.Vector Float)
      debugDown <- T.fromGPU ctx tlDownBuffer hiddenDim :: IO (V.Vector Float)  -- Use buffer instead of scoped variable
      debugOutput <- T.fromGPU ctx outputTensor hiddenDim :: IO (V.Vector Float)
      debugPrint $ "  DEBUG SECOND BATCH: attnOut (first 10): " ++ show (V.take 10 debugAttnOut)
      debugPrint $ "  DEBUG SECOND BATCH: downProj (first 10): " ++ show (V.take 10 debugDown)
      debugPrint $ "  DEBUG TransformerBlock OUTPUT (first 10): " ++ show (V.take 10 debugOutput)
      debugPrint $ "  DEBUG TransformerBlock OUTPUT all zeros? " ++ show (V.all (== 0.0) (debugOutput :: V.Vector Float))

  pure (outputTensor, updatedCache)
