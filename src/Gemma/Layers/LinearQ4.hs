{-# LANGUAGE OverloadedStrings #-}

{-|
Module: Gemma.Layers.LinearQ4
Description: Q4_0 quantized linear layer using fused dequantization + matmul

This module provides Q4_0 quantized matrix-vector multiplication for transformer layers.
Uses the validated Q4_0 GGUF implementation with fused dequantization and matmul.

Formula:
  output = (weights_q4 @ input) + bias  (no bias in Gemma 3)

Where weights_q4 are Q4_0 quantized (18 bytes per 32-element block).
The kernel fuses dequantization with matrix-vector multiply for efficiency.
-}

module Gemma.Layers.LinearQ4
  ( runLinearQ4
  , runLinearQ4WithContext
  , runLinearQ4GPU  -- For LinearQ4Fused compatibility
  ) where

import Graphics.WebGPU.Dawn.ContT
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import Data.Word (Word32)

import Gemma.Layers.DequantQ4GGUF (runMatmulQ4_0DSL)

-- | Run Q4_0 quantized linear layer: output = weights_q4 @ input
--
-- Parameters:
-- - weightsQ4: Q4_0 quantized weights as Word32 vector [nRows * nCols packed]
-- - input: Input vector [nCols]
-- - nRows: Output dimension
-- - nCols: Input dimension (must be divisible by 32 for Q4_0 block size)
-- - applySoftCap: Whether to apply Gemma 3 soft-capping (for final layer)
-- - softCapSigma: Soft-cap parameter (default 30.0 for Gemma 3)
--
-- Returns: Output vector [nRows]
runLinearQ4 :: Vector Word32  -- Q4_0 weights
            -> Vector Float   -- Input
            -> Int            -- nRows
            -> Int            -- nCols
            -> Bool           -- applySoftCap
            -> Float          -- softCapSigma
            -> ContT r IO (Vector Float)
runLinearQ4 = runMatmulQ4_0DSL

-- | Run Q4_0 linear layer with explicit context (for consistency with other layers)
--
-- Note: Currently ignores context parameter since Q4 matmul creates its own GPU context.
-- This interface is provided for API consistency with other layer functions.
runLinearQ4WithContext :: ctx           -- GPU context (unused)
                       -> Vector Word32  -- Q4_0 weights
                       -> Vector Float   -- Input
                       -> Int            -- nRows
                       -> Int            -- nCols
                       -> Bool           -- applySoftCap
                       -> Float          -- softCapSigma
                       -> ContT r IO (Vector Float)
runLinearQ4WithContext _ctx = runMatmulQ4_0DSL

-- | GPU-based Q4 linear layer (alias for compatibility)
--
-- This is an alias for runLinearQ4 to maintain compatibility with existing code.
runLinearQ4GPU :: Vector Word32  -- Q4_0 weights
               -> Vector Float   -- Input
               -> Int            -- nRows
               -> Int            -- nCols
               -> Bool           -- applySoftCap
               -> Float          -- softCapSigma
               -> ContT r IO (Vector Float)
runLinearQ4GPU = runLinearQ4
