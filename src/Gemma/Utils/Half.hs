{-# LANGUAGE BangPatterns #-}

{-|
Module: Gemma.Utils.Half
Description: Float ↔ Half (FP16) conversion utilities

Utilities for converting between FP32 (Float) and FP16 (Half) precision.
Uses the 'half' package for IEEE 754 half-precision conversions.
-}

module Gemma.Utils.Half
  ( floatToHalf
  , halfToFloat
  , vectorFloatToHalf
  , vectorHalfToFloat
  ) where

import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import qualified Numeric.Half as H
import Graphics.WebGPU.Dawn.Types (Half(..))
import Data.Word (Word16)
import Foreign.Storable (sizeOf, peek, poke)
import Foreign.Marshal.Alloc (alloca)
import Foreign.Ptr (Ptr, castPtr)
import System.IO.Unsafe (unsafePerformIO)

-- | Convert a single Float to Half (FP16)
-- Uses the half package's Half type and converts to Word16
floatToHalf :: Float -> Half
floatToHalf !f = unsafePerformIO $ do
  let h = H.toHalf f  -- Numeric.Half.Half
  alloca $ \ptr -> do
    poke ptr h
    w <- peek (castPtr ptr :: Ptr Word16)
    return (Half w)

-- | Convert a single Half (FP16) to Float
halfToFloat :: Half -> Float
halfToFloat !(Half w) = unsafePerformIO $ do
  alloca $ \ptr -> do
    poke ptr w
    h <- peek (castPtr ptr :: Ptr H.Half)
    return (H.fromHalf h)

-- | Convert a Vector of Floats to a Vector of Halves
vectorFloatToHalf :: Vector Float -> Vector Half
vectorFloatToHalf = V.map floatToHalf

-- | Convert a Vector of Halves to a Vector of Floats
vectorHalfToFloat :: Vector Half -> Vector Float
vectorHalfToFloat = V.map halfToFloat
