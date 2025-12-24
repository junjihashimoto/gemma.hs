{-# LANGUAGE BangPatterns #-}

{-|
Module: Gemma.Quantization.Q4
Description: 4-bit block-wise quantization for weight compression

This module implements 4-bit quantization with symmetric zero-centered quantization.
- Block size: 32 weights per block
- Packing: 8 nibbles (4-bit values) per Word32
- Scaling: 1 FP16 scale per block of 32 weights
- Range: Each nibble stores 0-15, representing quantized values

Quantization formula:
  quantized = round((weight - min) * 15 / (max - min))
  dequantized = (quantized / 15) * (max - min) + min

Symmetric variant (recommended, matches PyTorch):
  quantized = round((weight / scale) + 7.5)  where scale = max(abs(weight)) / 7.5
  dequantized = (quantized - 7.5) * scale
  Range: [-7.5, +7.5] mapped to [0, 15]
-}

module Gemma.Quantization.Q4
  ( -- * Quantization
    quantizeQ4
  , dequantizeQ4
    -- * Bit packing utilities
  , packNibbles
  , unpackNibbles
    -- * GPU tensor upload
  , uploadQ4Tensors
  ) where

import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import Data.Word (Word32)
import Data.Bits
import Control.Monad (forM_)
import Graphics.WebGPU.Dawn.Types (Context, Tensor, Shape(..), NumType(..))
import qualified Graphics.WebGPU.Dawn.Tensor as T

-- | Pack 8 nibbles (4-bit values) into one Word32
-- Nibbles are packed from LSB to MSB: nibble[0] in bits 0-3, nibble[7] in bits 28-31
packNibbles :: [Word32] -> Word32
packNibbles nibbles =
  case nibbles of
    [n0, n1, n2, n3, n4, n5, n6, n7] ->
      (n0 .&. 0xF)       .|.
      ((n1 .&. 0xF) `shiftL` 4)  .|.
      ((n2 .&. 0xF) `shiftL` 8)  .|.
      ((n3 .&. 0xF) `shiftL` 12) .|.
      ((n4 .&. 0xF) `shiftL` 16) .|.
      ((n5 .&. 0xF) `shiftL` 20) .|.
      ((n6 .&. 0xF) `shiftL` 24) .|.
      ((n7 .&. 0xF) `shiftL` 28)
    _ -> error $ "packNibbles: expected 8 nibbles, got " ++ show (length nibbles)

-- | Unpack one Word32 into 8 nibbles (4-bit values)
unpackNibbles :: Word32 -> [Word32]
unpackNibbles packed =
  [ packed .&. 0xF
  , (packed `shiftR` 4) .&. 0xF
  , (packed `shiftR` 8) .&. 0xF
  , (packed `shiftR` 12) .&. 0xF
  , (packed `shiftR` 16) .&. 0xF
  , (packed `shiftR` 20) .&. 0xF
  , (packed `shiftR` 24) .&. 0xF
  , (packed `shiftR` 28) .&. 0xF
  ]

-- | Quantize FP16 weights to Q4 format (block size 32, symmetric)
--
-- Args:
--   weights: Row-major weight matrix [outSize * inSize]
--   outSize: Number of output features (rows)
--   inSize: Number of input features (cols, must be multiple of 32)
--
-- Returns:
--   (packed, scales): Packed nibbles and per-block scales
--     packed: [outSize * inSize / 8] Word32 values
--     scales: [outSize * inSize / 32] Float values
quantizeQ4 :: Vector Float -> Int -> Int -> (Vector Word32, Vector Float)
quantizeQ4 weights outSize inSize
  | inSize `mod` 32 /= 0 = error $ "quantizeQ4: inSize must be multiple of 32, got " ++ show inSize
  | V.length weights /= outSize * inSize = error $ "quantizeQ4: weight size mismatch"
  | otherwise = (packedVec, scalesVec)
  where
    totalSize = outSize * inSize
    numBlocks = totalSize `div` 32
    blocksPerRow = inSize `div` 32

    -- Process each block of 32 weights
    (packedList, scalesList) = unzip $ map processBlock [0..numBlocks-1]

    packedVec = V.fromList $ concat packedList
    scalesVec = V.fromList scalesList

    processBlock :: Int -> ([Word32], Float)
    processBlock blockIdx =
      let blockStart = blockIdx * 32
          block = V.slice blockStart 32 weights

          -- Compute scale: max absolute value in block
          maxAbs = V.maximum (V.map abs block)
          -- For all zeros, use scale=1 and special handling
          -- For normal values, use maxAbs/7.5 for symmetric [-7.5, +7.5] range
          -- This matches PyTorch's Q4 implementation
          scale = if maxAbs < 1e-8 then 1e-8 else maxAbs / 7.5

          -- Quantize each weight in block
          quantized = V.toList $ V.map (\w -> quantizeWeight w scale maxAbs) block

          -- Pack into 4 Word32s (8 nibbles each)
          packed = map packNibbles
            [ take 8 $ drop 0 quantized
            , take 8 $ drop 8 quantized
            , take 8 $ drop 16 quantized
            , take 8 $ drop 24 quantized
            ]
      in (packed, scale)

    quantizeWeight :: Float -> Float -> Float -> Word32
    quantizeWeight w scale maxAbs =
      -- Special case: if all values are near zero, quantize to center (7.5 → nibble 8)
      if maxAbs < 1e-8 then 7  -- Center nibble for zero
      else
        let normalized = w / scale  -- Range: -7 to +7
            shifted = normalized + 7.5  -- Range: 0.5 to 14.5
            clamped = max 0 (min 15 shifted)  -- Clamp to 0-15
        in round clamped

-- | Dequantize Q4 back to FP16 (for testing and validation)
dequantizeQ4 :: Vector Word32 -> Vector Float -> Vector Float
dequantizeQ4 packed scales = V.fromList allWeights
  where
    numBlocks = V.length scales
    numPacked = V.length packed

    allWeights = concatMap dequantizeBlock [0..numBlocks-1]

    dequantizeBlock :: Int -> [Float]
    dequantizeBlock blockIdx =
      let scale = scales V.! blockIdx
          packedStart = blockIdx * 4  -- 4 Word32s per block

          -- Unpack 4 Word32s into 32 nibbles
          nibbles = concatMap unpackNibbles
            [ packed V.! (packedStart + 0)
            , packed V.! (packedStart + 1)
            , packed V.! (packedStart + 2)
            , packed V.! (packedStart + 3)
            ]

          -- Dequantize each nibble
          weights = map (\n -> dequantizeWeight n scale) nibbles
      in weights

    dequantizeWeight :: Word32 -> Float -> Float
    dequantizeWeight nibble scale =
      -- Handle zero scale (all zeros case)
      if scale < 1e-7 then 0.0
      else
        let shifted = fromIntegral nibble - 7.5  -- Range: -7.5 to +7.5
            denormalized = shifted * scale
        in denormalized

-- | Upload Q4 tensors to GPU
--
-- Args:
--   ctx: GPU context
--   packed: Packed nibbles (Word32 vector)
--   scales: Per-block scales (Float vector)
--
-- Returns:
--   (packedTensor, scalesTensor): GPU tensors ready for Q4 shaders
uploadQ4Tensors :: Context -> Vector Word32 -> Vector Float -> IO (Tensor dtype, Tensor dtype)
uploadQ4Tensors ctx packed scales = do
  let packedShape = Shape [V.length packed]
      scalesShape = Shape [V.length scales]

  -- Upload packed weights as U32 tensor
  packedTensor <- T.createTensorWithData ctx packedShape packed

  -- Upload scales as F32 tensor (will be cast to F16 in shader)
  scalesTensor <- T.createTensorWithData ctx scalesShape scales

  return (packedTensor, scalesTensor)
