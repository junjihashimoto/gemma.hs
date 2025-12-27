{-# LANGUAGE OverloadedStrings #-}

{-|
Memory-efficient Q4 quantization tool using SafeTensors save functions.

Usage:
  cabal run quantize-to-q4 -- <input.safetensors> <output-q4.safetensors>

Processes tensors one at a time to minimize memory usage.
-}

module Main where

import qualified Data.Vector.Storable as V
import qualified Data.Text as T
import Data.Word (Word32, Word16)
import Data.Int (Int32)
import Data.Bits (shiftR, (.&.), shiftL, (.|.))
import System.Environment (getArgs)
import System.Exit (exitFailure)
import Text.Printf (printf)
import Data.List (isSuffixOf)
import Foreign.Storable (sizeOf)
import GHC.Float (float2Double, double2Float, castWord32ToFloat, castFloatToWord32)

import Gemma.SafeTensors
import Graphics.WebGPU.Dawn.Types (Half(..))

-- | Q4 quantization: 32 weights per block, 4 bits per weight
blockSize :: Int
blockSize = 32

-- | Convert Float (FP32) to Half (FP16)
-- Uses the Half type from Graphics.WebGPU.Dawn.Types
-- Clamps values to FP16 range to avoid Inf
floatToHalf :: Float -> Half
floatToHalf f = Half (floatToWord16 (clampToFP16Range f))
  where
    -- FP16 max value is 65504
    maxFP16 :: Float
    maxFP16 = 65504.0

    clampToFP16Range :: Float -> Float
    clampToFP16Range x
      | x > maxFP16 = maxFP16
      | x < -maxFP16 = -maxFP16
      | otherwise = x

    floatToWord16 :: Float -> Word16
    floatToWord16 x =
      let bits = castFloatToWord32 x
          sign = (bits `shiftR` 16) .&. 0x8000
          exponent_raw = fromIntegral ((bits `shiftR` 23) .&. 0xFF) :: Int
          exponent_converted = exponent_raw - 127 + 15
          -- Clamp exponent to valid FP16 range [0, 30] (31 is reserved for Inf/NaN)
          exponent = max 0 (min 30 exponent_converted)
          mantissa = (bits `shiftR` 13) .&. 0x3FF

          -- Build FP16 value
          half16
            | exponent == 0 = sign  -- Zero or denormalized
            | otherwise = sign .|. (fromIntegral exponent `shiftL` 10) .|. mantissa
      in fromIntegral half16

main :: IO ()
main = do
  args <- getArgs
  case args of
    [inputPath, outputPath] -> do
      putStrLn "========================================"
      putStrLn "Q4 Quantization (Memory-Efficient)"
      putStrLn "========================================"
      putStrLn ""
      printf "Input:  %s\n" inputPath
      printf "Output: %s\n" outputPath
      putStrLn ""

      -- Load input model
      putStrLn "Loading model..."
      st <- loadSafeTensors inputPath

      let tensorNames = listTensors st
      printf "Found %d tensors\n" (length tensorNames)
      putStrLn ""

      -- Process each tensor (returns list of lists due to Q4 split)
      putStrLn "Processing tensors..."
      tensorsDataLists <- mapM (processTensor st) tensorNames
      let tensorsData = concat tensorsDataLists  -- Flatten

      -- Save to output
      putStrLn ""
      putStrLn "Writing output file..."
      saveSafeTensors outputPath tensorsData

      putStrLn ""
      putStrLn "✅ Quantization complete!"

    _ -> do
      putStrLn "Usage: quantize-to-q4 <input.safetensors> <output-q4.safetensors>"
      putStrLn ""
      putStrLn "Memory-efficient Q4 quantization for Gemma 3 models"
      exitFailure

-- | Process a single tensor: quantize if needed, otherwise keep original
-- Returns a list of TensorData (one for regular tensors, two for Q4)
processTensor :: SafeTensorsFile -> T.Text -> IO [TensorData]
processTensor st name = do
  let nameStr = T.unpack name

  if shouldQuantize nameStr then do
    printf "  Quantizing: %s\n" nameStr

    -- Get tensor data
    weights <- getTensor st name
    let shape = getTensorShape st name

    -- Quantize to Q4
    let (packed, scales) = quantizeTensorQ4 weights
        numWeights = V.length weights
        numPacked = V.length packed
        numScales = V.length scales

    printf "    %d weights -> %d packed + %d scales (%.1f%% size)\n"
      numWeights numPacked numScales
      (100.0 * fromIntegral (numPacked * 4 + numScales * 4) / fromIntegral (numWeights * 4) :: Double)

    -- Create Q4 tensors: {name}.q4_packed and {name}.q4_scales
    let packedName = name <> ".q4_packed"
        scalesName = name <> ".q4_scales"

        -- Packed shape: [num_uint32s] as 1D array
        -- Each uint32 holds 8 nibbles, so we have total_weights / 8 uint32s
        totalWeights = product shape
        numUint32s = (totalWeights + 7) `div` 8  -- Round up
        packedShape = [numUint32s]

        -- Scales shape: [num_blocks] where num_blocks = total_weights / 32
        numBlocks = (totalWeights + 31) `div` 32
        scalesShape = [numBlocks]

    -- Return TWO tensors for Q4: packed weights and scales
    -- Use I32 instead of U32 for PyTorch compatibility (reinterpret bits)
    let packedI32 = V.unsafeCast packed :: V.Vector Int32
    pure [ createTensorData packedName I32 packedShape packedI32
         , createTensorData scalesName F32 scalesShape scales
         ]

  else do
    printf "  Keeping:    %s (converting to FP16)\n" nameStr

    -- Convert to FP16 for memory savings
    -- Embeddings and layer norms don't need FP32 precision
    weights <- getTensor st name
    let shape = getTensorShape st name
        weightsF16 = V.map floatToHalf weights

    pure [createTensorData name F16 shape weightsF16]

-- | Should this tensor be quantized to Q4?
shouldQuantize :: String -> Bool
shouldQuantize name =
  any (`isSuffixOf` name)
    [ "q_proj.weight"
    , "k_proj.weight"
    , "v_proj.weight"
    , "o_proj.weight"
    , "gate_proj.weight"
    , "up_proj.weight"
    , "down_proj.weight"
    ]

-- | Quantize tensor to Q4 format
quantizeTensorQ4 :: V.Vector Float -> (V.Vector Word32, V.Vector Float)
quantizeTensorQ4 weights =
  let
    -- Pad to multiple of blockSize
    numWeights = V.length weights
    numPadded = ((numWeights + blockSize - 1) `div` blockSize) * blockSize
    padSize = numPadded - numWeights
    padded = if padSize > 0
             then weights V.++ V.replicate padSize 0.0
             else weights

    numBlocks = numPadded `div` blockSize

    -- Process each block
    blocks = [ V.slice (i * blockSize) blockSize padded | i <- [0 .. numBlocks - 1] ]
    results = map quantizeBlock blocks

    -- Extract packed and scales
    packed = V.concat [ V.fromList [w0, w1, w2, w3] | (w0, w1, w2, w3, _) <- results ]
    scales = V.fromList [ s | (_, _, _, _, s) <- results ]

  in (packed, scales)

-- | Quantize a block of 32 FP32 weights to Q4
quantizeBlock :: V.Vector Float -> (Word32, Word32, Word32, Word32, Float)
quantizeBlock block =
  let
    -- Compute scale: max absolute value / 7.5
    maxAbs = V.maximum $ V.map abs block
    scale = if maxAbs < 1e-7 then 0.0 else maxAbs / 7.5

    -- Quantize: round(weight/scale + 7.5), clamped to [0, 15]
    quantized :: V.Vector Word32
    quantized = if scale > 1e-7
                then V.map (\w -> clamp 0 15 $ round (w / scale + 7.5)) block
                else V.replicate blockSize 7

    clamp :: Ord a => a -> a -> a -> a
    clamp lo hi x = max lo $ min hi x

    -- Pack 8 nibbles into 4 uint32s
    packNibbles start =
      let nibbles = V.slice start 8 quantized
          word = sum [ fromIntegral (nibbles V.! i) * (2 ^ (i * 4))
                     | i <- [0..7] ]
      in word :: Word32

    packed0 = packNibbles 0
    packed1 = packNibbles 8
    packed2 = packNibbles 16
    packed3 = packNibbles 24

  in (packed0, packed1, packed2, packed3, scale)

-- | Dequantize Q4 back to FP32 (for validation)
dequantizeQ4Vector :: V.Vector Word32 -> V.Vector Float -> V.Vector Float
dequantizeQ4Vector packed scales =
  let numBlocks = V.length scales
      blocks = [ dequantizeBlock (V.slice (i * 4) 4 packed) (scales V.! i)
               | i <- [0 .. numBlocks - 1] ]
  in V.concat blocks

-- | Dequantize a block of Q4 data
dequantizeBlock :: V.Vector Word32 -> Float -> V.Vector Float
dequantizeBlock packed scale =
  let
    -- Unpack 4 uint32s into 32 nibbles
    unpackWord w = [ fromIntegral ((w `shiftR` (i * 4)) .&. 0xF)
                   | i <- [0..7] ] :: [Word32]

    nibbles = concatMap unpackWord (V.toList packed)

    -- Dequantize: (nibble - 7.5) * scale
    weights = map (\n -> (fromIntegral n - 7.5) * scale) nibbles

  in V.fromList weights
