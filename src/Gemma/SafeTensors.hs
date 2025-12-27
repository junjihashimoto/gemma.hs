{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE FunctionalDependencies #-}

{-|
Module: Gemma.SafeTensors
Description: Parser for SafeTensors format

This module provides functionality to load and parse .safetensors files,
which store model weights and activations in a simple binary format.

SafeTensors format:
- 8 bytes: header length (little-endian u64)
- N bytes: JSON header (UTF-8)
- M bytes: binary tensor data

The JSON header contains metadata for each tensor:
{
  "tensor_name": {
    "dtype": "F32",
    "shape": [batch, seq_len, hidden],
    "data_offsets": [start, end]
  },
  ...
}
-}

module Gemma.SafeTensors
  ( -- * Types
    SafeTensorsFile(..)
  , TensorInfo(..)
  , DType(..)
  , TensorData(..)
    -- * Loading
  , loadSafeTensors
    -- * Saving
  , saveSafeTensors
  , createTensorData
    -- * Querying
  , getTensor
  , getTensorTyped
  , getTensorU32
  , getTensorShape
  , getTensorDType
  , listTensors
  , hasTensor
    -- * Q4 Support
  , hasQ4Weight
  , loadQ4Weight
  , loadQ4WeightDequantized
    -- * Utilities
  , tensorSize
  ) where

import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BL
import Data.ByteString (ByteString)
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import qualified Data.Aeson as Aeson
import Data.Aeson ((.:), (.!=))
import qualified Data.Aeson.Types as Aeson
import qualified Data.Map.Strict as Map
import Data.Map.Strict (Map)
import Data.Text (Text)
import qualified Data.Text as T
import Data.Word
import Data.Int
import Foreign.Ptr (Ptr, castPtr, plusPtr)
import Foreign.ForeignPtr (ForeignPtr, newForeignPtr_, withForeignPtr)
import Foreign.Storable (Storable, sizeOf, peekElemOff, peek, poke)
import Foreign.Marshal.Alloc (allocaBytes)
import Data.Bits ((.&.), (.|.), shiftL, shiftR)
import System.IO.Unsafe (unsafePerformIO)
import Control.Monad (forM, when)
import qualified Data.Aeson.Key as Key
import qualified Data.Aeson.KeyMap as KM
import Data.Proxy (Proxy(..))
import qualified Graphics.WebGPU.Dawn.Types as GPU (NumType(..))
import Graphics.WebGPU.Dawn.Types (Half(..))

-- | Supported data types in SafeTensors
data DType
  = F32  -- Float32
  | F16  -- Float16
  | BF16 -- BFloat16
  | F64  -- Float64
  | I32  -- Int32
  | I64  -- Int64
  | U32  -- UInt32
  | U64  -- UInt64
  deriving (Show, Eq)

instance Aeson.FromJSON DType where
  parseJSON = Aeson.withText "DType" $ \t -> case t of
    "F32" -> pure F32
    "F16" -> pure F16
    "BF16" -> pure BF16
    "F64" -> pure F64
    "I32" -> pure I32
    "I64" -> pure I64
    "U32" -> pure U32
    "U64" -> pure U64
    _     -> fail $ "Unknown dtype: " ++ T.unpack t

-- | Information about a tensor in the file
data TensorInfo = TensorInfo
  { tiDType :: DType
  , tiShape :: [Int]
  , tiDataOffsets :: (Int, Int)  -- (start, end) in bytes
  } deriving (Show)

instance Aeson.FromJSON TensorInfo where
  parseJSON = Aeson.withObject "TensorInfo" $ \o -> do
    tiDType <- o .: "dtype"
    tiShape <- o .: "shape"
    offsets <- o .: "data_offsets"
    tiDataOffsets <- case offsets of
      [start, end] -> pure (start, end)
      _ -> fail "data_offsets must be [start, end]"
    pure TensorInfo{..}

-- | Loaded SafeTensors file
data SafeTensorsFile = SafeTensorsFile
  { stTensors :: Map Text TensorInfo
  , stData :: ByteString  -- Raw binary data
  } deriving (Show)

-- | Load a SafeTensors file from disk
loadSafeTensors :: FilePath -> IO SafeTensorsFile
loadSafeTensors path = do
  -- Read entire file
  fileData <- BS.readFile path

  -- Parse header length (first 8 bytes, little-endian u64)
  let headerLenBytes = BS.take 8 fileData
      headerLen = word64LE headerLenBytes

  -- Extract JSON header
  let headerBytes = BS.take (fromIntegral headerLen) (BS.drop 8 fileData)

  -- Parse JSON
  case Aeson.decodeStrict headerBytes of
    Nothing -> error $ "Failed to parse SafeTensors header: " ++ path
    Just obj -> do
      -- Parse tensor metadata
      tensors <- case Aeson.parseEither parseTensors obj of
        Left err -> error $ "Failed to parse tensor metadata: " ++ err
        Right t -> pure t

      -- Extract binary data (after header)
      let dataOffset = 8 + fromIntegral headerLen
          stData = BS.drop dataOffset fileData

      pure SafeTensorsFile { stTensors = tensors, stData = stData }

  where
    -- Parse little-endian u64
    word64LE :: ByteString -> Word64
    word64LE bs
      | BS.length bs < 8 = error "Not enough bytes for Word64"
      | otherwise =
          let [b0, b1, b2, b3, b4, b5, b6, b7] = map fromIntegral $ BS.unpack $ BS.take 8 bs
          in b0 + (b1 `shiftL` 8) + (b2 `shiftL` 16) + (b3 `shiftL` 24)
             + (b4 `shiftL` 32) + (b5 `shiftL` 40) + (b6 `shiftL` 48) + (b7 `shiftL` 56)

    parseTensors :: Aeson.Object -> Aeson.Parser (Map Text TensorInfo)
    parseTensors obj = do
      let pairs = KM.toList obj
      fmap Map.fromList $ forM pairs $ \(key, value) -> do
        let name = Key.toText key
        if name == "__metadata__"
          then pure (name, TensorInfo F32 [] (0, 0))  -- Skip metadata
          else do
            info <- Aeson.parseJSON value
            pure (name, info)

-- | Get a tensor as a Float vector
getTensor :: SafeTensorsFile -> Text -> IO (Vector Float)
getTensor SafeTensorsFile{..} name = case Map.lookup name stTensors of
  Nothing -> error $ "Tensor not found: " ++ T.unpack name
  Just TensorInfo{..} -> do
    let (start, end) = tiDataOffsets
        numBytes = end - start
        tensorBytes = BS.take numBytes (BS.drop start stData)

    case tiDType of
      F32 -> do
        -- Convert ByteString to Vector Float
        let numElements = numBytes `div` sizeOf (undefined :: Float)
        -- CRITICAL FIX: Copy data immediately to avoid use-after-free
        -- ByteString pointer is only valid during useAsCString callback
        BS.useAsCString tensorBytes $ \ptr -> do
          let floatPtr = castPtr ptr :: Ptr Float
          -- Force immediate copy of all elements
          V.generateM numElements $ \i -> peekElemOff floatPtr i

      F16 -> do
        -- Convert ByteString to Vector Float (FP16 stored as 16-bit, read and convert to Float)
        -- Note: WebGPU will handle the F16 format, we just read the bytes
        let numElements = numBytes `div` 2  -- 2 bytes per f16
        -- CRITICAL FIX: Copy data immediately to avoid use-after-free
        -- ByteString pointer is only valid during useAsCString callback
        BS.useAsCString tensorBytes $ \ptr -> do
          -- For now, we convert F16 to F32 on load
          -- TODO: Keep as F16 for better performance
          let halfPtr = castPtr ptr :: Ptr Word16
          V.generateM numElements $ \i -> do
            halfBits <- peekElemOff halfPtr i
            pure $ halfToFloat halfBits

      BF16 -> do
        -- Convert BFloat16 to Float32
        -- BF16 format: 1 sign bit, 8 exponent bits, 7 mantissa bits
        -- BF16 is essentially truncated FP32 (just drop the lower 16 bits of mantissa)
        let numElements = numBytes `div` 2  -- 2 bytes per bf16
        BS.useAsCString tensorBytes $ \ptr -> do
          let bf16Ptr = castPtr ptr :: Ptr Word16
          V.generateM numElements $ \i -> do
            bf16Bits <- peekElemOff bf16Ptr i
            pure $ bfloat16ToFloat bf16Bits

      _ -> error $ "Unsupported dtype for getTensor: " ++ show tiDType ++ " (only F32, F16, and BF16 supported)"

-- | Get tensor data as FP16 (Word16) without converting to Float
-- Returns Nothing if tensor is not FP16 format
-- This is useful for directly uploading FP16 data to GPU
getTensorFP16 :: SafeTensorsFile -> Text -> IO (Maybe (Vector Word16))
getTensorFP16 SafeTensorsFile{..} name = case Map.lookup name stTensors of
  Nothing -> error $ "Tensor not found: " ++ T.unpack name
  Just TensorInfo{tiDType = F16, tiDataOffsets = (start, end)} -> do
    let numBytes = end - start
        tensorBytes = BS.take numBytes (BS.drop start stData)
        numElements = numBytes `div` 2  -- 2 bytes per f16
    -- CRITICAL FIX: Copy data immediately to avoid use-after-free
    -- ByteString pointer is only valid during useAsCString callback
    vec <- BS.useAsCString tensorBytes $ \ptr -> do
      let halfPtr = castPtr ptr :: Ptr Word16
      -- Force immediate copy of all elements
      V.generateM numElements $ \i -> peekElemOff halfPtr i
    pure (Just vec)
  Just _ -> pure Nothing  -- Not FP16 format

-- | Get a tensor with type application to specify the desired output type
--
-- Usage:
--
-- > -- Load FP32 tensor as Float
-- > weights32 <- getTensorTyped @GPU.F32 st "model.weight"
-- > -- weights32 :: Vector Float
-- >
-- > -- Load FP16 tensor as Half (preserving FP16 precision)
-- > weights16 <- getTensorTyped @GPU.F16 st "model.weight"
-- > -- weights16 :: Vector Half
--
-- This uses TypeApplications to specify which dtype you want at the call site,
-- and the compiler infers the correct Vector element type from the HasNumTypeForSafeTensors instance.
--
-- The function will error if you try to load a tensor with a dtype that doesn't match
-- what's stored in the file (e.g., trying to load an FP32 tensor as @GPU.F16).
getTensorTyped :: forall (dtype :: GPU.NumType) a. HasNumTypeForSafeTensors dtype a
               => SafeTensorsFile -> Text -> IO (Vector a)
getTensorTyped sf@SafeTensorsFile{..} name = case Map.lookup name stTensors of
  Nothing -> error $ "Tensor not found: " ++ T.unpack name
  Just ti -> loadTensorAs @dtype @a sf name ti

-- | Type class mapping NumType to Haskell types for SafeTensors loading
class Storable a => HasNumTypeForSafeTensors (dtype :: GPU.NumType) a | dtype -> a where
  loadTensorAs :: SafeTensorsFile -> Text -> TensorInfo -> IO (Vector a)

-- | Load FP32 tensor as Float
instance HasNumTypeForSafeTensors 'GPU.F32 Float where
  loadTensorAs SafeTensorsFile{..} name TensorInfo{..} = case tiDType of
    F32 -> do
      let (start, end) = tiDataOffsets
          numBytes = end - start
          tensorBytes = BS.take numBytes (BS.drop start stData)
          numElements = numBytes `div` sizeOf (undefined :: Float)
      -- CRITICAL FIX: Copy data immediately to avoid use-after-free
      -- ByteString pointer is only valid during useAsCString callback
      BS.useAsCString tensorBytes $ \ptr -> do
        let floatPtr = castPtr ptr :: Ptr Float
        -- Force immediate copy of all elements
        V.generateM numElements $ \i -> peekElemOff floatPtr i
    _ -> error $ "Tensor " ++ T.unpack name ++ " is " ++ show tiDType ++ ", not F32"

-- | Load FP16 tensor as Half (keeping FP16 precision)
instance HasNumTypeForSafeTensors 'GPU.F16 Half where
  loadTensorAs SafeTensorsFile{..} name TensorInfo{..} = case tiDType of
    F16 -> do
      let (start, end) = tiDataOffsets
          numBytes = end - start
          tensorBytes = BS.take numBytes (BS.drop start stData)
          numElements = numBytes `div` 2  -- 2 bytes per f16
      -- CRITICAL FIX: Copy data immediately to avoid use-after-free
      -- ByteString pointer is only valid during useAsCString callback
      BS.useAsCString tensorBytes $ \ptr -> do
        let halfPtr = castPtr ptr :: Ptr Word16
        -- Read as Word16 and wrap in Half newtype, forcing immediate copy
        V.generateM numElements $ \i -> do
          w16 <- peekElemOff halfPtr i
          pure (Half w16)
    _ -> error $ "Tensor " ++ T.unpack name ++ " is " ++ show tiDType ++ ", not F16"

-- | Get the data type of a tensor
getTensorDType :: SafeTensorsFile -> Text -> DType
getTensorDType SafeTensorsFile{..} name = case Map.lookup name stTensors of
  Nothing -> error $ "Tensor not found: " ++ T.unpack name
  Just TensorInfo{..} -> tiDType

-- | Get the shape of a tensor
getTensorShape :: SafeTensorsFile -> Text -> [Int]
getTensorShape SafeTensorsFile{..} name = case Map.lookup name stTensors of
  Nothing -> error $ "Tensor not found: " ++ T.unpack name
  Just TensorInfo{..} -> tiShape

-- | List all tensor names in the file
listTensors :: SafeTensorsFile -> [Text]
listTensors SafeTensorsFile{..} = Map.keys stTensors

-- | Check if a tensor exists
hasTensor :: SafeTensorsFile -> Text -> Bool
hasTensor SafeTensorsFile{..} name = Map.member name stTensors

-- | Calculate total number of elements in a tensor
tensorSize :: [Int] -> Int
tensorSize = product

-- | Convert IEEE 754 binary16 (half precision) to Float
--
-- Format: 1 sign bit, 5 exponent bits, 10 mantissa bits
-- Reference: https://en.wikipedia.org/wiki/Half-precision_floating-point_format
halfToFloat :: Word16 -> Float
halfToFloat h = unsafePerformIO $ do
  allocaBytes 4 $ \ptr -> do
    let sign = (h `shiftR` 15) .&. 0x1
        exponent = (h `shiftR` 10) .&. 0x1F
        mantissa = h .&. 0x3FF

        sign32 = (fromIntegral sign :: Word32) `shiftL` 31

        floatBits = if exponent == 0
          then  -- Zero or subnormal
            if mantissa == 0
              then sign32  -- Zero (positive or negative)
              else sign32  -- Treat subnormals as zero for simplicity
          else if exponent == 0x1F
            then  -- Infinity or NaN
              sign32 .|. (0xFF `shiftL` 23) .|. ((fromIntegral mantissa :: Word32) `shiftL` 13)
            else  -- Normalized value
              let exp32 = (fromIntegral exponent - 15 + 127 :: Word32) `shiftL` 23
                  mant32 = (fromIntegral mantissa :: Word32) `shiftL` 13
              in sign32 .|. exp32 .|. mant32

    poke (castPtr ptr :: Ptr Word32) floatBits
    peek (castPtr ptr :: Ptr Float)

-- | Convert BFloat16 (brain floating point) to Float
--
-- BF16 format: 1 sign bit, 8 exponent bits, 7 mantissa bits
-- It's essentially Float32 with the lower 16 bits of mantissa truncated
-- This makes conversion very simple: just shift left by 16 bits
bfloat16ToFloat :: Word16 -> Float
bfloat16ToFloat bf16 = unsafePerformIO $ do
  allocaBytes 4 $ \ptr -> do
    -- BF16 is just the upper 16 bits of FP32
    -- To convert: shift left by 16 bits to restore FP32 format
    let floatBits = (fromIntegral bf16 :: Word32) `shiftL` 16
    poke (castPtr ptr :: Ptr Word32) floatBits
    peek (castPtr ptr :: Ptr Float)

-- | Get a tensor as U32 vector (for Q4 packed weights)
getTensorU32 :: SafeTensorsFile -> Text -> IO (Vector Word32)
getTensorU32 SafeTensorsFile{..} name = case Map.lookup name stTensors of
  Nothing -> error $ "Tensor not found: " ++ T.unpack name
  Just TensorInfo{..} -> do
    let (start, end) = tiDataOffsets
        numBytes = end - start
        tensorBytes = BS.take numBytes (BS.drop start stData)
    case tiDType of
      U32 -> do
        let numElements = numBytes `div` 4  -- 4 bytes per u32
        -- CRITICAL FIX: COPY data immediately to avoid use-after-free
        -- ByteString pointer is only valid during useAsCString callback
        BS.useAsCString tensorBytes $ \ptr -> do
          let wordPtr = castPtr ptr :: Ptr Word32
          -- Force immediate copy of all elements
          V.generateM numElements $ \i -> peekElemOff wordPtr i
      I32 -> do
        -- Q4 packed data is stored as I32 for PyTorch compatibility
        -- Reinterpret I32 bits as U32 (same bit pattern, different interpretation)
        let numElements = numBytes `div` 4
        BS.useAsCString tensorBytes $ \ptr -> do
          let int32Ptr = castPtr ptr :: Ptr Int32
          V.generateM numElements $ \i -> do
            i32 <- peekElemOff int32Ptr i
            pure (fromIntegral i32 :: Word32)  -- Reinterpret bits
      _ -> error $ "getTensorU32: Expected U32 or I32 dtype, got " ++ show tiDType

-- | Check if a tensor has Q4 quantized weights
-- Q4 weights are stored as two tensors:
--   - {name}_q4_packed: U32 array (8 nibbles per Word32)
--   - {name}_q4_scales: F32 array (one scale per 32-weight block)
hasQ4Weight :: SafeTensorsFile -> Text -> Bool
hasQ4Weight st name =
  hasTensor st (name <> ".q4_packed") &&
  hasTensor st (name <> ".q4_scales")

-- | Load Q4 quantized weight pair
-- Returns: (packed nibbles, scales)
--   - packed: [out_size * in_size / 8] U32 vector
--   - scales: [out_size * in_size / 32] Float vector
loadQ4Weight :: SafeTensorsFile -> Text -> IO (Vector Word32, Vector Float)
loadQ4Weight st name = do
  packed <- getTensorU32 st (name <> ".q4_packed")
  scales <- getTensor st (name <> ".q4_scales")
  pure (packed, scales)

-- | Load Q4 weight and dequantize to Float vector
-- Convenient for loading Q4 weights into existing FP16/FP32 pipeline
loadQ4WeightDequantized :: SafeTensorsFile -> Text -> IO (Vector Float)
loadQ4WeightDequantized st name = do
  (packed, scales) <- loadQ4Weight st name
  let dequantized = dequantizeQ4Vector packed scales

  -- DEBUG: Check for NaN/Inf and extreme values in dequantized weights
  let maxVal = V.maximum dequantized
      minVal = V.minimum dequantized
      hasNaN = V.any isNaN dequantized
      hasInf = V.any isInfinite dequantized
      minScale = V.minimum scales
      maxScale = V.maximum scales
      zeroScales = V.length $ V.filter (< 1e-8) scales

  when (hasNaN || hasInf || maxVal > 1e6 || minVal < -1e6) $ do
    putStrLn $ "⚠️  Q4 Weight Issue: " ++ T.unpack name
    putStrLn $ "    Dequantized Range: [" ++ show minVal ++ ", " ++ show maxVal ++ "]"
    putStrLn $ "    Scale Range: [" ++ show minScale ++ ", " ++ show maxScale ++ "]"
    putStrLn $ "    Zero Scales: " ++ show zeroScales ++ " / " ++ show (V.length scales)
    putStrLn $ "    Has NaN: " ++ show hasNaN
    putStrLn $ "    Has Inf: " ++ show hasInf

  pure dequantized

-- | Dequantize Q4 packed weights to Float vector
-- Each U32 contains 8 nibbles (4-bit values)
-- Each block of 32 weights shares one scale
dequantizeQ4Vector :: Vector Word32 -> Vector Float -> Vector Float
dequantizeQ4Vector packed scales =
  let numScales = V.length scales
      numWeights = numScales * 32  -- 32 weights per block
  in V.generate numWeights $ \idx ->
       let blockIdx = idx `div` 32
           weightIdxInBlock = idx `mod` 32
           scale = scales V.! blockIdx

           -- Find the packed word containing this weight
           wordIdx = (blockIdx * 4) + (weightIdxInBlock `div` 8)
           nibbleIdx = weightIdxInBlock `mod` 8
           packedWord = packed V.! wordIdx

           -- Extract nibble (4 bits)
           nibble = fromIntegral $ (packedWord `shiftR` (nibbleIdx * 4)) .&. 0xF

           -- Dequantize: weight = (nibble - 7.5) * scale
       in (nibble - 7.5) * scale

-- ============================================================================
-- Saving SafeTensors Files
-- ============================================================================

-- | Tensor data for saving
data TensorData = TensorData
  { tdName :: !Text
  , tdDType :: !DType
  , tdShape :: ![Int]
  , tdData :: !ByteString
  } deriving (Show)

-- | Create tensor data from a typed vector
createTensorData :: Storable a => Text -> DType -> [Int] -> Vector a -> TensorData
createTensorData name dtype shape vec = TensorData
  { tdName = name
  , tdDType = dtype
  , tdShape = shape
  , tdData = vectorToByteString vec
  }

-- | Save tensors to SafeTensors format
saveSafeTensors :: FilePath -> [TensorData] -> IO ()
saveSafeTensors path tensors = do
  -- Calculate offsets for each tensor
  let tensorsWithOffsets = calculateOffsets tensors

  -- Build JSON header
  let header = buildHeader tensorsWithOffsets
      headerBytes = BL.toStrict $ Aeson.encode header
      headerLen = BS.length headerBytes

  -- Build file: [8-byte header length][JSON header][tensor data]
  let headerLenBytes = word64ToLE (fromIntegral headerLen)
      allTensorData = BS.concat [tdData td | (td, _, _) <- tensorsWithOffsets]
      fileData = BS.concat [headerLenBytes, headerBytes, allTensorData]

  -- Write to file
  BS.writeFile path fileData

-- | Calculate byte offsets for each tensor
calculateOffsets :: [TensorData] -> [(TensorData, Int, Int)]
calculateOffsets = go 0
  where
    go _ [] = []
    go offset (td:tds) =
      let size = BS.length (tdData td)
          end = offset + size
      in (td, offset, end) : go end tds

-- | Build JSON header with tensor metadata
buildHeader :: [(TensorData, Int, Int)] -> Aeson.Value
buildHeader tensors =
  let tensorObjects = [ (Key.fromText (tdName td),
                         Aeson.object
                           [ ("dtype", dtypeToJSON (tdDType td))
                           , ("shape", Aeson.toJSON (tdShape td))
                           , ("data_offsets", Aeson.toJSON [start, end])
                           ])
                      | (td, start, end) <- tensors
                      ]
  in Aeson.object tensorObjects

-- | Convert DType to JSON string
dtypeToJSON :: DType -> Aeson.Value
dtypeToJSON F32 = Aeson.String "F32"
dtypeToJSON F16 = Aeson.String "F16"
dtypeToJSON BF16 = Aeson.String "BF16"
dtypeToJSON F64 = Aeson.String "F64"
dtypeToJSON I32 = Aeson.String "I32"
dtypeToJSON I64 = Aeson.String "I64"
dtypeToJSON U32 = Aeson.String "U32"
dtypeToJSON U64 = Aeson.String "U64"

-- | Encode Word64 as little-endian bytes
word64ToLE :: Word64 -> ByteString
word64ToLE w = BS.pack
  [ fromIntegral (w .&. 0xFF)
  , fromIntegral ((w `shiftR` 8) .&. 0xFF)
  , fromIntegral ((w `shiftR` 16) .&. 0xFF)
  , fromIntegral ((w `shiftR` 24) .&. 0xFF)
  , fromIntegral ((w `shiftR` 32) .&. 0xFF)
  , fromIntegral ((w `shiftR` 40) .&. 0xFF)
  , fromIntegral ((w `shiftR` 48) .&. 0xFF)
  , fromIntegral ((w `shiftR` 56) .&. 0xFF)
  ]

-- | Convert a storable vector to ByteString
vectorToByteString :: Storable a => Vector a -> ByteString
vectorToByteString vec = unsafePerformIO $ do
  V.unsafeWith vec $ \ptr -> do
    let numBytes = V.length vec * sizeOf (V.head vec)
    BS.packCStringLen (castPtr ptr, numBytes)
