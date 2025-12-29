{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE BangPatterns #-}

{-|
Module: Gemma.GGUF
Description: Parser for GGUF (GGML Universal File Format)

This module provides functionality to load and parse .gguf files,
which store quantized model weights in the GGML format.

GGUF format:
- Header (magic, version, tensor_count, metadata_kv_count)
- Metadata key-value pairs (model hyperparameters)
- Tensor info array (name, dimensions, type, offset)
- Padding to alignment
- Tensor data (aligned binary data)

Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
-}

module Gemma.GGUF
  ( -- * Types
    GGUFFile(..)
  , GGUFHeader(..)
  , GGMLType(..)
  , MetadataValue(..)
  , TensorInfo(..)
    -- * Loading
  , loadGGUF
    -- * Querying
  , getTensor
  , getTensorRaw
  , getTensorQ4_0Raw
  , getTensorShape
  , getTensorType
  , getMetadata
  , getMetadataInt
  , getMetadataFloat
  , getMetadataString
  , listTensors
  , hasTensor
  , hasMetadata
    -- * Utilities
  , ggmlTypeSize
  , ggmlTypeBlockSize
  ) where

import qualified Data.ByteString as BS
import Data.ByteString (ByteString)
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import qualified Data.Map.Strict as Map
import Data.Map.Strict (Map)
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import Data.Word
import Data.Int
import Foreign.Ptr (Ptr, castPtr, plusPtr)
import Foreign.Storable (Storable, sizeOf, peekElemOff, peek, poke)
import Foreign.Marshal.Alloc (allocaBytes)
import Data.Bits ((.&.), (.|.), shiftL, shiftR)
import System.IO.Unsafe (unsafePerformIO)
import Control.Monad (forM, when, replicateM)
import System.IO (hFlush, stdout)

-- | GGML data types (quantization formats)
data GGMLType
  = GGML_TYPE_F32      -- 0: Float32
  | GGML_TYPE_F16      -- 1: Float16
  | GGML_TYPE_Q4_0     -- 2: 4-bit quantization (block size 32)
  | GGML_TYPE_Q4_1     -- 3: 4-bit quantization with offset (block size 32)
  | GGML_TYPE_Q5_0     -- 6: 5-bit quantization
  | GGML_TYPE_Q5_1     -- 7: 5-bit quantization with offset
  | GGML_TYPE_Q8_0     -- 8: 8-bit quantization
  | GGML_TYPE_Q8_1     -- 9: 8-bit quantization with offset
  | GGML_TYPE_Q2_K     -- 10: 2-bit K-quant
  | GGML_TYPE_Q3_K     -- 11: 3-bit K-quant
  | GGML_TYPE_Q4_K     -- 12: 4-bit K-quant
  | GGML_TYPE_Q5_K     -- 13: 5-bit K-quant
  | GGML_TYPE_Q6_K     -- 14: 6-bit K-quant
  | GGML_TYPE_Q8_K     -- 15: 8-bit K-quant
  | GGML_TYPE_IQ2_XXS  -- 16: 2-bit IQ
  | GGML_TYPE_IQ2_XS   -- 17: 2-bit IQ
  | GGML_TYPE_IQ3_XXS  -- 18: 3-bit IQ
  | GGML_TYPE_IQ1_S    -- 19: 1-bit IQ
  | GGML_TYPE_IQ4_NL   -- 20: 4-bit IQ non-linear
  | GGML_TYPE_IQ3_S    -- 21: 3-bit IQ
  | GGML_TYPE_IQ2_S    -- 22: 2-bit IQ
  | GGML_TYPE_IQ4_XS   -- 23: 4-bit IQ
  | GGML_TYPE_I8       -- 24: Int8
  | GGML_TYPE_I16      -- 25: Int16
  | GGML_TYPE_I32      -- 26: Int32
  | GGML_TYPE_I64      -- 27: Int64
  | GGML_TYPE_F64      -- 28: Float64
  | GGML_TYPE_IQ1_M    -- 29: 1-bit IQ
  | GGML_TYPE_BF16     -- 30: BFloat16
  | GGML_TYPE_TQ1_0    -- 34: Ternary 1-bit
  | GGML_TYPE_TQ2_0    -- 35: Ternary 2-bit
  | GGML_TYPE_MXFP4    -- 39: MXFP4
  deriving (Show, Eq)

-- | Convert Word32 to GGMLType
word32ToGGMLType :: Word32 -> Maybe GGMLType
word32ToGGMLType = \case
  0  -> Just GGML_TYPE_F32
  1  -> Just GGML_TYPE_F16
  2  -> Just GGML_TYPE_Q4_0
  3  -> Just GGML_TYPE_Q4_1
  6  -> Just GGML_TYPE_Q5_0
  7  -> Just GGML_TYPE_Q5_1
  8  -> Just GGML_TYPE_Q8_0
  9  -> Just GGML_TYPE_Q8_1
  10 -> Just GGML_TYPE_Q2_K
  11 -> Just GGML_TYPE_Q3_K
  12 -> Just GGML_TYPE_Q4_K
  13 -> Just GGML_TYPE_Q5_K
  14 -> Just GGML_TYPE_Q6_K
  15 -> Just GGML_TYPE_Q8_K
  16 -> Just GGML_TYPE_IQ2_XXS
  17 -> Just GGML_TYPE_IQ2_XS
  18 -> Just GGML_TYPE_IQ3_XXS
  19 -> Just GGML_TYPE_IQ1_S
  20 -> Just GGML_TYPE_IQ4_NL
  21 -> Just GGML_TYPE_IQ3_S
  22 -> Just GGML_TYPE_IQ2_S
  23 -> Just GGML_TYPE_IQ4_XS
  24 -> Just GGML_TYPE_I8
  25 -> Just GGML_TYPE_I16
  26 -> Just GGML_TYPE_I32
  27 -> Just GGML_TYPE_I64
  28 -> Just GGML_TYPE_F64
  29 -> Just GGML_TYPE_IQ1_M
  30 -> Just GGML_TYPE_BF16
  34 -> Just GGML_TYPE_TQ1_0
  35 -> Just GGML_TYPE_TQ2_0
  39 -> Just GGML_TYPE_MXFP4
  _  -> Nothing

-- | Metadata value types
data MetadataValue
  = MetaUInt8 Word8
  | MetaInt8 Int8
  | MetaUInt16 Word16
  | MetaInt16 Int16
  | MetaUInt32 Word32
  | MetaInt32 Int32
  | MetaFloat32 Float
  | MetaBool Bool
  | MetaString Text
  | MetaArray [MetadataValue]
  | MetaUInt64 Word64
  | MetaInt64 Int64
  | MetaFloat64 Double
  deriving (Show, Eq)

-- | GGUF file header
data GGUFHeader = GGUFHeader
  { ghMagic :: Word32           -- Should be 0x46554747 ("GGUF")
  , ghVersion :: Word32         -- Format version (3 for current)
  , ghTensorCount :: Word64     -- Number of tensors
  , ghMetadataKVCount :: Word64 -- Number of metadata pairs
  } deriving (Show)

-- | Tensor information
data TensorInfo = TensorInfo
  { tiName :: Text              -- Tensor name
  , tiDimensions :: [Word64]    -- Dimensions (reverse order: [dim3, dim2, dim1, dim0])
  , tiType :: GGMLType          -- Quantization type
  , tiOffset :: Word64          -- Offset in tensor_data section
  } deriving (Show)

-- | Loaded GGUF file
data GGUFFile = GGUFFile
  { gfHeader :: GGUFHeader
  , gfMetadata :: Map Text MetadataValue
  , gfTensors :: Map Text TensorInfo
  , gfData :: ByteString        -- Raw tensor data
  , gfAlignment :: Word32       -- Global alignment (from metadata, default 32)
  } deriving (Show)

-- | Load a GGUF file from disk
loadGGUF :: FilePath -> IO GGUFFile
loadGGUF path = do
  putStrLn $ "Loading GGUF file: " ++ path
  hFlush stdout
  fileData <- BS.readFile path
  let !fileSize = BS.length fileData  -- Force evaluation
  putStrLn $ "  File size: " ++ show fileSize ++ " bytes"
  hFlush stdout

  -- Parse header
  putStrLn "  Parsing header..."
  hFlush stdout
  (header, offset1) <- parseHeader fileData
  putStrLn $ "    Tensor count: " ++ show (ghTensorCount header)
  putStrLn $ "    Metadata KV count: " ++ show (ghMetadataKVCount header)
  hFlush stdout

  -- Parse metadata key-value pairs
  putStrLn "  Parsing metadata..."
  hFlush stdout
  (metadata, offset2) <- parseMetadata fileData offset1 (ghMetadataKVCount header)
  putStrLn $ "    Parsed " ++ show (Map.size metadata) ++ " metadata entries"
  hFlush stdout

  -- Get alignment from metadata (default 32)
  let alignment = case Map.lookup "general.alignment" metadata of
        Just (MetaUInt32 a) -> a
        _ -> 32
  putStrLn $ "    Alignment: " ++ show alignment
  hFlush stdout

  -- Parse tensor infos
  putStrLn "  Parsing tensor infos..."
  hFlush stdout
  (tensors, offset3) <- parseTensorInfos fileData offset2 (ghTensorCount header)
  putStrLn $ "    Parsed " ++ show (Map.size tensors) ++ " tensors"
  hFlush stdout

  -- Calculate aligned offset for tensor data
  let tensorDataOffset = alignOffset offset3 alignment
  putStrLn $ "    Tensor data offset: " ++ show tensorDataOffset

  -- Extract tensor data
  let gfData = BS.drop (fromIntegral tensorDataOffset) fileData

  putStrLn "✅ GGUF file loaded successfully!"

  pure GGUFFile
    { gfHeader = header
    , gfMetadata = metadata
    , gfTensors = tensors
    , gfData = gfData
    , gfAlignment = alignment
    }

-- | Parse GGUF header
parseHeader :: ByteString -> IO (GGUFHeader, Int)
parseHeader bs = BS.useAsCString bs $ \ptr -> do
  let p = castPtr ptr :: Ptr Word32
  magic <- peekElemOff p 0
  version <- peekElemOff p 1

  when (magic /= 0x46554747) $
    error $ "Invalid GGUF magic number: 0x" ++ showHex magic

  when (version /= 3) $
    error $ "Unsupported GGUF version: " ++ show version ++ " (only version 3 supported)"

  let p64 = castPtr (plusPtr ptr 8) :: Ptr Word64
  tensorCount <- peekElemOff p64 0
  metadataKVCount <- peekElemOff p64 1

  let !header = GGUFHeader
        { ghMagic = magic
        , ghVersion = version
        , ghTensorCount = tensorCount
        , ghMetadataKVCount = metadataKVCount
        }

  pure (header, 24)  -- 4 + 4 + 8 + 8 = 24 bytes

-- | Parse metadata key-value pairs
parseMetadata :: ByteString -> Int -> Word64 -> IO (Map Text MetadataValue, Int)
parseMetadata bs offset count = go offset (fromIntegral count :: Int) Map.empty 1
  where
    go off 0 acc _ = pure (acc, off)
    go off n acc entryNum = do
      putStrLn $ "    Parsing metadata entry " ++ show entryNum ++ "/" ++ show (fromIntegral count :: Int)
      hFlush stdout
      (key, off1) <- parseString bs off
      putStrLn $ "      Key: " ++ show key
      hFlush stdout
      (valueType, off2) <- parseWord32 bs off1
      (value, off3) <- parseMetadataValue bs off2 valueType
      go off3 (n - 1) (Map.insert key value acc) (entryNum + 1)

-- | Parse a metadata value based on type
parseMetadataValue :: ByteString -> Int -> Word32 -> IO (MetadataValue, Int)
parseMetadataValue bs offset valueType = case valueType of
  0  -> do (v, o) <- parseWord8 bs offset; pure (MetaUInt8 v, o)
  1  -> do (v, o) <- parseInt8 bs offset; pure (MetaInt8 v, o)
  2  -> do (v, o) <- parseWord16 bs offset; pure (MetaUInt16 v, o)
  3  -> do (v, o) <- parseInt16 bs offset; pure (MetaInt16 v, o)
  4  -> do (v, o) <- parseWord32 bs offset; pure (MetaUInt32 v, o)
  5  -> do (v, o) <- parseInt32 bs offset; pure (MetaInt32 v, o)
  6  -> do (v, o) <- parseFloat32 bs offset; pure (MetaFloat32 v, o)
  7  -> do (v, o) <- parseBool bs offset; pure (MetaBool v, o)
  8  -> do (v, o) <- parseString bs offset; pure (MetaString v, o)
  9  -> do
    (arrType, o1) <- parseWord32 bs offset
    (len, o2) <- parseWord64 bs o1
    (values, o3) <- parseArrayValues bs o2 arrType (fromIntegral len)
    pure (MetaArray values, o3)
  10 -> do (v, o) <- parseWord64 bs offset; pure (MetaUInt64 v, o)
  11 -> do (v, o) <- parseInt64 bs offset; pure (MetaInt64 v, o)
  12 -> do (v, o) <- parseFloat64 bs offset; pure (MetaFloat64 v, o)
  _  -> error $ "Unknown metadata value type: " ++ show valueType

-- | Parse array values
parseArrayValues :: ByteString -> Int -> Word32 -> Int -> IO ([MetadataValue], Int)
parseArrayValues bs offset elemType count = go offset count []
  where
    go off 0 acc = pure (reverse acc, off)
    go off n acc = do
      (value, off1) <- parseMetadataValue bs off elemType
      go off1 (n - 1) (value : acc)

-- | Parse tensor infos
parseTensorInfos :: ByteString -> Int -> Word64 -> IO (Map Text TensorInfo, Int)
parseTensorInfos bs offset count = go offset (fromIntegral count :: Int) Map.empty
  where
    go off 0 acc = pure (acc, off)
    go off n acc = do
      (name, off1) <- parseString bs off
      (nDims, off2) <- parseWord32 bs off1
      (dims, off3) <- parseDimensions bs off2 (fromIntegral nDims)
      (typeVal, off4) <- parseWord32 bs off3
      (offset', off5) <- parseWord64 bs off4

      let tensorType = case word32ToGGMLType typeVal of
            Just t -> t
            Nothing -> error $ "Unknown GGML type: " ++ show typeVal

      let !info = TensorInfo
            { tiName = name
            , tiDimensions = dims
            , tiType = tensorType
            , tiOffset = offset'
            }
      go off5 (n - 1) (Map.insert name info acc)

-- | Parse dimensions array
parseDimensions :: ByteString -> Int -> Int -> IO ([Word64], Int)
parseDimensions bs offset count = go offset count []
  where
    go off 0 acc = pure (reverse acc, off)
    go off n acc = do
      (dim, off1) <- parseWord64 bs off
      go off1 (n - 1) (dim : acc)

-- ============================================================================
-- Primitive parsers
-- ============================================================================

parseWord8 :: ByteString -> Int -> IO (Word8, Int)
parseWord8 bs offset = BS.useAsCString bs $ \ptr -> do
  let p = plusPtr ptr offset :: Ptr Word8
  !v <- peek p
  pure (v, offset + 1)

parseInt8 :: ByteString -> Int -> IO (Int8, Int)
parseInt8 bs offset = BS.useAsCString bs $ \ptr -> do
  let p = plusPtr ptr offset :: Ptr Int8
  !v <- peek p
  pure (v, offset + 1)

parseWord16 :: ByteString -> Int -> IO (Word16, Int)
parseWord16 bs offset = BS.useAsCString bs $ \ptr -> do
  let p = castPtr (plusPtr ptr offset) :: Ptr Word16
  !v <- peek p
  pure (v, offset + 2)

parseInt16 :: ByteString -> Int -> IO (Int16, Int)
parseInt16 bs offset = BS.useAsCString bs $ \ptr -> do
  let p = castPtr (plusPtr ptr offset) :: Ptr Int16
  !v <- peek p
  pure (v, offset + 2)

parseWord32 :: ByteString -> Int -> IO (Word32, Int)
parseWord32 bs offset = BS.useAsCString bs $ \ptr -> do
  let p = castPtr (plusPtr ptr offset) :: Ptr Word32
  !v <- peek p
  pure (v, offset + 4)

parseInt32 :: ByteString -> Int -> IO (Int32, Int)
parseInt32 bs offset = BS.useAsCString bs $ \ptr -> do
  let p = castPtr (plusPtr ptr offset) :: Ptr Int32
  !v <- peek p
  pure (v, offset + 4)

parseWord64 :: ByteString -> Int -> IO (Word64, Int)
parseWord64 bs offset = BS.useAsCString bs $ \ptr -> do
  let p = castPtr (plusPtr ptr offset) :: Ptr Word64
  !v <- peek p
  pure (v, offset + 8)

parseInt64 :: ByteString -> Int -> IO (Int64, Int)
parseInt64 bs offset = BS.useAsCString bs $ \ptr -> do
  let p = castPtr (plusPtr ptr offset) :: Ptr Int64
  !v <- peek p
  pure (v, offset + 8)

parseFloat32 :: ByteString -> Int -> IO (Float, Int)
parseFloat32 bs offset = BS.useAsCString bs $ \ptr -> do
  let p = castPtr (plusPtr ptr offset) :: Ptr Float
  !v <- peek p
  pure (v, offset + 4)

parseFloat64 :: ByteString -> Int -> IO (Double, Int)
parseFloat64 bs offset = BS.useAsCString bs $ \ptr -> do
  let p = castPtr (plusPtr ptr offset) :: Ptr Double
  !v <- peek p
  pure (v, offset + 8)

parseBool :: ByteString -> Int -> IO (Bool, Int)
parseBool bs offset = BS.useAsCString bs $ \ptr -> do
  let p = plusPtr ptr offset :: Ptr Word8
  !v <- peek p
  pure (v /= 0, offset + 1)

parseString :: ByteString -> Int -> IO (Text, Int)
parseString bs offset = do
  (len, offset1) <- parseWord64 bs offset
  let strBytes = BS.take (fromIntegral len) (BS.drop offset1 bs)
      !text = TE.decodeUtf8 strBytes
  pure (text, offset1 + fromIntegral len)

-- ============================================================================
-- Utility functions
-- ============================================================================

-- | Align offset to next multiple of alignment
alignOffset :: Int -> Word32 -> Int
alignOffset offset alignment =
  let align = fromIntegral alignment
      remainder = offset `mod` align
  in if remainder == 0
     then offset
     else offset + (align - remainder)

-- | Show hex value
showHex :: Word32 -> String
showHex w = go w []
  where
    go 0 [] = "0"
    go 0 acc = acc
    go n acc =
      let digit = n .&. 0xF
          c = if digit < 10 then toEnum (fromEnum '0' + fromIntegral digit)
                            else toEnum (fromEnum 'A' + fromIntegral (digit - 10))
      in go (n `shiftR` 4) (c : acc)

-- ============================================================================
-- Query functions
-- ============================================================================

-- | Get a tensor as Float vector (dequantize if necessary)
getTensor :: GGUFFile -> Text -> IO (Vector Float)
getTensor gf@GGUFFile{..} name = case Map.lookup name gfTensors of
  Nothing -> error $ "Tensor not found: " ++ T.unpack name
  Just ti@TensorInfo{..} -> case tiType of
    GGML_TYPE_F32 -> loadF32Tensor gf ti
    GGML_TYPE_F16 -> loadF16Tensor gf ti
    GGML_TYPE_Q4_0 -> loadQ4_0Tensor gf ti
    GGML_TYPE_Q4_1 -> loadQ4_1Tensor gf ti
    _ -> error $ "Unsupported tensor type for getTensor: " ++ show tiType

-- | Get raw tensor data as ByteString (no dequantization)
getTensorRaw :: GGUFFile -> Text -> IO ByteString
getTensorRaw GGUFFile{..} name = case Map.lookup name gfTensors of
  Nothing -> error $ "Tensor not found: " ++ T.unpack name
  Just TensorInfo{..} -> do
    let numElements = product (map fromIntegral tiDimensions)
        numBlocks = case tiType of
          GGML_TYPE_Q4_0 -> (numElements + 31) `div` 32
          GGML_TYPE_Q4_1 -> (numElements + 31) `div` 32
          GGML_TYPE_Q5_0 -> (numElements + 31) `div` 32
          GGML_TYPE_Q5_1 -> (numElements + 31) `div` 32
          GGML_TYPE_Q8_0 -> (numElements + 31) `div` 32
          _ -> numElements
        blockSize = ggmlTypeSize tiType
        tensorSize = numBlocks * blockSize
        offset = fromIntegral tiOffset
    pure $ BS.take tensorSize (BS.drop offset gfData)

-- | Get Q4_0 tensor as Word32 vector for GPU usage
getTensorQ4_0Raw :: GGUFFile -> Text -> IO (Vector Word32)
getTensorQ4_0Raw gf@GGUFFile{..} name = case Map.lookup name gfTensors of
  Nothing -> error $ "Tensor not found: " ++ T.unpack name
  Just ti@TensorInfo{..} -> case tiType of
    GGML_TYPE_Q4_0 -> do
      rawBytes <- getTensorRaw gf name
      let bytes = BS.unpack rawBytes
          numWord32 = (length bytes + 3) `div` 4  -- Round up
      V.generateM numWord32 $ \i -> do
        let offset = i * 4
            b0 = if offset < length bytes then fromIntegral (bytes !! offset) else 0
            b1 = if offset + 1 < length bytes then fromIntegral (bytes !! (offset + 1)) else 0
            b2 = if offset + 2 < length bytes then fromIntegral (bytes !! (offset + 2)) else 0
            b3 = if offset + 3 < length bytes then fromIntegral (bytes !! (offset + 3)) else 0
            word32 = b0 .|. (b1 `shiftL` 8) .|. (b2 `shiftL` 16) .|. (b3 `shiftL` 24)
        pure word32
    _ -> error $ "Tensor is not Q4_0: " ++ T.unpack name ++ " (type: " ++ show tiType ++ ")"

-- | Load FP32 tensor
loadF32Tensor :: GGUFFile -> TensorInfo -> IO (Vector Float)
loadF32Tensor GGUFFile{..} TensorInfo{..} = do
  let numElements = product (map fromIntegral tiDimensions)
      offset = fromIntegral tiOffset
      tensorBytes = BS.drop offset gfData

  BS.useAsCString tensorBytes $ \ptr -> do
    let floatPtr = castPtr ptr :: Ptr Float
    V.generateM numElements $ \i -> peekElemOff floatPtr i

-- | Load FP16 tensor and convert to Float
loadF16Tensor :: GGUFFile -> TensorInfo -> IO (Vector Float)
loadF16Tensor GGUFFile{..} TensorInfo{..} = do
  let numElements = product (map fromIntegral tiDimensions)
      offset = fromIntegral tiOffset
      tensorBytes = BS.drop offset gfData

  BS.useAsCString tensorBytes $ \ptr -> do
    let halfPtr = castPtr ptr :: Ptr Word16
    V.generateM numElements $ \i -> do
      halfBits <- peekElemOff halfPtr i
      pure $ halfToFloat halfBits

-- | Load Q4_0 tensor and dequantize to Float
--
-- Q4_0 block structure (matching llama.cpp):
-- - 2 bytes: FP16 scale
-- - 16 bytes: Packed nibbles (32 weights, 2 per byte)
-- - Nibble layout: Elements 0-15 are low nibbles, 16-31 are high nibbles
loadQ4_0Tensor :: GGUFFile -> TensorInfo -> IO (Vector Float)
loadQ4_0Tensor GGUFFile{..} TensorInfo{..} = do
  let numElements = product (map fromIntegral tiDimensions)
      numBlocks = (numElements + 31) `div` 32  -- Round up to blocks of 32
      offset = fromIntegral tiOffset
      tensorBytes = BS.drop offset gfData

  -- Q4_0 block structure: 1 FP16 scale + 16 bytes of 4-bit weights (32 weights)
  -- Block size: 2 + 16 = 18 bytes
  BS.useAsCString tensorBytes $ \ptr -> do
    V.generateM numElements $ \i -> do
      let blockIdx = i `div` 32
          posInBlock = i `mod` 32
          blockOffset = blockIdx * 18

          -- Read scale (FP16 at start of block)
          scalePtr = castPtr (plusPtr ptr blockOffset) :: Ptr Word16
      scaleBits <- peek scalePtr
      let scale = halfToFloat scaleBits

          -- Read nibble (matching llama.cpp layout):
          -- Elements 0-15: low nibbles of bytes 0-15
          -- Elements 16-31: high nibbles of bytes 0-15
          nibbleByteIdx = if posInBlock < 16
                          then posInBlock
                          else posInBlock - 16
          nibbleByteOffset = blockOffset + 2 + nibbleByteIdx
          nibblePtr = plusPtr ptr nibbleByteOffset :: Ptr Word8
      nibbleByte <- peek nibblePtr
      let nibble = if posInBlock < 16
                   then nibbleByte .&. 0x0F        -- Low nibble
                   else (nibbleByte `shiftR` 4) .&. 0x0F  -- High nibble

          -- Dequantize: weight = (nibble - 8) * scale
      pure $ (fromIntegral nibble - 8.0) * scale

-- | Load Q4_1 tensor and dequantize to Float
loadQ4_1Tensor :: GGUFFile -> TensorInfo -> IO (Vector Float)
loadQ4_1Tensor GGUFFile{..} TensorInfo{..} = do
  let numElements = product (map fromIntegral tiDimensions)
      numBlocks = (numElements + 31) `div` 32  -- Round up to blocks of 32
      offset = fromIntegral tiOffset
      tensorBytes = BS.drop offset gfData

  -- Q4_1 block structure: 1 FP16 scale + 1 FP16 min + 16 bytes of 4-bit weights
  -- Block size: 2 + 2 + 16 = 20 bytes
  BS.useAsCString tensorBytes $ \ptr -> do
    V.generateM numElements $ \i -> do
      let blockIdx = i `div` 32
          withinBlock = i `mod` 32
          blockOffset = blockIdx * 20

          -- Read scale and min (FP16)
          scalePtr = castPtr (plusPtr ptr blockOffset) :: Ptr Word16
      scaleBits <- peek scalePtr
      minBits <- peekElemOff scalePtr 1
      let scale = halfToFloat scaleBits
          minVal = halfToFloat minBits

          -- Read nibble
          nibbleByteOffset = blockOffset + 4 + (withinBlock `div` 2)
          nibblePtr = plusPtr ptr nibbleByteOffset :: Ptr Word8
      nibbleByte <- peek nibblePtr
      let nibble = if withinBlock `mod` 2 == 0
                   then nibbleByte .&. 0x0F
                   else (nibbleByte `shiftR` 4) .&. 0x0F

          -- Dequantize: weight = nibble * scale + min
      pure $ fromIntegral nibble * scale + minVal

-- | Convert IEEE 754 binary16 (half precision) to Float
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
              then sign32  -- Zero
              else sign32  -- Treat subnormals as zero
          else if exponent == 0x1F
            then  -- Infinity or NaN
              sign32 .|. (0xFF `shiftL` 23) .|. ((fromIntegral mantissa :: Word32) `shiftL` 13)
            else  -- Normalized value
              let exp32 = (fromIntegral exponent - 15 + 127 :: Word32) `shiftL` 23
                  mant32 = (fromIntegral mantissa :: Word32) `shiftL` 13
              in sign32 .|. exp32 .|. mant32

    poke (castPtr ptr :: Ptr Word32) floatBits
    peek (castPtr ptr :: Ptr Float)

-- | Get tensor shape
getTensorShape :: GGUFFile -> Text -> [Word64]
getTensorShape GGUFFile{..} name = case Map.lookup name gfTensors of
  Nothing -> error $ "Tensor not found: " ++ T.unpack name
  Just TensorInfo{..} -> tiDimensions

-- | Get tensor type
getTensorType :: GGUFFile -> Text -> GGMLType
getTensorType GGUFFile{..} name = case Map.lookup name gfTensors of
  Nothing -> error $ "Tensor not found: " ++ T.unpack name
  Just TensorInfo{..} -> tiType

-- | Get metadata value
getMetadata :: GGUFFile -> Text -> Maybe MetadataValue
getMetadata GGUFFile{..} key = Map.lookup key gfMetadata

-- | Get metadata as Int64
getMetadataInt :: GGUFFile -> Text -> Maybe Int64
getMetadataInt gf key = case getMetadata gf key of
  Just (MetaInt64 v) -> Just v
  Just (MetaInt32 v) -> Just (fromIntegral v)
  Just (MetaInt16 v) -> Just (fromIntegral v)
  Just (MetaInt8 v) -> Just (fromIntegral v)
  Just (MetaUInt64 v) -> Just (fromIntegral v)
  Just (MetaUInt32 v) -> Just (fromIntegral v)
  Just (MetaUInt16 v) -> Just (fromIntegral v)
  Just (MetaUInt8 v) -> Just (fromIntegral v)
  _ -> Nothing

-- | Get metadata as Float
getMetadataFloat :: GGUFFile -> Text -> Maybe Float
getMetadataFloat gf key = case getMetadata gf key of
  Just (MetaFloat32 v) -> Just v
  Just (MetaFloat64 v) -> Just (realToFrac v)
  _ -> Nothing

-- | Get metadata as String
getMetadataString :: GGUFFile -> Text -> Maybe Text
getMetadataString gf key = case getMetadata gf key of
  Just (MetaString v) -> Just v
  _ -> Nothing

-- | List all tensor names
listTensors :: GGUFFile -> [Text]
listTensors GGUFFile{..} = Map.keys gfTensors

-- | Check if tensor exists
hasTensor :: GGUFFile -> Text -> Bool
hasTensor GGUFFile{..} name = Map.member name gfTensors

-- | Check if metadata key exists
hasMetadata :: GGUFFile -> Text -> Bool
hasMetadata GGUFFile{..} key = Map.member key gfMetadata

-- | Get size of one element of a GGML type (in bytes)
-- For quantized types, returns block size
ggmlTypeSize :: GGMLType -> Int
ggmlTypeSize GGML_TYPE_F32 = 4
ggmlTypeSize GGML_TYPE_F16 = 2
ggmlTypeSize GGML_TYPE_Q4_0 = 18  -- Block: 2 (scale) + 16 (weights)
ggmlTypeSize GGML_TYPE_Q4_1 = 20  -- Block: 2 (scale) + 2 (min) + 16 (weights)
ggmlTypeSize GGML_TYPE_Q5_0 = 22  -- Block: 2 (scale) + 4 (high bits) + 16 (low bits)
ggmlTypeSize GGML_TYPE_Q5_1 = 24  -- Block: 2 (scale) + 2 (min) + 4 (high bits) + 16 (low bits)
ggmlTypeSize GGML_TYPE_Q8_0 = 34  -- Block: 2 (scale) + 32 (weights)
ggmlTypeSize GGML_TYPE_I8 = 1
ggmlTypeSize GGML_TYPE_I16 = 2
ggmlTypeSize GGML_TYPE_I32 = 4
ggmlTypeSize GGML_TYPE_I64 = 8
ggmlTypeSize GGML_TYPE_F64 = 8
ggmlTypeSize GGML_TYPE_BF16 = 2
ggmlTypeSize t = error $ "ggmlTypeSize not implemented for: " ++ show t

-- | Get block size for quantized types (number of elements per block)
ggmlTypeBlockSize :: GGMLType -> Int
ggmlTypeBlockSize GGML_TYPE_F32 = 1
ggmlTypeBlockSize GGML_TYPE_F16 = 1
ggmlTypeBlockSize GGML_TYPE_Q4_0 = 32
ggmlTypeBlockSize GGML_TYPE_Q4_1 = 32
ggmlTypeBlockSize GGML_TYPE_Q5_0 = 32
ggmlTypeBlockSize GGML_TYPE_Q5_1 = 32
ggmlTypeBlockSize GGML_TYPE_Q8_0 = 32
ggmlTypeBlockSize GGML_TYPE_I8 = 1
ggmlTypeBlockSize GGML_TYPE_I16 = 1
ggmlTypeBlockSize GGML_TYPE_I32 = 1
ggmlTypeBlockSize GGML_TYPE_I64 = 1
ggmlTypeBlockSize GGML_TYPE_F64 = 1
ggmlTypeBlockSize GGML_TYPE_BF16 = 1
ggmlTypeBlockSize t = error $ "ggmlTypeBlockSize not implemented for: " ++ show t
