{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE BangPatterns #-}

{-|
Module      : Gemma.Tokenizer.Proto
Description : Protobuf parser for SentencePiece model files
Copyright   : (c) 2025
License     : BSD3
Maintainer  : your-email@example.com

Parses SentencePiece .spm model files (protobuf format) used by Gemma tokenizers.
-}

module Gemma.Tokenizer.Proto
  ( -- * Types
    ModelProto(..)
  , SentencePiece(..)
  , PieceType(..)
  , ModelType(..)
  , NormalizerSpec(..)
  , TrainerSpec(..)

    -- * Loading
  , loadModelProto
  , parseModelProto
  ) where

import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as BL
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import Data.Vector (Vector)
import qualified Data.Vector as V
import Data.Word
import Data.Int
import Data.Bits
import Control.Monad
import GHC.Generics
import qualified System.IO.Unsafe
import qualified Foreign.Marshal.Alloc
import qualified Foreign.Storable
import qualified Foreign.Ptr
import Foreign.Ptr (Ptr)

-- | Type of a sentence piece
data PieceType
  = NORMAL       -- ^ Normal token
  | UNKNOWN      -- ^ Unknown token (<unk>)
  | CONTROL      -- ^ Control token (<bos>, <eos>, etc.)
  | USER_DEFINED -- ^ User-defined token
  | BYTE         -- ^ Byte fallback token
  | UNUSED       -- ^ Unused
  deriving (Show, Eq, Ord, Generic, Enum)

-- | Sentence piece with its score and type
data SentencePiece = SentencePiece
  { spPiece :: !Text      -- ^ The token string
  , spScore :: !Float     -- ^ Log probability score
  , spType  :: !PieceType -- ^ Token type
  } deriving (Show, Eq, Generic)

-- | Model training type
data ModelType
  = UNIGRAM -- ^ Unigram language model
  | BPE     -- ^ Byte Pair Encoding
  | WORD    -- ^ Whitespace-delimited
  | CHAR    -- ^ Character sequence
  deriving (Show, Eq, Ord, Generic, Enum)

-- | Trainer specification (training config)
data TrainerSpec = TrainerSpec
  { tsModelType :: !ModelType
  , tsVocabSize :: !Int
  } deriving (Show, Eq, Generic)

-- | Normalizer specification (text preprocessing)
data NormalizerSpec = NormalizerSpec
  { nsName                  :: !Text
  , nsPrecompiledCharsmap   :: !BS.ByteString
  , nsAddDummyPrefix        :: !Bool
  , nsRemoveExtraWhitespaces :: !Bool
  , nsEscapeWhitespaces     :: !Bool
  } deriving (Show, Eq, Generic)

-- | Complete SentencePiece model
data ModelProto = ModelProto
  { mpPieces        :: !(Vector SentencePiece)
  , mpTrainerSpec   :: !(Maybe TrainerSpec)
  , mpNormalizerSpec :: !(Maybe NormalizerSpec)
  } deriving (Show, Eq, Generic)

-- | Load model from file
loadModelProto :: FilePath -> IO ModelProto
loadModelProto path = do
  bs <- BS.readFile path
  case parseModelProto bs of
    Left err -> error $ "Failed to parse model: " ++ err
    Right model -> return model

-- | Parse model from bytes
parseModelProto :: BS.ByteString -> Either String ModelProto
parseModelProto bs = parseModel bs

-- Parse complete model
parseModel :: BS.ByteString -> Either String ModelProto
parseModel input = do
  (pieces, trainerSpec, normalizerSpec) <- parseFields [] Nothing Nothing input
  return $ ModelProto (V.fromList $ reverse pieces) trainerSpec normalizerSpec

-- Parse protobuf fields (accumulate pieces in list for efficiency)
parseFields :: [SentencePiece] -> Maybe TrainerSpec -> Maybe NormalizerSpec
            -> BS.ByteString -> Either String ([SentencePiece], Maybe TrainerSpec, Maybe NormalizerSpec)
parseFields pieces trainer normalizer bs
  | BS.null bs = Right (pieces, trainer, normalizer)
  | otherwise = do
      (tag, bs') <- parseVarInt bs
      let fieldNum = tag `shiftR` 3
          wireType = tag .&. 0x7
      case fieldNum of
        1 -> do  -- repeated SentencePiece pieces
          (piece, bs'') <- parseSentencePiece bs'
          parseFields (piece : pieces) trainer normalizer bs''
        2 -> do  -- optional TrainerSpec
          (len, bs'') <- parseVarInt bs'
          let (specBytes, bs''') = BS.splitAt (fromIntegral len) bs''
          case parseTrainerSpec specBytes of
            Left err -> Left err
            Right spec -> parseFields pieces (Just spec) normalizer bs'''
        3 -> do  -- optional NormalizerSpec
          (len, bs'') <- parseVarInt bs'
          let (specBytes, bs''') = BS.splitAt (fromIntegral len) bs''
          case parseNormalizerSpec specBytes of
            Left err -> Left err
            Right spec -> parseFields pieces trainer (Just spec) bs'''
        _ -> do  -- Skip unknown fields
          bs'' <- skipField wireType bs'
          parseFields pieces trainer normalizer bs''

-- Parse a single SentencePiece
parseSentencePiece :: BS.ByteString -> Either String (SentencePiece, BS.ByteString)
parseSentencePiece bs = do
  (len, bs') <- parseVarInt bs
  let (pieceBytes, bs'') = BS.splitAt (fromIntegral len) bs'
  parsePieceFields (SentencePiece "" 0.0 NORMAL) pieceBytes >>= \piece ->
    Right (piece, bs'')

parsePieceFields :: SentencePiece -> BS.ByteString -> Either String SentencePiece
parsePieceFields piece bs
  | BS.null bs = Right piece
  | otherwise = do
      (tag, bs') <- parseVarInt bs
      let fieldNum = tag `shiftR` 3
          wireType = tag .&. 0x7
      case fieldNum of
        1 -> do  -- string piece
          (len, bs'') <- parseVarInt bs'
          let (str, bs''') = BS.splitAt (fromIntegral len) bs''
          parsePieceFields (piece { spPiece = TE.decodeUtf8 str }) bs'''
        2 -> do  -- float score
          let (scoreBytes, bs'') = BS.splitAt 4 bs'
              score = bytesToFloat scoreBytes
          parsePieceFields (piece { spScore = score }) bs''
        3 -> do  -- enum type
          (typeVal, bs'') <- parseVarInt bs'
          let pieceType = toEnum (fromIntegral typeVal - 1)  -- Protobuf enums start at 1
          parsePieceFields (piece { spType = pieceType }) bs''
        _ -> do
          bs'' <- skipField wireType bs'
          parsePieceFields piece bs''

-- Parse TrainerSpec
parseTrainerSpec :: BS.ByteString -> Either String TrainerSpec
parseTrainerSpec bs = parseTrainerFields (TrainerSpec UNIGRAM 8000) bs

parseTrainerFields :: TrainerSpec -> BS.ByteString -> Either String TrainerSpec
parseTrainerFields spec bs
  | BS.null bs = Right spec
  | otherwise = do
      (tag, bs') <- parseVarInt bs
      let fieldNum = tag `shiftR` 3
          wireType = tag .&. 0x7
      case fieldNum of
        3 -> do  -- model_type
          (typeVal, bs'') <- parseVarInt bs'
          let modelType = toEnum (fromIntegral typeVal - 1)
          parseTrainerFields (spec { tsModelType = modelType }) bs''
        4 -> do  -- vocab_size
          (size, bs'') <- parseVarInt bs'
          parseTrainerFields (spec { tsVocabSize = fromIntegral size }) bs''
        _ -> do
          bs'' <- skipField wireType bs'
          parseTrainerFields spec bs''

-- Parse NormalizerSpec
parseNormalizerSpec :: BS.ByteString -> Either String NormalizerSpec
parseNormalizerSpec bs =
  parseNormalizerFields (NormalizerSpec "" BS.empty True True True) bs

parseNormalizerFields :: NormalizerSpec -> BS.ByteString -> Either String NormalizerSpec
parseNormalizerFields spec bs
  | BS.null bs = Right spec
  | otherwise = do
      (tag, bs') <- parseVarInt bs
      let fieldNum = tag `shiftR` 3
          wireType = tag .&. 0x7
      case fieldNum of
        1 -> do  -- name
          (len, bs'') <- parseVarInt bs'
          let (str, bs''') = BS.splitAt (fromIntegral len) bs''
          parseNormalizerFields (spec { nsName = TE.decodeUtf8 str }) bs'''
        2 -> do  -- precompiled_charsmap
          (len, bs'') <- parseVarInt bs'
          let (charsmap, bs''') = BS.splitAt (fromIntegral len) bs''
          parseNormalizerFields (spec { nsPrecompiledCharsmap = charsmap }) bs'''
        3 -> do  -- add_dummy_prefix
          (val, bs'') <- parseVarInt bs'
          parseNormalizerFields (spec { nsAddDummyPrefix = val /= 0 }) bs''
        4 -> do  -- remove_extra_whitespaces
          (val, bs'') <- parseVarInt bs'
          parseNormalizerFields (spec { nsRemoveExtraWhitespaces = val /= 0 }) bs''
        5 -> do  -- escape_whitespaces
          (val, bs'') <- parseVarInt bs'
          parseNormalizerFields (spec { nsEscapeWhitespaces = val /= 0 }) bs''
        _ -> do
          bs'' <- skipField wireType bs'
          parseNormalizerFields spec bs''

-- Parse variable-length integer (varint)
parseVarInt :: BS.ByteString -> Either String (Word64, BS.ByteString)
parseVarInt bs = go 0 0 bs
  where
    go !acc !shift input
      | BS.null input = Left "Unexpected end of input parsing varint"
      | otherwise =
          let byte = BS.head input
              rest = BS.tail input
              val = acc .|. ((fromIntegral (byte .&. 0x7F)) `shiftL` shift)
          in if byte .&. 0x80 == 0
             then Right (val, rest)
             else go val (shift + 7) rest

-- Skip a field based on wire type
skipField :: Word64 -> BS.ByteString -> Either String BS.ByteString
skipField wireType bs = case wireType of
  0 -> do  -- Varint
    (_, bs') <- parseVarInt bs
    Right bs'
  1 -> do  -- 64-bit (fixed64, sfixed64, double)
    if BS.length bs < 8
      then Left "Unexpected end of input skipping 64-bit field"
      else Right $ BS.drop 8 bs
  2 -> do  -- Length-delimited (string, bytes, embedded message, packed repeated)
    (len, bs') <- parseVarInt bs
    if BS.length bs' < fromIntegral len
      then Left "Unexpected end of input skipping length-delimited field"
      else Right $ BS.drop (fromIntegral len) bs'
  5 -> do  -- 32-bit (fixed32, sfixed32, float)
    if BS.length bs < 4
      then Left "Unexpected end of input skipping 32-bit field"
      else Right $ BS.drop 4 bs
  _ -> Left $ "Unknown wire type: " ++ show wireType

-- Convert 4 bytes (little-endian) to Float
bytesToFloat :: BS.ByteString -> Float
bytesToFloat bs
  | BS.length bs /= 4 = 0.0
  | otherwise =
      let b0 = fromIntegral (BS.index bs 0)
          b1 = fromIntegral (BS.index bs 1)
          b2 = fromIntegral (BS.index bs 2)
          b3 = fromIntegral (BS.index bs 3)
          word32 = b0 .|. (b1 `shiftL` 8) .|. (b2 `shiftL` 16) .|. (b3 `shiftL` 24)
      in word32ToFloat word32

-- Convert Word32 bit pattern to Float (unsafe coercion via memory)
word32ToFloat :: Word32 -> Float
word32ToFloat w = System.IO.Unsafe.unsafePerformIO $ do
  Foreign.Marshal.Alloc.allocaBytes 4 $ \ptr -> do
    Foreign.Storable.poke (Foreign.Ptr.castPtr ptr :: Ptr Word32) w
    Foreign.Storable.peek (Foreign.Ptr.castPtr ptr :: Ptr Float)
