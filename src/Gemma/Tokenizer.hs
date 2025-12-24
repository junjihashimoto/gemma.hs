{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : Gemma.Tokenizer
Description : High-level tokenizer API for Gemma models
Copyright   : (c) 2025
License     : BSD3
Maintainer  : your-email@example.com

Main tokenizer interface for encoding and decoding text.
This is a pure Haskell implementation of SentencePiece tokenization.
-}

module Gemma.Tokenizer
  ( -- * Tokenizer
    Tokenizer(..)
  , loadTokenizer

    -- * Encoding and Decoding
  , encode
  , decode

    -- * Special tokens
  , bosId
  , eosId
  , unkId
  , padId

    -- * Vocabulary
  , vocabSize
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import Data.Vector (Vector)
import qualified Data.Vector as V
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Gemma.Tokenizer.Proto
import qualified Gemma.Tokenizer.Normalize as Norm
import qualified Gemma.Tokenizer.BPE as BPE
import qualified Gemma.Tokenizer.Decode as Dec

-- | Tokenizer with loaded model and vocabulary
data Tokenizer = Tokenizer
  { tokModel :: !ModelProto
  , tokVocab :: !BPE.Vocab
  , tokBosId :: !Int
  , tokEosId :: !Int
  , tokUnkId :: !Int
  , tokPadId :: !Int
  } deriving (Show, Eq)

-- | Load tokenizer from SentencePiece model file
loadTokenizer :: FilePath -> IO Tokenizer
loadTokenizer path = do
  model <- loadModelProto path
  let pieces = mpPieces model
      vocab = BPE.buildVocab pieces

      -- Find special token IDs
      bosIdx = findTokenId pieces "<bos>"
      eosIdx = findTokenId pieces "<eos>"
      unkIdx = findTokenId pieces "<unk>"
      padIdx = findTokenId pieces "<pad>"

  return $ Tokenizer
    { tokModel = model
    , tokVocab = vocab
    , tokBosId = bosIdx
    , tokEosId = eosIdx
    , tokUnkId = unkIdx
    , tokPadId = padIdx
    }

-- | Find the ID of a special token by name
findTokenId :: Vector SentencePiece -> Text -> Int
findTokenId pieces name =
  case V.findIndex (\p -> spPiece p == name) pieces of
    Just idx -> idx
    Nothing -> 0  -- Default to first token if not found

-- | Encode text to token IDs
-- Automatically adds BOS token at the beginning
encode :: Tokenizer -> Text -> [Int]
encode Tokenizer{..} text =
  let normalized = normalizeText tokModel text
      tokens = BPE.encode (mpPieces tokModel) tokVocab tokUnkId normalized
      -- Add BOS token at the beginning
  in tokBosId : tokens

-- | Decode token IDs to text
decode :: Tokenizer -> [Int] -> Text
decode Tokenizer{..} ids =
  let -- Remove special tokens (BOS, EOS, PAD)
      filteredIds = filter (\i -> i /= tokBosId && i /= tokEosId && i /= tokPadId) ids
  in Dec.decode (mpPieces tokModel) filteredIds

-- | Get BOS (Beginning of Sequence) token ID
bosId :: Tokenizer -> Int
bosId = tokBosId

-- | Get EOS (End of Sequence) token ID
eosId :: Tokenizer -> Int
eosId = tokEosId

-- | Get UNK (Unknown) token ID
unkId :: Tokenizer -> Int
unkId = tokUnkId

-- | Get PAD token ID
padId :: Tokenizer -> Int
padId = tokPadId

-- | Get vocabulary size
vocabSize :: Tokenizer -> Int
vocabSize Tokenizer{..} = V.length (mpPieces tokModel)

-- | Normalize text according to model configuration
normalizeText :: ModelProto -> Text -> Text
normalizeText model text =
  case mpNormalizerSpec model of
    Just spec -> Norm.normalizeWithSpec spec text
    Nothing -> Norm.normalize text  -- Default normalization
