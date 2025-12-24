{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : Gemma.Tokenizer.Decode
Description : Decode token IDs back to text
Copyright   : (c) 2025
License     : BSD3
Maintainer  : your-email@example.com

Decodes token IDs back into text by concatenating the token pieces
and unescaping whitespaces.
-}

module Gemma.Tokenizer.Decode
  ( -- * Decoding
    decode
  , decodeWithPieces
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import Data.Vector (Vector)
import qualified Data.Vector as V
import Gemma.Tokenizer.Proto (SentencePiece(..), PieceType(..))
import Gemma.Tokenizer.Normalize (unescapeWhitespaces)

-- | Decode token IDs to text
decode :: Vector SentencePiece -> [Int] -> Text
decode pieces ids = decodeWithPieces pieces ids

-- | Decode token IDs using the piece vector
decodeWithPieces :: Vector SentencePiece -> [Int] -> Text
decodeWithPieces pieces ids =
  let tokens = map (getPieceText pieces) ids
      concatenated = T.concat tokens
      -- Unescape whitespaces (replace ▁ with space)
      unescaped = unescapeWhitespaces concatenated
  in unescaped

-- | Get the text for a token ID, handling out-of-bounds
getPieceText :: Vector SentencePiece -> Int -> Text
getPieceText pieces idx
  | idx >= 0 && idx < V.length pieces =
      let piece = pieces V.! idx
      in case spType piece of
           CONTROL -> ""  -- Skip control tokens in output
           UNKNOWN -> ""  -- Skip unknown tokens
           _ -> spPiece piece
  | otherwise = ""  -- Out of bounds, return empty
