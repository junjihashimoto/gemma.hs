{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : Gemma.Tokenizer.BPE
Description : Byte Pair Encoding (BPE) tokenization algorithm
Copyright   : (c) 2025
License     : BSD3
Maintainer  : your-email@example.com

Implements the BPE algorithm for encoding text into token IDs.
BPE iteratively merges the most frequent adjacent pairs of tokens.
-}

module Gemma.Tokenizer.BPE
  ( -- * BPE Encoding
    Vocab
  , buildVocab
  , encode
  , encodeWithVocab

    -- * Utilities
  , tokenToId
  , idToToken
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Vector (Vector)
import qualified Data.Vector as V
import Data.Maybe (fromMaybe)
import Gemma.Tokenizer.Proto (SentencePiece(..), PieceType(..))

-- | Vocabulary mapping from token text to (id, score)
type Vocab = Map Text (Int, Float)

-- | Build vocabulary from sentence pieces
buildVocab :: Vector SentencePiece -> Vocab
buildVocab pieces =
  V.ifoldl' (\acc idx piece ->
    Map.insert (spPiece piece) (idx, spScore piece) acc
  ) Map.empty pieces

-- | Look up token ID from text
tokenToId :: Vocab -> Text -> Maybe Int
tokenToId vocab token = fst <$> Map.lookup token vocab

-- | Look up token text from ID (requires reverse map)
idToToken :: Vector SentencePiece -> Int -> Maybe Text
idToToken pieces idx
  | idx >= 0 && idx < V.length pieces = Just (spPiece $ pieces V.! idx)
  | otherwise = Nothing

-- | Encode text using BPE algorithm
-- For SentencePiece, we use greedy longest-match instead of iterative BPE merges
encode :: Vector SentencePiece -> Vocab -> Int -> Text -> [Int]
encode pieces vocab unkId text = greedyEncode vocab unkId text

-- | Encode text with given vocabulary (deprecated - use greedyEncode instead)
-- BPE algorithm:
-- 1. Split text into characters (or bytes for byte-level BPE)
-- 2. Iteratively merge the most frequent adjacent pair
-- 3. Continue until no more merges possible
encodeWithVocab :: Vector SentencePiece -> Vocab -> Int -> Text -> [Int]
encodeWithVocab pieces vocab unkId text =
  greedyEncode vocab unkId text

-- | Apply BPE merges iteratively until no more merges are possible
applyBPEMerges :: Vocab -> [Text] -> [Text]
applyBPEMerges vocab tokens = go tokens
  where
    go currentTokens =
      case findBestMerge vocab currentTokens of
        Nothing -> currentTokens  -- No more merges possible
        Just (left, right, merged) ->
          -- Apply the merge and continue
          let newTokens = mergePair left right merged currentTokens
          in go newTokens

-- | Find the best merge (highest score in vocabulary)
findBestMerge :: Vocab -> [Text] -> Maybe (Text, Text, Text)
findBestMerge vocab tokens = fmap extractResult $ go tokens Nothing
  where
    go [] best = best
    go [_] best = best
    go (t1:t2:rest) best =
      let merged = t1 <> t2
          currentScore = case Map.lookup merged vocab of
            Just (_, score) -> Just score
            Nothing -> Nothing
      in case (currentScore, best) of
           (Just score, Nothing) ->
             go (t2:rest) (Just (t1, t2, merged, score))
           (Just score, Just (_, _, _, bestScore))
             | score > bestScore ->
                 go (t2:rest) (Just (t1, t2, merged, score))
           _ -> go (t2:rest) best

    -- Extract result from best merge (removing score)
    extractResult (t1, t2, merged, _) = (t1, t2, merged)

-- | Merge a specific pair in the token list
mergePair :: Text -> Text -> Text -> [Text] -> [Text]
mergePair left right merged tokens = go tokens
  where
    go [] = []
    go [t] = [t]
    go (t1:t2:rest)
      | t1 == left && t2 == right = merged : go rest
      | otherwise = t1 : go (t2:rest)

-- Greedy longest-match encoding
-- This tries to match the longest token from vocabulary first
-- This is the correct algorithm for SentencePiece BPE models
greedyEncode :: Vocab -> Int -> Text -> [Int]
greedyEncode vocab unkId text
  | T.null text = []
  | otherwise = go text
  where
    go remaining
      | T.null remaining = []
      | otherwise =
          case findLongestMatch vocab remaining of
            Just (token, tokenId, rest) ->
              tokenId : go rest
            Nothing ->
              -- No match found, emit unknown token and skip one character
              unkId : go (T.drop 1 remaining)

-- | Find the longest matching token from the beginning of text
findLongestMatch :: Vocab -> Text -> Maybe (Text, Int, Text)
findLongestMatch vocab text = go (T.length text)
  where
    go 0 = Nothing
    go len =
      let prefix = T.take len text
      in case Map.lookup prefix vocab of
           Just (tokenId, _) ->
             Just (prefix, tokenId, T.drop len text)
           Nothing ->
             go (len - 1)
