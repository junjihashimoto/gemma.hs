{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : Gemma.Tokenizer.Normalize
Description : Text normalization for SentencePiece tokenizer
Copyright   : (c) 2025
License     : BSD3
Maintainer  : your-email@example.com

Text normalization functions for preprocessing input before tokenization.
Handles whitespace escaping, dummy prefix, and NFKC normalization.
-}

module Gemma.Tokenizer.Normalize
  ( -- * Normalization
    normalize
  , normalizeWithSpec

    -- * Individual transformations
  , nfkcNormalize
  , addDummyPrefix
  , removeExtraWhitespaces
  , escapeWhitespaces
  , unescapeWhitespaces
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import Data.Char (isSpace)
import Gemma.Tokenizer.Proto (NormalizerSpec(..))

-- | Normalize text using the given normalizer specification
normalizeWithSpec :: NormalizerSpec -> Text -> Text
normalizeWithSpec spec text =
  let step1 = if nsRemoveExtraWhitespaces spec then removeExtraWhitespaces text else text
      step2 = nfkcNormalize step1  -- Always apply NFKC
      step3 = if nsAddDummyPrefix spec then addDummyPrefix step2 else step2
      step4 = if nsEscapeWhitespaces spec then escapeWhitespaces step3 else step3
  in step4

-- | Normalize text with default settings (escape whitespaces only)
normalize :: Text -> Text
normalize = escapeWhitespaces

-- | Apply NFKC Unicode normalization
-- Uses the unicode-data or unicode-transforms package for proper NFKC
-- For now, we'll use a simple implementation that just passes through
-- TODO: Add proper NFKC normalization using unicode-transforms package
nfkcNormalize :: Text -> Text
nfkcNormalize text = text  -- Placeholder - identity for now
  -- Proper implementation would use:
  -- import qualified Data.Text.Normalize as Unicode
  -- nfkcNormalize = Unicode.normalize Unicode.NFKC

-- | Add a dummy prefix (space) to the beginning of text
-- This is used by some SentencePiece models to ensure consistent tokenization
addDummyPrefix :: Text -> Text
addDummyPrefix text
  | T.null text = text
  | otherwise = T.cons ' ' text

-- | Remove extra whitespaces (collapse multiple spaces into one)
removeExtraWhitespaces :: Text -> Text
removeExtraWhitespaces text =
  T.unwords $ filter (not . T.null) $ T.words text

-- | Escape whitespaces by replacing them with underscore '▁' (U+2581)
-- This is the standard SentencePiece whitespace marker
-- Note: Only spaces and tabs are escaped, newlines are preserved
escapeWhitespaces :: Text -> Text
escapeWhitespaces = T.map (\c -> if c == ' ' || c == '\t' then '▁' else c)

-- | Unescape whitespaces by replacing underscore '▁' back to space
-- Used during decoding
unescapeWhitespaces :: Text -> Text
unescapeWhitespaces = T.map (\c -> if c == '▁' then ' ' else c)
