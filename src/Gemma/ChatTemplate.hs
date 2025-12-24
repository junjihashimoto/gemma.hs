{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE OverloadedStrings #-}

{-|
Module      : Gemma.ChatTemplate
Description : Chat template formatting for Gemma models
Copyright   : (c) 2025
License     : BSD3
Maintainer  : your-email@example.com

Implements chat template formatting matching gemma.cpp's behavior.
Handles special markers like <start_of_turn>, <end_of_turn> for
multi-turn conversations.
-}

module Gemma.ChatTemplate
  ( -- * Chat Template
    ChatTemplate(..)
  , loadChatTemplate
  , buildChatTemplate

    -- * Formatting
  , PromptWrapping(..)
  , wrapAndTokenize
  , formatUserTurn
  , formatModelTurn
  , formatConversation
  , buildInferencePrompt

    -- * Special markers
  , startOfTurnMarker
  , endOfTurnMarker
  , userMarker
  , modelMarker
  ) where

import Data.Text (Text)
import qualified Data.Text as T
import Gemma.Tokenizer

-- | Chat template special markers
data ChatTemplate = ChatTemplate
  { ctStartOfTurn :: ![Int]       -- ^ "<start_of_turn>" token IDs
  , ctEndOfTurn   :: ![Int]       -- ^ "<end_of_turn>" token IDs
  , ctUserMarker  :: ![Int]       -- ^ "user\n" token IDs
  , ctModelMarker :: ![Int]       -- ^ "model\n" token IDs
  } deriving (Show, Eq)

-- | Prompt wrapping modes
data PromptWrapping
  = NoWrap      -- ^ No wrapping, just tokenize
  | WrapUser    -- ^ Wrap as user turn
  | WrapModel   -- ^ Wrap as model turn
  deriving (Show, Eq, Ord, Enum)

-- | Special marker strings
startOfTurnMarker :: Text
startOfTurnMarker = "<start_of_turn>"

endOfTurnMarker :: Text
endOfTurnMarker = "<end_of_turn>"

userMarker :: Text
userMarker = "user\n"

modelMarker :: Text
modelMarker = "model\n"

-- | Load and build chat template from tokenizer
loadChatTemplate :: Tokenizer -> ChatTemplate
loadChatTemplate = buildChatTemplate

-- | Build chat template by pre-encoding special markers
buildChatTemplate :: Tokenizer -> ChatTemplate
buildChatTemplate tokenizer =
  ChatTemplate
    { ctStartOfTurn = encodeWithoutBOS tokenizer startOfTurnMarker
    , ctEndOfTurn   = encodeWithoutBOS tokenizer endOfTurnMarker
    , ctUserMarker  = encodeWithoutBOS tokenizer userMarker
    , ctModelMarker = encodeWithoutBOS tokenizer modelMarker
    }

-- | Encode text without automatic BOS token prepending
encodeWithoutBOS :: Tokenizer -> Text -> [Int]
encodeWithoutBOS tokenizer text =
  let tokens = encode tokenizer text
  in case tokens of
       (x:xs) | x == bosId tokenizer -> xs  -- Remove BOS if present
       _ -> tokens

-- | Wrap and tokenize prompt with chat template
-- This matches gemma.cpp's WrapAndTokenize function
wrapAndTokenize :: Tokenizer -> ChatTemplate -> PromptWrapping
                -> Int -> Text -> [Int]
wrapAndTokenize tokenizer template wrapping pos prompt =
  case wrapping of
    NoWrap -> encode tokenizer prompt

    WrapUser ->
      -- Gemma 3 uses control tokens instead of BOS token in chat format
      formatUserTurn tokenizer template prompt

    WrapModel ->
      -- Gemma 3 uses control tokens instead of BOS token in chat format
      formatModelTurn tokenizer template prompt

-- | Format a user turn
-- Format: <start_of_turn>user\n{prompt}<end_of_turn>\n
formatUserTurn :: Tokenizer -> ChatTemplate -> Text -> [Int]
formatUserTurn tokenizer ChatTemplate{..} prompt =
  ctStartOfTurn
    ++ ctUserMarker
    ++ encodeWithoutBOS tokenizer prompt
    ++ ctEndOfTurn
    ++ encodeWithoutBOS tokenizer "\n"

-- | Format a model turn
-- Format: <start_of_turn>model\n{prompt}<end_of_turn>\n
formatModelTurn :: Tokenizer -> ChatTemplate -> Text -> [Int]
formatModelTurn tokenizer ChatTemplate{..} prompt =
  ctStartOfTurn
    ++ ctModelMarker
    ++ encodeWithoutBOS tokenizer prompt
    ++ ctEndOfTurn
    ++ encodeWithoutBOS tokenizer "\n"

-- | Format a multi-turn conversation
-- Each turn alternates between user and model
formatConversation :: Tokenizer -> ChatTemplate -> [(Bool, Text)] -> [Int]
formatConversation tokenizer template turns =
  let encodedTurns = zipWith formatTurn [0..] turns
  in concat encodedTurns
  where
    formatTurn pos (isUser, text) =
      let wrapping = if isUser then WrapUser else WrapModel
      in wrapAndTokenize tokenizer template wrapping pos text

-- | Build a complete prompt for inference
-- Includes the conversation history and starts a model turn
buildInferencePrompt :: Tokenizer -> ChatTemplate -> [(Bool, Text)] -> [Int]
buildInferencePrompt tokenizer template history =
  let bos = bosId tokenizer
      -- Official Gemma 3 chat template uses DOUBLE BOS at the start
      doubleBOS = [bos, bos]
      conversationTokens = formatConversation tokenizer template history
      -- Add start of model turn for generation
      modelTurnStart = ctStartOfTurn template ++ ctModelMarker template
  in doubleBOS ++ conversationTokens ++ modelTurnStart
