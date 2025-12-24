{-# LANGUAGE OverloadedStrings #-}
module Main where

import Gemma.Tokenizer
import Gemma.ChatTemplate
import qualified Data.Text.IO as TIO

main :: IO ()
main = do
  putStrLn "Loading Gemma tokenizer..."
  tokenizer <- loadTokenizer "../models/pytorch/gemma3-keras-gemma3_1b-v3/assets/tokenizer/vocabulary.spm"

  putStrLn "Building chat template..."
  let template = buildChatTemplate tokenizer

  putStrLn "\n=== Chat Template Info ==="
  putStrLn $ "Start of turn tokens: " ++ show (ctStartOfTurn template)
  putStrLn $ "End of turn tokens: " ++ show (ctEndOfTurn template)
  putStrLn $ "User marker tokens: " ++ show (ctUserMarker template)
  putStrLn $ "Model marker tokens: " ++ show (ctModelMarker template)

  putStrLn "\n=== Test 1: User Turn ===\n"
  let prompt1 = "Hello, how are you?"
  putStrLn $ "Prompt: " ++ show prompt1
  let tokens1 = wrapAndTokenize tokenizer template WrapUser 0 prompt1
  putStrLn $ "Token IDs: " ++ show tokens1
  putStrLn $ "Token count: " ++ show (length tokens1)
  let decoded1 = decode tokenizer tokens1
  TIO.putStr "Decoded: "
  TIO.putStrLn decoded1
  putStrLn ""

  putStrLn "\n=== Test 2: Multi-turn Conversation ===\n"
  let conversation =
        [ (True, "What is the capital of France?")  -- User
        , (False, "The capital of France is Paris.")  -- Model
        , (True, "What about Germany?")  -- User
        ]
  putStrLn "Conversation:"
  mapM_ (\(isUser, text) ->
    putStrLn $ "  " ++ (if isUser then "User: " else "Model: ") ++ show text
    ) conversation

  let conversationTokens = formatConversation tokenizer template conversation
  putStrLn $ "\nTotal tokens: " ++ show (length conversationTokens)
  putStrLn $ "Token IDs: " ++ show (take 50 conversationTokens) ++ "..."

  putStrLn "\n=== Test 3: Inference Prompt ===\n"
  let history =
        [ (True, "Write a greeting to the world.")
        ]
  let inferenceTokens = buildInferencePrompt tokenizer template history
  putStrLn $ "Inference prompt tokens: " ++ show (length inferenceTokens)
  putStrLn $ "Token IDs: " ++ show inferenceTokens

  putStrLn "\n✅ Chat template test complete!"
