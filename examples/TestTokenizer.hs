{-# LANGUAGE OverloadedStrings #-}
module Main where

import Gemma.Tokenizer
import qualified Data.Text as T
import qualified Data.Text.IO as TIO

main :: IO ()
main = do
  putStrLn "Loading Gemma tokenizer..."
  tokenizer <- loadTokenizer "../models/pytorch/gemma3-keras-gemma3_1b-v3/assets/tokenizer/vocabulary.spm"

  putStrLn "\n=== Tokenizer Info ==="
  putStrLn $ "Vocabulary size: " ++ show (vocabSize tokenizer)
  putStrLn $ "BOS ID: " ++ show (bosId tokenizer)
  putStrLn $ "EOS ID: " ++ show (eosId tokenizer)
  putStrLn $ "UNK ID: " ++ show (unkId tokenizer)
  putStrLn $ "PAD ID: " ++ show (padId tokenizer)

  putStrLn "\n=== Test Encoding ===\n"

  -- Test 1: Simple text
  let text1 = "Hello, world!"
  putStrLn $ "Input: " ++ show text1
  let ids1 = encode tokenizer text1
  putStrLn $ "Token IDs: " ++ show ids1
  let decoded1 = decode tokenizer ids1
  TIO.putStr "Decoded: "
  TIO.putStrLn decoded1
  putStrLn ""

  -- Test 2: Question
  let text2 = "What is the capital of France?"
  putStrLn $ "Input: " ++ show text2
  let ids2 = encode tokenizer text2
  putStrLn $ "Token IDs: " ++ show ids2
  putStrLn $ "Token count: " ++ show (length ids2)
  let decoded2 = decode tokenizer ids2
  TIO.putStr "Decoded: "
  TIO.putStrLn decoded2
  putStrLn ""

  -- Test 3: Round-trip test
  let text3 = "The quick brown fox jumps over the lazy dog."
  putStrLn $ "Round-trip test: " ++ show text3
  let ids3 = encode tokenizer text3
  let decoded3 = decode tokenizer ids3
  TIO.putStr "Decoded: "
  TIO.putStrLn decoded3
  if T.strip decoded3 == T.strip text3
    then putStrLn "✅ Round-trip successful!"
    else putStrLn "❌ Round-trip failed (this is expected with simple BPE)"

  putStrLn "\n✅ Tokenizer test complete!"
