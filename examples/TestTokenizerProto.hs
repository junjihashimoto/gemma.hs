{-# LANGUAGE OverloadedStrings #-}
module Main where

import Gemma.Tokenizer.Proto
import qualified Data.Vector as V
import qualified Data.Text as T

main :: IO ()
main = do
  putStrLn "Loading Gemma tokenizer model..."
  model <- loadModelProto "../models/pytorch/gemma3-keras-gemma3_1b-v3/assets/tokenizer/vocabulary.spm"

  putStrLn "\n=== Model Info ==="
  putStrLn $ "Vocabulary size: " ++ show (V.length (mpPieces model))

  case mpTrainerSpec model of
    Just spec -> do
      putStrLn $ "Model type: " ++ show (tsModelType spec)
      putStrLn $ "Configured vocab size: " ++ show (tsVocabSize spec)
    Nothing -> putStrLn "No trainer spec found"

  case mpNormalizerSpec model of
    Just spec -> do
      putStrLn "\n=== Normalizer Config ==="
      putStrLn $ "Name: " ++ T.unpack (nsName spec)
      putStrLn $ "Add dummy prefix: " ++ show (nsAddDummyPrefix spec)
      putStrLn $ "Remove extra whitespaces: " ++ show (nsRemoveExtraWhitespaces spec)
      putStrLn $ "Escape whitespaces: " ++ show (nsEscapeWhitespaces spec)
    Nothing -> putStrLn "No normalizer spec found"

  putStrLn "\n=== First 20 Tokens ==="
  V.imapM_ (\i piece ->
    putStrLn $ "  " ++ show i ++ ": " ++ show (spPiece piece)
            ++ " (score=" ++ show (spScore piece)
            ++ ", type=" ++ show (spType piece) ++ ")"
    ) (V.take 20 (mpPieces model))

  -- Find special tokens
  putStrLn "\n=== Special Token IDs ==="
  let findToken name = V.findIndex (\p -> spPiece p == name) (mpPieces model)
  case findToken "<bos>" of
    Just idx -> putStrLn $ "BOS: " ++ show idx
    Nothing -> putStrLn "BOS not found"
  case findToken "<eos>" of
    Just idx -> putStrLn $ "EOS: " ++ show idx
    Nothing -> putStrLn "EOS not found"
  case findToken "<unk>" of
    Just idx -> putStrLn $ "UNK: " ++ show idx
    Nothing -> putStrLn "UNK not found"
  case findToken "<pad>" of
    Just idx -> putStrLn $ "PAD: " ++ show idx
    Nothing -> putStrLn "PAD not found"

  putStrLn "\n✅ Proto parser test complete!"
