{-# LANGUAGE OverloadedStrings #-}

module Main where

import Gemma.HuggingFace
import Gemma.GGUF
import qualified Data.Text as T
import Control.Monad (forM_)

main :: IO ()
main = do
  putStrLn "=== Downloading Gemma 3 1B GGUF Model from Hugging Face ==="
  putStrLn ""

  -- Download and load the model
  putStrLn $ "Model ID: " ++ T.unpack gemma3_1b_q4_0
  putStrLn "This will download the model to ../models/"
  putStrLn ""

  gf <- downloadAndLoadGGUF gemma3_1b_q4_0 "../models"

  putStrLn ""
  putStrLn "=== Model Loaded Successfully! ==="
  putStrLn ""

  -- Display header info
  let header = gfHeader gf
  putStrLn "Header Information:"
  putStrLn $ "  Version: " ++ show (ghVersion header)
  putStrLn $ "  Tensor Count: " ++ show (ghTensorCount header)
  putStrLn $ "  Metadata KV Count: " ++ show (ghMetadataKVCount header)
  putStrLn $ "  Alignment: " ++ show (gfAlignment gf)
  putStrLn ""

  -- Display metadata
  putStrLn "Model Metadata:"
  let metadataKeys =
        [ ("Architecture", "general.architecture")
        , ("Name", "general.name")
        , ("Context Length", "llama.context_length")
        , ("Embedding Length", "llama.embedding_length")
        , ("Block Count", "llama.block_count")
        , ("Attention Heads", "llama.attention.head_count")
        , ("KV Heads", "llama.attention.head_count_kv")
        , ("RoPE Dimension", "llama.rope.dimension_count")
        , ("RoPE Freq Base", "llama.rope.freq_base")
        ]

  forM_ metadataKeys $ \(label, key) -> do
    case getMetadata gf key of
      Just val -> putStrLn $ "  " ++ label ++ ": " ++ show val
      Nothing -> pure ()

  putStrLn ""

  -- Display tensor summary
  let tensorNames = listTensors gf
  putStrLn $ "Total Tensors: " ++ show (length tensorNames)
  putStrLn ""

  putStrLn "Sample Tensors (first 10):"
  forM_ (take 10 tensorNames) $ \name -> do
    let shape = getTensorShape gf name
        tensorType = getTensorType gf name
        numElements = product shape
    putStrLn $ "  " ++ T.unpack name
    putStrLn $ "    Shape: " ++ show shape
    putStrLn $ "    Type: " ++ show tensorType
    putStrLn $ "    Elements: " ++ show numElements

  putStrLn ""
  putStrLn "✅ Model is ready to use!"
  putStrLn ""
  putStrLn "You can now use this model for inference by loading it with:"
  putStrLn "  gf <- loadGGUF \"../models/<model-file>.gguf\""
