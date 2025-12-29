{-# LANGUAGE OverloadedStrings #-}

module Main where

import Gemma.GGUF
import qualified Data.Text as T
import System.Environment (getArgs)
import System.Exit (exitFailure)
import Control.Monad (forM_)
import Data.Word
import Data.Bits

main :: IO ()
main = do
  args <- getArgs
  case args of
    [path] -> do
      putStrLn $ "Loading GGUF file: " ++ path
      putStrLn ""

      -- Load GGUF file
      gf <- loadGGUF path

      -- Display header
      let header = gfHeader gf
      putStrLn "=== GGUF Header ==="
      putStrLn $ "  Magic: 0x" ++ showHex (ghMagic header)
      putStrLn $ "  Version: " ++ show (ghVersion header)
      putStrLn $ "  Tensor Count: " ++ show (ghTensorCount header)
      putStrLn $ "  Metadata KV Count: " ++ show (ghMetadataKVCount header)
      putStrLn $ "  Alignment: " ++ show (gfAlignment gf)
      putStrLn ""

      -- Display metadata
      putStrLn "=== Metadata ==="
      let commonKeys =
            [ "general.architecture"
            , "general.name"
            , "general.file_type"
            , "llama.context_length"
            , "llama.embedding_length"
            , "llama.block_count"
            , "llama.attention.head_count"
            , "llama.attention.head_count_kv"
            , "llama.rope.dimension_count"
            , "llama.rope.freq_base"
            ]

      forM_ commonKeys $ \key -> do
        case getMetadata gf key of
          Just val -> putStrLn $ "  " ++ T.unpack key ++ ": " ++ show val
          Nothing -> pure ()
      putStrLn ""

      -- Display first 10 tensors
      let tensorNames = take 10 (listTensors gf)
      putStrLn "=== First 10 Tensors ==="
      forM_ tensorNames $ \name -> do
        let shape = getTensorShape gf name
            tensorType = getTensorType gf name
        putStrLn $ "  " ++ T.unpack name
        putStrLn $ "    Shape: " ++ show shape
        putStrLn $ "    Type: " ++ show tensorType
        putStrLn $ "    Elements: " ++ show (product shape)

      putStrLn ""
      putStrLn $ "Total tensors: " ++ show (length (listTensors gf))

    _ -> do
      putStrLn "Usage: test-gguf <path-to-gguf-file>"
      exitFailure

-- Helper to show hex
showHex :: Word32 -> String
showHex w = go w []
  where
    go 0 [] = "0"
    go 0 acc = acc
    go n acc =
      let digit = n .&. 0xF
          c = if digit < 10 then toEnum (fromEnum '0' + fromIntegral digit)
                            else toEnum (fromEnum 'A' + fromIntegral (digit - 10))
      in go (n `shiftR` 4) (c : acc)
