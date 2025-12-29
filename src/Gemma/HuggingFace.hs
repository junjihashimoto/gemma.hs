{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

{-|
Module: Gemma.HuggingFace
Description: Download and load models from Hugging Face Hub

This module provides utilities to download GGUF models from Hugging Face
and load them using the GGUF loader.
-}

module Gemma.HuggingFace
  ( -- * Downloading
    downloadModel
  , downloadAndLoadGGUF
    -- * Model IDs
  , gemma3_1b_q4_0
  , gemma3_4b_q4_0
  ) where

import qualified Data.Text as T
import Data.Text (Text)
import System.Process (callProcess, readProcess)
import System.Directory (doesFileExist, findExecutable, createDirectoryIfMissing)
import System.FilePath ((</>), takeFileName)
import System.Exit (exitFailure)
import Control.Monad (unless, when)
import qualified Control.Exception
import Gemma.GGUF (GGUFFile, loadGGUF)

-- | Gemma 3 1B Q4_0 quantized model
gemma3_1b_q4_0 :: Text
gemma3_1b_q4_0 = "google/gemma-3-1b-it-qat-q4_0-gguf"

-- | Gemma 3 4B Q4_0 quantized model
gemma3_4b_q4_0 :: Text
gemma3_4b_q4_0 = "google/gemma-3-4b-it-qat-q4_0-gguf"

-- | Download a model from Hugging Face Hub
--
-- This function:
-- 1. Checks if Python 3 and huggingface_hub are installed
-- 2. Downloads the model using the download_gguf.py script
-- 3. Returns the path to the downloaded GGUF file
--
-- Example:
-- > modelPath <- downloadModel gemma3_1b_q4_0 "../models"
-- > gf <- loadGGUF modelPath
downloadModel :: Text -> FilePath -> IO FilePath
downloadModel modelId outputDir = do
  -- Check if Python 3 is available
  pythonExe <- findExecutable "python3"
  case pythonExe of
    Nothing -> do
      putStrLn "Error: python3 not found in PATH"
      putStrLn "Please install Python 3 to download models from Hugging Face"
      exitFailure
    Just _ -> pure ()

  -- Check if huggingface_hub is installed
  putStrLn "Checking for huggingface_hub..."
  result <- readProcess "python3" ["-c", "import huggingface_hub"] ""
    `catch` \(_ :: IOError) -> do
      putStrLn "Error: huggingface_hub not installed"
      putStrLn "Install with: pip install huggingface_hub"
      exitFailure

  -- Create output directory
  createDirectoryIfMissing True outputDir

  -- Get the download script path
  let scriptPath = "scripts/download_gguf.py"

  -- Check if script exists
  scriptExists <- doesFileExist scriptPath
  unless scriptExists $ do
    putStrLn $ "Error: Download script not found at " ++ scriptPath
    exitFailure

  -- Download the model
  putStrLn $ "Downloading model: " ++ T.unpack modelId
  putStrLn $ "Output directory: " ++ outputDir
  putStrLn ""

  callProcess "python3" [scriptPath, T.unpack modelId, outputDir]

  -- Find the downloaded GGUF file
  -- The script creates a symlink in the output directory
  -- List all .gguf files (including symlinks)
  output <- readProcess "find" [outputDir, "-maxdepth", "1", "-name", "*.gguf"] ""
  let ggufFiles = filter (not . null) $ lines output

  case ggufFiles of
    (path:_) -> do
      putStrLn $ "Model ready at: " ++ path
      pure path
    [] -> do
      putStrLn $ "Error: No GGUF files found in " ++ outputDir
      putStrLn $ "Expected to find .gguf files after download"
      exitFailure

  where
    filterM :: Monad m => (a -> m Bool) -> [a] -> m [a]
    filterM _ [] = pure []
    filterM p (x:xs) = do
      b <- p x
      rest <- filterM p xs
      pure $ if b then x : rest else rest

    catch :: IO a -> (IOError -> IO a) -> IO a
    catch = Control.Exception.catch

-- | Download a model from Hugging Face and load it as a GGUFFile
--
-- This is a convenience function that combines downloadModel and loadGGUF.
--
-- Example:
-- > gf <- downloadAndLoadGGUF gemma3_1b_q4_0 "../models"
downloadAndLoadGGUF :: Text -> FilePath -> IO GGUFFile
downloadAndLoadGGUF modelId outputDir = do
  modelPath <- downloadModel modelId outputDir
  loadGGUF modelPath
