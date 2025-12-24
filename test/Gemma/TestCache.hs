{-# LANGUAGE OverloadedStrings #-}
module Gemma.TestCache
  ( loadOrGenerateCache
  ) where

import System.Process (callProcess)
import System.Directory (doesFileExist, getModificationTime)
import qualified Data.ByteString.Lazy as BL
import qualified Data.Aeson as JSON

-- | Load cached JSON or regenerate if Python script is newer
--
-- This function checks timestamps:
-- - If JSON doesn't exist, run Python to generate it
-- - If Python script is newer than JSON, regenerate
-- - Otherwise, use cached JSON
--
-- Usage:
--   reference <- loadOrGenerateCache
--     "test/Gemma/Regression/Q4InferenceSpec_layer0.py"
--     "test/Gemma/Regression/Q4InferenceSpec_layer0.json"
loadOrGenerateCache :: JSON.FromJSON a => FilePath -> FilePath -> IO a
loadOrGenerateCache pythonScript jsonCache = do
  needsRegeneration <- shouldRegenerate pythonScript jsonCache

  if needsRegeneration
    then do
      putStrLn $ "Cache outdated, regenerating: " ++ jsonCache
      callProcess "python3" [pythonScript]
      loadJSON jsonCache
    else do
      putStrLn $ "Using cached: " ++ jsonCache
      loadJSON jsonCache

-- | Check if cache needs regeneration
shouldRegenerate :: FilePath -> FilePath -> IO Bool
shouldRegenerate pythonScript jsonCache = do
  jsonExists <- doesFileExist jsonCache

  if not jsonExists
    then return True  -- No cache, need to generate
    else do
      pythonTime <- getModificationTime pythonScript
      jsonTime <- getModificationTime jsonCache
      return (pythonTime > jsonTime)  -- Regenerate if Python is newer

-- | Load and parse JSON file
loadJSON :: JSON.FromJSON a => FilePath -> IO a
loadJSON path = do
  content <- BL.readFile path
  case JSON.eitherDecode content of
    Left err -> error $ "Failed to parse " ++ path ++ ": " ++ err
    Right val -> return val
