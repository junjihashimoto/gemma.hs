{-# LANGUAGE OverloadedStrings #-}
module Main where

import Gemma.Tokenizer
import Gemma.ChatTemplate
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import System.Process
import System.Exit
import Data.List (intercalate)

-- Test cases to compare
testCases :: [(String, T.Text)]
testCases =
  [ ("Simple greeting", "Hello, world!")
  , ("Question", "What is the capital of France?")
  , ("Sentence", "The quick brown fox jumps over the lazy dog.")
  , ("With numbers", "There are 42 apples and 17 oranges.")
  , ("Special chars", "Hello! How are you? I'm fine, thanks.")
  , ("Multiline", "First line\nSecond line\nThird line")
  , ("Empty", "")
  , ("Single word", "Hello")
  , ("Chat marker", "<start_of_turn>")
  , ("User turn", "user\n")
  ]

-- Run Python tokenizer
runPythonTokenizer :: T.Text -> IO (Either String [Int])
runPythonTokenizer text = do
  let textStr = T.unpack text
  (exitCode, stdout, stderr) <- readProcessWithExitCode "python3"
    ["scripts/tokenize_simple.py",
     "../models/pytorch/gemma3-keras-gemma3_1b-v3/assets/tokenizer/vocabulary.spm",
     textStr] ""

  case exitCode of
    ExitSuccess -> do
      -- Parse comma-separated output
      let cleanOutput = filter (\c -> c `elem` (['0'..'9'] ++ [','])) stdout
      if null cleanOutput
        then return $ Left "Empty output from Python tokenizer"
        else do
          let tokens = map (read :: String -> Int) $ words $ map (\c -> if c == ',' then ' ' else c) cleanOutput
          return $ Right tokens
    ExitFailure _ -> return $ Left stderr

main :: IO ()
main = do
  putStrLn "=== Tokenizer Comparison Test ==="
  putStrLn "Comparing Haskell vs Python SentencePiece tokenizer"
  putStrLn ""

  -- Load Haskell tokenizer
  putStrLn "Loading Haskell tokenizer..."
  tokenizer <- loadTokenizer "../models/pytorch/gemma3-keras-gemma3_1b-v3/assets/tokenizer/vocabulary.spm"
  putStrLn ""

  -- Run comparison tests
  let totalTests = length testCases
  results <- mapM (testCase tokenizer) (zip [1..] testCases)

  let passed = length $ filter id results
      failed = totalTests - passed

  putStrLn ""
  putStrLn "=== Summary ==="
  putStrLn $ "Total tests: " ++ show totalTests
  putStrLn $ "Passed: " ++ show passed
  putStrLn $ "Failed: " ++ show failed

  if failed == 0
    then putStrLn "\n✅ All tests passed! Haskell tokenizer matches Python."
    else putStrLn $ "\n❌ " ++ show failed ++ " test(s) failed."

testCase :: Tokenizer -> (Int, (String, T.Text)) -> IO Bool
testCase tokenizer (num, (name, text)) = do
  putStrLn $ "\n[Test " ++ show num ++ "] " ++ name
  TIO.putStr "  Input: "
  TIO.putStrLn $ "\"" <> text <> "\""

  -- Haskell tokenizer
  let haskellTokens = encode tokenizer text
  putStrLn $ "  Haskell: " ++ show haskellTokens

  -- Python tokenizer
  pythonResult <- runPythonTokenizer text

  case pythonResult of
    Left err -> do
      putStrLn $ "  Python:  ERROR - " ++ err
      putStrLn "  Status:  ❌ FAILED (Python error)"
      return False

    Right pythonTokens -> do
      putStrLn $ "  Python:  " ++ show pythonTokens

      if haskellTokens == pythonTokens
        then do
          putStrLn "  Status:  ✅ PASS"
          return True
        else do
          putStrLn "  Status:  ❌ FAIL (mismatch)"
          putStrLn "  Difference:"
          putStrLn $ "    Haskell has " ++ show (length haskellTokens) ++ " tokens"
          putStrLn $ "    Python has " ++ show (length pythonTokens) ++ " tokens"
          return False
