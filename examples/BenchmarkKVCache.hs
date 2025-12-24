{-# LANGUAGE OverloadedStrings #-}

{-|
Module: BenchmarkKVCache
Description: Benchmark KV-cache performance

This benchmark measures:
1. Token generation speed with KV-cache
2. Tokens per second
3. Time per token
4. Cache efficiency

Usage:
  cabal run benchmark-kvcache -- \
    --model path/to/model.safetensors \
    --tokenizer path/to/tokenizer.spm \
    --num-tokens 50
-}

module Main where

import System.Environment (getArgs)
import System.Exit (exitFailure)
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import System.CPUTime (getCPUTime)
import Text.Printf (printf)
import Control.Monad (when)

import Gemma.Model
import Gemma.Tokenizer
import Gemma.ChatTemplate
import Foreign.C.Types

-- FFI for profiling
foreign import ccall unsafe "gpu_print_profile_stats"
  c_printProfileStats :: IO ()

foreign import ccall unsafe "gpu_reset_profile_stats"
  c_resetProfileStats :: IO ()

main :: IO ()
main = do
  args <- getArgs
  case args of
    ["--model", modelPath, "--tokenizer", tokPath, "--num-tokens", numStr] ->
      runBenchmark modelPath tokPath (read numStr)
    _ -> do
      putStrLn "Usage: benchmark-kvcache --model <path> --tokenizer <path> --num-tokens <n>"
      exitFailure

runBenchmark :: FilePath -> FilePath -> Int -> IO ()
runBenchmark modelPath tokPath numTokens = do
  putStrLn "=========================================="
  putStrLn "KV-Cache Performance Benchmark"
  putStrLn "=========================================="
  putStrLn ""

  -- Load tokenizer and model
  putStrLn "📦 Loading tokenizer..."
  tokenizer <- loadTokenizer tokPath
  putStrLn "✅ Tokenizer loaded"

  putStrLn "📦 Loading model (Gemma 3 1B)..."
  model <- loadGemmaModel modelPath gemma3_1BConfig
  putStrLn "✅ Model loaded"
  putStrLn ""

  -- Benchmark configuration
  let prompt = "Write a short story about"
      template = buildChatTemplate tokenizer
      promptTokens = buildInferencePrompt tokenizer template [(True, prompt)]
      promptVec = V.fromList promptTokens
      promptLen = length promptTokens

  putStrLn $ "Prompt: \"" ++ T.unpack prompt ++ "\""
  putStrLn $ "Prompt tokens: " ++ show promptLen
  putStrLn $ "Generate tokens: " ++ show numTokens
  putStrLn ""

  -- Warmup run (for shader compilation)
  putStrLn "🔥 Warmup run (compiling shaders)..."
  _ <- generateWithCache model tokenizer promptVec 5 False
  putStrLn "✅ Warmup complete"
  putStrLn ""

  -- Actual benchmark
  putStrLn "=========================================="
  putStrLn "Running Benchmark"
  putStrLn "=========================================="
  putStrLn ""

  startTime <- getCPUTime
  tokens <- generateWithCache model tokenizer promptVec numTokens True
  endTime <- getCPUTime

  let generatedTokens = drop promptLen tokens
      generatedCount = length generatedTokens
      totalTime = fromIntegral (endTime - startTime) / (10^12) :: Double
      tokensPerSec = fromIntegral generatedCount / totalTime
      msPerToken = (totalTime * 1000) / fromIntegral generatedCount

  -- Display results
  putStrLn ""
  putStrLn "=========================================="
  putStrLn "Benchmark Results"
  putStrLn "=========================================="
  putStrLn ""
  putStrLn $ "Total time: " ++ printf "%.2f" totalTime ++ " seconds"
  putStrLn $ "Tokens generated: " ++ show generatedCount
  putStrLn $ "Tokens per second: " ++ printf "%.2f" tokensPerSec
  putStrLn $ "Milliseconds per token: " ++ printf "%.2f" msPerToken
  putStrLn ""

  -- Performance classification
  putStrLn "Performance Rating:"
  if tokensPerSec >= 50 then
    putStrLn "  ⭐⭐⭐⭐⭐ Excellent! (50+ tok/s)"
  else if tokensPerSec >= 30 then
    putStrLn "  ⭐⭐⭐⭐ Very Good! (30-50 tok/s)"
  else if tokensPerSec >= 15 then
    putStrLn "  ⭐⭐⭐ Good! (15-30 tok/s)"
  else if tokensPerSec >= 5 then
    putStrLn "  ⭐⭐ Fair (5-15 tok/s)"
  else
    putStrLn "  ⭐ Slow (<5 tok/s)"
  putStrLn ""

  -- Expected vs actual
  putStrLn "Expected Performance (with KV-cache):"
  putStrLn "  - 30-100 tokens/second"
  putStrLn "  - 10-50× speedup over naive implementation"
  putStrLn ""

  -- Display generated text
  putStrLn "=========================================="
  putStrLn "Generated Text:"
  putStrLn "=========================================="
  putStrLn ""
  putStrLn $ "DEBUG: Generated token IDs: " ++ show generatedTokens
  let generatedText = decode tokenizer generatedTokens
  TIO.putStrLn generatedText
  putStrLn ""

  -- Memory estimate
  let cacheMemoryMB = (2048 * 1024 * 4 * 2 * 26) `div` (1024 * 1024) :: Int
  putStrLn "=========================================="
  putStrLn "Memory Usage (Estimated)"
  putStrLn "=========================================="
  putStrLn ""
  putStrLn $ "KV-Cache: ~" ++ show cacheMemoryMB ++ " MB"
  putStrLn $ "  - 26 layers"
  putStrLn $ "  - 2048 max sequence length"
  putStrLn $ "  - 1024 dimensions (4 heads × 256)"
  putStrLn $ "  - FP32 (4 bytes per value)"
  putStrLn ""

  putStrLn "=========================================="
  putStrLn "Benchmark Complete!"
  putStrLn "=========================================="
  putStrLn ""

  -- Print detailed profiling stats
  c_printProfileStats

-- | Generate tokens with KV-cache and optional printing
generateWithCache :: GemmaModel dtype -> Tokenizer -> Vector Int -> Int -> Bool -> IO [Int]
generateWithCache model tokenizer promptTokens maxTokens showOutput =
  let promptList = V.toList promptTokens
  in goPrompt promptList Nothing 0
  where
    -- Phase 1: Build cache from prompt
    goPrompt [] _cache _pos = error "Empty prompt"
    goPrompt [lastToken] cache pos = do
      when showOutput $ do
        putStr $ "Processing prompt token " ++ show (pos + 1) ++ "/" ++ show (length (V.toList promptTokens)) ++ "\r"
      (logits, finalCache) <- runGemmaInferenceCached model (V.singleton lastToken) cache

      -- Sample first generated token
      when showOutput $ do
        putStrLn $ "\nDEBUG first token logits (first 10): " ++ show (V.take 10 logits)
        putStrLn $ "DEBUG logits stats: min=" ++ show (V.minimum logits) ++ " max=" ++ show (V.maximum logits)
      let nextToken = sampleGreedy logits

      if nextToken == eosId tokenizer then
        return (V.toList promptTokens)
      else do
        when showOutput $ putStrLn ""  -- Clear progress line
        let tokens = V.toList promptTokens ++ [nextToken]
        goGenerate tokens (Just finalCache) (maxTokens - 1) 1

    goPrompt (token:rest) cache pos = do
      when showOutput $ do
        putStr $ "Processing prompt token " ++ show (pos + 1) ++ "/" ++ show (length (V.toList promptTokens)) ++ "\r"
      (_, newCache) <- runGemmaInferenceCached model (V.singleton token) cache
      goPrompt rest (Just newCache) (pos + 1)

    -- Phase 2: Generate with cache
    goGenerate tokens _cache 0 _count = return tokens
    goGenerate tokens (Just cache) remaining count = do
      when showOutput $ do
        putStr $ "Generating token " ++ show count ++ "/" ++ show maxTokens ++ "\r"

      (logits, newCache) <- runGemmaInferenceCached model (V.singleton (last tokens)) (Just cache)

      when showOutput $ do
        putStrLn $ "\nDEBUG token " ++ show count ++ " logits (first 10): " ++ show (V.take 10 logits)
        putStrLn $ "DEBUG logits stats: min=" ++ show (V.minimum logits) ++ " max=" ++ show (V.maximum logits)

      let nextToken = sampleGreedy logits

      if nextToken == eosId tokenizer then
        return tokens
      else do
        let newTokens = tokens ++ [nextToken]
        goGenerate newTokens (Just newCache) (remaining - 1) (count + 1)

    goGenerate _ Nothing _ _ = error "Cache should be initialized"

-- | Simple greedy sampling
sampleGreedy :: Vector Float -> Int
sampleGreedy logits = V.maxIndex logits
