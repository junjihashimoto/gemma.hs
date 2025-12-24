{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RankNTypes #-}

{-|
Module: Main
Description: Interactive CLI tool for Gemma inference with chat support

Command-line interface for running Gemma inference with full tokenization,
autoregressive generation, and interactive chat mode.

Usage:
  gemma-cli --model <path> --tokenizer <path> --prompt "Hello!"
  gemma-cli --model <path> --tokenizer <path> --chat
-}

module Main where

import System.Environment (getArgs)
import System.Exit (exitFailure)
import System.IO (hFlush, stdout)
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import Control.Monad (when, unless, forM_)
import Data.List (sortBy)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO

import Gemma.Model
import Gemma.Tokenizer
import Gemma.ChatTemplate
import Gemma.SafeTensors (loadSafeTensors, hasTensor)

main :: IO ()
main = do
  args <- getArgs
  case args of
    ["--help"] -> printHelp
    ["--test", modelPath] -> runTest modelPath
    ["--model", modelPath, "--tokenizer", tokPath, "--prompt", prompt] ->
      runSinglePrompt modelPath tokPath prompt
    ["--model", modelPath, "--tokenizer", tokPath, "--chat"] ->
      runInteractiveChat modelPath tokPath
    _ -> do
      putStrLn "❌ Invalid arguments. Use --help for usage information."
      exitFailure

printHelp :: IO ()
printHelp = do
  putStrLn "Gemma Inference CLI"
  putStrLn ""
  putStrLn "Usage:"
  putStrLn "  gemma-cli --model <path> --tokenizer <path> --prompt <text>"
  putStrLn "      Run single inference"
  putStrLn ""
  putStrLn "  gemma-cli --model <path> --tokenizer <path> --chat"
  putStrLn "      Interactive chat mode"
  putStrLn ""
  putStrLn "  gemma-cli --test <path>"
  putStrLn "      Run simple test (no tokenizer needed)"
  putStrLn ""
  putStrLn "  gemma-cli --help"
  putStrLn "      Show this help"
  putStrLn ""
  putStrLn "Examples:"
  putStrLn "  gemma-cli --model models/gemma-2-2b/model.safetensors \\"
  putStrLn "            --tokenizer models/gemma3-keras-gemma3_1b-v3/assets/tokenizer/vocabulary.spm \\"
  putStrLn "            --prompt \"What is the capital of France?\""
  putStrLn ""
  putStrLn "  gemma-cli --model models/gemma-2-2b/model.safetensors \\"
  putStrLn "            --tokenizer models/gemma3-keras-gemma3_1b-v3/assets/tokenizer/vocabulary.spm \\"
  putStrLn "            --chat"

-- | Test mode - just load model and run single token inference
runTest :: FilePath -> IO ()
runTest modelPath = do
  putStrLn "🔧 Testing Gemma Model Loading"
  putStrLn $ "📁 Model path: " ++ modelPath
  putStrLn ""

  -- Determine config based on path
  let (config, configName) =
        if any (\part -> part == "tiny" || part == "tiny-gemma") (words (map (\c -> if c == '/' || c == '-' then ' ' else c) modelPath))
        then (tinyGemmaConfig, "Tiny Gemma")
        else (gemma2_2BConfig, "Gemma 2 2B")

  -- Try to load the model
  putStrLn $ "📦 Loading model with " ++ configName ++ " config..."
  model <- loadGemmaModel modelPath config

  putStrLn "✅ Model loaded successfully!"
  putStrLn ""
  putStrLn "📊 Model info:"
  let modelConfig = gmConfig model
  putStrLn $ "  Vocab size:    " ++ show (gcVocabSize modelConfig)
  putStrLn $ "  Hidden dim:    " ++ show (gcHiddenDim modelConfig)
  putStrLn $ "  Num layers:    " ++ show (gcNumLayers modelConfig)
  putStrLn $ "  Num heads:     " ++ show (gcNumHeads modelConfig)
  putStrLn $ "  Head dim:      " ++ show (gcHeadDim modelConfig)
  putStrLn $ "  FFN dim:       " ++ show (gcFFNDim modelConfig)
  putStrLn ""

  -- Test with a simple token
  putStrLn "🚀 Running inference with token ID 1 (BOS token)..."
  let tokenIds = V.fromList [1]  -- BOS token

  logits <- runGemmaInference model tokenIds

  putStrLn $ "✅ Inference complete! Got " ++ show (V.length logits) ++ " logits"

  -- Find max logit
  let maxIdx = V.maxIndex logits
      maxVal = logits V.! maxIdx

  putStrLn ""
  putStrLn "🎯 Next token prediction:"
  putStrLn $ "  Token ID: " ++ show maxIdx
  putStrLn $ "  Logit:    " ++ show maxVal

-- | Auto-detect model configuration by inspecting which weights exist
autoDetectConfig :: FilePath -> Int -> IO GemmaConfig
autoDetectConfig modelPath vSize = do
  st <- loadSafeTensors modelPath

  -- Base config based on vocab size
  let baseConfig = if vSize > 260000
                   then gemma3_1BConfig  -- Gemma 3: 262144 vocab
                   else gemma2_2BConfig  -- Gemma 2: 256000 vocab

  -- Check which norm weights exist (layer 0 as sample)
  let hasPreFFNNorm = hasTensor st "model.layers.0.pre_feedforward_layernorm.weight"
      hasPostAttnNorm = hasTensor st "model.layers.0.post_attention_layernorm.weight" ||
                        hasTensor st "model.layers.0.post_attention_norm.weight"
      hasPostFFNNorm = hasTensor st "model.layers.0.post_feedforward_layernorm.weight"

  putStrLn $ "  Pre-FFN norm: " ++ if hasPreFFNNorm then "✅ Found" else "❌ Not found"
  putStrLn $ "  Post-attention norm: " ++ if hasPostAttnNorm then "✅ Found" else "❌ Not found"
  putStrLn $ "  Post-FFN norm: " ++ if hasPostFFNNorm then "✅ Found" else "❌ Not found"

  -- Gemma 3 architecture detection:
  -- The Keras config (config.json) specifies which features should be enabled.
  -- For Gemma 3 1B, the official config has:
  --   use_post_attention_norm: true
  --   use_post_ffw_norm: true
  --   use_query_key_norm: true (but no weights in safetensors - feature not implemented)
  --   use_sliding_window_attention: true
  --
  -- CORRECTED: We were incorrectly disabling these features. The baseConfig (gemma3_1BConfig)
  -- already has the correct settings. We should NOT override them unless weights are missing.

  -- Only disable features if the required weights are actually missing
  let usePostAttnNorm = hasPostAttnNorm  -- Enable if weights exist
      usePostFFNNorm = hasPostFFNNorm    -- Enable if weights exist

  putStrLn $ "  Config: gcUsePostAttnNorm=" ++ show usePostAttnNorm ++ ", gcUsePostFFNNorm=" ++ show usePostFFNNorm
  putStrLn $ "  Sliding window: " ++ if gcUseSlidingWindow baseConfig
                                     then "✅ Enabled (size=" ++ show (gcSlidingWindowSize baseConfig) ++ ")"
                                     else "❌ Disabled"
  putStrLn $ "  Zero-centered RMSNorm: " ++ if gcUseZeroCenteredRMSNorm baseConfig then "✅ Enabled" else "❌ Disabled"

  -- Adjust config based on detected weights (only override if weights missing)
  return baseConfig
    { gcUsePostAttnNorm = usePostAttnNorm
    , gcUsePostFFNNorm = usePostFFNNorm
    }

-- | Run inference on a single prompt
runSinglePrompt :: FilePath -> FilePath -> String -> IO ()
runSinglePrompt modelPath tokPath prompt = do
  putStrLn "🤖 Gemma Inference"
  putStrLn $ "📁 Model: " ++ modelPath
  putStrLn $ "📝 Tokenizer: " ++ tokPath
  putStrLn $ "💬 Prompt: " ++ prompt
  putStrLn ""

  -- Load tokenizer
  putStrLn "📦 Loading tokenizer..."
  tokenizer <- loadTokenizer tokPath
  let template = buildChatTemplate tokenizer
  putStrLn "✅ Tokenizer loaded"

  -- Load model with auto-detected config
  putStrLn "📦 Detecting model configuration..."
  let vSize = vocabSize tokenizer
  putStrLn $ "  Vocab size: " ++ show vSize
  putStrLn $ "  Base config: " ++ (if vSize > 260000 then "Gemma 3 1B" else "Gemma 2 2B")

  config <- autoDetectConfig modelPath vSize

  putStrLn "📦 Loading model..."
  model <- loadGemmaModel modelPath config
  putStrLn "✅ Model loaded"
  putStrLn ""

  -- Generate response (single user turn)
  response <- generateResponse model tokenizer template [(True, T.pack prompt)] 100

  putStrLn "🎯 Response:"
  TIO.putStrLn response

-- | Run interactive chat mode
runInteractiveChat :: FilePath -> FilePath -> IO ()
runInteractiveChat modelPath tokPath = do
  putStrLn "🤖 Gemma Interactive Chat"
  putStrLn "================================"
  putStrLn ""
  putStrLn $ "📁 Model: " ++ modelPath
  putStrLn $ "📝 Tokenizer: " ++ tokPath
  putStrLn ""

  -- Load tokenizer
  putStrLn "📦 Loading tokenizer..."
  tokenizer <- loadTokenizer tokPath
  let template = buildChatTemplate tokenizer
  putStrLn "✅ Tokenizer loaded"

  -- Load model with auto-detected config
  putStrLn "📦 Detecting model configuration..."
  let vSize = vocabSize tokenizer
  putStrLn $ "  Vocab size: " ++ show vSize
  putStrLn $ "  Base config: " ++ (if vSize > 260000 then "Gemma 3 1B" else "Gemma 2 2B")

  config <- autoDetectConfig modelPath vSize

  putStrLn "📦 Loading model..."
  model <- loadGemmaModel modelPath config
  putStrLn "✅ Model loaded"
  putStrLn ""

  putStrLn "Type 'exit' or 'quit' to end the conversation"
  putStrLn "================================"
  putStrLn ""

  -- Start chat loop
  chatLoop model tokenizer template []

-- | Interactive chat loop
chatLoop :: forall dtype. GemmaModel dtype -> Tokenizer -> ChatTemplate -> [(Bool, T.Text)] -> IO ()
chatLoop model tokenizer template history = do
  -- Get user input
  putStr "User: "
  hFlush stdout
  userInput <- TIO.getLine

  -- Check for exit
  when (T.toLower userInput `elem` ["exit", "quit"]) $ do
    putStrLn ""
    putStrLn "👋 Goodbye!"
    return ()

  unless (T.toLower userInput `elem` ["exit", "quit"]) $ do
    -- Add user message to history
    let newHistory = history ++ [(True, userInput)]

    -- Generate response using full conversation history
    putStr "Model: "
    hFlush stdout
    response <- generateResponse model tokenizer template newHistory 200

    TIO.putStrLn response
    putStrLn ""

    -- Add model response to history and continue
    let finalHistory = newHistory ++ [(False, response)]
    chatLoop model tokenizer template finalHistory

-- | Generate a response using autoregressive generation with streaming
generateResponse :: forall dtype. GemmaModel dtype -> Tokenizer -> ChatTemplate -> [(Bool, T.Text)] -> Int -> IO T.Text
generateResponse model tokenizer template history maxTokens = do
  -- Encode full conversation history with chat template
  let promptTokens = buildInferencePrompt tokenizer template history
      promptVec = V.fromList promptTokens

  -- DEBUG: Show what we're feeding to the model
  putStrLn $ "\nDEBUG: Prompt tokens (" ++ show (length promptTokens) ++ "): " ++ show (take 20 promptTokens)
  putStrLn $ "DEBUG: Decoded prompt: " ++ show (decode tokenizer promptTokens)

  -- Run autoregressive generation with streaming
  outputTokens <- autoregressiveGenerateStreaming model tokenizer promptVec maxTokens (length promptTokens)

  -- Decode output (skip the prompt tokens)
  let responseTokens = drop (length promptTokens) outputTokens
  putStrLn $ "\nDEBUG: Response tokens (" ++ show (length responseTokens) ++ "): " ++ show (take 20 responseTokens)
  return $ decode tokenizer responseTokens

-- | Autoregressive generation loop with streaming output and KV-cache
autoregressiveGenerateStreaming :: forall dtype. GemmaModel dtype -> Tokenizer -> Vector Int -> Int -> Int -> IO [Int]
autoregressiveGenerateStreaming model tokenizer promptTokens maxTokens _promptLen =
  -- Process prompt tokens one by one to build initial cache
  let promptList = V.toList promptTokens
  in goPrompt promptList Nothing 0
  where
    totalPromptLen = V.length promptTokens
    -- First phase: Process each prompt token to build cache
    goPrompt [] _cache _idx = do
      -- Should never happen if promptList is not empty
      error "Empty prompt"
    goPrompt [lastPromptToken] cache idx = do
      -- Last prompt token - after this, start generating
      putStrLn $ "\n=== Processing last prompt token " ++ show (idx + 1) ++ "/" ++ show totalPromptLen ++ ": " ++ show lastPromptToken ++ " ==="
      (logits, finalCache) <- runGemmaInferenceCached model (V.singleton lastPromptToken) cache

      -- DEBUG: Log logits statistics
      let maxLogit = V.maximum logits
          minLogit = V.minimum logits
          maxIndex = V.maxIndex logits
          topTokenLogit = logits V.! maxIndex
      putStrLn $ "Logits: max=" ++ show maxLogit ++ " min=" ++ show minLogit ++ " argmax=" ++ show maxIndex ++ " (logit=" ++ show topTokenLogit ++ ")"

      -- Show top 5 predictions
      let logitsList = V.toList logits
          indexedLogits = zip [0..] logitsList
          top5 = take 5 $ sortBy (\(_,a) (_,b) -> compare b a) indexedLogits
      putStrLn "Top 5 predictions:"
      forM_ top5 $ \(idx, logit) -> do
        let tokenText = decode tokenizer [idx]
        putStrLn $ "  token=" ++ show idx ++ " logit=" ++ show logit ++ " text=" ++ show tokenText

      -- Sample first generated token
      let temperature = 0.7
      nextToken <- sampleWithTemperature temperature logits

      -- DEBUG: Show what token was generated
      putStrLn $ "SAMPLED: token=" ++ show nextToken ++ " logit=" ++ show (logits V.! nextToken)

      if nextToken == eosId tokenizer
        then do
          putStrLn ""
          return (V.toList promptTokens)
        else do
          -- Stream the first generated token
          let tokenText = decode tokenizer [nextToken]
          TIO.putStr tokenText
          hFlush stdout

          -- Continue generating
          let tokens = V.toList promptTokens ++ [nextToken]
          goGenerate tokens (Just finalCache) (maxTokens - 1)

    goPrompt (token:rest) cache idx = do
      -- Process this prompt token and continue with rest
      putStrLn $ "Processing prompt token " ++ show (idx + 1) ++ "/" ++ show totalPromptLen ++ ": " ++ show token
      (logits, newCache) <- runGemmaInferenceCached model (V.singleton token) cache

      -- Log top prediction for this intermediate token
      let topIdx = V.maxIndex logits
          topLogit = logits V.! topIdx
      putStrLn $ "  Top prediction: token=" ++ show topIdx ++ " logit=" ++ show topLogit

      goPrompt rest (Just newCache) (idx + 1)

    -- Second phase: Generate new tokens with cache
    goGenerate tokens _cache 0 = do
      putStrLn ""
      return tokens
    goGenerate tokens (Just cache) remaining = do
      -- Process only the last token
      (logits, newCache) <- runGemmaInferenceCached model (V.singleton (last tokens)) (Just cache)

      -- Sample next token
      let temperature = 0.7
      nextToken <- sampleWithTemperature temperature logits

      if nextToken == eosId tokenizer
        then do
          putStrLn ""
          return tokens
        else do
          -- Stream the token
          let tokenText = decode tokenizer [nextToken]
          TIO.putStr tokenText
          hFlush stdout

          let newTokens = tokens ++ [nextToken]
          goGenerate newTokens (Just newCache) (remaining - 1)
    goGenerate _ Nothing _ = error "Cache should be initialized"

-- | Autoregressive generation loop (non-streaming, for backwards compatibility)
autoregressiveGenerate :: forall dtype. GemmaModel dtype -> Tokenizer -> Vector Int -> Int -> IO [Int]
autoregressiveGenerate model tokenizer promptTokens maxTokens = go (V.toList promptTokens) maxTokens
  where
    go tokens 0 = return tokens
    go tokens remaining = do
      -- Run inference on current sequence
      let inputVec = V.fromList tokens
      logits <- runGemmaInference model inputVec

      -- Sample next token with temperature
      let temperature = 0.7
      nextToken <- sampleWithTemperature temperature logits

      -- Check for EOS token
      if nextToken == eosId tokenizer
        then return tokens
        else do
          -- Add token and continue
          let newTokens = tokens ++ [nextToken]
          go newTokens (remaining - 1)

-- | Sample next token from logits (greedy sampling)
sampleToken :: Vector Float -> Int
sampleToken logits = V.maxIndex logits

-- | Sample with temperature (more diverse outputs)
sampleWithTemperature :: Float -> Vector Float -> IO Int
sampleWithTemperature temperature logits
  | temperature <= 0 = return $ sampleToken logits  -- Greedy if temp=0
  | otherwise = do
      -- Apply temperature scaling
      let scaledLogits = V.map (/ temperature) logits

      -- Convert to probabilities with softmax
      let maxLogit = V.maximum scaledLogits
          expLogits = V.map (\x -> exp (x - maxLogit)) scaledLogits
          sumExp = V.sum expLogits
          probs = V.map (/ sumExp) expLogits

      -- For now, still use greedy (argmax) - proper sampling needs random number generation
      -- TODO: Implement multinomial sampling with System.Random
      return $ V.maxIndex probs
