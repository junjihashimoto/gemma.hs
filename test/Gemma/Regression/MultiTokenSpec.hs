{-# LANGUAGE OverloadedStrings #-}

module Gemma.Regression.MultiTokenSpec (spec) where

import Test.Hspec
import qualified Data.Vector.Storable as V
import Gemma.Model
import Gemma.SafeTensors (loadSafeTensors, hasTensor)

spec :: Spec
spec = describe "Multi-Token Inference" $ do

  it "FP32: Two tokens with cache matches two tokens without cache" $ do
    putStrLn "\n=== FP32 Multi-Token TDD Test ==="

    -- Detect config
    st <- loadSafeTensors "../models/gemma3-1b.safetensors"
    let hasPreFFNNorm = hasTensor st "model.layers.0.pre_feedforward_layernorm.weight"
        hasPostAttnNorm = hasTensor st "model.layers.0.post_attention_layernorm.weight"
        usePostAttnNorm = hasPreFFNNorm && hasPostAttnNorm

    let config = gemma3_1BConfig {
          gcUseFP16 = False,
          gcUseQKNorm = False,
          gcUsePostAttnNorm = usePostAttnNorm,
          gcUsePostFFNNorm = False
        }

    model <- loadGemmaModel "../models/gemma3-1b.safetensors" config

    -- Test token
    let token1 = 6974  -- "Write"

    putStrLn "Step 1: Single token inference"
    logits1 <- runGemmaInference model (V.singleton token1)
    let top1 = V.maxIndex logits1
        logit1 = logits1 V.! top1
    putStrLn $ "  Input: [" ++ show token1 ++ "]"
    putStrLn $ "  Top: token=" ++ show top1 ++ ", logit=" ++ show logit1

    putStrLn "\nStep 2: Two tokens WITHOUT cache (full forward pass)"
    logits2NoCache <- runGemmaInference model (V.fromList [token1, top1])
    let top2NoCache = V.maxIndex logits2NoCache
        logit2NoCache = logits2NoCache V.! top2NoCache
    putStrLn $ "  Input: [" ++ show token1 ++ "," ++ show top1 ++ "]"
    putStrLn $ "  Top: token=" ++ show top2NoCache ++ ", logit=" ++ show logit2NoCache

    putStrLn "\nStep 3: Two tokens WITH cache"
    (logitsA, cacheA) <- runGemmaInferenceCached model (V.singleton token1) Nothing
    let topA = V.maxIndex logitsA
        logitA = logitsA V.! topA
    putStrLn $ "  After token 1: top=" ++ show topA ++ ", logit=" ++ show logitA

    (logits2WithCache, _cacheB) <- runGemmaInferenceCached model (V.singleton top1) (Just cacheA)
    let top2WithCache = V.maxIndex logits2WithCache
        logit2WithCache = logits2WithCache V.! top2WithCache
    putStrLn $ "  After token 2 (cached): top=" ++ show top2WithCache ++ ", logit=" ++ show logit2WithCache

    putStrLn "\nComparison:"
    putStrLn $ "  First token prediction matches: " ++ show (top1 == topA)
    putStrLn $ "  No cache: token=" ++ show top2NoCache ++ ", logit=" ++ show logit2NoCache
    putStrLn $ "  With cache: token=" ++ show top2WithCache ++ ", logit=" ++ show logit2WithCache

    -- Assert they match
    let tokenMatch = top2NoCache == top2WithCache
        logitClose = abs (logit2NoCache - logit2WithCache) < 0.1

    if tokenMatch && logitClose
      then putStrLn "\n✅ PASS: Cache produces same result as no-cache!"
      else do
        putStrLn "\n❌ FAIL: Cache mismatch!"
        putStrLn $ "  Token mismatch: " ++ show top2NoCache ++ " ≠ " ++ show top2WithCache
        putStrLn $ "  Logit difference: " ++ show (abs (logit2NoCache - logit2WithCache))
        expectationFailure "Cache does not match no-cache inference"
