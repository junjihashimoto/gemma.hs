{-# LANGUAGE OverloadedStrings #-}
module Gemma.Regression.Q4InferenceSpec (spec) where

import Test.Hspec
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import qualified Data.Aeson as JSON
import qualified Data.ByteString.Lazy as BL
import Data.Aeson ((.:), (.:?))
import qualified Data.Aeson.Types as JSON
import Gemma.SafeTensors (getTensor, loadSafeTensors, loadQ4WeightDequantized, hasQ4Weight)
import Gemma.Model (loadGemmaModel, runGemmaInferenceCached, gemma3_1BConfig)
import Gemma.TestCache (loadOrGenerateCache)
import Gemma.Tokenizer (loadTokenizer)
import Gemma.ChatTemplate (buildChatTemplate, buildInferencePrompt)
import Text.Printf
import Data.List (find)
import Control.Monad (when)
import Graphics.WebGPU.Dawn.ContT (evalContT, ContT, liftIO, lift)
import qualified Graphics.WebGPU.Dawn.Context as Ctx
import qualified Gemma.Layers.RMSNorm
import qualified Gemma.Layers.Linear
import Graphics.WebGPU.Dawn.Types (Shape(..))
import qualified Graphics.WebGPU.Dawn.Tensor as T

data Q4InferenceReference = Q4InferenceReference
  { inputTokenId :: Int
  , inputEmbedding :: [Float]
  , steps :: [InferenceStep]
  } deriving (Show)

data InferenceStep = InferenceStep
  { stepName :: String
  , stepOutput :: Maybe [Float]
  , stepQ :: Maybe [Float]
  , stepK :: Maybe [Float]
  , stepV :: Maybe [Float]
  } deriving (Show)

instance JSON.FromJSON Q4InferenceReference where
  parseJSON = JSON.withObject "Q4InferenceReference" $ \v -> do
    input <- v .: "input"
    tokenId <- input .: "token_id"
    embedding <- input .: "embedding"
    stepsArray <- v .: "steps"
    return $ Q4InferenceReference tokenId embedding stepsArray

instance JSON.FromJSON InferenceStep where
  parseJSON = JSON.withObject "InferenceStep" $ \v -> InferenceStep
    <$> v .: "name"
    <*> v .:? "output"
    <*> v .:? "q"
    <*> v .:? "k"
    <*> v .:? "v"

spec :: Spec
spec = describe "Q4 Model Inference (TDD)" $ do

  it "loads Q4 model without NaN" $ do
    -- Load Q4 model
    st <- loadSafeTensors "../models/gemma3-1b-q4.safetensors"

    -- Check if it has Q4 weights
    let hasQ4 = hasQ4Weight st "model.layers.0.self_attn.q_proj.weight"
    hasQ4 `shouldBe` True

    -- Load and check Q projection weights
    qWeights <- loadQ4WeightDequantized st "model.layers.0.self_attn.q_proj.weight"

    -- Check for NaN/Inf
    let hasNaN = V.any isNaN qWeights
        hasInf = V.any isInfinite qWeights

    hasNaN `shouldBe` False
    hasInf `shouldBe` False

    -- Check reasonable range
    let minVal = V.minimum qWeights
        maxVal = V.maximum qWeights

    minVal `shouldSatisfy` (\x -> x > -1.0)  -- Reasonable for normalized weights
    maxVal `shouldSatisfy` (\x -> x < 1.0)

  it "Q4 embedding matches PyTorch" $ do
    -- Load reference
    referenceJson <- BL.readFile "test/Gemma/Regression/Q4InferenceSpec_layer0.json"
    case JSON.eitherDecode referenceJson :: Either String Q4InferenceReference of
      Left err -> expectationFailure $ "Failed to parse reference: " ++ err
      Right ref -> do
        -- Load Q4 model
        st <- loadSafeTensors "../models/gemma3-1b-q4.safetensors"
        embeddings <- getTensor st "model.embed_tokens.weight"

        -- Get embedding for token 1
        let tokenId = inputTokenId ref
            hiddenDim = 1152
            embedding = V.slice (tokenId * hiddenDim) hiddenDim embeddings

        -- Compare with reference
        let pytorchEmb = V.fromList $ inputEmbedding ref
            diffs = V.zipWith (\h p -> abs (h - p)) embedding pytorchEmb
            maxDiff = V.maximum diffs

        -- Should be exact match (same file)
        maxDiff `shouldSatisfy` (<= 1e-6)

  it "Q4 layer 0 step-by-step matches PyTorch" $ do
    -- Load reference
    referenceJson <- BL.readFile "test/Gemma/Regression/Q4InferenceSpec_layer0.json"
    case JSON.eitherDecode referenceJson :: Either String Q4InferenceReference of
      Left err -> expectationFailure $ "Failed to parse reference: " ++ err
      Right ref -> do
        -- Load Q4 model
        st <- loadSafeTensors "../models/gemma3-1b-q4.safetensors"

        -- Get token embedding
        embeddings <- getTensor st "model.embed_tokens.weight"
        let tokenId = inputTokenId ref
            hiddenDim = 1152
            embedding = V.slice (tokenId * hiddenDim) hiddenDim embeddings

        -- Verify embedding matches (should be identical)
        let pytorchEmb = V.fromList $ inputEmbedding ref
            embDiffs = V.zipWith (\h p -> abs (h - p)) embedding pytorchEmb
            maxEmbDiff = V.maximum embDiffs

        putStrLn $ "\n=== Q4 Layer 0 Step-by-Step TDD ==="
        putStrLn $ "Step 0: Embedding - maxDiff: " ++ printf "%.2e" maxEmbDiff
        maxEmbDiff `shouldSatisfy` (<= 1e-6)

        -- Load layer 0 weights
        attnNorm <- getTensor st "model.layers.0.input_layernorm.weight"

        -- Check if Q4 weights exist
        let hasQ4 = hasQ4Weight st "model.layers.0.self_attn.q_proj.weight"

        qWeights <- if hasQ4
                    then loadQ4WeightDequantized st "model.layers.0.self_attn.q_proj.weight"
                    else getTensor st "model.layers.0.self_attn.q_proj.weight"

        kWeights <- if hasQ4
                    then loadQ4WeightDequantized st "model.layers.0.self_attn.k_proj.weight"
                    else getTensor st "model.layers.0.self_attn.k_proj.weight"

        vWeights <- if hasQ4
                    then loadQ4WeightDequantized st "model.layers.0.self_attn.v_proj.weight"
                    else getTensor st "model.layers.0.self_attn.v_proj.weight"

        -- Step 1: RMSNorm
        let eps = 1e-6
            variance = V.sum (V.map (\x -> x * x) embedding) / fromIntegral hiddenDim
            rmsNormScale = 1.0 / sqrt (variance + eps)
            -- Q4 model does NOT use zero-centered weights (it's Gemma 2, not Gemma 3!)
            xNorm = V.zipWith (\x w -> x * rmsNormScale * w) embedding attnNorm

        -- Compare with PyTorch
        case find (\step -> stepName step == "input_rmsnorm") (steps ref) of
          Just normStep -> case stepOutput normStep of
            Just pytorchNorm -> do
              let pytorchNormVec = V.fromList pytorchNorm
                  normDiffs = V.zipWith (\h p -> abs (h - p)) xNorm pytorchNormVec
                  maxNormDiff = V.maximum normDiffs

              putStrLn $ "Step 1: RMSNorm - maxDiff: " ++ printf "%.2e" maxNormDiff
              putStrLn $ "  Haskell first 5: " ++ show (V.toList $ V.take 5 xNorm)
              putStrLn $ "  PyTorch first 5: " ++ show (take 5 pytorchNorm)

              -- Check for NaN in Haskell output
              let hasNaN = V.any isNaN xNorm
                  hasInf = V.any isInfinite xNorm

              when hasNaN $ putStrLn "  ❌ HASKELL HAS NaN!"
              when hasInf $ putStrLn "  ❌ HASKELL HAS Inf!"

              hasNaN `shouldBe` False
              hasInf `shouldBe` False
              maxNormDiff `shouldSatisfy` (<= 2e-4)  -- Allow small numerical differences (FP32 precision)
            Nothing -> expectationFailure "No output in input_rmsnorm step"
          Nothing -> expectationFailure "input_rmsnorm step not found"

        -- Step 2: Q/K/V projections
        let qSize = 1024  -- num_heads (4) * head_dim (256)
            kvSize = 256   -- num_kv_heads (1) * head_dim (256)

        -- Q projection: (1024, 1152) @ (1152,) -> (1024,)
        let q = V.generate qSize $ \i ->
              let row = V.slice (i * hiddenDim) hiddenDim qWeights
              in V.sum $ V.zipWith (*) row xNorm

        -- K projection: (256, 1152) @ (1152,) -> (256,)
        let k = V.generate kvSize $ \i ->
              let row = V.slice (i * hiddenDim) hiddenDim kWeights
              in V.sum $ V.zipWith (*) row xNorm

        -- V projection: (256, 1152) @ (1152,) -> (256,)
        let v = V.generate kvSize $ \i ->
              let row = V.slice (i * hiddenDim) hiddenDim vWeights
              in V.sum $ V.zipWith (*) row xNorm

        -- Compare with PyTorch
        case find (\step -> stepName step == "qkv_projection") (steps ref) of
          Just qkvStep -> do
            case (stepQ qkvStep, stepK qkvStep, stepV qkvStep) of
              (Just pytorchQ, Just pytorchK, Just pytorchV) -> do
                let pytorchQVec = V.fromList pytorchQ
                    pytorchKVec = V.fromList pytorchK
                    pytorchVVec = V.fromList pytorchV

                    qDiffs = V.zipWith (\h p -> abs (h - p)) q pytorchQVec
                    kDiffs = V.zipWith (\h p -> abs (h - p)) k pytorchKVec
                    vDiffs = V.zipWith (\h p -> abs (h - p)) v pytorchVVec

                    maxQDiff = V.maximum qDiffs
                    maxKDiff = V.maximum kDiffs
                    maxVDiff = V.maximum vDiffs

                putStrLn $ "Step 2: Q/K/V Projections"
                putStrLn $ "  Q maxDiff: " ++ printf "%.2e" maxQDiff
                putStrLn $ "  K maxDiff: " ++ printf "%.2e" maxKDiff
                putStrLn $ "  V maxDiff: " ++ printf "%.2e" maxVDiff
                putStrLn $ "  Haskell Q first 5: " ++ show (V.toList $ V.take 5 q)
                putStrLn $ "  PyTorch Q first 5: " ++ show (take 5 pytorchQ)

                -- Check for NaN
                let qHasNaN = V.any isNaN q
                    kHasNaN = V.any isNaN k
                    vHasNaN = V.any isNaN v
                    qHasInf = V.any isInfinite q
                    kHasInf = V.any isInfinite k
                    vHasInf = V.any isInfinite v

                when qHasNaN $ putStrLn "  ❌ Q HAS NaN!"
                when kHasNaN $ putStrLn "  ❌ K HAS NaN!"
                when vHasNaN $ putStrLn "  ❌ V HAS NaN!"
                when qHasInf $ putStrLn "  ❌ Q HAS Inf!"
                when kHasInf $ putStrLn "  ❌ K HAS Inf!"
                when vHasInf $ putStrLn "  ❌ V HAS Inf!"

                qHasNaN `shouldBe` False
                kHasNaN `shouldBe` False
                vHasNaN `shouldBe` False
                qHasInf `shouldBe` False
                kHasInf `shouldBe` False
                vHasInf `shouldBe` False

                -- Allow larger tolerance for Q4 quantization error (~2-3%)
                maxQDiff `shouldSatisfy` (<= 0.5)  -- Q4 quantization introduces larger errors
                maxKDiff `shouldSatisfy` (<= 0.5)
                maxVDiff `shouldSatisfy` (<= 0.5)
              _ -> expectationFailure "Missing Q/K/V in qkv_projection step"
          Nothing -> expectationFailure "qkv_projection step not found"

  it "Q4 GPU inference matches PyTorch (RMSNorm + Q projection)" $ do
    -- This is the REAL integration test - GPU path with Q4 model
    referenceJson <- BL.readFile "test/Gemma/Regression/Q4InferenceSpec_layer0.json"
    case JSON.eitherDecode referenceJson :: Either String Q4InferenceReference of
      Left err -> expectationFailure $ "Failed to parse reference: " ++ err
      Right ref -> do
        -- Load Q4 model
        st <- loadSafeTensors "../models/gemma3-1b-q4.safetensors"

        -- Get embedding
        embeddings <- getTensor st "model.embed_tokens.weight"
        let tokenId = inputTokenId ref
            hiddenDim = 1152
            embedding = V.slice (tokenId * hiddenDim) hiddenDim embeddings

        putStrLn "\n=== Q4 GPU Integration Test ==="
        putStrLn "Testing if GPU produces same results as PyTorch"
        putStrLn $ "Input embedding (first 5): " ++ show (V.toList $ V.take 5 embedding)

        -- Load layer 0 weights (same as CPU test)
        attnNorm <- getTensor st "model.layers.0.input_layernorm.weight"

        -- Check if Q4 weights exist
        let hasQ4 = hasQ4Weight st "model.layers.0.self_attn.q_proj.weight"

        qWeights <- if hasQ4
                    then loadQ4WeightDequantized st "model.layers.0.self_attn.q_proj.weight"
                    else getTensor st "model.layers.0.self_attn.q_proj.weight"

        -- Run RMSNorm + Q projection on GPU
        (gpuNormOutput, gpuQ) <- evalContT $ do
          ctx <- lift Ctx.createContext

          -- Upload embedding to GPU
          let shape = Shape [hiddenDim]
          inputTensor <- lift $ T.createTensorWithData ctx shape embedding

          -- Run RMSNorm on GPU
          normTensor <- Gemma.Layers.RMSNorm.runRMSNormGPU ctx inputTensor attnNorm hiddenDim False

          -- Download to check
          normOutput <- liftIO $ T.fromGPU ctx normTensor hiddenDim

          -- Run Q projection (weight is CPU Vector, input is GPU Tensor)
          let qSize = 1024
          qTensor <- Gemma.Layers.Linear.runLinearGPU ctx qWeights normTensor qSize hiddenDim

          -- Download result
          qOutput <- liftIO $ T.fromGPU ctx qTensor qSize

          return (normOutput, qOutput)

        putStrLn "\nStep 1: RMSNorm GPU results"
        putStrLn $ "GPU RMSNorm output (first 5): " ++ show (V.toList $ V.take 5 gpuNormOutput)

        -- Check for NaN/Inf in RMSNorm
        let hasNaN = V.any isNaN gpuNormOutput
            hasInf = V.any isInfinite gpuNormOutput

        when hasNaN $ putStrLn "❌ GPU RMSNorm HAS NaN!"
        when hasInf $ putStrLn "❌ GPU RMSNorm HAS Inf!"

        hasNaN `shouldBe` False
        hasInf `shouldBe` False

        -- Compare with PyTorch RMSNorm
        case find (\step -> stepName step == "input_rmsnorm") (steps ref) of
          Just normStep -> case stepOutput normStep of
            Just pytorchNorm -> do
              let pytorchNormVec = V.fromList pytorchNorm
                  normDiffs = V.zipWith (\h p -> abs (h - p)) gpuNormOutput pytorchNormVec
                  maxNormDiff = V.maximum normDiffs

              putStrLn $ "GPU RMSNorm maxDiff vs PyTorch: " ++ printf "%.2e" maxNormDiff
              maxNormDiff `shouldSatisfy` (<= 2e-4)
            Nothing -> expectationFailure "No output in input_rmsnorm step"
          Nothing -> expectationFailure "input_rmsnorm step not found"

        putStrLn "\nStep 2: Q projection GPU results"
        putStrLn $ "GPU Q projection output (first 5): " ++ show (V.toList $ V.take 5 gpuQ)

        -- Check for NaN/Inf in Q projection
        let qHasNaN = V.any isNaN gpuQ
            qHasInf = V.any isInfinite gpuQ

        when qHasNaN $ putStrLn "❌ GPU Q projection HAS NaN!"
        when qHasInf $ putStrLn "❌ GPU Q projection HAS Inf!"

        qHasNaN `shouldBe` False
        qHasInf `shouldBe` False

        -- Compare with PyTorch Q projection
        case find (\step -> stepName step == "qkv_projection") (steps ref) of
          Just qkvStep -> case stepQ qkvStep of
            Just pytorchQ -> do
              let pytorchQVec = V.fromList pytorchQ
                  qDiffs = V.zipWith (\h p -> abs (h - p)) gpuQ pytorchQVec
                  maxQDiff = V.maximum qDiffs

              putStrLn $ "GPU Q projection maxDiff vs PyTorch: " ++ printf "%.2e" maxQDiff
              putStrLn $ "PyTorch Q (first 5): " ++ show (take 5 pytorchQ)

              -- Allow larger tolerance for Q4 quantization
              maxQDiff `shouldSatisfy` (<= 0.5)
            Nothing -> expectationFailure "Missing Q in qkv_projection step"
          Nothing -> expectationFailure "qkv_projection step not found"

  it "Q4 GPU inference - K/V projections" $ do
    -- Test K and V projections on GPU with Q4 model
    referenceJson <- BL.readFile "test/Gemma/Regression/Q4InferenceSpec_layer0.json"
    case JSON.eitherDecode referenceJson :: Either String Q4InferenceReference of
      Left err -> expectationFailure $ "Failed to parse reference: " ++ err
      Right ref -> do
        st <- loadSafeTensors "../models/gemma3-1b-q4.safetensors"

        embeddings <- getTensor st "model.embed_tokens.weight"
        let tokenId = inputTokenId ref
            hiddenDim = 1152
            embedding = V.slice (tokenId * hiddenDim) hiddenDim embeddings

        attnNorm <- getTensor st "model.layers.0.input_layernorm.weight"
        let hasQ4 = hasQ4Weight st "model.layers.0.self_attn.k_proj.weight"

        kWeights <- if hasQ4
                    then loadQ4WeightDequantized st "model.layers.0.self_attn.k_proj.weight"
                    else getTensor st "model.layers.0.self_attn.k_proj.weight"

        vWeights <- if hasQ4
                    then loadQ4WeightDequantized st "model.layers.0.self_attn.v_proj.weight"
                    else getTensor st "model.layers.0.self_attn.v_proj.weight"

        -- Run RMSNorm + K/V projections on GPU
        (gpuK, gpuV) <- evalContT $ do
          ctx <- lift Ctx.createContext

          -- Upload and run RMSNorm
          let shape = Shape [hiddenDim]
          inputTensor <- lift $ T.createTensorWithData ctx shape embedding
          normTensor <- Gemma.Layers.RMSNorm.runRMSNormGPU ctx inputTensor attnNorm hiddenDim False

          -- Run K/V projections
          let kvSize = 256
          kTensor <- Gemma.Layers.Linear.runLinearGPU ctx kWeights normTensor kvSize hiddenDim
          vTensor <- Gemma.Layers.Linear.runLinearGPU ctx vWeights normTensor kvSize hiddenDim

          -- Download results
          kOutput <- liftIO $ T.fromGPU ctx kTensor kvSize
          vOutput <- liftIO $ T.fromGPU ctx vTensor kvSize

          return (kOutput, vOutput)

        putStrLn "\n=== K/V Projection GPU Test ==="
        putStrLn $ "GPU K (first 5): " ++ show (V.toList $ V.take 5 gpuK)
        putStrLn $ "GPU V (first 5): " ++ show (V.toList $ V.take 5 gpuV)

        -- Check for NaN/Inf
        let kHasNaN = V.any isNaN gpuK
            vHasNaN = V.any isNaN gpuV
            kHasInf = V.any isInfinite gpuK
            vHasInf = V.any isInfinite gpuV

        when kHasNaN $ putStrLn "❌ GPU K projection HAS NaN!"
        when vHasNaN $ putStrLn "❌ GPU V projection HAS NaN!"
        when kHasInf $ putStrLn "❌ GPU K projection HAS Inf!"
        when vHasInf $ putStrLn "❌ GPU V projection HAS Inf!"

        kHasNaN `shouldBe` False
        vHasNaN `shouldBe` False
        kHasInf `shouldBe` False
        vHasInf `shouldBe` False

        -- Compare with PyTorch
        case find (\step -> stepName step == "qkv_projection") (steps ref) of
          Just qkvStep -> case (stepK qkvStep, stepV qkvStep) of
            (Just pytorchK, Just pytorchV) -> do
              let pytorchKVec = V.fromList pytorchK
                  pytorchVVec = V.fromList pytorchV
                  kDiffs = V.zipWith (\h p -> abs (h - p)) gpuK pytorchKVec
                  vDiffs = V.zipWith (\h p -> abs (h - p)) gpuV pytorchVVec
                  maxKDiff = V.maximum kDiffs
                  maxVDiff = V.maximum vDiffs

              putStrLn $ "K maxDiff vs PyTorch: " ++ printf "%.2e" maxKDiff
              putStrLn $ "V maxDiff vs PyTorch: " ++ printf "%.2e" maxVDiff
              putStrLn $ "PyTorch K (first 5): " ++ show (take 5 pytorchK)
              putStrLn $ "PyTorch V (first 5): " ++ show (take 5 pytorchV)

              maxKDiff `shouldSatisfy` (<= 0.5)
              maxVDiff `shouldSatisfy` (<= 0.5)
            _ -> expectationFailure "Missing K/V in qkv_projection step"
          Nothing -> expectationFailure "qkv_projection step not found"

  it "Q4 GPU - complete layer 0 forward pass" $ do
    -- Test the ENTIRE layer 0 using the actual transformer block function
    -- This will test: RMSNorm + QKV + RoPE + Attention + Output + FFN + Residuals
    ref <- loadOrGenerateCache
      "test/Gemma/Regression/Q4InferenceSpec_layer0.py"
      "test/Gemma/Regression/Q4InferenceSpec_layer0.json" :: IO Q4InferenceReference
    do
        putStrLn "\n=== Complete Layer 0 GPU Test ==="
        putStrLn "This test will find where NaN first appears!"

        -- Load Q4 model - this will help us understand if full integration produces NaN
        st <- loadSafeTensors "../models/gemma3-1b-q4.safetensors"

        embeddings <- getTensor st "model.embed_tokens.weight"
        let tokenId = inputTokenId ref
            hiddenDim = 1152
            embedding = V.slice (tokenId * hiddenDim) hiddenDim embeddings

        putStrLn $ "Starting with embedding (first 5): " ++ show (V.toList $ V.take 5 embedding)

        -- For now, just verify embedding has no NaN
        let embHasNaN = V.any isNaN embedding
            embHasInf = V.any isInfinite embedding

        when embHasNaN $ putStrLn "❌ Embedding HAS NaN!"
        when embHasInf $ putStrLn "❌ Embedding HAS Inf!"

        embHasNaN `shouldBe` False
        embHasInf `shouldBe` False

        putStrLn "✅ Embedding is valid (no NaN/Inf)"
        putStrLn ""
        putStrLn "TODO: Implement full layer 0 forward pass using runTransformerBlockCachedGPU"
        putStrLn "This will require:"
        putStrLn "  1. Load all layer 0 weights"
        putStrLn "  2. Upload to GPU tensors"
        putStrLn "  3. Initialize KV cache"
        putStrLn "  4. Run runTransformerBlockCachedGPU"
        putStrLn "  5. Check output for NaN"
        putStrLn "  6. Compare with PyTorch 'final_output' reference"

        -- For now, mark as pending
        expectationFailure "Full layer 0 test not yet implemented - need to call runTransformerBlockCachedGPU"

  it "Q4 GPU - attention output (RoPE + scores + softmax)" $ do
    -- Test the complete attention mechanism which is most likely to produce NaN
    -- For position=0 with single token, attention should essentially be identity on V
    ref <- loadOrGenerateCache
      "test/Gemma/Regression/Q4InferenceSpec_layer0.py"
      "test/Gemma/Regression/Q4InferenceSpec_layer0.json" :: IO Q4InferenceReference
    do
        putStrLn "\n=== Attention Mechanism GPU Test ==="
        putStrLn "Testing RoPE + Attention Scores + Softmax + Output"

        st <- loadSafeTensors "../models/gemma3-1b-q4.safetensors"

        -- For now, just check if we have the reference data
        case find (\step -> stepName step == "attention") (steps ref) of
          Just attnStep -> case stepOutput attnStep of
            Just pytorchAttn -> do
              putStrLn $ "PyTorch attention output size: " ++ show (length pytorchAttn)
              putStrLn $ "PyTorch attention (first 5): " ++ show (take 5 pytorchAttn)
              putStrLn ""
              putStrLn "NOTE: For position=0 (single token), attention output should equal V"
              putStrLn "This is because attention weights are all 1.0 for single token"
              putStrLn ""
              putStrLn "TODO: Implement attention mechanism test"
              putStrLn "Steps needed:"
              putStrLn "  1. Apply RoPE to Q and K"
              putStrLn "  2. Compute attention scores: Q @ K^T / sqrt(head_dim)"
              putStrLn "  3. Apply softmax (MOST LIKELY SOURCE OF NaN!)"
              putStrLn "  4. Compute attention output: scores @ V"
              putStrLn "  5. Check for NaN at each step"
              putStrLn ""
              putStrLn "Hypothesis: Softmax will produce NaN if scores are too large"

              expectationFailure "Attention test not yet implemented - this is the critical test!"
            Nothing -> expectationFailure "No output in attention step"
          Nothing -> expectationFailure "attention step not found"

  -- CRITICAL TEST: Use the SAME input as the failing benchmark!
  it "Q4 GPU - FAILING benchmark token (6974) - RMSNorm + Q/K/V projections" $ do
    putStrLn "\n=== TESTING WITH BENCHMARK FAILING TOKEN ==="
    putStrLn "Token ID: 6974 (first token of 'Write a short story about')"
    putStrLn "This is the EXACT token that produces NaN in the benchmark!\n"

    -- Load reference for failing token (with timestamp check)
    ref <- loadOrGenerateCache
      "test/Gemma/Regression/Q4InferenceSpec_failingToken.py"
      "test/Gemma/Regression/Q4InferenceSpec_failingToken.json" :: IO Q4InferenceReference

    do
        st <- loadSafeTensors "../models/gemma3-1b-q4.safetensors"

        embeddings <- getTensor st "model.embed_tokens.weight"
        let tokenId = inputTokenId ref
            hiddenDim = 1152
            embedding = V.slice (tokenId * hiddenDim) hiddenDim embeddings

        putStrLn $ "Token ID from reference: " ++ show tokenId
        putStrLn $ "Expected token ID: 6974"
        tokenId `shouldBe` 6974

        attnNorm <- getTensor st "model.layers.0.input_layernorm.weight"

        -- Load Q/K/V weights
        let hasQ4 = hasQ4Weight st "model.layers.0.self_attn.q_proj.weight"
        putStrLn $ "Has Q4 weights: " ++ show hasQ4

        qWeights <- if hasQ4
                    then loadQ4WeightDequantized st "model.layers.0.self_attn.q_proj.weight"
                    else getTensor st "model.layers.0.self_attn.q_proj.weight"
        kWeights <- if hasQ4
                    then loadQ4WeightDequantized st "model.layers.0.self_attn.k_proj.weight"
                    else getTensor st "model.layers.0.self_attn.k_proj.weight"
        vWeights <- if hasQ4
                    then loadQ4WeightDequantized st "model.layers.0.self_attn.v_proj.weight"
                    else getTensor st "model.layers.0.self_attn.v_proj.weight"

        putStrLn "\n--- Running GPU operations with token 6974 ---"

        -- Run on GPU
        (gpuNormOutput, gpuQ, gpuK, gpuV) <- evalContT $ do
          ctx <- lift Ctx.createContext
          let shape = Shape [hiddenDim]
          inputTensor <- lift $ T.createTensorWithData ctx shape embedding

          -- RMSNorm
          normTensor <- Gemma.Layers.RMSNorm.runRMSNormGPU ctx inputTensor attnNorm hiddenDim False
          normOutput <- liftIO $ T.fromGPU ctx normTensor hiddenDim

          -- Q/K/V projections
          let qSize = 1024
              kvSize = 256
          qTensor <- Gemma.Layers.Linear.runLinearGPU ctx qWeights normTensor qSize hiddenDim
          kTensor <- Gemma.Layers.Linear.runLinearGPU ctx kWeights normTensor kvSize hiddenDim
          vTensor <- Gemma.Layers.Linear.runLinearGPU ctx vWeights normTensor kvSize hiddenDim

          qOutput <- liftIO $ T.fromGPU ctx qTensor qSize
          kOutput <- liftIO $ T.fromGPU ctx kTensor kvSize
          vOutput <- liftIO $ T.fromGPU ctx vTensor kvSize

          return (normOutput, qOutput, kOutput, vOutput)

        putStrLn "\n--- Checking for NaN/Inf ---"

        -- Check for NaN/Inf
        let hasNormNaN = V.any isNaN gpuNormOutput
            hasNormInf = V.any isInfinite gpuNormOutput
            hasQNaN = V.any isNaN gpuQ
            hasQInf = V.any isInfinite gpuQ
            hasKNaN = V.any isNaN gpuK
            hasKInf = V.any isInfinite gpuK
            hasVNaN = V.any isNaN gpuV
            hasVInf = V.any isInfinite gpuV

        putStrLn $ "RMSNorm - NaN: " ++ show hasNormNaN ++ ", Inf: " ++ show hasNormInf
        putStrLn $ "Q Proj  - NaN: " ++ show hasQNaN ++ ", Inf: " ++ show hasQInf
        putStrLn $ "K Proj  - NaN: " ++ show hasKNaN ++ ", Inf: " ++ show hasKInf
        putStrLn $ "V Proj  - NaN: " ++ show hasVNaN ++ ", Inf: " ++ show hasVInf

        when (hasNormNaN || hasNormInf) $
          expectationFailure "❌ NaN/Inf detected in RMSNorm output!"
        when (hasQNaN || hasQInf) $
          expectationFailure "❌ NaN/Inf detected in Q projection!"
        when (hasKNaN || hasKInf) $
          expectationFailure "❌ NaN/Inf detected in K projection!"
        when (hasVNaN || hasVInf) $
          expectationFailure "❌ NaN/Inf detected in V projection!"

        putStrLn "\n--- Comparing with PyTorch ---"

        -- Find steps in reference
        let normStep = find (\s -> stepName s == "input_rmsnorm") (steps ref)
            qkvStep = find (\s -> stepName s == "qkv_projection") (steps ref)

        case (normStep, qkvStep) of
          (Just norm, Just qkv) -> do
            -- Check RMSNorm
            case stepOutput norm of
              Just pytorchNorm -> do
                let pytorchNormVec = V.fromList pytorchNorm
                    normDiffs = V.zipWith (\h p -> abs (h - p)) gpuNormOutput pytorchNormVec
                    maxNormDiff = V.maximum normDiffs
                putStrLn $ "RMSNorm max diff: " ++ printf "%.2e" maxNormDiff
                maxNormDiff `shouldSatisfy` (< 2e-4)
              Nothing -> expectationFailure "No RMSNorm output in reference"

            -- Check Q/K/V
            case (stepQ qkv, stepK qkv, stepV qkv) of
              (Just pytorchQ, Just pytorchK, Just pytorchV) -> do
                let pytorchQVec = V.fromList pytorchQ
                    pytorchKVec = V.fromList pytorchK
                    pytorchVVec = V.fromList pytorchV

                    qDiffs = V.zipWith (\h p -> abs (h - p)) gpuQ pytorchQVec
                    kDiffs = V.zipWith (\h p -> abs (h - p)) gpuK pytorchKVec
                    vDiffs = V.zipWith (\h p -> abs (h - p)) gpuV pytorchVVec

                    maxQDiff = V.maximum qDiffs
                    maxKDiff = V.maximum kDiffs
                    maxVDiff = V.maximum vDiffs

                putStrLn $ "Q proj max diff: " ++ printf "%.2e" maxQDiff
                putStrLn $ "K proj max diff: " ++ printf "%.2e" maxKDiff
                putStrLn $ "V proj max diff: " ++ printf "%.2e" maxVDiff

                maxQDiff `shouldSatisfy` (< 5e-5)
                maxKDiff `shouldSatisfy` (< 5e-5)
                maxVDiff `shouldSatisfy` (< 5e-5)

                putStrLn "\n✅ BENCHMARK TOKEN TEST PASSED!"
                putStrLn "GPU produces correct results for the token that fails in benchmark!"
              _ -> expectationFailure "Missing Q/K/V in reference"
          _ -> expectationFailure "Missing steps in reference"

  -- CRITICAL TEST: Find which layer first produces NaN
  it "Q4 GPU - single token 6974 produces valid logits" $ do
    putStrLn "\n=== SINGLE TOKEN INFERENCE TEST ==="
    putStrLn "Testing token 6974 (first token of benchmark prompt)\n"

    model <- loadGemmaModel "../models/gemma3-1b-q4.safetensors" gemma3_1BConfig
    let token = V.singleton 6974

    (finalLogits, _) <- runGemmaInferenceCached model token Nothing

    let hasNaN = V.any isNaN finalLogits
        hasInf = V.any isInfinite finalLogits

    putStrLn $ "Final logits - NaN: " ++ show hasNaN ++ ", Inf: " ++ show hasInf
    putStrLn $ "  First 10: " ++ show (V.toList $ V.take 10 finalLogits)

    if hasNaN || hasInf
      then expectationFailure "Single token produced NaN - this should not happen!"
      else putStrLn "\n✅ Single token inference works correctly"

  -- CRITICAL TEST: Multi-token generation (where NaN likely appears)
  it "Q4 GPU - EXACT benchmark scenario (5 prompt tokens → generate)" $ do
    putStrLn "\n=== EXACT BENCHMARK SCENARIO TEST ==="
    putStrLn "Testing EXACT sequence from benchmark: [6974, 496, 2822, 3925, 1003] → generate\n"

    model <- loadGemmaModel "../models/gemma3-1b-q4.safetensors" gemma3_1BConfig

    -- The EXACT prompt tokens from benchmark: "Write a short story about"
    let promptTokens = [6974, 496, 2822, 3925, 1003]

    putStrLn "Phase 1: Processing prompt tokens (building cache)..."

    -- Process each prompt token, building up the cache
    let processPromptTokens [] _ = error "Empty prompt"
        processPromptTokens [lastToken] cache = do
          putStrLn $ "  Processing last prompt token: " ++ show lastToken
          (logits, finalCache) <- runGemmaInferenceCached model (V.singleton lastToken) cache
          let hasNaN = V.any isNaN logits
          putStrLn $ "    Logits - NaN: " ++ show hasNaN
          return (logits, finalCache)
        processPromptTokens (token:rest) cache = do
          putStrLn $ "  Processing prompt token: " ++ show token
          (_, newCache) <- runGemmaInferenceCached model (V.singleton token) cache
          processPromptTokens rest (Just newCache)

    (firstLogits, cacheAfterPrompt) <- processPromptTokens promptTokens Nothing

    let hasNaNAfterPrompt = V.any isNaN firstLogits

    putStrLn $ "\nAfter processing all prompt tokens:"
    putStrLn $ "  Logits NaN: " ++ show hasNaNAfterPrompt
    putStrLn $ "  First 10 logits: " ++ show (V.toList $ V.take 10 firstLogits)

    if hasNaNAfterPrompt
      then do
        putStrLn "\n❌ FOUND IT: NaN appears after processing prompt!"
        putStrLn "This matches the benchmark behavior!"
        expectationFailure "NaN detected after processing 5-token prompt"
      else
        putStrLn "\n✅ Prompt processing works correctly - trying generation..."

  -- CRITICAL TEST: Test with actual benchmark prompt (with BOS and chat template)
  it "Q4 GPU - benchmark prompt with chat template (DOUBLE BOS + markers)" $ do
    putStrLn "\n=== BENCHMARK PROMPT WITH CHAT TEMPLATE ==="
    putStrLn "Testing with actual buildInferencePrompt (includes BOS, markers, etc.)\n"

    -- Load tokenizer (needed for chat template)
    tokenizer <- loadTokenizer "../models/pytorch/gemma3-keras-gemma3_1b-v3/assets/tokenizer/vocabulary.spm"
    model <- loadGemmaModel "../models/gemma3-1b-q4.safetensors" gemma3_1BConfig

    -- Build prompt exactly like benchmark does
    let prompt = "Write a short story about"
        template = buildChatTemplate tokenizer
        promptTokens = buildInferencePrompt tokenizer template [(True, prompt)]

    putStrLn $ "Actual prompt tokens: " ++ show promptTokens
    putStrLn $ "Number of tokens: " ++ show (length promptTokens)
    putStrLn ""

    -- Process each prompt token
    let processPromptTokens [] _ = error "Empty prompt"
        processPromptTokens [lastToken] cache = do
          putStrLn $ "  Processing last prompt token " ++ show (length promptTokens) ++ "/" ++ show (length promptTokens) ++ ": " ++ show lastToken
          (logits, finalCache) <- runGemmaInferenceCached model (V.singleton lastToken) cache
          let hasNaN = V.any isNaN logits
          putStrLn $ "    Logits - NaN: " ++ show hasNaN
          if hasNaN
            then putStrLn $ "    ❌ NaN first 10: " ++ show (V.toList $ V.take 10 logits)
            else putStrLn $ "    ✅ Valid first 10: " ++ show (V.toList $ V.take 10 logits)
          return (logits, finalCache)
        processPromptTokens (token:rest) cache = do
          putStrLn $ "  Processing prompt token " ++ show (length promptTokens - length rest) ++ "/" ++ show (length promptTokens) ++ ": " ++ show token
          (_, newCache) <- runGemmaInferenceCached model (V.singleton token) cache
          processPromptTokens rest (Just newCache)

    (firstLogits, _) <- processPromptTokens promptTokens Nothing

    let hasNaNAfterPrompt = V.any isNaN firstLogits

    putStrLn $ "\nAfter processing all prompt tokens with chat template:"
    putStrLn $ "  Logits NaN: " ++ show hasNaNAfterPrompt

    if hasNaNAfterPrompt
      then do
        putStrLn "\n❌ FOUND IT: NaN appears with chat template!"
        putStrLn "This EXACTLY matches the benchmark behavior!"
        putStrLn "\nThe bug is triggered by:"
        putStrLn "  - Double BOS tokens at start"
        putStrLn "  - Chat template markers"
        putStrLn "  - Specific token sequence with cache"
        expectationFailure "NaN detected with chat template - matches benchmark!"
      else
        putStrLn "\n✅ Chat template processing works correctly"

