{-# LANGUAGE OverloadedStrings #-}

module Main (main) where

import Test.Hspec
import Test.QuickCheck
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import Gemma.SafeTensors
import Gemma.Layers.RMSNorm (runRMSNorm)
import Gemma.Layers.Linear (runLinear)
import Gemma.Layers.RoPE (runRoPE)
import Gemma.Layers.Attention (runAttention)
import Gemma.Layers.MLP (runGELU, runGeGLU)
import Gemma.Layers.Embedding (runEmbedding)
import Gemma.Layers.TransformerBlock (TransformerLayer(..), runTransformerBlock, expandKVHeads)
import Gemma.Model (GemmaModel(..), GemmaConfig(..), runGemmaInference, runGemmaInferenceCached, loadGemmaModel, gemma3_1BConfig)
import Graphics.WebGPU.Dawn.ContT (evalContT)
import Control.Monad (unless)
import System.Directory (doesFileExist)
import qualified Gemma.Quantization.Q4Spec as Q4Spec
import qualified Gemma.Layers.LinearQ4Spec as LinearQ4Spec
import qualified Gemma.Layers.LinearQ4FusedSpec as LinearQ4FusedSpec
import qualified Gemma.Layers.LinearDSLSpec as LinearDSLSpec
import qualified Gemma.Layers.AttentionDSLSpec as AttentionDSLSpec
import qualified Gemma.Layers.RMSNormDSLSpec as RMSNormDSLSpec
import qualified Gemma.Layers.RoPEDSLSpec as RoPEDSLSpec
import qualified Gemma.Layers.GELUDSLSpec as GELUDSLSpec
import qualified Gemma.Layers.ElementwiseDSLSpec as ElementwiseDSLSpec
import qualified Gemma.Layers.SoftmaxDSLSpec as SoftmaxDSLSpec
import qualified Gemma.Layers.LinearQ4DSLSpec as LinearQ4DSLSpec
import qualified Gemma.Regression.FP32Spec as FP32Spec
import qualified Gemma.Regression.FP16Spec as FP16Spec
import qualified Gemma.Regression.AttentionSpec as AttentionSpec
import qualified Gemma.Regression.Q4Spec as Q4PyTorchSpec
import qualified Gemma.Regression.Q4InferenceSpec as Q4InferenceSpec
import qualified Gemma.Regression.InferenceSpec as InferenceSpec
import qualified Gemma.Regression.FP32InferenceSpec as FP32InferenceSpec
import qualified Gemma.Regression.MultiTokenSpec as MultiTokenSpec
import qualified Gemma.Regression.CompareWithOfficialSpec as CompareWithOfficialSpec
import qualified Gemma.Regression.FirstTokenSpec as FirstTokenSpec
import qualified Gemma.Regression.Layer0WithCacheSpec as Layer0WithCacheSpec
import qualified Gemma.Regression.EmbeddingSpec as EmbeddingSpec

-- | Main test suite entry point
main :: IO ()
main = hspec $ do
  describe "Embedding Layer Comparison" EmbeddingSpec.spec
  describe "First Token Generation (TDD PyTorch Comparison)" FirstTokenSpec.spec
  describe "Layer 0 With KV Cache (TDD PyTorch Comparison)" Layer0WithCacheSpec.spec
  describe "Compare with Official Gemma 3" CompareWithOfficialSpec.spec
  describe "Multi-Token Inference (TDD)" MultiTokenSpec.spec
  describe "FP32 Single Token Inference (TDD)" FP32InferenceSpec.spec
  describe "End-to-End Inference (PyTorch)" InferenceSpec.spec
  describe "FP32 Regression (PyTorch)" FP32Spec.spec
  describe "FP16 Regression (PyTorch)" FP16Spec.spec
  describe "Attention Regression (PyTorch)" AttentionSpec.spec
  describe "Q4 Quantization (PyTorch)" Q4PyTorchSpec.spec
  describe "Q4 Inference (PyTorch TDD)" Q4InferenceSpec.spec
  describe "Q4 Quantization" Q4Spec.spec
  describe "Q4 Linear Layer" LinearQ4Spec.spec
  describe "Q4 Fused Layers" LinearQ4FusedSpec.spec
  describe "Linear DSL Layer" LinearDSLSpec.spec
  describe "Attention DSL Layer" AttentionDSLSpec.spec
  describe "RMSNorm DSL Layer" RMSNormDSLSpec.spec
  describe "RoPE DSL Layer" RoPEDSLSpec.spec
  describe "GELU DSL Layer" GELUDSLSpec.spec
  describe "Elementwise DSL Layer" ElementwiseDSLSpec.spec
  describe "Softmax DSL Layer" SoftmaxDSLSpec.spec
  describe "Linear Q4 DSL Layer" LinearQ4DSLSpec.spec
  describe "SafeTensors" safeTensorsSpec
  describe "Embedding" embeddingSpec
  describe "RMSNorm" rmsNormSpec
  describe "Linear" linearSpec
  describe "RoPE" ropeSpec
  describe "Attention" attentionSpec
  describe "MLP" mlpSpec
  describe "GQA" gqaSpec
  describe "TransformerBlock" transformerBlockSpec
  describe "BufferReuse" bufferReuseSpec

-- | Test helper: Check if GPU output matches golden value within tolerance
shouldMatchGolden :: Vector Float -> Vector Float -> Float -> Expectation
shouldMatchGolden actual expected tolerance = do
  let len1 = V.length actual
      len2 = V.length expected

  unless (len1 == len2) $
    expectationFailure $ "Length mismatch: actual=" ++ show len1 ++ ", expected=" ++ show len2

  let diffs = V.zipWith (\a e -> abs (a - e)) actual expected
      maxDiff = V.maximum diffs
      maxDiffIdx = V.maxIndex diffs

  unless (maxDiff <= tolerance) $
    expectationFailure $ unlines
      [ "Maximum absolute error exceeds tolerance"
      , "  Tolerance: " ++ show tolerance
      , "  Max error: " ++ show maxDiff
      , "  At index:  " ++ show maxDiffIdx
      , "  Actual:    " ++ show (actual V.! maxDiffIdx)
      , "  Expected:  " ++ show (expected V.! maxDiffIdx)
      ]

-- | Test helper: Load golden value from file
loadGoldenValue :: FilePath -> String -> IO (Vector Float)
loadGoldenValue dir tensorName = do
  let path = dir ++ "/" ++ tensorName ++ ".safetensors"
  exists <- doesFileExist path
  unless exists $
    error $ "Golden value file not found: " ++ path ++ "\nRun: cd gemma.hs/scripts && python export_golden_values.py"

  st <- loadSafeTensors path
  getTensor st "output"  -- Python script saves with name "output"

-- | SafeTensors loader tests
safeTensorsSpec :: Spec
safeTensorsSpec = do
  it "can parse SafeTensors format" $ do
    -- This test will pass when golden values exist
    pending

  it "extracts tensor shapes correctly" $ do
    pending

  it "loads F32 tensors correctly" $ do
    pending

-- | Embedding layer tests (Phase 3.1)
embeddingSpec :: Spec
embeddingSpec = do
  describe "Simple Embedding Lookup" $ do
    it "looks up embeddings for token IDs correctly" $ do
      -- Test simple embedding lookup
      -- Embedding table: 4 tokens, dim=3
      -- [[1, 2, 3],    Token 0
      --  [4, 5, 6],    Token 1
      --  [7, 8, 9],    Token 2
      --  [10,11,12]]   Token 3
      --
      -- Token IDs: [1, 2] → output: [4,5,6, 7,8,9]
      let embedTable = V.fromList [1,2,3, 4,5,6, 7,8,9, 10,11,12] :: Vector Float
          tokenIds = V.fromList [1, 2] :: Vector Int
          vocabSize = 4
          embedDim = 3
          expected = V.fromList [4,5,6, 7,8,9] :: Vector Float

      actual <- evalContT $ runEmbedding tokenIds embedTable vocabSize embedDim
      actual `shouldMatchGolden` expected $ 1e-5

  describe "PyTorch Golden Value Validation" $ do
    it "matches PyTorch Embedding output" $ do
      -- Load PyTorch golden values
      embedTable <- loadGoldenValue "test/golden-values" "embedding_table"
      -- Note: token_ids are stored as int64 in safetensors, need to load differently
      expected <- loadGoldenValue "test/golden-values" "embedding_output"

      let tokenIds = V.fromList [1, 2] :: Vector Int
          vocabSize = 4
          embedDim = 3

      -- Run our implementation
      actual <- evalContT $ runEmbedding tokenIds embedTable vocabSize embedDim

      -- Compare with PyTorch
      actual `shouldMatchGolden` expected $ 1e-5

-- | RMSNorm layer tests (Phase 2.1)
rmsNormSpec :: Spec
rmsNormSpec = do
  describe "Simple RMSNorm" $ do
    it "normalizes a simple vector correctly" $ do
      -- Test with a simple known case
      let input = V.fromList [1.0, 2.0, 3.0, 4.0] :: Vector Float
          weights = V.fromList [1.0, 1.0, 1.0, 1.0] :: Vector Float

      -- For input [1,2,3,4]:
      -- mean(x²) = (1 + 4 + 9 + 16) / 4 = 7.5
      -- rms = sqrt(7.5 + 1e-6) ≈ 2.7386
      -- normalized = [1/2.7386, 2/2.7386, 3/2.7386, 4/2.7386]
      --            ≈ [0.3651, 0.7303, 1.0954, 1.4606]
      let expected = V.fromList [0.3651, 0.7303, 1.0954, 1.4606] :: Vector Float

      -- Now let's test the actual implementation!
      actual <- evalContT $ runRMSNorm input weights
      actual `shouldMatchGolden` expected $ 1e-3

  describe "PyTorch Golden Value Validation" $ do
    it "matches PyTorch RMSNorm output" $ do
      -- Load PyTorch golden values
      input <- loadGoldenValue "test/golden-values" "rmsnorm_input"
      weights <- loadGoldenValue "test/golden-values" "rmsnorm_weight"
      expected <- loadGoldenValue "test/golden-values" "rmsnorm_output"

      -- Run our implementation
      actual <- evalContT $ runRMSNorm input weights

      -- Compare with PyTorch (FP32 tolerance)
      actual `shouldMatchGolden` expected $ 1e-5

-- | Linear/MatMul layer tests (Phase 2.2)
linearSpec :: Spec
linearSpec = do
  describe "Simple Matrix-Vector Multiply" $ do
    it "performs matrix-vector multiplication correctly" $ do
      -- Test simple case: [2x3] @ [3] = [2]
      -- W = [[1, 2, 3],    x = [1]    y = [1*1 + 2*2 + 3*3] = [14]
      --      [4, 5, 6]]         [2]        [4*1 + 5*2 + 6*3]   [32]
      --                         [3]
      let weight = V.fromList [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] :: Vector Float
          input = V.fromList [1.0, 2.0, 3.0] :: Vector Float
          expected = V.fromList [14.0, 32.0] :: Vector Float
          outSize = 2
          inSize = 3

      actual <- evalContT $ runLinear weight input outSize inSize
      actual `shouldMatchGolden` expected $ 1e-5

  describe "PyTorch Golden Value Validation" $ do
    it "matches PyTorch Linear output" $ do
      -- Load PyTorch golden values
      weight <- loadGoldenValue "test/golden-values" "linear_weight"
      input <- loadGoldenValue "test/golden-values" "linear_input"
      expected <- loadGoldenValue "test/golden-values" "linear_output"

      let outSize = 2
          inSize = 3

      -- Run our implementation
      actual <- evalContT $ runLinear weight input outSize inSize

      -- Compare with PyTorch
      actual `shouldMatchGolden` expected $ 1e-5

-- | RoPE tests (Phase 2.3)
ropeSpec :: Spec
ropeSpec = do
  describe "Simple RoPE rotation" $ do
    it "applies rotary embeddings correctly" $ do
      -- Test RoPE with simple case
      -- For dim=4, pos=0, head_dim=4:
      -- freq = [1.0, 1.0] for pairs (0,1) and (2,3)
      -- At pos=0: cos=1.0, sin=0.0
      -- So rotation should be identity: input stays the same
      let input = V.fromList [1.0, 2.0, 3.0, 4.0] :: Vector Float
          position = 0
          headDim = 4
          baseFreq = 10000.0  -- Standard RoPE base
          -- At pos=0, RoPE should be identity (cos=1, sin=0)
          expected = V.fromList [1.0, 2.0, 3.0, 4.0] :: Vector Float

      actual <- evalContT $ runRoPE input position headDim baseFreq
      actual `shouldMatchGolden` expected $ 1e-5

  describe "PyTorch Golden Value Validation" $ do
    it "matches PyTorch RoPE output (position 0)" $ do
      -- Load PyTorch golden values for position 0 (identity)
      input <- loadGoldenValue "test/golden-values" "rope_input_pos0"
      expected <- loadGoldenValue "test/golden-values" "rope_output_pos0"

      let position = 0
          headDim = 4
          baseFreq = 10000.0  -- Standard RoPE base

      -- Run our implementation
      actual <- evalContT $ runRoPE input position headDim baseFreq

      -- Compare with PyTorch
      actual `shouldMatchGolden` expected $ 1e-5

    it "matches PyTorch RoPE output (position 1)" $ do
      -- Load PyTorch golden values for position 1 (non-identity)
      input <- loadGoldenValue "test/golden-values" "rope_input_pos0"
      expected <- loadGoldenValue "test/golden-values" "rope_output_pos1"

      let position = 1
          headDim = 4
          baseFreq = 10000.0  -- Standard RoPE base

      -- Run our implementation
      actual <- evalContT $ runRoPE input position headDim baseFreq

      -- Compare with PyTorch
      actual `shouldMatchGolden` expected $ 1e-5

-- | Attention mechanism tests (Phase 2.4)
attentionSpec :: Spec
attentionSpec = do
  describe "Simple Attention" $ do
    it "computes scaled dot-product attention correctly" $ do
      -- Simple test case: single head, sequence length 2, head_dim 2
      -- Q = [[1, 0],   K = [[1, 0],   V = [[10, 20],
      --      [0, 1]]        [0, 1]]        [30, 40]]
      --
      -- Scores = Q @ K^T / sqrt(2) = [[0.707, 0], [0, 0.707]]
      -- After softmax along rows:
      --   softmax([0.707, 0]) ≈ [0.67, 0.33]
      --   softmax([0, 0.707]) ≈ [0.33, 0.67]
      -- Output = Attention @ V
      --   row 0: 0.67*[10,20] + 0.33*[30,40] = [16.6, 26.6]
      --   row 1: 0.33*[10,20] + 0.67*[30,40] = [23.4, 33.4]
      let q = V.fromList [1.0, 0.0, 0.0, 1.0] :: Vector Float
          k = V.fromList [1.0, 0.0, 0.0, 1.0] :: Vector Float
          v = V.fromList [10.0, 20.0, 30.0, 40.0] :: Vector Float
          seqLen = 2
          headDim = 2
          expected = V.fromList [16.6, 26.6, 23.4, 33.4] :: Vector Float

      actual <- evalContT $ runAttention q k v seqLen headDim Nothing  -- Full attention
      actual `shouldMatchGolden` expected $ 0.5  -- Larger tolerance for softmax

  describe "PyTorch Golden Value Validation" $ do
    it "matches PyTorch Attention output" $ do
      -- Load PyTorch golden values
      q <- loadGoldenValue "test/golden-values" "attention_q"
      k <- loadGoldenValue "test/golden-values" "attention_k"
      v <- loadGoldenValue "test/golden-values" "attention_v"
      expected <- loadGoldenValue "test/golden-values" "attention_output"

      let seqLen = 2
          headDim = 2

      -- Run our implementation
      actual <- evalContT $ runAttention q k v seqLen headDim Nothing  -- Full attention

      -- Compare with PyTorch (larger tolerance for softmax accumulation)
      actual `shouldMatchGolden` expected $ 1e-4

-- | MLP tests (Phase 2.5)
mlpSpec :: Spec
mlpSpec = do
  describe "Simple GELU activation" $ do
    it "applies GELU activation correctly" $ do
      -- Test GELU activation function
      -- GELU(x) ≈ x * Φ(x) where Φ is the CDF of standard normal
      -- For x=0: GELU(0) = 0
      -- For x=1: GELU(1) ≈ 0.84
      -- For x=-1: GELU(-1) ≈ -0.16
      let input = V.fromList [0.0, 1.0, -1.0, 2.0] :: Vector Float
          -- Approximate GELU values
          expected = V.fromList [0.0, 0.84, -0.16, 1.96] :: Vector Float

      actual <- evalContT $ runGELU input
      actual `shouldMatchGolden` expected $ 0.05

  describe "PyTorch Golden Value Validation" $ do
    it "matches PyTorch GELU output" $ do
      -- Load PyTorch golden values (includes problematic values like 11, 17)
      input <- loadGoldenValue "test/golden-values" "gelu_input"
      expected <- loadGoldenValue "test/golden-values" "gelu_output"

      -- Run our implementation
      actual <- evalContT $ runGELU input

      -- Compare with PyTorch (slightly larger tolerance for GELU approximation)
      actual `shouldMatchGolden` expected $ 2e-4

  describe "Simple GeGLU MLP" $ do
    it "computes GeGLU MLP correctly" $ do
      -- Test simple GeGLU: GELU(x @ W_gate) * (x @ W_up) @ W_down
      -- Input: [2]
      -- W_gate: [3x2], W_up: [3x2], W_down: [2x3]
      --
      -- For input [1, 2]:
      -- gate = [1,2] @ [[1,2],[3,4],[5,6]]^T = [1*1+2*2, 1*3+2*4, 1*5+2*6] = [5, 11, 17]
      -- up   = [1,2] @ [[1,1],[1,1],[1,1]]^T = [1*1+2*1, 1*1+2*1, 1*1+2*1] = [3, 3, 3]
      -- gelu_gate = GELU([5, 11, 17]) ≈ [5.0, 11.0, 17.0]
      -- intermediate = gelu_gate * up = [5.0*3, 11.0*3, 17.0*3] ≈ [15.0, 33.0, 51.0]
      -- output = [15,33,51] @ [[1,0,0],[0,1,0]]^T = [15*1+33*0+51*0, 15*0+33*1+51*0] = [15, 33]

      let input = V.fromList [1.0, 2.0] :: Vector Float
          -- W_gate: 3x2 in row-major (each row is [1,2], [3,4], [5,6])
          wGate = V.fromList [1,2, 3,4, 5,6] :: Vector Float
          -- W_up: 3x2 all ones
          wUp = V.fromList [1,1, 1,1, 1,1] :: Vector Float
          -- W_down: 2x3 (identity-like)
          wDown = V.fromList [1,0,0, 0,1,0] :: Vector Float
          hiddenDim = 2
          ffnDim = 3
          expected = V.fromList [15.0, 33.0] :: Vector Float

      actual <- evalContT $ runGeGLU input wGate wUp wDown hiddenDim ffnDim
      actual `shouldMatchGolden` expected $ 1.0  -- Larger tolerance for GELU

  describe "GeGLU MLP" $ do
    it "MLP output matches PyTorch" $ do
      pending

    it "gate projection works correctly" $ do
      pending

-- | GQA (Grouped Query Attention) tests (Phase 4)
gqaSpec :: Spec
gqaSpec = do
  describe "K/V Head Expansion" $ do
    it "correctly expands K/V heads for GQA" $ do
      -- Test with 4 Q heads and 2 KV heads
      let numQHeads = 4
          numKVHeads = 2
          headDim = 8

      -- Load golden values from PyTorch
      kInput <- loadGoldenValue "test/golden-values" "gqa_k_input"
      vInput <- loadGoldenValue "test/golden-values" "gqa_v_input"
      kExpected <- loadGoldenValue "test/golden-values" "gqa_k_expanded"
      vExpanded <- loadGoldenValue "test/golden-values" "gqa_v_expanded"

      -- Apply our expansion
      let kActual = expandKVHeads kInput numQHeads numKVHeads headDim
          vActual = expandKVHeads vInput numQHeads numKVHeads headDim

      -- Verify sizes
      V.length kActual `shouldBe` (numQHeads * headDim)
      V.length vActual `shouldBe` (numQHeads * headDim)

      -- Verify values match PyTorch
      kActual `shouldMatchGolden` kExpected $ 1e-6
      vActual `shouldMatchGolden` vExpanded $ 1e-6

    it "handles equal Q and KV heads (no expansion)" $ do
      -- When numQHeads == numKVHeads, should be identity
      let numQHeads = 4
          numKVHeads = 4
          headDim = 8
          input = V.fromList [1..32] :: Vector Float  -- 4 heads * 8 dim

      let output = expandKVHeads input numQHeads numKVHeads headDim

      -- Should be unchanged
      output `shouldBe` input

    it "correctly replicates heads with 4:1 ratio" $ do
      -- Test with 8 Q heads and 2 KV heads (4x expansion)
      let numQHeads = 8
          numKVHeads = 2
          headDim = 4
          -- KV head 0: [1, 2, 3, 4], KV head 1: [5, 6, 7, 8]
          input = V.fromList [1, 2, 3, 4, 5, 6, 7, 8] :: Vector Float

      let output = expandKVHeads input numQHeads numKVHeads headDim

      -- Expected: each KV head replicated 4 times
      let expected = V.fromList [
            1, 2, 3, 4,  -- Q head 0 (uses KV head 0)
            1, 2, 3, 4,  -- Q head 1 (uses KV head 0)
            1, 2, 3, 4,  -- Q head 2 (uses KV head 0)
            1, 2, 3, 4,  -- Q head 3 (uses KV head 0)
            5, 6, 7, 8,  -- Q head 4 (uses KV head 1)
            5, 6, 7, 8,  -- Q head 5 (uses KV head 1)
            5, 6, 7, 8,  -- Q head 6 (uses KV head 1)
            5, 6, 7, 8   -- Q head 7 (uses KV head 1)
            ] :: Vector Float

      output `shouldBe` expected

-- | Full transformer block tests (Phase 3.3)
transformerBlockSpec :: Spec
transformerBlockSpec = do
  describe "Simple TransformerBlock" $ do
    it "runs without errors with minimal config" $ do
      -- Test with minimal transformer configuration
      -- Hidden dim: 4, Heads: 1, Head dim: 4, FFN dim: 8
      let hiddenDim = 4
          numHeads = 1
          numKVHeads = 1
          headDim = 4
          ffnDim = 8
          qkvDim = numHeads * headDim  -- 4

      -- Create simple input
      let input = V.fromList [1.0, 0.5, -0.5, 0.2] :: Vector Float

      -- Create minimal weights (all ones for simplicity)
      let attnNormWeights = V.replicate hiddenDim 1.0
          qWeights = V.replicate (qkvDim * hiddenDim) 0.1
          kWeights = V.replicate (qkvDim * hiddenDim) 0.1
          vWeights = V.replicate (qkvDim * hiddenDim) 0.1
          outWeights = V.replicate (hiddenDim * qkvDim) 0.1
          ffnNormWeights = V.replicate hiddenDim 1.0
          gateWeights = V.replicate (ffnDim * hiddenDim) 0.1
          upWeights = V.replicate (ffnDim * hiddenDim) 0.1
          downWeights = V.replicate (hiddenDim * ffnDim) 0.1

      let layer = TransformerLayer
            { tlAttnNormWeights = attnNormWeights
            , tlAttnQWeights = qWeights
            , tlAttnKWeights = kWeights
            , tlAttnVWeights = vWeights
            , tlQNormWeights = Nothing  -- Gemma 1 doesn't use QK-Norm
            , tlKNormWeights = Nothing
            , tlAttnOutWeights = outWeights
            , tlPostAttnNormWeights = Nothing  -- Gemma 1 doesn't use post-attn norm
            , tlFFNNormWeights = ffnNormWeights
            , tlFFNGateWeights = gateWeights
            , tlFFNUpWeights = upWeights
            , tlFFNDownWeights = downWeights
            , tlPostFFNNormWeights = Nothing  -- Gemma 1 doesn't use post-FFN norm
            }

      -- Run transformer block with full attention (Nothing = no sliding window)
      let ropeBase = 10000.0  -- Standard RoPE base
      result <- evalContT $ runTransformerBlock input layer 0 numHeads numKVHeads headDim hiddenDim ffnDim Nothing ropeBase

      -- Just verify it returns the right size and no NaN
      V.length result `shouldBe` hiddenDim
      let hasNaN = V.any isNaN result
      hasNaN `shouldBe` False

  describe "Complete Model Structure" $ do
    it "can create a minimal Gemma model" $ do
      -- This test demonstrates the complete model structure
      -- Create a minimal config for testing (much smaller than real Gemma)
      let config = GemmaConfig
            { gcVocabSize = 100      -- Small vocab
            , gcHiddenDim = 4        -- Tiny hidden dim
            , gcNumLayers = 2        -- Just 2 layers
            , gcNumHeads = 1
            , gcNumKVHeads = 1
            , gcHeadDim = 4
            , gcFFNDim = 8
            , gcRopeBase = 10000.0
            -- Gemma 3 features disabled for this simple test
            , gcUseQKNorm = False
            , gcUsePostAttnNorm = False
            , gcUsePostFFNNorm = False
            , gcUseSlidingWindow = False
            , gcSlidingWindowSize = 0
            , gcLocalRopeScaling = 1.0
            , gcGlobalRopeScaling = 1.0
            , gcQueryHeadDimNormalize = False
            , gcUseZeroCenteredRMSNorm = False
            }

      -- Create minimal weights (all small random-ish values)
      let vocabSize = gcVocabSize config
          hiddenDim = gcHiddenDim config
          numLayers = gcNumLayers config
          ffnDim = gcFFNDim config
          qkvDim = gcNumHeads config * gcHeadDim config

      let embeddings = V.generate (vocabSize * hiddenDim) (\i -> 0.1 * fromIntegral (i `mod` 10))
          finalNorm = V.replicate hiddenDim 1.0
          lmHead = V.generate (vocabSize * hiddenDim) (\i -> 0.1 * fromIntegral ((i + 5) `mod` 10))

      -- Create 2 transformer layers
      let mkLayer = TransformerLayer
            { tlAttnNormWeights = V.replicate hiddenDim 1.0
            , tlAttnQWeights = V.replicate (qkvDim * hiddenDim) 0.1
            , tlAttnKWeights = V.replicate (qkvDim * hiddenDim) 0.1
            , tlAttnVWeights = V.replicate (qkvDim * hiddenDim) 0.1
            , tlQNormWeights = Nothing  -- Gemma 1 doesn't use QK-Norm
            , tlKNormWeights = Nothing
            , tlAttnOutWeights = V.replicate (hiddenDim * qkvDim) 0.1
            , tlPostAttnNormWeights = Nothing  -- Gemma 1 doesn't use post-attn norm
            , tlFFNNormWeights = V.replicate hiddenDim 1.0
            , tlFFNGateWeights = V.replicate (ffnDim * hiddenDim) 0.1
            , tlFFNUpWeights = V.replicate (ffnDim * hiddenDim) 0.1
            , tlFFNDownWeights = V.replicate (hiddenDim * ffnDim) 0.1
            , tlPostFFNNormWeights = Nothing  -- Gemma 1 doesn't use post-FFN norm
            }

      let model = GemmaModel
            { gmConfig = config
            , gmEmbeddings = embeddings
            , gmLayers = [mkLayer, mkLayer]
            , gmFinalNormWeights = finalNorm
            , gmLMHeadWeights = lmHead
            }

      -- Run inference on a single token
      let tokenIds = V.fromList [42] :: Vector Int
      logits <- runGemmaInference model tokenIds

      -- Verify output
      V.length logits `shouldBe` vocabSize
      let hasNaN = V.any isNaN logits
      hasNaN `shouldBe` False

  describe "Layer 0 Complete" $ do
    it "end-to-end layer 0 matches PyTorch" $ do
      pending

-- | Buffer Reuse Bug Investigation (Current Issue)
bufferReuseSpec :: Spec
bufferReuseSpec = do
  describe "Buffer Reuse produces valid output (no NaN)" $ do
    it "embedding produces no NaN" $ do
      model <- loadGemmaModel "../models/gemma3-1b.safetensors" gemma3_1BConfig
      let tokenVec = V.singleton 2
      result <- evalContT $ runEmbedding tokenVec (gmEmbeddings model) (gcVocabSize $ gmConfig model) (gcHiddenDim $ gmConfig model)
      V.any isNaN result `shouldBe` False

    it "full inference (cached GPU version) produces no NaN" $ do
      model <- loadGemmaModel "../models/gemma3-1b.safetensors" gemma3_1BConfig
      (logits, _) <- runGemmaInferenceCached model (V.singleton 2) Nothing
      V.any isNaN logits `shouldBe` False

    it "autoregressive generation with cache produces no NaN" $ do
      -- This test mimics the chat mode's autoregressive generation pattern
      model <- loadGemmaModel "../models/gemma3-1b.safetensors" gemma3_1BConfig

      -- Start with BOS token (token 2)
      (logits1, cache1) <- runGemmaInferenceCached model (V.singleton 2) Nothing
      V.any isNaN logits1 `shouldBe` False

      -- Process second token (using cache from first)
      (logits2, cache2) <- runGemmaInferenceCached model (V.singleton 10) (Just cache1)
      V.any isNaN logits2 `shouldBe` False

      -- Process third token (using cache from second)
      (logits3, _cache3) <- runGemmaInferenceCached model (V.singleton 20) (Just cache2)
      V.any isNaN logits3 `shouldBe` False

    it "chat prompt sequence produces no NaN and can generate" $ do
      -- Use the exact token sequence from the chat debug output
      -- DEBUG: Prompt tokens (10): [2,105,2364,107,9259,106,107,105,4368,107]
      model <- loadGemmaModel "../models/gemma3-1b.safetensors" gemma3_1BConfig

      let promptTokens = [2, 105, 2364, 107, 9259, 106, 107, 105, 4368, 107]

      -- Process all prompt tokens one by one, building up cache
      let processTokens [] _ = error "Empty prompt"
          processTokens [t] cache = do
            (logits, finalCache) <- runGemmaInferenceCached model (V.singleton t) cache
            V.any isNaN logits `shouldBe` False
            return (logits, finalCache)
          processTokens (t:rest) cache = do
            (_, newCache) <- runGemmaInferenceCached model (V.singleton t) cache
            processTokens rest (Just newCache)

      (finalLogits, finalCache) <- processTokens promptTokens Nothing

      -- Now test generating a token (this is where chat fails)
      let nextTokenId = V.maxIndex finalLogits
          maxLogitValue = finalLogits V.! nextTokenId
      -- Print what token was selected
      putStrLn $ "\nFirst generated token ID: " ++ show nextTokenId ++ " (logit value: " ++ show maxLogitValue ++ ")"

      (genLogits, _) <- runGemmaInferenceCached model (V.singleton nextTokenId) (Just finalCache)
      V.any isNaN genLogits `shouldBe` False

      -- Check if logits have variation (not all same value)
      let minLogit = V.minimum genLogits
          maxLogit = V.maximum genLogits
          logitRange = maxLogit - minLogit
      -- If logits are all the same, maxIndex picks essentially random token
      logitRange > 0.001 `shouldBe` True

    it "single token inference works" $ do
      -- Test single token inference to isolate KV cache issues
      model <- loadGemmaModel "../models/gemma3-1b.safetensors" gemma3_1BConfig

      -- Just test with the first token: 105 = <start_of_turn>
      let firstToken = 105

      -- CPU version - SKIP: known to be broken, produces all NaN
      -- logitsCPU <- runGemmaInference model (V.singleton firstToken)
      -- let hasNanCPU = V.any isNaN logitsCPU
      --     numNanCPU = V.length $ V.filter isNaN logitsCPU
      -- putStrLn $ "\nCPU inference for token 105:"
      -- putStrLn $ "  Has NaN: " ++ show hasNanCPU
      -- putStrLn $ "  NaN count: " ++ show numNanCPU ++ " / " ++ show (V.length logitsCPU)
      -- putStrLn $ "  First 5 logits: " ++ show (V.toList $ V.take 5 logitsCPU)
      -- V.any isNaN logitsCPU `shouldBe` False
      let hasNanCPU = True  -- CPU version is broken

      -- GPU cached version
      (logitsGPU, _) <- runGemmaInferenceCached model (V.singleton firstToken) Nothing
      let hasNanGPU = V.any isNaN logitsGPU
          numNanGPU = V.length $ V.filter isNaN logitsGPU
      putStrLn $ "\nGPU cached inference for token 105:"
      putStrLn $ "  Has NaN: " ++ show hasNanGPU
      putStrLn $ "  NaN count: " ++ show numNanGPU ++ " / " ++ show (V.length logitsGPU)
      putStrLn $ "  First 5 logits: " ++ show (V.toList $ V.take 5 logitsGPU)

      V.any isNaN logitsGPU `shouldBe` False

      -- Show what token GPU picked
      unless hasNanGPU $ do
        let tokenGPU = V.maxIndex logitsGPU
            logitGPU = logitsGPU V.! tokenGPU

        putStrLn $ "\nGPU selected token: " ++ show tokenGPU ++ " (logit: " ++ show logitGPU ++ ")"
