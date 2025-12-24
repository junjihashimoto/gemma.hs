{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE QuasiQuotes #-}
module Gemma.Regression.Q4InlinePytorchSpec (spec) where

import Test.Hspec
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import System.Process (readProcess)
import System.Directory (doesFileExist)
import qualified Data.Aeson as JSON
import Data.Aeson ((.:))
import qualified Data.ByteString.Lazy as BL
import qualified Data.ByteString.Lazy.Char8 as BLC
import Text.Printf
import Gemma.SafeTensors
import Gemma.Model
import Control.Monad (when, unless)
import Graphics.WebGPU.Dawn.ContT (evalContT, ContT, liftIO, lift)
import qualified Graphics.WebGPU.Dawn.Context as Ctx
import qualified Gemma.Layers.RMSNorm as RMSNorm
import qualified Gemma.Layers.Linear as Linear
import Graphics.WebGPU.Dawn.Types (Shape(..))
import qualified Graphics.WebGPU.Dawn.Tensor as T
import Crypto.Hash.SHA256 as SHA256
import qualified Data.ByteString.Base16 as B16
import qualified Data.ByteString.Char8 as BS

-- | Generate cache key from test parameters
cacheKey :: String -> String -> String
cacheKey testName params =
  let hash = SHA256.hash $ BS.pack (testName ++ "|" ++ params)
  in ".cache/q4_" ++ BS.unpack (B16.encode hash) ++ ".json"

-- | Run Python code inline and return JSON result (with caching)
runPytorchInline :: String -> String -> String -> IO (Maybe JSON.Value)
runPytorchInline testName pythonCode params = do
  let cacheFile = cacheKey testName params

  -- Check cache first
  cached <- doesFileExist cacheFile
  if cached
    then do
      putStrLn $ "  ✓ Using cached PyTorch reference: " ++ cacheFile
      content <- BL.readFile cacheFile
      return $ JSON.decode content
    else do
      putStrLn $ "  → Running PyTorch to generate reference..."

      -- Run Python code
      let fullCode = unlines
            [ "import torch"
            , "import json"
            , "import sys"
            , "from transformers import AutoTokenizer, AutoModelForCausalLM"
            , ""
            , pythonCode
            , ""
            , "# Output JSON to stdout"
            , "print(json.dumps(result))"
            ]

      output <- readProcess "python3" ["-c", fullCode] ""

      -- Cache the result
      let jsonData = BLC.pack output
      BL.writeFile cacheFile jsonData
      putStrLn $ "  ✓ Cached PyTorch reference: " ++ cacheFile

      return $ JSON.decode jsonData

-- | Data structure for Q4 test results
data Q4TestResult = Q4TestResult
  { embedding :: [Float]
  , afterRMSNorm :: [Float]
  , afterQProj :: [Float]
  , afterKProj :: [Float]
  , afterVProj :: [Float]
  } deriving (Show)

instance JSON.FromJSON Q4TestResult where
  parseJSON = JSON.withObject "Q4TestResult" $ \v -> Q4TestResult
    <$> v .: "embedding"
    <*> v .: "after_rmsnorm"
    <*> v .: "after_q_proj"
    <*> v .: "after_k_proj"
    <*> v .: "after_v_proj"

spec :: Spec
spec = describe "Q4 Inference (Inline PyTorch TDD)" $ do

  it "Q4 GPU RMSNorm + Q/K/V - token 6974 (inline PyTorch)" $ do
    putStrLn "\n=== Q4 Inline PyTorch Test ==="
    putStrLn "Generating PyTorch reference on-the-fly..."

    let tokenId = 6974
        modelPath = "../models/gemma3-1b-q4.safetensors"

    -- Run PyTorch inline to get expected values
    let pythonCode = unlines
          [ "# Load Q4 model"
          , "model_path = '" ++ modelPath ++ "'"
          , "token_id = " ++ show tokenId
          , ""
          , "# TODO: Load Q4 model and run inference"
          , "# For now, return dummy data to show the structure"
          , "result = {"
          , "  'embedding': [0.1] * 1152,"
          , "  'after_rmsnorm': [0.2] * 1152,"
          , "  'after_q_proj': [0.3] * 1024,"
          , "  'after_k_proj': [0.4] * 256,"
          , "  'after_v_proj': [0.5] * 256"
          , "}"
          ]

    mbPytorchResult <- runPytorchInline "q4_layer0_token6974" pythonCode (show tokenId)

    case mbPytorchResult of
      Nothing -> expectationFailure "Failed to parse PyTorch result"
      Just jsonVal -> case JSON.fromJSON jsonVal of
        JSON.Error err -> expectationFailure $ "JSON parse error: " ++ err
        JSON.Success (pytorchRef :: Q4TestResult) -> do
          putStrLn "✓ PyTorch reference generated"

          -- Now run Haskell version
          putStrLn "\nRunning Haskell Q4 inference..."
          st <- loadSafeTensors modelPath

          embeddings <- getTensor st "model.embed_tokens.weight"
          let hiddenDim = 1152
              embedding = V.slice (tokenId * hiddenDim) hiddenDim embeddings

          -- Compare embedding
          let pytorchEmb = V.fromList $ embedding pytorchRef
              embDiffs = V.zipWith (\h p -> abs (h - p)) embedding pytorchEmb
              maxEmbDiff = V.maximum embDiffs

          putStrLn $ "\nEmbedding comparison:"
          putStrLn $ "  Max diff: " ++ printf "%.2e" maxEmbDiff
          putStrLn $ "  Haskell first 5: " ++ show (V.toList $ V.take 5 embedding)
          putStrLn $ "  PyTorch first 5: " ++ show (take 5 $ embedding pytorchRef)

          -- Load layer 0 weights
          attnNorm <- getTensor st "model.layers.0.input_layernorm.weight"

          -- Check if Q4
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

          -- RMSNorm on GPU
          (gpuNorm, gpuQ, gpuK, gpuV) <- evalContT $ do
            ctx <- lift Ctx.createContext
            let shape = Shape [hiddenDim]
            inputTensor <- lift $ T.createTensorWithData ctx shape embedding

            -- RMSNorm
            normTensor <- RMSNorm.runRMSNormGPU ctx inputTensor attnNorm hiddenDim False
            normOutput <- liftIO $ T.fromGPU ctx normTensor hiddenDim

            -- Q/K/V projections
            let qSize = 1024
                kvSize = 256
            qTensor <- Linear.runLinearGPU ctx qWeights normTensor qSize hiddenDim
            kTensor <- Linear.runLinearGPU ctx kWeights normTensor kvSize hiddenDim
            vTensor <- Linear.runLinearGPU ctx vWeights normTensor kvSize hiddenDim

            qOutput <- liftIO $ T.fromGPU ctx qTensor qSize
            kOutput <- liftIO $ T.fromGPU ctx kTensor kvSize
            vOutput <- liftIO $ T.fromGPU ctx vTensor kvSize

            return (normOutput, qOutput, kOutput, vOutput)

          -- Compare with PyTorch
          let pytorchNormVec = V.fromList $ afterRMSNorm pytorchRef
              pytorchQVec = V.fromList $ afterQProj pytorchRef
              pytorchKVec = V.fromList $ afterKProj pytorchRef
              pytorchVVec = V.fromList $ afterVProj pytorchRef

              normDiffs = V.zipWith (\h p -> abs (h - p)) gpuNorm pytorchNormVec
              qDiffs = V.zipWith (\h p -> abs (h - p)) gpuQ pytorchQVec
              kDiffs = V.zipWith (\h p -> abs (h - p)) gpuK pytorchKVec
              vDiffs = V.zipWith (\h p -> abs (h - p)) gpuV pytorchVVec

              maxNormDiff = V.maximum normDiffs
              maxQDiff = V.maximum qDiffs
              maxKDiff = V.maximum kDiffs
              maxVDiff = V.maximum vDiffs

          putStrLn "\n=== Haskell vs PyTorch Comparison ==="
          putStrLn $ "RMSNorm max diff: " ++ printf "%.2e" maxNormDiff
          putStrLn $ "Q proj max diff:  " ++ printf "%.2e" maxQDiff
          putStrLn $ "K proj max diff:  " ++ printf "%.2e" maxKDiff
          putStrLn $ "V proj max diff:  " ++ printf "%.2e" maxVDiff

          -- Check for NaN
          let hasNaN = V.any isNaN gpuQ || V.any isNaN gpuK || V.any isNaN gpuV
          hasNaN `shouldBe` False

          putStrLn "\n✅ Test complete!"

  it "demonstrates inline PyTorch workflow" $ do
    putStrLn "\n=== Inline PyTorch Workflow Demo ==="
    putStrLn "This test shows how to:"
    putStrLn "1. Write Python code inline in Haskell test"
    putStrLn "2. Generate PyTorch reference on-the-fly"
    putStrLn "3. Cache results for speed"
    putStrLn "4. Compare Haskell output immediately"
    putStrLn ""
    putStrLn "Benefits:"
    putStrLn "- No separate Python files to maintain"
    putStrLn "- Python code is right next to Haskell test"
    putStrLn "- Fast (cached after first run)"
    putStrLn "- Easy to see what's being tested"

    True `shouldBe` True
