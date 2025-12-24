{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Gemma.Regression.FP16Spec (spec) where

import Test.Hspec
import qualified Data.Vector.Storable as V
import qualified Data.Text as T
import qualified Data.Aeson as JSON
import qualified Data.ByteString.Lazy as BL
import Data.Aeson ((.:))
import qualified Data.Aeson.Types as JSON
import qualified Data.Aeson.KeyMap as KM
import qualified Data.Aeson.Key as K
import Gemma.SafeTensors
import Text.Printf

-- | Test case from PyTorch reference
data TestCase = TestCase
  { tcDescription :: String
  , tcShape :: [Int]
  , tcValues :: [Float]
  } deriving (Show)

-- | Reference output from PyTorch
data PyTorchReference = PyTorchReference
  { prModelPath :: String
  , prDType :: String
  , prTestCases :: [(String, TestCase)]
  } deriving (Show)

instance JSON.FromJSON TestCase where
  parseJSON = JSON.withObject "TestCase" $ \v -> TestCase
    <$> v .: "description"
    <*> v .: "shape"
    <*> v .: "values"

instance JSON.FromJSON PyTorchReference where
  parseJSON = JSON.withObject "PyTorchReference" $ \v -> do
    modelPath <- v .: "model_path"
    dtype <- v .: "dtype"
    testCasesObj <- v .: "test_cases" :: JSON.Parser JSON.Object
    -- Parse each test case from the KeyMap
    testCasesList <- mapM (\(k, val) -> do
        tc <- JSON.parseJSON val
        return (T.unpack $ K.toText k, tc)) (KM.toList testCasesObj)
    return $ PyTorchReference modelPath dtype testCasesList

-- | Compare two float values with tolerance
floatEqual :: Float -> Float -> Float -> Bool
floatEqual tolerance a b = abs (a - b) <= tolerance

-- | Compare two vectors with tolerance
compareVectors :: Float -> V.Vector Float -> [Float] -> (Bool, String)
compareVectors tolerance haskellVec pytorchList =
  let haskellList = V.toList haskellVec
      len = length pytorchList
  in if length haskellList /= len
     then (False, "Length mismatch: " ++ show (length haskellList) ++ " vs " ++ show len)
     else
       let differences = zipWith (\h p -> abs (h - p)) haskellList pytorchList
           maxDiff = maximum differences
           avgDiff = sum differences / fromIntegral len
           numMismatches = length $ filter (> tolerance) differences
           summary = printf "Max diff: %.10f, Avg diff: %.10f, Mismatches: %d" maxDiff avgDiff numMismatches
       in if numMismatches == 0
          then (True, "PASS - " ++ summary)
          else
            let firstMismatch = head $ filter (\(_, _, diff) -> diff > tolerance)
                                       $ zipWith3 (\i h p -> (i, (h, p), abs (h - p)))
                                                  ([0..] :: [Int]) haskellList pytorchList
                (idx, (h, p), diff) = firstMismatch
                detail = printf "\nFirst mismatch at index %d: Haskell=%.10f, PyTorch=%.10f, Diff=%.10f"
                               idx h p diff
            in (False, "FAIL - " ++ summary ++ detail)

spec :: Spec
spec = describe "FP16 PyTorch Comparison" $ do

  it "loads PyTorch reference and SafeTensors model" $ do
    -- Load PyTorch reference
    referenceJson <- BL.readFile "scripts/pytorch_fp16_reference.json"
    case JSON.eitherDecode referenceJson of
      Left err -> expectationFailure $ "Failed to parse reference JSON: " ++ err
      Right (ref :: PyTorchReference) -> do
        -- Load Haskell SafeTensors
        st <- loadSafeTensors "../models/gemma3-1b-fp16.safetensors"

        -- Verify it's FP16
        prDType ref `shouldBe` "FP16"

        -- Define tolerance for FP16 (lower precision than FP32)
        -- FP16 has ~3 decimal digits of precision, so tolerance should be around 1e-3
        let tolerance = 1e-3 :: Float

        -- Test case 1: Layer 0 Q projection (first 20)
        -- getTensor automatically converts FP16->FP32
        case lookup "layer0_q_proj_first20" (prTestCases ref) of
          Nothing -> expectationFailure "Missing test case: layer0_q_proj_first20"
          Just testCase -> do
            haskellTensor <- getTensor st "model.layers.0.self_attn.q_proj.weight"
            let pytorchValues = tcValues testCase
                numValues = length pytorchValues
                haskellValues = V.take numValues haskellTensor
                (pass, msg) = compareVectors tolerance haskellValues pytorchValues
            if not pass then expectationFailure msg else pass `shouldBe` True

        -- Test case 2: Embedding tokens (first 20)
        case lookup "embed_tokens_first20" (prTestCases ref) of
          Nothing -> expectationFailure "Missing test case: embed_tokens_first20"
          Just testCase -> do
            haskellTensor <- getTensor st "model.embed_tokens.weight"
            let pytorchValues = tcValues testCase
                numValues = length pytorchValues
                haskellValues = V.take numValues haskellTensor
                (pass, msg) = compareVectors tolerance haskellValues pytorchValues
            if not pass then expectationFailure msg else pass `shouldBe` True

        -- Test case 3: Layer 0 attention norm (all)
        case lookup "layer0_attn_norm_all" (prTestCases ref) of
          Nothing -> expectationFailure "Missing test case: layer0_attn_norm_all"
          Just testCase -> do
            haskellTensor <- getTensor st "model.layers.0.input_layernorm.weight"
            let pytorchValues = tcValues testCase
                haskellValues = haskellTensor  -- All values
                (pass, msg) = compareVectors tolerance haskellValues pytorchValues
            if not pass then expectationFailure msg else pass `shouldBe` True
