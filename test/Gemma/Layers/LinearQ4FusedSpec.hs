{-# LANGUAGE OverloadedStrings #-}

module Gemma.Layers.LinearQ4FusedSpec (spec) where

import Test.Hspec
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import Data.Word (Word32)
import Control.Monad.IO.Class (liftIO)

import Gemma.Quantization.Q4 (quantizeQ4, dequantizeQ4)
import Gemma.Layers.LinearQ4Fused
import Graphics.WebGPU.Dawn.ContT
import Graphics.WebGPU.Dawn.Types (Shape(..))

-- | Test Q4 fused linear operations
spec :: Spec
spec = describe "LinearQ4Fused" $ do

  describe "runLinearQ4GPU (basic)" $ do
    it "performs Q4 matrix-vector multiplication correctly" $ do
      -- Test: 128x128 matrix @ 128-vector (realistic size, inSize multiple of 32)
      let outSize = 128
          inSize = 128
          -- Weight matrix: simple pattern that's robust to quantization
          -- All weights = 0.1 (uniform, easy to quantize)
          weights = V.replicate (outSize * inSize) 0.1 :: Vector Float
          -- Input: all ones
          input = V.replicate inSize 1.0 :: Vector Float

          -- Quantize to Q4
          (packed, scales) = quantizeQ4 weights outSize inSize

      -- Run Q4 GPU linear
      result <- evalContT $ runLinearQ4GPU packed scales input outSize inSize

      -- Expected: each row sums to 0.1 * 128 = 12.8
      let expected = V.replicate outSize 12.8 :: Vector Float
          maxDiff = V.maximum $ V.zipWith (\a b -> abs (a - b)) result expected
          relError = maxDiff / 12.8

      -- Q4 quantization should have <10% relative error for uniform weights
      relError `shouldSatisfy` (< 0.10)

  describe "runRMSNormLinearQ4GPU (fused)" $ do
    it "fuses RMSNorm and Q4 linear correctly" $ do
      -- Test fused RMSNorm + Q4 Linear
      let hiddenDim = 128  -- Must be multiple of 32
          outSize = 64

          -- Input and norm weights
          input = V.replicate hiddenDim 0.5 :: Vector Float
          normWeights = V.replicate hiddenDim 1.0 :: Vector Float

          -- Linear weights: uniform for easy testing
          linearWeights = V.replicate (outSize * hiddenDim) 0.1 :: Vector Float

          -- Quantize linear weights
          (packed, scales) = quantizeQ4 linearWeights outSize hiddenDim

      -- Create GPU context and tensors
      resultVec <- evalContT $ do
        ctx <- createContext
        inputT <- createTensorWithData ctx (Shape [hiddenDim]) input
        normT <- createTensorWithData ctx (Shape [hiddenDim]) normWeights

        -- Run fused operation
        result <- runRMSNormLinearQ4GPU ctx inputT normT packed scales hiddenDim outSize False

        -- Download result
        liftIO $ fromGPU ctx result outSize

      -- Expected: RMSNorm(input) with zeroCentered=False
      -- input = 0.5, rms = sqrt(0.25) = 0.5
      -- normalized = (0.5 / 0.5) * 1.0 = 1.0
      -- Then: 1.0 * 0.1 * 128 ≈ 12.8 per output
      let expected = V.replicate outSize 12.8 :: Vector Float
          maxDiff = V.maximum $ V.zipWith (\a b -> abs (a - b)) resultVec expected
          relError = maxDiff / 12.8

      relError `shouldSatisfy` (< 0.15)  -- Q4 + fusion error

  describe "runRMSNormGateUpQ4GPU (fused)" $ do
    it "fuses RMSNorm with Q4 Gate and Q4 Up projections" $ do
      -- Test triple fusion: RMSNorm + Q4 Gate + Q4 Up
      let hiddenDim = 128  -- Must be multiple of 32
          ffnDim = 256

          -- Input and norm weights
          input = V.replicate hiddenDim 0.5 :: Vector Float
          normWeights = V.replicate hiddenDim 1.0 :: Vector Float

          -- Gate and Up weights: uniform for easy testing
          gateWeights = V.replicate (ffnDim * hiddenDim) 0.1 :: Vector Float
          upWeights = V.replicate (ffnDim * hiddenDim) 0.2 :: Vector Float

          -- Quantize both weight matrices
          (gatePacked, gateScales) = quantizeQ4 gateWeights ffnDim hiddenDim
          (upPacked, upScales) = quantizeQ4 upWeights ffnDim hiddenDim

      -- Create GPU context and run fused operation
      (gateVec, upVec) <- evalContT $ do
        ctx <- createContext
        inputT <- createTensorWithData ctx (Shape [hiddenDim]) input
        normT <- createTensorWithData ctx (Shape [hiddenDim]) normWeights

        -- Run fused operation
        (gateT, upT) <- runRMSNormGateUpQ4GPU ctx inputT normT
                                               gatePacked gateScales
                                               upPacked upScales
                                               hiddenDim ffnDim False

        -- Download both results
        gateResult <- liftIO $ fromGPU ctx gateT ffnDim
        upResult <- liftIO $ fromGPU ctx upT ffnDim
        return (gateResult, upResult)

      -- Expected values:
      -- RMSNorm(input) with zeroCentered=False
      -- input = 0.5, rms = sqrt(0.25) = 0.5
      -- normalized = (0.5 / 0.5) * 1.0 = 1.0
      -- Gate: 1.0 * 0.1 * 128 ≈ 12.8 per output
      -- Up: 1.0 * 0.2 * 128 ≈ 25.6 per output
      let gateExpected = V.replicate ffnDim 12.8 :: Vector Float
          upExpected = V.replicate ffnDim 25.6 :: Vector Float

          gateMaxDiff = V.maximum $ V.zipWith (\a b -> abs (a - b)) gateVec gateExpected
          upMaxDiff = V.maximum $ V.zipWith (\a b -> abs (a - b)) upVec upExpected

          gateRelError = gateMaxDiff / 12.8
          upRelError = upMaxDiff / 25.6

      -- Q4 + fusion error tolerance
      gateRelError `shouldSatisfy` (< 0.15)
      upRelError `shouldSatisfy` (< 0.15)

  describe "FP16 baseline protection" $ do
    it "does not affect existing FP16 linear operations" $ pending
    -- Verify FP16 tests still pass
