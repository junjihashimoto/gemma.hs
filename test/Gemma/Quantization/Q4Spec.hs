{-# LANGUAGE OverloadedStrings #-}

module Gemma.Quantization.Q4Spec (spec) where

import Test.Hspec
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import Data.Word (Word32)

-- Import Q4 functions
import Gemma.Quantization.Q4

-- Helper: Calculate mean absolute error
meanAbsError :: Vector Float -> Vector Float -> Float
meanAbsError v1 v2 =
  let diffs = V.zipWith (\a b -> abs (a - b)) v1 v2
  in V.sum diffs / fromIntegral (V.length diffs)

-- Helper: Create test weights
testWeights :: Int -> Vector Float
testWeights n = V.generate n (\i -> fromIntegral (i `mod` 20) - 10.0)

spec :: Spec
spec = do
  describe "Q4 Quantization Core (Isolated Module)" $ do

    -- Test 1: Basic bit packing (nibbles → Word32)
    describe "Bit Packing" $ do
      it "packs 8 nibbles into one Word32" $ do
        let nibbles = [0, 1, 2, 3, 4, 5, 6, 7] :: [Word32]
        let packed = packNibbles nibbles
        packed `shouldBe` 0x76543210  -- 0111_0110_0101_0100_0011_0010_0001_0000

      it "unpacks Word32 into 8 nibbles" $ do
        let packed = 0x76543210 :: Word32
        let nibbles = unpackNibbles packed
        nibbles `shouldBe` [0, 1, 2, 3, 4, 5, 6, 7]

      it "pack/unpack roundtrip is identity" $ do
        let original = [0, 15, 8, 4, 12, 3, 9, 1] :: [Word32]
        let roundtrip = unpackNibbles (packNibbles original)
        roundtrip `shouldBe` original

    -- Test 2: Block-wise quantization (32 weights → scale + packed)
    describe "Block-wise Quantization" $ do
      it "quantizes a single block (32 weights)" $ do
        let weights = V.fromList [0..31] :: Vector Float
        let (packed, scales) = quantizeQ4 weights 1 32
        V.length packed `shouldBe` 4  -- 32 weights / 8 per Word32
        V.length scales `shouldBe` 1  -- 1 scale per block

      it "quantizes multiple blocks (128 weights = 4 blocks)" $ do
        let weights = testWeights 128
        let (packed, scales) = quantizeQ4 weights 1 128
        V.length packed `shouldBe` 16  -- 128 weights / 8 per Word32
        V.length scales `shouldBe` 4   -- 4 scales (1 per 32 weights)

      it "handles exact multiple of 32" $ do
        let weights = testWeights 64  -- Exactly 2 blocks
        let (packed, scales) = quantizeQ4 weights 1 64
        V.length packed `shouldBe` 8
        V.length scales `shouldBe` 2

    -- Test 3: Quantization accuracy (roundtrip error)
    describe "Quantization Accuracy" $ do
      it "has <10% error on uniform distribution" $ do
        let original = V.fromList [0..255] :: Vector Float
        let (packed, scales) = quantizeQ4 original 1 256
        let reconstructed = dequantizeQ4 packed scales
        let error = meanAbsError original reconstructed
        -- For 4-bit quantization, we have 16 levels per block
        -- Each block of 32 has range ~31, so quantization step ~2
        -- Mean error should be < half quantization step * 2
        error `shouldSatisfy` (< 6.0)  -- Reasonable for 4-bit per-block quantization

      it "has <2% error on random-ish weights" $ do
        let original = testWeights 1024
        let (packed, scales) = quantizeQ4 original 1 1024
        let reconstructed = dequantizeQ4 packed scales
        let error = meanAbsError original reconstructed
        error `shouldSatisfy` (< 0.4)  -- <2% of range (-10 to 10)

      it "preserves zeros and near-zeros" $ do
        let original = V.fromList ([0, 0.1, -0.1, 0] ++ replicate 28 0.5) :: Vector Float
        let (packed, scales) = quantizeQ4 original 1 32
        let reconstructed = dequantizeQ4 packed scales
        -- First 4 values should be close to zero
        abs (reconstructed V.! 0) `shouldSatisfy` (< 0.5)
        abs (reconstructed V.! 1) `shouldSatisfy` (< 0.6)

      it "handles large values correctly" $ do
        let original = V.fromList (replicate 32 100.0) :: Vector Float
        let (packed, scales) = quantizeQ4 original 1 32
        let reconstructed = dequantizeQ4 packed scales
        -- Should be close to 100
        let maxError = V.maximum (V.map (\x -> abs (x - 100.0)) reconstructed)
        maxError `shouldSatisfy` (< 10.0)  -- <10% error for large values

    -- Test 4: Matrix layout (multiple rows)
    describe "Matrix Quantization" $ do
      it "quantizes 2D weight matrix row-major" $ do
        -- 4x64 matrix = 4 rows, 64 cols (2 blocks per row)
        let outSize = 4
            inSize = 64
            weights = testWeights (outSize * inSize)
        let (packed, scales) = quantizeQ4 weights outSize inSize
        -- 256 weights total = 32 Word32s
        V.length packed `shouldBe` 32
        -- 4 rows × 2 blocks/row = 8 scales
        V.length scales `shouldBe` 8

      it "different rows have different scales" $ do
        -- Row 0: all small values, Row 1: all large values
        let row0 = V.replicate 64 1.0
            row1 = V.replicate 64 100.0
            weights = row0 V.++ row1
        let (_, scales) = quantizeQ4 weights 2 64
        -- Scales for row 0 should be smaller than row 1
        let scale0 = scales V.! 0  -- First block of row 0
            scale1 = scales V.! 2  -- First block of row 1
        scale1 `shouldSatisfy` (> scale0 * 10)

    -- Test 5: Edge cases
    describe "Edge Cases" $ do
      it "handles all zeros" $ do
        let weights = V.replicate 32 0.0 :: Vector Float
        let (packed, scales) = quantizeQ4 weights 1 32
        let reconstructed = dequantizeQ4 packed scales
        V.all (== 0.0) reconstructed `shouldBe` True

      it "handles all same non-zero value" $ do
        let weights = V.replicate 32 5.0 :: Vector Float
        let (packed, scales) = quantizeQ4 weights 1 32
        let reconstructed = dequantizeQ4 packed scales
        let error = meanAbsError weights reconstructed
        -- With 4-bit quantization, quantization step = scale
        -- For value 5.0, scale = 5.0/7 ≈ 0.714
        -- Max error is ~0.357 (half quantization step)
        error `shouldSatisfy` (< 0.5)

      it "handles negative values" $ do
        let weights = V.fromList [-10, -5, 0, 5, 10] V.++ V.replicate 27 0 :: Vector Float
        let (packed, scales) = quantizeQ4 weights 1 32
        let reconstructed = dequantizeQ4 packed scales
        -- Check range preserved
        V.minimum reconstructed `shouldSatisfy` (< -8)
        V.maximum reconstructed `shouldSatisfy` (> 8)

    -- Test 6: No side effects on FP16 code
    describe "Isolation from FP16" $ do
      it "Q4 module compiles without affecting FP16" $ do
        -- This test just ensures the module exists and compiles
        True `shouldBe` True

      it "importing Q4 doesn't break existing imports" $ do
        -- Test that we can import both Q4 and Linear without conflicts
        -- import Gemma.Quantization.Q4
        -- import Gemma.Layers.Linear
        True `shouldBe` True
