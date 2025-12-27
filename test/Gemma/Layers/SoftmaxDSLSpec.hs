{-# LANGUAGE OverloadedStrings #-}

module Gemma.Layers.SoftmaxDSLSpec (spec) where

import Test.Hspec
import Graphics.WebGPU.Dawn.ContT
import Gemma.Layers.SoftmaxDSL
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)

-- | CPU reference implementation of softmax
--
-- Uses the same numerical stability trick as GPU:
-- softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
cpuSoftmax :: Vector Float -> Vector Float
cpuSoftmax input =
  let maxVal = V.maximum input
      exps = V.map (\x -> exp (x - maxVal)) input
      sumExps = V.sum exps
  in V.map (/ sumExps) exps

-- | CPU reference implementation of row-wise softmax
cpuSoftmaxRowwise :: Int -> Int -> Vector Float -> Vector Float
cpuSoftmaxRowwise numRows numCols input =
  V.concat [cpuSoftmax (V.slice (r * numCols) numCols input) | r <- [0..numRows-1]]

-- | Helper: Compare two vectors with tolerance
compareVectors :: Vector Float -> Vector Float -> (Float, Float)  -- (mean error, max error)
compareVectors expected actual =
  let diffs = V.zipWith (\e a -> abs (e - a)) expected actual
      meanError = V.sum diffs / fromIntegral (V.length diffs)
      maxError = V.maximum diffs
  in (meanError, maxError)

spec :: Spec
spec = describe "SoftmaxDSL" $ do

  describe "Vector Softmax (1D)" $ do

    it "computes softmax correctly (small vector, FP32)" $ do
      let size = 10
          input = V.fromList [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

      putStrLn $ "\n=== SoftmaxDSL Vector Test 1 (Small Vector) ==="
      putStrLn $ "Size: " ++ show size

      let cpuResult = cpuSoftmax input
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuResult)
      putStrLn $ "CPU output max: " ++ show (V.maximum cpuResult)

      gpuResult <- evalContT $ runSoftmaxDSL input
      putStrLn $ "GPU DSL output sum: " ++ show (V.sum gpuResult)
      putStrLn $ "GPU DSL output max: " ++ show (V.maximum gpuResult)

      let (meanErr, maxErr) = compareVectors cpuResult gpuResult
      putStrLn $ "Mean error: " ++ show meanErr
      putStrLn $ "Max error: " ++ show maxErr

      -- Check that output sums to ~1.0 (probability distribution)
      abs (V.sum gpuResult - 1.0) `shouldSatisfy` (< 1e-5)

      -- FP32 should have near-zero error
      maxErr `shouldSatisfy` (< 1e-6)

    it "computes softmax correctly (large vector, FP32)" $ do
      let size = 1024
          input = V.fromList [sin (fromIntegral x * 0.01) | x <- [0..size - 1]]

      putStrLn $ "\n=== SoftmaxDSL Vector Test 2 (Large Vector) ==="
      putStrLn $ "Size: " ++ show size

      let cpuResult = cpuSoftmax input
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuResult)

      gpuResult <- evalContT $ runSoftmaxDSL input
      putStrLn $ "GPU DSL output sum: " ++ show (V.sum gpuResult)

      let (meanErr, maxErr) = compareVectors cpuResult gpuResult
      putStrLn $ "Mean error: " ++ show meanErr
      putStrLn $ "Max error: " ++ show maxErr

      -- Check that output sums to ~1.0
      abs (V.sum gpuResult - 1.0) `shouldSatisfy` (< 1e-5)

      maxErr `shouldSatisfy` (< 1e-6)

    it "computes softmax correctly (FP16)" $ do
      let size = 256
          input = V.fromList [cos (fromIntegral x * 0.02) | x <- [0..size - 1]]

      putStrLn $ "\n=== SoftmaxDSL Vector Test 3 (FP16) ==="
      putStrLn $ "Size: " ++ show size

      let cpuResult = cpuSoftmax input
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuResult)

      gpuResult <- evalContT $ runSoftmaxDSLWithPrecision True False input
      putStrLn $ "GPU DSL (FP16) output sum: " ++ show (V.sum gpuResult)

      let (meanErr, maxErr) = compareVectors cpuResult gpuResult
      putStrLn $ "Mean error: " ++ show meanErr
      putStrLn $ "Max error: " ++ show maxErr

      -- Check that output sums to ~1.0 (allow more tolerance for FP16)
      abs (V.sum gpuResult - 1.0) `shouldSatisfy` (< 1e-3)

      -- FP16 has lower precision
      maxErr `shouldSatisfy` (< 1e-4)

    it "handles large values (numerical stability test)" $ do
      let size = 10
          -- Large values that would overflow without max subtraction
          input = V.fromList [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]

      putStrLn $ "\n=== SoftmaxDSL Vector Test 4 (Large Values) ==="
      putStrLn $ "Size: " ++ show size
      putStrLn $ "Input range: " ++ show (V.minimum input, V.maximum input)

      let cpuResult = cpuSoftmax input
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuResult)

      gpuResult <- evalContT $ runSoftmaxDSL input
      putStrLn $ "GPU DSL output sum: " ++ show (V.sum gpuResult)

      let (meanErr, maxErr) = compareVectors cpuResult gpuResult
      putStrLn $ "Mean error: " ++ show meanErr
      putStrLn $ "Max error: " ++ show maxErr

      -- Check no NaN or Inf
      V.all (not . isNaN) gpuResult `shouldBe` True
      V.all (not . isInfinite) gpuResult `shouldBe` True

      -- Check that output sums to ~1.0
      abs (V.sum gpuResult - 1.0) `shouldSatisfy` (< 1e-5)

      maxErr `shouldSatisfy` (< 1e-6)

    it "handles negative values correctly" $ do
      let size = 10
          input = V.fromList [-10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0]

      putStrLn $ "\n=== SoftmaxDSL Vector Test 5 (Negative Values) ==="
      putStrLn $ "Size: " ++ show size
      putStrLn $ "Input range: " ++ show (V.minimum input, V.maximum input)

      let cpuResult = cpuSoftmax input
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuResult)

      gpuResult <- evalContT $ runSoftmaxDSL input
      putStrLn $ "GPU DSL output sum: " ++ show (V.sum gpuResult)

      let (meanErr, maxErr) = compareVectors cpuResult gpuResult
      putStrLn $ "Mean error: " ++ show meanErr
      putStrLn $ "Max error: " ++ show maxErr

      -- Check that output sums to ~1.0
      abs (V.sum gpuResult - 1.0) `shouldSatisfy` (< 1e-5)

      maxErr `shouldSatisfy` (< 1e-6)

  describe "Matrix Row-wise Softmax (2D)" $ do

    it "computes row-wise softmax correctly (small matrix, FP32)" $ do
      let numRows = 4
          numCols = 8
          input = V.fromList [fromIntegral (r * numCols + c) * 0.1 | r <- [0..numRows-1], c <- [0..numCols-1]]

      putStrLn $ "\n=== SoftmaxDSL Rowwise Test 1 (Small Matrix) ==="
      putStrLn $ "Shape: " ++ show numRows ++ " x " ++ show numCols

      let cpuResult = cpuSoftmaxRowwise numRows numCols input
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuResult)

      gpuResult <- evalContT $ runSoftmaxRowwiseDSL numRows numCols input
      putStrLn $ "GPU DSL output sum: " ++ show (V.sum gpuResult)

      let (meanErr, maxErr) = compareVectors cpuResult gpuResult
      putStrLn $ "Mean error: " ++ show meanErr
      putStrLn $ "Max error: " ++ show maxErr

      -- Each row should sum to ~1.0
      let rowSums = [V.sum (V.slice (r * numCols) numCols gpuResult) | r <- [0..numRows-1]]
      putStrLn $ "Row sums: " ++ show rowSums
      all (\s -> abs (s - 1.0) < 1e-5) rowSums `shouldBe` True

      maxErr `shouldSatisfy` (< 1e-6)

    it "computes row-wise softmax correctly (large matrix, FP32)" $ do
      let numRows = 128
          numCols = 128
          input = V.fromList [sin (fromIntegral (r * numCols + c) * 0.001) | r <- [0..numRows-1], c <- [0..numCols-1]]

      putStrLn $ "\n=== SoftmaxDSL Rowwise Test 2 (Large Matrix) ==="
      putStrLn $ "Shape: " ++ show numRows ++ " x " ++ show numCols

      let cpuResult = cpuSoftmaxRowwise numRows numCols input
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuResult)

      gpuResult <- evalContT $ runSoftmaxRowwiseDSL numRows numCols input
      putStrLn $ "GPU DSL output sum: " ++ show (V.sum gpuResult)

      let (meanErr, maxErr) = compareVectors cpuResult gpuResult
      putStrLn $ "Mean error: " ++ show meanErr
      putStrLn $ "Max error: " ++ show maxErr

      -- Check first and last row sums
      let firstRowSum = V.sum (V.slice 0 numCols gpuResult)
          lastRowSum = V.sum (V.slice ((numRows-1) * numCols) numCols gpuResult)
      putStrLn $ "First row sum: " ++ show firstRowSum
      putStrLn $ "Last row sum: " ++ show lastRowSum

      abs (firstRowSum - 1.0) `shouldSatisfy` (< 1e-5)
      abs (lastRowSum - 1.0) `shouldSatisfy` (< 1e-5)

      maxErr `shouldSatisfy` (< 1e-5)

    it "computes row-wise softmax correctly (FP16)" $ do
      let numRows = 32
          numCols = 64
          input = V.fromList [cos (fromIntegral (r * numCols + c) * 0.01) | r <- [0..numRows-1], c <- [0..numCols-1]]

      putStrLn $ "\n=== SoftmaxDSL Rowwise Test 3 (FP16) ==="
      putStrLn $ "Shape: " ++ show numRows ++ " x " ++ show numCols

      let cpuResult = cpuSoftmaxRowwise numRows numCols input
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuResult)

      gpuResult <- evalContT $ runSoftmaxRowwiseDSLWithPrecision True False numRows numCols input
      putStrLn $ "GPU DSL (FP16) output sum: " ++ show (V.sum gpuResult)

      let (meanErr, maxErr) = compareVectors cpuResult gpuResult
      putStrLn $ "Mean error: " ++ show meanErr
      putStrLn $ "Max error: " ++ show maxErr

      -- Check row sums (allow more tolerance for FP16)
      let rowSums = [V.sum (V.slice (r * numCols) numCols gpuResult) | r <- [0, numRows `div` 2, numRows-1]]
      putStrLn $ "Sample row sums: " ++ show rowSums
      all (\s -> abs (s - 1.0) < 1e-2) rowSums `shouldBe` True

      -- FP16 has lower precision
      maxErr `shouldSatisfy` (< 1e-3)

    it "handles attention-like matrices (numerical stability)" $ do
      let numRows = 16
          numCols = 16
          -- Simulate attention scores with some large values
          input = V.fromList [if r == c then 100.0 else fromIntegral (abs (r - c)) * (-2.0)
                              | r <- [0..numRows-1], c <- [0..numCols-1]]

      putStrLn $ "\n=== SoftmaxDSL Rowwise Test 4 (Attention-like) ==="
      putStrLn $ "Shape: " ++ show numRows ++ " x " ++ show numCols
      putStrLn $ "Input range: " ++ show (V.minimum input, V.maximum input)

      let cpuResult = cpuSoftmaxRowwise numRows numCols input
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuResult)

      gpuResult <- evalContT $ runSoftmaxRowwiseDSL numRows numCols input
      putStrLn $ "GPU DSL output sum: " ++ show (V.sum gpuResult)

      let (meanErr, maxErr) = compareVectors cpuResult gpuResult
      putStrLn $ "Mean error: " ++ show meanErr
      putStrLn $ "Max error: " ++ show maxErr

      -- Check no NaN or Inf
      V.all (not . isNaN) gpuResult `shouldBe` True
      V.all (not . isInfinite) gpuResult `shouldBe` True

      -- Check row sums
      let rowSums = [V.sum (V.slice (r * numCols) numCols gpuResult) | r <- [0..numRows-1]]
      all (\s -> abs (s - 1.0) < 1e-5) rowSums `shouldBe` True

      maxErr `shouldSatisfy` (< 1e-5)
