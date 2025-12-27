{-# LANGUAGE OverloadedStrings #-}

module Gemma.Layers.ElementwiseDSLSpec (spec) where

import Test.Hspec
import Graphics.WebGPU.Dawn.ContT
import Gemma.Layers.ElementwiseDSL
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)

-- | CPU reference implementation of element-wise multiply
cpuElementwiseMultiply :: Vector Float -> Vector Float -> Vector Float
cpuElementwiseMultiply = V.zipWith (*)

-- | CPU reference implementation of element-wise add
cpuElementwiseAdd :: Vector Float -> Vector Float -> Vector Float
cpuElementwiseAdd = V.zipWith (+)

-- | Helper: Compare two vectors with tolerance
compareVectors :: Vector Float -> Vector Float -> (Float, Float)  -- (mean error, max error)
compareVectors expected actual =
  let diffs = V.zipWith (\e a -> abs (e - a)) expected actual
      meanError = V.sum diffs / fromIntegral (V.length diffs)
      maxError = V.maximum diffs
  in (meanError, maxError)

spec :: Spec
spec = describe "ElementwiseDSL" $ do

  describe "Element-wise Multiply" $ do

    it "computes element-wise multiply correctly (small vector, FP32)" $ do
      let size = 256
          a = V.fromList [x * 0.1 | x <- [1..fromIntegral size]]
          b = V.fromList [x * 0.2 | x <- [1..fromIntegral size]]

      putStrLn $ "\n=== ElementwiseDSL Multiply Test 1 (Small Vector) ==="
      putStrLn $ "Size: " ++ show size

      let cpuResult = cpuElementwiseMultiply a b
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuResult)

      gpuResult <- evalContT $ runElementwiseMultiplyDSL a b
      putStrLn $ "GPU DSL output sum: " ++ show (V.sum gpuResult)

      let (meanErr, maxErr) = compareVectors cpuResult gpuResult
      putStrLn $ "Mean error: " ++ show meanErr
      putStrLn $ "Max error: " ++ show maxErr

      -- FP32 should have near-zero error (just rounding)
      maxErr `shouldSatisfy` (< 1e-6)

    it "computes element-wise multiply correctly (large vector, FP32)" $ do
      let size = 4096
          a = V.fromList [sin (x * 0.01) | x <- [0..fromIntegral size - 1]]
          b = V.fromList [cos (x * 0.02) | x <- [0..fromIntegral size - 1]]

      putStrLn $ "\n=== ElementwiseDSL Multiply Test 2 (Large Vector) ==="
      putStrLn $ "Size: " ++ show size

      let cpuResult = cpuElementwiseMultiply a b
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuResult)

      gpuResult <- evalContT $ runElementwiseMultiplyDSL a b
      putStrLn $ "GPU DSL output sum: " ++ show (V.sum gpuResult)

      let (meanErr, maxErr) = compareVectors cpuResult gpuResult
      putStrLn $ "Mean error: " ++ show meanErr
      putStrLn $ "Max error: " ++ show maxErr

      maxErr `shouldSatisfy` (< 1e-6)

    it "computes element-wise multiply correctly (FP16)" $ do
      let size = 1024
          a = V.fromList [x * 0.05 - 10.0 | x <- [0..fromIntegral size - 1]]
          b = V.fromList [x * 0.03 + 5.0 | x <- [0..fromIntegral size - 1]]

      putStrLn $ "\n=== ElementwiseDSL Multiply Test 3 (FP16) ==="
      putStrLn $ "Size: " ++ show size

      let cpuResult = cpuElementwiseMultiply a b
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuResult)

      gpuResult <- evalContT $ runElementwiseMultiplyDSLWithPrecision True False a b
      putStrLn $ "GPU DSL (FP16) output sum: " ++ show (V.sum gpuResult)

      let (meanErr, maxErr) = compareVectors cpuResult gpuResult
      putStrLn $ "Mean error: " ++ show meanErr
      putStrLn $ "Max error: " ++ show maxErr

      -- FP16 has lower precision, especially for larger values
      -- With values up to ~1400, FP16 precision is ~1.4
      maxErr `shouldSatisfy` (< 2.0)

    it "computes element-wise multiply correctly (Vec4 SIMD, FP32)" $ do
      let size = 1024  -- Must be multiple of 4
          a = V.fromList [x * 0.1 | x <- [1..fromIntegral size]]
          b = V.fromList [x * 0.2 | x <- [1..fromIntegral size]]

      putStrLn $ "\n=== ElementwiseDSL Multiply Test 4 (Vec4 SIMD) ==="
      putStrLn $ "Size: " ++ show size

      let cpuResult = cpuElementwiseMultiply a b
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuResult)

      gpuResult <- evalContT $ runElementwiseMultiplyDSLWithPrecision False True a b
      putStrLn $ "GPU DSL (Vec4) output sum: " ++ show (V.sum gpuResult)

      let (meanErr, maxErr) = compareVectors cpuResult gpuResult
      putStrLn $ "Mean error: " ++ show meanErr
      putStrLn $ "Max error: " ++ show maxErr

      maxErr `shouldSatisfy` (< 1e-6)

  describe "Element-wise Add" $ do

    it "computes element-wise add correctly (small vector, FP32)" $ do
      let size = 256
          a = V.fromList [x * 0.1 | x <- [1..fromIntegral size]]
          b = V.fromList [x * 0.2 | x <- [1..fromIntegral size]]

      putStrLn $ "\n=== ElementwiseDSL Add Test 1 (Small Vector) ==="
      putStrLn $ "Size: " ++ show size

      let cpuResult = cpuElementwiseAdd a b
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuResult)

      gpuResult <- evalContT $ runElementwiseAddDSL a b
      putStrLn $ "GPU DSL output sum: " ++ show (V.sum gpuResult)

      let (meanErr, maxErr) = compareVectors cpuResult gpuResult
      putStrLn $ "Mean error: " ++ show meanErr
      putStrLn $ "Max error: " ++ show maxErr

      maxErr `shouldSatisfy` (< 1e-6)

    it "computes element-wise add correctly (large vector, FP32)" $ do
      let size = 4096
          a = V.fromList [sin (x * 0.01) | x <- [0..fromIntegral size - 1]]
          b = V.fromList [cos (x * 0.02) | x <- [0..fromIntegral size - 1]]

      putStrLn $ "\n=== ElementwiseDSL Add Test 2 (Large Vector) ==="
      putStrLn $ "Size: " ++ show size

      let cpuResult = cpuElementwiseAdd a b
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuResult)

      gpuResult <- evalContT $ runElementwiseAddDSL a b
      putStrLn $ "GPU DSL output sum: " ++ show (V.sum gpuResult)

      let (meanErr, maxErr) = compareVectors cpuResult gpuResult
      putStrLn $ "Mean error: " ++ show meanErr
      putStrLn $ "Max error: " ++ show maxErr

      maxErr `shouldSatisfy` (< 1e-6)

    it "computes element-wise add correctly (FP16)" $ do
      let size = 1024
          a = V.fromList [x * 0.05 - 10.0 | x <- [0..fromIntegral size - 1]]
          b = V.fromList [x * 0.03 + 5.0 | x <- [0..fromIntegral size - 1]]

      putStrLn $ "\n=== ElementwiseDSL Add Test 3 (FP16) ==="
      putStrLn $ "Size: " ++ show size

      let cpuResult = cpuElementwiseAdd a b
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuResult)

      gpuResult <- evalContT $ runElementwiseAddDSLWithPrecision True False a b
      putStrLn $ "GPU DSL (FP16) output sum: " ++ show (V.sum gpuResult)

      let (meanErr, maxErr) = compareVectors cpuResult gpuResult
      putStrLn $ "Mean error: " ++ show meanErr
      putStrLn $ "Max error: " ++ show maxErr

      -- FP16 has lower precision
      -- For values up to ~100, FP16 precision is ~0.05
      maxErr `shouldSatisfy` (< 0.1)

    it "computes element-wise add correctly (Vec4 SIMD, FP32)" $ do
      let size = 1024  -- Must be multiple of 4
          a = V.fromList [x * 0.1 | x <- [1..fromIntegral size]]
          b = V.fromList [x * 0.2 | x <- [1..fromIntegral size]]

      putStrLn $ "\n=== ElementwiseDSL Add Test 4 (Vec4 SIMD) ==="
      putStrLn $ "Size: " ++ show size

      let cpuResult = cpuElementwiseAdd a b
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuResult)

      gpuResult <- evalContT $ runElementwiseAddDSLWithPrecision False True a b
      putStrLn $ "GPU DSL (Vec4) output sum: " ++ show (V.sum gpuResult)

      let (meanErr, maxErr) = compareVectors cpuResult gpuResult
      putStrLn $ "Mean error: " ++ show meanErr
      putStrLn $ "Max error: " ++ show maxErr

      maxErr `shouldSatisfy` (< 1e-6)

    it "handles residual connections correctly (mixed signs)" $ do
      let size = 512
          a = V.fromList [x * 0.1 - 25.0 | x <- [0..fromIntegral size - 1]]
          b = V.fromList [x * 0.2 + 10.0 | x <- [0..fromIntegral size - 1]]

      putStrLn $ "\n=== ElementwiseDSL Add Test 5 (Residual Connection Pattern) ==="
      putStrLn $ "Size: " ++ show size
      putStrLn $ "A range: " ++ show (V.minimum a, V.maximum a)
      putStrLn $ "B range: " ++ show (V.minimum b, V.maximum b)

      let cpuResult = cpuElementwiseAdd a b
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuResult)

      gpuResult <- evalContT $ runElementwiseAddDSL a b
      putStrLn $ "GPU DSL output sum: " ++ show (V.sum gpuResult)

      let (meanErr, maxErr) = compareVectors cpuResult gpuResult
      putStrLn $ "Mean error: " ++ show meanErr
      putStrLn $ "Max error: " ++ show maxErr

      maxErr `shouldSatisfy` (< 1e-6)
