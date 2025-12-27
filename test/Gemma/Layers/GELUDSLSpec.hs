{-# LANGUAGE OverloadedStrings #-}

module Gemma.Layers.GELUDSLSpec (spec) where

import Test.Hspec
import Graphics.WebGPU.Dawn.ContT
import Gemma.Layers.GELUDSL
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)

-- | CPU reference implementation of GELU
--
-- Uses the same approximation as the GPU version:
-- GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
cpuGELU :: Vector Float -> Vector Float
cpuGELU input = V.map gelu input
  where
    sqrt2OverPi = 0.7978845608  -- √(2/π)
    coeff = 0.044715

    gelu :: Float -> Float
    gelu x =
      let xCubed = x * x * x
          inner = sqrt2OverPi * (x + coeff * xCubed)
          -- Clamp to prevent tanh overflow (same as GPU)
          clamped = max (-10.0) (min 10.0 inner)
          tanhVal = tanh clamped
      in 0.5 * x * (1.0 + tanhVal)

-- | Helper: Compare two vectors with tolerance
compareVectors :: Vector Float -> Vector Float -> (Float, Float)  -- (mean error, max error)
compareVectors expected actual =
  let diffs = V.zipWith (\e a -> abs (e - a)) expected actual
      meanError = V.sum diffs / fromIntegral (V.length diffs)
      maxError = V.maximum diffs
  in (meanError, maxError)

spec :: Spec
spec = describe "GELUDSL" $ do

  it "computes GELU correctly (small vector, FP32)" $ do
    let size = 256
        input = V.fromList [x * 0.1 | x <- [0..fromIntegral size - 1]]

    putStrLn $ "\n=== GELUDSL Test 1 (Small Vector) ==="
    putStrLn $ "Size: " ++ show size

    let cpuResult = cpuGELU input
    putStrLn $ "CPU output sum: " ++ show (V.sum cpuResult)

    gpuResult <- evalContT $ runGELUDSL input
    putStrLn $ "GPU DSL output sum: " ++ show (V.sum gpuResult)

    let (meanErr, maxErr) = compareVectors cpuResult gpuResult
    putStrLn $ "Mean error: " ++ show meanErr
    putStrLn $ "Max error: " ++ show maxErr

    -- FP32 should have very low error
    maxErr `shouldSatisfy` (< 1e-5)

  it "computes GELU correctly (large vector, FP32)" $ do
    let size = 4096
        input = V.fromList [sin (x * 0.01) | x <- [0..fromIntegral size - 1]]

    putStrLn $ "\n=== GELUDSL Test 2 (Large Vector) ==="
    putStrLn $ "Size: " ++ show size

    let cpuResult = cpuGELU input
    putStrLn $ "CPU output sum: " ++ show (V.sum cpuResult)

    gpuResult <- evalContT $ runGELUDSL input
    putStrLn $ "GPU DSL output sum: " ++ show (V.sum gpuResult)

    let (meanErr, maxErr) = compareVectors cpuResult gpuResult
    putStrLn $ "Mean error: " ++ show meanErr
    putStrLn $ "Max error: " ++ show maxErr

    maxErr `shouldSatisfy` (< 1e-5)

  it "computes GELU correctly (FP16)" $ do
    let size = 1024
        input = V.fromList [cos (x * 0.02) - 0.5 | x <- [0..fromIntegral size - 1]]

    putStrLn $ "\n=== GELUDSL Test 3 (FP16) ==="
    putStrLn $ "Size: " ++ show size

    let cpuResult = cpuGELU input
    putStrLn $ "CPU output sum: " ++ show (V.sum cpuResult)

    gpuResult <- evalContT $ runGELUDSLWithPrecision True False input
    putStrLn $ "GPU DSL (FP16) output sum: " ++ show (V.sum gpuResult)

    let (meanErr, maxErr) = compareVectors cpuResult gpuResult
    putStrLn $ "Mean error: " ++ show meanErr
    putStrLn $ "Max error: " ++ show maxErr

    -- FP16 has lower precision
    maxErr `shouldSatisfy` (< 5e-3)

  it "computes GELU correctly (Vec4 SIMD, FP32)" $ do
    let size = 1024  -- Must be multiple of 4
        input = V.fromList [x * 0.05 - 10.0 | x <- [0..fromIntegral size - 1]]

    putStrLn $ "\n=== GELUDSL Test 4 (Vec4 SIMD) ==="
    putStrLn $ "Size: " ++ show size

    let cpuResult = cpuGELU input
    putStrLn $ "CPU output sum: " ++ show (V.sum cpuResult)

    gpuResult <- evalContT $ runGELUDSLWithPrecision False True input
    putStrLn $ "GPU DSL (Vec4) output sum: " ++ show (V.sum gpuResult)

    let (meanErr, maxErr) = compareVectors cpuResult gpuResult
    putStrLn $ "Mean error: " ++ show meanErr
    putStrLn $ "Max error: " ++ show maxErr

    maxErr `shouldSatisfy` (< 1e-5)

  it "handles negative values correctly" $ do
    let size = 512
        input = V.fromList [x * 0.1 - 25.0 | x <- [0..fromIntegral size - 1]]

    putStrLn $ "\n=== GELUDSL Test 5 (Negative Values) ==="
    putStrLn $ "Size: " ++ show size
    putStrLn $ "Input range: " ++ show (V.minimum input, V.maximum input)

    let cpuResult = cpuGELU input
    putStrLn $ "CPU output sum: " ++ show (V.sum cpuResult)

    gpuResult <- evalContT $ runGELUDSL input
    putStrLn $ "GPU DSL output sum: " ++ show (V.sum gpuResult)

    let (meanErr, maxErr) = compareVectors cpuResult gpuResult
    putStrLn $ "Mean error: " ++ show meanErr
    putStrLn $ "Max error: " ++ show maxErr

    maxErr `shouldSatisfy` (< 1e-5)

  it "handles extreme values with clamping" $ do
    let size = 256
        -- Include some extreme values that would overflow tanh without clamping
        input = V.fromList $ [-50.0, -20.0, -10.0, -5.0] ++ [x * 0.2 | x <- [0..fromIntegral size - 5]] ++ [5.0, 10.0, 20.0, 50.0]

    putStrLn $ "\n=== GELUDSL Test 6 (Extreme Values) ==="
    putStrLn $ "Size: " ++ show (V.length input)
    putStrLn $ "Input range: " ++ show (V.minimum input, V.maximum input)

    let cpuResult = cpuGELU input
    putStrLn $ "CPU output sum: " ++ show (V.sum cpuResult)

    gpuResult <- evalContT $ runGELUDSL input
    putStrLn $ "GPU DSL output sum: " ++ show (V.sum gpuResult)

    let (meanErr, maxErr) = compareVectors cpuResult gpuResult
    putStrLn $ "Mean error: " ++ show meanErr
    putStrLn $ "Max error: " ++ show maxErr

    -- Check that no NaN or Inf values appear
    V.all (not . isNaN) gpuResult `shouldBe` True
    V.all (not . isInfinite) gpuResult `shouldBe` True

    maxErr `shouldSatisfy` (< 1e-5)
