{-# LANGUAGE OverloadedStrings #-}

module Gemma.Layers.AttentionDSLSpec (spec) where

import Test.Hspec
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import Graphics.WebGPU.Dawn.ContT (evalContT)

import Gemma.Layers.AttentionDSL

-- Helper: CPU implementation of scaled dot-product attention
-- Attn(Q, K, V) = softmax(Q @ K^T / sqrt(head_dim)) @ V
cpuAttention :: Vector Float -> Vector Float -> Vector Float -> Int -> Int -> Vector Float
cpuAttention q k v seqLen headDim =
  let -- Step 1: Q @ K^T (Q: [seqLen, headDim], K^T: [headDim, seqLen])
      scores = cpuMatMulTranspose q k seqLen headDim seqLen
      -- Step 2: Scale by 1/sqrt(head_dim)
      scale = 1.0 / sqrt (fromIntegral headDim)
      scoresScaled = V.map (* scale) scores
      -- Step 3: Softmax per row
      attn = cpuSoftmax scoresScaled seqLen seqLen
      -- Step 4: Attn @ V
      output = cpuMatMul attn v seqLen seqLen headDim
  in output

-- Helper: Matrix multiplication A @ B^T where B is transposed
-- A: [m, k], B: [n, k] (stored row-major, accessed as B^T: [k, n]) -> C: [m, n]
cpuMatMulTranspose :: Vector Float -> Vector Float -> Int -> Int -> Int -> Vector Float
cpuMatMulTranspose a b m k n =
  V.generate (m * n) $ \idx ->
    let row = idx `div` n
        col = idx `mod` n
    in V.sum $ V.zipWith (*)
         (V.slice (row * k) k a)
         (V.slice (col * k) k b)  -- Access B's row (which becomes column in B^T)

-- Helper: Matrix multiplication (A @ B)
-- A: [m, k], B: [k, n] -> C: [m, n]
cpuMatMul :: Vector Float -> Vector Float -> Int -> Int -> Int -> Vector Float
cpuMatMul a b m k n =
  V.generate (m * n) $ \idx ->
    let row = idx `div` n
        col = idx `mod` n
    in V.sum $ V.zipWith (*)
         (V.slice (row * k) k a)
         (V.generate k (\i -> b V.! (i * n + col)))

-- Helper: Softmax along rows
-- Input: [rows, cols], Output: [rows, cols]
cpuSoftmax :: Vector Float -> Int -> Int -> Vector Float
cpuSoftmax input rows cols =
  V.concat $ map procesRow [0..rows-1]
  where
    procesRow row =
      let rowStart = row * cols
          rowData = V.slice rowStart cols input
          -- Find max for numerical stability
          maxVal = V.maximum rowData
          -- Compute exp(x - max)
          expData = V.map (\x -> exp (x - maxVal)) rowData
          -- Normalize by sum
          sumExp = V.sum expData
      in V.map (/ sumExp) expData

-- Helper: Calculate mean absolute error
meanAbsError :: Vector Float -> Vector Float -> Float
meanAbsError v1 v2 =
  let diffs = V.zipWith (\a b -> abs (a - b)) v1 v2
  in V.sum diffs / fromIntegral (V.length diffs)

-- Helper: Calculate max absolute error
maxAbsError :: Vector Float -> Vector Float -> Float
maxAbsError v1 v2 =
  V.maximum $ V.zipWith (\a b -> abs (a - b)) v1 v2

spec :: Spec
spec = do
  describe "AttentionDSL (DSL-based GPU Attention with MMA)" $ do

    -- Test 1: Small attention to verify correctness
    it "DSL-based GPU attention matches CPU attention (small)" $ do
      -- Test: seqLen=8, headDim=16
      let seqLen = 8
          headDim = 16
          q = V.fromList [sin (fromIntegral i / 10.0) | i <- [0..seqLen*headDim-1]] :: Vector Float
          k = V.fromList [cos (fromIntegral i / 10.0) | i <- [0..seqLen*headDim-1]] :: Vector Float
          v = V.fromList [sin (fromIntegral i / 5.0) | i <- [0..seqLen*headDim-1]] :: Vector Float

      -- CPU baseline
      let cpuOutput = cpuAttention q k v seqLen headDim

      putStrLn $ "=== AttentionDSL Test 1 (Small) ==="
      putStrLn $ "Shape: seqLen=" ++ show seqLen ++ ", headDim=" ++ show headDim
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuOutput)

      -- GPU DSL-based attention
      gpuOutput <- evalContT $ runAttentionDSLWithMMA q k v seqLen headDim

      putStrLn $ "GPU DSL output sum: " ++ show (V.sum gpuOutput)
      putStrLn $ "Mean error: " ++ show (meanAbsError cpuOutput gpuOutput)
      putStrLn $ "Max error: " ++ show (maxAbsError cpuOutput gpuOutput)

      -- MMA uses FP16 internally, so allow error for precision loss
      let err = meanAbsError cpuOutput gpuOutput
      err `shouldSatisfy` (< 0.05)  -- 5% acceptable for FP16 attention pipeline

    -- Test 2: Medium attention (realistic size)
    it "DSL-based GPU attention matches CPU attention (medium)" $ do
      -- Test: seqLen=32, headDim=64 (typical for small models)
      let seqLen = 32
          headDim = 64
          q = V.generate (seqLen * headDim) (\i -> sin (fromIntegral i / 100.0)) :: Vector Float
          k = V.generate (seqLen * headDim) (\i -> cos (fromIntegral i / 100.0)) :: Vector Float
          v = V.generate (seqLen * headDim) (\i -> sin (fromIntegral i / 50.0)) :: Vector Float

      -- CPU baseline
      let cpuOutput = cpuAttention q k v seqLen headDim

      putStrLn $ "=== AttentionDSL Test 2 (Medium) ==="
      putStrLn $ "Shape: seqLen=" ++ show seqLen ++ ", headDim=" ++ show headDim
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuOutput)

      -- GPU DSL-based attention
      gpuOutput <- evalContT $ runAttentionDSLWithMMA q k v seqLen headDim

      putStrLn $ "GPU DSL output sum: " ++ show (V.sum gpuOutput)
      putStrLn $ "Mean error: " ++ show (meanAbsError cpuOutput gpuOutput)
      putStrLn $ "Max error: " ++ show (maxAbsError cpuOutput gpuOutput)

      -- MMA uses FP16 internally, so allow error for precision loss
      let err = meanAbsError cpuOutput gpuOutput
      err `shouldSatisfy` (< 0.05)  -- 5% acceptable for FP16 attention pipeline

    -- Test 3: Large attention (stress test)
    it "DSL-based GPU attention matches CPU attention (large)" $ do
      -- Test: seqLen=128, headDim=64 (typical for medium models)
      let seqLen = 128
          headDim = 64
          q = V.generate (seqLen * headDim) (\i -> fromIntegral (i `mod` 100) / 100.0) :: Vector Float
          k = V.generate (seqLen * headDim) (\i -> fromIntegral (i `mod` 50) / 50.0) :: Vector Float
          v = V.generate (seqLen * headDim) (\i -> fromIntegral (i `mod` 75) / 75.0) :: Vector Float

      -- CPU baseline
      let cpuOutput = cpuAttention q k v seqLen headDim

      putStrLn $ "=== AttentionDSL Test 3 (Large) ==="
      putStrLn $ "Shape: seqLen=" ++ show seqLen ++ ", headDim=" ++ show headDim
      putStrLn $ "CPU output sum: " ++ show (V.sum cpuOutput)

      -- GPU DSL-based attention
      gpuOutput <- evalContT $ runAttentionDSLWithMMA q k v seqLen headDim

      putStrLn $ "GPU DSL output sum: " ++ show (V.sum gpuOutput)
      putStrLn $ "Mean error: " ++ show (meanAbsError cpuOutput gpuOutput)
      putStrLn $ "Max error: " ++ show (maxAbsError cpuOutput gpuOutput)

      -- MMA uses FP16 internally, so allow error for precision loss
      let err = meanAbsError cpuOutput gpuOutput
      err `shouldSatisfy` (< 0.05)  -- 5% acceptable for FP16 attention pipeline
