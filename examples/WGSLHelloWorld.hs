{-# LANGUAGE GADTs #-}

module Main where

import Prelude hiding ((+), (-), (*), (/), (==), (<), (>))
import qualified Prelude as P

import WGSL.AST
import WGSL.Monad
import WGSL.CodeGen
import WGSL.Builder

-- | Example 1: Simple vector addition kernel
vectorAddKernel :: ShaderM ()
vectorAddKernel = do
  -- Get global invocation ID
  gid <- globalId
  let idx = VecX gid  -- Extract x component

  -- Declare a local variable
  result <- var TF32 (LitF32 0.0)

  -- Simple computation: a[i] + b[i]
  let a = Var "inputA"  -- Assume these are bound externally
      b = Var "inputB"
      valA = PtrIndex (Ptr "inputA" :: Ptr Storage (Array 1024 F32)) idx
      valB = PtrIndex (Ptr "inputB" :: Ptr Storage (Array 1024 F32)) idx

  -- result = a[idx] + b[idx]
  result <== Add valA valB

  -- Write to output
  let output = Ptr "outputC" :: Ptr Storage (Array 1024 F32)
  assign output (Deref result)

-- | Example 2: Shared memory tiling (demonstrates barriers)
tiledReductionKernel :: ShaderM ()
tiledReductionKernel = do
  lid <- localId
  gid <- globalId
  let localIdx = VecX lid
      globalIdx = VecX gid

  -- Declare shared memory
  sharedMem <- sharedNamed "tile" (TArray 256 TF32)

  -- Load from global to shared
  let inputPtr = Ptr "input" :: Ptr Storage (Array 4096 F32)
      inputVal = PtrIndex inputPtr globalIdx
  PtrIndex sharedMem localIdx `assign` inputVal

  -- Synchronize all threads in workgroup
  barrier

  -- Reduction within shared memory (simplified)
  result <- var TF32 (PtrIndex sharedMem (LitI32 0))

  -- Write result
  if_ (Eq localIdx (LitI32 0)) (do
    let outputPtr = Ptr "output" :: Ptr Storage (Array 16 F32)
    PtrIndex outputPtr (VecX gid) `assign` Deref result
    ) (return ())

-- | Example 3: Control flow demo
controlFlowDemo :: ShaderM ()
controlFlowDemo = do
  gid <- globalId
  let idx = VecX gid

  x <- var TF32 (LitF32 10.0)

  -- If-else
  if_ (Gt (Deref x) (LitF32 5.0))
    (x <== LitF32 100.0)
    (x <== LitF32 0.0)

  -- While loop
  counter <- var TI32 (LitI32 0)
  while_ (Lt (Deref counter) (LitI32 10)) $ do
    counter <== Add (Deref counter) (LitI32 1)

  -- For loop
  for_ "i" (LitI32 0) (LitI32 5) $ do
    x <== Add (Deref x) (LitF32 1.0)

  -- Write result
  let outputPtr = Ptr "output" :: Ptr Storage (Array 1024 F32)
  PtrIndex outputPtr idx `assign` Deref x

main :: IO ()
main = do
  putStrLn "=== WGSL EDSL Examples ===\n"

  putStrLn "--- Example 1: Vector Addition ---"
  let vecAddShader = computeShader "vectorAdd" vectorAddKernel
      vecAddCode = runShaderModule [vecAddShader]
        [ ("inputA", TArray 1024 TF32, MStorage)
        , ("inputB", TArray 1024 TF32, MStorage)
        , ("outputC", TArray 1024 TF32, MStorage)
        ]
  putStrLn vecAddCode

  putStrLn "\n--- Example 2: Tiled Reduction (Shared Memory + Barrier) ---"
  let tiledShader = computeShader "tiledReduction" tiledReductionKernel
      tiledCode = runShaderModule [tiledShader]
        [ ("input", TArray 4096 TF32, MStorage)
        , ("output", TArray 16 TF32, MStorage)
        ]
  putStrLn tiledCode

  putStrLn "\n--- Example 3: Control Flow Demo ---"
  let cfShader = computeShader "controlFlow" controlFlowDemo
      cfCode = runShaderModule [cfShader]
        [ ("output", TArray 1024 TF32, MStorage)
        ]
  putStrLn cfCode

  putStrLn "\n=== All examples generated successfully! ===\n"
