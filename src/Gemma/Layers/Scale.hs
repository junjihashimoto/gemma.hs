{-# LANGUAGE OverloadedStrings #-}

module Gemma.Layers.Scale
  ( scaleVectorShader
  , runScaleVectorGPU
  ) where

import Graphics.WebGPU.Dawn.ContT

-- | Shader to scale a vector by a constant
-- output[i] = input[i] * scale
scaleVectorShader :: Int -> Float -> String
scaleVectorShader size scale = unlines
  [ "@group(0) @binding(0) var<storage, read> input: array<f32>;"
  , "@group(0) @binding(1) var<storage, read_write> output: array<f32>;"
  , ""
  , "const SIZE: u32 = " ++ show size ++ "u;"
  , "const SCALE: f32 = " ++ show scale ++ ";"
  , ""
  , "@compute @workgroup_size(256)"
  , "fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {"
  , "  let idx = global_id.x;"
  , "  if (idx >= SIZE) { return; }"
  , "  "
  , "  output[idx] = input[idx] * SCALE;"
  , "}"
  ]

-- | Run scale operation on GPU
-- Takes a GPU tensor and scales it by a constant factor
runScaleVectorGPU :: Context -> Tensor dtype -> Float -> Int -> ContT r IO (Tensor dtype)
runScaleVectorGPU ctx inputTensor scale size = do
  -- Create output tensor
  let outputShape = Shape [size]
  outputTensor <- createTensor ctx outputShape F32

  -- Create and compile shader
  let shaderCode = scaleVectorShader size scale
  code <- createKernelCode shaderCode

  -- Calculate workgroups (256 threads per workgroup)
  let numWorkgroups = (size + 255) `div` 256

  -- Create and dispatch kernel
  kernel <- createKernel ctx code [inputTensor, outputTensor] (WorkgroupSize numWorkgroups 1 1)
  liftIO $ dispatchKernelAsync ctx kernel

  -- CRITICAL: Wait for GPU operation to complete before returning!
  -- Without this, the tensor may not have the scaled values yet
  liftIO $ waitAll ctx

  pure outputTensor
