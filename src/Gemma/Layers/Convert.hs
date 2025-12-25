{-# LANGUAGE OverloadedStrings #-}

module Gemma.Layers.Convert
  ( convertPrecisionShader
  , runConvertPrecisionGPU
  , PrecisionType(..)
  ) where

import Graphics.WebGPU.Dawn.ContT
import Graphics.WebGPU.Dawn.Types (Tensor(..), KernelCode, Context, Shape(..), NumType(..))

-- | Supported precision types for conversion
data PrecisionType = FP32 | FP16 | BF16
  deriving (Show, Eq)

-- | Get WGSL type name for precision type
precisionTypeName :: PrecisionType -> String
precisionTypeName FP32 = "f32"
precisionTypeName FP16 = "f16"
precisionTypeName BF16 = "f32"  -- BF16 stored as u32, converted to f32

-- | Check if we need to enable f16 extension
needsF16Extension :: PrecisionType -> PrecisionType -> Bool
needsF16Extension from to = from == FP16 || to == FP16

-- | Shader to convert between different precision types
-- Supports: FP32 ↔ FP16, FP32 ↔ BF16, FP16 ↔ BF16
convertPrecisionShader :: PrecisionType -> PrecisionType -> Int -> String
convertPrecisionShader fromType toType size =
  let enableF16 = if needsF16Extension fromType toType
                  then ["enable f16;", ""]
                  else []

      fromTypeName = precisionTypeName fromType
      toTypeName = precisionTypeName toType

      -- BF16 is stored as u16, but we need special conversion
      (inputBinding, outputBinding) = case (fromType, toType) of
        -- BF16 input: read as u16, convert to f32
        (BF16, FP32) ->
          ( "@group(0) @binding(0) var<storage, read> input: array<u32>;"
          , "@group(0) @binding(1) var<storage, read_write> output: array<f32>;"
          )
        (BF16, FP16) ->
          ( "@group(0) @binding(0) var<storage, read> input: array<u32>;"
          , "@group(0) @binding(1) var<storage, read_write> output: array<f16>;"
          )
        -- BF16 output: convert from f32, write as u16
        (FP32, BF16) ->
          ( "@group(0) @binding(0) var<storage, read> input: array<f32>;"
          , "@group(0) @binding(1) var<storage, read_write> output: array<u32>;"
          )
        (FP16, BF16) ->
          ( "@group(0) @binding(0) var<storage, read> input: array<f16>;"
          , "@group(0) @binding(1) var<storage, read_write> output: array<u32>;"
          )
        -- Standard conversions
        _ ->
          ( "@group(0) @binding(0) var<storage, read> input: array<" ++ fromTypeName ++ ">;"
          , "@group(0) @binding(1) var<storage, read_write> output: array<" ++ toTypeName ++ ">;"
          )

      conversionCode = case (fromType, toType) of
        -- BF16 to FP32: shift left by 16 bits
        (BF16, FP32) -> unlines
          [ "  let bf16_bits = input[idx];"
          , "  let fp32_bits = bf16_bits << 16u;"
          , "  output[idx] = bitcast<f32>(fp32_bits);"
          ]

        -- BF16 to FP16: convert via FP32 intermediate
        (BF16, FP16) -> unlines
          [ "  let bf16_bits = input[idx];"
          , "  let fp32_bits = bf16_bits << 16u;"
          , "  let fp32_val = bitcast<f32>(fp32_bits);"
          , "  output[idx] = f16(fp32_val);"
          ]

        -- FP32 to BF16: shift right by 16 bits (truncate mantissa)
        (FP32, BF16) -> unlines
          [ "  let fp32_bits = bitcast<u32>(input[idx]);"
          , "  let bf16_bits = fp32_bits >> 16u;"
          , "  output[idx] = bf16_bits;"
          ]

        -- FP16 to BF16: convert via FP32 intermediate
        (FP16, BF16) -> unlines
          [ "  let fp32_val = f32(input[idx]);"
          , "  let fp32_bits = bitcast<u32>(fp32_val);"
          , "  let bf16_bits = fp32_bits >> 16u;"
          , "  output[idx] = bf16_bits;"
          ]

        -- FP32 ↔ FP16: direct conversion
        (FP32, FP16) -> "  output[idx] = f16(input[idx]);"
        (FP16, FP32) -> "  output[idx] = f32(input[idx]);"

        -- Same type: direct copy (shouldn't happen, but handle it)
        (FP32, FP32) -> "  output[idx] = input[idx];"
        (FP16, FP16) -> "  output[idx] = input[idx];"
        (BF16, BF16) -> "  output[idx] = input[idx];"

  in unlines $ enableF16 ++
    [ "// Precision conversion: " ++ show fromType ++ " -> " ++ show toType
    , inputBinding
    , outputBinding
    , ""
    , "const SIZE: u32 = " ++ show size ++ "u;"
    , ""
    , "@compute @workgroup_size(256)"
    , "fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {"
    , "  let idx = global_id.x;"
    , "  if (idx >= SIZE) { return; }"
    , ""
    , conversionCode
    , "}"
    ]

-- | Run precision conversion on GPU
-- Converts a tensor from one precision to another
runConvertPrecisionGPU :: Context
                       -> Tensor dtype        -- Input tensor
                       -> PrecisionType       -- Source precision
                       -> PrecisionType       -- Target precision
                       -> Int                 -- Size
                       -> ContT r IO (Tensor dtype)
runConvertPrecisionGPU ctx inputTensor fromType toType size = do
  -- Create output tensor with appropriate type
  let outputShape = Shape [size]
      outputNumType = case toType of
        FP32 -> F32
        FP16 -> F16
        BF16 -> U32  -- BF16 stored as u32

  outputTensor <- createTensor ctx outputShape outputNumType

  -- Create and compile shader
  let shaderCode = convertPrecisionShader fromType toType size
  code <- createKernelCode shaderCode

  -- Calculate workgroups (256 threads per workgroup)
  let numWorkgroups = (size + 255) `div` 256

  -- Create and dispatch kernel
  kernel <- createKernel ctx code [inputTensor, outputTensor] (WorkgroupSize numWorkgroups 1 1)
  liftIO $ dispatchKernel ctx kernel

  pure outputTensor
