{-# LANGUAGE OverloadedStrings #-}

{-|
Module: Gemma.Layers.LinearQ4
Description: 4-bit quantized linear layer with GPU shader

This module implements Q4 quantized matrix-vector multiplication on GPU.
Uses 4-bit block-wise quantization for 4× memory bandwidth reduction.

Q4 Format:
- Weights are quantized to 4-bit per value (nibbles)
- Block size: 32 weights per block
- Each block has one FP16/FP32 scale factor
- Packing: 8 nibbles per Word32 (4 Word32s per block)

Memory savings:
- FP32: 4 bytes/weight
- FP16: 2 bytes/weight
- Q4: 0.5 bytes/weight (packed) + scale overhead
- Compression: 4× vs FP16, 8× vs FP32
-}

module Gemma.Layers.LinearQ4
  ( -- * Q4 Linear Layer
    runLinearQ4GPU
  , runLinearQ4PreloadedGPU
    -- * Q4 Shader
  , linearQ4Shader
  ) where

import Graphics.WebGPU.Dawn.ContT
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import Data.Word (Word32)

-- | WGSL compute shader for Q4 matrix-vector multiplication
--
-- Computes y = W @ x where:
-- - W is [out_size, in_size] quantized to Q4 format
-- - packed: [out_size * in_size / 8] packed nibbles (Word32)
-- - scales: [out_size * in_size / 32] per-block scales
-- - x is [in_size] input vector
-- - y is [out_size] output vector
--
-- Each thread computes one output element.
--
-- Q4 Dequantization:
--   weight = (nibble - 7.5) * scale
--
-- Block layout (32 weights):
--   - 4 Word32s of packed nibbles
--   - 1 scale (FP16 or FP32)
linearQ4Shader :: Int -> Int -> Bool -> String
linearQ4Shader outSize inSize useFP16 =
  let floatType = if useFP16 then "f16" else "f32"
      floatLit = if useFP16 then "h" else ""
      enableDirective = if useFP16 then ["enable f16;", ""] else []
      blocksPerRow = inSize `div` 32
  in unlines $
  [ "// Q4 Linear layer: quantized matrix-vector multiplication"
  , "// W: [out_size, in_size] in Q4 format (4-bit per weight)"
  , "// packed: nibbles packed into Word32 (8 nibbles per Word32)"
  , "// scales: per-block (32 weights) scale factors"
  , "// x: [in_size]"
  , "// y: [out_size]"
  , "//"
  , "// NOTE: Using read_write for integer buffers due to WebGPU/Dawn bug"
  , "// where read-only integer storage buffers return zeros"
  ] ++ enableDirective ++
  [ "@group(0) @binding(0) var<storage, read_write> packed: array<u32>;"  -- WORKAROUND: read_write instead of read
  , "@group(0) @binding(1) var<storage, read_write> scales: array<" ++ floatType ++ ">;"  -- WORKAROUND: read_write instead of read
  , "@group(0) @binding(2) var<storage, read_write> input: array<" ++ floatType ++ ">;"  -- WORKAROUND: read_write instead of read
  , "@group(0) @binding(3) var<storage, read_write> output: array<" ++ floatType ++ ">;"
  , ""
  , "const OUT_SIZE: u32 = " ++ show outSize ++ "u;"
  , "const IN_SIZE: u32 = " ++ show inSize ++ "u;"
  , "const BLOCKS_PER_ROW: u32 = " ++ show blocksPerRow ++ "u;"
  , ""
  , "// Extract nibble from packed Word32"
  , "fn getNibble(packed_val: u32, nibble_idx: u32) -> u32 {"
  , "  return (packed_val >> (nibble_idx * 4u)) & 0xFu;"
  , "}"
  , ""
  , "// Dequantize nibble to float"
  , "fn dequantize(nibble: u32, scale: " ++ floatType ++ ") -> " ++ floatType ++ " {"
  , "  if (scale < 1e-7" ++ floatLit ++ ") {"
  , "    return 0.0" ++ floatLit ++ ";"
  , "  }"
  , "  let shifted = " ++ floatType ++ "(nibble) - 7.5" ++ floatLit ++ ";"
  , "  return shifted * scale;"
  , "}"
  , ""
  , "@compute @workgroup_size(256)"
  , "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {"
  , "  let row = gid.x;"
  , "  "
  , "  if (row < OUT_SIZE) {"
  , "    var sum: " ++ floatType ++ " = 0.0" ++ floatLit ++ ";"
  , "    "
  , "    // Process each block (32 weights) in this row"
  , "    for (var block_idx: u32 = 0u; block_idx < BLOCKS_PER_ROW; block_idx = block_idx + 1u) {"
  , "      let block_num = row * BLOCKS_PER_ROW + block_idx;"
  , "      let scale = scales[block_num];"
  , "      "
  , "      // Each block has 4 Word32s (32 nibbles total)"
  , "      let packed_start = block_num * 4u;"
  , "      let input_start = block_idx * 32u;"
  , "      "
  , "      // Process 32 weights in this block"
  , "      for (var i: u32 = 0u; i < 32u; i = i + 1u) {"
  , "        let word_idx = i / 8u;  // Which Word32 (0-3)"
  , "        let nibble_idx = i % 8u;  // Which nibble in that Word32 (0-7)"
  , "        "
  , "        let packed_val = packed[packed_start + word_idx];"
  , "        let nibble = getNibble(packed_val, nibble_idx);"
  , "        let weight = dequantize(nibble, scale);"
  , "        let input_idx = input_start + i;"
  , "        sum = sum + weight * input[input_idx];"
  , "      }"
  , "    }"
  , "    "
  , "    output[row] = sum;"
  , "  }"
  , "}"
  ]

-- | Run Q4 Linear layer on GPU
--
-- Takes:
-- - packed: packed nibbles [out_size * in_size / 8] (Word32 vector)
-- - scales: per-block scales [out_size * in_size / 32] (Float vector)
-- - input: input vector [in_size]
-- - outSize: number of output features
-- - inSize: number of input features (must be multiple of 32)
--
-- Returns: output vector [out_size]
runLinearQ4GPU :: Vector Word32 -> Vector Float -> Vector Float -> Int -> Int -> ContT r IO (Vector Float)
runLinearQ4GPU packed scales input outSize inSize = do
  -- Validate inputs
  if inSize `mod` 32 /= 0
    then error $ "LinearQ4: inSize must be multiple of 32, got " ++ show inSize
    else pure ()

  if V.length packed /= outSize * inSize `div` 8
    then error $ "LinearQ4: packed size mismatch: " ++ show (V.length packed) ++ " vs " ++ show (outSize * inSize `div` 8)
    else pure ()

  if V.length scales /= outSize * inSize `div` 32
    then error $ "LinearQ4: scales size mismatch: " ++ show (V.length scales) ++ " vs " ++ show (outSize * inSize `div` 32)
    else pure ()

  if V.length input /= inSize
    then error $ "LinearQ4: input size mismatch: " ++ show (V.length input) ++ " vs " ++ show inSize
    else pure ()

  -- Create GPU context
  ctx <- createContext

  -- Create tensors
  let packedShape = Shape [V.length packed]
      scalesShape = Shape [V.length scales]
      inputShape = Shape [inSize]
      outputShape = Shape [outSize]

  -- Upload tensors
  liftIO $ putStrLn $ "Creating packed tensor with " ++ show (V.length packed) ++ " Word32 elements"
  packedTensor <- createTensorWithData ctx packedShape packed
  liftIO $ putStrLn "Packed tensor created"

  scalesTensor <- createTensorWithData ctx scalesShape scales
  liftIO $ putStrLn "Scales tensor created"

  inputTensor <- createTensorWithData ctx inputShape input
  liftIO $ putStrLn "Input tensor created"

  outputTensor <- createTensor ctx outputShape F32
  liftIO $ putStrLn "Output tensor created"

  -- Compile Q4 shader (use FP32 for now)
  let shaderCode = linearQ4Shader outSize inSize False
  liftIO $ putStrLn "=== Q4 SHADER CODE ==="
  liftIO $ putStrLn shaderCode
  liftIO $ putStrLn "=== END SHADER ==="
  liftIO $ putStrLn "Creating kernel code..."
  code <- createKernelCode shaderCode
  liftIO $ putStrLn "Kernel code created successfully"

  -- Create kernel with enough workgroups to cover all output rows
  let numWorkgroups = (outSize + 63) `div` 64  -- Ceiling division (workgroup size 64)
  liftIO $ putStrLn $ "Creating kernel with " ++ show numWorkgroups ++ " workgroups"
  kernel <- createKernel ctx code [packedTensor, scalesTensor, inputTensor, outputTensor]
            (WorkgroupSize numWorkgroups 1 1)
  liftIO $ putStrLn "Kernel created successfully"

  -- Dispatch kernel (synchronous for debugging)
  liftIO $ putStrLn "Dispatching kernel..."
  liftIO $ dispatchKernel ctx kernel  -- Changed to synchronous
  liftIO $ putStrLn "Kernel dispatched (synchronous)"

  -- Read result from GPU
  result <- liftIO $ fromGPU ctx outputTensor outSize
  liftIO $ putStrLn $ "fromGPU returned: " ++ show (V.toList result)
  liftIO $ putStrLn $ "fromGPU length: " ++ show (V.length result)

  pure result

-- | Run Q4 Linear with preloaded tensors (for pipelines)
runLinearQ4PreloadedGPU :: Context -> Tensor dtype -> Tensor dtype -> Tensor dtype -> Int -> Int -> ContT r IO (Vector Float)
runLinearQ4PreloadedGPU _ _ _ _ _ _ = error "runLinearQ4PreloadedGPU: not implemented yet (TDD RED phase)"
