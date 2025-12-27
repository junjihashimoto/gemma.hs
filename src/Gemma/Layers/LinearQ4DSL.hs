{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

{-|
Module: Gemma.Layers.LinearQ4DSL
Description: DSL-based 4-bit quantized linear layer

Linear Q4 layers perform quantized matrix-vector multiplication: y = W @ x
where:
  - W is a weight matrix of shape [out_size, in_size] quantized to 4-bit
  - Packing: 8 nibbles per Word32 (4 Word32s per 32-weight block)
  - Each block has one FP32 scale factor
  - x is an input vector of shape [in_size]
  - y is an output vector of shape [out_size]

Q4 Format:
  - Block size: 32 weights per block
  - Packed nibbles: [out_size * in_size / 8] Word32 elements
  - Scales: [out_size * in_size / 32] FP32 elements
  - Dequantization: weight = (nibble - 7.5) * scale

This module uses the type-safe WGSL DSL instead of string concatenation.
-}

module Gemma.Layers.LinearQ4DSL
  ( linearQ4KernelDSL
  , runLinearQ4DSL
  , runLinearQ4WithContextDSL
  ) where

import Graphics.WebGPU.Dawn.ContT
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import Data.Word (Word32)
import WGSL.DSL hiding ((<), (>), (<=), (>=), (==), (/=), (&&), (||), not)
import qualified WGSL.DSL as DSL
import WGSL.Execute (executeShaderNamed)
import Graphics.WebGPU.Dawn.Types (AnyTensor(..))
import WGSL.CodeGen (generateWGSL)
import Prelude

-- | Extract nibble from packed Word32
--
-- WGSL: fn getNibble(packed_val: u32, nibble_idx: u32) -> u32
--   return (packed_val >> (nibble_idx * 4u)) & 0xFu;
getNibble :: Exp U32 -> Exp U32 -> ShaderM (Exp U32)
getNibble packedVal nibbleIdx = do
  let shiftAmount = nibbleIdx * litU32 4
  let shifted = packedVal .>>. shiftAmount
  let nibble = shifted .&. litU32 0xF
  return nibble

-- | Dequantize nibble to float
--
-- WGSL: fn dequantize(nibble: u32, scale: f32) -> f32
--   if (scale < 1e-7) return 0.0;
--   let shifted = f32(nibble) - 7.5;
--   return shifted * scale;
dequantize :: Exp U32 -> Exp F32 -> ShaderM (Exp F32)
dequantize nibble scale = do
  -- Check for zero scale (avoid division issues)
  result <- var TF32 (litF32 0.0)

  if_ (scale DSL.>= litF32 1e-7)
    (do
      let nibbleF32 = I32ToF32 (U32ToI32 nibble)
      let shifted = nibbleF32 - litF32 7.5
      let weight = shifted * scale
      result <== weight
    )
    (result <== litF32 0.0)

  readPtr result

-- | DSL-based Q4 Linear kernel
--
-- Computes y = W @ x where:
-- - W is [out_size, in_size] quantized to Q4 format
-- - packed: nibbles packed into Word32 (8 nibbles per Word32)
-- - scales: per-block (32 weights) scale factors
-- - x is [in_size]
-- - y is [out_size]
--
-- Each thread computes one output element.
--
-- Parameters:
-- - outSize: Number of output features
-- - inSize: Number of input features (must be multiple of 32)
-- - useFP16: When True, uses FP16 for scales and input/output
linearQ4KernelDSL :: Int -> Int -> Bool -> ShaderM ()
linearQ4KernelDSL outSize inSize useFP16 = do
  -- Validate inSize is multiple of 32
  if inSize `mod` 32 /= 0
    then error $ "LinearQ4DSL: inSize must be multiple of 32, got " ++ show inSize
    else return ()

  let floatType = if useFP16 then TF16 else TF32
  let blocksPerRow = inSize `div` 32

  -- Declare buffers with automatic binding
  -- NOTE: Using read_write for Word32 buffers due to WebGPU/Dawn bug
  packed <- declareInputBuffer "packed" (TArray (outSize * inSize `div` 8) TU32)
  scales <- declareInputBuffer "scales" (TArray (outSize * blocksPerRow) floatType)
  input <- declareInputBuffer "input" (TArray inSize floatType)
  output <- declareOutputBuffer "output" (TArray outSize floatType)

  -- Get global thread ID
  gid <- globalId

  -- Check bounds
  if_ (i32 (vecX gid) DSL.< litI32 outSize)
    (do
      -- Accumulator (always use FP32 for accumulation)
      sum <- var TF32 (litF32 0.0)

      -- Process each block (32 weights) in this row
      loop (litI32 0) (litI32 blocksPerRow) (litI32 1) $ \blockIdx -> do
        let row = i32 (vecX gid)
        let blockNum = row * litI32 blocksPerRow + blockIdx

        -- Load scale for this block
        scaleVal <- readBuffer scales blockNum
        let scaleFP32 = if useFP16 then error "FP16 conversion not implemented" else scaleVal

        -- Each block has 4 Word32s (32 nibbles total)
        let packedStart = blockNum * litI32 4
        let inputStart = blockIdx * litI32 32

        -- Process 32 weights in this block
        loop (litI32 0) (litI32 32) (litI32 1) $ \i -> do
          let wordIdx = i ./. litI32 8        -- Which Word32 (0-3)
          let nibbleIdx_i32 = i .% litI32 8  -- Which nibble in that Word32 (0-7)
          let nibbleIdx = I32ToU32 nibbleIdx_i32  -- Convert to U32 for bit operations

          -- Read packed value
          packedVal <- readBuffer packed (packedStart + wordIdx)

          -- Extract nibble
          nibble <- getNibble packedVal nibbleIdx

          -- Dequantize
          weight <- dequantize nibble scaleFP32

          -- Read input
          let inputIdx = inputStart + i
          inVal <- readBuffer input inputIdx
          let inValFP32 = if useFP16 then error "FP16 conversion not implemented" else inVal

          -- Multiply and accumulate
          let prod = weight * inValFP32
          currentSum <- readPtr sum
          sum <== currentSum + prod

      -- Write output
      finalSum <- readPtr sum
      let finalOut = if useFP16 then error "FP16 conversion not implemented" else finalSum
      writeBuffer output (i32 (vecX gid)) finalOut
    )
    (return ())

-- | Run Q4 Linear layer (matrix-vector multiplication) on GPU using DSL
runLinearQ4DSL :: Vector Word32 -> Vector Float -> Vector Float
               -> Int -> Int -> ContT r IO (Vector Float)
runLinearQ4DSL packed scales input outSize inSize = do
  -- Validate inputs
  if inSize `mod` 32 /= 0
    then error $ "LinearQ4DSL: inSize must be multiple of 32, got " ++ show inSize
    else pure ()

  if V.length packed /= outSize * inSize `div` 8
    then error $ "LinearQ4DSL: packed size mismatch"
    else pure ()

  if V.length scales /= outSize * inSize `div` 32
    then error $ "LinearQ4DSL: scales size mismatch"
    else pure ()

  if V.length input /= inSize
    then error $ "LinearQ4DSL: input size mismatch"
    else pure ()

  -- Create GPU context
  ctx <- createContext

  -- Create tensors
  let packedShape = Shape [V.length packed]
      scalesShape = Shape [V.length scales]
      inputShape = Shape [inSize]
      outputShape = Shape [outSize]

  packedTensor <- createTensorWithData ctx packedShape packed
  scalesTensor <- createTensorWithData ctx scalesShape scales
  inputTensor <- createTensorWithData ctx inputShape input
  outputTensor <- createTensor ctx outputShape F32

  -- Generate shader using DSL (use FP32 for now)
  let shader = buildShaderWithAutoBinding (256, 1, 1) $
                linearQ4KernelDSL outSize inSize False

  -- Calculate workgroup size
  let numWorkgroups = (outSize + 255) `div` 256

  -- Execute shader
  liftIO $ executeShaderNamed ctx shader
    [ ("packed", AnyTensor packedTensor)
    , ("scales", AnyTensor scalesTensor)
    , ("input", AnyTensor inputTensor)
    , ("output", AnyTensor outputTensor)
    ]
    (WorkgroupSize numWorkgroups 1 1)

  -- Read result from GPU
  result <- liftIO $ fromGPU ctx outputTensor outSize

  pure result

-- | Run Q4 Linear with given context (for use in larger pipelines)
runLinearQ4WithContextDSL :: Context
                          -> Vector Word32 -> Vector Float -> Vector Float
                          -> Int -> Int -> ContT r IO (Vector Float)
runLinearQ4WithContextDSL ctx packed scales input outSize inSize = do
  -- Validate inputs
  if inSize `mod` 32 /= 0
    then error $ "LinearQ4DSL: inSize must be multiple of 32"
    else pure ()

  if V.length packed /= outSize * inSize `div` 8
    then error $ "LinearQ4DSL: packed size mismatch"
    else pure ()

  if V.length scales /= outSize * inSize `div` 32
    then error $ "LinearQ4DSL: scales size mismatch"
    else pure ()

  if V.length input /= inSize
    then error $ "LinearQ4DSL: input size mismatch"
    else pure ()

  -- Create tensors
  let packedShape = Shape [V.length packed]
      scalesShape = Shape [V.length scales]
      inputShape = Shape [inSize]
      outputShape = Shape [outSize]

  packedTensor <- createTensorWithData ctx packedShape packed
  scalesTensor <- createTensorWithData ctx scalesShape scales
  inputTensor <- createTensorWithData ctx inputShape input
  outputTensor <- createTensor ctx outputShape F32

  -- Generate shader using DSL
  let shader = buildShaderWithAutoBinding (256, 1, 1) $
                linearQ4KernelDSL outSize inSize False

  -- Calculate workgroup size
  let numWorkgroups = (outSize + 255) `div` 256

  -- Execute shader
  liftIO $ executeShaderNamed ctx shader
    [ ("packed", AnyTensor packedTensor)
    , ("scales", AnyTensor scalesTensor)
    , ("input", AnyTensor inputTensor)
    , ("output", AnyTensor outputTensor)
    ]
    (WorkgroupSize numWorkgroups 1 1)

  -- Read result from GPU
  result <- liftIO $ fromGPU ctx outputTensor outSize

  pure result
