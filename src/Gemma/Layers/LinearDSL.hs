{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedStrings #-}

{-|
Module: Gemma.Layers.LinearDSL
Description: DSL-based Linear (fully-connected) layer with matrix-vector multiplication

Linear layers perform the operation: y = W @ x
where:
  - W is a weight matrix of shape [out_size, in_size]
  - x is an input vector of shape [in_size]
  - y is an output vector of shape [out_size]

This module uses the type-safe WGSL DSL instead of string concatenation.
-}

module Gemma.Layers.LinearDSL
  ( -- * Configuration Types
    Precision(..)
  , LinearConfig(..)
  , Q4Weights(..)
  , defaultLinearConfig
    -- * Unified Linear Kernel
  , linearKernel
    -- * Legacy Interface (for backward compatibility)
  , linearKernelDSL
  , runLinearDSL
  , runLinearDSLWithPrecision
  , runLinearWithContextDSL
  , runLinearGPUDSL
  , runLinearPreloadedGPUDSL
  , linearKernelMMA
  , runLinearDSLWithMMA
    -- * New Unified Interface (Phase 2)
  , runLinearWithConfig
    -- * Backward-Compatible API (Phase 4) - Drop-in replacement for Gemma.Layers.Linear
  , linearShader
  , runLinear
  , runLinearWithContext
  , runLinearGPU
  , runLinearPreloadedGPU
  ) where

import Graphics.WebGPU.Dawn.ContT
import Graphics.WebGPU.Dawn (WGPUFeatureName(FeatureShaderF16, FeatureSubgroups, FeatureChromiumExperimentalSubgroupMatrix))
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import Data.Word (Word32)
import WGSL.DSL hiding ((<), (>), (<=), (>=), (==), (/=), (&&), (||), not)
import qualified WGSL.DSL as DSL
import WGSL.Execute (executeShaderNamed)
import Graphics.WebGPU.Dawn.Types (AnyTensor(..), Half)
import WGSL.CodeGen (generateWGSL)
import Gemma.Utils.Half (vectorFloatToHalf, vectorHalfToFloat)
import Prelude

-- ═══════════════════════════════════════════════════════════════
-- Phase 2: Unified Linear Configuration (dsl10.txt)
-- ═══════════════════════════════════════════════════════════════

-- | Precision modes for Linear layer computation
data Precision
  = FP32    -- ^ Standard 32-bit floating point
  | FP16    -- ^ Half precision (16-bit) for 2x memory bandwidth
  | Q4      -- ^ 4-bit quantized weights with FP16/FP32 activations
  deriving (Show, Eq)

-- | Configuration for Linear layer kernel generation
data LinearConfig = LinearConfig
  { lcPrecision :: Precision       -- ^ Precision mode (FP32, FP16, Q4)
  , lcUseSubgroups :: Bool          -- ^ Use chromium_experimental_subgroup_matrix (MMA)
  , lcUseVec4 :: Bool               -- ^ Use vec4 SIMD for 4x speedup (FP32/FP16 only)
  , lcQ4BlockSize :: Int            -- ^ Block size for Q4 quantization (default: 32)
  } deriving (Show, Eq)

-- | Default configuration: FP32, no optimizations
defaultLinearConfig :: LinearConfig
defaultLinearConfig = LinearConfig
  { lcPrecision = FP32
  , lcUseSubgroups = False
  , lcUseVec4 = False
  , lcQ4BlockSize = 32
  }

-- | Unified Linear kernel generator (Phase 2: dsl10.txt)
--
-- Generates WGSL kernel based on LinearConfig:
-- - FP32: Standard storage buffers with f32
-- - FP16: Uses 'enable f16', input/output buffers as f16
-- - Q4: Input vec4<f16>, weights array<u32>, scales array<f16>
--
-- Parameters:
-- - config: Configuration specifying precision and optimizations
-- - outSize: Number of output features
-- - inSize: Number of input features
linearKernel :: LinearConfig -> Int -> Int -> ShaderM ()
linearKernel config outSize inSize =
  case lcPrecision config of
    FP32 ->
      if lcUseSubgroups config
        then error "Subgroups not yet implemented for FP32"
        else linearKernelFP32 outSize inSize (lcUseVec4 config)

    FP16 ->
      if lcUseSubgroups config
        then linearKernelMMA outSize inSize
        else linearKernelFP16 outSize inSize (lcUseVec4 config)

    Q4 ->
      if lcUseSubgroups config
        then error "Subgroups not yet implemented for Q4"
        else linearKernelQ4 outSize inSize (lcQ4BlockSize config)

-- ═══════════════════════════════════════════════════════════════
-- Legacy Interface (preserved for backward compatibility)
-- ═══════════════════════════════════════════════════════════════

-- | DSL-based Linear kernel
--
-- Computes y = W @ x where:
-- - W is [out_size, in_size] stored in row-major order
-- - x is [in_size]
-- - y is [out_size]
--
-- Each thread computes one output element.
--
-- Parameters:
-- - outSize: Number of output features
-- - inSize: Number of input features
-- - useFP16: When True, uses FP16 for 2x memory bandwidth improvement
-- - useVec4: When True, uses vec4 SIMD for 4x additional speedup
linearKernelDSL :: Int -> Int -> Bool -> Bool -> ShaderM ()
linearKernelDSL outSize inSize useFP16 useVec4 = do
  if useFP16
    then linearKernelFP16 outSize inSize useVec4
    else linearKernelFP32 outSize inSize useVec4

-- | FP16 version of linear kernel
linearKernelFP16 :: Int -> Int -> Bool -> ShaderM ()
linearKernelFP16 outSize inSize useVec4 = do
  -- Declare buffers with FP16 storage
  weight <- declareInputBuffer "weight" (TArray (outSize * inSize) TF16)
  input <- declareInputBuffer "input" (TArray inSize TF16)
  output <- declareOutputBuffer "output" (TArray outSize TF16)

  gid <- globalId

  if_ (i32 (vecX gid) DSL.< litI32 outSize)
    (do
      -- Accumulate in FP32 for precision
      sum <- var TF32 (litF32 0.0)

      -- Use Vec4 SIMD if enabled and inSize is divisible by 4
      if useVec4 && (inSize `mod` 4 == 0)
        then do
          -- Vec4 path: process 4 elements at once
          let vec4Iters = inSize `div` 4
          loop (litI32 0) (litI32 vec4Iters) (litI32 1) $ \i -> do
            let idx = i * litI32 4
            let wBase = i32 (vecX gid) * litI32 inSize + idx

            -- Load 4 input elements (F16) and convert to F32
            in0_f16 <- readBuffer input idx
            in1_f16 <- readBuffer input (idx + litI32 1)
            in2_f16 <- readBuffer input (idx + litI32 2)
            in3_f16 <- readBuffer input (idx + litI32 3)

            let in0_f32 = F16ToF32 in0_f16
            let in1_f32 = F16ToF32 in1_f16
            let in2_f32 = F16ToF32 in2_f16
            let in3_f32 = F16ToF32 in3_f16

            -- Load 4 weight elements (F16) and convert to F32
            w0_f16 <- readBuffer weight wBase
            w1_f16 <- readBuffer weight (wBase + litI32 1)
            w2_f16 <- readBuffer weight (wBase + litI32 2)
            w3_f16 <- readBuffer weight (wBase + litI32 3)

            let w0_f32 = F16ToF32 w0_f16
            let w1_f32 = F16ToF32 w1_f16
            let w2_f32 = F16ToF32 w2_f16
            let w3_f32 = F16ToF32 w3_f16

            -- Compute 4 products in FP32
            let prod0 = in0_f32 * w0_f32
            let prod1 = in1_f32 * w1_f32
            let prod2 = in2_f32 * w2_f32
            let prod3 = in3_f32 * w3_f32

            -- Accumulate all 4
            currentSum <- readPtr sum
            sum <== currentSum + prod0 + prod1 + prod2 + prod3
        else do
          -- Scalar path
          loop (litI32 0) (litI32 inSize) (litI32 1) $ \i -> do
            let weightIdx = i32 (vecX gid) * litI32 inSize + i

            -- Read F16 and convert to F32
            wVal_f16 <- readBuffer weight weightIdx
            inVal_f16 <- readBuffer input i
            let wVal_f32 = F16ToF32 wVal_f16
            let inVal_f32 = F16ToF32 inVal_f16

            -- MAC in FP32
            let prod = wVal_f32 * inVal_f32
            currentSum <- readPtr sum
            sum <== currentSum + prod

      -- Convert result back to F16
      finalSum <- readPtr sum
      let finalOut = F32ToF16 finalSum
      writeBuffer output (i32 (vecX gid)) finalOut
    )
    (return ())

-- | FP32 version of linear kernel
linearKernelFP32 :: Int -> Int -> Bool -> ShaderM ()
linearKernelFP32 outSize inSize useVec4 = do
  -- Declare buffers with FP32 storage
  weight <- declareInputBuffer "weight" (TArray (outSize * inSize) TF32)
  input <- declareInputBuffer "input" (TArray inSize TF32)
  output <- declareOutputBuffer "output" (TArray outSize TF32)

  gid <- globalId

  if_ (i32 (vecX gid) DSL.< litI32 outSize)
    (do
      sum <- var TF32 (litF32 0.0)

      -- Use Vec4 SIMD if enabled and inSize is divisible by 4
      if useVec4 && (inSize `mod` 4 == 0)
        then do
          -- Vec4 path: process 4 elements at once
          let vec4Iters = inSize `div` 4
          loop (litI32 0) (litI32 vec4Iters) (litI32 1) $ \i -> do
            let idx = i * litI32 4
            let wBase = i32 (vecX gid) * litI32 inSize + idx

            -- Load 4 input elements
            in0 <- readBuffer input idx
            in1 <- readBuffer input (idx + litI32 1)
            in2 <- readBuffer input (idx + litI32 2)
            in3 <- readBuffer input (idx + litI32 3)

            -- Load 4 weight elements
            w0 <- readBuffer weight wBase
            w1 <- readBuffer weight (wBase + litI32 1)
            w2 <- readBuffer weight (wBase + litI32 2)
            w3 <- readBuffer weight (wBase + litI32 3)

            -- Compute 4 products
            let prod0 = in0 * w0
            let prod1 = in1 * w1
            let prod2 = in2 * w2
            let prod3 = in3 * w3

            -- Accumulate all 4
            currentSum <- readPtr sum
            sum <== currentSum + prod0 + prod1 + prod2 + prod3
        else do
          -- Scalar path
          loop (litI32 0) (litI32 inSize) (litI32 1) $ \i -> do
            let weightIdx = i32 (vecX gid) * litI32 inSize + i

            -- Read FP32 directly
            wVal <- readBuffer weight weightIdx
            inVal <- readBuffer input i

            -- MAC in FP32
            let prod = wVal * inVal
            currentSum <- readPtr sum
            sum <== currentSum + prod

      -- Write FP32 output
      finalSum <- readPtr sum
      writeBuffer output (i32 (vecX gid)) finalSum
    )
    (return ())

-- ═══════════════════════════════════════════════════════════════
-- Q4 Quantized Linear Kernel (Phase 2: dsl10.txt)
-- ═══════════════════════════════════════════════════════════════

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

-- | Q4 Quantized Linear kernel
--
-- Computes y = W @ x where:
-- - W is [out_size, in_size] quantized to Q4 format
-- - packed: nibbles packed into Word32 (8 nibbles per Word32)
-- - scales: per-block scale factors
-- - x is [in_size] in FP32
-- - y is [out_size] in FP32
--
-- Q4 Format:
-- - Block size: 32 weights per block (configurable)
-- - Packed nibbles: 8 nibbles per Word32
-- - Scales: one FP32 scale per block
-- - Dequantization: weight = (nibble - 7.5) * scale
--
-- Parameters:
-- - outSize: Number of output features
-- - inSize: Number of input features (must be multiple of blockSize)
-- - blockSize: Q4 block size (default: 32)
linearKernelQ4 :: Int -> Int -> Int -> ShaderM ()
linearKernelQ4 outSize inSize blockSize = do
  -- Validate inSize is multiple of blockSize
  if inSize `mod` blockSize /= 0
    then error $ "LinearQ4: inSize must be multiple of " ++ show blockSize ++ ", got " ++ show inSize
    else return ()

  let blocksPerRow = inSize `div` blockSize
  let nibblesPerU32 = 8
  let u32sPerBlock = blockSize `div` nibblesPerU32

  -- Declare buffers
  -- packed: [out_size * in_size / 8] Word32 elements
  -- scales: [out_size * blocksPerRow] FP32 elements
  -- input: [in_size] FP32 elements
  -- output: [out_size] FP32 elements
  packed <- declareInputBuffer "packed" (TArray (outSize * inSize `div` nibblesPerU32) TU32)
  scales <- declareInputBuffer "scales" (TArray (outSize * blocksPerRow) TF32)
  input <- declareInputBuffer "input" (TArray inSize TF32)
  output <- declareOutputBuffer "output" (TArray outSize TF32)

  -- Get global thread ID
  gid <- globalId

  -- Check bounds
  if_ (i32 (vecX gid) DSL.< litI32 outSize)
    (do
      -- Accumulator (always use FP32 for accumulation)
      sum <- var TF32 (litF32 0.0)

      -- Process each block in this row
      loop (litI32 0) (litI32 blocksPerRow) (litI32 1) $ \blockIdx -> do
        let row = i32 (vecX gid)
        let blockNum = row * litI32 blocksPerRow + blockIdx

        -- Load scale for this block
        scaleVal <- readBuffer scales blockNum

        -- Each block has u32sPerBlock Word32s
        let packedStart = blockNum * litI32 u32sPerBlock
        let inputStart = blockIdx * litI32 blockSize

        -- Process weights in this block
        loop (litI32 0) (litI32 blockSize) (litI32 1) $ \i -> do
          let wordIdx = i ./. litI32 nibblesPerU32        -- Which Word32
          let nibbleIdx_i32 = i .% litI32 nibblesPerU32  -- Which nibble in that Word32
          let nibbleIdx = I32ToU32 nibbleIdx_i32          -- Convert to U32 for bit operations

          -- Read packed value
          packedVal <- readBuffer packed (packedStart + wordIdx)

          -- Extract nibble
          nibble <- getNibble packedVal nibbleIdx

          -- Dequantize
          weight <- dequantize nibble scaleVal

          -- Read input
          let inputIdx = inputStart + i
          inVal <- readBuffer input inputIdx

          -- Multiply and accumulate
          let prod = weight * inVal
          currentSum <- readPtr sum
          sum <== currentSum + prod

      -- Write output
      finalSum <- readPtr sum
      writeBuffer output (i32 (vecX gid)) finalSum
    )
    (return ())

-- ═══════════════════════════════════════════════════════════════
-- Legacy Interface (preserved for backward compatibility)
-- ═══════════════════════════════════════════════════════════════

-- | Run Linear layer (matrix-vector multiplication) on GPU using DSL
runLinearDSL :: Vector Float -> Vector Float -> Int -> Int -> ContT r IO (Vector Float)
runLinearDSL weight input outSize inSize = do
  -- Validate inputs
  if V.length weight /= outSize * inSize
    then error $ "LinearDSL: weight size mismatch: " ++ show (V.length weight) ++ " vs " ++ show (outSize * inSize)
    else pure ()

  if V.length input /= inSize
    then error $ "LinearDSL: input size mismatch: " ++ show (V.length input) ++ " vs " ++ show inSize
    else pure ()

  -- Create GPU context
  ctx <- createContext

  -- Create tensors
  let weightShape = Shape [outSize * inSize]
      inputShape = Shape [inSize]
      outputShape = Shape [outSize]

  weightTensor <- createTensorWithData ctx weightShape weight
  inputTensor <- createTensorWithData ctx inputShape input
  outputTensor <- createTensor ctx outputShape F32

  -- Generate shader using DSL (use FP32, no vec4 for now)
  let shader = buildShaderWithAutoBinding (256, 1, 1) $
                linearKernelDSL outSize inSize False False

  -- Calculate workgroup size
  let numWorkgroups = (outSize + 255) `div` 256

  -- Execute shader
  liftIO $ executeShaderNamed ctx shader
    [ ("weight", AnyTensor weightTensor)
    , ("input", AnyTensor inputTensor)
    , ("output", AnyTensor outputTensor)
    ]
    (WorkgroupSize numWorkgroups 1 1)

  -- Read result from GPU
  result <- liftIO $ fromGPU ctx outputTensor outSize

  pure result

-- | Run Linear layer with configurable precision (FP16/FP32) and vectorization (Vec4)
runLinearDSLWithPrecision :: Vector Float -> Vector Float -> Int -> Int -> Bool -> Bool -> ContT r IO (Vector Float)
runLinearDSLWithPrecision weight input outSize inSize useFP16 useVec4 = do
  -- Validate inputs
  if V.length weight /= outSize * inSize
    then error $ "LinearDSL: weight size mismatch: " ++ show (V.length weight) ++ " vs " ++ show (outSize * inSize)
    else pure ()

  if V.length input /= inSize
    then error $ "LinearDSL: input size mismatch: " ++ show (V.length input) ++ " vs " ++ show inSize
    else pure ()

  -- Create GPU context
  ctx <- createContext

  let weightShape = Shape [outSize * inSize]
      inputShape = Shape [inSize]
      outputShape = Shape [outSize]

  -- Create tensors based on precision
  if useFP16
    then do
      -- Convert Float to Half for FP16 precision
      let weightHalf = vectorFloatToHalf weight
          inputHalf = vectorFloatToHalf input

      weightTensor <- createTensorWithData ctx weightShape weightHalf
      inputTensor <- createTensorWithData ctx inputShape inputHalf
      outputTensor <- createTensor ctx outputShape F16

      -- Generate shader with FP16 support
      let shader = (buildShaderWithAutoBinding (256, 1, 1) $ linearKernelDSL outSize inSize True useVec4)
                   { moduleExtensions = ["f16"] }

      let numWorkgroups = (outSize + 255) `div` 256

      -- Execute shader
      liftIO $ executeShaderNamed ctx shader
        [ ("weight", AnyTensor weightTensor)
        , ("input", AnyTensor inputTensor)
        , ("output", AnyTensor outputTensor)
        ]
        (WorkgroupSize numWorkgroups 1 1)

      -- Read FP16 result and convert back to Float
      resultHalf <- liftIO $ fromGPU ctx outputTensor outSize :: ContT r IO (Vector Half)
      pure $ vectorHalfToFloat resultHalf

    else do
      -- FP32 path
      weightTensor <- createTensorWithData ctx weightShape weight
      inputTensor <- createTensorWithData ctx inputShape input
      outputTensor <- createTensor ctx outputShape F32

      let shader = buildShaderWithAutoBinding (256, 1, 1) $ linearKernelDSL outSize inSize False useVec4
      let numWorkgroups = (outSize + 255) `div` 256

      liftIO $ executeShaderNamed ctx shader
        [ ("weight", AnyTensor weightTensor)
        , ("input", AnyTensor inputTensor)
        , ("output", AnyTensor outputTensor)
        ]
        (WorkgroupSize numWorkgroups 1 1)

      result <- liftIO $ fromGPU ctx outputTensor outSize
      pure result

-- | Run Linear with given context (for use in larger pipelines)
runLinearWithContextDSL :: Context -> Vector Float -> Vector Float -> Int -> Int -> ContT r IO (Vector Float)
runLinearWithContextDSL ctx weight input outSize inSize = do
  -- Validate inputs
  if V.length weight /= outSize * inSize
    then error $ "LinearDSL: weight size mismatch: " ++ show (V.length weight) ++ " vs " ++ show (outSize * inSize)
    else pure ()

  if V.length input /= inSize
    then error $ "LinearDSL: input size mismatch: " ++ show (V.length input) ++ " vs " ++ show inSize
    else pure ()

  -- Create tensors
  let weightShape = Shape [outSize * inSize]
      inputShape = Shape [inSize]
      outputShape = Shape [outSize]

  weightTensor <- createTensorWithData ctx weightShape weight
  inputTensor <- createTensorWithData ctx inputShape input
  outputTensor <- createTensor ctx outputShape F32

  -- Generate shader using DSL
  let shader = buildShaderWithAutoBinding (256, 1, 1) $
                linearKernelDSL outSize inSize False False

  -- Calculate workgroup size
  let numWorkgroups = (outSize + 255) `div` 256

  -- Execute shader
  liftIO $ executeShaderNamed ctx shader
    [ ("weight", AnyTensor weightTensor)
    , ("input", AnyTensor inputTensor)
    , ("output", AnyTensor outputTensor)
    ]
    (WorkgroupSize numWorkgroups 1 1)

  -- Read result from GPU
  result <- liftIO $ fromGPU ctx outputTensor outSize

  pure result

-- | GPU-resident Linear layer - keeps tensors on GPU (NO CPU transfers!)
runLinearGPUDSL :: Context
                -> Vector Float
                -> Tensor dtype
                -> Int
                -> Int
                -> ContT r IO (Tensor dtype)
runLinearGPUDSL ctx weight inputTensor outSize inSize = do
  -- Validate
  if V.length weight /= outSize * inSize
    then error $ "LinearGPUDSL: weight size mismatch"
    else pure ()

  -- Create weight and output tensors (input already on GPU!)
  let weightShape = Shape [outSize, inSize]
      outputShape = Shape [outSize]
  weightTensor <- createTensorWithData ctx weightShape weight
  outputTensor <- createTensor ctx outputShape F32

  -- Generate shader using DSL
  let shader = buildShaderWithAutoBinding (256, 1, 1) $
                linearKernelDSL outSize inSize False False

  -- Calculate workgroup size
  let numWorkgroups = (outSize + 255) `div` 256

  -- Execute shader
  liftIO $ executeShaderNamed ctx shader
    [ ("weight", AnyTensor weightTensor)
    , ("input", AnyTensor inputTensor)
    , ("output", AnyTensor outputTensor)
    ]
    (WorkgroupSize numWorkgroups 1 1)

  -- Return GPU tensor (NO download!)
  pure outputTensor

-- | GPU-resident Linear with PRE-UPLOADED weights (DSL version)
--
-- Note: For preloaded version, we need to pre-compile the shader module
-- and pass it as an argument, similar to the string-based version.
-- For now, this regenerates the shader (not optimal for performance).
runLinearPreloadedGPUDSL :: Context
                         -> Tensor dtype  -- Weight tensor on GPU
                         -> Tensor dtype  -- Input tensor on GPU
                         -> Tensor dtype  -- Output buffer (pre-allocated, REUSED!)
                         -> Int           -- outSize
                         -> Int           -- inSize
                         -> ContT r IO ()
runLinearPreloadedGPUDSL ctx weightTensor inputTensor outputTensor outSize inSize = do
  -- Generate shader using DSL
  -- TODO: In production, this should be pre-compiled and cached
  let shader = buildShaderWithAutoBinding (256, 1, 1) $
                linearKernelDSL outSize inSize False False

  -- Calculate workgroup size
  let numWorkgroups = (outSize + 255) `div` 256

  -- Execute shader
  liftIO $ executeShaderNamed ctx shader
    [ ("weight", AnyTensor weightTensor)
    , ("input", AnyTensor inputTensor)
    , ("output", AnyTensor outputTensor)
    ]
    (WorkgroupSize numWorkgroups 1 1)

-- | MMA-based Linear kernel using subgroup matrix operations
--
-- NOTE: This is designed for BATCHED operations (matrix-matrix multiplication)
-- For single vector (batch=1), use Vec4 SIMD instead - it's more efficient!
--
-- Uses hardware-accelerated tensor cores via chromium_experimental_subgroup_matrix
--
-- **Optimal Use Case**: Batched inference where input is [batchSize, inSize]
--   - batchSize should be multiple of 8 for full tile utilization
--   - Performs: output[batch, out] = weights[out, in] @ input[batch, in]
--
-- **Why not for single vectors?**
--   - Subgroup matrices expect 8×8 tiles
--   - Single vector is [inSize, 1] - wastes 7/8 of tile capacity
--   - Vec4 SIMD is ~4x faster for single vectors
--
-- Requires:
--   - FP16 precision
--   - Subgroup matrix feature
--   - Dimensions aligned to 8 (for 8x8 subgroup tiles)
--   - batchSize ≥ 8 for optimal performance
--
-- Performance: ~10-20x faster than Vec4 for batched operations
linearKernelMMA :: Int -> Int -> ShaderM ()
linearKernelMMA outSize inSize = do
  -- Declare buffers with FP16 storage (required for subgroup matrices)
  weight <- declareInputBuffer "weight" (TArray (outSize * inSize) TF16)
  input <- declareInputBuffer "input" (TArray inSize TF16)
  output <- declareOutputBuffer "output" (TArray outSize TF16)

  -- Get workgroup ID
  wg <- workgroupId

  let wgX = vecX wg

  -- Each workgroup processes 8 output rows (one subgroup tile)
  let rowStart = wgX * litU32 8

  -- Create accumulator subgroup matrix (8x8 result tile)
  acc <- newSubgroupMatrixZero ResultMatrix TF16 8 8

  -- Create left matrix (weight tile) and right matrix (input tile)
  a <- newSubgroupMatrix LeftMatrix TF16 8 8
  b <- newSubgroupMatrix RightMatrix TF16 8 8

  -- Loop over input dimension in 8-element chunks
  loop (litI32 0) (litI32 inSize) (litI32 8) $ \k -> do
    barrier

    let kU = u32 k
    let weightOffset = rowStart * litU32 (fromIntegral inSize) + kU
    let inputOffset = kU

    -- Load 8x8 tile from weight matrix
    loadMatrix a weight weightOffset (litU32 (fromIntegral inSize)) (TSubgroupMatrixLeft TF16 8 8)

    -- Load 8x1 column from input (treat as 8x8 with duplicated columns)
    loadMatrix b input inputOffset (litU32 1) (TSubgroupMatrixRight TF16 8 8)

    -- Multiply-accumulate: acc += a * b
    mma acc a b

  barrier

  -- Store result (8x8 tile, but we only use first column)
  let outputOffset = rowStart
  storeMatrix output outputOffset acc (litU32 1)

-- | Run Linear layer with MMA acceleration (requires compatible GPU)
--
-- Requirements:
--   - GPU must support chromium_experimental_subgroup_matrix
--   - Dimensions should be aligned to 8 for best performance
--   - Uses FP16 internally for tensor core compatibility
runLinearDSLWithMMA :: Vector Float -> Vector Float -> Int -> Int -> ContT r IO (Vector Float)
runLinearDSLWithMMA weight input outSize inSize = do
  -- Validate inputs
  if V.length weight /= outSize * inSize
    then error $ "LinearDSL MMA: weight size mismatch: " ++ show (V.length weight) ++ " vs " ++ show (outSize * inSize)
    else pure ()

  if V.length input /= inSize
    then error $ "LinearDSL MMA: input size mismatch: " ++ show (V.length input) ++ " vs " ++ show inSize
    else pure ()

  -- Create GPU context with subgroup matrix features
  ctx <- createContextWithFeatures
    ["allow_unsafe_apis"]
    [FeatureShaderF16, FeatureSubgroups, FeatureChromiumExperimentalSubgroupMatrix]

  let weightShape = Shape [outSize * inSize]
      inputShape = Shape [inSize]
      outputShape = Shape [outSize]

  -- Convert Float to Half for FP16 precision
  let weightHalf = vectorFloatToHalf weight
      inputHalf = vectorFloatToHalf input

  weightTensor <- createTensorWithData ctx weightShape weightHalf
  inputTensor <- createTensorWithData ctx inputShape inputHalf
  outputTensor <- createTensor ctx outputShape F16

  -- Generate shader with MMA support
  let shader = (buildShaderWithAutoBinding (32, 1, 1) $ linearKernelMMA outSize inSize)
               { moduleExtensions = ["f16", "chromium_experimental_subgroup_matrix"]
               , moduleDiagnostics = ["off, chromium.subgroup_matrix_uniformity"]
               }

  -- Calculate workgroups (one workgroup per 8 output rows)
  let numWorkgroups = (outSize + 7) `div` 8

  -- Execute shader
  liftIO $ executeShaderNamed ctx shader
    [ ("weight", AnyTensor weightTensor)
    , ("input", AnyTensor inputTensor)
    , ("output", AnyTensor outputTensor)
    ]
    (WorkgroupSize numWorkgroups 1 1)

  -- Read FP16 result and convert back to Float
  resultHalf <- liftIO $ fromGPU ctx outputTensor outSize :: ContT r IO (Vector Half)
  pure $ vectorHalfToFloat resultHalf

-- ═══════════════════════════════════════════════════════════════
-- Unified Interface (Phase 2: dsl10.txt)
-- ═══════════════════════════════════════════════════════════════

-- | Data type for Q4 weights
data Q4Weights = Q4Weights
  { q4Packed :: Vector Word32  -- ^ Packed nibbles (8 nibbles per Word32)
  , q4Scales :: Vector Float    -- ^ Per-block scale factors
  } deriving (Show)

-- | Unified Linear layer execution using LinearConfig
--
-- This is the primary interface for Phase 2 (dsl10.txt).
-- Supports FP32, FP16, and Q4 precisions with a single unified API.
--
-- Parameters:
-- - config: LinearConfig specifying precision and optimizations
-- - weights: Either (Vector Float) for FP32/FP16 or Q4Weights for Q4
-- - input: Input vector (always Vector Float)
-- - outSize: Number of output features
-- - inSize: Number of input features
--
-- Returns: Output vector (Vector Float)
runLinearWithConfig :: LinearConfig
                    -> Either (Vector Float) Q4Weights  -- weights
                    -> Vector Float                      -- input
                    -> Int                               -- outSize
                    -> Int                               -- inSize
                    -> ContT r IO (Vector Float)
runLinearWithConfig config weightsOrQ4 input outSize inSize =
  case lcPrecision config of
    FP32 -> do
      -- FP32 path: standard floating point weights
      weights <- case weightsOrQ4 of
        Left w -> pure w
        Right _ -> error "Q4Weights provided but FP32 precision requested"

      -- Validate
      if V.length weights /= outSize * inSize
        then error $ "LinearConfig: weight size mismatch: " ++ show (V.length weights) ++ " vs " ++ show (outSize * inSize)
        else pure ()

      if V.length input /= inSize
        then error $ "LinearConfig: input size mismatch"
        else pure ()

      -- Create context
      ctx <- createContext

      -- Create tensors
      let weightShape = Shape [outSize * inSize]
          inputShape = Shape [inSize]
          outputShape = Shape [outSize]

      weightTensor <- createTensorWithData ctx weightShape weights
      inputTensor <- createTensorWithData ctx inputShape input
      outputTensor <- createTensor ctx outputShape F32

      -- Generate shader
      let shader = buildShaderWithAutoBinding (256, 1, 1) $
                    linearKernel config outSize inSize

      let numWorkgroups = (outSize + 255) `div` 256

      -- Execute
      liftIO $ executeShaderNamed ctx shader
        [ ("weight", AnyTensor weightTensor)
        , ("input", AnyTensor inputTensor)
        , ("output", AnyTensor outputTensor)
        ]
        (WorkgroupSize numWorkgroups 1 1)

      -- Read result
      result <- liftIO $ fromGPU ctx outputTensor outSize
      pure result

    FP16 -> do
      -- FP16 path: half precision weights
      weights <- case weightsOrQ4 of
        Left w -> pure w
        Right _ -> error "Q4Weights provided but FP16 precision requested"

      -- Validate
      if V.length weights /= outSize * inSize
        then error $ "LinearConfig: weight size mismatch"
        else pure ()

      if V.length input /= inSize
        then error $ "LinearConfig: input size mismatch"
        else pure ()

      -- Create context with FP16 features
      ctx <- if lcUseSubgroups config
             then createContextWithFeatures
                    ["allow_unsafe_apis"]
                    [FeatureShaderF16, FeatureSubgroups, FeatureChromiumExperimentalSubgroupMatrix]
             else createContextWithFeatures [] [FeatureShaderF16]

      let weightShape = Shape [outSize * inSize]
          inputShape = Shape [inSize]
          outputShape = Shape [outSize]

      -- Convert to FP16
      let weightHalf = vectorFloatToHalf weights
          inputHalf = vectorFloatToHalf input

      weightTensor <- createTensorWithData ctx weightShape weightHalf
      inputTensor <- createTensorWithData ctx inputShape inputHalf
      outputTensor <- createTensor ctx outputShape F16

      -- Generate shader
      let extensions = if lcUseSubgroups config
                      then ["f16", "chromium_experimental_subgroup_matrix"]
                      else ["f16"]
          diagnostics = if lcUseSubgroups config
                       then ["off, chromium.subgroup_matrix_uniformity"]
                       else []
          shader = (buildShaderWithAutoBinding (if lcUseSubgroups config then 32 else 256, 1, 1) $
                     linearKernel config outSize inSize)
                   { moduleExtensions = extensions
                   , moduleDiagnostics = diagnostics
                   }

      let numWorkgroups = if lcUseSubgroups config
                         then (outSize + 7) `div` 8
                         else (outSize + 255) `div` 256

      -- Execute
      liftIO $ executeShaderNamed ctx shader
        [ ("weight", AnyTensor weightTensor)
        , ("input", AnyTensor inputTensor)
        , ("output", AnyTensor outputTensor)
        ]
        (WorkgroupSize numWorkgroups 1 1)

      -- Read FP16 result and convert to Float
      resultHalf <- liftIO $ fromGPU ctx outputTensor outSize :: ContT r IO (Vector Half)
      pure $ vectorHalfToFloat resultHalf

    Q4 -> do
      -- Q4 path: quantized weights
      Q4Weights packed scales <- case weightsOrQ4 of
        Right q4 -> pure q4
        Left _ -> error "Float weights provided but Q4 precision requested"

      -- Validate
      let blockSize = lcQ4BlockSize config
      if inSize `mod` blockSize /= 0
        then error $ "LinearConfig Q4: inSize must be multiple of " ++ show blockSize
        else pure ()

      if V.length packed /= outSize * inSize `div` 8
        then error $ "LinearConfig Q4: packed size mismatch"
        else pure ()

      if V.length scales /= outSize * inSize `div` blockSize
        then error $ "LinearConfig Q4: scales size mismatch"
        else pure ()

      if V.length input /= inSize
        then error $ "LinearConfig Q4: input size mismatch"
        else pure ()

      -- Create context
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

      -- Generate shader
      let shader = buildShaderWithAutoBinding (256, 1, 1) $
                    linearKernel config outSize inSize

      let numWorkgroups = (outSize + 255) `div` 256

      -- Execute
      liftIO $ executeShaderNamed ctx shader
        [ ("packed", AnyTensor packedTensor)
        , ("scales", AnyTensor scalesTensor)
        , ("input", AnyTensor inputTensor)
        , ("output", AnyTensor outputTensor)
        ]
        (WorkgroupSize numWorkgroups 1 1)

      -- Read result
      result <- liftIO $ fromGPU ctx outputTensor outSize
      pure result

-- ═══════════════════════════════════════════════════════════════
-- BACKWARD-COMPATIBLE API (Phase 4)
-- Drop-in replacement for Gemma.Layers.Linear
-- ═══════════════════════════════════════════════════════════════

-- | Generate WGSL shader code for linear layer (backward-compatible API)
--
-- This function provides the same interface as the old string-based linearShader,
-- but generates the WGSL code using the type-safe DSL.
--
-- Parameters:
-- - outSize: Output dimension
-- - inSize: Input dimension
-- - useFP16: Use FP16 precision (2x memory bandwidth)
-- - useVec4: Use vec4 SIMD (4x speedup)
linearShader :: Int -> Int -> Bool -> Bool -> String
linearShader outSize inSize useFP16 useVec4 =
  let precision = if useFP16 then FP16 else FP32
      config = LinearConfig
        { lcPrecision = precision
        , lcUseSubgroups = False
        , lcUseVec4 = useVec4
        , lcQ4BlockSize = 32
        }
      baseShader = buildShaderWithAutoBinding (256, 1, 1) $ linearKernel config outSize inSize
      extensions = if useFP16 then ["f16"] else []
      shader = baseShader { moduleExtensions = extensions }
      wgsl = generateWGSL shader
  in wgsl

-- | Run Linear layer (backward-compatible API)
--
-- Creates its own GPU context and runs the linear layer.
-- Same interface as Gemma.Layers.Linear.runLinear.
runLinear :: Vector Float -> Vector Float -> Int -> Int -> ContT r IO (Vector Float)
runLinear weight input outSize inSize = do
  -- Validate inputs
  if V.length weight /= outSize * inSize
    then error $ "Linear: weight size mismatch: " ++ show (V.length weight) ++ " vs " ++ show (outSize * inSize)
    else pure ()

  if V.length input /= inSize
    then error $ "Linear: input size mismatch: " ++ show (V.length input) ++ " vs " ++ show inSize
    else pure ()

  -- Use default FP32 config (matching old behavior)
  let config = defaultLinearConfig { lcPrecision = FP32 }
  runLinearWithConfig config (Left weight) input outSize inSize

-- | Run Linear with given context (backward-compatible API)
--
-- Uses an existing GPU context (for use in larger pipelines).
-- Same interface as Gemma.Layers.Linear.runLinearWithContext.
runLinearWithContext :: Context -> Vector Float -> Vector Float -> Int -> Int -> ContT r IO (Vector Float)
runLinearWithContext ctx weight input outSize inSize = do
  -- Validate inputs
  if V.length weight /= outSize * inSize
    then error $ "Linear: weight size mismatch: " ++ show (V.length weight) ++ " vs " ++ show (outSize * inSize)
    else pure ()

  if V.length input /= inSize
    then error $ "Linear: input size mismatch: " ++ show (V.length input) ++ " vs " ++ show inSize
    else pure ()

  -- Create tensors
  let weightShape = Shape [outSize * inSize]
      inputShape = Shape [inSize]
      outputShape = Shape [outSize]

  weightTensor <- createTensorWithData ctx weightShape weight
  inputTensor <- createTensorWithData ctx inputShape input
  outputTensor <- createTensor ctx outputShape F32

  -- Generate shader using DSL
  let config = defaultLinearConfig { lcPrecision = FP32 }
      shader = buildShaderWithAutoBinding (256, 1, 1) $ linearKernel config outSize inSize

  let numWorkgroups = (outSize + 255) `div` 256

  -- Execute
  liftIO $ executeShaderNamed ctx shader
    [ ("weight", AnyTensor weightTensor)
    , ("input", AnyTensor inputTensor)
    , ("output", AnyTensor outputTensor)
    ]
    (WorkgroupSize numWorkgroups 1 1)

  -- Read result from GPU
  result <- liftIO $ fromGPU ctx outputTensor outSize
  pure result

-- | GPU-resident Linear layer (backward-compatible API)
--
-- Keeps tensors on GPU, eliminating transfer overhead.
-- Same interface as Gemma.Layers.Linear.runLinearGPU.
runLinearGPU :: Context
             -> Vector Float
             -> Tensor dtype
             -> Int
             -> Int
             -> ContT r IO (Tensor dtype)
runLinearGPU ctx weight inputTensor outSize inSize = do
  -- Validate
  if V.length weight /= outSize * inSize
    then error $ "LinearGPU: weight size mismatch"
    else pure ()

  -- Create weight and output tensors (input already on GPU!)
  let weightShape = Shape [outSize, inSize]
      outputShape = Shape [outSize]
  weightTensor <- createTensorWithData ctx weightShape weight
  outputTensor <- createTensor ctx outputShape F32

  -- Generate shader using DSL
  let config = defaultLinearConfig { lcPrecision = FP32 }
      shader = buildShaderWithAutoBinding (256, 1, 1) $ linearKernel config outSize inSize

  -- Create and dispatch kernel
  let numWorkgroups = (outSize + 255) `div` 256

  liftIO $ executeShaderNamed ctx shader
    [ ("weight", AnyTensor weightTensor)
    , ("input", AnyTensor inputTensor)
    , ("output", AnyTensor outputTensor)
    ]
    (WorkgroupSize numWorkgroups 1 1)

  -- Return GPU tensor (NO download!)
  pure outputTensor

-- | GPU-resident Linear with pre-uploaded weights (backward-compatible API)
--
-- Takes GPU tensor for weights AND pre-compiled KernelCode.
-- Zero uploads, zero shader compilations - maximum performance!
-- Same interface as Gemma.Layers.Linear.runLinearPreloadedGPU.
runLinearPreloadedGPU :: Context
                      -> Tensor dtype  -- Weight tensor on GPU [outSize, inSize]
                      -> Tensor dtype  -- Input tensor on GPU [inSize]
                      -> Tensor dtype  -- Output buffer (pre-allocated, REUSED!)
                      -> KernelCode      -- PRE-COMPILED shader!
                      -> Int             -- outSize
                      -> ContT r IO ()
runLinearPreloadedGPU ctx weightTensor inputTensor outputTensor code outSize = do
  let numWorkgroups = (outSize + 255) `div` 256
  kernel <- createKernel ctx code [weightTensor, inputTensor, outputTensor]
            (WorkgroupSize numWorkgroups 1 1)
  liftIO $ dispatchKernelAsync ctx kernel
