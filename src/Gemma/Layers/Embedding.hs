{-# LANGUAGE OverloadedStrings #-}

{-|
Module: Gemma.Layers.Embedding
Description: Token Embedding Layer

Converts token IDs to dense embeddings by looking up in an embedding table.
For each token ID, retrieves the corresponding embedding vector from the table.

Formula:
  For token ID i: embedding_output[i] = embedding_table[token_id[i]]

This is a simple lookup operation that maps discrete token IDs to continuous vectors.
-}

module Gemma.Layers.Embedding
  ( runEmbedding
  , runEmbeddingGPU
  , embeddingShader
  ) where

import Graphics.WebGPU.Dawn.ContT
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import Data.Int (Int32)

-- | WGSL shader for embedding lookup
--
-- Each thread looks up one token's embedding from the table.
embeddingShader :: Bool -> Int -> Int -> String
embeddingShader useFP16 seqLen embedDim =
  let floatType = if useFP16 then "f16" else "f32"
      enableDirective = if useFP16 then ["enable f16;", ""] else []
  in unlines $
  [ "// Embedding lookup"
  , "// For each token ID, copy its embedding from the table"
  , ""
  ] ++ enableDirective ++
  [ "@group(0) @binding(0) var<storage, read_write> token_ids: array<i32>;"
  , "@group(0) @binding(1) var<storage, read_write> embed_table: array<" ++ floatType ++ ">;"
  , "@group(0) @binding(2) var<storage, read_write> output: array<" ++ floatType ++ ">;"
  , ""
  , "const SEQ_LEN: u32 = " ++ show seqLen ++ "u;"
  , "const EMBED_DIM: u32 = " ++ show embedDim ++ "u;"
  , ""
  , "@compute @workgroup_size(256)"
  , "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {"
  , "  let token_idx = gid.x;"
  , "  "
  , "  if (token_idx < SEQ_LEN) {"
  , "    // Get token ID"
  , "    let token_id = token_ids[token_idx];"
  , "    "
  , "    // Calculate offsets"
  , "    let embed_offset = u32(token_id) * EMBED_DIM;"
  , "    let output_offset = token_idx * EMBED_DIM;"
  , "    "
  , "    // Copy embedding vector for this token"
  , "    for (var i: u32 = 0u; i < EMBED_DIM; i = i + 1u) {"
  , "      output[output_offset + i] = embed_table[embed_offset + i];"
  , "    }"
  , "  }"
  , "}"
  ]

-- | Run Embedding lookup on GPU
--
-- Takes:
-- - tokenIds: token IDs [seq_len] as Int (will be converted to Int32 for GPU)
-- - embedTable: embedding table [vocab_size * embed_dim] in row-major order
-- - vocabSize: size of vocabulary
-- - embedDim: dimension of embeddings
--
-- Returns: embeddings [seq_len * embed_dim] in row-major order
runEmbedding :: Vector Int -> Vector Float -> Int -> Int -> ContT r IO (Vector Float)
runEmbedding tokenIds embedTable vocabSize embedDim = do
  let seqLen = V.length tokenIds

  -- Validate inputs
  if V.length embedTable /= vocabSize * embedDim
    then error $ "Embedding: table size mismatch: " ++ show (V.length embedTable) ++ " vs " ++ show (vocabSize * embedDim)
    else pure ()

  -- Convert Int to Int32 for GPU
  let tokenIds32 = V.map fromIntegral tokenIds :: Vector Int32

  -- Create GPU context
  ctx <- createContext

  -- Create tensors
  let tokenIdsShape = Shape [seqLen]
      embedTableShape = Shape [vocabSize * embedDim]
      outputShape = Shape [seqLen * embedDim]

  tokenIdsTensor <- createTensorWithData ctx tokenIdsShape tokenIds32
  embedTableTensor <- createTensorWithData ctx embedTableShape embedTable
  outputTensor <- createTensor ctx outputShape F32

  -- Compile shader
  let shaderCode = embeddingShader False seqLen embedDim  -- FP32 for now
  code <- createKernelCode shaderCode

  -- Create kernel - one thread per token
  let numWorkgroups = (seqLen + 255) `div` 256
  kernel <- createKernel ctx code [tokenIdsTensor, embedTableTensor, outputTensor]
            (WorkgroupSize numWorkgroups 1 1)

  -- Dispatch kernel
  liftIO $ dispatchKernel ctx kernel

  -- Read result from GPU
  result <- liftIO $ fromGPU ctx outputTensor (seqLen * embedDim)

  pure result

-- | GPU-resident embedding with PRE-UPLOADED table and PRE-COMPILED shader
--
-- Takes pre-uploaded embedding table on GPU and pre-compiled KernelCode.
-- Returns embedding tensor on GPU (no download) - ready for immediate use!
runEmbeddingGPU :: Context
                -> Vector Int        -- Token IDs [seqLen]
                -> Tensor dtype  -- Pre-uploaded embedding table on GPU
                -> KernelCode        -- Pre-compiled shader
                -> Bool              -- useFP16
                -> Int               -- embedDim
                -> ContT r IO (Tensor dtype)  -- Output embeddings on GPU [seqLen * embedDim]
runEmbeddingGPU ctx tokenIds embedTableTensor code useFP16 embedDim = do
  let seqLen = V.length tokenIds

  -- === VALIDATION 1: Check input token IDs ===
  let tokenIds32 = V.map fromIntegral tokenIds :: Vector Int32
  liftIO $ putStrLn $ "🔍 EMBEDDING CHECK: seqLen=" ++ show seqLen ++
                      ", tokenIds=" ++ show (V.toList tokenIds) ++
                      ", embedDim=" ++ show embedDim ++
                      ", useFP16=" ++ show useFP16

  -- === VALIDATION 2: Check embedding table tensor ===
  liftIO $ putStrLn "🔍 EMBEDDING TABLE: Pre-uploaded tensor received"

  -- Create tensors
  let tokenIdsShape = Shape [seqLen]
      outputShape = Shape [seqLen * embedDim]
      numType = if useFP16 then F16 else F32

  liftIO $ putStrLn $ "🔍 OUTPUT TENSOR: creating with type=" ++ show numType ++
                      ", shape=" ++ show outputShape

  tokenIdsTensor <- createTensorWithData ctx tokenIdsShape tokenIds32

  outputTensor <- createTensor ctx outputShape numType

  -- === VALIDATION 3: Verify tensor creation ===
  liftIO $ putStrLn "✅ Tensors created successfully"

  -- Create kernel - one thread per token
  let numWorkgroups = (seqLen + 255) `div` 256
  liftIO $ putStrLn $ "🔍 KERNEL: creating with " ++ show numWorkgroups ++ " workgroups"

  kernel <- createKernel ctx code [tokenIdsTensor, embedTableTensor, outputTensor]
            (WorkgroupSize numWorkgroups 1 1)

  liftIO $ putStrLn "✅ Kernel created successfully"

  -- === DISPATCH WITH SYNCHRONIZATION ===
  -- Use synchronous dispatch to catch errors immediately
  liftIO $ putStrLn "🚀 Dispatching embedding kernel (SYNCHRONOUS)..."
  liftIO $ dispatchKernel ctx kernel

  -- Wait for GPU to finish
  liftIO $ putStrLn "⏳ Waiting for GPU completion..."
  liftIO $ waitAll ctx
  liftIO $ putStrLn "✅ Embedding kernel completed"

  -- === VALIDATION 4: Sample output to verify it's not all zeros ===
  -- Read first 10 elements to check (always as Float for validation)
  liftIO $ putStrLn "🔍 Checking output tensor (first 10 elements)..."
  sampleOutput <- liftIO $ (fromGPU ctx outputTensor 10 :: IO (Vector Float))
  liftIO $ putStrLn $ "   Sample output: " ++ show (V.toList sampleOutput)

  let allZeros = V.all (== (0.0 :: Float)) sampleOutput
  if allZeros
    then liftIO $ putStrLn "❌ WARNING: Output is all zeros! Embedding lookup failed!"
    else liftIO $ putStrLn "✅ Output has non-zero values"

  -- Return GPU tensor (NO download!)
  pure outputTensor
