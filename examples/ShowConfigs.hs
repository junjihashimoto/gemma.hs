{-# LANGUAGE RecordWildCards #-}

{-|
Program: Show Gemma Configurations
Description: Displays and compares Gemma 1, 2, and 3 configurations
-}

module Main where

import Gemma.Model

-- | Pretty print a GemmaConfig
showConfig :: String -> GemmaConfig -> String
showConfig name GemmaConfig{..} = unlines
  [ "=== " ++ name ++ " ==="
  , ""
  , "Architecture:"
  , "  Vocabulary Size:  " ++ show gcVocabSize
  , "  Hidden Dimension: " ++ show gcHiddenDim
  , "  Num Layers:       " ++ show gcNumLayers
  , "  Num Q Heads:      " ++ show gcNumHeads
  , "  Num KV Heads:     " ++ show gcNumKVHeads
  , "  Head Dimension:   " ++ show gcHeadDim
  , "  FFN Dimension:    " ++ show gcFFNDim
  , "  RoPE Base:        " ++ show gcRopeBase
  , ""
  , "GQA Ratio: " ++ show (gcNumHeads `div` gcNumKVHeads) ++ ":1"
  , "Total Parameters: ~" ++ show (estimateParams GemmaConfig{..}) ++ "M"
  , ""
  , "Gemma 3 Features:"
  , "  QK-Norm:              " ++ show gcUseQKNorm
  , "  Post-Attn Norm:       " ++ show gcUsePostAttnNorm
  , "  Post-FFN Norm:        " ++ show gcUsePostFFNNorm
  , "  Sliding Window:       " ++ show gcUseSlidingWindow
  , "  Window Size:          " ++ show gcSlidingWindowSize
  , "  Local RoPE Scaling:   " ++ show gcLocalRopeScaling
  , "  Global RoPE Scaling:  " ++ show gcGlobalRopeScaling
  , "  Query Head Dim Norm:  " ++ show gcQueryHeadDimNormalize
  , "  Zero-Centered RMSNorm:" ++ show gcUseZeroCenteredRMSNorm
  , ""
  ]

-- | Estimate model parameters (very rough)
estimateParams :: GemmaConfig -> Int
estimateParams GemmaConfig{..} =
  let embedParams = gcVocabSize * gcHiddenDim

      -- Per-layer parameters
      attnQ = gcHiddenDim * gcNumHeads * gcHeadDim
      attnK = gcHiddenDim * gcNumKVHeads * gcHeadDim
      attnV = gcHiddenDim * gcNumKVHeads * gcHeadDim
      attnO = gcNumHeads * gcHeadDim * gcHiddenDim
      attnTotal = attnQ + attnK + attnV + attnO

      ffnGate = gcHiddenDim * gcFFNDim
      ffnUp = gcHiddenDim * gcFFNDim
      ffnDown = gcFFNDim * gcHiddenDim
      ffnTotal = ffnGate + ffnUp + ffnDown

      normParams = gcHiddenDim * 2  -- attn_norm + ffn_norm

      layerParams = attnTotal + ffnTotal + normParams
      allLayerParams = layerParams * gcNumLayers

      lmHeadParams = gcVocabSize * gcHiddenDim

      totalParams = embedParams + allLayerParams + lmHeadParams
  in totalParams `div` 1000000  -- Convert to millions

-- | Compare two configs and show differences
compareConfigs :: String -> GemmaConfig -> String -> GemmaConfig -> String
compareConfigs name1 cfg1 name2 cfg2 = unlines
  [ "=== Differences: " ++ name1 ++ " vs " ++ name2 ++ " ==="
  , ""
  , "Architecture Changes:"
  , diffLine "  Vocabulary Size" (gcVocabSize cfg1) (gcVocabSize cfg2)
  , diffLine "  Hidden Dimension" (gcHiddenDim cfg1) (gcHiddenDim cfg2)
  , diffLine "  Num Layers" (gcNumLayers cfg1) (gcNumLayers cfg2)
  , diffLine "  Num Q Heads" (gcNumHeads cfg1) (gcNumHeads cfg2)
  , diffLine "  Num KV Heads" (gcNumKVHeads cfg1) (gcNumKVHeads cfg2)
  , diffLine "  Head Dimension" (gcHeadDim cfg1) (gcHeadDim cfg2)
  , diffLine "  FFN Dimension" (gcFFNDim cfg1) (gcFFNDim cfg2)
  , ""
  , "New Features in " ++ name2 ++ ":"
  , boolDiff "  QK-Norm" (gcUseQKNorm cfg1) (gcUseQKNorm cfg2)
  , boolDiff "  Post-Attn Norm" (gcUsePostAttnNorm cfg1) (gcUsePostAttnNorm cfg2)
  , boolDiff "  Post-FFN Norm" (gcUsePostFFNNorm cfg1) (gcUsePostFFNNorm cfg2)
  , boolDiff "  Sliding Window" (gcUseSlidingWindow cfg1) (gcUseSlidingWindow cfg2)
  , boolDiff "  Zero-Centered RMSNorm" (gcUseZeroCenteredRMSNorm cfg1) (gcUseZeroCenteredRMSNorm cfg2)
  , ""
  ]
  where
    diffLine label v1 v2 =
      if v1 == v2
      then label ++ ": " ++ show v1 ++ " (unchanged)"
      else label ++ ": " ++ show v1 ++ " -> " ++ show v2 ++ " ✨"

    boolDiff label False True = label ++ ": disabled -> enabled ✨"
    boolDiff label True False = label ++ ": enabled -> disabled"
    boolDiff label _ _ = label ++ ": unchanged"

main :: IO ()
main = do
  putStrLn "Gemma Model Configurations\n"

  -- Show all configs
  putStrLn $ showConfig "Gemma 1 (1B)" gemma1BConfig
  putStrLn $ showConfig "Gemma 2 (2B)" gemma2_2BConfig
  putStrLn $ showConfig "Gemma 3 (1B)" gemma3_1BConfig

  -- Compare Gemma 1 vs Gemma 3
  putStrLn $ compareConfigs "Gemma 1" gemma1BConfig "Gemma 3" gemma3_1BConfig

  putStrLn "Key Architectural Improvements in Gemma 3:"
  putStrLn "  1. QK-Norm: Improved attention stability"
  putStrLn "  2. Post-layer norms: Better gradient flow"
  putStrLn "  3. Sliding window attention: Efficient long context"
  putStrLn "  4. Dual RoPE frequencies: Better position encoding"
  putStrLn "  5. Extreme GQA (4:1): Reduced memory, faster inference"
  putStrLn ""
  putStrLn "All features implemented with full backward compatibility!"
