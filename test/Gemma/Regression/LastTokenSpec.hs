{-# LANGUAGE OverloadedStrings #-}

module Gemma.Regression.LastTokenSpec (spec) where

import Test.Hspec
import qualified Data.Vector.Storable as V
import Data.Vector.Storable (Vector)
import Graphics.WebGPU.Dawn.ContT (evalContT)
import Data.Aeson (FromJSON(..), withObject, (.:))
import Control.Monad (foldM)
import System.Environment (setEnv)

import Gemma.Model
import Gemma.TestCache (loadOrGenerateCache)
import qualified Gemma.Tensor as T

data LastTokenReference = LastTokenReference
  { ltrInputTokens :: [Int]
  , ltrLastPos :: Int
  , ltrEmbeddings :: EmbStats
  , ltrLayer0 :: LayerStats
  , ltrLayer1 :: LayerStats
  } deriving (Show)

data EmbStats = EmbStats
  { esMean :: Float
  , esStd :: Float
  , esFirst10 :: [Float]
  } deriving (Show)

data LayerStats = LayerStats
  { lsMean :: Float
  , lsStd :: Float
  , lsMin :: Float
  , lsMax :: Float
  , lsFirst10 :: [Float]
  } deriving (Show)

instance FromJSON LastTokenReference where
  parseJSON = withObject "LastTokenReference" $ \v -> LastTokenReference
    <$> v .: "input_tokens"
    <$> v .: "last_token_position"
    <$> v .: "embeddings"
    <$> v .: "layer_0"
    <$> v .: "layer_1"

instance FromJSON EmbStats where
  parseJSON = withObject "EmbStats" $ \v -> EmbStats
    <$> v .: "mean"
    <$> v .: "std"
    <$> v .: "first_10"

instance FromJSON LayerStats where
  parseJSON = withObject "LayerStats" $ \v -> LayerStats
    <$> v .: "mean"
    <$> v .: "std"
    <$> v .: "min"
    <$> v .: "max"
    <$> v .: "first_10"

spec :: Spec
spec = do
  describe "Last Token Layer-by-Layer (TDD Debug)" $ do

    it "embeddings match PyTorch for last token" $ do
      -- Load PyTorch reference
      ref <- loadOrGenerateCache
        "test/Gemma/Regression/FirstTokenSpec_lastToken.py"
        "test/Gemma/Regression/FirstTokenSpec_lastToken.json" :: IO LastTokenReference

      -- Disable debug output for cleaner test
      setEnv "DEBUG" "0"

      putStrLn $ "\n=== Last Token Embeddings Comparison ==="
      putStrLn $ "Processing " ++ show (length (ltrInputTokens ref)) ++ " tokens"
      putStrLn $ "Comparing last token at position " ++ show (ltrLastPos ref)

      -- Load model
      let modelPath = "../models/gemma3-1b-official-instruct/model.safetensors"
          config = gemma3_1BConfig

      putStrLn "\n📦 Loading model..."
      model <- loadGemmaModel modelPath config

      -- Process all tokens to build cache (like the test does)
      let inputTokens = V.fromList $ map fromIntegral $ ltrInputTokens ref
      putStrLn $ "🚀 Processing " ++ show (V.length inputTokens) ++ " tokens one-by-one..."

      -- Process each token, keeping track of embeddings for the LAST one
      (_, lastEmbed, _) <- foldM
        (\(mCache, _, _) (idx, tok) -> do
          let singleToken = V.singleton tok
          -- Get embeddings and cache for this token
          result <- evalContT $ do
            let cache = case mCache of
                  Just c -> c
                  Nothing -> initKVCache (gcNumLayers config) (gcNumKVHeads config) (gcHeadDim config) 2048
            let position = case mCache of
                  Nothing -> 0
                  Just c -> cacheLength (kvLayers c BV.! 0)

            -- Get raw embedding
            rawEmbed <- runEmbeddingGPU (gmContext model) singleToken (gmEmbeddingTensor model) (gmEmbeddingShader model) False (gcHiddenDim config)
            liftIO $ T.waitAll (gmContext model)
            rawEmbedCPU <- liftIO $ T.fromGPU (gmContext model) rawEmbed (gcHiddenDim config)

            -- Scale embedding (Gemma 3 requirement)
            let embeddingScale = sqrt (fromIntegral (gcHiddenDim config) :: Float)
                scaledEmbed = V.map (* embeddingScale) rawEmbedCPU

            pure (cache, scaledEmbed)

          -- Run full inference to update cache
          (_, cache') <- runGemmaInferenceCached model singleToken mCache

          -- Return updated cache and embedding (keep last one)
          pure (Just cache', result, idx == V.length inputTokens - 1)
        )
        (Nothing, V.empty, False)
        (zip [0..] (V.toList inputTokens))

      putStrLn $ "\n✅ Got last token embeddings: " ++ show (V.length lastEmbed) ++ " values"

      -- Compare with PyTorch
      let pyEmbed = ltrEmbeddings ref
          pyMean = esMean pyEmbed
          pyStd = esStd pyEmbed
          pyFirst10 = esFirst10 pyEmbed

          hsMean = V.sum lastEmbed / fromIntegral (V.length lastEmbed)
          hsStd = sqrt $ V.sum (V.map (\x -> (x - hsMean)^2) lastEmbed) / fromIntegral (V.length lastEmbed)
          hsFirst10 = V.toList $ V.take 10 lastEmbed

      putStrLn $ "\nPyTorch embeddings:"
      putStrLn $ "  Mean: " ++ show pyMean ++ ", Std: " ++ show pyStd
      putStrLn $ "  First 10: " ++ show pyFirst10

      putStrLn $ "\nHaskell embeddings:"
      putStrLn $ "  Mean: " ++ show hsMean ++ ", Std: " ++ show hsStd
      putStrLn $ "  First 10: " ++ show hsFirst10

      -- Check if they match within tolerance
      let meanDiff = abs (hsMean - pyMean)
          stdDiff = abs (hsStd - pyStd)
          first10Diff = maximum $ zipWith (\a b -> abs (a - b)) hsFirst10 pyFirst10

      putStrLn $ "\nDifferences:"
      putStrLn $ "  Mean diff: " ++ show meanDiff
      putStrLn $ "  Std diff: " ++ show stdDiff
      putStrLn $ "  Max first-10 diff: " ++ show first10Diff

      -- Embeddings should match within 1e-6 tolerance
      meanDiff `shouldSatisfy` (< 1e-5)
      stdDiff `shouldSatisfy` (< 1e-5)
      first10Diff `shouldSatisfy` (< 1e-5)
