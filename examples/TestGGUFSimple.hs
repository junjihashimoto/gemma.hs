{-# LANGUAGE OverloadedStrings #-}

module Main where

import qualified Data.ByteString as BS
import Foreign.Ptr (Ptr, castPtr, plusPtr)
import Foreign.Storable (peekElemOff, peek)
import Data.Word

main :: IO ()
main = do
  let path = "/Users/junji.hashimoto/git/dawn/models/gemma-3-1b-it-q4_0.gguf"
  putStrLn $ "Reading GGUF file: " ++ path

  -- Read just the header (first 256 bytes)
  fileData <- BS.readFile path
  putStrLn $ "File size: " ++ show (BS.length fileData) ++ " bytes"

  -- Parse header manually
  BS.useAsCString fileData $ \ptr -> do
    let p = castPtr ptr :: Ptr Word32
    magic <- peekElemOff p 0
    version <- peekElemOff p 1

    let p64 = castPtr (plusPtr ptr 8) :: Ptr Word64
    tensorCount <- peekElemOff p64 0
    metadataKVCount <- peekElemOff p64 1

    putStrLn $ "Magic: 0x" ++ showHex magic ""
    putStrLn $ "Version: " ++ show version
    putStrLn $ "Tensor count: " ++ show tensorCount
    putStrLn $ "Metadata KV count: " ++ show metadataKVCount

    putStrLn "✅ Header parsed successfully!"

showHex :: Word32 -> String -> String
showHex 0 acc = if null acc then "0" else acc
showHex n acc =
  let digit = n `mod` 16
      c = if digit < 10 then toEnum (fromEnum '0' + fromIntegral digit)
                        else toEnum (fromEnum 'A' + fromIntegral (digit - 10))
  in showHex (n `div` 16) (c : acc)
