#!/usr/bin/env python3
"""
Generate Q4 quantization reference from PyTorch Gemma model.

This script:
1. Loads a weight matrix from the PyTorch Gemma model
2. Quantizes it using PyTorch's approach (if available)
3. Saves reference data for Haskell to compare against
"""

import torch
import numpy as np
import json
import sys
from pathlib import Path

def load_gemma_weights(model_path, layer_name="model.layers.0.self_attn.q_proj.weight"):
    """Load a weight tensor from Gemma SafeTensors"""
    from safetensors import safe_open

    with safe_open(model_path, framework="pt", device="cpu") as f:
        tensor = f.get_tensor(layer_name)

    return tensor

def quantize_q4_symmetric(weights, block_size=32):
    """
    Symmetric Q4 quantization (range: -7.5 to 7.5 with 4-bit integers)

    This matches the Haskell implementation:
    - Block size: 32 elements per scale
    - Range: 4-bit unsigned (0-15) mapped to [-7.5, 7.5]
    - Formula: quantized = round((weight / scale) + 7.5)
              dequantized = (quantized - 7.5) * scale
    """
    if isinstance(weights, np.ndarray):
        weights_np = weights.flatten()
    else:
        weights_np = weights.cpu().numpy().flatten()
    n = len(weights_np)

    # Ensure weights are multiple of block_size
    if n % block_size != 0:
        raise ValueError(f"Weight count {n} must be multiple of block_size {block_size}")

    num_blocks = n // block_size
    quantized = np.zeros(n, dtype=np.uint8)
    scales = np.zeros(num_blocks, dtype=np.float32)

    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = start + block_size
        block = weights_np[start:end]

        # Compute scale: max absolute value in block divided by 7.5
        # This maps the range [-max_abs, max_abs] to [-7.5, 7.5]
        max_abs = np.max(np.abs(block))
        scale = max_abs / 7.5 if max_abs > 0 else 1.0
        scales[block_idx] = scale

        # Quantize: map to [0, 15]
        normalized = block / scale  # Now in range [-7.5, 7.5]
        shifted = normalized + 7.5   # Now in range [0, 15]
        quantized_block = np.clip(np.round(shifted), 0, 15).astype(np.uint8)
        quantized[start:end] = quantized_block

    return quantized, scales

def dequantize_q4_symmetric(quantized, scales, block_size=32):
    """Dequantize Q4 data back to FP32"""
    n = len(quantized)
    num_blocks = len(scales)
    dequantized = np.zeros(n, dtype=np.float32)

    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = start + block_size
        scale = scales[block_idx]

        # Dequantize: (quantized - 7.5) * scale
        block_quant = quantized[start:end].astype(np.float32)
        dequantized[start:end] = (block_quant - 7.5) * scale

    return dequantized

def pack_nibbles(quantized):
    """Pack 8 x 4-bit values into Word32 (matches Haskell implementation)"""
    n = len(quantized)
    if n % 8 != 0:
        raise ValueError(f"Length {n} must be multiple of 8 for packing")

    packed = []
    for i in range(0, n, 8):
        # Pack 8 nibbles into one 32-bit word
        # Nibble 0 at bits 0-3, nibble 1 at bits 4-7, etc.
        word = 0
        for j in range(8):
            nibble = int(quantized[i + j]) & 0xF
            word |= (nibble << (j * 4))
        packed.append(word)

    return np.array(packed, dtype=np.uint32)

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_q4_reference.py <model_path> [layer_name]")
        print("Example: python generate_q4_reference.py ../models/gemma3-1b.safetensors")
        sys.exit(1)

    model_path = sys.argv[1]
    layer_name = sys.argv[2] if len(sys.argv) > 2 else "model.layers.0.self_attn.q_proj.weight"

    print(f"Loading weights from {model_path}")
    print(f"Layer: {layer_name}")

    # Load weights
    weights = load_gemma_weights(model_path, layer_name)
    print(f"Weight shape: {weights.shape}")
    print(f"Weight dtype: {weights.dtype}")

    # Convert to FP32 if needed
    if weights.dtype == torch.float16:
        weights_fp32 = weights.float()
        print("Converted FP16 to FP32 for quantization")
    else:
        weights_fp32 = weights

    # Flatten to 1D for quantization
    original_shape = weights.shape
    weights_flat = weights_fp32.reshape(-1)

    print(f"\nQuantizing {len(weights_flat)} weights with Q4 symmetric...")

    # Quantize
    quantized, scales = quantize_q4_symmetric(weights_flat.cpu().numpy())

    print(f"Quantized to {len(quantized)} bytes")
    print(f"Scales: {len(scales)} values")

    # Pack into nibbles
    packed = pack_nibbles(quantized)
    print(f"Packed to {len(packed)} Word32 values")

    # Dequantize for validation
    dequantized = dequantize_q4_symmetric(quantized, scales)

    # Compute error
    original_np = weights_flat.cpu().numpy()
    max_diff = np.max(np.abs(original_np - dequantized))
    mean_diff = np.mean(np.abs(original_np - dequantized))
    rel_error = mean_diff / (np.mean(np.abs(original_np)) + 1e-8)

    print(f"\nQuantization Error:")
    print(f"  Max absolute diff: {max_diff:.6f}")
    print(f"  Mean absolute diff: {mean_diff:.6f}")
    print(f"  Relative error: {rel_error*100:.4f}%")

    # Save reference data
    output = {
        "model_path": model_path,
        "layer_name": layer_name,
        "original_shape": list(original_shape),
        "block_size": 32,
        "num_elements": int(len(weights_flat)),
        "num_packed": int(len(packed)),
        "num_scales": int(len(scales)),

        # Statistics
        "original_stats": {
            "min": float(np.min(original_np)),
            "max": float(np.max(original_np)),
            "mean": float(np.mean(original_np)),
            "std": float(np.std(original_np))
        },

        # Quantization error
        "quantization_error": {
            "max_diff": float(max_diff),
            "mean_diff": float(mean_diff),
            "relative_error": float(rel_error)
        },

        # Sample data for validation (first 256 elements)
        "original_sample": original_np[:256].tolist(),
        "quantized_sample": quantized[:256].tolist(),
        "scales_sample": scales[:8].tolist(),  # First 8 scales
        "packed_sample": packed[:32].tolist(),  # First 32 packed words
        "dequantized_sample": dequantized[:256].tolist()
    }

    output_path = "test/Gemma/Regression/Q4InferenceSpec_quantization.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Reference data saved to {output_path}")
    print("\nYou can now validate the Haskell Q4 implementation against this reference.")

if __name__ == "__main__":
    main()
