#!/usr/bin/env python3
"""
Generate PyTorch reference for the FAILING benchmark input
This uses the exact same prompt tokens as the benchmark: [6974, 496, 2822, 3925, 1003]
"""

import torch
from safetensors import safe_open
import numpy as np
import json

def dequantize_q4(packed, scales):
    """Dequantize Q4 weights"""
    num_scales = len(scales)
    num_weights = num_scales * 32
    result = np.zeros(num_weights, dtype=np.float32)

    for block_idx in range(num_scales):
        scale = scales[block_idx]
        for weight_in_block in range(32):
            weight_idx = block_idx * 32 + weight_in_block
            word_idx = weight_in_block // 8
            nibble_idx = weight_in_block % 8

            packed_idx = block_idx * 4 + word_idx
            packed_word = packed[packed_idx]
            nibble = (packed_word >> (nibble_idx * 4)) & 0xF

            result[weight_idx] = (float(nibble) - 7.5) * scale

    return torch.from_numpy(result)

def rmsnorm(x, weight, eps=1e-6):
    """Standard RMSNorm"""
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return x_normed * weight

print("=== Generating Q4 Reference for FAILING Benchmark Input ===\n")

# The exact tokens from the benchmark prompt "Write a short story about"
BENCHMARK_TOKENS = [6974, 496, 2822, 3925, 1003]
print(f"Benchmark prompt tokens: {BENCHMARK_TOKENS}")
print(f"First token (will test): {BENCHMARK_TOKENS[0]}\n")

# Load Q4 model
print("Loading Q4 model...")
with safe_open("../models/gemma3-1b-q4.safetensors", framework="pt", device="cpu") as f:
    # Load embeddings (FP32)
    embeddings = f.get_tensor("model.embed_tokens.weight").float()

    # Load layer 0 weights
    prefix = "model.layers.0"

    # Dequantize Q4 weights
    print("Dequantizing Q4 weights...")
    q_packed = f.get_tensor(f"{prefix}.self_attn.q_proj.weight_q4_packed").numpy().astype(np.uint32)
    q_scales = f.get_tensor(f"{prefix}.self_attn.q_proj.weight_q4_scales").float().numpy()
    q_weight = dequantize_q4(q_packed, q_scales).reshape(1024, 1152)

    k_packed = f.get_tensor(f"{prefix}.self_attn.k_proj.weight_q4_packed").numpy().astype(np.uint32)
    k_scales = f.get_tensor(f"{prefix}.self_attn.k_proj.weight_q4_scales").float().numpy()
    k_weight = dequantize_q4(k_packed, k_scales).reshape(256, 1152)

    v_packed = f.get_tensor(f"{prefix}.self_attn.v_proj.weight_q4_packed").numpy().astype(np.uint32)
    v_scales = f.get_tensor(f"{prefix}.self_attn.v_proj.weight_q4_scales").float().numpy()
    v_weight = dequantize_q4(v_packed, v_scales).reshape(256, 1152)

    # Load norm weights (FP32)
    input_norm = f.get_tensor(f"{prefix}.input_layernorm.weight").float()

print("Running layer 0 forward pass for FIRST TOKEN (6974)...\n")

# Get embedding for token 6974 (first token of failing benchmark)
token_id = BENCHMARK_TOKENS[0]
x = embeddings[token_id].clone()
print(f"Step 0: Input embedding (token {token_id})")
print(f"  First 10: {x[:10].tolist()}")
print(f"  Stats: min={x.min():.6f}, max={x.max():.6f}")

# Check for NaN/Inf in embedding
if torch.isnan(x).any() or torch.isinf(x).any():
    print("  ⚠️  WARNING: NaN/Inf in embedding!")
else:
    print("  ✅ No NaN/Inf in embedding")

# Store intermediate values
reference = {
    'input': {
        'token_id': token_id,
        'benchmark_prompt': 'Write a short story about',
        'all_tokens': BENCHMARK_TOKENS,
        'embedding': x.tolist(),
        'stats': {'min': float(x.min()), 'max': float(x.max())}
    },
    'steps': []
}

# Step 1: Pre-attention norm
x_norm = rmsnorm(x, input_norm, eps=1e-6)
print(f"\nStep 1: After input RMSNorm")
print(f"  First 10: {x_norm[:10].tolist()}")
print(f"  Stats: min={x_norm.min():.6f}, max={x_norm.max():.6f}")
if torch.isnan(x_norm).any() or torch.isinf(x_norm).any():
    print("  ⚠️  WARNING: NaN/Inf after RMSNorm!")
else:
    print("  ✅ No NaN/Inf")

reference['steps'].append({
    'name': 'input_rmsnorm',
    'output': x_norm.tolist(),
    'stats': {'min': float(x_norm.min()), 'max': float(x_norm.max())}
})

# Step 2: Q/K/V projections
q = torch.nn.functional.linear(x_norm, q_weight)
k = torch.nn.functional.linear(x_norm, k_weight)
v = torch.nn.functional.linear(x_norm, v_weight)
print(f"\nStep 2: Q/K/V projections")
print(f"  Q first 10: {q[:10].tolist()}")
print(f"  Q stats: min={q.min():.6f}, max={q.max():.6f}")
print(f"  K first 10: {k[:10].tolist()}")
print(f"  K stats: min={k.min():.6f}, max={k.max():.6f}")
print(f"  V first 10: {v[:10].tolist()}")
print(f"  V stats: min={v.min():.6f}, max={v.max():.6f}")

if torch.isnan(q).any() or torch.isinf(q).any():
    print("  ⚠️  WARNING: NaN/Inf in Q!")
if torch.isnan(k).any() or torch.isinf(k).any():
    print("  ⚠️  WARNING: NaN/Inf in K!")
if torch.isnan(v).any() or torch.isinf(v).any():
    print("  ⚠️  WARNING: NaN/Inf in V!")
if not (torch.isnan(q).any() or torch.isnan(k).any() or torch.isnan(v).any()):
    print("  ✅ No NaN/Inf in Q/K/V")

reference['steps'].append({
    'name': 'qkv_projection',
    'q': q.tolist(),
    'k': k.tolist(),
    'v': v.tolist(),
    'q_stats': {'min': float(q.min()), 'max': float(q.max())},
    'k_stats': {'min': float(k.min()), 'max': float(k.max())},
    'v_stats': {'min': float(v.min()), 'max': float(v.max())}
})

# Save reference
output_path = "test/Gemma/Regression/Q4InferenceSpec_failingToken.json"
with open(output_path, 'w') as f:
    json.dump(reference, f, indent=2)

print(f"\n✅ Saved Q4 failing token reference to {output_path}")
print(f"   Token ID: {token_id}")
print(f"   Total steps: {len(reference['steps'])}")
print("\nUse this to test if Haskell GPU produces NaN with the SAME input as benchmark!")
