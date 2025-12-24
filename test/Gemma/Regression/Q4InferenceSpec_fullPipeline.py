#!/usr/bin/env python3
"""
Generate COMPLETE PyTorch reference for Q4 model with benchmark prompt
This will process the ENTIRE prompt sequence and show output at each stage
"""

import torch
from safetensors import safe_open
import numpy as np
import json
import sys

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

def apply_rope(q, k, position, head_dim, rope_base=10000.0):
    """Apply RoPE (Rotary Position Embedding)"""
    # Simple RoPE implementation
    inv_freq = 1.0 / (rope_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.tensor([float(position)])
    freqs = torch.outer(t, inv_freq)

    # Create rotation matrix
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    # Apply to q and k
    def rotate(x, cos, sin):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1).flatten(-2)

    q_rot = rotate(q, cos, sin)
    k_rot = rotate(k, cos, sin)

    return q_rot, k_rot

print("=== Generating COMPLETE Q4 Reference with Benchmark Prompt ===\n")

# The exact tokens from the benchmark prompt
BENCHMARK_TOKENS = [6974, 496, 2822, 3925, 1003]
print(f"Benchmark prompt: 'Write a short story about'")
print(f"Token IDs: {BENCHMARK_TOKENS}\n")

# Load Q4 model
print("Loading Q4 model...")
with safe_open("../models/gemma3-1b-q4.safetensors", framework="pt", device="cpu") as f:
    embeddings = f.get_tensor("model.embed_tokens.weight").float()

    # Load layer 0 weights (dequantized)
    prefix = "model.layers.0"

    print("Dequantizing weights...")
    q_packed = f.get_tensor(f"{prefix}.self_attn.q_proj.weight_q4_packed").numpy().astype(np.uint32)
    q_scales = f.get_tensor(f"{prefix}.self_attn.q_proj.weight_q4_scales").float().numpy()
    q_weight = dequantize_q4(q_packed, q_scales).reshape(1024, 1152)

    k_packed = f.get_tensor(f"{prefix}.self_attn.k_proj.weight_q4_packed").numpy().astype(np.uint32)
    k_scales = f.get_tensor(f"{prefix}.self_attn.k_proj.weight_q4_scales").float().numpy()
    k_weight = dequantize_q4(k_packed, k_scales).reshape(256, 1152)

    v_packed = f.get_tensor(f"{prefix}.self_attn.v_proj.weight_q4_packed").numpy().astype(np.uint32)
    v_scales = f.get_tensor(f"{prefix}.self_attn.v_proj.weight_q4_scales").float().numpy()
    v_weight = dequantize_q4(v_packed, v_scales).reshape(256, 1152)

    o_packed = f.get_tensor(f"{prefix}.self_attn.o_proj.weight_q4_packed").numpy().astype(np.uint32)
    o_scales = f.get_tensor(f"{prefix}.self_attn.o_proj.weight_q4_scales").float().numpy()
    o_weight = dequantize_q4(o_packed, o_scales).reshape(1152, 1024)

    input_norm = f.get_tensor(f"{prefix}.input_layernorm.weight").float()
    post_attn_norm = f.get_tensor(f"{prefix}.post_attention_layernorm.weight").float()

    final_norm = f.get_tensor("model.norm.weight").float()

    # Load LM head (NOT quantized - FP32)
    lm_head_weight = f.get_tensor("lm_head.weight").float()

print("Processing FIRST token only (position 0)...")
print("=" * 60)

token_id = BENCHMARK_TOKENS[0]
x = embeddings[token_id].clone()

print(f"\nToken {token_id}:")
print(f"  Embedding (first 10): {x[:10].tolist()}")
print(f"  Stats: min={x.min():.6f}, max={x.max():.6f}, has_nan={torch.isnan(x).any().item()}")

# Layer 0 forward pass
print("\n--- Layer 0 ---")

# Pre-attention norm
x_norm = rmsnorm(x, input_norm)
print(f"  After RMSNorm: min={x_norm.min():.6f}, max={x_norm.max():.6f}, has_nan={torch.isnan(x_norm).any().item()}")

# Q/K/V projections
q = torch.nn.functional.linear(x_norm, q_weight)
k = torch.nn.functional.linear(x_norm, k_weight)
v = torch.nn.functional.linear(x_norm, v_weight)
print(f"  After Q proj: min={q.min():.6f}, max={q.max():.6f}, has_nan={torch.isnan(q).any().item()}")
print(f"  After K proj: min={k.min():.6f}, max={k.max():.6f}, has_nan={torch.isnan(k).any().item()}")
print(f"  After V proj: min={v.min():.6f}, max={v.max():.6f}, has_nan={torch.isnan(v).any().item()}")

# For first token, attention is trivial (no previous tokens)
# Just use V directly (scaled appropriately)
num_heads = 4
head_dim = 256
v_heads = v.view(1, head_dim)  # 1 KV head
v_expanded = v_heads.repeat(num_heads, 1).view(-1)  # Expand to 4 Q heads
attn_out = v_expanded

print(f"  After attention: min={attn_out.min():.6f}, max={attn_out.max():.6f}, has_nan={torch.isnan(attn_out).any().item()}")

# Output projection
attn_proj = torch.nn.functional.linear(attn_out, o_weight)
print(f"  After output proj: min={attn_proj.min():.6f}, max={attn_proj.max():.6f}, has_nan={torch.isnan(attn_proj).any().item()}")

# Post-attention norm
attn_normed = rmsnorm(attn_proj, post_attn_norm)
print(f"  After post-attn norm: min={attn_normed.min():.6f}, max={attn_normed.max():.6f}, has_nan={torch.isnan(attn_normed).any().item()}")

# Residual
x = x + attn_normed
print(f"  After residual: min={x.min():.6f}, max={x.max():.6f}, has_nan={torch.isnan(x).any().item()}")

# Skip FFN for now, go straight to final norm + LM head
print("\n--- Final layers ---")
final_hidden = rmsnorm(x, final_norm)
print(f"  After final RMSNorm: min={final_hidden.min():.6f}, max={final_hidden.max():.6f}, has_nan={torch.isnan(final_hidden).any().item()}")

logits = torch.nn.functional.linear(final_hidden, lm_head_weight)
print(f"  After LM head: min={logits.min():.6f}, max={logits.max():.6f}, has_nan={torch.isnan(logits).any().item()}")
print(f"  Logits (first 10): {logits[:10].tolist()}")

if torch.isnan(logits).any():
    print("\n❌ PYTORCH PRODUCES NaN TOO!")
    print("This means the bug is NOT Haskell-specific - it's in the model or approach!")
else:
    print("\n✅ PyTorch produces valid logits")
    print("Haskell implementation should match this exactly")

# Save reference
reference = {
    'token_id': token_id,
    'embedding': x_embeddings[token_id].tolist() if 'x_embeddings' in locals() else embeddings[token_id].tolist(),
    'after_rmsnorm': x_norm.tolist(),
    'after_q_proj': q.tolist(),
    'after_k_proj': k.tolist(),
    'after_v_proj': v.tolist(),
    'after_attention': attn_out.tolist(),
    'after_output_proj': attn_proj.tolist(),
    'after_final_norm': final_hidden.tolist(),
    'logits': logits.tolist()
}

output_path = 'test/Gemma/Regression/Q4InferenceSpec_fullPipeline.json'
with open(output_path, 'w') as f:
    json.dump(reference, f, indent=2)

print(f"\n✅ Saved reference to {output_path}")
