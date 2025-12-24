#!/usr/bin/env python3
"""
Generate FP32 single token inference reference from PyTorch Gemma model.

This script generates reference output for the EXACT token that gemma-cli uses
in the benchmark prompt "Write a short story about a cat".
"""

import torch
from safetensors import safe_open
import numpy as np
import json
import sys

def load_fp32_model(model_path):
    """Load FP32 Gemma model weights"""
    weights = {}
    with safe_open(model_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key).float()
    return weights

def rmsnorm(x, weight, eps=1e-6):
    """Standard RMSNorm"""
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return x_normed * weight

def apply_rope(q, k, position, head_dim, rope_base=10000.0):
    """Apply RoPE (Rotary Position Embedding)"""
    inv_freq = 1.0 / (rope_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.tensor([float(position)])
    freqs = torch.outer(t, inv_freq)

    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    def rotate(x, cos, sin):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1).flatten(-2)

    q_rot = rotate(q, cos, sin)
    k_rot = rotate(k, cos, sin)

    return q_rot, k_rot

print("=== Generating FP32 Reference for Single Token ===\n")

# The first token from "Write a short story about a cat"
# After chat template: [2, 2, 106, 1645, ...]
# We'll use token 6974 which is "Write" (same as Q4 benchmark)
TOKEN_ID = 6974
print(f"Token ID: {TOKEN_ID}")
print(f"Token text: 'Write'")
print()

# Load FP32 model (original Gemma 3 that Q4 was derived from)
print("Loading Gemma 3 FP32 model...")
model_path = "../models/gemma3-1b.safetensors"  # From /Users/junji.hashimoto/git/dawn/gemma.hs
weights = load_fp32_model(model_path)
print(f"Loaded {len(weights)} weight tensors")

# Model dimensions
hidden_dim = 1152
num_heads = 4
num_kv_heads = 1
head_dim = 256
ffn_dim = 6912

# Get embedding
print(f"\nProcessing token {TOKEN_ID}...")
embeddings = weights["model.embed_tokens.weight"]
x = embeddings[TOKEN_ID].clone()

print(f"  Embedding shape: {x.shape}")
print(f"  Embedding (first 10): {x[:10].tolist()}")
print(f"  Stats: min={x.min():.6f}, max={x.max():.6f}, mean={x.mean():.6f}")

# Run through ALL 26 layers (Gemma 3 1B has 26 transformer layers)
print("\n--- Running through all 26 transformer layers ---")
for layer_idx in range(26):
    prefix = f"model.layers.{layer_idx}"

    if layer_idx == 0 or layer_idx == 25:  # Print details for first and last layer
        print(f"\n--- Layer {layer_idx} ---")

    # Pre-attention norm
    input_norm = weights[f"{prefix}.input_layernorm.weight"]
    x_norm = rmsnorm(x, input_norm)
    if layer_idx == 0:
        print(f"  After input RMSNorm: min={x_norm.min():.6f}, max={x_norm.max():.6f}")

    # Q/K/V projections
    q_weight = weights[f"{prefix}.self_attn.q_proj.weight"]
    k_weight = weights[f"{prefix}.self_attn.k_proj.weight"]
    v_weight = weights[f"{prefix}.self_attn.v_proj.weight"]

    q = torch.nn.functional.linear(x_norm, q_weight)
    k = torch.nn.functional.linear(x_norm, k_weight)
    v = torch.nn.functional.linear(x_norm, v_weight)

    if layer_idx == 0:
        print(f"  After Q proj: shape={q.shape}, min={q.min():.6f}, max={q.max():.6f}")
        print(f"  After K proj: shape={k.shape}, min={k.min():.6f}, max={k.max():.6f}")
        print(f"  After V proj: shape={v.shape}, min={v.min():.6f}, max={v.max():.6f}")

    # QK-Norm (Gemma 3 specific) - skip if not present
    if f"{prefix}.self_attn.q_norm.weight" in weights:
        q_norm_weight = weights[f"{prefix}.self_attn.q_norm.weight"]
        k_norm_weight = weights[f"{prefix}.self_attn.k_norm.weight"]

        q_heads = q.view(num_heads, head_dim)
        k_heads = k.view(num_kv_heads, head_dim)

        q_normed = torch.zeros_like(q_heads)
        for h in range(num_heads):
            q_normed[h] = rmsnorm(q_heads[h], q_norm_weight)

        k_normed = torch.zeros_like(k_heads)
        for h in range(num_kv_heads):
            k_normed[h] = rmsnorm(k_heads[h], k_norm_weight)

        q = q_normed.view(-1)
        k = k_normed.view(-1)

        if layer_idx == 0:
            print(f"  After QK-Norm Q: min={q.min():.6f}, max={q.max():.6f}")
            print(f"  After QK-Norm K: min={k.min():.6f}, max={k.max():.6f}")
    elif layer_idx == 0:
        print(f"  QK-Norm: Not present (older Gemma model)")

    # Apply RoPE (head-wise)
    q_heads = q.view(num_heads, head_dim)
    k_heads = k.view(num_kv_heads, head_dim)

    q_rot_heads = []
    for h in range(num_heads):
        q_h, _ = apply_rope(q_heads[h], q_heads[h], position=0, head_dim=head_dim)
        q_rot_heads.append(q_h)
    q_rot = torch.cat(q_rot_heads)

    k_rot_heads = []
    for h in range(num_kv_heads):
        _, k_h = apply_rope(k_heads[h], k_heads[h], position=0, head_dim=head_dim)
        k_rot_heads.append(k_h)
    k_rot = torch.cat(k_rot_heads)

    if layer_idx == 0:
        print(f"  After RoPE Q: min={q_rot.min():.6f}, max={q_rot.max():.6f}")
        print(f"  After RoPE K: min={k_rot.min():.6f}, max={k_rot.max():.6f}")

    # For first token, attention is simple (no previous tokens)
    # Attention output = V (scaled appropriately)
    v_heads = v.view(num_kv_heads, head_dim)
    v_expanded = v_heads.repeat(num_heads, 1).view(-1)
    attn_out = v_expanded

    if layer_idx == 0:
        print(f"  After attention: min={attn_out.min():.6f}, max={attn_out.max():.6f}")

    # Output projection
    o_weight = weights[f"{prefix}.self_attn.o_proj.weight"]
    attn_proj = torch.nn.functional.linear(attn_out, o_weight)
    if layer_idx == 0:
        print(f"  After output proj: min={attn_proj.min():.6f}, max={attn_proj.max():.6f}")

    # Residual
    x = x + attn_proj
    if layer_idx == 0:
        print(f"  After first residual: min={x.min():.6f}, max={x.max():.6f}")

    # Post-attention norm (Gemma 3) - use appropriate key
    post_attn_key = f"{prefix}.post_attention_layernorm.weight" if f"{prefix}.post_attention_layernorm.weight" in weights else f"{prefix}.post_attention_norm.weight"
    post_attn_norm = weights[post_attn_key]
    x_norm2 = rmsnorm(x, post_attn_norm)
    if layer_idx == 0:
        print(f"  After post-attn RMSNorm: min={x_norm2.min():.6f}, max={x_norm2.max():.6f}")

    # FFN
    gate_weight = weights[f"{prefix}.mlp.gate_proj.weight"]
    up_weight = weights[f"{prefix}.mlp.up_proj.weight"]
    down_weight = weights[f"{prefix}.mlp.down_proj.weight"]

    gate = torch.nn.functional.linear(x_norm2, gate_weight)
    up = torch.nn.functional.linear(x_norm2, up_weight)

    # GELU activation
    gelu_gate = torch.nn.functional.gelu(gate, approximate='tanh')
    ffn_hidden = gelu_gate * up

    if layer_idx == 0:
        print(f"  After gate: min={gate.min():.6f}, max={gate.max():.6f}")
        print(f"  After GELU: min={gelu_gate.min():.6f}, max={gelu_gate.max():.6f}")
        print(f"  After up: min={up.min():.6f}, max={up.max():.6f}")
        print(f"  After gelu*up: min={ffn_hidden.min():.6f}, max={ffn_hidden.max():.6f}")

    # Down projection
    ffn_out = torch.nn.functional.linear(ffn_hidden, down_weight)
    if layer_idx == 0:
        print(f"  After down proj: min={ffn_out.min():.6f}, max={ffn_out.max():.6f}")

    # Second residual
    x = x + ffn_out
    if layer_idx == 0 or layer_idx == 25:
        print(f"  After second residual: min={x.min():.6f}, max={x.max():.6f}")

# Final norm
final_norm = weights["model.norm.weight"]
final_hidden = rmsnorm(x, final_norm)
print(f"  After final RMSNorm: min={final_hidden.min():.6f}, max={final_hidden.max():.6f}")

# LM head (use tied weights - embeddings)
# Gemma 3 typically ties embedding and output weights
if "lm_head.weight" in weights:
    lm_head = weights["lm_head.weight"]
else:
    # Use tied weights (embeddings)
    lm_head = embeddings
    print(f"  Using tied weights (embedding table) for LM head")

logits = torch.nn.functional.linear(final_hidden, lm_head)
print(f"\n  Final logits: shape={logits.shape}, min={logits.min():.6f}, max={logits.max():.6f}")
print(f"  Logits (first 10): {logits[:10].tolist()}")
print(f"  Has NaN: {torch.isnan(logits).any().item()}")

# Get top 5 tokens
top5_logits, top5_indices = torch.topk(logits, 5)
print(f"\n  Top 5 tokens:")
for i in range(5):
    print(f"    Token {top5_indices[i].item()}: logit={top5_logits[i].item():.6f}")

# Save reference
reference = {
    'token_id': TOKEN_ID,
    'embedding': embeddings[TOKEN_ID].tolist(),
    'after_input_norm': x_norm.tolist(),
    'after_q_proj': q.tolist() if isinstance(q, torch.Tensor) else q_rot.tolist(),
    'after_k_proj': k.tolist() if isinstance(k, torch.Tensor) else k_rot.tolist(),
    'after_v_proj': v.tolist(),
    'after_rope_q': q_rot.tolist(),
    'after_rope_k': k_rot.tolist(),
    'after_attention': attn_out.tolist(),
    'after_output_proj': attn_proj.tolist(),
    'after_first_residual': x.tolist(),
    'after_post_attn_norm': x_norm2.tolist(),
    'after_gate': gate.tolist(),
    'after_gelu': gelu_gate.tolist(),
    'after_up': up.tolist(),
    'after_gelu_mul_up': ffn_hidden.tolist(),
    'after_down_proj': ffn_out.tolist(),
    'after_second_residual': x.tolist(),
    'after_final_norm': final_hidden.tolist(),
    'logits': logits.tolist(),
    'top5_tokens': top5_indices.tolist(),
    'top5_logits': top5_logits.tolist()
}

output_path = 'FP32InferenceSpec_singleToken.json'
with open(output_path, 'w') as f:
    json.dump(reference, f, indent=2)

print(f"\n✅ Saved FP32 reference to {output_path}")
