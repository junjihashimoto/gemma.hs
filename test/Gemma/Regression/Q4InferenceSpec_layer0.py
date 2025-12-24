#!/usr/bin/env python3
"""
Generate PyTorch reference for Q4 model inference (Layer 0 only)
This will be used for TDD to find the Haskell Q4 bug
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

print("=== Generating Q4 Inference Reference ===\n")

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

    o_packed = f.get_tensor(f"{prefix}.self_attn.o_proj.weight_q4_packed").numpy().astype(np.uint32)
    o_scales = f.get_tensor(f"{prefix}.self_attn.o_proj.weight_q4_scales").float().numpy()
    o_weight = dequantize_q4(o_packed, o_scales).reshape(1152, 1024)

    gate_packed = f.get_tensor(f"{prefix}.mlp.gate_proj.weight_q4_packed").numpy().astype(np.uint32)
    gate_scales = f.get_tensor(f"{prefix}.mlp.gate_proj.weight_q4_scales").float().numpy()
    gate_weight = dequantize_q4(gate_packed, gate_scales).reshape(6912, 1152)

    up_packed = f.get_tensor(f"{prefix}.mlp.up_proj.weight_q4_packed").numpy().astype(np.uint32)
    up_scales = f.get_tensor(f"{prefix}.mlp.up_proj.weight_q4_scales").float().numpy()
    up_weight = dequantize_q4(up_packed, up_scales).reshape(6912, 1152)

    down_packed = f.get_tensor(f"{prefix}.mlp.down_proj.weight_q4_packed").numpy().astype(np.uint32)
    down_scales = f.get_tensor(f"{prefix}.mlp.down_proj.weight_q4_scales").float().numpy()
    down_weight = dequantize_q4(down_packed, down_scales).reshape(1152, 6912)

    # Load norm weights (FP32)
    input_norm = f.get_tensor(f"{prefix}.input_layernorm.weight").float()
    post_attn_norm = f.get_tensor(f"{prefix}.post_attention_layernorm.weight").float()
    post_ffn_norm = f.get_tensor(f"{prefix}.post_feedforward_layernorm.weight").float()

print("Running layer 0 forward pass...\n")

# Get embedding for token 1
x = embeddings[1].clone()
print(f"Step 0: Input embedding (token 1)")
print(f"  First 10: {x[:10].tolist()}")
print(f"  Stats: min={x.min():.6f}, max={x.max():.6f}")

# Store intermediate values
reference = {
    'input': {
        'token_id': 1,
        'embedding': x.tolist(),
        'stats': {'min': float(x.min()), 'max': float(x.max())}
    },
    'steps': []
}

# Step 1: Pre-attention norm
x_norm = rmsnorm(x, input_norm, eps=1e-6)
print(f"\nStep 1: After input RMSNorm")
print(f"  First 10: {x_norm[:10].tolist()}")
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
print(f"  K first 10: {k[:10].tolist()}")
print(f"  V first 10: {v[:10].tolist()}")
reference['steps'].append({
    'name': 'qkv_projection',
    'q': q.tolist(),
    'k': k.tolist(),
    'v': v.tolist(),
    'q_stats': {'min': float(q.min()), 'max': float(q.max())},
    'k_stats': {'min': float(k.min()), 'max': float(k.max())},
    'v_stats': {'min': float(v.min()), 'max': float(v.max())}
})

# Step 3: Attention (single token - just use V)
num_heads = 4
num_kv_heads = 1
head_dim = 256
heads_per_kv = num_heads // num_kv_heads

v_heads = v.view(num_kv_heads, head_dim)
v_expanded = v_heads.repeat_interleave(heads_per_kv, dim=0)
attn_out = v_expanded.view(-1)
print(f"\nStep 3: Attention output")
print(f"  First 10: {attn_out[:10].tolist()}")
reference['steps'].append({
    'name': 'attention',
    'output': attn_out.tolist(),
    'stats': {'min': float(attn_out.min()), 'max': float(attn_out.max())}
})

# Step 4: Output projection
attn_proj = torch.nn.functional.linear(attn_out, o_weight)
print(f"\nStep 4: After output projection")
print(f"  First 10: {attn_proj[:10].tolist()}")
reference['steps'].append({
    'name': 'output_projection',
    'output': attn_proj.tolist(),
    'stats': {'min': float(attn_proj.min()), 'max': float(attn_proj.max())}
})

# Step 5: Post-attention norm
attn_normed = rmsnorm(attn_proj, post_attn_norm, eps=1e-6)
print(f"\nStep 5: After post-attention norm")
print(f"  First 10: {attn_normed[:10].tolist()}")
reference['steps'].append({
    'name': 'post_attention_norm',
    'output': attn_normed.tolist(),
    'stats': {'min': float(attn_normed.min()), 'max': float(attn_normed.max())}
})

# Step 6: Residual
x = x + attn_normed
print(f"\nStep 6: After attention residual")
print(f"  First 10: {x[:10].tolist()}")
reference['steps'].append({
    'name': 'attention_residual',
    'output': x.tolist(),
    'stats': {'min': float(x.min()), 'max': float(x.max())}
})

# Step 7: Pre-FFN norm (using same post_attn_norm for Gemma 3)
x_ffn_norm = rmsnorm(x, post_attn_norm, eps=1e-6)
print(f"\nStep 7: After pre-FFN norm")
print(f"  First 10: {x_ffn_norm[:10].tolist()}")
reference['steps'].append({
    'name': 'pre_ffn_norm',
    'output': x_ffn_norm.tolist(),
    'stats': {'min': float(x_ffn_norm.min()), 'max': float(x_ffn_norm.max())}
})

# Step 8: FFN
gate = torch.nn.functional.linear(x_ffn_norm, gate_weight)
up = torch.nn.functional.linear(x_ffn_norm, up_weight)
gate_gelu = torch.nn.functional.gelu(gate, approximate='tanh')
hidden = gate_gelu * up
ffn_out = torch.nn.functional.linear(hidden, down_weight)
print(f"\nStep 8: FFN")
print(f"  Gate first 10: {gate[:10].tolist()}")
print(f"  Up first 10: {up[:10].tolist()}")
print(f"  FFN out first 10: {ffn_out[:10].tolist()}")
reference['steps'].append({
    'name': 'ffn',
    'gate': gate.tolist(),
    'up': up.tolist(),
    'ffn_output': ffn_out.tolist(),
    'stats': {'min': float(ffn_out.min()), 'max': float(ffn_out.max())}
})

# Step 9: Post-FFN norm
ffn_normed = rmsnorm(ffn_out, post_ffn_norm, eps=1e-6)
print(f"\nStep 9: After post-FFN norm")
print(f"  First 10: {ffn_normed[:10].tolist()}")
reference['steps'].append({
    'name': 'post_ffn_norm',
    'output': ffn_normed.tolist(),
    'stats': {'min': float(ffn_normed.min()), 'max': float(ffn_normed.max())}
})

# Step 10: Final residual
output = x + ffn_normed
print(f"\nStep 10: Final output (layer 0)")
print(f"  First 10: {output[:10].tolist()}")
print(f"  Stats: min={output.min():.6f}, max={output.max():.6f}")
reference['steps'].append({
    'name': 'final_output',
    'output': output.tolist(),
    'stats': {'min': float(output.min()), 'max': float(output.max())}
})

# Save reference
output_path = "test/Gemma/Regression/Q4InferenceSpec_layer0.json"
with open(output_path, 'w') as f:
    json.dump(reference, f, indent=2)

print(f"\n✅ Saved Q4 inference reference to {output_path}")
print(f"   Total steps: {len(reference['steps'])}")
