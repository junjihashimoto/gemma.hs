#!/usr/bin/env python3
"""
PyTorch reference for 17-token prompt: "What is 2+2?"
Generates expected prediction after full prompt processing with KV cache.
"""

import torch
import torch.nn.functional as F
from safetensors import safe_open
import json

def load_fp32_model(path):
    """Load FP32 weights from SafeTensors file."""
    weights = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    return weights

def rmsnorm(x, weight, eps=1e-6, zero_centered=False):
    """RMSNorm with optional zero-centering.

    Args:
        x: Input tensor
        weight: Normalization weight
        eps: Epsilon for numerical stability
        zero_centered: If True, apply (1 + weight) * normalized (Gemma 3)
                      If False, apply weight * normalized (standard)
    """
    rms = torch.sqrt(torch.mean(x * x) + eps)
    normalized = x / rms
    if zero_centered:
        return normalized * (1.0 + weight)
    else:
        return normalized * weight

def apply_rope(q, k, position, head_dim=256, rope_base=1000000.0):
    """Apply Rotary Position Embedding.
    
    Args:
        q: Query tensor, shape [num_q_heads * head_dim] = [1024] for 4 heads
        k: Key tensor, shape [num_kv_heads * head_dim] = [256] for 1 head
        position: Current position in sequence
        head_dim: Dimension per head (256)
    """
    # Compute frequencies for RoPE
    freqs = 1.0 / (rope_base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    
    # Compute angles for this position
    angles = position * freqs  # shape: [128]
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    
    def rotate_half(x):
        """Rotate half the hidden dims of the input."""
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)
    
    def apply_rotary_pos_emb(x, cos, sin):
        """Apply rotary position embedding to tensor x."""
        # x shape: [num_heads * head_dim]
        # Reshape to [num_heads, head_dim]
        num_heads = x.shape[0] // head_dim
        x_reshaped = x.view(num_heads, head_dim)
        
        # Apply rotation per head
        cos_expanded = cos.unsqueeze(0).expand(num_heads, -1)  # [num_heads, 128]
        sin_expanded = sin.unsqueeze(0).expand(num_heads, -1)  # [num_heads, 128]
        
        # Split into even and odd indices
        x_even = x_reshaped[:, 0::2]  # [num_heads, 128]
        x_odd = x_reshaped[:, 1::2]   # [num_heads, 128]
        
        # Apply rotation
        rotated_even = x_even * cos_expanded - x_odd * sin_expanded
        rotated_odd = x_even * sin_expanded + x_odd * cos_expanded
        
        # Interleave back
        result = torch.zeros_like(x_reshaped)
        result[:, 0::2] = rotated_even
        result[:, 1::2] = rotated_odd
        
        return result.view(-1)  # Flatten back to [num_heads * head_dim]
    
    q_rot = apply_rotary_pos_emb(q, cos, sin)
    k_rot = apply_rotary_pos_emb(k, cos, sin)
    
    return q_rot, k_rot

def forward_layer(x, layer_idx, weights, position, kv_cache=None):
    """Forward pass through one transformer layer with KV caching.
    
    Args:
        x: Input tensor [hidden_dim] = [1152]
        layer_idx: Which transformer layer (0-25)
        weights: Model weights dict
        position: Current position in sequence
        kv_cache: Optional dict with 'k' and 'v' lists of cached tensors
    
    Returns:
        (output, updated_cache)
    """
    prefix = f"model.layers.{layer_idx}"
    
    # Pre-attention norm
    input_norm = weights[f"{prefix}.input_layernorm.weight"]
    x_norm = rmsnorm(x, input_norm)
    
    # Attention projections
    q_weight = weights[f"{prefix}.self_attn.q_proj.weight"]
    k_weight = weights[f"{prefix}.self_attn.k_proj.weight"]
    v_weight = weights[f"{prefix}.self_attn.v_proj.weight"]
    
    q = F.linear(x_norm, q_weight)  # [1024] = 4 heads * 256 dim
    k = F.linear(x_norm, k_weight)  # [256] = 1 head * 256 dim
    v = F.linear(x_norm, v_weight)  # [256] = 1 head * 256 dim
    
    # Apply RoPE
    q, k = apply_rope(q, k, position)
    
    # Update KV cache
    if kv_cache is None:
        kv_cache = {'k': [k], 'v': [v]}
    else:
        kv_cache['k'].append(k)
        kv_cache['v'].append(v)
    
    # Concatenate all cached K/V: [seq_len, kv_dim]
    k_cached = torch.stack(kv_cache['k'], dim=0)  # [seq_len, 256]
    v_cached = torch.stack(kv_cache['v'], dim=0)  # [seq_len, 256]
    
    # Reshape for GQA (Grouped Query Attention)
    # Q: 4 query heads, K/V: 1 kv head (each query head shares same kv head)
    q = q.view(4, 256)                    # [4, 256]
    k_cached = k_cached.view(-1, 1, 256)  # [seq_len, 1, 256]
    v_cached = v_cached.view(-1, 1, 256)  # [seq_len, 1, 256]
    
    # Expand KV to match all query heads
    seq_len = k_cached.shape[0]
    k_expanded = k_cached.expand(seq_len, 4, 256)  # [seq_len, 4, 256]
    v_expanded = v_cached.expand(seq_len, 4, 256)  # [seq_len, 4, 256]
    
    # Compute attention scores: Q @ K^T / sqrt(head_dim)
    # q: [4, 256], k_expanded: [seq_len, 4, 256]
    # We want [4, seq_len] scores
    scores = torch.einsum('hd,shd->hs', q, k_expanded) / (256 ** 0.5)  # [4, seq_len]
    
    # Apply softmax
    attn_weights = F.softmax(scores, dim=-1)  # [4, seq_len]
    
    # Apply attention to values: attn @ V
    # attn_weights: [4, seq_len], v_expanded: [seq_len, 4, 256]
    attn_out = torch.einsum('hs,shd->hd', attn_weights, v_expanded)  # [4, 256]
    attn_out = attn_out.reshape(-1)  # [1024]
    
    # Output projection
    out_weight = weights[f"{prefix}.self_attn.o_proj.weight"]
    attn_proj = F.linear(attn_out, out_weight)  # [1152]
    
    # First residual
    x = x + attn_proj
    
    # Post-attention norm (serves as pre-FFN norm in Gemma 3)
    post_attn_key = f"{prefix}.post_attention_layernorm.weight"
    if post_attn_key not in weights:
        post_attn_key = f"{prefix}.post_attention_norm.weight"
    post_attn_norm = weights[post_attn_key]
    x_norm2 = rmsnorm(x, post_attn_norm)
    
    # FFN
    gate_weight = weights[f"{prefix}.mlp.gate_proj.weight"]
    up_weight = weights[f"{prefix}.mlp.up_proj.weight"]
    down_weight = weights[f"{prefix}.mlp.down_proj.weight"]
    
    gate = F.linear(x_norm2, gate_weight)
    up = F.linear(x_norm2, up_weight)
    
    # GELU activation (approximate='tanh' matches Gemma)
    gelu_gate = F.gelu(gate, approximate='tanh')
    ffn_hidden = gelu_gate * up
    
    # Down projection
    ffn_out = F.linear(ffn_hidden, down_weight)

    # Post-FFN normalization (Gemma 3 feature - CORRECTED!)
    # The Keras config says use_post_ffw_norm: true
    post_ffn_key = f"{prefix}.post_feedforward_layernorm.weight"
    if post_ffn_key in weights:
        post_ffn_norm = weights[post_ffn_key]
        ffn_out = rmsnorm(ffn_out, post_ffn_norm)

    # Second residual
    x = x + ffn_out

    return x, kv_cache

print("=== PyTorch Reference: 17-Token Prompt ===\n")

# Load model
print("Loading FP32 model...")
weights = load_fp32_model("../models/gemma3-1b.safetensors")
print(f"Loaded {len(weights)} weight tensors\n")

# Prompt tokens from Haskell DEBUG output
prompt_tokens = [2,2,105,2364,107,3689,563,236743,236778,236862,236778,236881,106,107,105,4368,107]
print(f"Prompt tokens ({len(prompt_tokens)}): {prompt_tokens}")
print("Decoded: <start_of_turn>user\\nWhat is 2+2?<end_of_turn>\\n<start_of_turn>model\\n\n")

# Get embeddings
embed_table = weights["model.embed_tokens.weight"]
final_norm = weights["model.norm.weight"]

# Process prompt tokens one by one, building KV cache
kv_caches = [None] * 26  # One cache per layer
x = None

print("\nProcessing prompt tokens...")
for token_idx, token in enumerate(prompt_tokens):
    print(f"Token {token_idx + 1}/{len(prompt_tokens)}: {token}")
    
    # Get embedding
    x = embed_table[token]
    
    # Forward through all 26 layers
    for layer_idx in range(26):
        x, kv_caches[layer_idx] = forward_layer(x, layer_idx, weights, token_idx, kv_caches[layer_idx])
    
    # Final norm
    x_norm = rmsnorm(x, final_norm)
    
    # LM head (tied with embeddings)
    logits = F.linear(x_norm, embed_table)
    
    # Get top token
    top_token = torch.argmax(logits).item()
    top_logit = logits[top_token].item()
    
    # Only show details for last token
    if token_idx == len(prompt_tokens) - 1:
        print(f"\n=== AFTER PROCESSING ALL PROMPT TOKENS ===")
        print(f"Top predicted token: {top_token}")
        print(f"Top logit: {top_logit:.4f}")
        
        # Show top 5
        top5_values, top5_indices = torch.topk(logits, 5)
        print("\nTop 5 predictions:")
        for i in range(5):
            print(f"  token={top5_indices[i].item()} logit={top5_values[i].item():.4f}")
        
        print("\n\nHaskell output:")
        print("  Top: token=143055 logit=64.50845")
        print("  Top 5: [143055, 234605, 139602, 98627, 178327]")
        
        # Save reference
        reference = {
            "prompt_tokens": prompt_tokens,
            "top_token": top_token,
            "top_logit": top_logit,
            "top5_tokens": [top5_indices[i].item() for i in range(5)],
            "top5_logits": [top5_values[i].item() for i in range(5)]
        }
        
        with open("test/Gemma/Regression/MultiTokenPromptSpec_prompt17.json", "w") as f:
            json.dump(reference, f, indent=2)
        print("\n✅ Saved reference to MultiTokenPromptSpec_prompt17.json")
