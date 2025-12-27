#!/usr/bin/env python3
"""
Generate Layer 0 RoPE and attention values for the last token WITH KV cache.

This extracts:
- Q, K after RoPE
- Attention scores (after softmax)
- Attention output (scores @ V)

Following TDD convention: co-located with Layer0WithCacheSpec.hs
"""

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import numpy as np

model_path = os.path.abspath('../models/gemma3-1b-official-instruct')
print(f'Loading from: {model_path}')

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map='cpu',
    local_files_only=True,
    output_hidden_states=True,
    output_attentions=True  # We need attention scores!
)
model.eval()

# Same prompt as Haskell
prompt = 'What is 2+2?'
messages = [{'role': 'user', 'content': prompt}]
chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(chat_prompt, return_tensors='pt')
input_ids = inputs['input_ids']

print(f'Input tokens: {input_ids[0].tolist()}')
print(f'Total tokens: {len(input_ids[0])}')

# Process tokens autoregressively up to position 16 (last token)
print(f'\n=== Processing tokens 0-15 to build KV cache ===\n')

past_key_values = None
for i in range(16):  # Process tokens 0-15
    token = input_ids[0, i:i+1].unsqueeze(0)
    print(f'Token {i}: caching...', end='')

    with torch.no_grad():
        outputs = model(
            token,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=False,
            output_attentions=False
        )

    past_key_values = outputs.past_key_values
    print(f' cache_len={i+1}')

# Now process the LAST token (position 16) with full instrumentation
print(f'\n=== Processing LAST token (position 16) with instrumentation ===\n')

last_token = input_ids[0, 16:17].unsqueeze(0)  # Token ID 107
print(f'Last token ID: {last_token[0,0].item()}')
print(f'Cache length before: 16 tokens')

# We need to manually extract Layer 0 RoPE and attention values
# Access the first transformer layer
layer0 = model.model.layers[0]

# Step 1: Get embedding and apply scaling
with torch.no_grad():
    # Embedding (already scaled by HuggingFace)
    token_embedding = model.model.embed_tokens(last_token)  # [1, 1, hidden_dim]
    hidden_states = token_embedding

    # Step 2: Pre-attention RMSNorm
    normed = layer0.input_layernorm(hidden_states)

    # Step 3: Q, K, V projections
    num_heads = model.config.num_attention_heads  # 4
    num_kv_heads = model.config.num_key_value_heads  # 1
    head_dim = model.config.head_dim  # 256

    q = layer0.self_attn.q_proj(normed)  # [1, 1, num_heads * head_dim]
    k = layer0.self_attn.k_proj(normed)  # [1, 1, num_kv_heads * head_dim]
    v = layer0.self_attn.v_proj(normed)  # [1, 1, num_kv_heads * head_dim]

    # Reshape for multi-head attention
    bsz, q_len = q.size(0), q.size(1)
    q = q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)  # [1, num_heads, 1, head_dim]
    k_new = k.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)  # [1, num_kv_heads, 1, head_dim]
    v_new = v.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

    # Step 4: QK-Norm
    if hasattr(layer0.self_attn, 'q_norm'):
        q = layer0.self_attn.q_norm(q)
        k_new = layer0.self_attn.k_norm(k_new)

    # Step 5: Apply RoPE
    # Manually implement RoPE for Gemma 3
    position = 16
    rope_theta = getattr(model.config, 'rope_theta', 10000.0)

    # Compute RoPE frequencies
    # inv_freq shape: [head_dim // 2]
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))

    # Compute position encodings for position 16
    # position_ids shape: [1]
    position_ids = torch.tensor([position], dtype=torch.float32)

    # freqs shape: [1, head_dim // 2]
    freqs = torch.outer(position_ids, inv_freq)

    # emb shape: [1, head_dim] (interleaved cos and sin)
    emb = torch.cat((freqs, freqs), dim=-1)

    # cos/sin shape: [1, 1, 1, head_dim]
    cos_cached = emb.cos().unsqueeze(0).unsqueeze(0)
    sin_cached = emb.sin().unsqueeze(0).unsqueeze(0)

    # Apply rotation
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_rope = (q * cos_cached) + (rotate_half(q) * sin_cached)
    k_rope = (k_new * cos_cached) + (rotate_half(k_new) * sin_cached)

    print(f'\n5. Q and K after RoPE:')
    q_rope_np = q_rope[0, :, 0, :].cpu().numpy().flatten()
    k_rope_np = k_rope[0, :, 0, :].cpu().numpy().flatten()
    print(f'   Q: mean={q_rope_np.mean():.6f}, std={q_rope_np.std():.6f}, first_10: {q_rope_np[:10].tolist()}')
    print(f'   K: mean={k_rope_np.mean():.6f}, std={k_rope_np.std():.6f}, first_10: {k_rope_np[:10].tolist()}')

    # Step 6: Prepare K/V cache (we have 16 cached tokens + 1 new token)
    # For simplicity, we'll extract from the full model run
    # Get past K/V from the cache
    cached_k = past_key_values[0][0]  # Layer 0, K cache: [1, num_kv_heads, 16, head_dim]
    cached_v = past_key_values[0][1]  # Layer 0, V cache: [1, num_kv_heads, 16, head_dim]

    # Append new K/V to cache
    k_cache = torch.cat([cached_k, k_rope], dim=2)  # [1, num_kv_heads, 17, head_dim]
    v_cache = torch.cat([cached_v, v_new], dim=2)   # [1, num_kv_heads, 17, head_dim]

    # Step 7: Compute attention scores
    # Expand K/V if using GQA (num_heads > num_kv_heads)
    if num_heads != num_kv_heads:
        k_cache = k_cache.repeat_interleave(num_heads // num_kv_heads, dim=1)
        v_cache = v_cache.repeat_interleave(num_heads // num_kv_heads, dim=1)

    # Q @ K^T / sqrt(head_dim)
    scores = torch.matmul(q_rope, k_cache.transpose(-2, -1)) / (head_dim ** 0.5)  # [1, num_heads, 1, 17]

    # Apply softmax
    attn_weights = torch.softmax(scores, dim=-1)  # [1, num_heads, 1, 17]

    print(f'\n6. Attention scores (after softmax):')
    attn_weights_np = attn_weights[0, :, 0, :].cpu().numpy().flatten()
    print(f'   mean={attn_weights_np.mean():.6f}, std={attn_weights_np.std():.6f}')
    print(f'   first_10: {attn_weights_np[:10].tolist()}')

    # Step 8: Compute attention output (scores @ V)
    attn_output = torch.matmul(attn_weights, v_cache)  # [1, num_heads, 1, head_dim]

    # Reshape back
    attn_output = attn_output.transpose(1, 2).contiguous()  # [1, 1, num_heads, head_dim]
    attn_output = attn_output.view(bsz, q_len, num_heads * head_dim)  # [1, 1, hidden_dim]

    print(f'\n7. Attention output (scores @ V):')
    attn_output_np = attn_output[0, 0, :].cpu().numpy()
    print(f'   mean={attn_output_np.mean():.6f}, std={attn_output_np.std():.6f}')
    print(f'   first_10: {attn_output_np[:10].tolist()}')

# Save the RoPE and attention values for TDD comparison
reference = {
    'test_description': 'Layer 0 RoPE and attention values for LAST token (position 16) with KV cache',
    'position': 16,
    'token_id': last_token[0,0].item(),
    'cache_len': 17,
    'rope_and_attention': {
        'q_after_rope': {
            'mean': float(q_rope_np.mean()),
            'std': float(q_rope_np.std()),
            'first_10': q_rope_np[:10].tolist()
        },
        'k_after_rope': {
            'mean': float(k_rope_np.mean()),
            'std': float(k_rope_np.std()),
            'first_10': k_rope_np[:10].tolist()
        },
        'attention_scores': {
            'mean': float(attn_weights_np.mean()),
            'std': float(attn_weights_np.std()),
            'first_10': attn_weights_np[:10].tolist()
        },
        'attention_output': {
            'mean': float(attn_output_np.mean()),
            'std': float(attn_output_np.std()),
            'first_10': attn_output_np[:10].tolist()
        }
    }
}

output_file = "test/Gemma/Regression/Layer0WithCacheSpec_rope_attention.json"
with open(output_file, 'w') as f:
    json.dump(reference, f, indent=2)

print(f'\n✅ RoPE and attention values saved to: {output_file}')
