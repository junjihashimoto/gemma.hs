#!/usr/bin/env python3
"""
Generate RoPE intermediate values for position 16 with KV cache.

This script extracts Q/K values BEFORE and AFTER RoPE at position 16
to debug the multi-token bug. Co-located with Layer0WithCacheSpec.hs.

Following TDD convention: Layer0WithCacheSpec_rope.py
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
    output_attentions=False
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

# We need to hook into layer 0's attention module to extract RoPE intermediate values
# Get the actual model (unwrap from PreTrainedModel)
gemma_model = model.model  # Access the underlying GemmaModel

# Hook to capture Q/K before and after RoPE
rope_intermediates = {}

def rope_hook(module, input, output):
    """
    Hook for the attention module. We'll capture values manually.
    Note: This runs AFTER the module's forward pass.
    """
    pass

# Register hook on layer 0 self_attn
layer_0_attn = gemma_model.layers[0].self_attn
handle = layer_0_attn.register_forward_hook(rope_hook)

# We'll manually capture by modifying the forward pass
# For now, let's use a simpler approach: run inference and manually extract

print(f'\n=== Processing tokens autoregressively to position 16 ===\n')

past_key_values = None
for i in range(len(input_ids[0])):
    token = input_ids[0, i:i+1].unsqueeze(0)
    token_id = token[0, 0].item()

    print(f'Token {i} (id={token_id})')

    # For position 16 (last token), we'll manually extract intermediate values
    if i == 16:
        print(f'\n🔍 POSITION 16: Extracting RoPE intermediate values...\n')

        # Get layer 0 attention module
        attn = gemma_model.layers[0].self_attn

        # Get hidden states after embedding + positional encoding
        with torch.no_grad():
            # Forward through embedding
            hidden_states = gemma_model.embed_tokens(token)
            # Apply layer 0's input_layernorm (pre-attention RMSNorm)
            residual = hidden_states
            hidden_states = gemma_model.layers[0].input_layernorm(hidden_states)

            # Now we're at the input to self-attention
            # Extract Q, K, V projections
            bsz, q_len, _ = hidden_states.size()
            query_states = attn.q_proj(hidden_states)
            key_states = attn.k_proj(hidden_states)
            value_states = attn.v_proj(hidden_states)

            # Reshape to heads (use config for Gemma3)
            config = model.config
            num_heads = config.num_attention_heads
            num_kv_heads = config.num_key_value_heads
            head_dim = config.head_dim

            query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

            # Apply QK normalization (Gemma 3 specific)
            query_states = attn.q_norm(query_states)
            key_states = attn.k_norm(key_states)

            # Flatten back for comparison
            q_before_rope = query_states.transpose(1, 2).contiguous().view(bsz, q_len, -1)[0, 0, :].cpu().numpy()
            k_before_rope = key_states.transpose(1, 2).contiguous().view(bsz, q_len, -1)[0, 0, :].cpu().numpy()

            print(f'Q before RoPE: shape={q_before_rope.shape}')
            print(f'  Mean: {q_before_rope.mean():.6f}')
            print(f'  Std: {q_before_rope.std():.6f}')
            print(f'  First 10: {q_before_rope[:10].tolist()}')

            print(f'\nK before RoPE: shape={k_before_rope.shape}')
            print(f'  Mean: {k_before_rope.mean():.6f}')
            print(f'  Std: {k_before_rope.std():.6f}')
            print(f'  First 10: {k_before_rope[:10].tolist()}')

            # Now apply RoPE
            # Position for this token considering the cache
            kv_seq_len = i if past_key_values is None else past_key_values[0][0].shape[2] + q_len
            position_ids = torch.arange(kv_seq_len - q_len, kv_seq_len, dtype=torch.long, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0)

            # Apply RoPE
            cos, sin = attn.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # Flatten for comparison
            q_after_rope = query_states.transpose(1, 2).contiguous().view(bsz, q_len, -1)[0, 0, :].cpu().numpy()
            k_after_rope = key_states.transpose(1, 2).contiguous().view(bsz, q_len, -1)[0, 0, :].cpu().numpy()

            print(f'\nQ after RoPE: shape={q_after_rope.shape}')
            print(f'  Mean: {q_after_rope.mean():.6f}')
            print(f'  Std: {q_after_rope.std():.6f}')
            print(f'  First 10: {q_after_rope[:10].tolist()}')

            print(f'\nK after RoPE: shape={k_after_rope.shape}')
            print(f'  Mean: {k_after_rope.mean():.6f}')
            print(f'  Std: {k_after_rope.std():.6f}')
            print(f'  First 10: {k_after_rope[:10].tolist()}')

            # Save reference
            rope_intermediates = {
                'position': i,
                'token_id': token_id,
                'q_before_rope': {
                    'mean': float(q_before_rope.mean()),
                    'std': float(q_before_rope.std()),
                    'first_10': q_before_rope[:10].tolist(),
                },
                'k_before_rope': {
                    'mean': float(k_before_rope.mean()),
                    'std': float(k_before_rope.std()),
                    'first_10': k_before_rope[:10].tolist(),
                },
                'q_after_rope': {
                    'mean': float(q_after_rope.mean()),
                    'std': float(q_after_rope.std()),
                    'first_10': q_after_rope[:10].tolist(),
                },
                'k_after_rope': {
                    'mean': float(k_after_rope.mean()),
                    'std': float(k_after_rope.std()),
                    'first_10': k_after_rope[:10].tolist(),
                },
            }

    # Now run normal inference to update cache
    with torch.no_grad():
        outputs = model(
            token,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True
        )

    past_key_values = outputs.past_key_values

# Helper function (from transformers library)
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Apply rotary position embedding."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# Save reference
output_path = "test/Gemma/Regression/Layer0WithCacheSpec_rope.json"
with open(output_path, 'w') as f:
    json.dump(rope_intermediates, f, indent=2)

print(f'\n✅ RoPE intermediate values saved to {output_path}')

# Cleanup
handle.remove()
