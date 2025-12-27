#!/usr/bin/env python3
"""
Generate Layer 0 INTERNAL values for the last token WITH KV cache.

This extracts intermediate values to help debug the Layer 0 divergence:
- Input to Layer 0 (after embedding)
- After pre-attention RMSNorm
- Q, K, V projections
- Q, K after QK-Norm
- Q, K after RoPE
- Attention scores (before softmax)
- Attention output (after softmax @ V)
- After output projection
- After post-attention norm
- Final output (after residual add)

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
            output_hidden_states=False
        )

    past_key_values = outputs.past_key_values
    print(f' cache_len={i+1}')

# Now process the LAST token (position 16) with full instrumentation
print(f'\n=== Processing LAST token (position 16) with instrumentation ===\n')

last_token = input_ids[0, 16:17].unsqueeze(0)  # Token ID 107
print(f'Last token ID: {last_token[0,0].item()}')
print(f'Cache length before: 16 tokens')

# We need to manually extract Layer 0 internals
# Access the first transformer layer
layer0 = model.model.layers[0]

# Step 1: Get embedding for last token
with torch.no_grad():
    # Embedding
    token_embedding = model.model.embed_tokens(last_token)  # [1, 1, hidden_dim]

    # DEBUG: Show embedding output from embed_tokens
    # NOTE: HuggingFace Gemma 3 embed_tokens ALREADY applies sqrt(hidden_dim) scaling internally!
    # So we do NOT need to scale it again manually!
    print(f'\n1. Embedding from embed_tokens (token_id={last_token[0,0].item()}):')
    inp = token_embedding[0, 0, :].cpu().numpy()
    print(f'   mean={inp.mean():.6f}, std={inp.std():.6f}')
    print(f'   first_10: {inp[:10].tolist()}')
    print(f'   NOTE: HuggingFace Gemma embed_tokens already scales by sqrt(hidden_dim)!')

    # Use token_embedding directly (it's already scaled)
    hidden_states = token_embedding

    # Step 2: Pre-attention RMSNorm
    normed = layer0.input_layernorm(hidden_states)
    print(f'\n2. After pre-attention RMSNorm:')
    normed_np = normed[0, 0, :].cpu().numpy()
    print(f'   mean={normed_np.mean():.6f}, std={normed_np.std():.6f}')
    print(f'   first_10: {normed_np[:10].tolist()}')

    # Step 3: Q, K, V projections
    num_heads = model.config.num_attention_heads  # 4
    num_kv_heads = model.config.num_key_value_heads  # 1
    head_dim = model.config.head_dim  # 256

    q = layer0.self_attn.q_proj(normed)  # [1, 1, num_heads * head_dim]
    k = layer0.self_attn.k_proj(normed)  # [1, 1, num_kv_heads * head_dim]
    v = layer0.self_attn.v_proj(normed)  # [1, 1, num_kv_heads * head_dim]

    print(f'\n3. Q, K, V projections (BEFORE QK-Norm, BEFORE RoPE):')
    q_np = q[0, 0, :].cpu().numpy()
    k_np = k[0, 0, :].cpu().numpy()
    v_np = v[0, 0, :].cpu().numpy()
    print(f'   Q: mean={q_np.mean():.6f}, std={q_np.std():.6f}, first_10: {q_np[:10].tolist()}')
    print(f'   K: mean={k_np.mean():.6f}, std={k_np.std():.6f}, first_10: {k_np[:10].tolist()}')
    print(f'   V: mean={v_np.mean():.6f}, std={v_np.std():.6f}, first_10: {v_np[:10].tolist()}')

    # Reshape for multi-head attention
    bsz, q_len = q.size(0), q.size(1)
    q = q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)  # [1, num_heads, 1, head_dim]
    k_new = k.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)  # [1, num_kv_heads, 1, head_dim]
    v_new = v.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

    # Step 4: QK-Norm (if enabled)
    if hasattr(layer0.self_attn, 'q_norm'):
        q = layer0.self_attn.q_norm(q)
        k_new = layer0.self_attn.k_norm(k_new)
        print(f'\n4. After QK-Norm (BEFORE RoPE):')
        q_normed = q[0, :, 0, :].cpu().numpy().flatten()
        k_normed = k_new[0, :, 0, :].cpu().numpy().flatten()
        print(f'   Q: mean={q_normed.mean():.6f}, std={q_normed.std():.6f}, first_10: {q_normed[:10].tolist()}')
        print(f'   K: mean={k_normed.mean():.6f}, std={k_normed.std():.6f}, first_10: {k_normed[:10].tolist()}')

    # For debugging, we'll stop here and compare Q, K, V projections and QK-Norm
    # This is enough to identify if the issue is in:
    # - Pre-attention RMSNorm
    # - QKV projections
    # - QK-Norm
    #
    # If these match, the bug is likely in:
    # - RoPE application
    # - Attention score computation
    # - Or later stages

    print(f'\n=== Summary ===')
    print(f'This extracts intermediate values for debugging Layer 0 attention.')
    print(f'Compare these with Haskell to find the divergence point!')

# Save the intermediate values for TDD comparison
reference = {
    'test_description': 'Layer 0 internal values for LAST token (position 16) with KV cache',
    'position': 16,
    'token_id': last_token[0,0].item(),
    'cache_len': 17,
    'intermediates': {
        'input': {
            'mean': float(inp.mean()),
            'std': float(inp.std()),
            'first_10': inp[:10].tolist()
        },
        'after_prenorm': {
            'mean': float(normed_np.mean()),
            'std': float(normed_np.std()),
            'first_10': normed_np[:10].tolist()
        },
        'q_before_norm': {
            'mean': float(q_np.mean()),
            'std': float(q_np.std()),
            'first_10': q_np[:10].tolist()
        },
        'k_before_norm': {
            'mean': float(k_np.mean()),
            'std': float(k_np.std()),
            'first_10': k_np[:10].tolist()
        },
        'v_projection': {
            'mean': float(v_np.mean()),
            'std': float(v_np.std()),
            'first_10': v_np[:10].tolist()
        }
    }
}

if hasattr(layer0.self_attn, 'q_norm'):
    with torch.no_grad():
        # QK-Norm is applied, save those values
        reference['intermediates']['q_after_norm'] = {
            'mean': float(q_normed.mean()),
            'std': float(q_normed.std()),
            'first_10': q_normed[:10].tolist()
        }
        reference['intermediates']['k_after_norm'] = {
            'mean': float(k_normed.mean()),
            'std': float(k_normed.std()),
            'first_10': k_normed[:10].tolist()
        }

output_file = "test/Gemma/Regression/Layer0WithCacheSpec_internals.json"
with open(output_file, 'w') as f:
    json.dump(reference, f, indent=2)

print(f'\n✅ Layer 0 internals saved to: {output_file}')
