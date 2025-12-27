#!/usr/bin/env python3
"""
Generate Layer 0 reference for token 17 (LAST token) WITH KV cache.

This tests the autoregressive generation path:
- Tokens 0-15 are cached
- Token 16 uses those 16 cached tokens for attention
- This is exactly what Haskell does!

Following TDD convention: co-located with FirstTokenSpec.hs
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
    output_attentions=False  # We'll manually extract what we need
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

# Process tokens one-by-one like Haskell does (autoregressive with KV cache)
print(f'\n=== Simulating Haskell\'s autoregressive generation with KV cache ===\n')

past_key_values = None
layer0_outputs = []

for i in range(len(input_ids[0])):
    token = input_ids[0, i:i+1].unsqueeze(0)  # Single token [1, 1]
    token_id = token[0, 0].item()

    print(f'Token {i} (id={token_id}): processing...', end='')

    with torch.no_grad():
        outputs = model(
            token,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True
        )

    # Update cache for next iteration
    past_key_values = outputs.past_key_values

    # Extract layer 0 output (hidden_states[1] is after layer 0)
    layer0_out = outputs.hidden_states[1][0, -1, :].cpu().numpy()  # [hidden_dim]
    layer0_outputs.append({
        'position': i,
        'token_id': token_id,
        'cache_len_before': i,  # How many tokens were in cache before this one
        'cache_len_after': i + 1,  # How many tokens in cache after appending this one
        'output': {
            'mean': float(layer0_out.mean()),
            'std': float(layer0_out.std()),
            'min': float(layer0_out.min()),
            'max': float(layer0_out.max()),
            'first_10': layer0_out[:10].tolist(),
        }
    })

    print(f' cache_len={i+1}, mean={layer0_out.mean():.4f}')

# Focus on the LAST token (position 16) which uses 16 cached tokens
last_token = layer0_outputs[-1]
print(f'\n=== LAST TOKEN (position {last_token["position"]}) ===')
print(f'Token ID: {last_token["token_id"]}')
print(f'Cache length before: {last_token["cache_len_before"]} tokens')
print(f'Cache length after: {last_token["cache_len_after"]} tokens')
print(f'Layer 0 output:')
print(f'  Mean: {last_token["output"]["mean"]:.6f}')
print(f'  Std: {last_token["output"]["std"]:.6f}')
print(f'  Min: {last_token["output"]["min"]:.6f}')
print(f'  Max: {last_token["output"]["max"]:.6f}')
print(f'  First 10: {last_token["output"]["first_10"]}')

# Save reference
reference = {
    'test_description': 'Layer 0 output for LAST token (position 16) using KV cache',
    'methodology': 'Autoregressive generation (one token at a time with cache)',
    'input_tokens': input_ids[0].tolist(),
    'all_tokens': layer0_outputs,
    'last_token': last_token,
}

output_file = "test/Gemma/Regression/FirstTokenSpec_layer0_withcache.json"
with open(output_file, 'w') as f:
    json.dump(reference, f, indent=2)

print(f'\n✅ Layer 0 with-cache reference saved to: {output_file}')
print(f'\nThis reference uses PyTorch KV cache (autoregressive) - same as Haskell!')
print(f'Compare Haskell\'s Layer 0 output at position 16 with: {last_token["output"]["first_10"][:5]}')
