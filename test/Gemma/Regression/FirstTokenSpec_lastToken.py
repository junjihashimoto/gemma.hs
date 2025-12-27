#!/usr/bin/env python3
"""
Generate layer-by-layer output for LAST TOKEN (position 16) when processing all 17 tokens.
This is the critical test: PyTorch processes all tokens in parallel with causal masking,
Haskell processes them one-by-one with KV cache. These should be mathematically equivalent!
"""

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_path = os.path.abspath('../models/gemma3-1b-official-instruct')
print(f'Loading from: {model_path}')

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map='cpu',
    local_files_only=True,
    output_hidden_states=True
)
model.eval()

# Same prompt as Haskell
prompt = 'What is 2+2?'
messages = [{'role': 'user', 'content': prompt}]
chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print(f'Chat prompt: {chat_prompt}')

inputs = tokenizer(chat_prompt, return_tensors='pt')
input_ids = inputs['input_ids']

print(f'Input tokens ({len(input_ids[0])} tokens): {input_ids[0].tolist()}')
print(f'\nProcessing all {len(input_ids[0])} tokens in parallel (PyTorch)')
print(f'Will extract outputs for LAST token (position {len(input_ids[0])-1})')

# Process ALL tokens in parallel with causal masking
with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states

    # Extract LAST token outputs (position 16, index -1)
    last_pos = -1

    # Embeddings (after normalization) for last token
    embeddings = hidden_states[0][0, last_pos, :].cpu().numpy()

    # Layer 0 output for last token
    layer0_output = hidden_states[1][0, last_pos, :].cpu().numpy()

    # Layer 1 output for last token
    layer1_output = hidden_states[2][0, last_pos, :].cpu().numpy()

    # Final layer output for last token
    final_layer_output = hidden_states[-1][0, last_pos, :].cpu().numpy()

    # Final logits for last token
    logits = outputs.logits[0, last_pos, :].cpu().numpy()

print(f'\n=== LAST TOKEN (position {len(input_ids[0])-1}) OUTPUTS ===')

print(f'\nEmbeddings stats:')
print(f'  Mean: {embeddings.mean():.6f}, Std: {embeddings.std():.6f}')
print(f'  First 10: {embeddings[:10].tolist()}')

print(f'\nLayer 0 output stats:')
print(f'  Mean: {layer0_output.mean():.6f}, Std: {layer0_output.std():.6f}')
print(f'  Min: {layer0_output.min():.6f}, Max: {layer0_output.max():.6f}')
print(f'  First 10: {layer0_output[:10].tolist()}')

print(f'\nLayer 1 output stats:')
print(f'  Mean: {layer1_output.mean():.6f}, Std: {layer1_output.std():.6f}')
print(f'  Min: {layer1_output.min():.6f}, Max: {layer1_output.max():.6f}')
print(f'  First 10: {layer1_output[:10].tolist()}')

print(f'\nFinal layer output stats:')
print(f'  Mean: {final_layer_output.mean():.6f}, Std: {final_layer_output.std():.6f}')
print(f'  Min: {final_layer_output.min():.6f}, Max: {final_layer_output.max():.6f}')

print(f'\nLogits stats:')
print(f'  Mean: {logits.mean():.6f}')
print(f'  Top token: {logits.argmax()} (logit={logits.max():.2f})')

# Save reference
reference = {
    "prompt": prompt,
    "chat_prompt": chat_prompt,
    "input_tokens": input_ids[0].tolist(),
    "last_token_position": len(input_ids[0]) - 1,
    "embeddings": {
        "mean": float(embeddings.mean()),
        "std": float(embeddings.std()),
        "first_10": embeddings[:10].tolist(),
    },
    "layer_0": {
        "mean": float(layer0_output.mean()),
        "std": float(layer0_output.std()),
        "min": float(layer0_output.min()),
        "max": float(layer0_output.max()),
        "first_10": layer0_output[:10].tolist(),
    },
    "layer_1": {
        "mean": float(layer1_output.mean()),
        "std": float(layer1_output.std()),
        "min": float(layer1_output.min()),
        "max": float(layer1_output.max()),
        "first_10": layer1_output[:10].tolist(),
    },
    "final_layer": {
        "mean": float(final_layer_output.mean()),
        "std": float(final_layer_output.std()),
        "min": float(final_layer_output.min()),
        "max": float(final_layer_output.max()),
    },
    "logits": {
        "mean": float(logits.mean()),
        "top_token": int(logits.argmax()),
        "top_logit": float(logits.max()),
    }
}

output_file = "test/Gemma/Regression/FirstTokenSpec_lastToken.json"
with open(output_file, 'w') as f:
    json.dump(reference, f, indent=2)

print(f'\n✅ Last token reference saved to: {output_file}')
