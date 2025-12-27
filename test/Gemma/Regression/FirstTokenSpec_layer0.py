#!/usr/bin/env python3
"""
Generate layer 0 output for FIRST TOKEN ONLY (no caching).
This should match Haskell's first token processing exactly.
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

# Process ONLY the first token
first_token_id = 2  # <bos>

print(f'\nProcessing FIRST token only: {first_token_id}')

input_ids = torch.tensor([[first_token_id]])

with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states

    # Embeddings (after normalization)
    embeddings = hidden_states[0][0, 0, :].cpu().numpy()

    # Layer 0 output
    layer0_output = hidden_states[1][0, 0, :].cpu().numpy()

    # Final logits
    logits = outputs.logits[0, 0, :].cpu().numpy()

print(f'\nEmbeddings stats:')
print(f'  Mean: {embeddings.mean():.6f}, Std: {embeddings.std():.6f}')
print(f'  First 10: {embeddings[:10].tolist()}')

print(f'\nLayer 0 output stats:')
print(f'  Mean: {layer0_output.mean():.6f}, Std: {layer0_output.std():.6f}')
print(f'  Min: {layer0_output.min():.6f}, Max: {layer0_output.max():.6f}')
print(f'  First 10: {layer0_output[:10].tolist()}')

print(f'\nLogits stats:')
print(f'  Mean: {logits.mean():.6f}')
print(f'  Top token: {logits.argmax()} (logit={logits.max():.2f})')

# Save reference
reference = {
    "first_token_id": first_token_id,
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
    "logits": {
        "mean": float(logits.mean()),
        "top_token": int(logits.argmax()),
        "top_logit": float(logits.max()),
    }
}

output_file = "test/Gemma/Regression/FirstTokenSpec_layer0.json"
with open(output_file, 'w') as f:
    json.dump(reference, f, indent=2)

print(f'\n✅ First token layer 0 reference saved to: {output_file}')
