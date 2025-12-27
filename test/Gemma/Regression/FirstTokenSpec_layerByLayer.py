#!/usr/bin/env python3
"""
Generate layer-by-layer intermediate outputs to find where Haskell diverges from PyTorch.
This will help us pinpoint the exact layer causing the bug.
"""

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_path = os.path.abspath('../models/gemma3-1b-official-instruct')
print(f'Loading from: {model_path}')

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,  # FP32 for exact comparison
    device_map='cpu',
    local_files_only=True,
    output_hidden_states=True  # CRITICAL: Get all layer outputs
)
model.eval()

# Same prompt as FirstTokenSpec
prompt = 'What is 2+2?'
messages = [{'role': 'user', 'content': prompt}]
chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print(f'Chat prompt: {chat_prompt}')

inputs = tokenizer(chat_prompt, return_tensors='pt')
input_ids = inputs['input_ids']

print(f'Input tokens ({len(input_ids[0])} tokens): {input_ids[0].tolist()}')

# Get outputs with hidden states
with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=True)

    # outputs.hidden_states is a tuple of (num_layers + 1) tensors
    # Index 0: embeddings (before any transformer layer)
    # Index 1: after layer 0
    # Index 2: after layer 1
    # ...
    # Index -1: after final layer (same as outputs.logits before LM head)

    hidden_states = outputs.hidden_states
    logits = outputs.logits[0, -1, :]  # Last position logits

print(f'\nNumber of hidden states: {len(hidden_states)} (embeddings + {len(hidden_states)-1} layers)')

# Extract layer outputs for the LAST token position (which predicts the first generated token)
last_position = -1

layer_outputs = {}

# Embeddings (after normalization)
embeddings = hidden_states[0][0, last_position, :].cpu().numpy()
layer_outputs['embeddings'] = {
    'mean': float(embeddings.mean()),
    'std': float(embeddings.std()),
    'min': float(embeddings.min()),
    'max': float(embeddings.max()),
    'first_10': embeddings[:10].tolist(),
    'last_10': embeddings[-10:].tolist(),
}

print(f'\nEmbeddings (last token, after normalization):')
print(f'  Shape: {hidden_states[0].shape}')
print(f'  Mean: {embeddings.mean():.6f}, Std: {embeddings.std():.6f}')
print(f'  Min: {embeddings.min():.6f}, Max: {embeddings.max():.6f}')
print(f'  First 10: {embeddings[:10].tolist()}')

# Layer 0 output (after first transformer block)
layer0_output = hidden_states[1][0, last_position, :].cpu().numpy()
layer_outputs['layer_0'] = {
    'mean': float(layer0_output.mean()),
    'std': float(layer0_output.std()),
    'min': float(layer0_output.min()),
    'max': float(layer0_output.max()),
    'first_10': layer0_output[:10].tolist(),
    'last_10': layer0_output[-10:].tolist(),
}

print(f'\nLayer 0 output (last token):')
print(f'  Mean: {layer0_output.mean():.6f}, Std: {layer0_output.std():.6f}')
print(f'  Min: {layer0_output.min():.6f}, Max: {layer0_output.max():.6f}')
print(f'  First 10: {layer0_output[:10].tolist()}')

# Layer 1 output
layer1_output = hidden_states[2][0, last_position, :].cpu().numpy()
layer_outputs['layer_1'] = {
    'mean': float(layer1_output.mean()),
    'std': float(layer1_output.std()),
    'min': float(layer1_output.min()),
    'max': float(layer1_output.max()),
    'first_10': layer1_output[:10].tolist(),
    'last_10': layer1_output[-10:].tolist(),
}

print(f'\nLayer 1 output (last token):')
print(f'  Mean: {layer1_output.mean():.6f}, Std: {layer1_output.std():.6f}')
print(f'  Min: {layer1_output.min():.6f}, Max: {layer1_output.max():.6f}')
print(f'  First 10: {layer1_output[:10].tolist()}')

# Final layer output (before LM head)
final_hidden = hidden_states[-1][0, last_position, :].cpu().numpy()
layer_outputs['final_hidden'] = {
    'mean': float(final_hidden.mean()),
    'std': float(final_hidden.std()),
    'min': float(final_hidden.min()),
    'max': float(final_hidden.max()),
    'first_10': final_hidden[:10].tolist(),
    'last_10': final_hidden[-10:].tolist(),
}

print(f'\nFinal hidden state (before LM head):')
print(f'  Mean: {final_hidden.mean():.6f}, Std: {final_hidden.std():.6f}')
print(f'  Min: {final_hidden.min():.6f}, Max: {final_hidden.max():.6f}')
print(f'  First 10: {final_hidden[:10].tolist()}')

# Logits
logits_np = logits.cpu().numpy()
layer_outputs['logits'] = {
    'mean': float(logits_np.mean()),
    'std': float(logits_np.std()),
    'min': float(logits_np.min()),
    'max': float(logits_np.max()),
    'first_10': logits_np[:10].tolist(),
    'top_token': int(logits.argmax().item()),
    'top_logit': float(logits.max().item()),
}

print(f'\nLogits:')
print(f'  Mean: {logits_np.mean():.6f}, Std: {logits_np.std():.6f}')
print(f'  Min: {logits_np.min():.6f}, Max: {logits_np.max():.6f}')
print(f'  Top token: {logits.argmax().item()} (logit={logits.max().item():.2f})')

# Save reference
reference = {
    "prompt": prompt,
    "chat_prompt": chat_prompt,
    "input_tokens": input_ids[0].tolist(),
    "num_layers": len(hidden_states) - 1,
    "layer_outputs": layer_outputs,
}

output_file = "test/Gemma/Regression/FirstTokenSpec_layerByLayer.json"
with open(output_file, 'w') as f:
    json.dump(reference, f, indent=2)

print(f'\n✅ Layer-by-layer reference saved to: {output_file}')
