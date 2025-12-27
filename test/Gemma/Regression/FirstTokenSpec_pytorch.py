#!/usr/bin/env python3
"""
Generate PyTorch reference for FIRST TOKEN logits comparison.
This will help us debug why Haskell generates wrong first token.
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
    local_files_only=True
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

# Get logits for the LAST prompt token (which predicts first generated token)
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits[0, -1, :]  # Last position logits

print(f'\nLogits shape: {logits.shape}')
print(f'Vocab size: {len(logits)}')

# Get top 10 predictions
top_k = 10
top_values, top_indices = torch.topk(logits, top_k)

print(f'\nTop {top_k} predictions for FIRST generated token:')
for i, (idx, logit) in enumerate(zip(top_indices.tolist(), top_values.tolist())):
    token_text = tokenizer.decode([idx])
    print(f'  {i+1}. Token {idx:6d} (logit={logit:8.2f}): "{token_text}"')

# Expected first token based on our test
expected_first = 236778  # Should be '2'
expected_logit = logits[expected_first].item()
expected_rank = (logits > expected_logit).sum().item() + 1

print(f'\nExpected first token: {expected_first} ("{tokenizer.decode([expected_first])}")')
print(f'  Logit: {expected_logit:.2f}')
print(f'  Rank: {expected_rank}')

# Save reference
reference = {
    "prompt": prompt,
    "chat_prompt": chat_prompt,
    "input_tokens": input_ids[0].tolist(),
    "top_10_tokens": top_indices.tolist(),
    "top_10_logits": top_values.tolist(),
    "expected_first_token": expected_first,
    "expected_first_logit": expected_logit,
    "expected_first_rank": expected_rank,
    "vocab_size": len(logits)
}

output_file = "test/Gemma/Regression/FirstTokenSpec_reference.json"
with open(output_file, 'w') as f:
    json.dump(reference, f, indent=2)

print(f'\n✅ Reference saved to: {output_file}')
