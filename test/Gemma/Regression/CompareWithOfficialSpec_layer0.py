#!/usr/bin/env python3
"""
Generate Layer 0 output reference from official Gemma 3 1B model.
Co-located with CompareWithOfficialSpec.hs following project convention.

Extracts:
- Embedding output (normalized) - already validated ✅
- Layer 0 output - for validation

Output: CompareWithOfficialSpec_layer0.json
"""

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os

def get_tensor_stats(tensor):
    """Get statistics from a tensor."""
    return {
        "mean": float(tensor.mean()),
        "std": float(tensor.std()),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "first_10": tensor[:10].tolist(),
        "last_10": tensor[-10:].tolist()
    }

def main():
    print("=== Generating Layer 0 Reference Data ===")
    print()

    # Load official INSTRUCT model - use absolute path
    model_path = "/Users/junji.hashimoto/git/dawn/models/gemma3-1b-official-instruct"
    print(f"Loading INSTRUCT model from: {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        local_files_only=True,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True
    )

    # Test with simple prompt - just a single token for layer validation
    # Using formatted chat template
    prompt = "<bos><start_of_turn>user\nWhat is 2+2?<end_of_turn>\n<start_of_turn>model\n"

    print(f"Prompt: {repr(prompt)}")

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    tokens = inputs.input_ids[0].tolist()

    print(f"Tokens: {tokens}")
    print(f"Number of tokens: {len(tokens)}")
    print(f"Last token: {tokens[-1]}")
    print()

    # Run inference with all hidden states
    print("Running inference to extract layer outputs...")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

        # hidden_states is a tuple of tensors:
        # [0] = embedding output (after normalization!)
        # [1] = layer 0 output
        # [2] = layer 1 output
        # ...
        # [26] = layer 25 output
        hidden_states = outputs.hidden_states

        print(f"Total hidden states: {len(hidden_states)}")
        print(f"  [0] = embedding (normalized)")
        for i in range(1, len(hidden_states)):
            print(f"  [{i}] = layer {i-1} output")
        print()

        # Extract last token's hidden state for each layer
        embedding_output = hidden_states[0][0, -1, :]  # [hidden_dim]
        layer0_output = hidden_states[1][0, -1, :]     # [hidden_dim]

        # Get stats
        embedding_stats = get_tensor_stats(embedding_output)
        layer0_stats = get_tensor_stats(layer0_output)

        print("Embedding output (normalized):")
        print(f"  Mean: {embedding_stats['mean']:.6f}")
        print(f"  Std:  {embedding_stats['std']:.6f}")
        print(f"  Min:  {embedding_stats['min']:.6f}")
        print(f"  Max:  {embedding_stats['max']:.6f}")
        print()

        print("Layer 0 output:")
        print(f"  Mean: {layer0_stats['mean']:.6f}")
        print(f"  Std:  {layer0_stats['std']:.6f}")
        print(f"  Min:  {layer0_stats['min']:.6f}")
        print(f"  Max:  {layer0_stats['max']:.6f}")
        print()

    # Model config
    config_dict = {
        "num_hidden_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "num_attention_heads": model.config.num_attention_heads,
        "num_key_value_heads": model.config.num_key_value_heads,
        "head_dim": model.config.head_dim,
        "intermediate_size": model.config.intermediate_size,
        "rope_theta": model.config.rope_theta
    }

    # Create reference
    reference = {
        "prompt": prompt,
        "tokens": tokens,
        "last_token": tokens[-1],
        "config": config_dict,
        "embedding_output": embedding_stats,
        "layer0_output": layer0_stats,
        "note": "Embedding output includes normalization by sqrt(hidden_dim)"
    }

    # Save
    output_path = "/Users/junji.hashimoto/git/dawn/gemma.hs/test/Gemma/Regression/CompareWithOfficialSpec_layer0.json"
    with open(output_path, "w") as f:
        json.dump(reference, f, indent=2)

    print(f"✅ Reference data saved to: {output_path}")
    print(f"   Last token: {tokens[-1]}")
    print(f"   Embedding mean: {embedding_stats['mean']:.6f}")
    print(f"   Layer 0 mean: {layer0_stats['mean']:.6f}")
    print()
    print("Next step: Run Haskell test to compare our Layer 0 implementation")

if __name__ == "__main__":
    main()
