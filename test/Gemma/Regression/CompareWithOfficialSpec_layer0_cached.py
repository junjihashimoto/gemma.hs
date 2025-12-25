#!/usr/bin/env python3
"""
Generate Layer 0 output reference using INCREMENTAL KV-CACHED inference.
This matches Haskell's processing mode: tokens processed one-by-one with cache.

Co-located with CompareWithOfficialSpec.hs following project convention.

Extracts:
- Layer 0 output for LAST token position using incremental inference with KV cache

Output: CompareWithOfficialSpec_layer0_cached.json
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
    print("=== Generating Layer 0 Reference with KV CACHE (Incremental Mode) ===")
    print()

    # Load official INSTRUCT model
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

    # Test prompt - use ACTUAL newlines, not escaped sequences!
    prompt = "<bos><start_of_turn>user\nWhat is 2+2?<end_of_turn>\n<start_of_turn>model\n"
    print(f"Prompt: {repr(prompt)}")

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    tokens = inputs.input_ids[0].tolist()

    print(f"Tokens: {tokens}")
    print(f"Number of tokens: {len(tokens)}")
    print()

    # INCREMENTAL PROCESSING WITH KV CACHE (matching Haskell)
    print("Processing tokens incrementally with KV cache...")
    print("This matches Haskell's runGemmaInferenceCached behavior")
    print()

    past_key_values = None
    layer0_outputs_per_token = []

    with torch.no_grad():
        for i, token_id in enumerate(tokens):
            print(f"Processing token {i}: ID {token_id}")

            # Process ONE token at a time
            single_token = torch.tensor([[token_id]])

            # Run forward pass with cache
            outputs = model(
                single_token,
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=True  # Enable KV caching!
            )

            # Update cache for next iteration
            past_key_values = outputs.past_key_values

            # Extract Layer 0 output for this token
            # hidden_states[0] = embedding
            # hidden_states[1] = layer 0
            layer0_output = outputs.hidden_states[1][0, -1, :]  # [hidden_dim]
            layer0_outputs_per_token.append(layer0_output)

            stats = get_tensor_stats(layer0_output)
            print(f"  Layer 0 output: mean={stats['mean']:.6f}, std={stats['std']:.6f}")
            print(f"  First 3: {stats['first_10'][:3]}")

    print()
    print("="*70)
    print("FINAL TOKEN (position 15) Layer 0 output:")
    final_layer0 = layer0_outputs_per_token[-1]
    final_stats = get_tensor_stats(final_layer0)

    print(f"  Mean: {final_stats['mean']:.6f}")
    print(f"  Std:  {final_stats['std']:.6f}")
    print(f"  Min:  {final_stats['min']:.6f}")
    print(f"  Max:  {final_stats['max']:.6f}")
    print(f"  First 10: {final_stats['first_10']}")
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
        "layer0_output_cached": final_stats,
        "note": "Layer 0 output using INCREMENTAL KV-CACHED inference (use_cache=True, one token at a time)",
        "mode": "incremental_cached"
    }

    # Save
    output_path = "/Users/junji.hashimoto/git/dawn/gemma.hs/test/Gemma/Regression/CompareWithOfficialSpec_layer0_cached.json"
    with open(output_path, "w") as f:
        json.dump(reference, f, indent=2)

    print(f"✅ Reference data saved to: {output_path}")
    print(f"   Mode: Incremental KV-cached (matches Haskell)")
    print(f"   Last token: {tokens[-1]}")
    print(f"   Layer 0 mean: {final_stats['mean']:.6f}")
    print()
    print("Next step: Compare this with Haskell's KV-cached output")
    print("If they match, KV cache is correct!")
    print("If they differ, there's a bug in Haskell's KV cache implementation")

if __name__ == "__main__":
    main()
