#!/usr/bin/env python3
"""
Generate reference data from official Gemma 3 1B model for embedding layer test.
Following project convention: co-located with CompareWithOfficialSpec.hs
Output: CompareWithOfficialSpec_embedding.json (or OfficialReference_prompt17.json)
"""

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

def main():
    print("=== Generating Official Gemma 3 Reference Data ===")
    print()

    # Load official model
    model_path = "../models/gemma3-1b-official-instruct"
    print(f"Loading model from: {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Test prompt: same as CompareWithOfficialSpec.hs
    prompt = "<bos><start_of_turn>user\nWhat is 2+2?<end_of_turn>\n<start_of_turn>model\n"

    print(f"Prompt: {repr(prompt)}")

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    tokens = inputs.input_ids[0].tolist()

    print(f"Tokens: {tokens}")
    print(f"Number of tokens: {len(tokens)}")
    print(f"Last token: {tokens[-1]}")
    print()

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        # Get all hidden states
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

        # Get final logits
        logits = outputs.logits[0, -1, :]  # Last token logits

        # Get top predictions
        top5_values, top5_indices = torch.topk(logits, 5)

        predictions = []
        for i in range(5):
            token_id = top5_indices[i].item()
            logit = top5_values[i].item()
            token_str = tokenizer.decode([token_id])
            predictions.append({
                "token": token_id,
                "logit": float(logit),
                "text": token_str
            })
            print(f"  Top {i+1}: token={token_id} ({repr(token_str)}) logit={logit:.4f}")

        print()

        # Extract layer outputs (embedding + 26 transformer layers)
        hidden_states = outputs.hidden_states  # Tuple of (embedding, layer0, ..., layer25)
        print(f"Number of hidden states: {len(hidden_states)} (embedding + {len(hidden_states)-1} layers)")

        layer_outputs = []
        for i, hidden in enumerate(hidden_states):
            # Get last token's hidden state
            last_token_hidden = hidden[0, -1, :]  # [hidden_dim]

            layer_name = "embedding" if i == 0 else f"layer_{i-1}"
            stats = {
                "layer": layer_name,
                "mean": float(last_token_hidden.mean()),
                "std": float(last_token_hidden.std()),
                "min": float(last_token_hidden.min()),
                "max": float(last_token_hidden.max()),
                "first_10": last_token_hidden[:10].tolist(),
                "last_10": last_token_hidden[-10:].tolist()
            }

            print(f"  {layer_name}: mean={stats['mean']:.6f} std={stats['std']:.6f}")
            layer_outputs.append(stats)

    # Get model config
    config_dict = {
        "num_hidden_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "num_attention_heads": model.config.num_attention_heads,
        "num_key_value_heads": model.config.num_key_value_heads,
        "head_dim": model.config.head_dim,
        "intermediate_size": model.config.intermediate_size,
        "rope_theta": model.config.rope_theta
    }

    # Create reference output
    reference = {
        "tokens": tokens,
        "predictions": predictions,
        "layer_outputs": layer_outputs,
        "config": config_dict
    }

    # Save to JSON
    output_path = "test/Gemma/Regression/OfficialReference_prompt17.json"
    with open(output_path, "w") as f:
        json.dump(reference, f, indent=2)

    print()
    print(f"✅ Reference data saved to: {output_path}")
    print(f"   Tokens: {len(tokens)}")
    print(f"   Top prediction: token={predictions[0]['token']} ({repr(predictions[0]['text'])}) logit={predictions[0]['logit']:.4f}")
    print(f"   Layers: {len(layer_outputs)} (embedding + {len(layer_outputs)-1} transformer layers)")

if __name__ == "__main__":
    main()
