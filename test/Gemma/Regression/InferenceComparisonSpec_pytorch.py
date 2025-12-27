#!/usr/bin/env python3
"""
Generate PyTorch reference for inference comparison test.
This uses the official Gemma 3 1B instruct model to generate expected output.
"""

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    print("=" * 70)
    print("PyTorch Reference Generator: Gemma 3 1B Instruct")
    print("=" * 70)

    model_path = "../../../models/gemma3-1b-official-instruct"

    print(f"\n📦 Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # Use FP32 for exact comparison
        device_map="cpu"
    )
    model.eval()

    print("✅ Model loaded successfully\n")

    # Test prompt
    prompt = "What is 2+2?"

    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    print(f"💬 Original Prompt: {prompt}")
    print(f"📝 Chat Template: {chat_prompt}")

    # Tokenize
    inputs = tokenizer(chat_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    print(f"\n🔢 Input tokens: {input_ids[0].tolist()}")
    print(f"   Token count: {len(input_ids[0])}")

    # Generate with greedy decoding (no sampling for reproducibility)
    print("\n🚀 Generating (greedy decoding, max 100 tokens)...")

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=100,
            do_sample=False,  # Greedy decoding
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Get only the generated tokens (excluding prompt)
    generated_ids = outputs[0][len(input_ids[0]):]

    print(f"\n✅ Generated {len(generated_ids)} tokens")
    print(f"   Token IDs: {generated_ids.tolist()[:20]}...")  # First 20

    # Decode
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    print("\n" + "=" * 70)
    print("📊 PYTORCH OUTPUT:")
    print("=" * 70)
    print(generated_text)
    print("=" * 70)

    # Save reference
    reference = {
        "prompt": prompt,
        "chat_prompt": chat_prompt,
        "input_tokens": input_ids[0].tolist(),
        "generated_tokens": generated_ids.tolist(),
        "generated_text": generated_text,
        "model_path": model_path,
        "dtype": "float32"
    }

    output_file = "test/Gemma/Regression/InferenceComparisonSpec_reference.json"
    with open(output_file, 'w') as f:
        json.dump(reference, f, indent=2)

    print(f"\n✅ Reference saved to: {output_file}")

if __name__ == "__main__":
    main()
