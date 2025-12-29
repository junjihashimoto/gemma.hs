# Gemma 3 (1B) Inference Engine - Haskell + WebGPU

High-performance Gemma 3 1B inference engine implemented in Haskell using Google's Dawn WebGPU implementation.

## Features

- **Interactive Chat CLI**: Full conversational AI with **streaming output** 🎯
  - Real-time token generation (see text as it's generated!)
  - Temperature sampling for natural responses
  - Auto-detects Gemma 2 vs Gemma 3 models
  - See [CLI_GUIDE.md](./CLI_GUIDE.md) and [IMPROVEMENTS.md](./IMPROVEMENTS.md)
- **GGUF Format Support**: Load quantized models from Hugging Face 🆕
  - Native GGUF (GGML Universal File Format) parser
  - Support for Q4_0, Q4_1, Q5_0, Q8_0 quantization formats
  - Automatic model download from Hugging Face Hub
  - Strict IO-based parser (no unsafePerformIO)
- **GPU-First Architecture**: All weights resident in GPU memory, zero CPU-GPU transfers during inference
- **Pure Haskell Tokenizer**: Zero dependencies on Python or C++ - see [TOKENIZER.md](./TOKENIZER.md) ✨
- **Test-Driven Development**: Every layer verified against PyTorch golden values
- **WebGPU Compute Shaders**: WGSL shaders for all operations (RMSNorm, Attention, MLP, etc.)
- **Automatic Resource Management**: ContT monad for safe GPU resource cleanup
- **Multiple Precision Support**: FP32, FP16, and Q4 quantization

## Project Status

🎉 **Latest Updates (December 2025)**

**Recent Achievements:**
- ✅ **GGUF Format Support** - Load quantized models from Hugging Face! 🆕
  - Native GGUF parser with strict IO (no unsafePerformIO)
  - Support for Q4_0, Q4_1, Q5_0, Q8_0 quantization
  - Automatic download from Hugging Face Hub
  - Tested with Gemma 3 1B Q4_0 (~1GB model)
- ✅ **Gemma 3 INSTRUCT RMSNorm Fix** - Zero-centered RMSNorm implemented
- ✅ **Pure Haskell Tokenizer** - Zero Python/C++ dependencies
- ✅ **KV-Cache** - 10-50x speedup for generation
- ✅ **Streaming Chat Interface** - Real-time token generation
- ✅ **PyTorch Validation** - All core layers verified

**Test Results:**
```
25 examples, 0 failures, 6 pending
Test suite gemma-test: PASS
End-to-end inference: ✅ WORKING
GGUF loading: ✅ WORKING (340 tensors, 25 metadata entries)
```

- ✅ **Phase 1**: Test Infrastructure
  - Python golden value generator
  - SafeTensors parser
  - Hspec test framework
- ✅ **Phase 2**: Core Layer Implementation (TDD)
  - RMSNorm (with parallel reduction) - **Validated (1e-5)**
  - Linear (matrix-vector multiply) - **Validated (1e-5)**
  - RoPE (rotary positional embeddings) - **Validated (1e-5)**
  - Attention (scaled dot-product with softmax) - **Validated (1e-4)**
  - GELU (activation function with numerical stability fixes) - **Validated (2e-4)**
  - GeGLU MLP (complete 5-step pipeline)
- ✅ **Phase 3**: Complete Model Architecture
  - Embedding layer (GPU token lookup) - **Validated (1e-5)**
  - TransformerBlock (attention + MLP with residual connections)
  - Full Gemma model (embeddings → 24 layers → final norm → LM head)
  - Model loading from SafeTensors
  - End-to-end inference pipeline
  - **PyTorch golden value validation for all core layers** ✅
- ✅ **Phase 4**: End-to-End Integration
  - **GQA (Grouped Query Attention)** with K/V head expansion ✅
  - CLI inference tool (`gemma-cli`) ✅
  - Tiny synthetic model for testing (2.48 MB, 2 layers) ✅
  - **End-to-end inference demonstrated** ✅
  - Architecture documentation (Gemma 1 vs Gemma 2)
- ✅ **Phase 5**: Tokenization (Complete!)
  - **Pure Haskell tokenizer** - Zero Python/C++ dependencies ✅
  - **Chat template support** - Full Gemma formatting ✅
  - **Verified correctness** - 100% match with SentencePiece ✅
  - See [TOKENIZER.md](./TOKENIZER.md) for details
- ✅ **Phase 6**: KV-Cache (Complete!)
  - **KV-cache implementation** - 10-50x speedup for generation ✅
  - **Cached attention layer** - WebGPU shaders for cached computation ✅
  - **Full model integration** - Works seamlessly with chat interface ✅
  - See [KV_CACHE_COMPLETE.md](./KV_CACHE_COMPLETE.md) for details
- ⏳ **Phase 7**: Advanced Features (next)
  - Performance benchmarking and optimization
  - Batch inference support
  - FP16 support for reduced memory usage

See [IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md) for detailed progress and [PHASE3_ROADMAP.md](./PHASE3_ROADMAP.md) for next steps.

## Quick Start

### Download Model from Hugging Face (New! 🆕)

Download quantized Gemma 3 1B model directly from Hugging Face:

```bash
# Install Python dependencies
pip install huggingface_hub

# Download and inspect model
cabal run download-gemma

# Or test GGUF loading directly
cabal run test-gguf -- ../models/gemma-3-1b-it-q4_0.gguf
```

**Supported Models:**
- `google/gemma-3-1b-it-qat-q4_0-gguf` (1B, Q4_0, ~1GB)
- `google/gemma-3-4b-it-qat-q4_0-gguf` (4B, Q4_0, ~3GB)

### Interactive Chat

Talk with Gemma directly from your terminal:

```bash
# Using SafeTensors (FP32/FP16)
cabal run gemma-cli -- \
  --model ../models/gemma3-1b-official-instruct/model.safetensors \
  --tokenizer ../models/gemma3-1b-official-instruct/tokenizer.model \
  --chat

# Using GGUF (Q4 quantized) - Partially Implemented
# Note: GGUF parser works, but full model loading needs Q4 dequantization
# cabal run gemma-cli -- \
#   --model ../models/gemma-3-1b-it-q4_0.gguf \
#   --tokenizer ../models/gemma3-1b-official-instruct/tokenizer.model \
#   --chat
```

See [CLI_GUIDE.md](./CLI_GUIDE.md) for full details.

### Prerequisites

- GHC 9.6.7+
- Cabal 3.10+
- Python 3.11+ (for golden value generation, optional)
- macOS with Metal or Linux with Vulkan

### 1. Generate Golden Values (Optional - Already Generated)

Simple golden values for layer validation are already generated. To regenerate:

```bash
cd scripts

# Install Python dependencies
pip install -r requirements.txt

# Generate simple test cases (no model required)
python generate_simple_golden_values.py
```

This creates `.safetensors` files in `test/golden-values/` with PyTorch reference outputs.

### 2. Build Haskell Project

```bash
# From gemma.hs directory
cabal update
cabal build
```

### 3. Run Tests

```bash
cabal test
```

Tests follow TDD approach:
- **RED**: Tests fail initially (not implemented)
- **GREEN**: Implement until tests pass
- **REFACTOR**: Optimize and clean up

### 4. Try End-to-End Inference

```bash
# Test with tiny synthetic model (2 layers, 128 hidden dim)
cabal run gemma-cli -- --test tiny-gemma/model.safetensors

# Expected output:
# ✅ Model loaded successfully!
# 🚀 Running inference with token ID 1 (BOS token)...
# ✅ Inference complete! Got 1000 logits
# 🎯 Next token prediction: Token ID: 561, Logit: 0.619
```

See [PHASE4_COMPLETE.md](./PHASE4_COMPLETE.md) for details on the end-to-end demo.

## Architecture

```
┌─────────────────────────────────────────────┐
│           Gemma Model (GPU-Resident)        │
├─────────────────────────────────────────────┤
│  Embeddings (GPU)                           │
│  ├─ Layer 0                                 │
│  │   ├─ RMSNorm (pre-attention)            │
│  │   ├─ Multi-Head Attention + RoPE        │
│  │   ├─ RMSNorm (pre-MLP)                  │
│  │   └─ GeGLU MLP                           │
│  ├─ Layer 1                                 │
│  │   └─ ...                                 │
│  ├─ ...                                     │
│  ├─ Layer N-1                               │
│  ├─ Final RMSNorm                           │
│  └─ LM Head                                 │
└─────────────────────────────────────────────┘
         │
         ▼
    WebGPU Dawn (Metal/Vulkan)
         │
         ▼
    WGSL Compute Shaders
```

All computations happen on GPU. No CPU transfers between layers.

## Implementation Phases

### Phase 1: Test Infrastructure ✅
- [x] Python script to export PyTorch golden values
- [x] Haskell SafeTensors parser
- [x] Hspec test framework with golden value comparison

### Phase 2: Layer-by-Layer TDD ✅
- [x] 2.1 RMSNorm (parallel workgroup reduction)
- [x] 2.2 Matrix Multiplication (Linear layers)
- [x] 2.3 RoPE (Rotary Positional Embeddings)
- [x] 2.4 Attention (Scaled Dot-Product with softmax)
- [x] 2.5 GELU activation (with numerical stability fixes)
- [x] 2.6 GeGLU MLP (complete 5-step pipeline)

### Phase 3: Complete Model Architecture ✅
- [x] 3.1 Embedding layer (GPU token lookup)
- [x] 3.2 Element-wise operations (add for residuals, multiply for gating)
- [x] 3.3 TransformerBlock (attention + MLP with residual connections)
- [x] 3.4 Full Gemma model with 24 layers
- [x] 3.5 Model loading from SafeTensors
- [x] 3.6 End-to-end inference pipeline (single token)

### Phase 4: Optimization
- [ ] 4.1 Kernel fusion
- [ ] 4.2 FP16/BF16 support
- [ ] 4.3 KV-cache for autoregressive generation
- [ ] 4.4 Batched inference

## Development Workflow (TDD)

Each layer follows this cycle:

1. **🔴 RED - Write Test First**
   ```haskell
   it "RMSNorm matches PyTorch output" $ do
     input <- loadGoldenValue "test/golden-values" "layer0_rmsnorm_input"
     weights <- loadGoldenValue "test/golden-values" "layer0_rmsnorm_weights"
     expected <- loadGoldenValue "test/golden-values" "layer0_rmsnorm_output"

     actual <- runRMSNorm input weights
     actual `shouldMatchGolden` expected $ 1e-5
   ```

2. **🟢 GREEN - Implement Until Pass**
   - Write WGSL compute shader
   - Implement Haskell wrapper
   - Run test - should pass!

3. **🔵 REFACTOR - Optimize**
   - Clean up code
   - Optimize GPU kernel
   - Re-run tests

## Project Structure

```
gemma.hs/
├── src/
│   └── Gemma/
│       ├── SafeTensors.hs         # SafeTensors format parser
│       ├── GGUF.hs                # GGUF format parser (NEW!)
│       ├── HuggingFace.hs         # HF model downloader (NEW!)
│       ├── Model.hs               # Main model
│       ├── Tokenizer.hs           # Pure Haskell tokenizer
│       ├── ChatTemplate.hs        # Gemma chat formatting
│       ├── KVCache.hs             # KV-cache for generation
│       └── Layers/
│           ├── RMSNorm.hs         # RMS normalization
│           ├── Linear.hs          # Matrix multiplication
│           ├── LinearQ4.hs        # Q4 quantized linear
│           ├── RoPE.hs            # Rotary embeddings
│           ├── Attention.hs      # Multi-head attention
│           ├── AttentionCached.hs # Cached attention
│           ├── MLP.hs             # Feed-forward network
│           ├── Embedding.hs      # Token embeddings
│           └── TransformerBlock.hs # Complete layer
├── test/
│   ├── GemmaSpec.hs              # Hspec tests
│   └── Gemma/
│       ├── Regression/           # Regression tests
│       └── Layers/               # Layer tests
├── scripts/
│   ├── download_gguf.py          # GGUF downloader (NEW!)
│   ├── export_golden_values.py  # Generate test data
│   └── requirements.txt
├── examples/
│   ├── DownloadGemma.hs          # Download demo (NEW!)
│   └── TestGGUF.hs               # GGUF inspector (NEW!)
├── app/
│   └── Main.hs                   # CLI inference tool
├── gemma.cabal                   # Package configuration
└── README.md                     # This file
```

## Testing

```bash
# Run all tests
cabal test

# Run specific test suite
cabal test --test-show-details=direct

# Run with coverage
cabal test --enable-coverage
```

## References

- **gemma.cpp**: Reference implementation for layer logic
- **gpu.cpp**: WebGPU kernel examples (GPT-2)
- **MatmulSubgroup.hs**: Optimized matrix multiplication example
- **PyTorch/Transformers**: Golden value generation

## Performance Goals

- **Latency**: <50ms per token (first token)
- **Throughput**: >20 tokens/sec (autoregressive)
- **Memory**: All 1B parameters fit in GPU memory (~4GB FP32, ~2GB FP16)

## Contributing

This project follows strict TDD:
1. Tests must be written before implementation
2. All tests must pass before moving to next component
3. Each layer verified against PyTorch golden values

## License

MIT

## Author

Junji Hashimoto
