# Speculative Decoding for MLX-Swift

Native speculative decoding implementation for fast LLM inference on Apple Silicon using [MLX-Swift](https://github.com/ml-explore/mlx-swift).

## Overview

Speculative decoding accelerates LLM inference by using a smaller "draft" model to propose multiple tokens, which are then verified in parallel by a larger "target" model. This achieves **2-3x speedups** while maintaining exact output distribution equivalence.

## Features

- Native Swift implementation optimized for Apple Silicon
- Simple high-level API and streaming support
- Compatible with any MLX-Swift LLM models
- Full Swift concurrency support (async/await)
- Greedy and temperature-based sampling
- Comprehensive statistics and benchmarking

## Requirements

- macOS 14.0+ / iOS 16.0+
- Xcode 15.0+
- Swift 5.9+
- Apple Silicon (M1/M2/M3/M4)

## Installation

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/lulzx/speculative-decoding", branch: "main"),
]
```

Then add the dependency to your target:

```swift
.target(
    name: "YourApp",
    dependencies: [
        .product(name: "SpeculativeDecoding", package: "speculative-decoding"),
    ]
),
```

## Quick Start

### Simple Generation

```swift
import SpeculativeDecoding

let output = try await SpeculativeDecoding.generate(
    prompt: "Explain quantum computing:",
    draftModelId: "mlx-community/Qwen3-0.6B-4bit",
    targetModelId: "mlx-community/Qwen3-8B-4bit"
)
print(output)
```

### Streaming Generation

```swift
let stream = try await SpeculativeDecoding.generateStream(
    prompt: "Write a haiku about Swift:",
    draftModelId: "mlx-community/Qwen3-0.6B-4bit",
    targetModelId: "mlx-community/Qwen3-8B-4bit"
)

for await event in stream {
    switch event {
    case .text(let chunk):
        print(chunk, terminator: "")
    case .result(let result):
        print("\n\(result.summary())")
    case .error(let error):
        print("Error: \(error)")
    }
}
```

### Advanced Usage

```swift
// Load models once, generate multiple times
let modelPair = try await DraftTargetPair.load(
    draftModelId: "mlx-community/Qwen3-0.6B-4bit",
    targetModelId: "mlx-community/Qwen3-8B-4bit"
)

let parameters = SpeculativeParameters(
    numDraftTokens: 6,
    draftTemperature: 0.7,
    targetTemperature: 0.7,
    maxTokens: 512
)

let generator = SpeculativeGenerator(modelPair: modelPair, parameters: parameters)
let input = try modelPair.prepare(prompt: "Hello, world!")
let result = try await generator.generate(input: input) { tokens in
    return .more  // Continue generating
}

print(result.summary())
```

## CLI Tool

Build and run the CLI:

```bash
swift build -c release
.build/release/speculative-cli generate \
    --draft-model mlx-community/Qwen3-0.6B-4bit \
    --target-model mlx-community/Qwen3-8B-4bit \
    --prompt "Explain neural networks:" \
    --max-tokens 256 \
    --stats
```

### Commands

```bash
# Generate text
speculative-cli generate --prompt "Your prompt" --stats

# Benchmark speculative vs standard decoding
speculative-cli benchmark --prompt "Test prompt" --iterations 3

# List recommended model pairs
speculative-cli list-models
```

## Recommended Model Pairs

| Draft Model | Target Model | Use Case |
|-------------|--------------|----------|
| Qwen3-0.6B-4bit | Qwen3-8B-4bit | General purpose |
| Qwen3-0.6B-4bit | Qwen3-4B-4bit | Memory constrained |
| Llama-3.2-1B-4bit | Llama-3.2-3B-4bit | Llama family |
| SmolLM3-0.5B-4bit | SmolLM3-3B-4bit | Lightweight |
| gemma-3-1b-4bit | gemma-3-4b-4bit | Strong reasoning |

## Configuration Options

```swift
SpeculativeParameters(
    numDraftTokens: 5,        // Tokens to draft per iteration (4-8 typical)
    draftTemperature: 0.6,    // Draft model temperature
    targetTemperature: 0.6,   // Target model temperature
    draftTopP: 0.9,           // Top-p sampling for draft
    targetTopP: 0.9,          // Top-p sampling for target
    maxTokens: nil,           // Max tokens (nil = unlimited)
    prefillStepSize: 512      // Prompt processing chunk size
)

// Presets
SpeculativeParameters.default      // Balanced
SpeculativeParameters.greedy       // Deterministic (temp=0)
SpeculativeParameters.creative     // Higher temperature
SpeculativeParameters.conservative // Lower draft count
```

## How It Works

1. **Draft Phase**: Small model generates K candidate tokens autoregressively
2. **Verify Phase**: Large model processes all K tokens in a single forward pass
3. **Accept/Reject**: Rejection sampling validates each token against target distribution
4. **Repeat**: Continue with accepted tokens plus one new token from target

The key insight is that verification is parallelizable - the target model can check all draft tokens simultaneously, amortizing the cost of its larger size.

## Performance

Results on MacBook Pro M4 Pro with Qwen3 models:

| Drafter | Target | Speed | Acceptance | Tokens/Step |
|---------|--------|-------|------------|-------------|
| Qwen3-0.6B | Qwen3-4B | 81.4 tok/s | 76.3% | 4.85 |
| Mamba-130M | Qwen3-4B | 98.1 tok/s | 100% | 6.05 |

Speedup depends on:
- Draft/target model size ratio
- Task similarity between models
- Temperature settings
- Hardware capabilities

## Mamba Drafters (Experimental)

This library includes experimental support for using **Mamba** (state-space models) as draft models instead of transformers.

### Why Mamba for Drafting?

| Property | Transformer | Mamba |
|----------|-------------|-------|
| Memory per token | O(n) KV-cache | O(1) constant |
| Inference complexity | O(n²) attention | O(n) linear |
| Model size for quality | ~500M | ~130M |

### Supported Mamba Models

```bash
speculative-cli list-models

# Mamba Drafters:
# - state-spaces/mamba-130m-hf (768 hidden, 24 layers)
# - state-spaces/mamba-370m-hf (1024 hidden, 48 layers)
# - state-spaces/mamba-790m-hf (1536 hidden, 48 layers)
```

### Mamba API

```swift
import SpeculativeDecoding

// Load Mamba drafter with transformer target
let pair = try await MambaDraftTargetPair.load(
    draftModelId: "state-spaces/mamba-130m-hf",
    targetModelId: "mlx-community/Qwen3-8B-4bit"
)

let generator = MambaSpeculativeGenerator(modelPair: pair)
let result = try await generator.generate(prompt: "Hello") { _ in .more }
```

### Memory Comparison at Different Sequence Lengths

| Sequence Length | Transformer (135M) | Mamba (130M) |
|----------------|-------------------|--------------|
| 512 tokens | ~334 MB | ~260 MB |
| 2048 tokens | ~526 MB | ~260 MB |
| 8192 tokens | ~1.3 GB | ~260 MB |

### Benchmark Results

Comparing Mamba (130M) vs Transformer (0.6B) as draft models with Qwen3-4B target on MacBook Pro M4 Pro:

| Drafter | Speed | Acceptance | Tokens/Step |
|---------|-------|------------|-------------|
| **Transformer (0.6B)** | 81.4 tok/s | 76.3% | 4.85 |
| **Mamba (130M)** | 98.1 tok/s | 100% | 6.05 |

**Result: Mamba achieves 1.21x speedup** with higher acceptance rates.

#### Why Mamba Performs Well

1. **Faster drafting**: 130M Mamba is smaller with O(1) memory per token
2. **High acceptance**: Nearly all drafted tokens (6.05/step) are accepted
3. **Efficient verification**: More tokens per target model forward pass

#### Benchmark Command

```bash
.build/release/speculative-cli benchmark \
    --transformer-draft mlx-community/Qwen3-0.6B-4bit \
    --target-model mlx-community/Qwen3-4B-4bit \
    --tokens 128 --runs 3
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
- [MLX-Swift](https://github.com/ml-explore/mlx-swift) - Swift bindings
- [MLX-Swift-LM](https://github.com/ml-explore/mlx-swift-lm) - LLM implementations
- [Mamba](https://github.com/state-spaces/mamba) - State-space models
- Original paper: "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al., 2023)
- Mamba paper: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)
