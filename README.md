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
    draftModelId: "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    targetModelId: "mlx-community/Qwen2.5-7B-Instruct-4bit"
)
print(output)
```

### Streaming Generation

```swift
let stream = try await SpeculativeDecoding.generateStream(
    prompt: "Write a haiku about Swift:",
    draftModelId: "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    targetModelId: "mlx-community/Qwen2.5-7B-Instruct-4bit"
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
    draftModelId: "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    targetModelId: "mlx-community/Qwen2.5-7B-Instruct-4bit"
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
    --draft-model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    --target-model mlx-community/Qwen2.5-7B-Instruct-4bit \
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
| Qwen2.5-0.5B-4bit | Qwen2.5-7B-4bit | General purpose |
| Qwen2.5-0.5B-4bit | Qwen2.5-3B-4bit | Memory constrained |
| Llama-3.2-1B-4bit | Llama-3.2-3B-4bit | Llama family |
| SmolLM2-135M-4bit | SmolLM2-1.7B-4bit | Lightweight |
| gemma-2-2b-4bit | gemma-2-9b-4bit | Strong reasoning |

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

Typical results on Apple Silicon:

| Device | Draft | Target | Speedup | Acceptance |
|--------|-------|--------|---------|------------|
| M1 Pro | 0.5B | 7B | 2.1x | 72% |
| M2 Max | 0.5B | 7B | 2.4x | 75% |
| M3 Max | 1B | 8B | 2.6x | 78% |

Speedup depends on:
- Draft/target model size ratio
- Task similarity between models
- Temperature settings
- Hardware capabilities

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
- [MLX-Swift](https://github.com/ml-explore/mlx-swift) - Swift bindings
- [MLX-Swift-LM](https://github.com/ml-explore/mlx-swift-lm) - LLM implementations
- Original paper: "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al., 2023)
