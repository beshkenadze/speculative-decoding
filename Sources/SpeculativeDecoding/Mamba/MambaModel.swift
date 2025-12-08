// Copyright © 2024 speculative-decoding contributors
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN
import MLXFast

/// Residual block containing a Mamba layer with normalization.
public class MambaResidualBlock: Module {

    @ModuleInfo var mixer: MambaBlock
    @ModuleInfo var norm: RMSNorm

    let config: MambaConfig

    public init(_ config: MambaConfig) {
        self.config = config
        self._mixer.wrappedValue = MambaBlock(config)
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.layerNormEpsilon)
        super.init()
    }

    /// Forward pass: residual + mixer(norm(x))
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Pre-norm architecture
        let normalized = norm(x)
        let mixed = mixer(normalized)

        // Residual connection
        if config.residualInFp32 {
            return (x.asType(.float32) + mixed.asType(.float32)).asType(x.dtype)
        } else {
            return x + mixed
        }
    }

    /// Single-step forward for generation
    public func step(_ x: MLXArray, cache: inout MambaCache.LayerState) -> MLXArray {
        let normalized = norm(x)
        let mixed = mixer.step(normalized, cache: &cache)

        if config.residualInFp32 {
            return (x.asType(.float32) + mixed.asType(.float32)).asType(x.dtype)
        } else {
            return x + mixed
        }
    }
}

/// Complete Mamba language model.
///
/// Architecture:
/// - Embedding layer: tokens -> vectors
/// - N x MambaResidualBlock: core computation
/// - RMSNorm: final normalization
/// - LM Head: vectors -> logits (tied with embedding)
public class MambaLM: Module {

    /// Model configuration
    public let config: MambaConfig

    /// Token embedding
    @ModuleInfo var embedding: Embedding

    /// Mamba layers
    @ModuleInfo var layers: [MambaResidualBlock]

    /// Final normalization
    @ModuleInfo(key: "norm_f") var normF: RMSNorm

    /// LM head (may be tied to embedding)
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    /// Whether LM head is tied to embedding
    let tiedEmbedding: Bool

    public init(_ config: MambaConfig, tiedEmbedding: Bool = true) {
        self.config = config
        self.tiedEmbedding = tiedEmbedding

        self._embedding.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )

        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            MambaResidualBlock(config)
        }

        self._normF.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEpsilon
        )

        if !tiedEmbedding {
            self._lmHead.wrappedValue = Linear(
                config.hiddenSize,
                config.vocabSize,
                bias: false
            )
        }

        super.init()
    }

    // MARK: - Forward Pass

    /// Forward pass for full sequence.
    ///
    /// - Parameter tokens: Input token IDs [B, L]
    /// - Returns: Logits [B, L, vocab_size]
    public func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        // Embedding
        var x = embedding(tokens)  // [B, L, D]

        // Mamba layers
        for layer in layers {
            x = layer(x)
        }

        // Final norm
        x = normF(x)

        // LM head
        if let lmHead {
            return lmHead(x)
        } else {
            // Tied embedding: use embedding weight as LM head
            return MLX.matmul(x, embedding.weight.T)
        }
    }

    /// Forward pass with cache for generation.
    ///
    /// - Parameters:
    ///   - tokens: Input token IDs [B, L] or [B, 1] for single-step
    ///   - cache: Mamba cache for stateful generation
    /// - Returns: Logits for last position [B, vocab_size]
    public func forward(_ tokens: MLXArray, cache: MambaCache) -> MLXArray {
        if tokens.dim(1) > 1 {
            // Prefill: process full sequence
            return prefill(tokens, cache: cache)
        } else {
            // Generate: single token step
            return step(tokens, cache: cache)
        }
    }

    /// Prefill: process initial sequence and populate cache.
    private func prefill(_ tokens: MLXArray, cache: MambaCache) -> MLXArray {
        // For prefill, we run the full sequence through
        // and extract the final states for the cache
        var x = embedding(tokens)

        for (i, layer) in layers.enumerated() {
            // Run full sequence
            x = layer(x)

            // Update cache with final state
            // This is a simplification - proper implementation would
            // track state throughout the sequence
            cache.layers[i].offset = tokens.dim(1)
        }

        x = normF(x)

        // Get logits for last position only
        let lastX = x[0..., -1, 0...]  // [B, D]

        if let lmHead {
            return lmHead(lastX)
        } else {
            return MLX.matmul(lastX, embedding.weight.T)
        }
    }

    /// Single-step generation with cache.
    private func step(_ tokens: MLXArray, cache: MambaCache) -> MLXArray {
        var x = embedding(tokens).squeezed(axis: 1)  // [B, D]

        for (i, layer) in layers.enumerated() {
            x = layer.step(x, cache: &cache.layers[i])
            cache.layers[i].offset += 1
        }

        x = normF(x)

        if let lmHead {
            return lmHead(x)
        } else {
            return MLX.matmul(x, embedding.weight.T)
        }
    }

    // MARK: - Generation

    /// Generate tokens autoregressively.
    ///
    /// - Parameters:
    ///   - prompt: Initial token IDs [L]
    ///   - maxTokens: Maximum tokens to generate
    ///   - temperature: Sampling temperature (0 = greedy)
    ///   - topK: Top-k sampling (0 = disabled)
    ///   - topP: Top-p (nucleus) sampling (1.0 = disabled)
    /// - Returns: Generated token sequence including prompt
    public func generate(
        prompt: MLXArray,
        maxTokens: Int = 100,
        temperature: Float = 1.0,
        topK: Int = 0,
        topP: Float = 1.0
    ) -> MLXArray {
        let cache = MambaCache(config: config, batchSize: 1)

        // Prepare prompt: [L] -> [1, L]
        let tokens = prompt.reshaped(1, -1)
        var generated = [Int32]()

        // Prefill
        let prefillLogits = forward(tokens, cache: cache)
        var nextToken = sampleToken(prefillLogits, temperature: temperature, topK: topK, topP: topP)
        generated.append(nextToken.item(Int32.self))

        // Generate
        for _ in 0..<(maxTokens - 1) {
            let tokenInput = MLXArray([nextToken.item(Int32.self)]).reshaped(1, 1)
            let logits = forward(tokenInput, cache: cache)
            nextToken = sampleToken(logits, temperature: temperature, topK: topK, topP: topP)
            generated.append(nextToken.item(Int32.self))
        }

        // Combine prompt and generated
        let promptTokens = prompt.asType(.int32)
        let generatedArray = MLXArray(generated)
        return MLX.concatenated([promptTokens, generatedArray], axis: 0)
    }

    /// Sample a token from logits.
    private func sampleToken(
        _ logits: MLXArray,
        temperature: Float,
        topK: Int,
        topP: Float
    ) -> MLXArray {
        var logits = logits

        // Handle batch dimension
        if logits.ndim > 1 {
            logits = logits.squeezed()
        }

        if temperature == 0 {
            // Greedy
            return argMax(logits, axis: -1)
        }

        // Apply temperature
        logits = logits / temperature

        // Top-k filtering
        if topK > 0 && topK < logits.dim(-1) {
            // Sort and get top-k threshold
            let sorted = MLX.sorted(logits, axis: -1)
            let threshold = sorted[logits.dim(-1) - topK]
            logits = MLX.where(logits .< threshold, MLXArray(-Float.infinity), logits)
        }

        // Top-p (nucleus) filtering
        if topP < 1.0 {
            let probs = softmax(logits, axis: -1)
            let sortedIndices = argSort(probs, axis: -1)
            let sortedProbs = takeAlong(probs, sortedIndices, axis: -1)
            let cumProbs = cumsum(sortedProbs, axis: -1)

            // Find cutoff
            let mask = cumProbs .> (1.0 - topP)
            let cutoffIdx = argMax(mask.asType(.int32), axis: -1)

            // Zero out tokens below threshold
            let threshold = takeAlong(sortedProbs, cutoffIdx.expandedDimensions(axis: -1), axis: -1)
            logits = MLX.where(probs .< threshold, MLXArray(-Float.infinity), logits)
        }

        // Sample from distribution
        let probs = softmax(logits, axis: -1)
        return categorical(probs)
    }
}
