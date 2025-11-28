// Copyright 2024 MLX Speculative Decoding Contributors
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon
import MLXLLM
import Tokenizers

/// Main entry point for speculative decoding in MLX-Swift.
public enum SpeculativeDecoding {
    
    /// Generates text using speculative decoding.
    public static func generate(
        prompt: String,
        draftModelId: String,
        targetModelId: String,
        parameters: SpeculativeParameters = .default
    ) async throws -> String {
        let modelPair = try await DraftTargetPair.load(
            draftModelId: draftModelId,
            targetModelId: targetModelId
        )
        return try await generate(prompt: prompt, modelPair: modelPair, parameters: parameters)
    }
    
    /// Generates text using a pre-loaded model pair.
    public static func generate(
        prompt: String,
        modelPair: DraftTargetPair,
        parameters: SpeculativeParameters = .default
    ) async throws -> String {
        let input = try await modelPair.prepare(prompt: prompt)
        let generator = SpeculativeGenerator(modelPair: modelPair, parameters: parameters)
        let result = try await generator.generate(input: input) { _ in .more }
        let tokenizer = await modelPair.tokenizer
        return result.decode(with: tokenizer)
    }
    
    /// Generates text with detailed results.
    public static func generateWithResult(
        prompt: String,
        modelPair: DraftTargetPair,
        parameters: SpeculativeParameters = .default
    ) async throws -> (text: String, result: SpeculativeGenerateResult) {
        let input = try await modelPair.prepare(prompt: prompt)
        let generator = SpeculativeGenerator(modelPair: modelPair, parameters: parameters)
        let result = try await generator.generate(input: input) { _ in .more }
        let tokenizer = await modelPair.tokenizer
        let text = result.decode(with: tokenizer)
        return (text, result)
    }
    
    /// Generates text with streaming output.
    public static func generateStream(
        prompt: String,
        draftModelId: String,
        targetModelId: String,
        parameters: SpeculativeParameters = .default
    ) async throws -> AsyncStream<SpeculativeGenerationEvent> {
        let modelPair = try await DraftTargetPair.load(
            draftModelId: draftModelId,
            targetModelId: targetModelId
        )
        return try await generateStream(prompt: prompt, modelPair: modelPair, parameters: parameters)
    }
    
    /// Generates text with streaming output using a pre-loaded model pair.
    public static func generateStream(
        prompt: String,
        modelPair: DraftTargetPair,
        parameters: SpeculativeParameters = .default
    ) async throws -> AsyncStream<SpeculativeGenerationEvent> {
        let input = try await modelPair.prepare(prompt: prompt)
        let generator = SpeculativeGenerator(modelPair: modelPair, parameters: parameters)
        return generator.generateStream(input: input)
    }
    
    /// Generates text using a recommended model pair from a model family.
    public static func generate(
        prompt: String,
        family: String,
        parameters: SpeculativeParameters = .default
    ) async throws -> String {
        let modelPair = try await DraftTargetPair.loadRecommended(family: family)
        return try await generate(prompt: prompt, modelPair: modelPair, parameters: parameters)
    }
}

public typealias GenerateDisposition = MLXLMCommon.GenerateDisposition
