// Copyright 2024 MLX Speculative Decoding Contributors
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon
import MLXLLM
import Tokenizers

/// Manages a pair of draft and target language models for speculative decoding.
public final class DraftTargetPair: @unchecked Sendable {
    
    public let draftContainer: ModelContainer
    public let targetContainer: ModelContainer
    
    public init(draftContainer: ModelContainer, targetContainer: ModelContainer) {
        self.draftContainer = draftContainer
        self.targetContainer = targetContainer
    }
    
    /// Access draft model via perform block
    public func withDraftModel<R>(_ action: @Sendable (any LanguageModel) async throws -> R) async rethrows -> R {
        try await draftContainer.perform { context in
            try await action(context.model)
        }
    }
    
    /// Access target model via perform block  
    public func withTargetModel<R>(_ action: @Sendable (any LanguageModel) async throws -> R) async rethrows -> R {
        try await targetContainer.perform { context in
            try await action(context.model)
        }
    }
    
    /// Access both models
    public func withModels<R>(_ action: @Sendable (any LanguageModel, any LanguageModel) async throws -> R) async rethrows -> R {
        let draftModel = await draftContainer.perform { $0.model }
        let targetModel = await targetContainer.perform { $0.model }
        return try await action(draftModel, targetModel)
    }
    
    /// Get draft model
    public var draftModel: any LanguageModel {
        get async {
            await draftContainer.perform { $0.model }
        }
    }
    
    /// Get target model
    public var targetModel: any LanguageModel {
        get async {
            await targetContainer.perform { $0.model }
        }
    }
    
    /// Get tokenizer
    public var tokenizer: Tokenizer {
        get async {
            await targetContainer.perform { $0.tokenizer }
        }
    }
    
    /// Get configuration
    public var configuration: ModelConfiguration {
        get async {
            await targetContainer.configuration
        }
    }
    
    /// Load models from IDs
    public static func load(
        draftModelId: String,
        targetModelId: String
    ) async throws -> DraftTargetPair {
        async let draftTask = LLMModelFactory.shared.loadContainer(
            configuration: ModelConfiguration(id: draftModelId)
        )
        async let targetTask = LLMModelFactory.shared.loadContainer(
            configuration: ModelConfiguration(id: targetModelId)
        )
        
        let (draftContainer, targetContainer) = try await (draftTask, targetTask)
        return DraftTargetPair(draftContainer: draftContainer, targetContainer: targetContainer)
    }
    
    /// Prepare input
    public func prepare(prompt: String) async throws -> LMInput {
        try await targetContainer.perform { context in
            try await context.processor.prepare(input: .init(prompt: prompt))
        }
    }
    
    /// Prepare user input
    public func prepare(input: UserInput) async throws -> LMInput {
        try await targetContainer.perform { context in
            try await context.processor.prepare(input: input)
        }
    }
    
    // Recommended pairs
    public struct RecommendedPair: Sendable {
        public let draftModelId: String
        public let targetModelId: String
        public let family: String
        public let description: String
    }
    
    public static let recommendedPairs: [RecommendedPair] = [
        RecommendedPair(
            draftModelId: "mlx-community/Qwen3-0.6B-4bit",
            targetModelId: "mlx-community/Qwen3-8B-4bit",
            family: "Qwen3",
            description: "Qwen3 family - general purpose"
        ),
        RecommendedPair(
            draftModelId: "mlx-community/Llama-3.2-1B-Instruct-4bit",
            targetModelId: "mlx-community/Llama-3.2-3B-Instruct-4bit",
            family: "Llama-3.2",
            description: "Llama 3.2 family"
        ),
        RecommendedPair(
            draftModelId: "mlx-community/SmolLM3-0.5B-Instruct-4bit",
            targetModelId: "mlx-community/SmolLM3-3B-Instruct-4bit",
            family: "SmolLM3",
            description: "SmolLM3 family - lightweight"
        ),
        RecommendedPair(
            draftModelId: "mlx-community/gemma-3-1b-it-4bit",
            targetModelId: "mlx-community/gemma-3-4b-it-4bit",
            family: "gemma-3",
            description: "Gemma 3 family - strong reasoning"
        ),
    ]
    
    public static func loadRecommended(family: String) async throws -> DraftTargetPair {
        guard let pair = recommendedPairs.first(where: { $0.family.lowercased() == family.lowercased() }) else {
            throw SpeculativeDecodingError.unknownModelFamily(family)
        }
        return try await load(draftModelId: pair.draftModelId, targetModelId: pair.targetModelId)
    }
}

public enum SpeculativeDecodingError: Error, LocalizedError, Sendable {
    case incompatibleModels(reason: String)
    case unknownModelFamily(String)
    case generationFailed(String)
    case cacheError(String)
    
    public var errorDescription: String? {
        switch self {
        case .incompatibleModels(let reason): return "Incompatible models: \(reason)"
        case .unknownModelFamily(let family): return "Unknown model family: \(family)"
        case .generationFailed(let msg): return "Generation failed: \(msg)"
        case .cacheError(let msg): return "Cache error: \(msg)"
        }
    }
}
