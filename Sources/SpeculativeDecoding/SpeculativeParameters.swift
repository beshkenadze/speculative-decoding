// Copyright 2024 MLX Speculative Decoding Contributors
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon

/// Configuration parameters for speculative decoding.
///
/// Speculative decoding uses a smaller draft model to propose multiple tokens,
/// which are then verified in parallel by the larger target model. These parameters
/// control the behavior of both the drafting and verification phases.
///
/// ## Example Usage
/// ```swift
/// let params = SpeculativeParameters(
///     numDraftTokens: 5,
///     draftTemperature: 0.7,
///     targetTemperature: 0.7,
///     maxTokens: 256
/// )
/// ```
public struct SpeculativeParameters: Sendable, Hashable {
    
    // Draft Configuration
    
    /// Number of tokens to draft speculatively per iteration.
    ///
    /// Higher values can improve throughput when acceptance rate is high,
    /// but may waste computation when acceptance rate is low.
    /// Typical values: 4-8 tokens.
    public var numDraftTokens: Int
    
    /// Temperature for draft model sampling.
    ///
    /// Lower temperatures make the draft model more deterministic,
    /// which can improve acceptance rate but reduce diversity.
    public var draftTemperature: Float
    
    /// Top-p (nucleus) sampling threshold for draft model.
    ///
    /// Only tokens with cumulative probability up to `topP` are considered.
    /// Set to 1.0 to disable top-p sampling.
    public var draftTopP: Float
    
    // Target Configuration
    
    /// Temperature for target model verification and sampling.
    ///
    /// This should typically match or be close to the draft temperature
    /// for optimal acceptance rates.
    public var targetTemperature: Float
    
    /// Top-p (nucleus) sampling threshold for target model.
    public var targetTopP: Float
    
    // Generation Limits
    
    /// Maximum number of tokens to generate (excluding prompt).
    ///
    /// Set to `nil` for unlimited generation (until EOS token).
    public var maxTokens: Int?
    
    /// Step size for processing the prompt (prefill phase).
    ///
    /// Larger values use more memory but can be faster for long prompts.
    public var prefillStepSize: Int
    
    // Advanced Options
    
    /// Whether to use tree-based speculation for higher parallelism.
    ///
    /// Tree speculation generates multiple candidate sequences in parallel,
    /// potentially improving throughput at the cost of memory.
    /// Note: This is an advanced feature and may not be fully supported.
    public var useTreeSpeculation: Bool
    
    /// Number of tree branches when using tree speculation.
    public var treeBranches: Int
    
    /// Minimum acceptance probability threshold.
    ///
    /// Tokens with acceptance probability below this threshold are
    /// automatically rejected without full verification.
    /// Set to 0.0 to disable early rejection.
    public var minAcceptanceProbability: Float
    
    /// Whether to use greedy decoding (temperature = 0).
    ///
    /// When true, both draft and target models use argmax sampling,
    /// which simplifies verification to exact match checking.
    public var useGreedyDecoding: Bool {
        draftTemperature == 0 && targetTemperature == 0
    }
    
    // Initialization
    
    /// Creates speculative decoding parameters with the specified configuration.
    ///
    /// - Parameters:
    ///   - numDraftTokens: Number of tokens to draft per iteration (default: 5)
    ///   - draftTemperature: Temperature for draft model (default: 0.6)
    ///   - targetTemperature: Temperature for target model (default: 0.6)
    ///   - draftTopP: Top-p sampling for draft model (default: 0.9)
    ///   - targetTopP: Top-p sampling for target model (default: 0.9)
    ///   - maxTokens: Maximum tokens to generate (default: nil)
    ///   - prefillStepSize: Prefill step size (default: 512)
    ///   - useTreeSpeculation: Enable tree speculation (default: false)
    ///   - treeBranches: Number of tree branches (default: 4)
    ///   - minAcceptanceProbability: Minimum acceptance threshold (default: 0.0)
    public init(
        numDraftTokens: Int = 5,
        draftTemperature: Float = 0.6,
        targetTemperature: Float = 0.6,
        draftTopP: Float = 0.9,
        targetTopP: Float = 0.9,
        maxTokens: Int? = nil,
        prefillStepSize: Int = 512,
        useTreeSpeculation: Bool = false,
        treeBranches: Int = 4,
        minAcceptanceProbability: Float = 0.0
    ) {
        precondition(numDraftTokens > 0, "numDraftTokens must be positive")
        precondition(draftTemperature >= 0, "draftTemperature must be non-negative")
        precondition(targetTemperature >= 0, "targetTemperature must be non-negative")
        precondition(draftTopP > 0 && draftTopP <= 1, "draftTopP must be in (0, 1]")
        precondition(targetTopP > 0 && targetTopP <= 1, "targetTopP must be in (0, 1]")
        precondition(prefillStepSize > 0, "prefillStepSize must be positive")
        
        self.numDraftTokens = numDraftTokens
        self.draftTemperature = draftTemperature
        self.targetTemperature = targetTemperature
        self.draftTopP = draftTopP
        self.targetTopP = targetTopP
        self.maxTokens = maxTokens
        self.prefillStepSize = prefillStepSize
        self.useTreeSpeculation = useTreeSpeculation
        self.treeBranches = treeBranches
        self.minAcceptanceProbability = minAcceptanceProbability
    }
    
    // Presets
    
    /// Default parameters optimized for balanced speed and quality.
    public static let `default` = SpeculativeParameters()
    
    /// Parameters optimized for maximum throughput (higher draft count).
    public static let fastThroughput = SpeculativeParameters(
        numDraftTokens: 8,
        draftTemperature: 0.5,
        targetTemperature: 0.5,
        draftTopP: 0.85
    )
    
    /// Parameters for greedy (deterministic) decoding.
    public static let greedy = SpeculativeParameters(
        numDraftTokens: 6,
        draftTemperature: 0,
        targetTemperature: 0,
        draftTopP: 1.0,
        targetTopP: 1.0
    )
    
    /// Parameters for creative/diverse generation.
    public static let creative = SpeculativeParameters(
        numDraftTokens: 4,
        draftTemperature: 0.9,
        targetTemperature: 0.9,
        draftTopP: 0.95,
        targetTopP: 0.95
    )
    
    /// Conservative parameters with lower draft count for better acceptance.
    public static let conservative = SpeculativeParameters(
        numDraftTokens: 3,
        draftTemperature: 0.6,
        targetTemperature: 0.6,
        draftTopP: 0.9
    )
}

// MLXLMCommon Integration

extension SpeculativeParameters {
    /// Convert to GenerateParameters for draft model.
    public func draftGenerateParameters() -> GenerateParameters {
        GenerateParameters(
            maxTokens: maxTokens,
            temperature: draftTemperature,
            topP: draftTopP,
            prefillStepSize: prefillStepSize
        )
    }
    
    /// Convert to GenerateParameters for target model.
    public func targetGenerateParameters() -> GenerateParameters {
        GenerateParameters(
            maxTokens: maxTokens,
            temperature: targetTemperature,
            topP: targetTopP,
            prefillStepSize: prefillStepSize
        )
    }
}

// CustomStringConvertible

extension SpeculativeParameters: CustomStringConvertible {
    public var description: String {
        """
        SpeculativeParameters(
            numDraftTokens: \(numDraftTokens),
            draftTemperature: \(draftTemperature),
            targetTemperature: \(targetTemperature),
            draftTopP: \(draftTopP),
            targetTopP: \(targetTopP),
            maxTokens: \(maxTokens?.description ?? "nil"),
            prefillStepSize: \(prefillStepSize),
            useTreeSpeculation: \(useTreeSpeculation)
        )
        """
    }
}
