// Copyright 2024 MLX Speculative Decoding Contributors
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXRandom

/// Implements the speculative decoding verification algorithm.
///
/// This sampler verifies draft tokens against target model logits using
/// rejection sampling, as described in:
/// "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al., 2023)
///
/// ## Algorithm Overview
/// For each draft token:
/// 1. Compute acceptance probability: min(1, p_target(x) / p_draft(x))
/// 2. Accept with this probability
/// 3. On rejection, sample from adjusted distribution: max(0, p_target - p_draft)
///
/// This guarantees the output distribution matches the target model exactly.
public struct VerificationSampler: Sendable {
    
    /// Random state for reproducible sampling.
    private let randomState: MLXRandom.RandomState
    
    /// Small epsilon for numerical stability.
    private let epsilon: Float = 1e-10
    
    // Initialization
    
    /// Creates a new verification sampler with a fresh random state.
    public init() {
        self.randomState = MLXRandom.RandomState()
    }
    
    /// Creates a verification sampler with a specific random seed.
    ///
    /// - Parameter seed: Seed for reproducible results
    public init(seed: UInt64) {
        self.randomState = MLXRandom.RandomState(seed: seed)
    }
    
    // Verification
    
    /// Verifies draft tokens against target model logits.
    ///
    /// This implements the core speculative decoding verification algorithm.
    /// It accepts tokens that match the target distribution and rejects
    /// tokens that diverge, sampling a correction token when needed.
    ///
    /// - Parameters:
    ///   - draftTokens: Array of token IDs proposed by the draft model
    ///   - draftLogits: Logits from the draft model for each proposed token
    ///   - targetLogits: Logits from the target model (shape: [n+1, vocab])
    ///   - temperature: Temperature for softmax (0 for greedy)
    /// - Returns: Verification result with accepted tokens and next token
    public func verify(
        draftTokens: [Int],
        draftLogits: [MLXArray],
        targetLogits: MLXArray,
        temperature: Float
    ) -> VerificationResult {
        let n = draftTokens.count
        
        // Handle greedy decoding separately (exact match)
        if temperature == 0 {
            return verifyGreedy(
                draftTokens: draftTokens,
                targetLogits: targetLogits
            )
        }
        
        var acceptedTokens: [Int] = []
        var nextToken: Int?
        var totalAcceptanceProb: Float = 0
        
        withRandomState(randomState) {
            for i in 0..<n {
                let draftToken = draftTokens[i]
                
                // Get probability distributions
                let pDraft = computeProbabilities(draftLogits[i], temperature: temperature)
                let pTarget = computeProbabilities(targetLogits[i], temperature: temperature)
                
                // Get probabilities for the specific draft token
                let pDraftToken = pDraft[draftToken].item(Float.self)
                let pTargetToken = pTarget[draftToken].item(Float.self)
                
                // Compute acceptance probability: min(1, p_target / p_draft)
                let acceptanceProb = min(1.0, pTargetToken / max(pDraftToken, epsilon))
                totalAcceptanceProb += acceptanceProb
                
                // Sample uniform random for acceptance test
                let u = MLXRandom.uniform().item(Float.self)
                
                if u < acceptanceProb {
                    // Accept the draft token
                    acceptedTokens.append(draftToken)
                } else {
                    // Reject: sample from adjusted distribution
                    nextToken = sampleAdjustedDistribution(
                        pTarget: pTarget,
                        pDraft: pDraft,
                        fallbackLogits: targetLogits[i],
                        temperature: temperature
                    )
                    break
                }
            }
            
            // If all tokens were accepted, sample next from target's final position
            if nextToken == nil {
                nextToken = sampleFromLogits(targetLogits[n], temperature: temperature)
            }
        }
        
        let avgAcceptanceProb = n > 0 ? totalAcceptanceProb / Float(n) : 0
        
        return VerificationResult(
            acceptedTokens: acceptedTokens,
            nextToken: nextToken!,
            acceptanceRate: Float(acceptedTokens.count) / Float(max(n, 1)),
            averageAcceptanceProbability: avgAcceptanceProb
        )
    }
    
    /// Verifies draft tokens using greedy (argmax) comparison.
    ///
    /// In greedy mode, a token is accepted if and only if it matches
    /// the argmax of the target logits.
    private func verifyGreedy(
        draftTokens: [Int],
        targetLogits: MLXArray
    ) -> VerificationResult {
        var acceptedTokens: [Int] = []
        var nextToken: Int?
        
        let n = draftTokens.count
        
        for i in 0..<n {
            let draftToken = draftTokens[i]
            let targetToken = argMax(targetLogits[i], axis: -1).item(Int.self)
            
            if draftToken == targetToken {
                acceptedTokens.append(draftToken)
            } else {
                nextToken = targetToken
                break
            }
        }
        
        // If all accepted, get next token from final position
        if nextToken == nil {
            nextToken = argMax(targetLogits[n], axis: -1).item(Int.self)
        }
        
        return VerificationResult(
            acceptedTokens: acceptedTokens,
            nextToken: nextToken!,
            acceptanceRate: Float(acceptedTokens.count) / Float(max(n, 1)),
            averageAcceptanceProbability: Float(acceptedTokens.count) / Float(max(n, 1))
        )
    }
    
    // Probability Helpers
    
    /// Computes softmax probabilities from logits.
    private func computeProbabilities(_ logits: MLXArray, temperature: Float) -> MLXArray {
        var logits = logits
        
        // Handle bfloat16
        if logits.dtype == .bfloat16 {
            logits = logits.asType(.float32)
        }
        
        // Ensure 1D
        if logits.ndim > 1 {
            logits = logits.squeezed()
        }
        
        return softmax(logits / temperature, axis: -1)
    }
    
    /// Samples from the adjusted distribution: max(0, p_target - p_draft).
    ///
    /// This is used when a draft token is rejected. The adjustment ensures
    /// the overall output distribution matches the target distribution.
    private func sampleAdjustedDistribution(
        pTarget: MLXArray,
        pDraft: MLXArray,
        fallbackLogits: MLXArray,
        temperature: Float
    ) -> Int {
        // Compute adjusted probabilities
        let pAdjusted = maximum(pTarget - pDraft, MLXArray(Float(0)))
        let pAdjustedSum = pAdjusted.sum().item(Float.self)
        
        if pAdjustedSum > epsilon {
            // Normalize and sample
            let pNormalized = pAdjusted / pAdjustedSum
            // Add small epsilon before log to avoid -inf
            let logProbs = log(pNormalized + epsilon)
            return categorical(logProbs).item(Int.self)
        } else {
            // Fallback: sample from target distribution
            return sampleFromLogits(fallbackLogits, temperature: temperature)
        }
    }
    
    /// Samples a token from logits with temperature.
    private func sampleFromLogits(_ logits: MLXArray, temperature: Float) -> Int {
        var logits = logits
        
        if logits.dtype == .bfloat16 {
            logits = logits.asType(.float32)
        }
        
        if logits.ndim > 1 {
            logits = logits.squeezed()
        }
        
        if temperature == 0 {
            return argMax(logits, axis: -1).item(Int.self)
        }
        
        return categorical(logits / temperature).item(Int.self)
    }
}

// Verification Result

/// Result of the speculative decoding verification step.
public struct VerificationResult: Sendable {
    /// Tokens that were accepted from the draft.
    public let acceptedTokens: [Int]
    
    /// The next token to continue generation from.
    ///
    /// This is either:
    /// - A correction token sampled from adjusted distribution (on rejection)
    /// - The next token sampled from target's final logits (if all accepted)
    public let nextToken: Int
    
    /// Acceptance rate for this verification step.
    ///
    /// Computed as: acceptedCount / draftCount
    public let acceptanceRate: Float
    
    /// Average acceptance probability across all draft tokens.
    ///
    /// This measures how well the draft model matches the target.
    public let averageAcceptanceProbability: Float
    
    /// Whether all draft tokens were accepted.
    public var allAccepted: Bool {
        acceptanceRate >= 1.0 - 1e-6
    }
    
    /// Total number of new tokens (accepted + next).
    public var totalNewTokens: Int {
        acceptedTokens.count + 1
    }
    
    /// All tokens from this verification step (accepted + next).
    public var allTokens: [Int] {
        acceptedTokens + [nextToken]
    }
}

// Batch Verification

extension VerificationSampler {
    /// Verifies multiple draft sequences in parallel (for tree speculation).
    ///
    /// - Parameters:
    ///   - draftSequences: Array of draft token sequences
    ///   - draftLogitsSequences: Logits for each draft sequence
    ///   - targetLogitsSequences: Target logits for each sequence
    ///   - temperature: Sampling temperature
    /// - Returns: Array of verification results, one per sequence
    public func verifyBatch(
        draftSequences: [[Int]],
        draftLogitsSequences: [[MLXArray]],
        targetLogitsSequences: [MLXArray],
        temperature: Float
    ) -> [VerificationResult] {
        precondition(draftSequences.count == draftLogitsSequences.count)
        precondition(draftSequences.count == targetLogitsSequences.count)
        
        return zip(zip(draftSequences, draftLogitsSequences), targetLogitsSequences)
            .map { pair, targetLogits in
                let (draftTokens, draftLogits) = pair
                return verify(
                    draftTokens: draftTokens,
                    draftLogits: draftLogits,
                    targetLogits: targetLogits,
                    temperature: temperature
                )
            }
    }
}

// Statistics

extension VerificationResult: CustomStringConvertible {
    public var description: String {
        """
        VerificationResult:
          Accepted: \(acceptedTokens.count) tokens
          Next Token: \(nextToken)
          Acceptance Rate: \(String(format: "%.1f%%", acceptanceRate * 100))
          Avg Acceptance Prob: \(String(format: "%.3f", averageAcceptanceProbability))
        """
    }
}
