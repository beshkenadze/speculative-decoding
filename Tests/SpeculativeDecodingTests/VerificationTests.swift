// Copyright 2024 MLX Speculative Decoding Contributors
// SPDX-License-Identifier: MIT

import XCTest
import MLX
import MLXRandom
@testable import SpeculativeDecoding

final class VerificationTests: XCTestCase {
    
    // Greedy Verification Tests
    
    func testGreedyVerificationAllAccepted() {
        // Setup: Draft tokens match target's argmax
        let verifier = VerificationSampler(seed: 42)
        
        // Create logits where draft tokens are the argmax
        let vocabSize = 100
        let draftTokens = [10, 20, 30]
        
        // Create draft logits (draft tokens are argmax)
        var draftLogits: [MLXArray] = []
        for token in draftTokens {
            var logits = MLXArray.zeros([vocabSize])
            logits[token] = MLXArray([Float(10.0)])  // High logit for draft token
            draftLogits.append(logits)
        }
        
        // Create target logits (same argmax as draft)
        var targetLogitsData: [[Float]] = []
        for token in draftTokens {
            var logits = [Float](repeating: 0, count: vocabSize)
            logits[token] = 10.0  // Same argmax
            targetLogitsData.append(logits)
        }
        // Add final position logits
        var finalLogits = [Float](repeating: 0, count: vocabSize)
        finalLogits[50] = 10.0  // Next token will be 50
        targetLogitsData.append(finalLogits)
        
        let targetLogits = MLXArray(targetLogitsData)
        
        // Verify with temperature=0 (greedy)
        let result = verifier.verify(
            draftTokens: draftTokens,
            draftLogits: draftLogits,
            targetLogits: targetLogits,
            temperature: 0
        )
        
        XCTAssertEqual(result.acceptedTokens, draftTokens, "All draft tokens should be accepted")
        XCTAssertEqual(result.nextToken, 50, "Next token should be from final position")
        XCTAssertEqual(result.acceptanceRate, 1.0, accuracy: 0.001)
    }
    
    func testGreedyVerificationPartialAcceptance() {
        let verifier = VerificationSampler(seed: 42)
        
        let vocabSize = 100
        let draftTokens = [10, 20, 30]  // Draft proposes these
        let targetArgmax = [10, 25, 35]  // Target has different argmax at positions 1 and 2
        
        // Create draft logits
        var draftLogits: [MLXArray] = []
        for token in draftTokens {
            var logits = MLXArray.zeros([vocabSize])
            logits[token] = MLXArray([Float(10.0)])
            draftLogits.append(logits)
        }
        
        // Create target logits with different argmax
        var targetLogitsData: [[Float]] = []
        for token in targetArgmax {
            var logits = [Float](repeating: 0, count: vocabSize)
            logits[token] = 10.0
            targetLogitsData.append(logits)
        }
        // Final position
        var finalLogits = [Float](repeating: 0, count: vocabSize)
        finalLogits[99] = 10.0
        targetLogitsData.append(finalLogits)
        
        let targetLogits = MLXArray(targetLogitsData)
        
        let result = verifier.verify(
            draftTokens: draftTokens,
            draftLogits: draftLogits,
            targetLogits: targetLogits,
            temperature: 0
        )
        
        // First token (10) matches, second (20 vs 25) doesn't
        XCTAssertEqual(result.acceptedTokens, [10], "Only first token should be accepted")
        XCTAssertEqual(result.nextToken, 25, "Next token should be target's argmax at rejection point")
        XCTAssertEqual(result.acceptanceRate, 1.0/3.0, accuracy: 0.001)
    }
    
    func testGreedyVerificationFirstRejected() {
        let verifier = VerificationSampler(seed: 42)
        
        let vocabSize = 50
        let draftTokens = [5, 10, 15]
        
        // Draft logits
        var draftLogits: [MLXArray] = []
        for token in draftTokens {
            var logits = MLXArray.zeros([vocabSize])
            logits[token] = MLXArray([Float(10.0)])
            draftLogits.append(logits)
        }
        
        // Target logits - first position has different argmax
        var targetLogitsData: [[Float]] = []
        var firstLogits = [Float](repeating: 0, count: vocabSize)
        firstLogits[99 % vocabSize] = 10.0  // Different from draft
        targetLogitsData.append(firstLogits)
        
        // Add remaining positions
        for _ in 1..<draftTokens.count + 1 {
            var logits = [Float](repeating: 0, count: vocabSize)
            logits[0] = 10.0
            targetLogitsData.append(logits)
        }
        
        let targetLogits = MLXArray(targetLogitsData)
        
        let result = verifier.verify(
            draftTokens: draftTokens,
            draftLogits: draftLogits,
            targetLogits: targetLogits,
            temperature: 0
        )
        
        XCTAssertEqual(result.acceptedTokens.count, 0, "No tokens should be accepted")
        XCTAssertEqual(result.acceptanceRate, 0.0, accuracy: 0.001)
    }
    
    // Stochastic Verification Tests
    
    func testStochasticVerificationBasic() {
        let verifier = VerificationSampler(seed: 123)
        
        let vocabSize = 10
        let draftTokens = [0]
        
        // Draft: uniform distribution
        let draftLogits = [MLXArray([Float](repeating: 0, count: vocabSize))]
        
        // Target: also uniform
        var targetLogitsData: [[Float]] = []
        targetLogitsData.append([Float](repeating: 0, count: vocabSize))
        targetLogitsData.append([Float](repeating: 0, count: vocabSize))
        let targetLogits = MLXArray(targetLogitsData)
        
        // With uniform distributions, acceptance probability should be 1.0
        // for any token (p_target / p_draft = 1)
        let result = verifier.verify(
            draftTokens: draftTokens,
            draftLogits: draftLogits,
            targetLogits: targetLogits,
            temperature: 1.0
        )
        
        // With identical distributions, token should be accepted
        XCTAssertEqual(result.acceptedTokens.count, 1, "Token should be accepted with matching distributions")
    }
    
    // Verification Result Tests
    
    func testVerificationResultProperties() {
        let result = VerificationResult(
            acceptedTokens: [1, 2, 3],
            nextToken: 4,
            acceptanceRate: 0.75,
            averageAcceptanceProbability: 0.8
        )
        
        XCTAssertEqual(result.allTokens, [1, 2, 3, 4])
        XCTAssertEqual(result.totalNewTokens, 4)
        XCTAssertFalse(result.allAccepted)  // 0.75 < 1.0
        
        let perfectResult = VerificationResult(
            acceptedTokens: [1, 2],
            nextToken: 3,
            acceptanceRate: 1.0,
            averageAcceptanceProbability: 1.0
        )
        
        XCTAssertTrue(perfectResult.allAccepted)
    }
    
    // Edge Cases
    
    func testEmptyDraftTokens() {
        let verifier = VerificationSampler(seed: 42)
        
        let vocabSize = 10
        let draftTokens: [Int] = []
        let draftLogits: [MLXArray] = []
        
        // Target logits for position 0 only (next token position)
        var logits = [Float](repeating: 0, count: vocabSize)
        logits[5] = 10.0
        let targetLogits = MLXArray([logits])
        
        let result = verifier.verify(
            draftTokens: draftTokens,
            draftLogits: draftLogits,
            targetLogits: targetLogits,
            temperature: 0
        )
        
        XCTAssertEqual(result.acceptedTokens.count, 0)
        XCTAssertEqual(result.nextToken, 5)
    }
    
    func testSingleDraftToken() {
        let verifier = VerificationSampler(seed: 42)
        
        let vocabSize = 10
        let draftTokens = [3]
        
        var logits = MLXArray.zeros([vocabSize])
        logits[3] = MLXArray([Float(10.0)])
        let draftLogits = [logits]
        
        var targetData: [[Float]] = []
        var t0 = [Float](repeating: 0, count: vocabSize)
        t0[3] = 10.0  // Match draft
        targetData.append(t0)
        var t1 = [Float](repeating: 0, count: vocabSize)
        t1[7] = 10.0  // Next token
        targetData.append(t1)
        
        let targetLogits = MLXArray(targetData)
        
        let result = verifier.verify(
            draftTokens: draftTokens,
            draftLogits: draftLogits,
            targetLogits: targetLogits,
            temperature: 0
        )
        
        XCTAssertEqual(result.acceptedTokens, [3])
        XCTAssertEqual(result.nextToken, 7)
        XCTAssertEqual(result.acceptanceRate, 1.0, accuracy: 0.001)
    }
}

// Speculative Parameters Tests

final class SpeculativeParametersTests: XCTestCase {
    
    func testDefaultParameters() {
        let params = SpeculativeParameters.default
        
        XCTAssertEqual(params.numDraftTokens, 5)
        XCTAssertEqual(params.draftTemperature, 0.6, accuracy: 0.001)
        XCTAssertEqual(params.targetTemperature, 0.6, accuracy: 0.001)
        XCTAssertFalse(params.useGreedyDecoding)
    }
    
    func testGreedyParameters() {
        let params = SpeculativeParameters.greedy
        
        XCTAssertEqual(params.draftTemperature, 0)
        XCTAssertEqual(params.targetTemperature, 0)
        XCTAssertTrue(params.useGreedyDecoding)
    }
    
    func testCreativeParameters() {
        let params = SpeculativeParameters.creative
        
        XCTAssertEqual(params.draftTemperature, 0.9, accuracy: 0.001)
        XCTAssertEqual(params.targetTemperature, 0.9, accuracy: 0.001)
        XCTAssertEqual(params.draftTopP, 0.95, accuracy: 0.001)
    }
    
    func testCustomParameters() {
        let params = SpeculativeParameters(
            numDraftTokens: 8,
            draftTemperature: 0.5,
            targetTemperature: 0.7,
            maxTokens: 100
        )
        
        XCTAssertEqual(params.numDraftTokens, 8)
        XCTAssertEqual(params.draftTemperature, 0.5, accuracy: 0.001)
        XCTAssertEqual(params.targetTemperature, 0.7, accuracy: 0.001)
        XCTAssertEqual(params.maxTokens, 100)
    }
    
    func testGenerateParametersConversion() {
        let params = SpeculativeParameters(
            numDraftTokens: 5,
            draftTemperature: 0.5,
            targetTemperature: 0.8,
            draftTopP: 0.85,
            targetTopP: 0.95,
            maxTokens: 200
        )
        
        let draftParams = params.draftGenerateParameters()
        XCTAssertEqual(draftParams.temperature, 0.5, accuracy: 0.001)
        XCTAssertEqual(draftParams.topP, 0.85, accuracy: 0.001)
        
        let targetParams = params.targetGenerateParameters()
        XCTAssertEqual(targetParams.temperature, 0.8, accuracy: 0.001)
        XCTAssertEqual(targetParams.topP, 0.95, accuracy: 0.001)
    }
}

// Cache Tests

final class SpeculativeCacheTests: XCTestCase {
    
    func testCacheDebugState() {
        // This is a basic structural test since we can't easily create real KVCache instances
        // without loading actual models
        
        let state = SpeculativeCache.CacheDebugState(
            draftOffset: 100,
            targetOffset: 95,
            draftLayerCount: 32,
            targetLayerCount: 32,
            rollbackCount: 2
        )
        
        XCTAssertEqual(state.draftOffset, 100)
        XCTAssertEqual(state.targetOffset, 95)
        XCTAssertEqual(state.rollbackCount, 2)
        
        let description = state.description
        XCTAssertTrue(description.contains("100"))
        XCTAssertTrue(description.contains("95"))
        XCTAssertTrue(description.contains("Synchronized: false"))
    }
    
    func testCacheStatistics() {
        let stats = SpeculativeCache.Statistics(
            rollbackCount: 5,
            finalDraftOffset: 200,
            finalTargetOffset: 200
        )
        
        XCTAssertEqual(stats.rollbackCount, 5)
        XCTAssertEqual(stats.finalDraftOffset, 200)
        XCTAssertEqual(stats.finalTargetOffset, 200)
        XCTAssertGreaterThan(stats.estimatedMemoryBytes, 0)
    }
}

// Generation Statistics Tests

final class GenerationStatisticsTests: XCTestCase {
    
    func testStatisticsCalculations() {
        var stats = GenerationStatistics()
        stats.promptTokenCount = 50
        stats.generatedTokenCount = 100
        stats.totalDrafted = 120
        stats.totalAccepted = 90
        stats.verificationSteps = 25
        stats.promptTime = 0.5
        stats.generateTime = 2.0
        stats.totalTime = 2.5
        
        XCTAssertEqual(stats.acceptanceRate, 0.75, accuracy: 0.001)
        XCTAssertEqual(stats.tokensPerSecond, 50.0, accuracy: 0.001)
        XCTAssertEqual(stats.promptTokensPerSecond, 100.0, accuracy: 0.001)
        XCTAssertEqual(stats.avgTokensPerStep, 4.0, accuracy: 0.001)
        XCTAssertEqual(stats.speculativeSpeedup, 4.0, accuracy: 0.001)
    }
    
    func testStatisticsZeroDivision() {
        let stats = GenerationStatistics()
        
        XCTAssertEqual(stats.acceptanceRate, 0)
        XCTAssertEqual(stats.tokensPerSecond, 0)
        XCTAssertEqual(stats.avgTokensPerStep, 0)
    }
}
