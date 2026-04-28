// Copyright 2024 MLX Speculative Decoding Contributors
// SPDX-License-Identifier: MIT

import Foundation
import MLX
@preconcurrency import MLXLMCommon
import Tokenizers

/// Main generator for speculative decoding.
public struct SpeculativeGenerator: Sendable {
    
    private let modelPair: DraftTargetPair
    private let parameters: SpeculativeParameters
    private let verifier: VerificationSampler
    
    public init(
        modelPair: DraftTargetPair,
        parameters: SpeculativeParameters = .default
    ) {
        self.modelPair = modelPair
        self.parameters = parameters
        self.verifier = VerificationSampler()
    }
    
    public init(
        modelPair: DraftTargetPair,
        parameters: SpeculativeParameters,
        verifier: VerificationSampler
    ) {
        self.modelPair = modelPair
        self.parameters = parameters
        self.verifier = verifier
    }
    
    public func generate(
        input: LMInput,
        didGenerate: @escaping ([Int]) -> GenerateDisposition
    ) async throws -> SpeculativeGenerateResult {
        let draftModel = await modelPair.draftModel
        let targetModel = await modelPair.targetModel
        let tokenizer = await modelPair.tokenizer
        let configuration = await modelPair.configuration
        
        let cache = SpeculativeCache.create(
            draftModel: draftModel,
            targetModel: targetModel
        )
        
        var stats = GenerationStatistics()
        let overallStart = Date()
        
        let prefillStart = Date()
        try prefillModels(input: input, cache: cache, draftModel: draftModel, targetModel: targetModel)
        stats.promptTime = Date().timeIntervalSince(prefillStart)
        stats.promptTokenCount = input.text.tokens.size
        
        let generateStart = Date()
        var currentToken = try getInitialToken(
            input: input, cache: cache, draftModel: draftModel, targetModel: targetModel
        )
        var allTokens: [Int] = [currentToken]
        
        let eosTokenId = tokenizer.eosTokenId
        let additionalEOSTokenIds = Set(
            (configuration.extraEOSTokens).compactMap { tokenizer.convertTokenToId($0) }
        )
        
        while true {
            if let maxTokens = parameters.maxTokens, allTokens.count >= maxTokens { break }
            
            if currentToken == eosTokenId || additionalEOSTokenIds.contains(currentToken) {
                if allTokens.last == currentToken { allTokens.removeLast() }
                break
            }
            
            let (draftTokens, draftLogits) = try generateDraftTokens(
                startToken: currentToken, cache: cache, count: parameters.numDraftTokens,
                draftModel: draftModel, eosTokenId: eosTokenId
            )
            
            stats.totalDrafted += draftTokens.count
            
            var effectiveDraftTokens = draftTokens
            var effectiveDraftLogits = draftLogits
            if let eosIndex = draftTokens.firstIndex(where: { $0 == eosTokenId || additionalEOSTokenIds.contains($0) }) {
                effectiveDraftTokens = Array(draftTokens.prefix(eosIndex))
                effectiveDraftLogits = Array(draftLogits.prefix(eosIndex))
            }
            
            if effectiveDraftTokens.isEmpty {
                let nextToken = try getNextToken(previousToken: currentToken, cache: cache, targetModel: targetModel)
                allTokens.append(nextToken)
                currentToken = nextToken
                stats.verificationSteps += 1
                if didGenerate(allTokens) == .stop { break }
                continue
            }
            
            let targetLogits = try verifyBatch(
                previousToken: currentToken, draftTokens: effectiveDraftTokens,
                cache: cache, targetModel: targetModel
            )
            
            let result = verifier.verify(
                draftTokens: effectiveDraftTokens, draftLogits: effectiveDraftLogits,
                targetLogits: targetLogits, temperature: parameters.targetTemperature
            )
            
            stats.totalAccepted += result.acceptedTokens.count
            stats.verificationSteps += 1
            
            let newTokens = result.allTokens
            allTokens.append(contentsOf: newTokens)
            currentToken = result.nextToken
            
            if effectiveDraftTokens.count - result.acceptedTokens.count > 0 {
                cache.synchronizeDraftToTarget()
            }
            
            if didGenerate(allTokens) == .stop { break }
            
            if newTokens.contains(where: { $0 == eosTokenId || additionalEOSTokenIds.contains($0) }) {
                while let last = allTokens.last, last == eosTokenId || additionalEOSTokenIds.contains(last) {
                    allTokens.removeLast()
                }
                break
            }
        }
        
        stats.generateTime = Date().timeIntervalSince(generateStart)
        stats.totalTime = Date().timeIntervalSince(overallStart)
        stats.generatedTokenCount = allTokens.count
        stats.cacheStatistics = cache.statistics
        
        Stream().synchronize()
        
        return SpeculativeGenerateResult(tokens: allTokens, statistics: stats)
    }
    
    private func prefillModels(
        input: LMInput, cache: SpeculativeCache,
        draftModel: any LanguageModel, targetModel: any LanguageModel
    ) throws {
        _ = try draftModel.prepare(input, cache: cache.draftCache, windowSize: parameters.prefillStepSize)
        _ = try targetModel.prepare(input, cache: cache.targetCache, windowSize: parameters.prefillStepSize)
        cache.synchronize()
    }
    
    private func getInitialToken(
        input: LMInput, cache: SpeculativeCache,
        draftModel: any LanguageModel, targetModel: any LanguageModel
    ) throws -> Int {
        let lastTokenIdx = input.text.tokens.size - 1
        let lastToken = input.text.tokens[lastTokenIdx].item(Int32.self)
        let tokenInput = LMInput.Text(tokens: MLXArray([lastToken]))
        
        let output = targetModel(tokenInput[text: .newAxis], cache: cache.targetCache, state: nil)
        _ = draftModel(tokenInput[text: .newAxis], cache: cache.draftCache, state: nil)
        
        let logits = output.logits[0..., -1, 0...].squeezed(axis: 0)
        let token: Int
        if parameters.targetTemperature == 0 {
            token = argMax(logits, axis: -1).item(Int.self)
        } else {
            var logits32 = logits
            if logits32.dtype == .bfloat16 { logits32 = logits32.asType(.float32) }
            token = categorical(logits32 / parameters.targetTemperature).item(Int.self)
        }
        
        eval(cache.draftCache.flatMap { $0.innerState() })
        eval(cache.targetCache.flatMap { $0.innerState() })
        return token
    }
    
    private func getNextToken(
        previousToken: Int, cache: SpeculativeCache, targetModel: any LanguageModel
    ) throws -> Int {
        let input = LMInput.Text(tokens: MLXArray([Int32(previousToken)]))
        let output = targetModel(input[text: .newAxis], cache: cache.targetCache, state: nil)
        let logits = output.logits[0..., -1, 0...].squeezed(axis: 0)
        
        let token: Int
        if parameters.targetTemperature == 0 {
            token = argMax(logits, axis: -1).item(Int.self)
        } else {
            var logits32 = logits
            if logits32.dtype == .bfloat16 { logits32 = logits32.asType(.float32) }
            token = categorical(logits32 / parameters.targetTemperature).item(Int.self)
        }
        eval(cache.targetCache.flatMap { $0.innerState() })
        return token
    }
    
    private func generateDraftTokens(
        startToken: Int, cache: SpeculativeCache, count: Int,
        draftModel: any LanguageModel, eosTokenId: Int?
    ) throws -> ([Int], [MLXArray]) {
        var tokens: [Int] = []
        var logits: [MLXArray] = []
        var currentToken = startToken
        
        for _ in 0..<count {
            let input = LMInput.Text(tokens: MLXArray([Int32(currentToken)]))
            let output = draftModel(input[text: .newAxis], cache: cache.draftCache, state: nil)
            let stepLogits = output.logits[0..., -1, 0...].squeezed(axis: 0)
            logits.append(stepLogits)
            
            let nextToken: Int
            if parameters.draftTemperature == 0 {
                nextToken = argMax(stepLogits, axis: -1).item(Int.self)
            } else {
                var logits32 = stepLogits
                if logits32.dtype == .bfloat16 { logits32 = logits32.asType(.float32) }
                nextToken = categorical(logits32 / parameters.draftTemperature).item(Int.self)
            }
            
            tokens.append(nextToken)
            currentToken = nextToken
            if let eos = eosTokenId, nextToken == eos { break }
        }
        
        eval(cache.draftCache.flatMap { $0.innerState() })
        return (tokens, logits)
    }
    
    private func verifyBatch(
        previousToken: Int, draftTokens: [Int],
        cache: SpeculativeCache, targetModel: any LanguageModel
    ) throws -> MLXArray {
        let allTokens = [Int32(previousToken)] + draftTokens.map { Int32($0) }
        let input = LMInput.Text(tokens: MLXArray(allTokens))
        let output = targetModel(input[text: .newAxis], cache: cache.targetCache, state: nil)
        eval(cache.targetCache.flatMap { $0.innerState() })
        return output.logits.squeezed(axis: 0)
    }
    
    public func generateStream(input: LMInput) -> AsyncStream<SpeculativeGenerationEvent> {
        let modelPair = self.modelPair
        let selfCopy = self
        
        return AsyncStream { continuation in
            Task {
                do {
                    let tokenizer = await modelPair.tokenizer
                    var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
                    var lastTokenCount = 0
                    
                    let result = try await selfCopy.generate(input: input) { tokens in
                        for i in lastTokenCount..<tokens.count {
                            detokenizer.append(token: tokens[i])
                            if let text = detokenizer.next() { continuation.yield(.text(text)) }
                        }
                        lastTokenCount = tokens.count
                        return .more
                    }
                    continuation.yield(.result(result))
                    continuation.finish()
                } catch {
                    continuation.yield(.error(error))
                    continuation.finish()
                }
            }
        }
    }
}

public struct GenerationStatistics: Sendable {
    public var promptTokenCount: Int = 0
    public var generatedTokenCount: Int = 0
    public var totalDrafted: Int = 0
    public var totalAccepted: Int = 0
    public var verificationSteps: Int = 0
    public var promptTime: TimeInterval = 0
    public var generateTime: TimeInterval = 0
    public var totalTime: TimeInterval = 0
    public var cacheStatistics: SpeculativeCache.Statistics?
    
    public var acceptanceRate: Float {
        guard totalDrafted > 0 else { return 0 }
        return Float(totalAccepted) / Float(totalDrafted)
    }
    public var tokensPerSecond: Double {
        guard generateTime > 0 else { return 0 }
        return Double(generatedTokenCount) / generateTime
    }
    public var promptTokensPerSecond: Double {
        guard promptTime > 0 else { return 0 }
        return Double(promptTokenCount) / promptTime
    }
    public var avgTokensPerStep: Double {
        guard verificationSteps > 0 else { return 0 }
        return Double(generatedTokenCount) / Double(verificationSteps)
    }
    public var speculativeSpeedup: Double { avgTokensPerStep }
}

public struct SpeculativeGenerateResult: Sendable {
    public let tokens: [Int]
    public let statistics: GenerationStatistics
    
    public var promptTime: TimeInterval { statistics.promptTime }
    public var generateTime: TimeInterval { statistics.generateTime }
    public var totalTime: TimeInterval { statistics.totalTime }
    public var acceptanceRate: Float { statistics.acceptanceRate }
    public var tokensPerSecond: Double { statistics.tokensPerSecond }
    
    public func decode(with tokenizer: any Tokenizer) -> String {
        tokenizer.decode(tokens: tokens)
    }
    
    public func summary() -> String {
        """
        Generation Complete:
          Tokens: \(tokens.count)
          Time: \(String(format: "%.2f", totalTime))s
          Speed: \(String(format: "%.1f", tokensPerSecond)) tok/s
          Acceptance Rate: \(String(format: "%.1f%%", acceptanceRate * 100))
          Avg Tokens/Step: \(String(format: "%.2f", statistics.avgTokensPerStep))
        """
    }
}

public enum SpeculativeGenerationEvent: Sendable {
    case text(String)
    case result(SpeculativeGenerateResult)
    case error(Error)
}

struct NaiveStreamingDetokenizer {
    private let tokenizer: any Tokenizer
    private var tokens: [Int] = []
    private var lastDecodedLength: Int = 0
    
    init(tokenizer: any Tokenizer) { self.tokenizer = tokenizer }
    
    mutating func append(token: Int) { tokens.append(token) }
    
    mutating func next() -> String? {
        let decoded = tokenizer.decode(tokens: tokens)
        let newLength = decoded.count
        if newLength > lastDecodedLength {
            let startIndex = decoded.index(decoded.startIndex, offsetBy: lastDecodedLength)
            let newText = String(decoded[startIndex...])
            lastDecodedLength = newLength
            return newText
        }
        return nil
    }
}
