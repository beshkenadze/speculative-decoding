// Copyright © 2024 speculative-decoding contributors
// SPDX-License-Identifier: MIT

import Foundation
@preconcurrency import MLX
import MLXNN
@preconcurrency import MLXLMCommon
import MLXLLM
import Tokenizers

/// A Mamba-based drafter for speculative decoding.
///
/// Uses a Mamba model (SSM) as the draft model instead of a transformer.
/// Key advantages:
/// - O(1) memory per token (vs O(n) for transformer KV-cache)
/// - Linear-time inference
/// - Potentially faster draft generation
///
/// The drafter generates candidate tokens that are then verified by a transformer target model.
public final class MambaDrafter: @unchecked Sendable {

    /// The Mamba model used for drafting
    public let model: MambaLM

    /// Model configuration
    public let config: MambaConfig

    /// Current cache state
    private var cache: MambaCache

    /// Tokenizer for the model
    public let tokenizer: Tokenizer

    /// Lock for thread safety
    private let lock = NSLock()

    /// Initialize with a loaded model
    public init(model: MambaLM, config: MambaConfig, tokenizer: Tokenizer) {
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.cache = MambaCache(config: config, batchSize: 1)
    }

    /// Load a Mamba drafter from HuggingFace
    public static func load(modelId: String) async throws -> MambaDrafter {
        let (model, config, modelDirectory) = try await MambaLoader.load(modelId: modelId)
        // Load tokenizer from the model directory
        let tokenizer = try await AutoTokenizer.from(modelFolder: modelDirectory)
        return MambaDrafter(model: model, config: config, tokenizer: tokenizer)
    }

    /// Load a pretrained Mamba drafter
    public static func loadPretrained(_ preset: MambaLoader.PretrainedModel) async throws -> MambaDrafter {
        try await load(modelId: preset.modelId)
    }

    // MARK: - Draft Generation

    /// Reset the cache state
    public func reset() {
        lock.lock()
        defer { lock.unlock() }
        cache.reset()
    }

    /// Prefill with input tokens
    public func prefill(tokens: MLXArray) -> MLXArray {
        lock.lock()
        defer { lock.unlock() }

        let input = tokens.reshaped(1, -1)
        let logits = model.forward(input, cache: cache)
        eval(logits)
        return logits
    }

    /// Generate a single draft token given current state
    public func generateToken(previousToken: Int, temperature: Float = 0.6) -> (token: Int, logits: MLXArray) {
        lock.lock()
        defer { lock.unlock() }

        let input = MLXArray([Int32(previousToken)]).reshaped(1, 1)
        let logits = model.forward(input, cache: cache)
        eval(logits)

        let token = sampleToken(logits.squeezed(), temperature: temperature)
        return (token, logits.squeezed())
    }

    /// Generate multiple draft tokens
    public func generateDraftTokens(
        startToken: Int,
        count: Int,
        temperature: Float = 0.6,
        eosTokenId: Int? = nil
    ) -> (tokens: [Int], logits: [MLXArray]) {
        var tokens: [Int] = []
        var logits: [MLXArray] = []
        var currentToken = startToken

        for _ in 0..<count {
            let (token, stepLogits) = generateToken(previousToken: currentToken, temperature: temperature)
            tokens.append(token)
            logits.append(stepLogits)
            currentToken = token

            if let eos = eosTokenId, token == eos {
                break
            }
        }

        return (tokens, logits)
    }

    /// Sample a token from logits
    private func sampleToken(_ logits: MLXArray, temperature: Float) -> Int {
        let logits = logits

        if temperature == 0 {
            return argMax(logits, axis: -1).item(Int.self)
        }

        var logits32 = logits
        if logits32.dtype == .bfloat16 {
            logits32 = logits32.asType(.float32)
        }

        return categorical(logits32 / temperature).item(Int.self)
    }

    // MARK: - State Management for Speculative Decoding

    /// Save current cache state for potential rollback
    public func saveState() -> [MambaCache.LayerState] {
        lock.lock()
        defer { lock.unlock() }
        return cache.snapshot()
    }

    /// Restore cache state (rollback)
    public func restoreState(_ state: [MambaCache.LayerState]) {
        lock.lock()
        defer { lock.unlock() }
        cache.restore(from: state)
    }

    /// Get current sequence position
    public var currentOffset: Int {
        cache.offset
    }

    /// Cache statistics
    public var cacheStatistics: MambaCache.Statistics {
        cache.statistics()
    }
}

// MARK: - MambaDraftTargetPair

/// A model pair using Mamba as the draft model and a transformer as the target.
public final class MambaDraftTargetPair: @unchecked Sendable {

    /// The Mamba draft model
    public let drafter: MambaDrafter

    /// The transformer target model container
    public let targetContainer: ModelContainer

    /// Initialize with loaded models
    public init(drafter: MambaDrafter, targetContainer: ModelContainer) {
        self.drafter = drafter
        self.targetContainer = targetContainer
    }

    /// Load a Mamba-Transformer pair
    ///
    /// Uses the target model's tokenizer for both models since:
    /// 1. Mamba's GPTNeoXTokenizer may not be supported
    /// 2. For speculative decoding, token IDs must match between models
    ///
    /// - Parameters:
    ///   - draftModelId: HuggingFace ID for Mamba model (e.g., "state-spaces/mamba-130m-hf")
    ///   - targetModelId: HuggingFace ID for target transformer
    public static func load(
        draftModelId: String,
        targetModelId: String
    ) async throws -> MambaDraftTargetPair {
        // Load transformer target first to get its tokenizer
        let targetContainer = try await LLMModelFactory.shared.loadContainer(
            configuration: ModelConfiguration(id: targetModelId)
        )

        // Get the target tokenizer
        let targetTokenizer = await targetContainer.perform { $0.tokenizer }

        // Load Mamba model (without its own tokenizer)
        let (model, config, _) = try await MambaLoader.load(modelId: draftModelId)

        // Create drafter using target's tokenizer
        let drafter = MambaDrafter(model: model, config: config, tokenizer: targetTokenizer)

        return MambaDraftTargetPair(drafter: drafter, targetContainer: targetContainer)
    }

    /// Access the target model
    public func withTargetModel<R: Sendable>(_ action: @Sendable (any LanguageModel) async throws -> R) async rethrows -> R {
        try await targetContainer.perform { context in
            try await action(context.model)
        }
    }

    /// Get the target model's tokenizer
    public var targetTokenizer: Tokenizer {
        get async {
            await targetContainer.perform { $0.tokenizer }
        }
    }

    /// Get the target model configuration
    public var targetConfiguration: ModelConfiguration {
        get async {
            await targetContainer.configuration
        }
    }
}

// MARK: - Mamba Speculative Generator

/// Speculative decoding generator using Mamba draft model.
public struct MambaSpeculativeGenerator: Sendable {

    private let modelPair: MambaDraftTargetPair
    private let parameters: SpeculativeParameters
    private let verifier: VerificationSampler

    public init(
        modelPair: MambaDraftTargetPair,
        parameters: SpeculativeParameters = .default
    ) {
        self.modelPair = modelPair
        self.parameters = parameters
        self.verifier = VerificationSampler()
    }

    /// Generate tokens using Mamba-based speculative decoding
    public func generate(
        prompt: String,
        didGenerate: @escaping ([Int]) -> GenerateDisposition
    ) async throws -> SpeculativeGenerateResult {
        let tokenizer = await modelPair.targetTokenizer

        // Tokenize prompt
        let promptTokens = tokenizer.encode(text: prompt)
        let promptArray = MLXArray(promptTokens.map { Int32($0) })

        // Reset drafter
        modelPair.drafter.reset()

        // Create target cache
        let targetCache = await modelPair.withTargetModel { model in
            model.newCache(parameters: nil)
        }

        var stats = GenerationStatistics()
        let overallStart = Date()

        // Prefill
        let prefillStart = Date()
        _ = modelPair.drafter.prefill(tokens: promptArray)

        // Prefill target
        let targetInput = LMInput(text: LMInput.Text(tokens: promptArray))
        try await modelPair.withTargetModel { model in
            _ = try model.prepare(targetInput, cache: targetCache, windowSize: parameters.prefillStepSize)
        }

        stats.promptTime = Date().timeIntervalSince(prefillStart)
        stats.promptTokenCount = promptTokens.count

        // Get initial token from target
        let generateStart = Date()
        let configuration = await modelPair.targetConfiguration

        var currentToken = await getInitialToken(
            promptTokens: promptArray,
            targetCache: targetCache
        )

        var allTokens: [Int] = [currentToken]

        let eosTokenId = tokenizer.eosTokenId
        let additionalEOSTokenIds = Set(
            (configuration.extraEOSTokens).compactMap { tokenizer.convertTokenToId($0) }
        )

        // Main generation loop
        while true {
            if let maxTokens = parameters.maxTokens, allTokens.count >= maxTokens { break }

            if currentToken == eosTokenId || additionalEOSTokenIds.contains(currentToken) {
                if allTokens.last == currentToken { allTokens.removeLast() }
                break
            }

            // Save drafter state for potential rollback
            let savedState = modelPair.drafter.saveState()

            // Generate draft tokens using Mamba
            let (draftTokens, draftLogits) = modelPair.drafter.generateDraftTokens(
                startToken: currentToken,
                count: parameters.numDraftTokens,
                temperature: parameters.draftTemperature,
                eosTokenId: eosTokenId
            )

            stats.totalDrafted += draftTokens.count

            // Filter out EOS tokens from draft
            var effectiveDraftTokens = draftTokens
            var effectiveDraftLogits = draftLogits
            if let eosIndex = draftTokens.firstIndex(where: { $0 == eosTokenId || additionalEOSTokenIds.contains($0) }) {
                effectiveDraftTokens = Array(draftTokens.prefix(eosIndex))
                effectiveDraftLogits = Array(draftLogits.prefix(eosIndex))
            }

            if effectiveDraftTokens.isEmpty {
                // No draft tokens, fall back to regular generation
                let nextToken = await getNextToken(
                    previousToken: currentToken,
                    targetCache: targetCache
                )
                allTokens.append(nextToken)
                currentToken = nextToken
                stats.verificationSteps += 1
                if didGenerate(allTokens) == .stop { break }
                continue
            }

            // Verify draft tokens with target model
            let targetLogits = await verifyBatch(
                previousToken: currentToken,
                draftTokens: effectiveDraftTokens,
                targetCache: targetCache
            )

            // Run verification
            let result = verifier.verify(
                draftTokens: effectiveDraftTokens,
                draftLogits: effectiveDraftLogits,
                targetLogits: targetLogits,
                temperature: parameters.targetTemperature
            )

            stats.totalAccepted += result.acceptedTokens.count
            stats.verificationSteps += 1

            let newTokens = result.allTokens
            allTokens.append(contentsOf: newTokens)
            currentToken = result.nextToken

            // Handle rejection: rollback drafter state
            if effectiveDraftTokens.count - result.acceptedTokens.count > 0 {
                // Restore drafter to saved state
                modelPair.drafter.restoreState(savedState)

                // Re-process only accepted tokens
                for token in result.acceptedTokens {
                    _ = modelPair.drafter.generateToken(previousToken: token, temperature: 0)
                }
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

        Stream().synchronize()

        return SpeculativeGenerateResult(tokens: allTokens, statistics: stats)
    }

    private func getInitialToken(
        promptTokens: MLXArray,
        targetCache: [KVCache]
    ) async -> Int {
        await modelPair.withTargetModel { model in
            let lastTokenIdx = promptTokens.size - 1
            let lastToken = promptTokens[lastTokenIdx].item(Int32.self)
            let tokenInput = LMInput.Text(tokens: MLXArray([lastToken]))

            let output = model(tokenInput[text: .newAxis], cache: targetCache, state: nil)
            let logits = output.logits[0..., -1, 0...].squeezed(axis: 0)

            let token: Int
            if parameters.targetTemperature == 0 {
                token = argMax(logits, axis: -1).item(Int.self)
            } else {
                var logits32 = logits
                if logits32.dtype == .bfloat16 { logits32 = logits32.asType(.float32) }
                token = categorical(logits32 / parameters.targetTemperature).item(Int.self)
            }

            eval(targetCache.flatMap { $0.innerState() })
            return token
        }
    }

    private func getNextToken(
        previousToken: Int,
        targetCache: [KVCache]
    ) async -> Int {
        await modelPair.withTargetModel { model in
            let input = LMInput.Text(tokens: MLXArray([Int32(previousToken)]))
            let output = model(input[text: .newAxis], cache: targetCache, state: nil)
            let logits = output.logits[0..., -1, 0...].squeezed(axis: 0)

            let token: Int
            if parameters.targetTemperature == 0 {
                token = argMax(logits, axis: -1).item(Int.self)
            } else {
                var logits32 = logits
                if logits32.dtype == .bfloat16 { logits32 = logits32.asType(.float32) }
                token = categorical(logits32 / parameters.targetTemperature).item(Int.self)
            }

            eval(targetCache.flatMap { $0.innerState() })
            return token
        }
    }

    private func verifyBatch(
        previousToken: Int,
        draftTokens: [Int],
        targetCache: [KVCache]
    ) async -> MLXArray {
        await modelPair.withTargetModel { model in
            let allTokens = [Int32(previousToken)] + draftTokens.map { Int32($0) }
            let input = LMInput.Text(tokens: MLXArray(allTokens))
            let output = model(input[text: .newAxis], cache: targetCache, state: nil)
            eval(targetCache.flatMap { $0.innerState() })
            return output.logits.squeezed(axis: 0)
        }
    }
}

// MARK: - Public API Extension

extension MambaSpeculativeGenerator {

    /// Generate with streaming output
    public func generateStream(prompt: String) -> AsyncStream<SpeculativeGenerationEvent> {
        let modelPair = self.modelPair
        let selfCopy = self

        return AsyncStream { continuation in
            Task {
                do {
                    let tokenizer = await modelPair.targetTokenizer
                    var detokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
                    var lastTokenCount = 0

                    let result = try await selfCopy.generate(prompt: prompt) { tokens in
                        for i in lastTokenCount..<tokens.count {
                            detokenizer.append(token: tokens[i])
                            if let text = detokenizer.next() {
                                continuation.yield(.text(text))
                            }
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
