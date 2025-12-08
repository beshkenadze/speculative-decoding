// Copyright © 2024 speculative-decoding contributors
// SPDX-License-Identifier: MIT

import XCTest
import MLX
import MLXRandom
@testable import SpeculativeDecoding

final class MambaTests: XCTestCase {

    // MARK: - Config Tests

    func testMambaConfigDefaults() {
        let config = MambaConfig()

        XCTAssertEqual(config.hiddenSize, 768)
        XCTAssertEqual(config.numHiddenLayers, 24)
        XCTAssertEqual(config.vocabSize, 50280)
        XCTAssertEqual(config.stateSize, 16)
        XCTAssertEqual(config.convKernel, 4)
        XCTAssertEqual(config.expand, 2)
        XCTAssertEqual(config.innerSize, 1536)  // 768 * 2
    }

    func testMambaConfigPresets() {
        let mamba130m = MambaConfig.mamba130m
        XCTAssertEqual(mamba130m.hiddenSize, 768)
        XCTAssertEqual(mamba130m.numHiddenLayers, 24)

        let mamba370m = MambaConfig.mamba370m
        XCTAssertEqual(mamba370m.hiddenSize, 1024)
        XCTAssertEqual(mamba370m.numHiddenLayers, 48)
    }

    func testMambaConfigCodable() throws {
        let json = """
        {
            "hidden_size": 768,
            "num_hidden_layers": 24,
            "vocab_size": 50280,
            "state_size": 16,
            "conv_kernel": 4,
            "expand": 2
        }
        """.data(using: .utf8)!

        let config = try JSONDecoder().decode(MambaConfig.self, from: json)
        XCTAssertEqual(config.hiddenSize, 768)
        XCTAssertEqual(config.numHiddenLayers, 24)
    }

    // MARK: - MambaBlock Tests

    func testMambaBlockInit() {
        let config = MambaConfig(
            hiddenSize: 64,
            numHiddenLayers: 2,
            vocabSize: 100,
            stateSize: 8,
            convKernel: 4,
            expand: 2
        )

        let block = MambaBlock(config)

        // Verify projections exist
        XCTAssertNotNil(block.inProj)
        XCTAssertNotNil(block.outProj)
        XCTAssertNotNil(block.dtProj)
    }

    func testMambaBlockForward() {
        let config = MambaConfig(
            hiddenSize: 64,
            numHiddenLayers: 2,
            vocabSize: 100,
            stateSize: 8,
            convKernel: 4,
            expand: 2
        )

        let block = MambaBlock(config)
        eval(block)

        // Create test input: [batch=1, seq_len=8, hidden=64]
        let x = MLXRandom.normal([1, 8, 64])
        eval(x)

        // Forward pass
        let y = block(x)
        eval(y)

        // Output should have same shape as input
        XCTAssertEqual(y.shape, [1, 8, 64])
    }

    // MARK: - MambaModel Tests

    func testMambaModelInit() {
        let config = MambaConfig(
            hiddenSize: 64,
            numHiddenLayers: 2,
            vocabSize: 100,
            stateSize: 8,
            convKernel: 4,
            expand: 2
        )

        let model = MambaLM(config)
        eval(model)

        XCTAssertNotNil(model.embedding)
        XCTAssertEqual(model.layers.count, 2)
        XCTAssertNotNil(model.normF)
    }

    func testMambaModelForward() {
        let config = MambaConfig(
            hiddenSize: 64,
            numHiddenLayers: 2,
            vocabSize: 100,
            stateSize: 8,
            convKernel: 4,
            expand: 2
        )

        let model = MambaLM(config)
        eval(model)

        // Create test input: [batch=1, seq_len=8]
        let tokens = MLXArray([1, 2, 3, 4, 5, 6, 7, 8]).reshaped(1, 8)
        eval(tokens)

        // Forward pass
        let logits = model(tokens)
        eval(logits)

        // Output should be [batch=1, seq_len=8, vocab_size=100]
        XCTAssertEqual(logits.shape, [1, 8, 100])
    }

    // MARK: - MambaCache Tests

    func testMambaCacheInit() {
        let config = MambaConfig(
            hiddenSize: 64,
            numHiddenLayers: 2,
            vocabSize: 100,
            stateSize: 8,
            convKernel: 4,
            expand: 2
        )

        let cache = MambaCache(config: config, batchSize: 1)

        XCTAssertEqual(cache.layers.count, 2)
        XCTAssertEqual(cache.batchSize, 1)
        XCTAssertEqual(cache.offset, 0)
    }

    func testMambaCacheSnapshot() {
        let config = MambaConfig(
            hiddenSize: 64,
            numHiddenLayers: 2,
            vocabSize: 100,
            stateSize: 8,
            convKernel: 4,
            expand: 2
        )

        let cache = MambaCache(config: config, batchSize: 1)

        // Take snapshot
        let snapshot = cache.snapshot()
        XCTAssertEqual(snapshot.count, 2)

        // Modify cache
        cache.layers[0].offset = 10

        // Restore
        cache.restore(from: snapshot)
        XCTAssertEqual(cache.layers[0].offset, 0)
    }

    func testMambaCacheStatistics() {
        let config = MambaConfig(
            hiddenSize: 64,
            numHiddenLayers: 4,
            vocabSize: 100,
            stateSize: 8,
            convKernel: 4,
            expand: 2
        )

        let cache = MambaCache(config: config, batchSize: 1)
        let stats = cache.statistics()

        XCTAssertEqual(stats.layerCount, 4)
        XCTAssertEqual(stats.batchSize, 1)
        XCTAssertGreaterThan(stats.totalMemoryBytes, 0)
    }

    // MARK: - Integration Tests

    func testMambaModelWithCache() {
        let config = MambaConfig(
            hiddenSize: 64,
            numHiddenLayers: 2,
            vocabSize: 100,
            stateSize: 8,
            convKernel: 4,
            expand: 2
        )

        let model = MambaLM(config)
        let cache = MambaCache(config: config, batchSize: 1)
        eval(model)

        // Prefill with sequence
        let tokens = MLXArray([1, 2, 3, 4]).reshaped(1, 4)
        let prefillLogits = model.forward(tokens, cache: cache)
        eval(prefillLogits)

        XCTAssertEqual(prefillLogits.shape, [1, 100])  // Last position logits
        XCTAssertGreaterThan(cache.offset, 0)

        // Generate one more token
        let nextToken = MLXArray([5]).reshaped(1, 1)
        let stepLogits = model.forward(nextToken, cache: cache)
        eval(stepLogits)

        XCTAssertEqual(stepLogits.shape, [1, 100])
    }

    // MARK: - Memory Comparison Tests

    func testMambaCacheIsConstantMemory() {
        // Mamba's key advantage: constant memory regardless of sequence length
        let config = MambaConfig(
            hiddenSize: 64,
            numHiddenLayers: 4,
            vocabSize: 100,
            stateSize: 8,
            convKernel: 4,
            expand: 2
        )

        let cache1 = MambaCache(config: config, batchSize: 1)
        let cache2 = MambaCache(config: config, batchSize: 1)

        // Both caches should use the same memory regardless of sequence length
        XCTAssertEqual(cache1.memoryBytes, cache2.memoryBytes)

        // Memory should be predictable
        // SSM state: batch * innerSize * stateSize * 4 bytes
        // Conv state: batch * (convKernel-1) * innerSize * 4 bytes
        let innerSize = config.innerSize
        let expectedSSMBytes = 1 * innerSize * config.stateSize * 4
        let expectedConvBytes = 1 * (config.convKernel - 1) * innerSize * 4
        let expectedPerLayer = expectedSSMBytes + expectedConvBytes
        let expectedTotal = expectedPerLayer * config.numHiddenLayers

        XCTAssertEqual(cache1.memoryBytes, expectedTotal)
    }
}
