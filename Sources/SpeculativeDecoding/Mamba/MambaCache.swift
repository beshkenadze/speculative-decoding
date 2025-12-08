// Copyright © 2024 speculative-decoding contributors
// SPDX-License-Identifier: MIT

import Foundation
@preconcurrency import MLX

/// Cache for Mamba inference state.
///
/// Unlike transformer KV-cache which grows with sequence length, Mamba's cache is constant-size:
/// - SSM hidden state: [B, ED, N] - the recurrent state
/// - Conv inputs: [B, k-1, ED] - last (kernelSize-1) inputs for causal convolution
///
/// This enables O(1) memory per token during generation, a key advantage over transformers.
public final class MambaCache: @unchecked Sendable {

    /// State for a single Mamba layer
    public struct LayerState: Sendable {
        /// SSM hidden state: [B, ED, N]
        public var ssmState: MLXArray

        /// Convolution input cache: [B, k-1, ED]
        public var convState: MLXArray

        /// Current sequence position
        public var offset: Int

        public init(ssmState: MLXArray, convState: MLXArray, offset: Int = 0) {
            self.ssmState = ssmState
            self.convState = convState
            self.offset = offset
        }

        /// Create empty state for a given batch size and config
        public static func empty(batchSize: Int, config: MambaConfig) -> LayerState {
            LayerState(
                ssmState: MLXArray.zeros([batchSize, config.innerSize, config.stateSize]),
                convState: MLXArray.zeros([batchSize, config.convKernel - 1, config.innerSize]),
                offset: 0
            )
        }
    }

    /// Per-layer states
    public var layers: [LayerState]

    /// Configuration
    public let config: MambaConfig

    /// Batch size
    public let batchSize: Int

    /// Lock for thread-safe access
    private let lock = NSLock()

    /// Initialize empty cache for a model
    public init(config: MambaConfig, batchSize: Int = 1) {
        self.config = config
        self.batchSize = batchSize
        self.layers = (0..<config.numHiddenLayers).map { _ in
            LayerState.empty(batchSize: batchSize, config: config)
        }
    }

    /// Reset all layer states to zero
    public func reset() {
        lock.lock()
        defer { lock.unlock() }

        for i in 0..<layers.count {
            layers[i] = LayerState.empty(batchSize: batchSize, config: config)
        }
    }

    /// Reset to a specific position (for speculative decoding rollback)
    public func resetToPosition(_ position: Int) {
        lock.lock()
        defer { lock.unlock() }

        // For Mamba, we can't easily rollback state like KV-cache
        // We need to re-process from beginning or checkpoint
        // For speculative decoding, we'll use a different strategy
        for i in 0..<layers.count {
            layers[i].offset = position
        }
    }

    /// Create a snapshot of current state (for speculative decoding)
    public func snapshot() -> [LayerState] {
        lock.lock()
        defer { lock.unlock() }
        return layers.map { layer in
            LayerState(
                ssmState: layer.ssmState,
                convState: layer.convState,
                offset: layer.offset
            )
        }
    }

    /// Restore from a snapshot
    public func restore(from snapshot: [LayerState]) {
        lock.lock()
        defer { lock.unlock() }

        precondition(snapshot.count == layers.count, "Snapshot layer count mismatch")
        layers = snapshot
    }

    /// Current sequence offset
    public var offset: Int {
        layers.first?.offset ?? 0
    }

    /// Estimated memory usage in bytes
    public var memoryBytes: Int {
        guard !layers.isEmpty else { return 0 }

        let ssmBytes = batchSize * config.innerSize * config.stateSize * 4  // float32
        let convBytes = batchSize * (config.convKernel - 1) * config.innerSize * 4
        let perLayer = ssmBytes + convBytes

        return perLayer * config.numHiddenLayers
    }

    /// Debug description of cache state
    public var debugDescription: String {
        """
        MambaCache:
          Layers: \(layers.count)
          Batch size: \(batchSize)
          Offset: \(offset)
          Memory: \(memoryBytes / 1024) KB
          Config:
            - hiddenSize: \(config.hiddenSize)
            - innerSize: \(config.innerSize)
            - stateSize: \(config.stateSize)
            - convKernel: \(config.convKernel)
        """
    }
}

// MARK: - Cache Statistics

extension MambaCache {

    public struct Statistics: Sendable {
        public let layerCount: Int
        public let batchSize: Int
        public let offset: Int
        public let ssmStateShape: [Int]
        public let convStateShape: [Int]
        public let totalMemoryBytes: Int

        public var totalMemoryMB: Float {
            Float(totalMemoryBytes) / (1024 * 1024)
        }
    }

    public func statistics() -> Statistics {
        Statistics(
            layerCount: layers.count,
            batchSize: batchSize,
            offset: offset,
            ssmStateShape: layers.first?.ssmState.shape ?? [],
            convStateShape: layers.first?.convState.shape ?? [],
            totalMemoryBytes: memoryBytes
        )
    }
}
