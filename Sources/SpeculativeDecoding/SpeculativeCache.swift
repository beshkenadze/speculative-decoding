// Copyright 2024 MLX Speculative Decoding Contributors
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXLMCommon

/// Manages synchronized KV caches for draft and target models.
public final class SpeculativeCache: @unchecked Sendable {
    
    public private(set) var draftCache: [KVCache]
    public private(set) var targetCache: [KVCache]
    private let lock = NSLock()
    public private(set) var rollbackCount: Int = 0
    
    public var targetOffset: Int {
        lock.lock()
        defer { lock.unlock() }
        return targetCache.first?.offset ?? 0
    }
    
    public var draftOffset: Int {
        lock.lock()
        defer { lock.unlock() }
        return draftCache.first?.offset ?? 0
    }
    
    public init(draftCache: [KVCache], targetCache: [KVCache]) {
        self.draftCache = draftCache
        self.targetCache = targetCache
    }
    
    public static func create(
        draftModel: any LanguageModel,
        targetModel: any LanguageModel,
        parameters: GenerateParameters? = nil
    ) -> SpeculativeCache {
        SpeculativeCache(
            draftCache: draftModel.newCache(parameters: parameters),
            targetCache: targetModel.newCache(parameters: parameters)
        )
    }
    
    @discardableResult
    public func rollbackDraft(toOffset offset: Int) -> Int {
        lock.lock()
        defer { lock.unlock() }
        
        var totalTrimmed = 0
        for cache in draftCache {
            let currentOffset = cache.offset
            if currentOffset > offset {
                let trimAmount = currentOffset - offset
                let trimmed = cache.trim(trimAmount)
                totalTrimmed = max(totalTrimmed, trimmed)
            }
        }
        if totalTrimmed > 0 { rollbackCount += 1 }
        return totalTrimmed
    }
    
    @discardableResult
    public func synchronizeDraftToTarget() -> Int {
        rollbackDraft(toOffset: targetOffset)
    }
    
    public func synchronize() {
        lock.lock()
        defer { lock.unlock() }
        
        var arrays: [MLXArray] = []
        for cache in draftCache { arrays.append(contentsOf: cache.innerState()) }
        for cache in targetCache { arrays.append(contentsOf: cache.innerState()) }
        if !arrays.isEmpty { eval(arrays) }
    }
    
    public var statistics: Statistics {
        lock.lock()
        defer { lock.unlock() }
        return Statistics(
            rollbackCount: rollbackCount,
            finalDraftOffset: draftCache.first?.offset ?? 0,
            finalTargetOffset: targetCache.first?.offset ?? 0
        )
    }
    
    public struct Statistics: Sendable {
        public let rollbackCount: Int
        public let finalDraftOffset: Int
        public let finalTargetOffset: Int
        public var estimatedMemoryBytes: Int {
            finalTargetOffset * 4 * 64 * 32
        }
    }
}
