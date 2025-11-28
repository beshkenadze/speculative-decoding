// Copyright 2024 MLX Speculative Decoding Contributors
// SPDX-License-Identifier: MIT

import ArgumentParser
import Foundation
import MLX
import MLXLMCommon
import MLXLLM
import SpeculativeDecoding

@main
struct SpeculativeCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "speculative-cli",
        abstract: "Generate text using speculative decoding with MLX",
        version: "1.0.0",
        subcommands: [Generate.self, ListModels.self],
        defaultSubcommand: Generate.self
    )
}

struct Generate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "generate",
        abstract: "Generate text using speculative decoding"
    )
    
    @Option(name: .long, help: "Draft model ID")
    var draftModel: String = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    
    @Option(name: .long, help: "Target model ID")
    var targetModel: String = "mlx-community/Qwen2.5-7B-Instruct-4bit"
    
    @Option(name: .shortAndLong, help: "Input prompt")
    var prompt: String
    
    @Option(name: .long, help: "Number of draft tokens (default: 5)")
    var draftTokens: Int = 5
    
    @Option(name: .long, help: "Maximum tokens (default: 256)")
    var maxTokens: Int = 256
    
    @Option(name: .long, help: "Temperature (default: 0.7)")
    var temperature: Float = 0.7
    
    @Flag(name: .long, help: "Use greedy decoding")
    var greedy: Bool = false
    
    @Flag(name: .long, help: "Show statistics")
    var stats: Bool = false
    
    @Flag(name: .long, help: "Quiet mode")
    var quiet: Bool = false
    
    mutating func run() async throws {
        if !quiet {
            print("Loading models...")
            print("  Draft:  \(draftModel)")
            print("  Target: \(targetModel)")
            print()
        }
        
        let modelPair = try await DraftTargetPair.load(
            draftModelId: draftModel,
            targetModelId: targetModel
        )
        
        let parameters = SpeculativeParameters(
            numDraftTokens: draftTokens,
            draftTemperature: greedy ? 0 : temperature,
            targetTemperature: greedy ? 0 : temperature,
            maxTokens: maxTokens
        )
        
        if !quiet {
            print("Generating...")
            print("---")
        }
        
        let stream = try await SpeculativeDecoding.generateStream(
            prompt: prompt,
            modelPair: modelPair,
            parameters: parameters
        )
        
        var finalResult: SpeculativeGenerateResult?
        
        for await event in stream {
            switch event {
            case .text(let chunk):
                print(chunk, terminator: "")
                fflush(stdout)
            case .result(let result):
                finalResult = result
            case .error(let error):
                print("\nError: \(error.localizedDescription)")
            }
        }
        
        print()
        if !quiet { print("---") }
        
        if stats, let result = finalResult {
            print()
            print(result.summary())
        } else if !quiet, let result = finalResult {
            print()
            print("Generated \(result.tokens.count) tokens @ \(String(format: "%.1f", result.tokensPerSecond)) tok/s")
            print("Acceptance: \(String(format: "%.1f%%", result.acceptanceRate * 100))")
        }
    }
}

struct ListModels: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "list-models",
        abstract: "List recommended model pairs"
    )
    
    mutating func run() async throws {
        print("Recommended Model Pairs")
        print()
        for pair in DraftTargetPair.recommendedPairs {
            print("[\(pair.family)]")
            print("  Draft:  \(pair.draftModelId)")
            print("  Target: \(pair.targetModelId)")
            print()
        }
    }
}
