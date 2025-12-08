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
        subcommands: [Generate.self, ListModels.self, Benchmark.self],
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
        print("=== Transformer Drafters ===")
        for pair in DraftTargetPair.recommendedPairs {
            print("[\(pair.family)]")
            print("  Draft:  \(pair.draftModelId)")
            print("  Target: \(pair.targetModelId)")
            print()
        }

        print("=== Mamba Drafters ===")
        for model in MambaLoader.PretrainedModel.allCases {
            print("[\(model)]")
            print("  ID: \(model.modelId)")
            print("  Hidden: \(model.config.hiddenSize), Layers: \(model.config.numHiddenLayers)")
            print()
        }
    }
}

struct Benchmark: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "benchmark",
        abstract: "Benchmark Mamba vs Transformer drafters"
    )

    @Option(name: .long, help: "Target model ID")
    var targetModel: String = "mlx-community/Qwen2.5-7B-Instruct-4bit"

    @Option(name: .long, help: "Transformer draft model ID")
    var transformerDraft: String = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

    @Option(name: .long, help: "Mamba draft model ID (default: state-spaces/mamba-130m-hf)")
    var mambaDraft: String = "state-spaces/mamba-130m-hf"

    @Option(name: .shortAndLong, help: "Test prompt")
    var prompt: String = "Explain the theory of relativity in simple terms."

    @Option(name: .long, help: "Number of tokens to generate")
    var tokens: Int = 128

    @Option(name: .long, help: "Number of draft tokens per step")
    var draftTokens: Int = 5

    @Option(name: .long, help: "Number of benchmark runs")
    var runs: Int = 3

    mutating func run() async throws {
        print("=" * 60)
        print("Speculative Decoding Benchmark: Mamba vs Transformer")
        print("=" * 60)
        print()
        print("Configuration:")
        print("  Target model:      \(targetModel)")
        print("  Transformer draft: \(transformerDraft)")
        print("  Mamba draft:       \(mambaDraft)")
        print("  Tokens:            \(tokens)")
        print("  Draft tokens/step: \(draftTokens)")
        print("  Runs:              \(runs)")
        print()

        // Run transformer benchmark
        print("-" * 60)
        print("Benchmarking Transformer Drafter...")
        print("-" * 60)

        let transformerResults = try await benchmarkTransformer()

        print()
        print("-" * 60)
        print("Benchmarking Mamba Drafter...")
        print("-" * 60)

        let mambaResults = try await benchmarkMamba()

        // Print comparison summary
        print()
        print("=" * 60)
        print("Results Comparison")
        print("=" * 60)
        print()
        print("Transformer Drafter:")
        printResults(transformerResults)
        print()
        print("Mamba Drafter:")
        printResults(mambaResults)

        // Calculate speedup/slowdown
        if !transformerResults.isEmpty && !mambaResults.isEmpty {
            let transformerSpeed = transformerResults.map(\.tokensPerSecond).reduce(0, +) / Double(transformerResults.count)
            let mambaSpeed = mambaResults.map(\.tokensPerSecond).reduce(0, +) / Double(mambaResults.count)
            let speedRatio = mambaSpeed / transformerSpeed

            print()
            print("Mamba vs Transformer: \(String(format: "%.2fx", speedRatio)) speed ratio")
            if speedRatio > 1.0 {
                print("  Mamba is \(String(format: "%.1f%%", (speedRatio - 1) * 100)) faster")
            } else {
                print("  Transformer is \(String(format: "%.1f%%", (1/speedRatio - 1) * 100)) faster")
            }
        }
    }

    private func benchmarkMamba() async throws -> [BenchmarkResult] {
        print("Loading Mamba model: \(mambaDraft)")

        let pair = try await MambaDraftTargetPair.load(
            draftModelId: mambaDraft,
            targetModelId: targetModel
        )

        print("Model loaded successfully")

        let parameters = SpeculativeParameters(
            numDraftTokens: draftTokens,
            draftTemperature: 0,
            targetTemperature: 0,
            maxTokens: tokens
        )

        var results: [BenchmarkResult] = []

        for i in 1...runs {
            print("Run \(i)/\(runs)...")

            let generator = MambaSpeculativeGenerator(modelPair: pair, parameters: parameters)
            let result = try await generator.generate(prompt: prompt) { _ in .more }

            results.append(BenchmarkResult(
                tokensGenerated: result.tokens.count,
                totalTime: result.totalTime,
                tokensPerSecond: result.tokensPerSecond,
                acceptanceRate: result.acceptanceRate,
                avgTokensPerStep: result.statistics.avgTokensPerStep
            ))

            print("  Tokens: \(result.tokens.count), Speed: \(String(format: "%.1f", result.tokensPerSecond)) tok/s, Acceptance: \(String(format: "%.1f%%", result.acceptanceRate * 100))")
        }

        return results
    }

    private func benchmarkTransformer() async throws -> [BenchmarkResult] {
        print("Loading models...")
        let modelPair = try await DraftTargetPair.load(
            draftModelId: transformerDraft,
            targetModelId: targetModel
        )

        let parameters = SpeculativeParameters(
            numDraftTokens: draftTokens,
            draftTemperature: 0,
            targetTemperature: 0,
            maxTokens: tokens
        )

        var results: [BenchmarkResult] = []

        for i in 1...runs {
            print("Run \(i)/\(runs)...")
            let input = try await modelPair.prepare(prompt: prompt)

            var result: SpeculativeGenerateResult?
            let generator = SpeculativeGenerator(modelPair: modelPair, parameters: parameters)

            let genResult = try await generator.generate(input: input) { _ in .more }
            result = genResult

            if let r = result {
                results.append(BenchmarkResult(
                    tokensGenerated: r.tokens.count,
                    totalTime: r.totalTime,
                    tokensPerSecond: r.tokensPerSecond,
                    acceptanceRate: r.acceptanceRate,
                    avgTokensPerStep: r.statistics.avgTokensPerStep
                ))

                print("  Tokens: \(r.tokens.count), Speed: \(String(format: "%.1f", r.tokensPerSecond)) tok/s, Acceptance: \(String(format: "%.1f%%", r.acceptanceRate * 100))")
            }
        }

        return results
    }

    private func printResults(_ results: [BenchmarkResult]) {
        guard !results.isEmpty else {
            print("No results available.")
            return
        }

        let avgSpeed = results.map(\.tokensPerSecond).reduce(0, +) / Double(results.count)
        let avgAcceptance = results.map(\.acceptanceRate).reduce(0, +) / Float(results.count)
        let avgTokensPerStep = results.map(\.avgTokensPerStep).reduce(0, +) / Double(results.count)

        print("  Average Speed:           \(String(format: "%.1f", avgSpeed)) tok/s")
        print("  Average Acceptance Rate: \(String(format: "%.1f%%", avgAcceptance * 100))")
        print("  Average Tokens/Step:     \(String(format: "%.2f", avgTokensPerStep))")
    }
}

struct BenchmarkResult {
    let tokensGenerated: Int
    let totalTime: TimeInterval
    let tokensPerSecond: Double
    let acceptanceRate: Float
    let avgTokensPerStep: Double
}

extension String {
    static func * (str: String, count: Int) -> String {
        String(repeating: str, count: count)
    }
}
