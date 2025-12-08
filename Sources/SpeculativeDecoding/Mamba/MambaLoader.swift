// Copyright © 2024 speculative-decoding contributors
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN
@preconcurrency import Hub

/// Loader for Mamba models from HuggingFace.
///
/// Supports loading from:
/// - `state-spaces/mamba-130m-hf`
/// - `state-spaces/mamba-370m-hf`
/// - Other HuggingFace Mamba models
public enum MambaLoader {

    /// Errors that can occur during loading
    public enum LoadError: Error {
        case configNotFound(String)
        case weightsNotFound(String)
        case invalidFormat(String)
        case downloadFailed(String)
    }

    /// Load a Mamba model from HuggingFace.
    ///
    /// - Parameters:
    ///   - modelId: HuggingFace model ID (e.g., "state-spaces/mamba-130m-hf")
    ///   - progressHandler: Optional callback for download progress
    /// - Returns: Tuple of (model, config, tokenizer directory)
    public static func load(
        modelId: String,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> (MambaLM, MambaConfig, URL) {
        // Download model files
        let hub = HubApi()
        let repo = Hub.Repo(id: modelId)

        // Download all needed files
        let modelDirectory = try await hub.snapshot(
            from: repo,
            matching: ["*.safetensors", "*.json"],
            progressHandler: progressHandler
        )

        // Load config
        let configURL = modelDirectory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw LoadError.configNotFound(configURL.path)
        }

        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(MambaConfig.self, from: configData)

        // Find weights file
        let weightsPath = modelDirectory.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: weightsPath.path) else {
            throw LoadError.weightsNotFound(weightsPath.path)
        }

        // Create model
        let model = MambaLM(config)

        // Load weights
        try loadWeights(model: model, from: weightsPath, config: config)

        return (model, config, modelDirectory)
    }

    /// Load weights from safetensors file into model.
    public static func loadWeights(
        model: MambaLM,
        from path: URL,
        config: MambaConfig
    ) throws {
        // Load safetensors
        let weights = try MLX.loadArrays(url: path)

        // Map HuggingFace keys to our model structure and process weights
        var mappedWeights = [String: MLXArray]()
        for (key, value) in weights {
            let mappedKey = mapKey(key)
            let processedValue = processWeight(mappedKey, value)
            mappedWeights[mappedKey] = processedValue
        }

        // Build the nested parameter dictionary that MLX-Swift expects
        let parameters = buildNestedParameters(from: mappedWeights)

        // Update model parameters
        try model.update(parameters: parameters, verify: [])
    }

    /// Map HuggingFace weight keys to our model structure.
    private static func mapKey(_ hfKey: String) -> String {
        var key = hfKey

        // HuggingFace format: backbone.layers.0.mixer.in_proj.weight
        // Our format: layers.0.mixer.inProj.weight

        // Remove "backbone." prefix if present
        if key.hasPrefix("backbone.") {
            key = String(key.dropFirst("backbone.".count))
        }

        // Handle embedding
        key = key.replacingOccurrences(of: "embeddings.weight", with: "embedding.weight")

        // Handle final norm
        key = key.replacingOccurrences(of: "norm_f.weight", with: "normF.weight")

        // Handle dt_proj
        key = key.replacingOccurrences(of: "dt_proj", with: "dtProj")

        // Handle projections with underscores
        key = key.replacingOccurrences(of: "in_proj", with: "inProj")
        key = key.replacingOccurrences(of: "out_proj", with: "outProj")
        key = key.replacingOccurrences(of: "x_proj", with: "xProj")

        // Handle conv1d - map to flat path
        key = key.replacingOccurrences(of: "conv1d.weight", with: "conv1d\\.weight")
        key = key.replacingOccurrences(of: "conv1d.bias", with: "conv1d\\.bias")

        return key
    }

    /// Process weights after mapping - handle any transpositions needed
    private static func processWeight(_ key: String, _ value: MLXArray) -> MLXArray {
        // Conv1d weights: HF stores as [C_out, 1, k], MLX expects [C_out, k, 1]
        if key.contains("conv1d") && key.hasSuffix("weight") {
            // Transpose from [ED, 1, k] to [ED, k, 1]
            return value.transposed(0, 2, 1)
        }
        return value
    }

    /// Build nested parameter structure from flat weight dictionary
    private static func buildNestedParameters(from weights: [String: MLXArray]) -> ModuleParameters {
        var result = ModuleParameters()

        for (key, value) in weights {
            insertNested(into: &result, path: key.split(separator: ".").map(String.init), value: value)
        }

        // Convert "layers" from dictionary with numeric keys to array
        if case .dictionary(let layersDict) = result["layers"] {
            let sortedKeys = layersDict.keys.compactMap { Int($0) }.sorted()
            var layersArray = [NestedItem<String, MLXArray>]()

            for key in sortedKeys {
                if let item = layersDict[String(key)] {
                    layersArray.append(item)
                }
            }

            result["layers"] = .array(layersArray)
        }

        return result
    }

    private static func insertNested(into dict: inout ModuleParameters, path: [String], value: MLXArray) {
        guard !path.isEmpty else { return }

        if path.count == 1 {
            dict[path[0]] = .value(value)
        } else {
            let key = path[0]
            let rest = Array(path.dropFirst())

            // Get existing nested dictionary or create new one
            var nested: ModuleParameters
            if case .dictionary(let existing) = dict[key] {
                nested = ModuleParameters(values: existing)
            } else {
                nested = ModuleParameters()
            }

            insertNested(into: &nested, path: rest, value: value)
            dict[key] = nested.asItem()
        }
    }

    /// Available pretrained Mamba models.
    public enum PretrainedModel: String, CaseIterable {
        case mamba130m = "state-spaces/mamba-130m-hf"
        case mamba370m = "state-spaces/mamba-370m-hf"
        case mamba790m = "state-spaces/mamba-790m-hf"
        case mamba1_4b = "state-spaces/mamba-1.4b-hf"
        case mamba2_8b = "state-spaces/mamba-2.8b-hf"

        public var modelId: String { rawValue }

        public var config: MambaConfig {
            switch self {
            case .mamba130m: return .mamba130m
            case .mamba370m: return .mamba370m
            case .mamba790m: return .mamba790m
            case .mamba1_4b: return .mamba1_4b
            case .mamba2_8b: return .mamba2_8b
            }
        }
    }

    /// Load a pretrained Mamba model.
    public static func loadPretrained(
        _ model: PretrainedModel,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> (MambaLM, MambaConfig, URL) {
        try await load(modelId: model.modelId, progressHandler: progressHandler)
    }
}
