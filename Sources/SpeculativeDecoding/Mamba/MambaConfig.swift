// Copyright © 2024 speculative-decoding contributors
// SPDX-License-Identifier: MIT

import Foundation

/// Configuration for Mamba models loaded from HuggingFace.
///
/// Supports both `state-spaces/mamba-*` and `state-spaces/mamba-*-hf` model formats.
public struct MambaConfig: Codable, Sendable {

    // MARK: - Core Architecture

    /// Model dimension (d_model). Default: 768 for mamba-130m
    public var hiddenSize: Int

    /// Number of Mamba layers. Default: 24 for mamba-130m
    public var numHiddenLayers: Int

    /// Vocabulary size. Default: 50280
    public var vocabSize: Int

    // MARK: - SSM Parameters

    /// SSM state expansion factor (N in paper). Default: 16
    public var stateSize: Int

    /// Convolution kernel width. Default: 4
    public var convKernel: Int

    /// Block expansion factor. Default: 2
    public var expand: Int

    // MARK: - Time Step Parameters

    /// Rank of dt projection. Default: "auto" -> hiddenSize / 16
    public var timeStepRank: Int

    /// Minimum time step for initialization
    public var timeStepMin: Float

    /// Maximum time step for initialization
    public var timeStepMax: Float

    /// Floor for time step clamping
    public var timeStepFloor: Float

    /// Scale for time step
    public var timeStepScale: Float

    // MARK: - Normalization & Precision

    /// Use RMS normalization (vs LayerNorm)
    public var rmsNorm: Bool

    /// Keep residuals in FP32 for stability
    public var residualInFp32: Bool

    /// Layer norm epsilon
    public var layerNormEpsilon: Float

    // MARK: - Projection Options

    /// Use bias in linear projections
    public var useBias: Bool

    /// Use bias in convolution
    public var useConvBias: Bool

    // MARK: - Computed Properties

    /// Inner dimension after expansion: hiddenSize * expand
    public var innerSize: Int {
        hiddenSize * expand
    }

    // MARK: - Coding Keys (HuggingFace format)

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case vocabSize = "vocab_size"
        case stateSize = "state_size"
        case convKernel = "conv_kernel"
        case expand
        case timeStepRank = "time_step_rank"
        case timeStepMin = "time_step_min"
        case timeStepMax = "time_step_max"
        case timeStepFloor = "time_step_floor"
        case timeStepScale = "time_step_scale"
        case rmsNorm = "rms_norm"
        case residualInFp32 = "residual_in_fp32"
        case layerNormEpsilon = "layer_norm_epsilon"
        case useBias = "use_bias"
        case useConvBias = "use_conv_bias"

        // Alternative keys from non-HF format
        case dModel = "d_model"
        case nLayer = "n_layer"
        case dState = "d_state"
        case dConv = "d_conv"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        // Handle both HF and original format
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize)
            ?? container.decodeIfPresent(Int.self, forKey: .dModel)
            ?? 768

        self.numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers)
            ?? container.decodeIfPresent(Int.self, forKey: .nLayer)
            ?? 24

        self.vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 50280

        self.stateSize = try container.decodeIfPresent(Int.self, forKey: .stateSize)
            ?? container.decodeIfPresent(Int.self, forKey: .dState)
            ?? 16

        self.convKernel = try container.decodeIfPresent(Int.self, forKey: .convKernel)
            ?? container.decodeIfPresent(Int.self, forKey: .dConv)
            ?? 4

        self.expand = try container.decodeIfPresent(Int.self, forKey: .expand) ?? 2

        // Time step rank: "auto" means hiddenSize / 16
        if let rank = try? container.decode(Int.self, forKey: .timeStepRank) {
            self.timeStepRank = rank
        } else {
            self.timeStepRank = self.hiddenSize / 16
        }

        self.timeStepMin = try container.decodeIfPresent(Float.self, forKey: .timeStepMin) ?? 0.001
        self.timeStepMax = try container.decodeIfPresent(Float.self, forKey: .timeStepMax) ?? 0.1
        self.timeStepFloor = try container.decodeIfPresent(Float.self, forKey: .timeStepFloor) ?? 0.0001
        self.timeStepScale = try container.decodeIfPresent(Float.self, forKey: .timeStepScale) ?? 1.0

        self.rmsNorm = try container.decodeIfPresent(Bool.self, forKey: .rmsNorm) ?? true
        self.residualInFp32 = try container.decodeIfPresent(Bool.self, forKey: .residualInFp32) ?? true
        self.layerNormEpsilon = try container.decodeIfPresent(Float.self, forKey: .layerNormEpsilon) ?? 1e-5

        self.useBias = try container.decodeIfPresent(Bool.self, forKey: .useBias) ?? false
        self.useConvBias = try container.decodeIfPresent(Bool.self, forKey: .useConvBias) ?? true
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(numHiddenLayers, forKey: .numHiddenLayers)
        try container.encode(vocabSize, forKey: .vocabSize)
        try container.encode(stateSize, forKey: .stateSize)
        try container.encode(convKernel, forKey: .convKernel)
        try container.encode(expand, forKey: .expand)
        try container.encode(timeStepRank, forKey: .timeStepRank)
        try container.encode(timeStepMin, forKey: .timeStepMin)
        try container.encode(timeStepMax, forKey: .timeStepMax)
        try container.encode(timeStepFloor, forKey: .timeStepFloor)
        try container.encode(timeStepScale, forKey: .timeStepScale)
        try container.encode(rmsNorm, forKey: .rmsNorm)
        try container.encode(residualInFp32, forKey: .residualInFp32)
        try container.encode(layerNormEpsilon, forKey: .layerNormEpsilon)
        try container.encode(useBias, forKey: .useBias)
        try container.encode(useConvBias, forKey: .useConvBias)
    }

    /// Create config for mamba-130m
    public static var mamba130m: MambaConfig {
        MambaConfig(
            hiddenSize: 768,
            numHiddenLayers: 24,
            vocabSize: 50280,
            stateSize: 16,
            convKernel: 4,
            expand: 2
        )
    }

    /// Create config for mamba-370m
    public static var mamba370m: MambaConfig {
        MambaConfig(
            hiddenSize: 1024,
            numHiddenLayers: 48,
            vocabSize: 50280,
            stateSize: 16,
            convKernel: 4,
            expand: 2
        )
    }

    /// Create config for mamba-790m
    public static var mamba790m: MambaConfig {
        MambaConfig(
            hiddenSize: 1536,
            numHiddenLayers: 48,
            vocabSize: 50280,
            stateSize: 16,
            convKernel: 4,
            expand: 2
        )
    }

    /// Create config for mamba-1.4b
    public static var mamba1_4b: MambaConfig {
        MambaConfig(
            hiddenSize: 2048,
            numHiddenLayers: 48,
            vocabSize: 50280,
            stateSize: 16,
            convKernel: 4,
            expand: 2
        )
    }

    /// Create config for mamba-2.8b
    public static var mamba2_8b: MambaConfig {
        MambaConfig(
            hiddenSize: 2560,
            numHiddenLayers: 64,
            vocabSize: 50280,
            stateSize: 16,
            convKernel: 4,
            expand: 2
        )
    }

    /// Initialize with explicit parameters
    public init(
        hiddenSize: Int = 768,
        numHiddenLayers: Int = 24,
        vocabSize: Int = 50280,
        stateSize: Int = 16,
        convKernel: Int = 4,
        expand: Int = 2,
        timeStepRank: Int? = nil,
        timeStepMin: Float = 0.001,
        timeStepMax: Float = 0.1,
        timeStepFloor: Float = 0.0001,
        timeStepScale: Float = 1.0,
        rmsNorm: Bool = true,
        residualInFp32: Bool = true,
        layerNormEpsilon: Float = 1e-5,
        useBias: Bool = false,
        useConvBias: Bool = true
    ) {
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.vocabSize = vocabSize
        self.stateSize = stateSize
        self.convKernel = convKernel
        self.expand = expand
        self.timeStepRank = timeStepRank ?? (hiddenSize / 16)
        self.timeStepMin = timeStepMin
        self.timeStepMax = timeStepMax
        self.timeStepFloor = timeStepFloor
        self.timeStepScale = timeStepScale
        self.rmsNorm = rmsNorm
        self.residualInFp32 = residualInFp32
        self.layerNormEpsilon = layerNormEpsilon
        self.useBias = useBias
        self.useConvBias = useConvBias
    }
}
