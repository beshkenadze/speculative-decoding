// Copyright © 2024 speculative-decoding contributors
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import MLXNN
import MLXFast

/// Mamba selective state space block.
///
/// Implements the selective SSM from "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
/// (Gu & Dao, 2023). This is the core building block that replaces attention in transformers.
///
/// The block consists of:
/// 1. Input projection (expand D -> 2*ED for x and z branches)
/// 2. Depthwise 1D convolution on x branch
/// 3. Selective SSM with input-dependent Δ, B, C
/// 4. Gating with z branch via SiLU
/// 5. Output projection (contract ED -> D)
public class MambaBlock: Module {

    // MARK: - Configuration

    let config: MambaConfig

    // MARK: - Projections

    /// Input projection: D -> 2*ED (x and z branches)
    @ModuleInfo(key: "in_proj") var inProj: Linear

    /// Output projection: ED -> D
    @ModuleInfo(key: "out_proj") var outProj: Linear

    // MARK: - Convolution

    /// Depthwise 1D convolution weight
    /// HuggingFace stores as [ED, 1, k], we transpose during loading to [ED, k, 1]
    @ParameterInfo(key: "conv1d.weight") var convWeight: MLXArray

    /// Convolution bias: [ED]
    @ParameterInfo(key: "conv1d.bias") var convBias: MLXArray

    // MARK: - SSM Parameters

    /// Combined x_proj: ED -> dt_rank + 2*stateSize (for dt, B, C)
    /// HuggingFace format stores these combined
    @ModuleInfo(key: "x_proj") var xProj: Linear

    /// Time step projection: dt_rank -> ED
    @ModuleInfo(key: "dt_proj") var dtProj: Linear

    /// A parameter: [ED, stateSize] - diagonal state matrix (stored as log for stability)
    @ParameterInfo(key: "A_log") var aLog: MLXArray

    /// D parameter: [ED] - skip connection
    @ParameterInfo(key: "D") var d: MLXArray

    // MARK: - Initialization

    public init(_ config: MambaConfig) {
        self.config = config

        let innerSize = config.innerSize  // ED = D * expand

        // Input projection: D -> 2*ED (x and z)
        self._inProj.wrappedValue = Linear(
            config.hiddenSize,
            innerSize * 2,
            bias: config.useBias
        )

        // Output projection: ED -> D
        self._outProj.wrappedValue = Linear(
            innerSize,
            config.hiddenSize,
            bias: config.useBias
        )

        // Depthwise convolution: groups=ED for depthwise
        // Weight shape for MLX conv1d: [C_out, k, C_in/groups] = [ED, k, 1]
        let convScale = sqrt(1.0 / Float(config.convKernel))
        self._convWeight.wrappedValue = MLXRandom.uniform(
            low: -convScale,
            high: convScale,
            [innerSize, config.convKernel, 1]
        )
        self._convBias.wrappedValue = MLXArray.zeros([innerSize])

        // Combined x_proj: ED -> dt_rank + 2*N (for dt, B, C)
        // This matches HuggingFace format
        let xProjSize = config.timeStepRank + 2 * config.stateSize
        self._xProj.wrappedValue = Linear(innerSize, xProjSize, bias: false)

        // Time step projection: dt_rank -> ED
        self._dtProj.wrappedValue = Linear(
            config.timeStepRank,
            innerSize,
            bias: true  // dt_proj always has bias
        )

        // A matrix: initialized as -exp(linspace) for stability
        // Shape: [ED, N] where N is stateSize
        let aRange = MLXArray((1...config.stateSize).map { Float($0) })
        let aInit = MLX.broadcast(
            MLX.log(aRange),
            to: [innerSize, config.stateSize]
        )
        self._aLog.wrappedValue = aInit

        // D: skip connection, initialized to ones
        self._d.wrappedValue = MLXArray.ones([innerSize])

        super.init()
    }

    // MARK: - Forward Pass

    /// Forward pass for sequence processing (training/prefill).
    ///
    /// - Parameters:
    ///   - x: Input tensor of shape [B, L, D]
    /// - Returns: Output tensor of shape [B, L, D]
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, seqLen, _) = (x.dim(0), x.dim(1), x.dim(2))

        // 1. Input projection: [B, L, D] -> [B, L, 2*ED]
        let projected = inProj(x)

        // Split into x and z branches
        let xz = projected.split(parts: 2, axis: -1)
        var xBranch = xz[0]  // [B, L, ED]
        let z = xz[1]        // [B, L, ED]

        // 2. Causal convolution on x branch
        // MLX conv1d expects input [B, L, C] and weight [C_out, k, C_in/groups]
        // Causal padding: pad left with (kernelSize - 1) zeros on sequence dimension
        let padSize = config.convKernel - 1
        xBranch = MLX.padded(xBranch, widths: [[0, 0], [padSize, 0], [0, 0]])

        // Apply depthwise conv1d (groups = ED for depthwise)
        xBranch = MLX.conv1d(xBranch, convWeight, stride: 1, padding: 0, dilation: 1, groups: config.innerSize)
        xBranch = xBranch + convBias

        // 3. SiLU activation on x
        xBranch = silu(xBranch)

        // 4. Selective SSM
        let y = selectiveSSM(xBranch, seqLen: seqLen, batch: batch)

        // 5. Gating: y * silu(z)
        let gated = y * silu(z)

        // 6. Output projection
        return outProj(gated)
    }

    /// Selective SSM computation using parallel scan.
    private func selectiveSSM(_ x: MLXArray, seqLen: Int, batch: Int) -> MLXArray {
        let innerSize = config.innerSize
        let stateSize = config.stateSize
        let dtRank = config.timeStepRank

        // Compute input-dependent parameters via combined x_proj
        // x_proj output: [B, L, dt_rank + 2*N]
        let xProjOut = xProj(x)  // [B, L, dt_rank + 2*N]

        // Split into dt, B, C
        let dtInput = xProjOut[0..., 0..., 0..<dtRank]  // [B, L, dt_rank]
        let B = xProjOut[0..., 0..., dtRank..<(dtRank + stateSize)]  // [B, L, N]
        let C = xProjOut[0..., 0..., (dtRank + stateSize)...]  // [B, L, N]

        // dt projection: dt_rank -> ED, with softplus
        var delta = dtProj(softplus(dtInput))  // [B, L, ED]

        // Clamp delta to prevent numerical issues
        delta = MLX.clip(delta, min: config.timeStepFloor)

        // A: diagonal state matrix (negative for stability)
        let A = -MLX.exp(aLog)  // [ED, N]

        // Discretize: A_bar = exp(delta * A), B_bar = delta * B
        // delta: [B, L, ED], A: [ED, N]
        // deltaA: [B, L, ED, N]
        let deltaExpanded = delta.expandedDimensions(axis: -1)  // [B, L, ED, 1]
        let aExpanded = A.reshaped(1, 1, innerSize, stateSize)  // [1, 1, ED, N]
        let deltaA = MLX.exp(deltaExpanded * aExpanded)  // [B, L, ED, N]

        // deltaB: [B, L, ED, N]
        let bExpanded = B.expandedDimensions(axis: 2)  // [B, L, 1, N]
        let deltaB = deltaExpanded * bExpanded  // [B, L, ED, N]

        // x contribution: deltaB * x
        let xExpanded = x.expandedDimensions(axis: -1)  // [B, L, ED, 1]
        let BX = deltaB * xExpanded  // [B, L, ED, N]

        // Parallel scan: h[t] = deltaA[t] * h[t-1] + BX[t]
        let hs = parallelScan(deltaA, BX)  // [B, L, ED, N]

        // Output: y = (h @ C) + D * x
        // hs: [B, L, ED, N], C: [B, L, N]
        let cExpanded = C.expandedDimensions(axis: 2)  // [B, L, 1, N]
        let y = (hs * cExpanded).sum(axis: -1)  // [B, L, ED]

        // Skip connection
        return y + d * x
    }

    /// Parallel scan for SSM recurrence: h[t] = A[t] * h[t-1] + X[t]
    ///
    /// Uses associative scan for O(L) parallel computation instead of O(L) sequential.
    private func parallelScan(_ A: MLXArray, _ X: MLXArray) -> MLXArray {
        // For now, use sequential scan for correctness
        // TODO: Implement parallel associative scan for performance
        return sequentialScan(A, X)
    }

    /// Sequential scan (fallback for correctness)
    private func sequentialScan(_ A: MLXArray, _ X: MLXArray) -> MLXArray {
        let seqLen = A.dim(1)
        var h = MLXArray.zeros([A.dim(0), A.dim(2), A.dim(3)])  // [B, ED, N]
        var outputs = [MLXArray]()

        for t in 0..<seqLen {
            let at = A[0..., t, 0..., 0...]  // [B, ED, N]
            let xt = X[0..., t, 0..., 0...]  // [B, ED, N]
            h = at * h + xt
            outputs.append(h.expandedDimensions(axis: 1))
        }

        return MLX.concatenated(outputs, axis: 1)  // [B, L, ED, N]
    }

    // MARK: - Step Function (for fast inference)

    /// Single-step forward pass for autoregressive generation.
    ///
    /// Uses cached convolution inputs and SSM state for O(1) inference per token.
    ///
    /// - Parameters:
    ///   - x: Input tensor of shape [B, D]
    ///   - cache: Mamba state cache containing conv inputs and SSM hidden state
    /// - Returns: Tuple of (output [B, D], updated cache)
    public func step(_ x: MLXArray, cache: inout MambaCache.LayerState) -> MLXArray {
        // 1. Input projection: [B, D] -> [B, 2*ED]
        let projected = inProj(x)

        let xz = projected.split(parts: 2, axis: -1)
        var xBranch = xz[0]  // [B, ED]
        let z = xz[1]        // [B, ED]

        // 2. Update conv cache and compute convolution
        // Cache stores last (kernelSize - 1) inputs for causal convolution
        // For each step, we need the last k inputs (including current)
        let k = config.convKernel

        // Get the full window of k inputs for convolution
        let convInputs: MLXArray
        if cache.convState.dim(1) >= k - 1 {
            // Normal case: have enough history
            // Take last k-1 from cache, append current input
            let history = cache.convState[0..., (cache.convState.dim(1) - (k - 1))..., 0...]  // [B, k-1, ED]
            convInputs = MLX.concatenated([history, xBranch.expandedDimensions(axis: 1)], axis: 1)  // [B, k, ED]
        } else {
            // First few steps: pad with zeros
            let historyLen = cache.convState.dim(1)
            let padLen = k - 1 - historyLen
            let padding = MLXArray.zeros([x.dim(0), padLen, config.innerSize])
            if historyLen > 0 {
                convInputs = MLX.concatenated([padding, cache.convState, xBranch.expandedDimensions(axis: 1)], axis: 1)
            } else {
                convInputs = MLX.concatenated([padding, xBranch.expandedDimensions(axis: 1)], axis: 1)
            }
        }

        // Update cache: keep last k-1 inputs (drop oldest, add current)
        if cache.convState.dim(1) >= k - 1 {
            let newCache = MLX.concatenated([
                cache.convState[0..., 1..., 0...],  // Drop oldest
                xBranch.expandedDimensions(axis: 1)  // Add current
            ], axis: 1)
            cache.convState = newCache
        } else {
            // Growing the cache
            cache.convState = MLX.concatenated([cache.convState, xBranch.expandedDimensions(axis: 1)], axis: 1)
        }

        // Compute convolution manually for single step
        // convInputs: [B, k, ED], convWeight: [ED, k, 1]
        // Need: [B, ED]

        // Reshape for depthwise convolution dot product
        // inputs: [B, k, ED] -> [B, ED, k]
        // weights: [ED, k, 1] -> [ED, k]
        let inputsForConv = convInputs.transposed(0, 2, 1)  // [B, ED, k]
        let weightsForConv = convWeight.squeezed(axis: -1)  // [ED, k]

        // Element-wise multiply and sum over k dimension
        xBranch = (inputsForConv * weightsForConv).sum(axis: -1)  // [B, ED]
        xBranch = xBranch + convBias

        // 3. SiLU activation
        xBranch = silu(xBranch)

        // 4. Selective SSM step
        let y = ssmStep(xBranch, cache: &cache)

        // 5. Gating
        let gated = y * silu(z)

        // 6. Output projection
        return outProj(gated)
    }

    /// Single SSM step with state update.
    private func ssmStep(_ x: MLXArray, cache: inout MambaCache.LayerState) -> MLXArray {
        let stateSize = config.stateSize
        let dtRankSize = config.timeStepRank

        // Compute input-dependent parameters via combined x_proj
        let xProjOut = xProj(x)  // [B, dt_rank + 2*N]

        // Split into dt, B, C
        let dtInput = xProjOut[0..., 0..<dtRankSize]  // [B, dt_rank]
        let B = xProjOut[0..., dtRankSize..<(dtRankSize + stateSize)]  // [B, N]
        let C = xProjOut[0..., (dtRankSize + stateSize)...]  // [B, N]

        // dt projection with softplus
        var delta = dtProj(softplus(dtInput))  // [B, ED]
        delta = MLX.clip(delta, min: config.timeStepFloor)

        let A = -MLX.exp(aLog)  // [ED, N]

        // Discretize
        let deltaA = MLX.exp(delta.expandedDimensions(axis: -1) * A)  // [B, ED, N]
        let deltaB = delta.expandedDimensions(axis: -1) * B.expandedDimensions(axis: 1)  // [B, ED, N]
        let BX = deltaB * x.expandedDimensions(axis: -1)  // [B, ED, N]

        // Update hidden state: h = deltaA * h + BX
        cache.ssmState = deltaA * cache.ssmState + BX

        // Output: y = (h @ C) + D * x
        let y = (cache.ssmState * C.expandedDimensions(axis: 1)).sum(axis: -1) + d * x

        return y
    }
}
