import Foundation
@preconcurrency import Metal
import simd

// MARK: - Metal Force Compute

/// Manages Metal compute pipeline for GPU-accelerated charge force computation.
/// Falls back to CPU when Metal is unavailable or node count is too small.
///
/// Thread safety: not actor-isolated. Safe because `ForceSimulation3D.tickInFlight`
/// guarantees at most one concurrent call to `dispatchChargeForces`.
/// Metal device/queue/pipeline are inherently thread-safe.
final class MetalForceCompute: @unchecked Sendable {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipelineState: MTLComputePipelineState

    // Persistent buffers — resized only when capacity exceeded (2x over-allocation)
    private var nodeBuffer: MTLBuffer?
    private var forceBuffer: MTLBuffer?
    private var paramBuffer: MTLBuffer?
    private var currentNodeCount: Int = 0
    private var bufferCapacity: Int = 0

    /// Matches the `ForceNode` struct in Shaders.metal.
    struct GPUForceNode {
        var px: Float, py: Float, pz: Float
        var projectGroup: Int32
        var topicGroup: Int32
        var _pad: Int32
    }

    /// Matches the `ForceParams` struct in Shaders.metal.
    struct GPUForceParams {
        var chargeStrength: Float
        var crossChargeMultiplier: Float
        var sameTopicChargeScale: Float
        var sameProjectChargeScale: Float
        var cutoffSq: Float
        var nodeCount: UInt32
    }

    init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let library = device.makeDefaultLibrary(),
              let function = library.makeFunction(name: "compute_charge_forces"),
              let pipeline = try? device.makeComputePipelineState(function: function)
        else { return nil }

        self.device = device
        self.commandQueue = queue
        self.pipelineState = pipeline
    }

    /// Dispatch charge force computation on GPU. Non-blocking — calls completion on main thread.
    /// Result from GPU charge force computation.
    struct ChargeResult: Sendable {
        let fx: [Float], fy: [Float], fz: [Float]
    }

    func dispatchChargeForces(
        positions: (x: [Float], y: [Float], z: [Float]),
        projectGroups: [Int],
        topicGroups: [Int],
        chargeStrength: Float,
        crossChargeMultiplier: Float,
        sameTopicChargeScale: Float,
        sameProjectChargeScale: Float
    ) async -> ChargeResult {
        let n = positions.x.count
        guard n > 0 else { return ChargeResult(fx: [], fy: [], fz: []) }

        // Resize buffers only when capacity exceeded (2x over-allocation avoids
        // reallocation on every single addNode during sync)
        currentNodeCount = n
        if n > bufferCapacity {
            let capacity = max(n * 2, 512)
            let nodeSize = MemoryLayout<GPUForceNode>.stride * capacity
            let forceSize = MemoryLayout<SIMD3<Float>>.stride * capacity
            nodeBuffer = device.makeBuffer(length: nodeSize, options: .storageModeShared)
            forceBuffer = device.makeBuffer(length: forceSize, options: .storageModeShared)
            paramBuffer = device.makeBuffer(length: MemoryLayout<GPUForceParams>.stride, options: .storageModeShared)
            bufferCapacity = capacity
        }

        let zeros = ChargeResult(
            fx: [Float](repeating: 0, count: n),
            fy: [Float](repeating: 0, count: n),
            fz: [Float](repeating: 0, count: n)
        )
        guard let nodeBuffer, let forceBuffer, let paramBuffer else { return zeros }

        // Pack node data
        let nodePtr = nodeBuffer.contents().bindMemory(to: GPUForceNode.self, capacity: n)
        for i in 0..<n {
            nodePtr[i] = GPUForceNode(
                px: positions.x[i], py: positions.y[i], pz: positions.z[i],
                projectGroup: Int32(projectGroups[i]),
                topicGroup: Int32(i < topicGroups.count ? topicGroups[i] : -1),
                _pad: 0
            )
        }

        // Set params
        let paramPtr = paramBuffer.contents().bindMemory(to: GPUForceParams.self, capacity: 1)
        paramPtr.pointee = GPUForceParams(
            chargeStrength: chargeStrength,
            crossChargeMultiplier: crossChargeMultiplier,
            sameTopicChargeScale: sameTopicChargeScale,
            sameProjectChargeScale: sameProjectChargeScale,
            cutoffSq: 500 * 500,
            nodeCount: UInt32(n)
        )

        // Zero force buffer
        memset(forceBuffer.contents(), 0, MemoryLayout<SIMD3<Float>>.stride * n)

        guard let cmdBuffer = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuffer.makeComputeCommandEncoder() else { return zeros }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(nodeBuffer, offset: 0, index: 0)
        encoder.setBuffer(forceBuffer, offset: 0, index: 1)
        encoder.setBuffer(paramBuffer, offset: 0, index: 2)

        let threadGroupSize = min(pipelineState.maxTotalThreadsPerThreadgroup, n)
        let gridSize = MTLSize(width: n, height: 1, depth: 1)
        let threadGroup = MTLSize(width: threadGroupSize, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadGroup)
        encoder.endEncoding()

        // Use continuation to bridge Metal's callback-based API to async/await
        let capturedForceBuffer = forceBuffer
        let capturedN = n
        return await withCheckedContinuation { continuation in
            cmdBuffer.addCompletedHandler { @Sendable _ in
                let forcePtr = capturedForceBuffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: capturedN)
                var fx = [Float](repeating: 0, count: capturedN)
                var fy = [Float](repeating: 0, count: capturedN)
                var fz = [Float](repeating: 0, count: capturedN)
                for i in 0..<capturedN {
                    fx[i] = forcePtr[i].x
                    fy[i] = forcePtr[i].y
                    fz[i] = forcePtr[i].z
                }
                continuation.resume(returning: ChargeResult(fx: fx, fy: fy, fz: fz))
            }
            cmdBuffer.commit()
        }
    }
}
