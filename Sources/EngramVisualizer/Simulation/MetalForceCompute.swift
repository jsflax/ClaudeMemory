import Foundation
import CEngramSceneTypes
@preconcurrency import Metal
import simd
import os

// MARK: - Metal Force Compute

/// Manages Metal compute pipelines for GPU-accelerated force simulation.
/// Falls back to CPU when Metal is unavailable or node count is too small.
///
/// Thread safety: `gpuInFlight` gate ensures at most one command buffer is
/// encoding/executing at a time. Concurrent callers get zero results immediately.
final class MetalForceCompute: @unchecked Sendable {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipelineState: MTLComputePipelineState

    /// Serial gate — prevents concurrent command buffer encoding on shared buffers.
    /// Set true before encoding, cleared in completion handler.
    private let gpuInFlight = OSAllocatedUnfairLock(initialState: false)

    // Persistent buffers for charge-only path — resized only when capacity exceeded (2x over-allocation)
    private var nodeBuffer: MTLBuffer?
    private var forceBuffer: MTLBuffer?
    private var paramBuffer: MTLBuffer?
    private var currentNodeCount: Int = 0
    private var bufferCapacity: Int = 0

    /// Matches the `ForceParams` struct in Shaders.metal.
    struct GPUForceParams {
        var chargeStrength: Float
        var crossChargeMultiplier: Float
        var sameTopicChargeScale: Float
        var sameProjectChargeScale: Float
        var cutoffSq: Float
        var nodeCount: UInt32
    }

    /// Initialize with an external device and library (shares GPU with renderer).
    init?(device: MTLDevice, library: MTLLibrary) {
        guard let queue = device.makeCommandQueue(),
              let function = library.makeFunction(name: "compute_charge_forces"),
              let pipeline = try? device.makeComputePipelineState(function: function)
        else { return nil }

        self.device = device
        self.commandQueue = queue
        self.pipelineState = pipeline

        // Build full simulation pipelines
        if let fn = library.makeFunction(name: "compute_spring_forces") {
            springPipeline = try? device.makeComputePipelineState(function: fn)
        }
        if let fn = library.makeFunction(name: "compute_group_centroids") {
            centroidPipeline = try? device.makeComputePipelineState(function: fn)
        }
        if let fn = library.makeFunction(name: "compute_centroid_repulsion") {
            centroidRepulsionPipeline = try? device.makeComputePipelineState(function: fn)
        }
        if let fn = library.makeFunction(name: "apply_cohesion_forces") {
            cohesionPipeline = try? device.makeComputePipelineState(function: fn)
        }
        fullSimReady = springPipeline != nil && centroidPipeline != nil
            && centroidRepulsionPipeline != nil && cohesionPipeline != nil

        simParamBuffer = device.makeBuffer(length: MemoryLayout<ForceSimParams>.stride, options: .storageModeShared)
        groupTypeBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
        nodeCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
        repulsionBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)
        groupCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
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

        // Build full simulation pipelines
        if let fn = library.makeFunction(name: "compute_spring_forces") {
            springPipeline = try? device.makeComputePipelineState(function: fn)
        }
        if let fn = library.makeFunction(name: "compute_group_centroids") {
            centroidPipeline = try? device.makeComputePipelineState(function: fn)
        }
        if let fn = library.makeFunction(name: "compute_centroid_repulsion") {
            centroidRepulsionPipeline = try? device.makeComputePipelineState(function: fn)
        }
        if let fn = library.makeFunction(name: "apply_cohesion_forces") {
            cohesionPipeline = try? device.makeComputePipelineState(function: fn)
        }
        fullSimReady = springPipeline != nil && centroidPipeline != nil
            && centroidRepulsionPipeline != nil && cohesionPipeline != nil

        // Constant-size buffers
        simParamBuffer = device.makeBuffer(length: MemoryLayout<ForceSimParams>.stride, options: .storageModeShared)
        groupTypeBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
        nodeCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
        repulsionBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)
        groupCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
    }

    /// Whether the full GPU simulation pipeline is available.
    var isFullSimAvailable: Bool { fullSimReady }

    /// Node count from last encodeForces call (for readback sizing).
    private var lastEncodedNodeCount: Int = 0

    /// Encode all force compute passes into an external compute encoder.
    /// Called from MetalGraphRenderer.drawFrame() BEFORE render passes.
    /// Forces accumulate in fullForceBuffer; read back via readBackForces() after GPU completes.
    func encodeForces(
        encoder: MTLComputeCommandEncoder,
        x: [Float], y: [Float], z: [Float],
        projectGroups: [Int], topicGroups: [Int],
        chargeStrength: Float, crossChargeMultiplier: Float,
        sameTopicChargeScale: Float, sameProjectChargeScale: Float,
        springLength: Float, crossProjectSpringLength: Float, springStrength: Float,
        cohesionStrength: Float, centroidRepulsion: Float,
        topicCohesionStrength: Float, topicCentroidRepulsion: Float,
        centerStrength: Float, center: SIMD3<Float>,
        alpha: Float, damping: Float, maxSpeed: Float
    ) {
        let n = x.count
        guard n > 1, fullSimReady else { return }

        let projectCount = UInt32(max((projectGroups.max() ?? 0) + 1, 1))
        let topicCount = UInt32(max((topicGroups.max() ?? 0) + 1, 1))
        let maxGroups = max(Int(projectCount), Int(topicCount))

        // Ensure buffer capacity
        if n > fullNodeCapacity {
            let cap = max(n * 2, 512)
            fullNodeBuffer = device.makeBuffer(length: cap * MemoryLayout<ForceNodeFull>.stride, options: .storageModeShared)
            fullForceBuffer = device.makeBuffer(length: cap * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)
            fullNodeCapacity = cap
        }
        if maxGroups > fullGroupCapacity {
            let cap = max(maxGroups * 2, 64)
            projCentroidBuffer = device.makeBuffer(length: cap * MemoryLayout<GroupCentroid>.stride, options: .storageModeShared)
            topicCentroidBuffer = device.makeBuffer(length: cap * MemoryLayout<GroupCentroid>.stride, options: .storageModeShared)
            projForceBuffer = device.makeBuffer(length: cap * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)
            topicForceBuffer = device.makeBuffer(length: cap * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)
            fullGroupCapacity = cap
        }

        guard let nodeBuf = fullNodeBuffer, let forceBuf = fullForceBuffer,
              let projCBuf = projCentroidBuffer, let topicCBuf = topicCentroidBuffer,
              let projFBuf = projForceBuffer, let topicFBuf = topicForceBuffer,
              let simPBuf = simParamBuffer, let gtBuf = groupTypeBuffer,
              let repBuf = repulsionBuffer, let gcBuf = groupCountBuffer
        else { return }

        lastEncodedNodeCount = n

        // Pack node data
        let nodePtr = nodeBuf.contents().bindMemory(to: ForceNodeFull.self, capacity: n)
        for i in 0..<n {
            nodePtr[i] = ForceNodeFull(
                px: x[i], py: y[i], pz: z[i], vx: 0, vy: 0, vz: 0,
                projectGroup: Int32(projectGroups[i]),
                topicGroup: Int32(i < topicGroups.count ? topicGroups[i] : -1))
        }

        // Zero buffers
        memset(forceBuf.contents(), 0, MemoryLayout<SIMD3<Float>>.stride * n)
        memset(projCBuf.contents(), 0, MemoryLayout<GroupCentroid>.stride * Int(projectCount))
        memset(topicCBuf.contents(), 0, MemoryLayout<GroupCentroid>.stride * Int(topicCount))
        memset(projFBuf.contents(), 0, MemoryLayout<SIMD3<Float>>.stride * Int(projectCount))
        memset(topicFBuf.contents(), 0, MemoryLayout<SIMD3<Float>>.stride * Int(topicCount))

        // Set params
        if paramBuffer == nil {
            paramBuffer = device.makeBuffer(length: MemoryLayout<GPUForceParams>.stride, options: .storageModeShared)
        }
        if let chargePBuf = paramBuffer {
            chargePBuf.contents().bindMemory(to: GPUForceParams.self, capacity: 1).pointee = GPUForceParams(
                chargeStrength: chargeStrength, crossChargeMultiplier: crossChargeMultiplier,
                sameTopicChargeScale: sameTopicChargeScale, sameProjectChargeScale: sameProjectChargeScale,
                cutoffSq: 500 * 500, nodeCount: UInt32(n))
        }
        simPBuf.contents().bindMemory(to: ForceSimParams.self, capacity: 1).pointee = ForceSimParams(
            springLength: springLength, crossProjectSpringLength: crossProjectSpringLength,
            springStrength: springStrength, cohesionStrength: cohesionStrength,
            centroidRepulsion: centroidRepulsion, topicCohesionStrength: topicCohesionStrength,
            topicCentroidRepulsion: topicCentroidRepulsion, centerStrength: centerStrength,
            center: center, alpha: alpha, damping: damping, maxSpeed: maxSpeed,
            nodeCount: UInt32(n), edgeCount: UInt32(csrEdgeCount),
            projectGroupCount: projectCount, topicGroupCount: topicCount, _pad0: 0, _pad1: 0)

        let tgSize = 256

        // 1. Charge (O(n²))
        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(nodeBuf, offset: 0, index: 0)
        encoder.setBuffer(forceBuf, offset: 0, index: 1)
        if let chargePBuf = paramBuffer { encoder.setBuffer(chargePBuf, offset: 0, index: 2) }
        let chargeTg = min(Int(pipelineState.maxTotalThreadsPerThreadgroup), tgSize)
        encoder.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: chargeTg, height: 1, depth: 1))
        encoder.memoryBarrier(scope: .buffers)

        // 2. Springs
        if csrEdgeCount > 0, let springPL = springPipeline,
           let adjOffBuf = adjOffsetsBuffer, let adjNbrBuf = adjNeighborsBuffer {
            encoder.setComputePipelineState(springPL)
            encoder.setBuffer(adjOffBuf, offset: 0, index: 0)
            encoder.setBuffer(adjNbrBuf, offset: 0, index: 1)
            encoder.setBuffer(nodeBuf, offset: 0, index: 2)
            encoder.setBuffer(forceBuf, offset: 0, index: 3)
            encoder.setBuffer(simPBuf, offset: 0, index: 4)
            let springTg = min(Int(springPL.maxTotalThreadsPerThreadgroup), tgSize)
            encoder.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: springTg, height: 1, depth: 1))
            encoder.memoryBarrier(scope: .buffers)
        }

        // 3-6. Project centroids, repulsion, cohesion
        if let centPL = centroidPipeline, projectCount > 1,
           let projOffBuf = projMemberOffsetsBuffer, let projMemBuf = projMembersBuffer {
            encoder.setComputePipelineState(centPL)
            encoder.setBuffer(projOffBuf, offset: 0, index: 0)
            encoder.setBuffer(projMemBuf, offset: 0, index: 1)
            encoder.setBuffer(nodeBuf, offset: 0, index: 2)
            encoder.setBuffer(projCBuf, offset: 0, index: 3)
            let centTg = min(Int(centPL.maxTotalThreadsPerThreadgroup), tgSize)
            encoder.dispatchThreads(MTLSize(width: Int(projectCount), height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: min(centTg, Int(projectCount)), height: 1, depth: 1))
            encoder.memoryBarrier(scope: .buffers)

            if let repPL = centroidRepulsionPipeline {
                encoder.setComputePipelineState(repPL)
                encoder.setBuffer(projCBuf, offset: 0, index: 0)
                encoder.setBuffer(projFBuf, offset: 0, index: 1)
                encoder.setBuffer(simPBuf, offset: 0, index: 2)
                gcBuf.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = projectCount
                encoder.setBuffer(gcBuf, offset: 0, index: 3)
                gtBuf.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = 0
                encoder.setBuffer(gtBuf, offset: 0, index: 4)
                repBuf.contents().bindMemory(to: Float.self, capacity: 1).pointee = centroidRepulsion
                encoder.setBuffer(repBuf, offset: 0, index: 5)
                let repTg = min(Int(repPL.maxTotalThreadsPerThreadgroup), tgSize)
                encoder.dispatchThreads(MTLSize(width: Int(projectCount), height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: min(repTg, Int(projectCount)), height: 1, depth: 1))
                encoder.memoryBarrier(scope: .buffers)
            }

            if let cohPL = cohesionPipeline {
                encoder.setComputePipelineState(cohPL)
                encoder.setBuffer(projOffBuf, offset: 0, index: 0)
                encoder.setBuffer(projMemBuf, offset: 0, index: 1)
                encoder.setBuffer(nodeBuf, offset: 0, index: 2)
                encoder.setBuffer(projCBuf, offset: 0, index: 3)
                encoder.setBuffer(projFBuf, offset: 0, index: 4)
                encoder.setBuffer(forceBuf, offset: 0, index: 5)
                encoder.setBuffer(simPBuf, offset: 0, index: 6)
                let cohTg = min(Int(cohPL.maxTotalThreadsPerThreadgroup), tgSize)
                encoder.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: cohTg, height: 1, depth: 1))
                encoder.memoryBarrier(scope: .buffers)
            }
        }

        // Topic centroids, repulsion, cohesion
        if let centPL = centroidPipeline, topicCount > 1,
           let topicOffBuf = topicMemberOffsetsBuffer, let topicMemBuf = topicMembersBuffer {
            encoder.setComputePipelineState(centPL)
            encoder.setBuffer(topicOffBuf, offset: 0, index: 0)
            encoder.setBuffer(topicMemBuf, offset: 0, index: 1)
            encoder.setBuffer(nodeBuf, offset: 0, index: 2)
            encoder.setBuffer(topicCBuf, offset: 0, index: 3)
            let centTg = min(Int(centPL.maxTotalThreadsPerThreadgroup), tgSize)
            encoder.dispatchThreads(MTLSize(width: Int(topicCount), height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: min(centTg, Int(topicCount)), height: 1, depth: 1))
            encoder.memoryBarrier(scope: .buffers)

            if let repPL = centroidRepulsionPipeline {
                encoder.setComputePipelineState(repPL)
                encoder.setBuffer(topicCBuf, offset: 0, index: 0)
                encoder.setBuffer(topicFBuf, offset: 0, index: 1)
                encoder.setBuffer(simPBuf, offset: 0, index: 2)
                gcBuf.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = topicCount
                encoder.setBuffer(gcBuf, offset: 0, index: 3)
                gtBuf.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = 1
                encoder.setBuffer(gtBuf, offset: 0, index: 4)
                repBuf.contents().bindMemory(to: Float.self, capacity: 1).pointee = topicCentroidRepulsion
                encoder.setBuffer(repBuf, offset: 0, index: 5)
                let repTg = min(Int(repPL.maxTotalThreadsPerThreadgroup), tgSize)
                encoder.dispatchThreads(MTLSize(width: Int(topicCount), height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: min(repTg, Int(topicCount)), height: 1, depth: 1))
                encoder.memoryBarrier(scope: .buffers)
            }

            if let cohPL = cohesionPipeline {
                encoder.setComputePipelineState(cohPL)
                encoder.setBuffer(topicOffBuf, offset: 0, index: 0)
                encoder.setBuffer(topicMemBuf, offset: 0, index: 1)
                encoder.setBuffer(nodeBuf, offset: 0, index: 2)
                encoder.setBuffer(topicCBuf, offset: 0, index: 3)
                encoder.setBuffer(topicFBuf, offset: 0, index: 4)
                encoder.setBuffer(forceBuf, offset: 0, index: 5)
                encoder.setBuffer(simPBuf, offset: 0, index: 6)
                let cohTg = min(Int(cohPL.maxTotalThreadsPerThreadgroup), tgSize)
                encoder.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: cohTg, height: 1, depth: 1))
                encoder.memoryBarrier(scope: .buffers)
            }
        }
    }

    /// Read back force results from GPU shared buffer after command buffer completion.
    /// Call this from the command buffer's completion handler or after waitUntilCompleted.
    func readBackForces() -> (fx: [Float], fy: [Float], fz: [Float])? {
        let n = lastEncodedNodeCount
        guard n > 0, let forceBuf = fullForceBuffer else { return nil }
        let forcePtr = forceBuf.contents().bindMemory(to: SIMD3<Float>.self, capacity: n)
        var fx = [Float](repeating: 0, count: n)
        var fy = [Float](repeating: 0, count: n)
        var fz = [Float](repeating: 0, count: n)
        for i in 0..<n { fx[i] = forcePtr[i].x; fy[i] = forcePtr[i].y; fz[i] = forcePtr[i].z }
        return (fx, fy, fz)
    }

    // Full GPU force simulation pipelines
    private var springPipeline: MTLComputePipelineState?
    private var centroidPipeline: MTLComputePipelineState?
    private var centroidRepulsionPipeline: MTLComputePipelineState?
    private var cohesionPipeline: MTLComputePipelineState?

    // Persistent GPU buffers for full simulation
    private var fullNodeBuffer: MTLBuffer?      // ForceNodeFull[N]
    private var fullForceBuffer: MTLBuffer?     // float3[N] accumulated forces
    private var projCentroidBuffer: MTLBuffer?  // GroupCentroid[maxGroups]
    private var topicCentroidBuffer: MTLBuffer? // GroupCentroid[maxGroups]
    private var projForceBuffer: MTLBuffer?     // float3[maxGroups]
    private var topicForceBuffer: MTLBuffer?    // float3[maxGroups]
    private var simParamBuffer: MTLBuffer?      // ForceSimParams
    private var groupTypeBuffer: MTLBuffer?     // uint (0=project, 1=topic)
    private var nodeCountBuffer: MTLBuffer?     // uint
    private var repulsionBuffer: MTLBuffer?     // float
    private var groupCountBuffer: MTLBuffer?    // uint
    private var fullNodeCapacity: Int = 0
    private var fullGroupCapacity: Int = 0
    private var fullSimReady: Bool = false

    // CSR topology buffers (rebuilt on topology change, not per-frame)
    private var adjOffsetsBuffer: MTLBuffer?       // [N+1] uint — edge adjacency offsets
    private var adjNeighborsBuffer: MTLBuffer?     // [2*E] uint — edge adjacency neighbors
    private var projMemberOffsetsBuffer: MTLBuffer? // [G+1] uint — project group offsets
    private var projMembersBuffer: MTLBuffer?      // [N] uint — project group members
    private var topicMemberOffsetsBuffer: MTLBuffer? // [G+1] uint — topic group offsets
    private var topicMembersBuffer: MTLBuffer?     // [N] uint — topic group members
    private var adjNeighborCapacity: Int = 0
    private var csrNodeCapacity: Int = 0
    private var csrProjectGroupCapacity: Int = 0
    private var csrTopicGroupCapacity: Int = 0
    private var csrEdgeCount: Int = 0  // tracks current edge count for spring dispatch guard

    /// Result from full GPU simulation.
    struct FullSimResult: Sendable {
        let fx: [Float], fy: [Float], fz: [Float]
        let maxSpeedSq: Float
        let totalKineticEnergy: Float
    }

    /// Result from GPU charge force computation.
    struct ChargeResult: Sendable {
        let fx: [Float], fy: [Float], fz: [Float]
    }

    // MARK: - Charge-Only Path (hybrid fallback)

    /// Dispatch charge force computation on GPU. Non-blocking — returns via async/await.
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

        // Serial gate: if a previous command buffer is still in-flight, return zeros
        let acquired = gpuInFlight.withLock { inFlight -> Bool in
            if inFlight { return false }
            inFlight = true
            return true
        }
        if !acquired { return ChargeResult(fx: [Float](repeating: 0, count: n),
                                           fy: [Float](repeating: 0, count: n),
                                           fz: [Float](repeating: 0, count: n)) }

        // Resize buffers only when capacity exceeded (2x over-allocation avoids
        // reallocation on every single addNode during sync)
        currentNodeCount = n
        if n > bufferCapacity {
            let capacity = max(n * 2, 512)
            let nodeSize = MemoryLayout<ForceNodeFull>.stride * capacity
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

        // Pack node data as ForceNodeFull (velocity fields zeroed — charge kernel ignores them)
        let nodePtr = nodeBuffer.contents().bindMemory(to: ForceNodeFull.self, capacity: n)
        for i in 0..<n {
            nodePtr[i] = ForceNodeFull(
                px: positions.x[i], py: positions.y[i], pz: positions.z[i],
                vx: 0, vy: 0, vz: 0,
                projectGroup: Int32(projectGroups[i]),
                topicGroup: Int32(i < topicGroups.count ? topicGroups[i] : -1)
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
        let gate = gpuInFlight
        return await withCheckedContinuation { continuation in
            cmdBuffer.addCompletedHandler { @Sendable buffer in
                defer { gate.withLock { $0 = false } }
                if buffer.status == .error {
                    continuation.resume(returning: ChargeResult(
                        fx: [Float](repeating: 0, count: capturedN),
                        fy: [Float](repeating: 0, count: capturedN),
                        fz: [Float](repeating: 0, count: capturedN)))
                    return
                }
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

    // MARK: - CSR Topology Rebuild

    /// Build CSR (Compressed Sparse Row) index structures for gather-pattern GPU kernels.
    /// Called on topology change (addNode/removeNodes/addEdge/updateGraph), not per-frame.
    /// O(N+E), sub-millisecond for typical graph sizes.
    func rebuildTopologyBuffers(
        nodeCount n: Int,
        edges: [(Int, Int)],
        projectGroups: [Int],
        topicGroups: [Int]
    ) {
        let e = edges.count

        // ── Edge adjacency CSR: adjOffsets[N+1] + adjNeighbors[2*E] ──────────
        let neighborCount = e * 2
        if n + 1 > csrNodeCapacity || neighborCount > adjNeighborCapacity {
            let nodeCap = max((n + 1) * 2, 512)
            let neighborCap = max(neighborCount * 2, 1024)
            adjOffsetsBuffer = device.makeBuffer(length: nodeCap * MemoryLayout<UInt32>.stride, options: .storageModeShared)
            adjNeighborsBuffer = device.makeBuffer(length: neighborCap * MemoryLayout<UInt32>.stride, options: .storageModeShared)
            csrNodeCapacity = nodeCap
            adjNeighborCapacity = neighborCap
        }

        if let offsetsBuf = adjOffsetsBuffer, let neighborsBuf = adjNeighborsBuffer {
            let offsets = offsetsBuf.contents().bindMemory(to: UInt32.self, capacity: n + 1)
            let neighbors = neighborsBuf.contents().bindMemory(to: UInt32.self, capacity: neighborCount)

            // Count neighbors per node
            memset(offsets, 0, (n + 1) * MemoryLayout<UInt32>.stride)
            for (s, t) in edges {
                offsets[s] &+= 1
                offsets[t] &+= 1  // bidirectional
            }

            // Prefix sum → offsets
            var running: UInt32 = 0
            for i in 0...n {
                let count = offsets[i]
                offsets[i] = running
                running &+= count
            }

            // Fill neighbor lists (use a copy of offsets as write cursors)
            var cursors = [UInt32](repeating: 0, count: n)
            for i in 0..<n { cursors[i] = offsets[i] }
            for (s, t) in edges {
                neighbors[Int(cursors[s])] = UInt32(t)
                cursors[s] &+= 1
                neighbors[Int(cursors[t])] = UInt32(s)
                cursors[t] &+= 1
            }
        }
        csrEdgeCount = e

        // ── Project group membership CSR ─────────────────────────────────────
        let projectCount = max((projectGroups.max() ?? 0) + 1, 1)
        if projectCount + 1 > csrProjectGroupCapacity {
            let cap = max((projectCount + 1) * 2, 64)
            projMemberOffsetsBuffer = device.makeBuffer(length: cap * MemoryLayout<UInt32>.stride, options: .storageModeShared)
            csrProjectGroupCapacity = cap
        }
        // Members buffer needs N entries
        if n > csrNodeCapacity {
            // Already resized above, but ensure members buffer exists
        }
        projMembersBuffer = ensureBuffer(projMembersBuffer, count: max(n, 1), stride: MemoryLayout<UInt32>.stride)

        if let offsetsBuf = projMemberOffsetsBuffer, let membersBuf = projMembersBuffer {
            buildGroupCSR(offsets: offsetsBuf, members: membersBuf,
                          groups: projectGroups, nodeCount: n, groupCount: projectCount)
        }

        // ── Topic group membership CSR ───────────────────────────────────────
        let topicCount = max((topicGroups.max() ?? 0) + 1, 1)
        if topicCount + 1 > csrTopicGroupCapacity {
            let cap = max((topicCount + 1) * 2, 64)
            topicMemberOffsetsBuffer = device.makeBuffer(length: cap * MemoryLayout<UInt32>.stride, options: .storageModeShared)
            csrTopicGroupCapacity = cap
        }
        topicMembersBuffer = ensureBuffer(topicMembersBuffer, count: max(n, 1), stride: MemoryLayout<UInt32>.stride)

        if let offsetsBuf = topicMemberOffsetsBuffer, let membersBuf = topicMembersBuffer {
            buildGroupCSR(offsets: offsetsBuf, members: membersBuf,
                          groups: topicGroups, nodeCount: n, groupCount: topicCount)
        }
    }

    /// Helper: build group membership CSR into pre-allocated buffers.
    private func buildGroupCSR(offsets offsetsBuf: MTLBuffer, members membersBuf: MTLBuffer,
                               groups: [Int], nodeCount n: Int, groupCount: Int) {
        let offsets = offsetsBuf.contents().bindMemory(to: UInt32.self, capacity: groupCount + 1)
        let members = membersBuf.contents().bindMemory(to: UInt32.self, capacity: n)

        // Count members per group
        memset(offsets, 0, (groupCount + 1) * MemoryLayout<UInt32>.stride)
        for i in 0..<n {
            let g = groups[i]
            if g >= 0 && g < groupCount { offsets[g] &+= 1 }
        }

        // Prefix sum → offsets
        var running: UInt32 = 0
        for g in 0...groupCount {
            let count = offsets[g]
            offsets[g] = running
            running &+= count
        }

        // Fill member lists
        var cursors = [UInt32](repeating: 0, count: groupCount)
        for g in 0..<groupCount { cursors[g] = offsets[g] }
        for i in 0..<n {
            let g = groups[i]
            if g >= 0 && g < groupCount {
                members[Int(cursors[g])] = UInt32(i)
                cursors[g] &+= 1
            }
        }
    }

    /// Ensure a buffer exists with at least the given capacity.
    private func ensureBuffer(_ existing: MTLBuffer?, count: Int, stride: Int) -> MTLBuffer? {
        if let buf = existing, buf.length >= count * stride { return buf }
        let cap = max(count * 2, 512)
        return device.makeBuffer(length: cap * stride, options: .storageModeShared)
    }

    // MARK: - Full GPU Force Simulation

    /// Dispatch the complete force simulation pipeline on GPU:
    /// charge → springs (CSR gather) → centroids (CSR gather) → centroid repulsion → cohesion.
    /// Returns combined forces. Integration stays on CPU (position readback needed anyway).
    /// No per-frame buffer allocation. Topology-dependent CSR structures pre-built.
    func dispatchFullSimulation(
        positions: (x: [Float], y: [Float], z: [Float]),
        projectGroups: [Int],
        topicGroups: [Int],
        chargeStrength: Float,
        crossChargeMultiplier: Float,
        sameTopicChargeScale: Float,
        sameProjectChargeScale: Float,
        springLength: Float,
        crossProjectSpringLength: Float,
        springStrength: Float,
        cohesionStrength: Float,
        centroidRepulsion: Float,
        topicCohesionStrength: Float,
        topicCentroidRepulsion: Float,
        centerStrength: Float,
        center: SIMD3<Float>,
        alpha: Float,
        damping: Float,
        maxSpeed: Float
    ) async -> FullSimResult {
        let n = positions.x.count
        let zeroResult = FullSimResult(fx: [Float](repeating: 0, count: n),
                                       fy: [Float](repeating: 0, count: n),
                                       fz: [Float](repeating: 0, count: n),
                                       maxSpeedSq: 0, totalKineticEnergy: 0)
        guard n > 1, fullSimReady else { return zeroResult }

        // Serial gate: reject concurrent dispatch to prevent buffer races
        let acquired = gpuInFlight.withLock { inFlight -> Bool in
            if inFlight { return false }
            inFlight = true
            return true
        }
        if !acquired { return zeroResult }

        let projectCount = UInt32(max((projectGroups.max() ?? 0) + 1, 1))
        let topicCount = UInt32(max((topicGroups.max() ?? 0) + 1, 1))
        let maxGroups = max(Int(projectCount), Int(topicCount))

        // Ensure GPU buffer capacity (2x over-allocation)
        if n > fullNodeCapacity {
            let cap = max(n * 2, 512)
            fullNodeBuffer = device.makeBuffer(length: cap * MemoryLayout<ForceNodeFull>.stride, options: .storageModeShared)
            fullForceBuffer = device.makeBuffer(length: cap * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)
            fullNodeCapacity = cap
        }
        if maxGroups > fullGroupCapacity {
            let cap = max(maxGroups * 2, 64)
            projCentroidBuffer = device.makeBuffer(length: cap * MemoryLayout<GroupCentroid>.stride, options: .storageModeShared)
            topicCentroidBuffer = device.makeBuffer(length: cap * MemoryLayout<GroupCentroid>.stride, options: .storageModeShared)
            projForceBuffer = device.makeBuffer(length: cap * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)
            topicForceBuffer = device.makeBuffer(length: cap * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)
            fullGroupCapacity = cap
        }

        guard let nodeBuf = fullNodeBuffer,
              let forceBuf = fullForceBuffer,
              let projCBuf = projCentroidBuffer,
              let topicCBuf = topicCentroidBuffer,
              let projFBuf = projForceBuffer,
              let topicFBuf = topicForceBuffer,
              let simPBuf = simParamBuffer,
              let gtBuf = groupTypeBuffer,
              let repBuf = repulsionBuffer,
              let gcBuf = groupCountBuffer
        else { return zeroResult }

        // Pack node data (positions + groups; velocity fields zeroed — not used by gather kernels)
        let nodePtr = nodeBuf.contents().bindMemory(to: ForceNodeFull.self, capacity: n)
        for i in 0..<n {
            nodePtr[i] = ForceNodeFull(
                px: positions.x[i], py: positions.y[i], pz: positions.z[i],
                vx: 0, vy: 0, vz: 0,
                projectGroup: Int32(projectGroups[i]),
                topicGroup: Int32(i < topicGroups.count ? topicGroups[i] : -1)
            )
        }

        // Zero force buffer
        memset(forceBuf.contents(), 0, MemoryLayout<SIMD3<Float>>.stride * n)

        // Set charge params (reuse persistent paramBuffer from charge-only path)
        if paramBuffer == nil {
            paramBuffer = device.makeBuffer(length: MemoryLayout<GPUForceParams>.stride, options: .storageModeShared)
        }
        if let chargePBuf = paramBuffer {
            chargePBuf.contents().bindMemory(to: GPUForceParams.self, capacity: 1).pointee = GPUForceParams(
                chargeStrength: chargeStrength,
                crossChargeMultiplier: crossChargeMultiplier,
                sameTopicChargeScale: sameTopicChargeScale,
                sameProjectChargeScale: sameProjectChargeScale,
                cutoffSq: 500 * 500,
                nodeCount: UInt32(n)
            )
        }

        // Set simulation params (for spring + cohesion kernels)
        let simParams = ForceSimParams(
            springLength: springLength,
            crossProjectSpringLength: crossProjectSpringLength,
            springStrength: springStrength,
            cohesionStrength: cohesionStrength,
            centroidRepulsion: centroidRepulsion,
            topicCohesionStrength: topicCohesionStrength,
            topicCentroidRepulsion: topicCentroidRepulsion,
            centerStrength: centerStrength,
            center: center,
            alpha: alpha,
            damping: damping,
            maxSpeed: maxSpeed,
            nodeCount: UInt32(n),
            edgeCount: UInt32(csrEdgeCount),
            projectGroupCount: projectCount,
            topicGroupCount: topicCount,
            _pad0: 0, _pad1: 0
        )
        simPBuf.contents().bindMemory(to: ForceSimParams.self, capacity: 1).pointee = simParams

        // Zero centroid and group force buffers
        memset(projCBuf.contents(), 0, MemoryLayout<GroupCentroid>.stride * Int(projectCount))
        memset(topicCBuf.contents(), 0, MemoryLayout<GroupCentroid>.stride * Int(topicCount))
        memset(projFBuf.contents(), 0, MemoryLayout<SIMD3<Float>>.stride * Int(projectCount))
        memset(topicFBuf.contents(), 0, MemoryLayout<SIMD3<Float>>.stride * Int(topicCount))

        // Split into two command buffers: the O(n²) charge kernel gets its own
        // command buffer so the GPU scheduler can interleave WindowServer compositing
        // between charge and the remaining O(n)/O(E) passes. A single monolithic
        // command buffer with all passes starves the display compositor on Apple Silicon.

        let tgSize = 256

        // ── Command buffer 1: Charge forces (O(n²)) ────────────────────────
        guard let chargeCmdBuf = commandQueue.makeCommandBuffer(),
              let chargeEnc = chargeCmdBuf.makeComputeCommandEncoder()
        else {
            gpuInFlight.withLock { $0 = false }
            return zeroResult
        }

        chargeEnc.setComputePipelineState(pipelineState)
        chargeEnc.setBuffer(nodeBuf, offset: 0, index: 0)
        chargeEnc.setBuffer(forceBuf, offset: 0, index: 1)
        if let chargePBuf = paramBuffer {
            chargeEnc.setBuffer(chargePBuf, offset: 0, index: 2)
        }
        let chargeTg = min(Int(pipelineState.maxTotalThreadsPerThreadgroup), tgSize)
        chargeEnc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: chargeTg, height: 1, depth: 1))
        chargeEnc.endEncoding()
        chargeCmdBuf.commit()

        // ── Command buffer 2: Springs + centroids + cohesion (all O(n) or O(E)) ──
        guard let cmdBuffer = commandQueue.makeCommandBuffer(),
              let enc = cmdBuffer.makeComputeCommandEncoder()
        else {
            gpuInFlight.withLock { $0 = false }
            return zeroResult
        }

        // 2. Spring forces (O(E) gather via CSR — one thread per node)
        if csrEdgeCount > 0,
           let springPL = springPipeline,
           let adjOffBuf = adjOffsetsBuffer,
           let adjNbrBuf = adjNeighborsBuffer {
            enc.setComputePipelineState(springPL)
            enc.setBuffer(adjOffBuf, offset: 0, index: 0)
            enc.setBuffer(adjNbrBuf, offset: 0, index: 1)
            enc.setBuffer(nodeBuf, offset: 0, index: 2)
            enc.setBuffer(forceBuf, offset: 0, index: 3)
            enc.setBuffer(simPBuf, offset: 0, index: 4)
            let springTg = min(Int(springPL.maxTotalThreadsPerThreadgroup), tgSize)
            enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: springTg, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
        }

        // 3. Project centroid accumulation (gather — one thread per group)
        if let centPL = centroidPipeline, projectCount > 1,
           let projOffBuf = projMemberOffsetsBuffer,
           let projMemBuf = projMembersBuffer {
            enc.setComputePipelineState(centPL)
            enc.setBuffer(projOffBuf, offset: 0, index: 0)
            enc.setBuffer(projMemBuf, offset: 0, index: 1)
            enc.setBuffer(nodeBuf, offset: 0, index: 2)
            enc.setBuffer(projCBuf, offset: 0, index: 3)
            let centTg = min(Int(centPL.maxTotalThreadsPerThreadgroup), tgSize)
            enc.dispatchThreads(MTLSize(width: Int(projectCount), height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: min(centTg, Int(projectCount)), height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)

            // Project centroid repulsion (O(groups²/2) — one thread per pair)
            if let repPL = centroidRepulsionPipeline, projectCount > 1 {
                enc.setComputePipelineState(repPL)
                enc.setBuffer(projCBuf, offset: 0, index: 0)
                enc.setBuffer(projFBuf, offset: 0, index: 1)
                repBuf.contents().bindMemory(to: Float.self, capacity: 1).pointee = centroidRepulsion
                enc.setBuffer(repBuf, offset: 0, index: 2)
                gcBuf.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = projectCount
                enc.setBuffer(gcBuf, offset: 0, index: 3)
                let pairs = Int(projectCount) * (Int(projectCount) - 1) / 2
                let repTg = min(Int(repPL.maxTotalThreadsPerThreadgroup), tgSize)
                let repGroups = max(1, (pairs + repTg - 1) / repTg)
                enc.dispatchThreadgroups(MTLSize(width: repGroups, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: repTg, height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)
            }

            // Apply project cohesion forces (one thread per node)
            if let cohPL = cohesionPipeline {
                enc.setComputePipelineState(cohPL)
                enc.setBuffer(nodeBuf, offset: 0, index: 0)
                enc.setBuffer(projCBuf, offset: 0, index: 1)
                enc.setBuffer(projFBuf, offset: 0, index: 2)
                enc.setBuffer(forceBuf, offset: 0, index: 3)
                enc.setBuffer(simPBuf, offset: 0, index: 4)
                gtBuf.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = 0
                enc.setBuffer(gtBuf, offset: 0, index: 5)
                let cohTg = min(Int(cohPL.maxTotalThreadsPerThreadgroup), tgSize)
                enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: cohTg, height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)
            }
        }

        // 4. Topic centroid accumulation + repulsion + cohesion
        if let centPL = centroidPipeline, topicCount > 1,
           let topicOffBuf = topicMemberOffsetsBuffer,
           let topicMemBuf = topicMembersBuffer {
            enc.setComputePipelineState(centPL)
            enc.setBuffer(topicOffBuf, offset: 0, index: 0)
            enc.setBuffer(topicMemBuf, offset: 0, index: 1)
            enc.setBuffer(nodeBuf, offset: 0, index: 2)
            enc.setBuffer(topicCBuf, offset: 0, index: 3)
            let centTg = min(Int(centPL.maxTotalThreadsPerThreadgroup), tgSize)
            enc.dispatchThreads(MTLSize(width: Int(topicCount), height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: min(centTg, Int(topicCount)), height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)

            if let repPL = centroidRepulsionPipeline, topicCount > 1 {
                enc.setComputePipelineState(repPL)
                enc.setBuffer(topicCBuf, offset: 0, index: 0)
                enc.setBuffer(topicFBuf, offset: 0, index: 1)
                repBuf.contents().bindMemory(to: Float.self, capacity: 1).pointee = topicCentroidRepulsion
                enc.setBuffer(repBuf, offset: 0, index: 2)
                gcBuf.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = topicCount
                enc.setBuffer(gcBuf, offset: 0, index: 3)
                let pairs = Int(topicCount) * (Int(topicCount) - 1) / 2
                let repTg = min(Int(repPL.maxTotalThreadsPerThreadgroup), tgSize)
                let repGroups = max(1, (pairs + repTg - 1) / repTg)
                enc.dispatchThreadgroups(MTLSize(width: repGroups, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: repTg, height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)
            }

            if let cohPL = cohesionPipeline {
                enc.setComputePipelineState(cohPL)
                enc.setBuffer(nodeBuf, offset: 0, index: 0)
                enc.setBuffer(topicCBuf, offset: 0, index: 1)
                enc.setBuffer(topicFBuf, offset: 0, index: 2)
                enc.setBuffer(forceBuf, offset: 0, index: 3)
                enc.setBuffer(simPBuf, offset: 0, index: 4)
                gtBuf.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = 1
                enc.setBuffer(gtBuf, offset: 0, index: 5)
                let cohTg = min(Int(cohPL.maxTotalThreadsPerThreadgroup), tgSize)
                enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: cohTg, height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)
            }
        }

        enc.endEncoding()

        // Read back forces (integration stays on CPU for position readback compatibility)
        let capturedForceBuf = forceBuf
        let capturedN = n
        let gate = gpuInFlight
        return await withCheckedContinuation { continuation in
            cmdBuffer.addCompletedHandler { @Sendable buffer in
                defer { gate.withLock { $0 = false } }
                if buffer.status == .error {
                    continuation.resume(returning: FullSimResult(
                        fx: [Float](repeating: 0, count: capturedN),
                        fy: [Float](repeating: 0, count: capturedN),
                        fz: [Float](repeating: 0, count: capturedN),
                        maxSpeedSq: 0, totalKineticEnergy: 0))
                    return
                }
                let forcePtr = capturedForceBuf.contents().bindMemory(to: SIMD3<Float>.self, capacity: capturedN)
                var fx = [Float](repeating: 0, count: capturedN)
                var fy = [Float](repeating: 0, count: capturedN)
                var fz = [Float](repeating: 0, count: capturedN)
                for i in 0..<capturedN {
                    fx[i] = forcePtr[i].x
                    fy[i] = forcePtr[i].y
                    fz[i] = forcePtr[i].z
                }
                continuation.resume(returning: FullSimResult(
                    fx: fx, fy: fy, fz: fz, maxSpeedSq: 0, totalKineticEnergy: 0
                ))
            }
            cmdBuffer.commit()
        }
    }

    /// Synchronous GPU force dispatch — blocks until GPU completes.
    /// Eliminates the async feedback delay that causes micro-oscillations.
    /// Forces are computed from current-frame positions and returned immediately.
    func dispatchFullSimulationSync(
        x: [Float], y: [Float], z: [Float],
        projectGroups: [Int], topicGroups: [Int],
        chargeStrength: Float, crossChargeMultiplier: Float,
        sameTopicChargeScale: Float, sameProjectChargeScale: Float,
        springLength: Float, crossProjectSpringLength: Float,
        springStrength: Float,
        cohesionStrength: Float, centroidRepulsion: Float,
        topicCohesionStrength: Float, topicCentroidRepulsion: Float,
        centerStrength: Float, center: SIMD3<Float>,
        alpha: Float, damping: Float, maxSpeed: Float
    ) -> (fx: [Float], fy: [Float], fz: [Float]) {
        let n = x.count
        let zero = ([Float](repeating: 0, count: n), [Float](repeating: 0, count: n), [Float](repeating: 0, count: n))
        guard n > 1, fullSimReady else { return zero }

        let acquired = gpuInFlight.withLock { inFlight -> Bool in
            if inFlight { return false }
            inFlight = true
            return true
        }
        if !acquired { return zero }

        let projectCount = UInt32(max((projectGroups.max() ?? 0) + 1, 1))
        let topicCount = UInt32(max((topicGroups.max() ?? 0) + 1, 1))
        let maxGroups = max(Int(projectCount), Int(topicCount))

        if n > fullNodeCapacity {
            let cap = max(n * 2, 512)
            fullNodeBuffer = device.makeBuffer(length: cap * MemoryLayout<ForceNodeFull>.stride, options: .storageModeShared)
            fullForceBuffer = device.makeBuffer(length: cap * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)
            fullNodeCapacity = cap
        }
        if maxGroups > fullGroupCapacity {
            let cap = max(maxGroups * 2, 64)
            projCentroidBuffer = device.makeBuffer(length: cap * MemoryLayout<GroupCentroid>.stride, options: .storageModeShared)
            topicCentroidBuffer = device.makeBuffer(length: cap * MemoryLayout<GroupCentroid>.stride, options: .storageModeShared)
            projForceBuffer = device.makeBuffer(length: cap * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)
            topicForceBuffer = device.makeBuffer(length: cap * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)
            fullGroupCapacity = cap
        }

        guard let nodeBuf = fullNodeBuffer,
              let forceBuf = fullForceBuffer,
              let projCBuf = projCentroidBuffer,
              let topicCBuf = topicCentroidBuffer,
              let projFBuf = projForceBuffer,
              let topicFBuf = topicForceBuffer,
              let simPBuf = simParamBuffer,
              let gtBuf = groupTypeBuffer,
              let repBuf = repulsionBuffer,
              let gcBuf = groupCountBuffer
        else {
            gpuInFlight.withLock { $0 = false }
            return zero
        }

        // Pack node data
        let nodePtr = nodeBuf.contents().bindMemory(to: ForceNodeFull.self, capacity: n)
        for i in 0..<n {
            nodePtr[i] = ForceNodeFull(
                px: x[i], py: y[i], pz: z[i],
                vx: 0, vy: 0, vz: 0,
                projectGroup: Int32(projectGroups[i]),
                topicGroup: Int32(i < topicGroups.count ? topicGroups[i] : -1)
            )
        }

        memset(forceBuf.contents(), 0, MemoryLayout<SIMD3<Float>>.stride * n)
        if paramBuffer == nil {
            paramBuffer = device.makeBuffer(length: MemoryLayout<GPUForceParams>.stride, options: .storageModeShared)
        }
        if let chargePBuf = paramBuffer {
            chargePBuf.contents().bindMemory(to: GPUForceParams.self, capacity: 1).pointee = GPUForceParams(
                chargeStrength: chargeStrength,
                crossChargeMultiplier: crossChargeMultiplier,
                sameTopicChargeScale: sameTopicChargeScale,
                sameProjectChargeScale: sameProjectChargeScale,
                cutoffSq: 500 * 500,
                nodeCount: UInt32(n)
            )
        }

        let simParams = ForceSimParams(
            springLength: springLength,
            crossProjectSpringLength: crossProjectSpringLength,
            springStrength: springStrength,
            cohesionStrength: cohesionStrength,
            centroidRepulsion: centroidRepulsion,
            topicCohesionStrength: topicCohesionStrength,
            topicCentroidRepulsion: topicCentroidRepulsion,
            centerStrength: centerStrength,
            center: center,
            alpha: alpha,
            damping: damping,
            maxSpeed: maxSpeed,
            nodeCount: UInt32(n),
            edgeCount: UInt32(csrEdgeCount),
            projectGroupCount: projectCount,
            topicGroupCount: topicCount,
            _pad0: 0, _pad1: 0
        )
        simPBuf.contents().bindMemory(to: ForceSimParams.self, capacity: 1).pointee = simParams

        memset(projCBuf.contents(), 0, MemoryLayout<GroupCentroid>.stride * Int(projectCount))
        memset(topicCBuf.contents(), 0, MemoryLayout<GroupCentroid>.stride * Int(topicCount))
        memset(projFBuf.contents(), 0, MemoryLayout<SIMD3<Float>>.stride * Int(projectCount))
        memset(topicFBuf.contents(), 0, MemoryLayout<SIMD3<Float>>.stride * Int(topicCount))

        let tgSize = 256

        // Single command buffer — charge + springs + cohesion all in one submission
        guard let cmdBuffer = commandQueue.makeCommandBuffer(),
              let enc = cmdBuffer.makeComputeCommandEncoder()
        else {
            gpuInFlight.withLock { $0 = false }
            return zero
        }

        // 1. Charge forces (O(n²))
        enc.setComputePipelineState(pipelineState)
        enc.setBuffer(nodeBuf, offset: 0, index: 0)
        enc.setBuffer(forceBuf, offset: 0, index: 1)
        if let chargePBuf = paramBuffer {
            enc.setBuffer(chargePBuf, offset: 0, index: 2)
        }
        let chargeTg = min(Int(pipelineState.maxTotalThreadsPerThreadgroup), tgSize)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: chargeTg, height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)

        // 2. Spring forces
        if csrEdgeCount > 0,
           let springPL = springPipeline,
           let adjOffBuf = adjOffsetsBuffer,
           let adjNbrBuf = adjNeighborsBuffer {
            enc.setComputePipelineState(springPL)
            enc.setBuffer(adjOffBuf, offset: 0, index: 0)
            enc.setBuffer(adjNbrBuf, offset: 0, index: 1)
            enc.setBuffer(nodeBuf, offset: 0, index: 2)
            enc.setBuffer(forceBuf, offset: 0, index: 3)
            enc.setBuffer(simPBuf, offset: 0, index: 4)
            let springTg = min(Int(springPL.maxTotalThreadsPerThreadgroup), tgSize)
            enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: springTg, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
        }

        // 3-6. Project/topic centroid, repulsion, cohesion (same as async path)
        if let centPL = centroidPipeline, projectCount > 1,
           let projOffBuf = projMemberOffsetsBuffer,
           let projMemBuf = projMembersBuffer {
            enc.setComputePipelineState(centPL)
            enc.setBuffer(projOffBuf, offset: 0, index: 0)
            enc.setBuffer(projMemBuf, offset: 0, index: 1)
            enc.setBuffer(nodeBuf, offset: 0, index: 2)
            enc.setBuffer(projCBuf, offset: 0, index: 3)
            let centTg = min(Int(centPL.maxTotalThreadsPerThreadgroup), tgSize)
            enc.dispatchThreads(MTLSize(width: Int(projectCount), height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: min(centTg, Int(projectCount)), height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)

            if let repPL = centroidRepulsionPipeline {
                enc.setComputePipelineState(repPL)
                enc.setBuffer(projCBuf, offset: 0, index: 0)
                enc.setBuffer(projFBuf, offset: 0, index: 1)
                enc.setBuffer(simPBuf, offset: 0, index: 2)
                gcBuf.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = projectCount
                enc.setBuffer(gcBuf, offset: 0, index: 3)
                gtBuf.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = 0
                enc.setBuffer(gtBuf, offset: 0, index: 4)
                repBuf.contents().bindMemory(to: Float.self, capacity: 1).pointee = centroidRepulsion
                enc.setBuffer(repBuf, offset: 0, index: 5)
                let repTg = min(Int(repPL.maxTotalThreadsPerThreadgroup), tgSize)
                enc.dispatchThreads(MTLSize(width: Int(projectCount), height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: min(repTg, Int(projectCount)), height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)
            }

            if let cohPL = cohesionPipeline {
                enc.setComputePipelineState(cohPL)
                enc.setBuffer(projOffBuf, offset: 0, index: 0)
                enc.setBuffer(projMemBuf, offset: 0, index: 1)
                enc.setBuffer(nodeBuf, offset: 0, index: 2)
                enc.setBuffer(projCBuf, offset: 0, index: 3)
                enc.setBuffer(projFBuf, offset: 0, index: 4)
                enc.setBuffer(forceBuf, offset: 0, index: 5)
                enc.setBuffer(simPBuf, offset: 0, index: 6)
                let cohTg = min(Int(cohPL.maxTotalThreadsPerThreadgroup), tgSize)
                enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: cohTg, height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)
            }
        }

        if let centPL = centroidPipeline, topicCount > 1,
           let topicOffBuf = topicMemberOffsetsBuffer,
           let topicMemBuf = topicMembersBuffer {
            enc.setComputePipelineState(centPL)
            enc.setBuffer(topicOffBuf, offset: 0, index: 0)
            enc.setBuffer(topicMemBuf, offset: 0, index: 1)
            enc.setBuffer(nodeBuf, offset: 0, index: 2)
            enc.setBuffer(topicCBuf, offset: 0, index: 3)
            let centTg = min(Int(centPL.maxTotalThreadsPerThreadgroup), tgSize)
            enc.dispatchThreads(MTLSize(width: Int(topicCount), height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: min(centTg, Int(topicCount)), height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)

            if let repPL = centroidRepulsionPipeline {
                enc.setComputePipelineState(repPL)
                enc.setBuffer(topicCBuf, offset: 0, index: 0)
                enc.setBuffer(topicFBuf, offset: 0, index: 1)
                enc.setBuffer(simPBuf, offset: 0, index: 2)
                gcBuf.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = topicCount
                enc.setBuffer(gcBuf, offset: 0, index: 3)
                gtBuf.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = 1
                enc.setBuffer(gtBuf, offset: 0, index: 4)
                repBuf.contents().bindMemory(to: Float.self, capacity: 1).pointee = topicCentroidRepulsion
                enc.setBuffer(repBuf, offset: 0, index: 5)
                let repTg = min(Int(repPL.maxTotalThreadsPerThreadgroup), tgSize)
                enc.dispatchThreads(MTLSize(width: Int(topicCount), height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: min(repTg, Int(topicCount)), height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)
            }

            if let cohPL = cohesionPipeline {
                enc.setComputePipelineState(cohPL)
                enc.setBuffer(topicOffBuf, offset: 0, index: 0)
                enc.setBuffer(topicMemBuf, offset: 0, index: 1)
                enc.setBuffer(nodeBuf, offset: 0, index: 2)
                enc.setBuffer(topicCBuf, offset: 0, index: 3)
                enc.setBuffer(topicFBuf, offset: 0, index: 4)
                enc.setBuffer(forceBuf, offset: 0, index: 5)
                enc.setBuffer(simPBuf, offset: 0, index: 6)
                let cohTg = min(Int(cohPL.maxTotalThreadsPerThreadgroup), tgSize)
                enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: cohTg, height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)
            }
        }

        enc.endEncoding()
        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()
        gpuInFlight.withLock { $0 = false }

        if cmdBuffer.status == .error {
            return zero
        }

        // Read back forces directly from shared buffer
        let forcePtr = forceBuf.contents().bindMemory(to: SIMD3<Float>.self, capacity: n)
        var fx = [Float](repeating: 0, count: n)
        var fy = [Float](repeating: 0, count: n)
        var fz = [Float](repeating: 0, count: n)
        for i in 0..<n {
            fx[i] = forcePtr[i].x
            fy[i] = forcePtr[i].y
            fz[i] = forcePtr[i].z
        }
        return (fx, fy, fz)
    }
}
