import Foundation
import CEngramSceneTypes
@preconcurrency import Metal
import simd
import os

// MARK: - Metal Force Compute

/// GPU-accelerated force simulation.
/// encodeForces() encodes into an external compute encoder (same CB as render).
/// No separate command queue — forces run synchronously in the render frame.
public final class MetalForceCompute: @unchecked Sendable {
    private let device: MTLDevice
    private let pipelineState: MTLComputePipelineState

    // Persistent buffers — resized only when capacity exceeded (2x over-allocation)
    private var nodeBuffer: MTLBuffer?
    private var forceBuffer: MTLBuffer?
    private var paramBuffer: MTLBuffer?
    private var currentNodeCount: Int = 0
    private var bufferCapacity: Int = 0

    /// Simple flag — only one dispatch at a time. Checked/set on main thread,
    /// cleared via Task { @MainActor } from completion handler. No cross-thread lock.
    public var inFlight = false

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
    public init?(device: MTLDevice, library: MTLLibrary) {
        guard let function = library.makeFunction(name: "compute_charge_forces"),
              let pipeline = try? device.makeComputePipelineState(function: function)
        else { return nil }

        self.device = device
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
        if let fn = library.makeFunction(name: "compute_charge_forces_bh") {
            bhChargePipeline = try? device.makeComputePipelineState(function: fn)
        }
        fullSimReady = springPipeline != nil && centroidPipeline != nil
            && centroidRepulsionPipeline != nil && cohesionPipeline != nil

        simParamBuffer = device.makeBuffer(length: MemoryLayout<ForceSimParams>.stride, options: .storageModeShared)
        groupTypeBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
        nodeCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
        repulsionBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)
        groupCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
    }

    /// Whether the full GPU simulation pipeline is available.
    public var isFullSimAvailable: Bool { fullSimReady }

    /// Node count from last encodeForces call (for readback sizing).
    private var lastEncodedNodeCount: Int = 0

    /// Encode all force compute passes into an external compute encoder.
    /// Called from MetalGraphRenderer.drawFrame() BEFORE render passes.
    /// Forces accumulate in fullForceBuffer; read back via readBackForces() after GPU completes.
    public func encodeForces(
        encoder: MTLComputeCommandEncoder,
        x: [Float], y: [Float], z: [Float],
        projectGroups: [Int], topicGroups: [Int],
        galaxyGroups: [Int] = [], galaxyCenters: [SIMD3<Float>] = [],
        chargeStrength: Float, crossChargeMultiplier: Float,
        sameTopicChargeScale: Float, sameProjectChargeScale: Float,
        springLength: Float, crossProjectSpringLength: Float, springStrength: Float,
        cohesionStrength: Float, centroidRepulsion: Float,
        topicCohesionStrength: Float, topicCentroidRepulsion: Float,
        centerStrength: Float, center: SIMD3<Float>,
        alpha: Float, damping: Float, maxSpeed: Float
    ) {
        let n = x.count
        guard n > 1, fullSimReady else {
            GPULog.log("ENCODE SKIP n=\(n) ready=\(fullSimReady)")
            return
        }
        GPULog.log("ENCODE START n=\(n) cap=\(fullNodeCapacity) grpCap=\(fullGroupCapacity)")

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
                topicGroup: Int32(i < topicGroups.count ? topicGroups[i] : -1),
                galaxyGroup: Int32(i < galaxyGroups.count ? galaxyGroups[i] : 0),
                _pad: 0)
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
            projectGroupCount: projectCount, topicGroupCount: topicCount,
            galaxyGroupCount: UInt32(max(galaxyCenters.count, 1)))

        let tgSize = 256

        // 1. Charge forces.
        // Swap in async-built octree if available.
        if let newTree = pendingTreeNodes.withLock({ val -> [OctreeNode]? in
            let r = val; val = nil; return r
        }) {
            cachedTreeNodes = newTree
            treeCacheFrameAge = 0
        } else {
            treeCacheFrameAge += 1
        }

        // Trigger async rebuild if tree is stale or dirty.
        if bhChargePipeline != nil, n > 256,
           cachedTreeNodes == nil || bhTreeDirty || treeCacheFrameAge >= treeCacheMaxAge {
            rebuildOctreeAsync(x: x, y: y, z: z, n: n)
            bhTreeDirty = false
        }

        // BH GPU kernel disabled — hangs when concurrent with render pipeline.
        // CPU Barnes-Hut in computeAllForces() handles charge instead.
        if false, n > 256, let bhPL = bhChargePipeline, let treeNodes = cachedTreeNodes {
            let treeCount = treeNodes.count

            if treeCount > bhTreeCapacity {
                let cap = max(treeCount * 2, 512)
                bhTreeBuffer = device.makeBuffer(length: cap * MemoryLayout<BHOctreeNode>.stride, options: .storageModeShared)
                bhTreeCapacity = cap
            }
            if bhParamBuffer == nil {
                bhParamBuffer = device.makeBuffer(length: MemoryLayout<BHChargeParams>.stride, options: .storageModeShared)
            }

            if let treeBuf = bhTreeBuffer, let bhPBuf = bhParamBuffer {
                let treePtr = treeBuf.contents().bindMemory(to: BHOctreeNode.self, capacity: treeCount)
                // Validate tree before packing
                var maxChildIdx: Int32 = 0
                var invalidChildren = 0
                var maxDepthSeen = 0
                for i in 0..<treeCount {
                    let src = treeNodes[i]
                    treePtr[i] = BHOctreeNode(
                        cx: src.cx, cy: src.cy, cz: src.cz, halfSize: src.halfSize,
                        comX: src.comX, comY: src.comY, comZ: src.comZ, mass: src.mass,
                        children: (src.child(0), src.child(1), src.child(2), src.child(3),
                                   src.child(4), src.child(5), src.child(6), src.child(7)),
                        bodyIndex: src.bodyIndex, _pad: 0)
                    for oct in 0..<8 {
                        let c = src.child(oct)
                        if c >= Int32(treeCount) { invalidChildren += 1 }
                        if c > maxChildIdx { maxChildIdx = c }
                    }
                    if src.bodyIndex >= Int32(n) { invalidChildren += 1 }
                }

                // Walk tree to find max depth
                var stack = [(Int, Int)]() // (nodeIdx, depth)
                stack.append((0, 0))
                while let (idx, depth) = stack.popLast() {
                    guard idx >= 0, idx < treeCount else { continue }
                    maxDepthSeen = max(maxDepthSeen, depth)
                    if depth > 50 { break } // safety
                    let node = treeNodes[idx]
                    for oct in 0..<8 {
                        let c = Int(node.child(oct))
                        if c > 0 && c < treeCount { stack.append((c, depth + 1)) }
                    }
                }

                // Check ALL positions for NaN/Inf and compute bounding box
                var nanCount = 0
                var infCount = 0
                var minP = SIMD3<Float>(Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude)
                var maxP = SIMD3<Float>(-Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude)
                let nodeCheck = nodeBuf.contents().bindMemory(to: ForceNodeFull.self, capacity: n)
                for i in 0..<n {
                    let p = nodeCheck[i]
                    if p.px.isNaN || p.py.isNaN || p.pz.isNaN { nanCount += 1 }
                    if p.px.isInfinite || p.py.isInfinite || p.pz.isInfinite { infCount += 1 }
                    minP.x = min(minP.x, p.px); minP.y = min(minP.y, p.py); minP.z = min(minP.z, p.pz)
                    maxP.x = max(maxP.x, p.px); maxP.y = max(maxP.y, p.py); maxP.z = max(maxP.z, p.pz)
                }
                let span = maxP - minP

                GPULog.log("BH charge: n=\(n) tree=\(treeCount) cap=\(bhTreeCapacity) maxChild=\(maxChildIdx) invalid=\(invalidChildren) depth=\(maxDepthSeen) nan=\(nanCount) inf=\(infCount) span=(\(String(format:"%.1f",span.x)),\(String(format:"%.1f",span.y)),\(String(format:"%.1f",span.z)))")

                // cutoffSq must be relative to the graph's span, not hardcoded.
                // With span ~500 and cutoffSq = 250000 (500²), every node is within cutoff
                // of every other → BH degenerates to O(n²) → 1-second GPU hang.
                // Use 30% of max span dimension — far nodes beyond this contribute negligible force.
                let maxSpan = max(span.x, max(span.y, span.z))
                let cutoff = maxSpan * 0.3
                let cutoffSq = cutoff * cutoff

                bhPBuf.contents().bindMemory(to: BHChargeParams.self, capacity: 1).pointee = BHChargeParams(
                    chargeStrength: chargeStrength, crossChargeMultiplier: crossChargeMultiplier,
                    sameTopicChargeScale: sameTopicChargeScale, sameProjectChargeScale: sameProjectChargeScale,
                    cutoffSq: cutoffSq, thetaSq: 0.7 * 0.7,
                    nodeCount: UInt32(n), treeNodeCount: UInt32(treeCount))

                encoder.setComputePipelineState(bhPL)
                encoder.setBuffer(nodeBuf, offset: 0, index: 0)
                encoder.setBuffer(forceBuf, offset: 0, index: 1)
                encoder.setBuffer(treeBuf, offset: 0, index: 2)
                encoder.setBuffer(bhPBuf, offset: 0, index: 3)
                let chargeTg = min(Int(bhPL.maxTotalThreadsPerThreadgroup), tgSize)
                encoder.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: chargeTg, height: 1, depth: 1))
                encoder.memoryBarrier(scope: .buffers)
            }
        } else if n <= 2048 {
            // Brute force O(n²) — only safe for small counts.
            // For larger n, skip charge this frame; CPU fallback handles it
            // until the async BH tree is ready.
            GPULog.log("BRUTE charge: n=\(n)")
            encoder.setComputePipelineState(pipelineState)
            encoder.setBuffer(nodeBuf, offset: 0, index: 0)
            encoder.setBuffer(forceBuf, offset: 0, index: 1)
            if let chargePBuf = paramBuffer { encoder.setBuffer(chargePBuf, offset: 0, index: 2) }
            let chargeTg = min(Int(pipelineState.maxTotalThreadsPerThreadgroup), tgSize)
            encoder.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: chargeTg, height: 1, depth: 1))
            encoder.memoryBarrier(scope: .buffers)
        } else {
            GPULog.log("SKIPPED charge: n=\(n) (waiting for BH tree)")
        }

        // 2. Springs
        GPULog.log("SPRINGS: edges=\(csrEdgeCount) nodeBuf=\(nodeBuf.length) forceBuf=\(forceBuf.length)")
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
        GPULog.log("CENTROIDS: projCount=\(projectCount) topicCount=\(topicCount)")
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
        GPULog.log("ENCODE END n=\(n)")
    }

    /// Read back force results from GPU shared buffer after command buffer completion.
    /// Call this from the command buffer's completion handler or after waitUntilCompleted.
    public func readBackForces() -> (fx: [Float], fy: [Float], fz: [Float])? {
        let n = lastEncodedNodeCount
        guard n > 0, let forceBuf = fullForceBuffer else { return nil }
        let forcePtr = forceBuf.contents().bindMemory(to: SIMD3<Float>.self, capacity: n)
        var fx = [Float](repeating: 0, count: n)
        var fy = [Float](repeating: 0, count: n)
        var fz = [Float](repeating: 0, count: n)
        for i in 0..<n { fx[i] = forcePtr[i].x; fy[i] = forcePtr[i].y; fz[i] = forcePtr[i].z }
        return (fx, fy, fz)
    }

    // dispatchForces() DELETED — forces now encode synchronously in the render CB
    // via FrameOrchestrator. No separate command queue, no completion handler race.

    /// Stash edges for topology rebuild inside dispatchForces
    public private(set) var topologyDirtyForDispatch = true
    private var dispatchEdges: [(Int, Int)] = []

    public func setTopologyDirty(edges: [(Int, Int)]) {
        dispatchEdges = edges
        topologyDirtyForDispatch = true
    }

    /// Rebuild topology buffers (CSR adjacency, group membership) if dirty.
    /// Call before encodeForces when encoding externally (not via dispatchForces).
    public func rebuildTopology(nodeCount: Int, projectGroups: [Int], topicGroups: [Int]) {
        guard topologyDirtyForDispatch else { return }
        rebuildTopologyBuffers(nodeCount: nodeCount, edges: dispatchEdges,
                              projectGroups: projectGroups, topicGroups: topicGroups)
        topologyDirtyForDispatch = false
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

    // Barnes-Hut octree (CPU-built on background thread, GPU-walked)
    private var bhChargePipeline: MTLComputePipelineState?
    private var bhTreeBuffer: MTLBuffer?       // BHOctreeNode[]
    private var bhParamBuffer: MTLBuffer?      // BHChargeParams
    private var bhTreeCapacity: Int = 0
    private var bhTreeNodeCount: Int = 0       // current tree size in buffer
    private var bhTreeDirty: Bool = true       // rebuild tree on next encode

    // Async octree building — tree is built on a background thread to avoid
    // blocking the main thread (buildOctree is O(n log n), ~20-30ms at 10K).
    // The GPU uses the cached tree; brute force is used as fallback until ready.
    private var cachedTreeNodes: [OctreeNode]?
    private let pendingTreeNodes = OSAllocatedUnfairLock<[OctreeNode]?>(initialState: nil)
    private let treeRebuildInFlight = OSAllocatedUnfairLock<Bool>(initialState: false)
    private var treeCacheFrameAge: Int = 0
    private let treeCacheMaxAge: Int = 10      // rebuild every 10 encode-frames

    /// Kick off an async octree build on a background thread.
    /// The result is stored in `pendingTreeNodes` and swapped in on the next encode.
    func rebuildOctreeAsync(x: [Float], y: [Float], z: [Float], n: Int) {
        let alreadyBuilding = treeRebuildInFlight.withLock { val -> Bool in
            if val { return true }
            val = true
            return false
        }
        guard !alreadyBuilding else { return }
        // x/y/z are value types — Swift COW gives the task its own snapshot
        let pending = pendingTreeNodes
        let inFlight = treeRebuildInFlight
        Task.detached(priority: .userInitiated) {
            let tree = buildOctree(x: x, y: y, z: z, n: n)
            pending.withLock { $0 = tree }
            inFlight.withLock { $0 = false }
        }
    }

    // MARK: - Topology Buffer Management

    func rebuildTopologyBuffers(
        nodeCount n: Int,
        edges: [(Int, Int)],
        projectGroups: [Int],
        topicGroups: [Int]
    ) {
        let e = edges.count

        // Edge adjacency CSR: adjOffsets[N+1] + adjNeighbors[2*E]
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

            memset(offsets, 0, (n + 1) * MemoryLayout<UInt32>.stride)
            for (s, t) in edges {
                offsets[s] &+= 1
                offsets[t] &+= 1
            }

            var running: UInt32 = 0
            for i in 0...n {
                let count = offsets[i]
                offsets[i] = running
                running &+= count
            }

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

        // Project group membership CSR
        let projectCount = max((projectGroups.max() ?? 0) + 1, 1)
        if projectCount + 1 > csrProjectGroupCapacity {
            let cap = max((projectCount + 1) * 2, 64)
            projMemberOffsetsBuffer = device.makeBuffer(length: cap * MemoryLayout<UInt32>.stride, options: .storageModeShared)
            csrProjectGroupCapacity = cap
        }
        projMembersBuffer = ensureBuffer(projMembersBuffer, count: max(n, 1), stride: MemoryLayout<UInt32>.stride)

        if let offsetsBuf = projMemberOffsetsBuffer, let membersBuf = projMembersBuffer {
            buildGroupCSR(offsets: offsetsBuf, members: membersBuf,
                          groups: projectGroups, nodeCount: n, groupCount: projectCount)
        }

        // Topic group membership CSR
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

    private func buildGroupCSR(offsets offsetsBuf: MTLBuffer, members membersBuf: MTLBuffer,
                               groups: [Int], nodeCount n: Int, groupCount: Int) {
        let offsets = offsetsBuf.contents().bindMemory(to: UInt32.self, capacity: groupCount + 1)
        let members = membersBuf.contents().bindMemory(to: UInt32.self, capacity: n)

        memset(offsets, 0, (groupCount + 1) * MemoryLayout<UInt32>.stride)
        for i in 0..<n {
            let g = groups[i]
            if g >= 0 && g < groupCount { offsets[g] &+= 1 }
        }

        var running: UInt32 = 0
        for g in 0...groupCount {
            let count = offsets[g]
            offsets[g] = running
            running &+= count
        }

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

    private func ensureBuffer(_ existing: MTLBuffer?, count: Int, stride: Int) -> MTLBuffer? {
        if let buf = existing, buf.length >= count * stride { return buf }
        let cap = max(count * 2, 512)
        return device.makeBuffer(length: cap * stride, options: .storageModeShared)
    }
}
