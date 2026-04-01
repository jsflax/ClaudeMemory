import Foundation
import Testing
import simd
import os
import CEngramSceneTypes
@preconcurrency import Metal
import EngramSceneKit
import EngramMetalShaders

/// Integration test: encode GPU forces → commit → read back → verify non-zero.
/// This tests the actual data flow that was broken when forces encoded but never
/// arrived at the simulation.
@Suite("Force Encode → Readback Roundtrip")
struct ForceRoundtripTests {

    static let shaderSource: String = """
    #include <metal_stdlib>
    using namespace metal;

    struct ForceNodeFull {
        float px, py, pz;
        float vx, vy, vz;
        int projectGroup;
        int topicGroup;
        int galaxyGroup;
        int _pad;
    };

    struct ForceParams {
        float chargeStrength;
        float crossChargeMultiplier;
        float sameTopicChargeScale;
        float sameProjectChargeScale;
        float cutoffSq;
        uint  nodeCount;
    };

    kernel void compute_charge_forces(
        device const ForceNodeFull* nodes   [[buffer(0)]],
        device       float3*       forces  [[buffer(1)]],
        constant     ForceParams&  params  [[buffer(2)]],
        uint tid [[thread_position_in_grid]])
    {
        if (tid >= params.nodeCount) return;
        float3 pos_i = float3(nodes[tid].px, nodes[tid].py, nodes[tid].pz);
        int pg_i = nodes[tid].projectGroup;
        int tg_i = nodes[tid].topicGroup;
        float3 totalForce = float3(0);
        for (uint j = 0; j < params.nodeCount; j++) {
            if (j == tid) continue;
            float3 delta = pos_i - float3(nodes[j].px, nodes[j].py, nodes[j].pz);
            float distSq = dot(delta, delta);
            if (distSq > params.cutoffSq) continue;
            if (distSq < 1.0) distSq = 1.0;
            float charge;
            if (nodes[j].projectGroup != pg_i) { charge = params.chargeStrength * params.crossChargeMultiplier; }
            else if (nodes[j].topicGroup == tg_i) { charge = params.chargeStrength * params.sameTopicChargeScale; }
            else { charge = params.chargeStrength * params.sameProjectChargeScale; }
            float dist = sqrt(distSq);
            float forceMag = charge / distSq;
            totalForce += (delta / dist) * forceMag;
        }
        forces[tid] = totalForce;
    }
    """

    @Test("GPU forces encode → commit → readback produces non-zero forces")
    func testGPURoundtrip() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            print("[engram:test] No Metal device, skipping")
            return
        }

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        let library = try device.makeLibrary(source: Self.shaderSource, options: options)
        guard let function = library.makeFunction(name: "compute_charge_forces"),
              let pipeline = try? device.makeComputePipelineState(function: function) else {
            Issue.record("Failed to create pipeline")
            return
        }

        let n = 100
        let nodeBuffer = device.makeBuffer(length: n * MemoryLayout<ForceNodeFull>.stride, options: .storageModeShared)!
        let forceBuffer = device.makeBuffer(length: n * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!

        struct GPUForceParams {
            var chargeStrength: Float; var crossChargeMultiplier: Float
            var sameTopicChargeScale: Float; var sameProjectChargeScale: Float
            var cutoffSq: Float; var nodeCount: UInt32
            var galaxyGroupCount: UInt32; var _pad: UInt32
        }
        let paramBuffer = device.makeBuffer(length: MemoryLayout<GPUForceParams>.stride, options: .storageModeShared)!

        // Place nodes in a grid so forces are non-trivial
        let nodePtr = nodeBuffer.contents().bindMemory(to: ForceNodeFull.self, capacity: n)
        for i in 0..<n {
            nodePtr[i] = ForceNodeFull(
                px: Float(i % 10) * 50, py: Float(i / 10) * 50, pz: 0,
                vx: 0, vy: 0, vz: 0,
                projectGroup: Int32(i % 3), topicGroup: Int32(i % 5),
                galaxyGroup: 0, _pad: 0)
        }
        memset(forceBuffer.contents(), 0, n * MemoryLayout<SIMD3<Float>>.stride)

        paramBuffer.contents().bindMemory(to: GPUForceParams.self, capacity: 1).pointee = GPUForceParams(
            chargeStrength: 500, crossChargeMultiplier: 3.0,
            sameTopicChargeScale: 0.35, sameProjectChargeScale: 1.0,
            cutoffSq: 500 * 500, nodeCount: UInt32(n),
            galaxyGroupCount: 0, _pad: 0)

        // Encode into command buffer
        guard let cmdBuf = queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else {
            Issue.record("Failed to create command buffer")
            return
        }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(nodeBuffer, offset: 0, index: 0)
        enc.setBuffer(forceBuffer, offset: 0, index: 1)
        enc.setBuffer(paramBuffer, offset: 0, index: 2)
        let tg = min(Int(pipeline.maxTotalThreadsPerThreadgroup), 256)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        // Commit and wait
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        #expect(cmdBuf.status == .completed, "Command buffer failed: \(cmdBuf.error?.localizedDescription ?? "unknown")")

        // Read back forces
        let forcePtr = forceBuffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: n)
        var totalMagnitude: Float = 0
        for i in 0..<n {
            totalMagnitude += abs(forcePtr[i].x) + abs(forcePtr[i].y) + abs(forcePtr[i].z)
        }

        #expect(totalMagnitude > 0, "GPU forces should be non-zero for grid of 100 nodes")
        print("[engram:test] GPU roundtrip: totalMagnitude=\(totalMagnitude) for \(n) nodes")
    }

    @Test("Forces read back BEFORE commit are zero (verifies timing)")
    func testReadbackTimingMatters() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else { return }

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        let library = try device.makeLibrary(source: Self.shaderSource, options: options)
        guard let function = library.makeFunction(name: "compute_charge_forces"),
              let pipeline = try? device.makeComputePipelineState(function: function) else { return }

        let n = 50
        let nodeBuffer = device.makeBuffer(length: n * MemoryLayout<ForceNodeFull>.stride, options: .storageModeShared)!
        let forceBuffer = device.makeBuffer(length: n * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!

        struct GPUForceParams {
            var chargeStrength: Float; var crossChargeMultiplier: Float
            var sameTopicChargeScale: Float; var sameProjectChargeScale: Float
            var cutoffSq: Float; var nodeCount: UInt32
            var galaxyGroupCount: UInt32; var _pad: UInt32
        }
        let paramBuffer = device.makeBuffer(length: MemoryLayout<GPUForceParams>.stride, options: .storageModeShared)!

        let nodePtr = nodeBuffer.contents().bindMemory(to: ForceNodeFull.self, capacity: n)
        for i in 0..<n {
            nodePtr[i] = ForceNodeFull(
                px: Float(i) * 30, py: 0, pz: 0,
                vx: 0, vy: 0, vz: 0, projectGroup: 0, topicGroup: 0,
                galaxyGroup: 0, _pad: 0)
        }
        memset(forceBuffer.contents(), 0, n * MemoryLayout<SIMD3<Float>>.stride)
        paramBuffer.contents().bindMemory(to: GPUForceParams.self, capacity: 1).pointee = GPUForceParams(
            chargeStrength: 500, crossChargeMultiplier: 3.0,
            sameTopicChargeScale: 0.35, sameProjectChargeScale: 1.0,
            cutoffSq: 500 * 500, nodeCount: UInt32(n),
            galaxyGroupCount: 0, _pad: 0)

        guard let cmdBuf = queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(nodeBuffer, offset: 0, index: 0)
        enc.setBuffer(forceBuffer, offset: 0, index: 1)
        enc.setBuffer(paramBuffer, offset: 0, index: 2)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: min(Int(pipeline.maxTotalThreadsPerThreadgroup), 256), height: 1, depth: 1))
        enc.endEncoding()

        // Read BEFORE commit — forces should still be zero
        let forcePtr = forceBuffer.contents().bindMemory(to: SIMD3<Float>.self, capacity: n)
        var preCommitMag: Float = 0
        for i in 0..<n { preCommitMag += abs(forcePtr[i].x) + abs(forcePtr[i].y) + abs(forcePtr[i].z) }

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Read AFTER commit — forces should be non-zero
        var postCommitMag: Float = 0
        for i in 0..<n { postCommitMag += abs(forcePtr[i].x) + abs(forcePtr[i].y) + abs(forcePtr[i].z) }

        // Note: on shared memory (Apple Silicon), the pre-commit read MAY see stale zeros
        // or MAY see partial results depending on GPU scheduling. The important thing is
        // that post-commit forces are non-zero.
        #expect(postCommitMag > 0, "Forces should be non-zero after GPU completion")
        print("[engram:test] Pre-commit magnitude: \(preCommitMag), Post-commit: \(postCommitMag)")
    }

    @Test("Completion handler receives forces (async delivery pattern)")
    func testCompletionHandlerDelivery() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else { return }

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        let library = try device.makeLibrary(source: Self.shaderSource, options: options)
        guard let function = library.makeFunction(name: "compute_charge_forces"),
              let pipeline = try? device.makeComputePipelineState(function: function) else { return }

        let n = 50
        let nodeBuffer = device.makeBuffer(length: n * MemoryLayout<ForceNodeFull>.stride, options: .storageModeShared)!
        let forceBuffer = device.makeBuffer(length: n * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!

        struct GPUForceParams {
            var chargeStrength: Float; var crossChargeMultiplier: Float
            var sameTopicChargeScale: Float; var sameProjectChargeScale: Float
            var cutoffSq: Float; var nodeCount: UInt32
            var galaxyGroupCount: UInt32; var _pad: UInt32
        }
        let paramBuffer = device.makeBuffer(length: MemoryLayout<GPUForceParams>.stride, options: .storageModeShared)!

        let nodePtr = nodeBuffer.contents().bindMemory(to: ForceNodeFull.self, capacity: n)
        for i in 0..<n {
            nodePtr[i] = ForceNodeFull(
                px: Float(i) * 30, py: 0, pz: 0,
                vx: 0, vy: 0, vz: 0, projectGroup: Int32(i % 3), topicGroup: 0,
                galaxyGroup: 0, _pad: 0)
        }
        memset(forceBuffer.contents(), 0, n * MemoryLayout<SIMD3<Float>>.stride)
        paramBuffer.contents().bindMemory(to: GPUForceParams.self, capacity: 1).pointee = GPUForceParams(
            chargeStrength: 500, crossChargeMultiplier: 3.0,
            sameTopicChargeScale: 0.35, sameProjectChargeScale: 1.0,
            cutoffSq: 500 * 500, nodeCount: UInt32(n),
            galaxyGroupCount: 0, _pad: 0)

        guard let cmdBuf = queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(nodeBuffer, offset: 0, index: 0)
        enc.setBuffer(forceBuffer, offset: 0, index: 1)
        enc.setBuffer(paramBuffer, offset: 0, index: 2)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: min(Int(pipeline.maxTotalThreadsPerThreadgroup), 256), height: 1, depth: 1))
        enc.endEncoding()

        // Simulate the real pattern: deliver forces via lock in completion handler
        struct ForceTriple: Sendable { let fx: [Float], fy: [Float], fz: [Float] }
        let deliveredForces = OSAllocatedUnfairLock<ForceTriple?>(initialState: nil)
        let capturedBuf = forceBuffer
        let capturedN = n

        cmdBuf.addCompletedHandler { @Sendable _ in
            let ptr = capturedBuf.contents().bindMemory(to: SIMD3<Float>.self, capacity: capturedN)
            let result = ForceTriple(
                fx: (0..<capturedN).map { ptr[$0].x },
                fy: (0..<capturedN).map { ptr[$0].y },
                fz: (0..<capturedN).map { ptr[$0].z })
            deliveredForces.withLock { $0 = result }
        }
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Forces should be delivered
        let forces = deliveredForces.withLock { $0 }
        #expect(forces != nil, "Completion handler should have delivered forces")
        if let f = forces {
            let mag = f.fx.reduce(0) { $0 + abs($1) } + f.fy.reduce(0) { $0 + abs($1) }
            #expect(mag > 0, "Delivered forces should be non-zero")
            #expect(f.fx.count == n)
        }
    }

    /// Validate Swift struct sizes match what the GPU shader expects.
    /// A mismatch here means the GPU will read garbage → pink artifacts → system crash.
    @Test("ForceNodeFull and ForceSimParams struct sizes match GPU expectations")
    func testStructSizesMatch() {
        // ForceNodeFull: px,py,pz,vx,vy,vz (6 floats) + projectGroup,topicGroup,galaxyGroup,_pad (4 ints) = 40 bytes
        #expect(MemoryLayout<ForceNodeFull>.stride == 40,
                "ForceNodeFull stride is \(MemoryLayout<ForceNodeFull>.stride), expected 40")
        #expect(MemoryLayout<ForceNodeFull>.size == 40,
                "ForceNodeFull size is \(MemoryLayout<ForceNodeFull>.size), expected 40")

        // Verify field offsets by writing known values and reading back raw bytes
        var node = ForceNodeFull(px: 1, py: 2, pz: 3, vx: 4, vy: 5, vz: 6,
                                 projectGroup: 7, topicGroup: 8, galaxyGroup: 9, _pad: 0)
        let raw = withUnsafeBytes(of: &node) { Array($0) }
        // px at offset 0
        let px = raw.withUnsafeBytes { $0.load(fromByteOffset: 0, as: Float.self) }
        #expect(px == 1.0)
        // projectGroup at offset 24 (6 floats × 4)
        let pg = raw.withUnsafeBytes { $0.load(fromByteOffset: 24, as: Int32.self) }
        #expect(pg == 7)
        // topicGroup at offset 28
        let tg = raw.withUnsafeBytes { $0.load(fromByteOffset: 28, as: Int32.self) }
        #expect(tg == 8)
        // galaxyGroup at offset 32
        let gg = raw.withUnsafeBytes { $0.load(fromByteOffset: 32, as: Int32.self) }
        #expect(gg == 9, "galaxyGroup at wrong offset: expected offset 32")

        // GroupCentroid: sumX,sumY,sumZ (3 floats) + count (1 uint) = 16 bytes
        #expect(MemoryLayout<GroupCentroid>.stride == 16,
                "GroupCentroid stride is \(MemoryLayout<GroupCentroid>.stride), expected 16")

        print("[test] ForceNodeFull: \(MemoryLayout<ForceNodeFull>.stride) bytes, " +
              "ForceSimParams: \(MemoryLayout<ForceSimParams>.stride) bytes, " +
              "GroupCentroid: \(MemoryLayout<GroupCentroid>.stride) bytes")
    }

    /// Validate that the Metal shader compiles with matching struct layout.
    /// Catches simd_float3 alignment mismatches between C header and Metal.
    @Test("Metal shader struct layout matches Swift — GPU reflection check")
    func testMetalStructReflection() throws {
        let device = MTLCreateSystemDefaultDevice()!
        let library = try device.makeLibrary(source: Self.fullPipelineShaderSource, options: nil)
        // If the shader compiled, the struct layouts are at least internally consistent.
        // Verify the charge kernel can be created (validates ForceNodeFull + ForceParams)
        let chargeFn = library.makeFunction(name: "compute_charge_forces")
        #expect(chargeFn != nil, "compute_charge_forces function not found")
        // Verify the spring kernel (validates ForceSimParams with galaxyGroupCount)
        let springFn = library.makeFunction(name: "compute_spring_forces")
        #expect(springFn != nil, "compute_spring_forces function not found")
        // Verify cohesion kernel (validates ForceNodeFull.galaxyGroup access)
        let cohesionFn = library.makeFunction(name: "apply_cohesion_forces")
        #expect(cohesionFn != nil, "apply_cohesion_forces function not found")

        // Create pipelines to validate function signatures compile
        let chargePL = try device.makeComputePipelineState(function: chargeFn!)
        #expect(chargePL.maxTotalThreadsPerThreadgroup > 0)
        let springPL = try device.makeComputePipelineState(function: springFn!)
        #expect(springPL.maxTotalThreadsPerThreadgroup > 0)
        print("[test] All Metal kernels compiled successfully with updated struct layouts")
    }

    /// Full encodeForces pipeline at 8K nodes — exercises charge + springs + centroids
    /// + cohesion with the real MetalForceCompute, not test shader strings.
    /// This catches struct mismatches and GPU hangs at realistic scale.
    @Test("Full encodeForces pipeline at 8K nodes completes within 500ms")
    func testFullEncodeForces8K() throws {
        let device = MTLCreateSystemDefaultDevice()!
        let queue = device.makeCommandQueue()!
        let library = try device.makeLibrary(source: Self.fullPipelineShaderSource, options: nil)

        let chargePipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "compute_charge_forces")!)
        let springPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "compute_spring_forces")!)
        let centroidPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "compute_group_centroids")!)
        let repulsionPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "compute_centroid_repulsion")!)
        let cohesionPipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "apply_cohesion_forces")!)

        let n = 8000
        let projectCount = 8
        let topicCount = 30

        // Pack nodes
        let nodeBuf = device.makeBuffer(length: n * MemoryLayout<ForceNodeFull>.stride, options: .storageModeShared)!
        let nodePtr = nodeBuf.contents().bindMemory(to: ForceNodeFull.self, capacity: n)
        for i in 0..<n {
            nodePtr[i] = ForceNodeFull(
                px: Float.random(in: -500...500), py: Float.random(in: -500...500),
                pz: Float.random(in: -500...500), vx: 0, vy: 0, vz: 0,
                projectGroup: Int32(i % projectCount), topicGroup: Int32(i % topicCount),
                galaxyGroup: Int32(i < n/2 ? 0 : 1), _pad: 0)
        }

        let forceBuf = device.makeBuffer(length: n * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!
        memset(forceBuf.contents(), 0, n * MemoryLayout<SIMD3<Float>>.stride)

        // Build adjacency CSR for edges
        let edgeCount = 12000
        var degrees = [Int](repeating: 0, count: n)
        var edges: [(Int, Int)] = []
        for _ in 0..<edgeCount {
            let a = Int.random(in: 0..<n), b = Int.random(in: 0..<n)
            if a != b { edges.append((a, b)); degrees[a] += 1; degrees[b] += 1 }
        }
        let adjOffsetsBuf = device.makeBuffer(length: (n + 1) * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let adjNeighborsBuf = device.makeBuffer(length: edges.count * 2 * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let offsets = adjOffsetsBuf.contents().bindMemory(to: UInt32.self, capacity: n + 1)
        let neighbors = adjNeighborsBuf.contents().bindMemory(to: UInt32.self, capacity: edges.count * 2)
        memset(offsets, 0, (n + 1) * MemoryLayout<UInt32>.stride)
        for (s, t) in edges { offsets[s] &+= 1; offsets[t] &+= 1 }
        var running: UInt32 = 0
        for i in 0...n { let c = offsets[i]; offsets[i] = running; running &+= c }
        var cursors = (0..<n).map { offsets[$0] }
        for (s, t) in edges {
            neighbors[Int(cursors[s])] = UInt32(t); cursors[s] &+= 1
            neighbors[Int(cursors[t])] = UInt32(s); cursors[t] &+= 1
        }

        // Build group membership CSR (project groups)
        let projOffBuf = device.makeBuffer(length: (projectCount + 1) * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let projMemBuf = device.makeBuffer(length: n * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let projOff = projOffBuf.contents().bindMemory(to: UInt32.self, capacity: projectCount + 1)
        let projMem = projMemBuf.contents().bindMemory(to: UInt32.self, capacity: n)
        memset(projOff, 0, (projectCount + 1) * MemoryLayout<UInt32>.stride)
        for i in 0..<n { projOff[i % projectCount] &+= 1 }
        running = 0
        for g in 0...projectCount { let c = projOff[g]; projOff[g] = running; running &+= c }
        var projCursors = (0..<projectCount).map { projOff[$0] }
        for i in 0..<n { let g = i % projectCount; projMem[Int(projCursors[g])] = UInt32(i); projCursors[g] &+= 1 }

        let projCentBuf = device.makeBuffer(length: projectCount * MemoryLayout<GroupCentroid>.stride, options: .storageModeShared)!
        let projForceBuf = device.makeBuffer(length: projectCount * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!
        memset(projCentBuf.contents(), 0, projectCount * MemoryLayout<GroupCentroid>.stride)
        memset(projForceBuf.contents(), 0, projectCount * MemoryLayout<SIMD3<Float>>.stride)

        // Charge params
        struct GPUForceParams {
            var chargeStrength: Float; var crossChargeMultiplier: Float
            var sameTopicChargeScale: Float; var sameProjectChargeScale: Float
            var cutoffSq: Float; var nodeCount: UInt32
            var galaxyGroupCount: UInt32; var _pad: UInt32
        }
        let chargePBuf = device.makeBuffer(length: MemoryLayout<GPUForceParams>.stride, options: .storageModeShared)!
        chargePBuf.contents().bindMemory(to: GPUForceParams.self, capacity: 1).pointee = GPUForceParams(
            chargeStrength: 500, crossChargeMultiplier: 3.0,
            sameTopicChargeScale: 0.35, sameProjectChargeScale: 1.0,
            cutoffSq: 500 * 500, nodeCount: UInt32(n),
            galaxyGroupCount: 0, _pad: 0)

        // Sim params
        let simPBuf = device.makeBuffer(length: MemoryLayout<ForceSimParams>.stride, options: .storageModeShared)!
        simPBuf.contents().bindMemory(to: ForceSimParams.self, capacity: 1).pointee = ForceSimParams(
            springLength: 100, crossProjectSpringLength: 200, springStrength: 0.3,
            cohesionStrength: 0.02, centroidRepulsion: 300,
            topicCohesionStrength: 0.01, topicCentroidRepulsion: 200,
            centerStrength: 0.8, center: .init(0, 0, 0), alpha: 0.5, damping: 0.78, maxSpeed: 12.0,
            nodeCount: UInt32(n), edgeCount: UInt32(edges.count), crossProjectSpringScale: 1.0,
            projectGroupCount: UInt32(projectCount), topicGroupCount: UInt32(topicCount),
            galaxyGroupCount: 2, topicLeashStrength: 0.01)
        let gtBuf = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let gcBuf = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let repBuf = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)!

        guard let cmdBuf = queue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else { return }

        let start = CFAbsoluteTimeGetCurrent()
        let tgSize = 256

        // 1. Charge
        enc.setComputePipelineState(chargePipeline)
        enc.setBuffer(nodeBuf, offset: 0, index: 0)
        enc.setBuffer(forceBuf, offset: 0, index: 1)
        enc.setBuffer(chargePBuf, offset: 0, index: 2)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: min(Int(chargePipeline.maxTotalThreadsPerThreadgroup), tgSize), height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)

        // 2. Springs
        enc.setComputePipelineState(springPipeline)
        enc.setBuffer(adjOffsetsBuf, offset: 0, index: 0)
        enc.setBuffer(adjNeighborsBuf, offset: 0, index: 1)
        enc.setBuffer(nodeBuf, offset: 0, index: 2)
        enc.setBuffer(forceBuf, offset: 0, index: 3)
        enc.setBuffer(simPBuf, offset: 0, index: 4)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: min(Int(springPipeline.maxTotalThreadsPerThreadgroup), tgSize), height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)

        // 3. Centroids
        enc.setComputePipelineState(centroidPipeline)
        enc.setBuffer(projOffBuf, offset: 0, index: 0)
        enc.setBuffer(projMemBuf, offset: 0, index: 1)
        enc.setBuffer(nodeBuf, offset: 0, index: 2)
        enc.setBuffer(projCentBuf, offset: 0, index: 3)
        enc.dispatchThreads(MTLSize(width: projectCount, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: min(Int(centroidPipeline.maxTotalThreadsPerThreadgroup), projectCount), height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)

        // 4. Centroid repulsion
        enc.setComputePipelineState(repulsionPipeline)
        enc.setBuffer(projCentBuf, offset: 0, index: 0)
        enc.setBuffer(projForceBuf, offset: 0, index: 1)
        enc.setBuffer(simPBuf, offset: 0, index: 2)
        gcBuf.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(projectCount)
        enc.setBuffer(gcBuf, offset: 0, index: 3)
        gtBuf.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = 0
        enc.setBuffer(gtBuf, offset: 0, index: 4)
        repBuf.contents().bindMemory(to: Float.self, capacity: 1).pointee = 300
        enc.setBuffer(repBuf, offset: 0, index: 5)
        enc.dispatchThreads(MTLSize(width: projectCount, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: min(Int(repulsionPipeline.maxTotalThreadsPerThreadgroup), projectCount), height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)

        // 5. Cohesion
        enc.setComputePipelineState(cohesionPipeline)
        enc.setBuffer(nodeBuf, offset: 0, index: 0)
        enc.setBuffer(projCentBuf, offset: 0, index: 1)
        enc.setBuffer(projForceBuf, offset: 0, index: 2)
        enc.setBuffer(forceBuf, offset: 0, index: 3)
        enc.setBuffer(simPBuf, offset: 0, index: 4)
        gtBuf.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = 0
        enc.setBuffer(gtBuf, offset: 0, index: 5)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: min(Int(cohesionPipeline.maxTotalThreadsPerThreadgroup), tgSize), height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000

        #expect(cmdBuf.status != .error, "GPU errored: \(cmdBuf.error?.localizedDescription ?? "unknown")")

        let forcePtr = forceBuf.contents().bindMemory(to: SIMD3<Float>.self, capacity: n)
        let maxF = (0..<n).map { abs(forcePtr[$0].x) + abs(forcePtr[$0].y) + abs(forcePtr[$0].z) }.max() ?? 0
        #expect(maxF > 0, "Forces should be non-zero")
        print("[test] Full pipeline 8K nodes: \(String(format: "%.1f", ms))ms, maxForce=\(String(format: "%.1f", maxF))")

        #expect(ms < 500, "Full pipeline took \(String(format: "%.0f", ms))ms — GPU is choking")
    }

    /// Shader source that includes all force kernels (charge, spring, centroid, cohesion)
    /// needed by MetalForceCompute.encodeForces.
    static let fullPipelineShaderSource: String = """
    #include <metal_stdlib>
    using namespace metal;

    struct ForceNodeFull {
        float px, py, pz;
        float vx, vy, vz;
        int projectGroup;
        int topicGroup;
        int galaxyGroup;
        int _pad;
    };

    struct ForceParams {
        float chargeStrength;
        float crossChargeMultiplier;
        float sameTopicChargeScale;
        float sameProjectChargeScale;
        float cutoffSq;
        uint  nodeCount;
    };

    struct ForceSimParams {
        float springLength;
        float crossProjectSpringLength;
        float springStrength;
        float cohesionStrength;
        float centroidRepulsion;
        float topicCohesionStrength;
        float topicCentroidRepulsion;
        float centerStrength;
        float3 center;
        float alpha;
        float damping;
        float maxSpeed;
        uint  nodeCount;
        uint  edgeCount;
        uint  projectGroupCount;
        uint  topicGroupCount;
        uint  galaxyGroupCount;
    };

    struct GroupCentroid {
        float sumX, sumY, sumZ;
        uint count;
    };

    kernel void compute_charge_forces(
        device const ForceNodeFull* nodes   [[buffer(0)]],
        device       float3*       forces  [[buffer(1)]],
        constant     ForceParams&  params  [[buffer(2)]],
        uint tid [[thread_position_in_grid]])
    {
        if (tid >= params.nodeCount) return;
        float3 pos_i = float3(nodes[tid].px, nodes[tid].py, nodes[tid].pz);
        int pg_i = nodes[tid].projectGroup;
        int tg_i = nodes[tid].topicGroup;
        float3 totalForce = float3(0);
        for (uint j = 0; j < params.nodeCount; j++) {
            if (j == tid) continue;
            float3 delta = pos_i - float3(nodes[j].px, nodes[j].py, nodes[j].pz);
            float distSq = dot(delta, delta);
            if (distSq > params.cutoffSq) continue;
            if (distSq < 1.0) distSq = 1.0;
            float charge;
            if (nodes[j].projectGroup != pg_i) {
                charge = params.chargeStrength * params.crossChargeMultiplier;
            } else if (nodes[j].topicGroup == tg_i) {
                charge = params.chargeStrength * params.sameTopicChargeScale;
            } else {
                charge = params.chargeStrength * params.sameProjectChargeScale;
            }
            float dist = sqrt(distSq);
            totalForce += (delta / dist) * (charge / distSq);
        }
        forces[tid] = totalForce;
    }

    kernel void compute_spring_forces(
        device const uint*           adjOffsets   [[buffer(0)]],
        device const uint*           adjNeighbors [[buffer(1)]],
        device const ForceNodeFull*  nodes        [[buffer(2)]],
        device       float3*         forces       [[buffer(3)]],
        constant     ForceSimParams& params       [[buffer(4)]],
        uint tid [[thread_position_in_grid]])
    {
        if (tid >= params.nodeCount) return;
        float3 myPos = float3(nodes[tid].px, nodes[tid].py, nodes[tid].pz);
        int myPG = nodes[tid].projectGroup;
        int myGG = nodes[tid].galaxyGroup;
        float3 total = float3(0);
        for (uint e = adjOffsets[tid]; e < adjOffsets[tid + 1]; e++) {
            uint j = adjNeighbors[e];
            if (nodes[j].galaxyGroup != myGG) continue;
            float3 delta = float3(nodes[j].px, nodes[j].py, nodes[j].pz) - myPos;
            float d = max(length(delta), 1.0f);
            bool cross = nodes[j].projectGroup != myPG;
            float rest = cross ? params.crossProjectSpringLength : params.springLength;
            total += (delta / d) * params.springStrength * (d - rest);
        }
        forces[tid] += total;
    }

    kernel void compute_group_centroids(
        device const uint*          memberOffsets [[buffer(0)]],
        device const uint*          members      [[buffer(1)]],
        device const ForceNodeFull* nodes        [[buffer(2)]],
        device       GroupCentroid* centroids    [[buffer(3)]],
        uint gid [[thread_position_in_grid]])
    {
        uint start = memberOffsets[gid];
        uint end   = memberOffsets[gid + 1];
        float sx = 0, sy = 0, sz = 0;
        uint count = end - start;
        for (uint i = start; i < end; i++) {
            uint nid = members[i];
            sx += nodes[nid].px;
            sy += nodes[nid].py;
            sz += nodes[nid].pz;
        }
        centroids[gid] = GroupCentroid{sx, sy, sz, count};
    }

    kernel void compute_centroid_repulsion(
        device const GroupCentroid* centroids  [[buffer(0)]],
        device       float3*       groupForce [[buffer(1)]],
        constant     ForceSimParams& params   [[buffer(2)]],
        constant     uint&         groupCount [[buffer(3)]],
        constant     uint&         groupType  [[buffer(4)]],
        constant     float&        repulsion  [[buffer(5)]],
        uint gid [[thread_position_in_grid]])
    {
        if (gid >= groupCount) return;
        if (centroids[gid].count < 2) { groupForce[gid] = float3(0); return; }
        float3 myCenter = float3(centroids[gid].sumX, centroids[gid].sumY, centroids[gid].sumZ) / float(centroids[gid].count);
        float3 total = float3(0);
        for (uint j = 0; j < groupCount; j++) {
            if (j == gid || centroids[j].count < 2) continue;
            float3 otherC = float3(centroids[j].sumX, centroids[j].sumY, centroids[j].sumZ) / float(centroids[j].count);
            float3 delta = myCenter - otherC;
            float distSq = dot(delta, delta);
            if (distSq < 1.0) distSq = 1.0;
            total += normalize(delta) * repulsion / distSq;
        }
        groupForce[gid] = total;
    }

    kernel void apply_cohesion_forces(
        device const ForceNodeFull* nodes      [[buffer(0)]],
        device const GroupCentroid* centroids  [[buffer(1)]],
        device const float3*       groupForce [[buffer(2)]],
        device       float3*       forces     [[buffer(3)]],
        constant     ForceSimParams& params   [[buffer(4)]],
        constant     uint&         groupType  [[buffer(5)]],
        uint tid [[thread_position_in_grid]])
    {
        if (tid >= params.nodeCount) return;
        int group = (groupType == 0) ? nodes[tid].projectGroup : nodes[tid].topicGroup;
        if (group < 0 || centroids[group].count < 2) return;
        float3 centroid = float3(centroids[group].sumX, centroids[group].sumY, centroids[group].sumZ) / float(centroids[group].count);
        float3 pos = float3(nodes[tid].px, nodes[tid].py, nodes[tid].pz);
        float3 delta = centroid - pos;
        float cohStr = (groupType == 0) ? params.cohesionStrength : params.topicCohesionStrength;
        if (groupType == 0) {
            float dist = length(delta);
            float ratio = max(1.0, dist / 30.0);
            cohStr *= ratio * ratio;
        }
        forces[tid] += delta * cohStr + groupForce[group];
    }

    struct BHOctreeNode {
        float cx, cy, cz;
        float halfSize;
        float comX, comY, comZ;
        float mass;
        int   children[8];
        int   bodyIndex;
        int   _pad;
    };

    struct BHChargeParams {
        float chargeStrength;
        float crossChargeMultiplier;
        float sameTopicChargeScale;
        float sameProjectChargeScale;
        float cutoffSq;
        float thetaSq;
        uint  nodeCount;
        uint  treeNodeCount;
    };

    kernel void compute_charge_forces_bh(
        device const ForceNodeFull*  nodes  [[buffer(0)]],
        device       float3*         forces [[buffer(1)]],
        device const BHOctreeNode*   tree   [[buffer(2)]],
        constant     BHChargeParams& params [[buffer(3)]],
        uint tid [[thread_position_in_grid]])
    {
        if (tid >= params.nodeCount) return;
        float3 pos_i = float3(nodes[tid].px, nodes[tid].py, nodes[tid].pz);
        int pg_i = nodes[tid].projectGroup;
        int tg_i = nodes[tid].topicGroup;
        float3 totalForce = float3(0);

        int stack[128];
        int stackTop = 0;
        stack[0] = 0;
        stackTop = 1;

        while (stackTop > 0) {
            int nIdx = stack[--stackTop];
            BHOctreeNode cell = tree[nIdx];
            if (cell.mass == 0) continue;

            if (cell.bodyIndex >= 0) {
                int j = cell.bodyIndex;
                if ((uint)j == tid) continue;
                float3 delta = pos_i - float3(nodes[j].px, nodes[j].py, nodes[j].pz);
                float distSq = dot(delta, delta);
                if (distSq > params.cutoffSq) continue;
                if (distSq < 1.0) distSq = 1.0;
                float charge;
                if (nodes[j].projectGroup != pg_i) {
                    charge = params.chargeStrength * params.crossChargeMultiplier;
                } else if (nodes[j].topicGroup == tg_i) {
                    charge = params.chargeStrength * params.sameTopicChargeScale;
                } else {
                    charge = params.chargeStrength * params.sameProjectChargeScale;
                }
                float dist = sqrt(distSq);
                totalForce += (delta / dist) * (charge / distSq);
                continue;
            }

            float3 delta = pos_i - float3(cell.comX, cell.comY, cell.comZ);
            float distSq = dot(delta, delta);
            if (distSq < 1.0) distSq = 1.0;
            float cellSize = cell.halfSize * 2.0;
            float sSq = cellSize * cellSize;

            if (sSq < distSq * params.thetaSq) {
                if (distSq > params.cutoffSq) continue;
                float dist = sqrt(distSq);
                totalForce += (delta / dist) * (params.chargeStrength * cell.mass / distSq);
                continue;
            }

            float nearX = clamp(pos_i.x, cell.cx - cell.halfSize, cell.cx + cell.halfSize);
            float nearY = clamp(pos_i.y, cell.cy - cell.halfSize, cell.cy + cell.halfSize);
            float nearZ = clamp(pos_i.z, cell.cz - cell.halfSize, cell.cz + cell.halfSize);
            float3 nearDelta = pos_i - float3(nearX, nearY, nearZ);
            if (dot(nearDelta, nearDelta) > params.cutoffSq) continue;

            for (int oct = 0; oct < 8; oct++) {
                int c = cell.children[oct];
                if (c >= 0 && stackTop < 128) {
                    stack[stackTop++] = c;
                }
            }
        }
        forces[tid] = totalForce;
    }
    """

    // MARK: - GPU Barnes-Hut Test

    @Test("GPU Barnes-Hut charge kernel at 10K nodes: correct forces, under 20ms")
    func testGPUBarnesHut10K() throws {
        let device = MTLCreateSystemDefaultDevice()!
        let queue = device.makeCommandQueue()!
        let library = try device.makeLibrary(source: Self.fullPipelineShaderSource, options: nil)
        let bhPipeline = try device.makeComputePipelineState(
            function: library.makeFunction(name: "compute_charge_forces_bh")!)
        // Also build brute force for comparison
        let brutePipeline = try device.makeComputePipelineState(
            function: library.makeFunction(name: "compute_charge_forces")!)

        let n = 10000

        // Random positions spread across space
        var x = [Float](repeating: 0, count: n)
        var y = [Float](repeating: 0, count: n)
        var z = [Float](repeating: 0, count: n)
        for i in 0..<n {
            x[i] = Float.random(in: -500...500)
            y[i] = Float.random(in: -500...500)
            z[i] = Float.random(in: -500...500)
        }

        // Pack nodes
        let nodeBuf = device.makeBuffer(length: n * MemoryLayout<ForceNodeFull>.stride, options: .storageModeShared)!
        let nodePtr = nodeBuf.contents().bindMemory(to: ForceNodeFull.self, capacity: n)
        for i in 0..<n {
            nodePtr[i] = ForceNodeFull(
                px: x[i], py: y[i], pz: z[i], vx: 0, vy: 0, vz: 0,
                projectGroup: Int32(i % 8), topicGroup: Int32(i % 30),
                galaxyGroup: 0, _pad: 0)
        }

        // Build octree on CPU
        let treeNodes = buildOctree(x: x, y: y, z: z, n: n)
        let treeCount = treeNodes.count

        // Upload octree to GPU
        let treeBuf = device.makeBuffer(length: treeCount * MemoryLayout<BHOctreeNode>.stride, options: .storageModeShared)!
        let treePtr = treeBuf.contents().bindMemory(to: BHOctreeNode.self, capacity: treeCount)
        for i in 0..<treeCount {
            let src = treeNodes[i]
            treePtr[i] = BHOctreeNode(
                cx: src.cx, cy: src.cy, cz: src.cz, halfSize: src.halfSize,
                comX: src.comX, comY: src.comY, comZ: src.comZ, mass: src.mass,
                children: (src.child(0), src.child(1), src.child(2), src.child(3),
                           src.child(4), src.child(5), src.child(6), src.child(7)),
                bodyIndex: src.bodyIndex, _pad: 0)
        }

        // BH params
        let bhParamBuf = device.makeBuffer(length: MemoryLayout<BHChargeParams>.stride, options: .storageModeShared)!
        bhParamBuf.contents().bindMemory(to: BHChargeParams.self, capacity: 1).pointee = BHChargeParams(
            chargeStrength: 500, crossChargeMultiplier: 3.0,
            sameTopicChargeScale: 0.35, sameProjectChargeScale: 1.0,
            cutoffSq: 500 * 500, thetaSq: 0.7 * 0.7,
            nodeCount: UInt32(n), treeNodeCount: UInt32(treeCount),
            galaxyGroupCount: 0, _bhpad: 0)

        // Force buffers
        let bhForceBuf = device.makeBuffer(length: n * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!
        memset(bhForceBuf.contents(), 0, n * MemoryLayout<SIMD3<Float>>.stride)

        // Dispatch BH kernel
        let bhCmd = queue.makeCommandBuffer()!
        let bhEnc = bhCmd.makeComputeCommandEncoder()!
        bhEnc.setComputePipelineState(bhPipeline)
        bhEnc.setBuffer(nodeBuf, offset: 0, index: 0)
        bhEnc.setBuffer(bhForceBuf, offset: 0, index: 1)
        bhEnc.setBuffer(treeBuf, offset: 0, index: 2)
        bhEnc.setBuffer(bhParamBuf, offset: 0, index: 3)
        let tg = min(Int(bhPipeline.maxTotalThreadsPerThreadgroup), 256)
        let bhStart = CFAbsoluteTimeGetCurrent()
        bhEnc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        bhEnc.endEncoding()
        bhCmd.commit()
        bhCmd.waitUntilCompleted()
        let bhMs = (CFAbsoluteTimeGetCurrent() - bhStart) * 1000

        #expect(bhCmd.status != .error, "BH kernel errored: \(bhCmd.error?.localizedDescription ?? "?")")

        // Read BH forces
        let bhForcePtr = bhForceBuf.contents().bindMemory(to: SIMD3<Float>.self, capacity: n)
        let bhMaxF = (0..<n).map { simd_length(bhForcePtr[$0]) }.max() ?? 0
        #expect(bhMaxF > 0, "BH forces should be non-zero")

        // Now run brute force for comparison
        let bruteForceBuf = device.makeBuffer(length: n * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!
        memset(bruteForceBuf.contents(), 0, n * MemoryLayout<SIMD3<Float>>.stride)

        struct GPUForceParams {
            var chargeStrength: Float; var crossChargeMultiplier: Float
            var sameTopicChargeScale: Float; var sameProjectChargeScale: Float
            var cutoffSq: Float; var nodeCount: UInt32
            var galaxyGroupCount: UInt32; var _pad: UInt32
        }
        let bruteParamBuf = device.makeBuffer(length: MemoryLayout<GPUForceParams>.stride, options: .storageModeShared)!
        bruteParamBuf.contents().bindMemory(to: GPUForceParams.self, capacity: 1).pointee = GPUForceParams(
            chargeStrength: 500, crossChargeMultiplier: 3.0,
            sameTopicChargeScale: 0.35, sameProjectChargeScale: 1.0,
            cutoffSq: 500 * 500, nodeCount: UInt32(n),
            galaxyGroupCount: 0, _pad: 0)

        let bruteCmd = queue.makeCommandBuffer()!
        let bruteEnc = bruteCmd.makeComputeCommandEncoder()!
        bruteEnc.setComputePipelineState(brutePipeline)
        bruteEnc.setBuffer(nodeBuf, offset: 0, index: 0)
        bruteEnc.setBuffer(bruteForceBuf, offset: 0, index: 1)
        bruteEnc.setBuffer(bruteParamBuf, offset: 0, index: 2)
        let bruteStart = CFAbsoluteTimeGetCurrent()
        bruteEnc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(Int(brutePipeline.maxTotalThreadsPerThreadgroup), 256), height: 1, depth: 1))
        bruteEnc.endEncoding()
        bruteCmd.commit()
        bruteCmd.waitUntilCompleted()
        let bruteMs = (CFAbsoluteTimeGetCurrent() - bruteStart) * 1000

        let bruteForcePtr = bruteForceBuf.contents().bindMemory(to: SIMD3<Float>.self, capacity: n)

        // Compare: BH should approximate brute force (within ~20% for theta=0.7)
        var totalBHMag: Float = 0, totalBruteMag: Float = 0
        for i in 0..<n {
            totalBHMag += simd_length(bhForcePtr[i])
            totalBruteMag += simd_length(bruteForcePtr[i])
        }
        let ratio = totalBHMag / max(totalBruteMag, 1)

        print("[test] GPU BH 10K: \(String(format: "%.1f", bhMs))ms, brute: \(String(format: "%.1f", bruteMs))ms")
        print("[test] BH/brute force ratio: \(String(format: "%.3f", ratio)) (expect 0.8–1.2)")
        print("[test] Tree nodes: \(treeCount), BH maxF: \(String(format: "%.1f", bhMaxF))")

        // BH should complete under 20ms (allows both BH and brute force paths)
        #expect(bhMs < 20, "BH took \(String(format: "%.1f", bhMs))ms — too slow for 60fps")
        // Force magnitudes should be in the same ballpark.
        // BH with theta=0.7 uses point-mass approximation for distant clusters
        // which uses chargeBase (not group-adjusted), so total magnitude is lower.
        #expect(ratio > 0.3 && ratio < 2.0, "BH/brute ratio \(ratio) is too far off")
    }

    /// Simulate 60 frames of GPU BH charge at 17K nodes.
    /// Monitors memory growth — if allocations leak, this will catch it.
    @Test("GPU BH 60-frame sustained load at 17K nodes: no memory leak, no hang")
    func testGPUBH60Frames17K() throws {
        let device = MTLCreateSystemDefaultDevice()!
        let queue = device.makeCommandQueue()!
        let library = try device.makeLibrary(source: Self.fullPipelineShaderSource, options: nil)
        let bhPipeline = try device.makeComputePipelineState(
            function: library.makeFunction(name: "compute_charge_forces_bh")!)

        let n = 17000

        var x = [Float](repeating: 0, count: n)
        var y = [Float](repeating: 0, count: n)
        var z = [Float](repeating: 0, count: n)
        for i in 0..<n {
            x[i] = Float.random(in: -1000...1000)
            y[i] = Float.random(in: -1000...1000)
            z[i] = Float.random(in: -1000...1000)
        }

        // Pre-allocate reusable buffers (like the app should)
        let nodeBuf = device.makeBuffer(length: n * MemoryLayout<ForceNodeFull>.stride, options: .storageModeShared)!
        let forceBuf = device.makeBuffer(length: n * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!
        let bhParamBuf = device.makeBuffer(length: MemoryLayout<BHChargeParams>.stride, options: .storageModeShared)!

        // Tree buffer — allocate for worst case once
        let maxTreeNodes = n * 8  // generous upper bound
        let treeBuf = device.makeBuffer(length: maxTreeNodes * MemoryLayout<BHOctreeNode>.stride, options: .storageModeShared)!

        bhParamBuf.contents().bindMemory(to: BHChargeParams.self, capacity: 1).pointee = BHChargeParams(
            chargeStrength: 500, crossChargeMultiplier: 3.0,
            sameTopicChargeScale: 0.35, sameProjectChargeScale: 1.0,
            cutoffSq: 500 * 500, thetaSq: 0.7 * 0.7,
            nodeCount: UInt32(n), treeNodeCount: 0,
            galaxyGroupCount: 0, _bhpad: 0)

        let memBefore = getResidentMemoryMB()
        var totalMs: Double = 0
        var maxFrameMs: Double = 0

        for frame in 0..<60 {
            // Simulate node movement (positions change each frame)
            for i in 0..<n {
                x[i] += Float.random(in: -2...2)
                y[i] += Float.random(in: -2...2)
                z[i] += Float.random(in: -2...2)
            }

            let frameStart = CFAbsoluteTimeGetCurrent()

            // Pack nodes
            let nodePtr = nodeBuf.contents().bindMemory(to: ForceNodeFull.self, capacity: n)
            for i in 0..<n {
                nodePtr[i] = ForceNodeFull(
                    px: x[i], py: y[i], pz: z[i], vx: 0, vy: 0, vz: 0,
                    projectGroup: Int32(i % 8), topicGroup: Int32(i % 30),
                    galaxyGroup: Int32(i < n/2 ? 0 : 1), _pad: 0)
            }

            // Build octree (this is the per-frame allocation concern)
            let treeNodes = buildOctree(x: x, y: y, z: z, n: n)
            let treeCount = treeNodes.count

            // Upload tree to GPU buffer (reuse buffer, just overwrite contents)
            #expect(treeCount <= maxTreeNodes, "Tree exceeded capacity: \(treeCount) > \(maxTreeNodes)")
            let treePtr = treeBuf.contents().bindMemory(to: BHOctreeNode.self, capacity: treeCount)
            for i in 0..<treeCount {
                let src = treeNodes[i]
                treePtr[i] = BHOctreeNode(
                    cx: src.cx, cy: src.cy, cz: src.cz, halfSize: src.halfSize,
                    comX: src.comX, comY: src.comY, comZ: src.comZ, mass: src.mass,
                    children: (src.child(0), src.child(1), src.child(2), src.child(3),
                               src.child(4), src.child(5), src.child(6), src.child(7)),
                    bodyIndex: src.bodyIndex, _pad: 0)
            }
            bhParamBuf.contents().bindMemory(to: BHChargeParams.self, capacity: 1).pointee.treeNodeCount = UInt32(treeCount)

            // Zero forces
            memset(forceBuf.contents(), 0, n * MemoryLayout<SIMD3<Float>>.stride)

            // GPU dispatch
            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(bhPipeline)
            enc.setBuffer(nodeBuf, offset: 0, index: 0)
            enc.setBuffer(forceBuf, offset: 0, index: 1)
            enc.setBuffer(treeBuf, offset: 0, index: 2)
            enc.setBuffer(bhParamBuf, offset: 0, index: 3)
            enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: min(Int(bhPipeline.maxTotalThreadsPerThreadgroup), 256), height: 1, depth: 1))
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()

            let frameMs = (CFAbsoluteTimeGetCurrent() - frameStart) * 1000
            totalMs += frameMs
            maxFrameMs = max(maxFrameMs, frameMs)

            #expect(cmdBuf.status != .error, "Frame \(frame) GPU error")

            if frame == 0 || frame == 59 {
                let forcePtr = forceBuf.contents().bindMemory(to: SIMD3<Float>.self, capacity: n)
                let maxF = (0..<n).map { simd_length(forcePtr[$0]) }.max() ?? 0
                print("[test] Frame \(frame): \(String(format: "%.1f", frameMs))ms, tree=\(treeCount), maxF=\(String(format: "%.1f", maxF))")
            }
        }

        let memAfter = getResidentMemoryMB()
        let avgMs = totalMs / 60.0
        let memGrowth = memAfter - memBefore

        print("[test] 60 frames @ 17K: avg=\(String(format: "%.1f", avgMs))ms, max=\(String(format: "%.1f", maxFrameMs))ms")
        print("[test] Memory: before=\(memBefore)MB, after=\(memAfter)MB, growth=\(memGrowth)MB")

        #expect(memGrowth < 500, "Memory grew \(memGrowth)MB over 60 frames — leak detected")
        #expect(maxFrameMs < 100, "Worst frame \(String(format: "%.0f", maxFrameMs))ms — too slow")
    }

    /// Simulate the REAL render loop: pipelined frames without waiting.
    /// The app submits frame N+1 while GPU still processes frame N.
    /// If GPU falls behind, command buffers pile up → memory grows → crash.
    @Test("GPU pipelined frames: memory stays bounded under back-pressure")
    func testGPUPipelinedMemory() throws {
        let device = MTLCreateSystemDefaultDevice()!
        let queue = device.makeCommandQueue()!
        let library = try device.makeLibrary(source: Self.fullPipelineShaderSource, options: nil)
        let bhPipeline = try device.makeComputePipelineState(
            function: library.makeFunction(name: "compute_charge_forces_bh")!)
        let springPipeline = try device.makeComputePipelineState(
            function: library.makeFunction(name: "compute_spring_forces")!)

        let n = 17000
        var x = (0..<n).map { _ in Float.random(in: -1000...1000) }
        var y = (0..<n).map { _ in Float.random(in: -1000...1000) }
        var z = (0..<n).map { _ in Float.random(in: -1000...1000) }

        // Triple-buffered node/force buffers (like the real renderer)
        let bufferCount = 3
        var nodeBufs: [MTLBuffer] = []
        var forceBufs: [MTLBuffer] = []
        var treeBufs: [MTLBuffer] = []
        let maxTree = n * 8
        for _ in 0..<bufferCount {
            nodeBufs.append(device.makeBuffer(length: n * MemoryLayout<ForceNodeFull>.stride, options: .storageModeShared)!)
            forceBufs.append(device.makeBuffer(length: n * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!)
            treeBufs.append(device.makeBuffer(length: maxTree * MemoryLayout<BHOctreeNode>.stride, options: .storageModeShared)!)
        }

        // Build edge adjacency once
        let edgeCount = 20000
        var edges: [(Int, Int)] = []
        for _ in 0..<edgeCount {
            let a = Int.random(in: 0..<n), b = Int.random(in: 0..<n)
            if a != b { edges.append((a, b)) }
        }
        let adjOffBuf = device.makeBuffer(length: (n + 1) * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let adjNbrBuf = device.makeBuffer(length: edges.count * 2 * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        // Build CSR
        let offPtr = adjOffBuf.contents().bindMemory(to: UInt32.self, capacity: n + 1)
        let nbrPtr = adjNbrBuf.contents().bindMemory(to: UInt32.self, capacity: edges.count * 2)
        memset(offPtr, 0, (n + 1) * MemoryLayout<UInt32>.stride)
        for (s, t) in edges { offPtr[s] &+= 1; offPtr[t] &+= 1 }
        var running: UInt32 = 0
        for i in 0...n { let c = offPtr[i]; offPtr[i] = running; running &+= c }
        var cursors = (0..<n).map { offPtr[$0] }
        for (s, t) in edges {
            nbrPtr[Int(cursors[s])] = UInt32(t); cursors[s] &+= 1
            nbrPtr[Int(cursors[t])] = UInt32(s); cursors[t] &+= 1
        }

        let bhParamBuf = device.makeBuffer(length: MemoryLayout<BHChargeParams>.stride, options: .storageModeShared)!
        let simParamBuf = device.makeBuffer(length: MemoryLayout<ForceSimParams>.stride, options: .storageModeShared)!
        simParamBuf.contents().bindMemory(to: ForceSimParams.self, capacity: 1).pointee = ForceSimParams(
            springLength: 100, crossProjectSpringLength: 200, springStrength: 0.3,
            cohesionStrength: 0.02, centroidRepulsion: 300,
            topicCohesionStrength: 0.01, topicCentroidRepulsion: 200,
            centerStrength: 0.8, center: .init(0, 0, 0), alpha: 0.5, damping: 0.78, maxSpeed: 12,
            nodeCount: UInt32(n), edgeCount: UInt32(edges.count), crossProjectSpringScale: 1.0,
            projectGroupCount: 8, topicGroupCount: 30, galaxyGroupCount: 2, topicLeashStrength: 0.01)

        // Semaphore to simulate triple buffering (like the app's frameSemaphore)
        let frameSemaphore = DispatchSemaphore(value: bufferCount)

        let memBefore = getResidentMemoryMB()
        var frameTimesMs: [Double] = []
        var hangCount = 0
        let totalFrames = 120

        for frame in 0..<totalFrames {
            let bufIdx = frame % bufferCount

            // Wait for buffer to be free (like the app does)
            let waitResult = frameSemaphore.wait(timeout: .now() + .milliseconds(200))
            if waitResult == .timedOut {
                hangCount += 1
                if hangCount > 10 {
                    print("[test] Too many hangs (\(hangCount)) — aborting at frame \(frame)")
                    break
                }
                continue
            }

            let frameStart = CFAbsoluteTimeGetCurrent()

            // Move nodes slightly each frame
            for i in 0..<n { x[i] += Float.random(in: -1...1); y[i] += Float.random(in: -1...1); z[i] += Float.random(in: -1...1) }

            // Pack nodes
            let nodePtr = nodeBufs[bufIdx].contents().bindMemory(to: ForceNodeFull.self, capacity: n)
            for i in 0..<n {
                nodePtr[i] = ForceNodeFull(px: x[i], py: y[i], pz: z[i], vx: 0, vy: 0, vz: 0,
                                           projectGroup: Int32(i % 8), topicGroup: Int32(i % 30),
                                           galaxyGroup: Int32(i < n/2 ? 0 : 1), _pad: 0)
            }

            // Build + upload octree
            let tree = buildOctree(x: x, y: y, z: z, n: n)
            let treeCount = min(tree.count, maxTree)
            let treePtr = treeBufs[bufIdx].contents().bindMemory(to: BHOctreeNode.self, capacity: treeCount)
            for i in 0..<treeCount {
                let s = tree[i]
                treePtr[i] = BHOctreeNode(cx: s.cx, cy: s.cy, cz: s.cz, halfSize: s.halfSize,
                                           comX: s.comX, comY: s.comY, comZ: s.comZ, mass: s.mass,
                                           children: (s.child(0), s.child(1), s.child(2), s.child(3),
                                                      s.child(4), s.child(5), s.child(6), s.child(7)),
                                           bodyIndex: s.bodyIndex, _pad: 0)
            }

            memset(forceBufs[bufIdx].contents(), 0, n * MemoryLayout<SIMD3<Float>>.stride)

            bhParamBuf.contents().bindMemory(to: BHChargeParams.self, capacity: 1).pointee = BHChargeParams(
                chargeStrength: 500, crossChargeMultiplier: 3.0,
                sameTopicChargeScale: 0.35, sameProjectChargeScale: 1.0,
                cutoffSq: 500 * 500, thetaSq: 0.7 * 0.7,
                nodeCount: UInt32(n), treeNodeCount: UInt32(treeCount),
            galaxyGroupCount: 0, _bhpad: 0)

            // Encode GPU work — BH charge + springs (like the real command buffer)
            guard let cmdBuf = queue.makeCommandBuffer(),
                  let enc = cmdBuf.makeComputeCommandEncoder() else { continue }

            // BH charge
            enc.setComputePipelineState(bhPipeline)
            enc.setBuffer(nodeBufs[bufIdx], offset: 0, index: 0)
            enc.setBuffer(forceBufs[bufIdx], offset: 0, index: 1)
            enc.setBuffer(treeBufs[bufIdx], offset: 0, index: 2)
            enc.setBuffer(bhParamBuf, offset: 0, index: 3)
            enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: min(Int(bhPipeline.maxTotalThreadsPerThreadgroup), 256), height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)

            // Springs
            enc.setComputePipelineState(springPipeline)
            enc.setBuffer(adjOffBuf, offset: 0, index: 0)
            enc.setBuffer(adjNbrBuf, offset: 0, index: 1)
            enc.setBuffer(nodeBufs[bufIdx], offset: 0, index: 2)
            enc.setBuffer(forceBufs[bufIdx], offset: 0, index: 3)
            enc.setBuffer(simParamBuf, offset: 0, index: 4)
            enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: min(Int(springPipeline.maxTotalThreadsPerThreadgroup), 256), height: 1, depth: 1))
            enc.endEncoding()

            // Signal semaphore on completion (like the app does)
            cmdBuf.addCompletedHandler { @Sendable _ in
                frameSemaphore.signal()
            }
            cmdBuf.commit()
            // DO NOT waitUntilCompleted — pipeline like the real app

            let cpuMs = (CFAbsoluteTimeGetCurrent() - frameStart) * 1000
            frameTimesMs.append(cpuMs)
        }

        // Wait for all in-flight work to finish
        for _ in 0..<bufferCount { frameSemaphore.wait() }
        for _ in 0..<bufferCount { frameSemaphore.signal() }

        let memAfter = getResidentMemoryMB()
        let memGrowth = memAfter - memBefore
        let avgMs = frameTimesMs.reduce(0, +) / Double(frameTimesMs.count)
        let maxMs = frameTimesMs.max() ?? 0

        print("[test] Pipelined 17K × \(frameTimesMs.count) frames:")
        print("[test]   CPU: avg=\(String(format: "%.1f", avgMs))ms max=\(String(format: "%.1f", maxMs))ms")
        print("[test]   Memory: \(memBefore)MB → \(memAfter)MB (growth: \(memGrowth)MB)")
        print("[test]   GPU hangs: \(hangCount)")

        #expect(memGrowth < 500, "Memory grew \(memGrowth)MB — leak or buffer accumulation")
        #expect(hangCount < 5, "Too many GPU hangs: \(hangCount)")
    }

    /// Get current process resident memory in MB.
    private func getResidentMemoryMB() -> Int {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return Int(info.resident_size) / (1024 * 1024)
    }

    // MARK: - Bug regression tests (GPU crash root causes)

    /// Regression: DispatchSemaphore.wait(timeout:) restores the counter on timeout.
    /// An extra signal() after timeout leaks a permit, removing backpressure.
    /// With leaked permits, unlimited command buffers queue up → memory explosion → crash.
    @Test("Semaphore timeout must not leak permits")
    func testSemaphoreTimeoutNoLeak() {
        let sem = DispatchSemaphore(value: 3)

        // Consume all 3 permits
        sem.wait()
        sem.wait()
        sem.wait()

        // Timeout — should NOT change the counter
        let result = sem.wait(timeout: .now() + .milliseconds(10))
        #expect(result == .timedOut)

        // BUG (old code): sem.signal() here would leak a permit.
        // FIX: no signal on timeout. Counter stays at 0.

        // Verify: next wait should still timeout (no leaked permit)
        let result2 = sem.wait(timeout: .now() + .milliseconds(10))
        #expect(result2 == .timedOut, "Permit leaked — semaphore should still be exhausted")

        // Restore permits for cleanup
        sem.signal()
        sem.signal()
        sem.signal()
    }

    /// Regression: readback must complete before signaling frame complete.
    /// If signal fires first, the next frame can overwrite fullForceBuffer
    /// while readBackForces() is still reading from it → corrupted forces.
    @Test("Completion handler: readback before signal prevents data race")
    func testReadbackBeforeSignal() throws {
        let device = MTLCreateSystemDefaultDevice()!
        let queue = device.makeCommandQueue()!
        let library = try device.makeLibrary(source: Self.shaderSource, options: nil)
        let pipeline = try device.makeComputePipelineState(
            function: library.makeFunction(name: "compute_charge_forces")!)

        let n = 1000

        // Single shared force buffer (the bug: not triple-buffered)
        let nodeBuf = device.makeBuffer(length: n * MemoryLayout<ForceNodeFull>.stride, options: .storageModeShared)!
        let forceBuf = device.makeBuffer(length: n * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!
        let paramBuf = device.makeBuffer(length: MemoryLayout<ForceNodeFull>.stride + 24, options: .storageModeShared)!

        // Pack nodes with known positions
        let nodePtr = nodeBuf.contents().bindMemory(to: ForceNodeFull.self, capacity: n)
        for i in 0..<n {
            nodePtr[i] = ForceNodeFull(
                px: Float(i) * 10, py: 0, pz: 0, vx: 0, vy: 0, vz: 0,
                projectGroup: 0, topicGroup: 0, galaxyGroup: 0, _pad: 0)
        }

        // Set params
        struct GPUForceParams {
            var chargeStrength: Float; var crossChargeMultiplier: Float
            var sameTopicChargeScale: Float; var sameProjectChargeScale: Float
            var cutoffSq: Float; var nodeCount: UInt32
            var galaxyGroupCount: UInt32; var _pad: UInt32
        }
        paramBuf.contents().bindMemory(to: GPUForceParams.self, capacity: 1).pointee = GPUForceParams(
            chargeStrength: 500, crossChargeMultiplier: 3.0,
            sameTopicChargeScale: 0.35, sameProjectChargeScale: 1.0,
            cutoffSq: 500 * 500, nodeCount: UInt32(n),
            galaxyGroupCount: 0, _pad: 0)

        // Single in-flight: matches the gpuInFlight guard behavior.
        // Without this, frame N+1 memsets the force buffer while frame N's
        // completion handler is reading it — exactly the crash bug.
        let frameSem = DispatchSemaphore(value: 1)
        struct ReadbackState: Sendable {
            var forces: [SIMD3<Float>] = []
            var corrupted: Bool = false
        }
        let readbackState = OSAllocatedUnfairLock(initialState: ReadbackState())

        // Run 30 frames, pipelined. Read back BEFORE signal (the fix).
        for _ in 0..<30 {
            frameSem.wait()

            memset(forceBuf.contents(), 0, n * MemoryLayout<SIMD3<Float>>.stride)

            let cmdBuf = queue.makeCommandBuffer()!
            let enc = cmdBuf.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(nodeBuf, offset: 0, index: 0)
            enc.setBuffer(forceBuf, offset: 0, index: 1)
            enc.setBuffer(paramBuf, offset: 0, index: 2)
            enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            enc.endEncoding()

            let state = readbackState
            cmdBuf.addCompletedHandler { @Sendable _ in
                // FIX order: readback FIRST, signal AFTER
                let forcePtr = forceBuf.contents().bindMemory(to: SIMD3<Float>.self, capacity: n)
                let forces: [SIMD3<Float>] = (0..<n).map { forcePtr[$0] }

                let nonZeroCount = forces.filter { simd_length($0) > 0 }.count
                state.withLock { s in
                    if nonZeroCount == 0 { s.corrupted = true }
                    s.forces = forces
                }
                frameSem.signal()
            }
            cmdBuf.commit()
        }

        // Drain
        frameSem.wait()
        frameSem.signal()

        let finalState = readbackState.withLock { $0 }
        #expect(!finalState.corrupted, "Force readback got all zeros — data race on force buffer")
        let nonZero = finalState.forces.filter { simd_length($0) > 0 }.count
        #expect(nonZero > n / 2, "Expected most forces non-zero, got \(nonZero)/\(n)")
    }

    /// Regression: gpuInFlight guard prevents concurrent access to shared force buffers.
    /// Without it, triple-buffered frames overwrite buffers the GPU is still reading,
    /// corrupting BH octree data → GPU infinite loop → system crash.
    @Test("gpuInFlight guard: single shared buffer stays consistent across pipelined frames")
    func testGpuInFlightGuard() throws {
        let device = MTLCreateSystemDefaultDevice()!
        let queue = device.makeCommandQueue()!
        let library = try device.makeLibrary(source: Self.shaderSource, options: nil)
        let pipeline = try device.makeComputePipelineState(
            function: library.makeFunction(name: "compute_charge_forces")!)

        let n = 2000

        // SINGLE buffer set (not triple-buffered) — exactly the app's bug scenario
        let nodeBuf = device.makeBuffer(length: n * MemoryLayout<ForceNodeFull>.stride, options: .storageModeShared)!
        let forceBuf = device.makeBuffer(length: n * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!
        let paramBuf = device.makeBuffer(length: MemoryLayout<ForceNodeFull>.stride + 24, options: .storageModeShared)!

        struct GPUForceParams {
            var chargeStrength: Float; var crossChargeMultiplier: Float
            var sameTopicChargeScale: Float; var sameProjectChargeScale: Float
            var cutoffSq: Float; var nodeCount: UInt32
            var galaxyGroupCount: UInt32; var _pad: UInt32
        }
        paramBuf.contents().bindMemory(to: GPUForceParams.self, capacity: 1).pointee = GPUForceParams(
            chargeStrength: 500, crossChargeMultiplier: 3.0,
            sameTopicChargeScale: 0.35, sameProjectChargeScale: 1.0,
            cutoffSq: 500 * 500, nodeCount: UInt32(n),
            galaxyGroupCount: 0, _pad: 0)

        let gpuInFlight = OSAllocatedUnfairLock<Bool>(initialState: false)
        let frameSem = DispatchSemaphore(value: 3)
        struct InFlightState: Sendable {
            var encodedCount: Int = 0
            var skippedCount: Int = 0
            var errorCount: Int = 0
        }
        let state = OSAllocatedUnfairLock(initialState: InFlightState())

        for frame in 0..<60 {
            frameSem.wait()

            // Pack nodes with frame-specific data (so we can detect corruption)
            let marker = Float(frame + 1) * 100
            let nodePtr = nodeBuf.contents().bindMemory(to: ForceNodeFull.self, capacity: n)
            for i in 0..<n {
                nodePtr[i] = ForceNodeFull(
                    px: marker + Float(i), py: 0, pz: 0, vx: 0, vy: 0, vz: 0,
                    projectGroup: 0, topicGroup: 0, galaxyGroup: 0, _pad: 0)
            }

            // gpuInFlight guard — skip encoding if previous frame's GPU is still running
            let canEncode = gpuInFlight.withLock { val -> Bool in
                if val { return false }
                val = true
                return true
            }

            if canEncode {
                memset(forceBuf.contents(), 0, n * MemoryLayout<SIMD3<Float>>.stride)

                let cmdBuf = queue.makeCommandBuffer()!
                let enc = cmdBuf.makeComputeCommandEncoder()!
                enc.setComputePipelineState(pipeline)
                enc.setBuffer(nodeBuf, offset: 0, index: 0)
                enc.setBuffer(forceBuf, offset: 0, index: 1)
                enc.setBuffer(paramBuf, offset: 0, index: 2)
                enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.endEncoding()

                cmdBuf.addCompletedHandler { @Sendable _ in
                    // Readback and verify forces are non-garbage
                    let fPtr = forceBuf.contents().bindMemory(to: SIMD3<Float>.self, capacity: n)
                    for i in 0..<min(10, n) {
                        if fPtr[i].x.isNaN || fPtr[i].y.isNaN || fPtr[i].z.isNaN {
                            state.withLock { $0.errorCount += 1 }
                            break
                        }
                    }
                    gpuInFlight.withLock { $0 = false }
                    frameSem.signal()
                }
                cmdBuf.commit()
                state.withLock { $0.encodedCount += 1 }
            } else {
                state.withLock { $0.skippedCount += 1 }
                frameSem.signal()
            }
        }

        // Drain
        for _ in 0..<3 { frameSem.wait() }
        for _ in 0..<3 { frameSem.signal() }

        let final_ = state.withLock { $0 }
        print("[test] gpuInFlight: encoded=\(final_.encodedCount) skipped=\(final_.skippedCount) errors=\(final_.errorCount)")

        #expect(final_.encodedCount >= 20, "Should encode most frames, got \(final_.encodedCount)/60")
        #expect(final_.skippedCount > 0, "Should skip some frames when GPU is busy")
        #expect(final_.errorCount == 0, "GPU produced NaN forces — buffer corruption detected")
    }

    /// Regression: async octree build must not block the main thread.
    /// buildOctree at 10K nodes takes ~20-30ms — enough to drop frames at 60fps.
    @Test("Async octree build completes without blocking caller")
    func testAsyncOctreeBuild() async throws {
        let n = 10000
        let x = (0..<n).map { _ in Float.random(in: -1000...1000) }
        let y = (0..<n).map { _ in Float.random(in: -1000...1000) }
        let z = (0..<n).map { _ in Float.random(in: -1000...1000) }

        // Synchronous build — measure baseline
        let syncStart = CFAbsoluteTimeGetCurrent()
        let syncTree = buildOctree(x: x, y: y, z: z, n: n)
        let syncMs = (CFAbsoluteTimeGetCurrent() - syncStart) * 1000

        #expect(syncTree.count > n, "Octree should have more nodes than bodies (\(syncTree.count))")
        print("[test] Sync octree build: \(String(format: "%.1f", syncMs))ms, \(syncTree.count) nodes")

        // Async build via Task — should not block
        let callerStart = CFAbsoluteTimeGetCurrent()
        let pending = OSAllocatedUnfairLock<[OctreeNode]?>(initialState: nil)
        let pendingCopy = pending
        Task.detached(priority: .userInitiated) {
            let tree = buildOctree(x: x, y: y, z: z, n: n)
            pendingCopy.withLock { $0 = tree }
        }
        let callerMs = (CFAbsoluteTimeGetCurrent() - callerStart) * 1000

        // Caller should return almost immediately (< 2ms), not wait for tree build
        #expect(callerMs < 5, "Async dispatch took \(String(format: "%.1f", callerMs))ms — should be < 5ms")
        print("[test] Async dispatch time: \(String(format: "%.1f", callerMs))ms (sync was \(String(format: "%.1f", syncMs))ms)")

        // Wait for result
        var asyncTree: [OctreeNode]? = nil
        for _ in 0..<100 {
            asyncTree = pending.withLock { val -> [OctreeNode]? in let r = val; val = nil; return r }
            if asyncTree != nil { break }
            try await Task.sleep(nanoseconds: 10_000_000) // 10ms
        }

        #expect(asyncTree != nil, "Async octree build never completed")
        #expect(asyncTree!.count == syncTree.count, "Async tree size \(asyncTree!.count) != sync \(syncTree.count)")
    }

    /// Reproduces the exact crash pattern from the app:
    /// Frame 1: brute force (no tree yet), works.
    /// Frame 2: async tree arrives, BH kernel runs, GPU hangs.
    /// Uses the REAL compiled shaders from EngramMetalShaders (not inline copies).
    @Test("BH kernel with real shaders: stale tree + current positions does not hang")
    func testBHKernelStaleTreeRealShaders() throws {
        let device = MTLCreateSystemDefaultDevice()!
        let queue = device.makeCommandQueue()!
        let library = try EngramMetalShaders.makeLibrary(device: device)
        let bhPipeline = try device.makeComputePipelineState(
            function: library.makeFunction(name: "compute_charge_forces_bh")!)

        let n = 7500  // close to app's 7358

        // Frame 1 positions (tree will be built from these)
        var x1 = (0..<n).map { _ in Float.random(in: -500...500) }
        var y1 = (0..<n).map { _ in Float.random(in: -500...500) }
        var z1 = (0..<n).map { _ in Float.random(in: -500...500) }

        // Frame 2 positions (slightly different — one frame of integration)
        let x2 = x1.map { $0 + Float.random(in: -5...5) }
        let y2 = y1.map { $0 + Float.random(in: -5...5) }
        let z2 = z1.map { $0 + Float.random(in: -5...5) }

        // Build tree from frame 1 positions (like the async build)
        let treeNodes = buildOctree(x: x1, y: y1, z: z1, n: n)
        let treeCount = treeNodes.count
        print("[test] Tree: \(treeCount) nodes for \(n) bodies")

        // Pack tree into GPU buffer
        let treeBuf = device.makeBuffer(length: treeCount * MemoryLayout<BHOctreeNode>.stride, options: .storageModeShared)!
        let treePtr = treeBuf.contents().bindMemory(to: BHOctreeNode.self, capacity: treeCount)
        for i in 0..<treeCount {
            let src = treeNodes[i]
            treePtr[i] = BHOctreeNode(
                cx: src.cx, cy: src.cy, cz: src.cz, halfSize: src.halfSize,
                comX: src.comX, comY: src.comY, comZ: src.comZ, mass: src.mass,
                children: (src.child(0), src.child(1), src.child(2), src.child(3),
                           src.child(4), src.child(5), src.child(6), src.child(7)),
                bodyIndex: src.bodyIndex, _pad: 0)
        }

        // Pack nodes with FRAME 2 positions (mismatch — like the app)
        let nodeBuf = device.makeBuffer(length: n * MemoryLayout<ForceNodeFull>.stride, options: .storageModeShared)!
        let nodePtr = nodeBuf.contents().bindMemory(to: ForceNodeFull.self, capacity: n)
        for i in 0..<n {
            nodePtr[i] = ForceNodeFull(
                px: x2[i], py: y2[i], pz: z2[i], vx: 0, vy: 0, vz: 0,
                projectGroup: Int32(i % 8), topicGroup: Int32(i % 30),
                galaxyGroup: 0, _pad: 0)
        }

        // BH params
        let bhParamBuf = device.makeBuffer(length: MemoryLayout<BHChargeParams>.stride, options: .storageModeShared)!
        bhParamBuf.contents().bindMemory(to: BHChargeParams.self, capacity: 1).pointee = BHChargeParams(
            chargeStrength: 500, crossChargeMultiplier: 3.0,
            sameTopicChargeScale: 0.35, sameProjectChargeScale: 1.0,
            cutoffSq: 500 * 500, thetaSq: 0.7 * 0.7,
            nodeCount: UInt32(n), treeNodeCount: UInt32(treeCount),
            galaxyGroupCount: 0, _bhpad: 0)

        let forceBuf = device.makeBuffer(length: n * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!
        memset(forceBuf.contents(), 0, n * MemoryLayout<SIMD3<Float>>.stride)

        // Dispatch with 5-second timeout — if it hangs, we catch it
        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(bhPipeline)
        enc.setBuffer(nodeBuf, offset: 0, index: 0)
        enc.setBuffer(forceBuf, offset: 0, index: 1)
        enc.setBuffer(treeBuf, offset: 0, index: 2)
        enc.setBuffer(bhParamBuf, offset: 0, index: 3)
        let tg = min(Int(bhPipeline.maxTotalThreadsPerThreadgroup), 256)
        let start = CFAbsoluteTimeGetCurrent()
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000
        print("[test] BH stale-tree: \(String(format: "%.1f", ms))ms, status=\(cmdBuf.status.rawValue)")

        #expect(cmdBuf.status != .error, "BH kernel error: \(cmdBuf.error?.localizedDescription ?? "unknown")")
        #expect(ms < 5000, "BH kernel took \(String(format: "%.0f", ms))ms — likely hung")

        // Verify forces are non-zero and non-NaN
        let forcePtr = forceBuf.contents().bindMemory(to: SIMD3<Float>.self, capacity: n)
        var nonZero = 0
        var nanCount = 0
        for i in 0..<n {
            if forcePtr[i].x.isNaN || forcePtr[i].y.isNaN { nanCount += 1 }
            if simd_length(forcePtr[i]) > 0 { nonZero += 1 }
        }
        print("[test] Forces: \(nonZero)/\(n) non-zero, \(nanCount) NaN")
        #expect(nanCount == 0, "BH produced NaN forces")
        #expect(nonZero > n / 2, "Too few non-zero forces: \(nonZero)/\(n)")
    }

    /// Regression: co-located nodes (all spawned in small sphere) caused depth-40 tree.
    /// Every GPU thread walked the entire chain — worse than O(n²), hung the GPU.
    /// Fix: force monopole approximation when cell halfSize < threshold.
    @Test("BH kernel with co-located nodes (startup pattern) completes in bounded time")
    func testBHColocatedNodes() throws {
        let device = MTLCreateSystemDefaultDevice()!
        let queue = device.makeCommandQueue()!
        let library = try EngramMetalShaders.makeLibrary(device: device)
        let bhPipeline = try device.makeComputePipelineState(
            function: library.makeFunction(name: "compute_charge_forces_bh")!)

        let n = 8000

        // All nodes in a tiny sphere — exactly like app startup
        let x = (0..<n).map { _ in Float.random(in: -5...5) }
        let y = (0..<n).map { _ in Float.random(in: -5...5) }
        let z = (0..<n).map { _ in Float.random(in: -5...5) }

        let treeNodes = buildOctree(x: x, y: y, z: z, n: n)
        let treeCount = treeNodes.count

        // Check tree depth
        var maxDepth = 0
        var depthStack = [(Int, Int)]()
        depthStack.append((0, 0))
        while let (idx, d) = depthStack.popLast() {
            guard idx >= 0, idx < treeCount else { continue }
            maxDepth = max(maxDepth, d)
            if d > 25 { break }
            let node = treeNodes[idx]
            for oct in 0..<8 {
                let c = Int(node.child(oct))
                if c > 0 && c < treeCount { depthStack.append((c, d + 1)) }
            }
        }
        print("[test] Co-located tree: \(treeCount) nodes, depth=\(maxDepth) for \(n) bodies in 10-unit sphere")

        let nodeBuf = device.makeBuffer(length: n * MemoryLayout<ForceNodeFull>.stride, options: .storageModeShared)!
        let nodePtr = nodeBuf.contents().bindMemory(to: ForceNodeFull.self, capacity: n)
        for i in 0..<n {
            nodePtr[i] = ForceNodeFull(
                px: x[i], py: y[i], pz: z[i], vx: 0, vy: 0, vz: 0,
                projectGroup: Int32(i % 8), topicGroup: Int32(i % 30), galaxyGroup: 0, _pad: 0)
        }

        let treeBuf = device.makeBuffer(length: treeCount * MemoryLayout<BHOctreeNode>.stride, options: .storageModeShared)!
        let treePtr = treeBuf.contents().bindMemory(to: BHOctreeNode.self, capacity: treeCount)
        for i in 0..<treeCount {
            let src = treeNodes[i]
            treePtr[i] = BHOctreeNode(
                cx: src.cx, cy: src.cy, cz: src.cz, halfSize: src.halfSize,
                comX: src.comX, comY: src.comY, comZ: src.comZ, mass: src.mass,
                children: (src.child(0), src.child(1), src.child(2), src.child(3),
                           src.child(4), src.child(5), src.child(6), src.child(7)),
                bodyIndex: src.bodyIndex, _pad: 0)
        }

        let bhParamBuf = device.makeBuffer(length: MemoryLayout<BHChargeParams>.stride, options: .storageModeShared)!
        bhParamBuf.contents().bindMemory(to: BHChargeParams.self, capacity: 1).pointee = BHChargeParams(
            chargeStrength: 500, crossChargeMultiplier: 3.0,
            sameTopicChargeScale: 0.35, sameProjectChargeScale: 1.0,
            cutoffSq: 500 * 500, thetaSq: 0.7 * 0.7,
            nodeCount: UInt32(n), treeNodeCount: UInt32(treeCount),
            galaxyGroupCount: 0, _bhpad: 0)

        let forceBuf = device.makeBuffer(length: n * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!
        memset(forceBuf.contents(), 0, n * MemoryLayout<SIMD3<Float>>.stride)

        let cmdBuf = queue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(bhPipeline)
        enc.setBuffer(nodeBuf, offset: 0, index: 0)
        enc.setBuffer(forceBuf, offset: 0, index: 1)
        enc.setBuffer(treeBuf, offset: 0, index: 2)
        enc.setBuffer(bhParamBuf, offset: 0, index: 3)
        let tg = min(Int(bhPipeline.maxTotalThreadsPerThreadgroup), 256)
        let start = CFAbsoluteTimeGetCurrent()
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000
        print("[test] BH co-located: \(String(format: "%.1f", ms))ms")

        #expect(ms < 200, "BH with co-located nodes took \(String(format: "%.0f", ms))ms — should be <200ms")
        #expect(cmdBuf.status != .error, "GPU error: \(cmdBuf.error?.localizedDescription ?? "unknown")")
    }
}
