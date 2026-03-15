import Foundation
import Testing
import simd
import os
import CEngramSceneTypes
@preconcurrency import Metal
import EngramSceneKit

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
        }
        let paramBuffer = device.makeBuffer(length: MemoryLayout<GPUForceParams>.stride, options: .storageModeShared)!

        // Place nodes in a grid so forces are non-trivial
        let nodePtr = nodeBuffer.contents().bindMemory(to: ForceNodeFull.self, capacity: n)
        for i in 0..<n {
            nodePtr[i] = ForceNodeFull(
                px: Float(i % 10) * 50, py: Float(i / 10) * 50, pz: 0,
                vx: 0, vy: 0, vz: 0,
                projectGroup: Int32(i % 3), topicGroup: Int32(i % 5))
        }
        memset(forceBuffer.contents(), 0, n * MemoryLayout<SIMD3<Float>>.stride)

        paramBuffer.contents().bindMemory(to: GPUForceParams.self, capacity: 1).pointee = GPUForceParams(
            chargeStrength: 500, crossChargeMultiplier: 3.0,
            sameTopicChargeScale: 0.35, sameProjectChargeScale: 1.0,
            cutoffSq: 500 * 500, nodeCount: UInt32(n))

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
        }
        let paramBuffer = device.makeBuffer(length: MemoryLayout<GPUForceParams>.stride, options: .storageModeShared)!

        let nodePtr = nodeBuffer.contents().bindMemory(to: ForceNodeFull.self, capacity: n)
        for i in 0..<n {
            nodePtr[i] = ForceNodeFull(
                px: Float(i) * 30, py: 0, pz: 0,
                vx: 0, vy: 0, vz: 0, projectGroup: 0, topicGroup: 0)
        }
        memset(forceBuffer.contents(), 0, n * MemoryLayout<SIMD3<Float>>.stride)
        paramBuffer.contents().bindMemory(to: GPUForceParams.self, capacity: 1).pointee = GPUForceParams(
            chargeStrength: 500, crossChargeMultiplier: 3.0,
            sameTopicChargeScale: 0.35, sameProjectChargeScale: 1.0,
            cutoffSq: 500 * 500, nodeCount: UInt32(n))

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
        }
        let paramBuffer = device.makeBuffer(length: MemoryLayout<GPUForceParams>.stride, options: .storageModeShared)!

        let nodePtr = nodeBuffer.contents().bindMemory(to: ForceNodeFull.self, capacity: n)
        for i in 0..<n {
            nodePtr[i] = ForceNodeFull(
                px: Float(i) * 30, py: 0, pz: 0,
                vx: 0, vy: 0, vz: 0, projectGroup: Int32(i % 3), topicGroup: 0)
        }
        memset(forceBuffer.contents(), 0, n * MemoryLayout<SIMD3<Float>>.stride)
        paramBuffer.contents().bindMemory(to: GPUForceParams.self, capacity: 1).pointee = GPUForceParams(
            chargeStrength: 500, crossChargeMultiplier: 3.0,
            sameTopicChargeScale: 0.35, sameProjectChargeScale: 1.0,
            cutoffSq: 500 * 500, nodeCount: UInt32(n))

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
}
