import Foundation
import Testing
import simd
import CEngramSceneTypes
@preconcurrency import Metal

/// GPU force computation perf tests using real Metal device.
/// Compiles shaders from source at runtime — no app bundle needed.
@Suite("GPU Force Performance")
struct GPUForcePerfTests {

    /// Minimal shader source for the charge kernel — no framework dependencies.
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

    static let device: MTLDevice? = MTLCreateSystemDefaultDevice()

    private func measure(_ label: String, iterations: Int = 5, block: () -> Void) -> Double {
        block()
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations { block() }
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let avgMs = (elapsed / Double(iterations)) * 1000.0
        print("[engram:perf] \(label): \(String(format: "%.2f", avgMs))ms")
        return avgMs
    }

    @Test("GPU charge kernel 8K nodes: under 15ms")
    func testGPUCharge8K() throws {
        guard let device = Self.device else {
            print("[engram:perf] No Metal device available, skipping GPU test")
            return
        }
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        let library = try device.makeLibrary(source: Self.shaderSource, options: options)
        guard let function = library.makeFunction(name: "compute_charge_forces"),
              let pipeline = try? device.makeComputePipelineState(function: function),
              let queue = device.makeCommandQueue() else {
            Issue.record("Failed to create Metal pipeline")
            return
        }

        let n = 8684
        let nodeBuffer = device.makeBuffer(length: n * MemoryLayout<ForceNodeFull>.stride, options: .storageModeShared)!
        let forceBuffer = device.makeBuffer(length: n * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!

        // Fill with random positions
        let nodePtr = nodeBuffer.contents().bindMemory(to: ForceNodeFull.self, capacity: n)
        for i in 0..<n {
            nodePtr[i] = ForceNodeFull(
                px: Float.random(in: -500...500),
                py: Float.random(in: -500...500),
                pz: Float.random(in: -500...500),
                vx: 0, vy: 0, vz: 0,
                projectGroup: Int32(i % 8),
                topicGroup: Int32(i % 30),
                galaxyGroup: 0, _pad: 0)
        }

        struct GPUForceParams {
            var chargeStrength: Float
            var crossChargeMultiplier: Float
            var sameTopicChargeScale: Float
            var sameProjectChargeScale: Float
            var cutoffSq: Float
            var nodeCount: UInt32
            var galaxyGroupCount: UInt32
            var _pad: UInt32
        }
        let paramBuffer = device.makeBuffer(length: MemoryLayout<GPUForceParams>.stride, options: .storageModeShared)!
        paramBuffer.contents().bindMemory(to: GPUForceParams.self, capacity: 1).pointee = GPUForceParams(
            chargeStrength: 500,
            crossChargeMultiplier: 3.0,
            sameTopicChargeScale: 0.35,
            sameProjectChargeScale: 1.0,
            cutoffSq: 500 * 500,
            nodeCount: UInt32(n),
            galaxyGroupCount: 0, _pad: 0)

        let tgSize = min(Int(pipeline.maxTotalThreadsPerThreadgroup), 256)

        let ms = measure("gpu-charge-8K") {
            memset(forceBuffer.contents(), 0, n * MemoryLayout<SIMD3<Float>>.stride)
            guard let cmdBuf = queue.makeCommandBuffer(),
                  let enc = cmdBuf.makeComputeCommandEncoder() else { return }
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(nodeBuffer, offset: 0, index: 0)
            enc.setBuffer(forceBuffer, offset: 0, index: 1)
            enc.setBuffer(paramBuffer, offset: 0, index: 2)
            enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        #expect(ms < 15.0, "GPU charge took \(String(format: "%.1f", ms))ms, budget 15ms")
    }

    @Test("GPU charge kernel 20K nodes: under 50ms")
    func testGPUCharge20K() throws {
        guard let device = Self.device else { return }
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        let library = try device.makeLibrary(source: Self.shaderSource, options: options)
        guard let function = library.makeFunction(name: "compute_charge_forces"),
              let pipeline = try? device.makeComputePipelineState(function: function),
              let queue = device.makeCommandQueue() else { return }

        let n = 20000
        let nodeBuffer = device.makeBuffer(length: n * MemoryLayout<ForceNodeFull>.stride, options: .storageModeShared)!
        let forceBuffer = device.makeBuffer(length: n * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!

        let nodePtr = nodeBuffer.contents().bindMemory(to: ForceNodeFull.self, capacity: n)
        for i in 0..<n {
            nodePtr[i] = ForceNodeFull(
                px: Float.random(in: -500...500), py: Float.random(in: -500...500),
                pz: Float.random(in: -500...500), vx: 0, vy: 0, vz: 0,
                projectGroup: Int32(i % 8), topicGroup: Int32(i % 30),
                galaxyGroup: 0, _pad: 0)
        }

        struct GPUForceParams {
            var chargeStrength: Float; var crossChargeMultiplier: Float
            var sameTopicChargeScale: Float; var sameProjectChargeScale: Float
            var cutoffSq: Float; var nodeCount: UInt32
            var galaxyGroupCount: UInt32; var _pad: UInt32
        }
        let paramBuffer = device.makeBuffer(length: MemoryLayout<GPUForceParams>.stride, options: .storageModeShared)!
        paramBuffer.contents().bindMemory(to: GPUForceParams.self, capacity: 1).pointee = GPUForceParams(
            chargeStrength: 500, crossChargeMultiplier: 3.0,
            sameTopicChargeScale: 0.35, sameProjectChargeScale: 1.0,
            cutoffSq: 500 * 500, nodeCount: UInt32(n),
            galaxyGroupCount: 0, _pad: 0)

        let tgSize = min(Int(pipeline.maxTotalThreadsPerThreadgroup), 256)

        let ms = measure("gpu-charge-20K") {
            memset(forceBuffer.contents(), 0, n * MemoryLayout<SIMD3<Float>>.stride)
            guard let cmdBuf = queue.makeCommandBuffer(),
                  let enc = cmdBuf.makeComputeCommandEncoder() else { return }
            enc.setComputePipelineState(pipeline)
            enc.setBuffer(nodeBuffer, offset: 0, index: 0)
            enc.setBuffer(forceBuffer, offset: 0, index: 1)
            enc.setBuffer(paramBuffer, offset: 0, index: 2)
            enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
            enc.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }
        #expect(ms < 50.0, "GPU charge took \(String(format: "%.1f", ms))ms, budget 50ms")
    }
}
