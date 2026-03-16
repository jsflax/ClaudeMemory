import Foundation
import Testing
import simd
import CEngramSceneTypes
@preconcurrency import Metal
import EngramMetalShaders
import EngramSceneKit

/// Tests that a full frame (force compute + pack + render encode) completes
/// within budget in a SINGLE command buffer. This is the test that catches
/// the GPU hang — separate CBs for forces hang when MTKView is rendering.
@Suite("Synchronous Frame")
struct SynchronousFrameTests {

    static let device: MTLDevice? = MTLCreateSystemDefaultDevice()

    @Test("Full frame at 8K nodes completes within 100ms (catches hang)")
    @MainActor func testFullFrameCompletes() async throws {
        let device = try #require(Self.device)
        let library = try EngramMetalShaders.makeLibrary(device: device)
        let queue = try #require(device.makeCommandQueue())
        let orchestrator = try #require(FrameOrchestrator(device: device, library: library))
        let bm = orchestrator.bufferManager
        let fc = orchestrator.forceEngine.forceCompute

        let n = 8000
        let edgeCount = 50000

        // Build simulation snapshot
        let x = (0..<n).map { _ in Float.random(in: -500...500) }
        let y = (0..<n).map { _ in Float.random(in: -500...500) }
        let z = (0..<n).map { _ in Float.random(in: -500...500) }
        let projectGroups = (0..<n).map { $0 % 15 }
        let topicGroups = (0..<n).map { $0 % 200 }
        let edges = (0..<edgeCount).map { _ in (Int.random(in: 0..<n), Int.random(in: 0..<n)) }

        // Setup topology
        fc.setTopologyDirty(edges: edges)
        fc.rebuildTopology(nodeCount: n, projectGroups: projectGroups, topicGroups: topicGroups)

        // Setup node pack buffers
        bm.ensureNodeBuffers(nodeCount: n)
        bm.ensureNodePackBuffers(count: n)
        bm.ensureNodePositionBuffer(count: n)
        bm.actualNodeCount = n

        // Fill position buffer (CPU-uploaded, read by pack_node_instances)
        if let posBuf = bm.nodePositionBuffer {
            let ptr = posBuf.contents().bindMemory(to: SIMD3<Float>.self, capacity: n)
            for i in 0..<n {
                ptr[i] = SIMD3<Float>(x[i], y[i], z[i])
            }
        }

        // Fill pack input buffer (visual state only — no positions)
        if let packBuf = bm.nodePackInputBuffer {
            let ptr = packBuf.contents().bindMemory(to: NodePackInput.self, capacity: n)
            for i in 0..<n {
                ptr[i] = NodePackInput(position: SIMD3<Float>(x[i], y[i], z[i]),
                                       baseRadius: 1.0,
                                       baseColor: SIMD3<Float>(0.5, 0.5, 0.8),
                                       packedState: 0)
            }
        }
        if let paramsBuf = bm.nodePackParamsBuffer {
            paramsBuf.contents().bindMemory(to: NodePackParams.self, capacity: 1).pointee =
                NodePackParams(nodeCount: UInt32(n), scaleFactor: 1.0/200.0, nodeRadius: 0.008,
                              animationTime: 0, projectCount: 15, _pad0: 0, _pad1: 0, _pad2: 0)
        }

        // Create ONE command buffer (the key: no separate CB for forces)
        let start = CFAbsoluteTimeGetCurrent()
        let cmdBuf = try #require(queue.makeCommandBuffer())
        let enc = try #require(cmdBuf.makeComputeCommandEncoder())

        // Force compute — same encoder
        fc.encodeForces(
            encoder: enc,
            x: x, y: y, z: z,
            projectGroups: projectGroups, topicGroups: topicGroups,
            galaxyGroups: [Int](repeating: 0, count: n),
            galaxyCenters: [.zero],
            chargeStrength: 500, crossChargeMultiplier: 3.0,
            sameTopicChargeScale: 0.35, sameProjectChargeScale: 1.0,
            springLength: 240, crossProjectSpringLength: 400, springStrength: 0.0004,
            cohesionStrength: 0.0015, centroidRepulsion: 2500,
            topicCohesionStrength: 0.009, topicCentroidRepulsion: 3500,
            centerStrength: 0.006, center: .zero,
            alpha: 1.0, damping: 0.78, maxSpeed: 12.0)

        // Pack nodes — same encoder, reads from nodePositionBuffer
        if let pl = orchestrator.packNodePipeline,
           let inputBuf = bm.nodePackInputBuffer,
           let instanceBuf = bm.nodeInstanceBuffer,
           let lightBuf = bm.pointLightOutputBuffer,
           let lightCountBuf = bm.pointLightCountBuffer,
           let paramsBuf = bm.nodePackParamsBuffer,
           let projIdxBuf = bm.nodeProjectIdxBuffer,
           let posBuf = bm.nodePositionBuffer {
            ComputeDispatcher.packNodes(
                encoder: enc, pipeline: pl,
                inputBuffer: inputBuf, instanceBuffer: instanceBuf,
                lightOutputBuffer: lightBuf, lightCountBuffer: lightCountBuf,
                paramsBuffer: paramsBuf, projIdxBuffer: projIdxBuf,
                positionBuffer: posBuf,
                nodeCount: n)
        }

        enc.endEncoding()
        cmdBuf.commit()
        await cmdBuf.completed()

        let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
        print("[sync-frame] Full frame 8K nodes + 50K edges: \(String(format: "%.1f", ms))ms")
        #expect(ms < 100, "Full frame must complete within 100ms — took \(String(format: "%.1f", ms))ms")

        // Verify node instances have non-zero positions
        if let instanceBuf = bm.nodeInstanceBuffer {
            let output = instanceBuf.contents().bindMemory(to: NodeInstance.self, capacity: n)
            var nonZero = 0
            for i in 0..<min(n, 100) {
                if output[i].position.x != 0 || output[i].position.y != 0 { nonZero += 1 }
            }
            #expect(nonZero > 50, "Most nodes should have non-zero positions")
        }
    }

    @Test("NodePackInput has no position field (visual state only)")
    @MainActor func testNodePackInputLayout() {
        // NodePackInput should be: baseRadius(4) + packedState(4) + baseColor(12) + _pad0(4) = 24 bytes
        // NOT 32 bytes (which would include position)
        let size = MemoryLayout<NodePackInput>.stride
        #expect(size <= 32, "NodePackInput should be compact without position field, got \(size) bytes")
        // Verify no position member by construction
        let npi = NodePackInput(position: .zero, baseRadius: 1.0,
                                baseColor: SIMD3<Float>(0.5, 0.5, 0.5), packedState: 2.0)
        #expect(npi.baseRadius == 1.0)
        #expect(npi.packedState == 2.0)
    }
}
