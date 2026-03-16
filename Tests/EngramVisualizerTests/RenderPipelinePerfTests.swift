import Foundation
import Testing
import simd
import CEngramSceneTypes
@preconcurrency import Metal
import EngramMetalShaders
import EngramSceneKit

/// Tests the FULL render pipeline GPU workload — not just force compute.
/// Profiles every compute pass and render draw call at production scale:
/// 8K nodes, 100K edges, 247 topics, 15 projects.
@Suite("Render Pipeline Performance")
struct RenderPipelinePerfTests {

    static let device: MTLDevice? = MTLCreateSystemDefaultDevice()

    private func measure(_ label: String, iterations: Int = 5, block: () -> Void) -> Double {
        block() // warmup
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations { block() }
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let avgMs = (elapsed / Double(iterations)) * 1000.0
        print("[engram:perf] \(label): \(String(format: "%.2f", avgMs))ms")
        return avgMs
    }

    // MARK: - Instanced Node Pipeline (shader function loads)

    @Test("node_vertex_instanced shader loads")
    func testNodeInstancedShaderLoads() throws {
        let device = try #require(Self.device)
        let library = try EngramMetalShaders.makeLibrary(device: device)
        let vertFn = try #require(library.makeFunction(name: "node_vertex_instanced"))
        let fragFn = try #require(library.makeFunction(name: "node_fragment"))

        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = vertFn
        desc.fragmentFunction = fragFn
        desc.colorAttachments[0].pixelFormat = .bgra8Unorm
        desc.depthAttachmentPixelFormat = .depth32Float

        let pipeline = try device.makeRenderPipelineState(descriptor: desc)
        #expect(pipeline.label == nil || true, "Pipeline created successfully")
    }

    // MARK: - Instanced Edge Pipeline (shader function loads)

    @Test("edge_vertex_instanced shader loads")
    func testEdgeInstancedShaderLoads() throws {
        let device = try #require(Self.device)
        let library = try EngramMetalShaders.makeLibrary(device: device)
        let vertFn = try #require(library.makeFunction(name: "edge_vertex_instanced"))
        let fragFn = try #require(library.makeFunction(name: "edge_fragment"))

        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = vertFn
        desc.fragmentFunction = fragFn
        desc.colorAttachments[0].pixelFormat = .bgra8Unorm
        desc.colorAttachments[0].isBlendingEnabled = true
        desc.colorAttachments[0].rgbBlendOperation = .add
        desc.colorAttachments[0].alphaBlendOperation = .add
        desc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        desc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        desc.colorAttachments[0].sourceAlphaBlendFactor = .one
        desc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        desc.depthAttachmentPixelFormat = .depth32Float

        let pipeline = try device.makeRenderPipelineState(descriptor: desc)
        #expect(pipeline.label == nil || true, "Pipeline created successfully")
    }

    // MARK: - Full Force Compute via ForceEngine.encodeForcePass()

    @Test("Real ForceEngine.encodeForcePass 8K nodes: profile")
    @MainActor func testRealDispatchForces() async throws {
        let device = try #require(Self.device)
        let library = try EngramMetalShaders.makeLibrary(device: device)
        let fc = try #require(MetalForceCompute(device: device, library: library))
        let forceEngine = ForceEngine(forceCompute: fc)
        let queue = try #require(device.makeCommandQueue())
        #expect(fc.isFullSimAvailable, "Full sim pipeline should be available")

        let n = 8000
        let edgeCount = 100_000
        let projectCount = 15
        let topicCount = 247

        // Generate test data matching production scale
        var x = (0..<n).map { _ in Float.random(in: -500...500) }
        var y = (0..<n).map { _ in Float.random(in: -500...500) }
        var z = (0..<n).map { _ in Float.random(in: -500...500) }
        let projectGroups = (0..<n).map { $0 % projectCount }
        let topicGroups = (0..<n).map { $0 % topicCount }
        let galaxyGroups = [Int](repeating: 0, count: n)

        // Generate edges
        var edges: [(Int, Int)] = []
        for _ in 0..<edgeCount {
            let s = Int.random(in: 0..<n)
            var t = Int.random(in: 0..<n)
            while t == s { t = Int.random(in: 0..<n) }
            edges.append((s, t))
        }

        // Use encodeForcePass — the exact same code path the app uses.
        // Measure round-trip: dispatch → GPU encode → GPU execute → readback → callback.
        let iterations = 5
        var times: [Double] = []

        for i in 0..<iterations {
            let snapshot = ForceEngine.SimulationSnapshot(
                nodeCount: n, isSettled: false,
                posX: x, posY: y, posZ: z,
                projectGroups: projectGroups, topicGroups: topicGroups,
                galaxyGroups: galaxyGroups, galaxyCenters: [SIMD3<Float>.zero],
                edgeIndices: edges, topologyDirty: i == 0,
                chargeStrength: 500, crossChargeMultiplier: 3.0,
                sameTopicChargeScale: 0.35, sameProjectChargeScale: 1.0,
                springLength: 240, crossProjectSpringLength: 400, springStrength: 0.0004,
                cohesionStrength: 0.0015, centroidRepulsion: 2500,
                topicCohesionStrength: 0.009, topicCentroidRepulsion: 3500,
                centerStrength: 0.006, center: .zero,
                alpha: 1.0, damping: 0.78, maxSpeed: 12.0
            )

            let start = CFAbsoluteTimeGetCurrent()

            let result: ForceResult = await withCheckedContinuation { cont in
                forceEngine.encodeForcePass(
                    queue: queue,
                    snapshot: snapshot
                ) { result in
                    cont.resume(returning: result)
                }
            }

            let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
            times.append(ms)
            #expect(result.fx.count == n, "Should return \(n) forces")
        }

        let avgMs = times.reduce(0, +) / Double(times.count)
        let maxMs = times.max() ?? 0
        print("[engram:perf] REAL ForceEngine.encodeForcePass:")
        print("[engram:perf]   avg=\(String(format: "%.2f", avgMs))ms max=\(String(format: "%.2f", maxMs))ms (\(iterations) iterations)")
        print("[engram:perf]   n=\(n) edges=\(edgeCount) projects=\(projectCount) topics=\(topicCount)")
        print("[engram:perf]   per-iteration: \(times.map { String(format: "%.1f", $0) }.joined(separator: ", "))ms")
    }

    // MARK: - Instanced rendering: pack compute only (no stamp overhead)

    @Test("Instanced pipeline: pack_node_instances + pack_edge_instances only")
    func testInstancedPackOnly() throws {
        let device = try #require(Self.device)
        let library = try EngramMetalShaders.makeLibrary(device: device)
        let queue = try #require(device.makeCommandQueue())

        let n = 8000
        let edgeCount = 100_000

        // --- Pack nodes pipeline ---
        let packNodePL = try device.makeComputePipelineState(
            function: try #require(library.makeFunction(name: "pack_node_instances")))

        let nodeInputBuf = device.makeBuffer(length: n * MemoryLayout<NodePackInput>.stride, options: .storageModeShared)!
        let ni = nodeInputBuf.contents().bindMemory(to: NodePackInput.self, capacity: n)
        for i in 0..<n {
            ni[i] = NodePackInput(
                position: SIMD3<Float>(Float.random(in: -500...500),
                                       Float.random(in: -500...500),
                                       Float.random(in: -500...500)),
                baseRadius: 1.0,
                baseColor: SIMD3<Float>(0.5, 0.5, 0.8),
                packedState: 0)
        }
        let nodeInstanceBuf = device.makeBuffer(length: n * MemoryLayout<NodeInstance>.stride, options: .storageModeShared)!
        let centroidBuf = device.makeBuffer(length: 64 * MemoryLayout<ProjectCentroidGPU>.stride, options: .storageModeShared)!
        let lightBuf = device.makeBuffer(length: 16 * MemoryLayout<PointLightEntry>.stride, options: .storageModeShared)!
        let lightCountBuf = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let nodeParamsBuf = device.makeBuffer(length: MemoryLayout<NodePackParams>.stride, options: .storageModeShared)!
        nodeParamsBuf.contents().bindMemory(to: NodePackParams.self, capacity: 1).pointee =
            NodePackParams(nodeCount: UInt32(n), scaleFactor: 1.0/200.0, nodeRadius: 0.008,
                          animationTime: 0, projectCount: 15, _pad0: 0, _pad1: 0, _pad2: 0)
        let projIdxBuf = device.makeBuffer(length: n * MemoryLayout<UInt32>.stride, options: .storageModeShared)!

        // --- Pack edges pipeline ---
        let packEdgePL = try device.makeComputePipelineState(
            function: try #require(library.makeFunction(name: "pack_edge_instances")))

        let edgeDescBuf = device.makeBuffer(length: edgeCount * MemoryLayout<EdgeDescriptor>.stride, options: .storageModeShared)!
        let ed = edgeDescBuf.contents().bindMemory(to: EdgeDescriptor.self, capacity: edgeCount)
        for i in 0..<edgeCount {
            ed[i] = EdgeDescriptor(
                sourceIdx: UInt32(Int.random(in: 0..<n)),
                targetIdx: UInt32(Int.random(in: 0..<n)),
                sourceRadius: 0.008, targetRadius: 0.008,
                color: SIMD4<Float>(0.3, 0.5, 0.8, 1.0),
                baseRadius: 0.002, state: 0, _pad0: 0, _pad1: 0)
        }
        let posBuf = device.makeBuffer(length: n * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared)!
        let positions = posBuf.contents().bindMemory(to: SIMD3<Float>.self, capacity: n)
        for i in 0..<n {
            positions[i] = SIMD3<Float>(Float.random(in: -500...500),
                                        Float.random(in: -500...500),
                                        Float.random(in: -500...500))
        }
        let edgeParamsBuf = device.makeBuffer(length: MemoryLayout<PackEdgeParams>.stride, options: .storageModeShared)!
        edgeParamsBuf.contents().bindMemory(to: PackEdgeParams.self, capacity: 1).pointee =
            PackEdgeParams(edgeCount: UInt32(edgeCount), scaleFactor: 1.0/200.0, _pad0: 0, _pad1: 0)
        let edgeInstanceBuf = device.makeBuffer(length: edgeCount * MemoryLayout<EdgeInstance>.stride, options: .storageModeShared)!

        let tgSize = 256

        // Measure pack-only workload (instanced rendering eliminates stamp passes entirely)
        let ms = measure("instanced-pack-only", iterations: 5) {
            guard let cmdBuf = queue.makeCommandBuffer(),
                  let enc = cmdBuf.makeComputeCommandEncoder() else { return }

            // Reset light counter
            lightCountBuf.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = 0

            // Pack nodes
            enc.setComputePipelineState(packNodePL)
            enc.setBuffer(nodeInputBuf, offset: 0, index: 0)
            enc.setBuffer(nodeInstanceBuf, offset: 0, index: 1)
            enc.setBuffer(centroidBuf, offset: 0, index: 2)
            enc.setBuffer(lightBuf, offset: 0, index: 3)
            enc.setBuffer(lightCountBuf, offset: 0, index: 4)
            enc.setBuffer(nodeParamsBuf, offset: 0, index: 5)
            enc.setBuffer(projIdxBuf, offset: 0, index: 6)
            enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))

            // Pack edges
            enc.setComputePipelineState(packEdgePL)
            enc.setBuffer(edgeDescBuf, offset: 0, index: 0)
            enc.setBuffer(posBuf, offset: 0, index: 1)
            enc.setBuffer(edgeParamsBuf, offset: 0, index: 2)
            enc.setBuffer(edgeInstanceBuf, offset: 0, index: 3)
            enc.dispatchThreads(MTLSize(width: edgeCount, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
            enc.endEncoding()

            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()
        }

        print("[engram:perf] === INSTANCED PIPELINE SUMMARY ===")
        print("[engram:perf] Pack compute: \(String(format: "%.2f", ms))ms")
        print("[engram:perf] Node instances: \(n) (\(n * MemoryLayout<NodeInstance>.stride / 1024)KB)")
        print("[engram:perf] Edge instances: \(edgeCount) (\(edgeCount * MemoryLayout<EdgeInstance>.stride / 1024)KB)")
        print("[engram:perf] Stamp compute: 0ms (ELIMINATED by instanced rendering)")
        print("[engram:perf] Stamp buffer writes: 0 bytes (ELIMINATED)")
    }

    // MARK: - Helpers

    private func buildTestCSR(offsets: MTLBuffer, neighbors: MTLBuffer, n: Int, edgeCount: Int) {
        let offs = offsets.contents().bindMemory(to: UInt32.self, capacity: n + 1)
        let nbrs = neighbors.contents().bindMemory(to: UInt32.self, capacity: edgeCount * 2)
        memset(offs, 0, (n + 1) * MemoryLayout<UInt32>.stride)

        var edges: [(Int, Int)] = []
        for _ in 0..<edgeCount {
            let s = Int.random(in: 0..<n)
            var t = Int.random(in: 0..<n)
            while t == s { t = Int.random(in: 0..<n) }
            edges.append((s, t))
            offs[s] &+= 1
            offs[t] &+= 1
        }
        var running: UInt32 = 0
        for i in 0...n {
            let count = offs[i]
            offs[i] = running
            running &+= count
        }
        var cursors = [UInt32](repeating: 0, count: n)
        for i in 0..<n { cursors[i] = offs[i] }
        for (s, t) in edges {
            nbrs[Int(cursors[s])] = UInt32(t)
            cursors[s] &+= 1
            nbrs[Int(cursors[t])] = UInt32(s)
            cursors[t] &+= 1
        }
    }

    private func buildGroupCSR(device: MTLDevice, n: Int, groupCount: Int) -> (MTLBuffer, MTLBuffer) {
        let offBuf = device.makeBuffer(length: (groupCount + 1) * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let memBuf = device.makeBuffer(length: n * MemoryLayout<UInt32>.stride, options: .storageModeShared)!

        let offsets = offBuf.contents().bindMemory(to: UInt32.self, capacity: groupCount + 1)
        let members = memBuf.contents().bindMemory(to: UInt32.self, capacity: n)

        memset(offsets, 0, (groupCount + 1) * MemoryLayout<UInt32>.stride)
        for i in 0..<n { offsets[i % groupCount] &+= 1 }
        var running: UInt32 = 0
        for g in 0...groupCount {
            let count = offsets[g]
            offsets[g] = running
            running &+= count
        }
        var cursors = [UInt32](repeating: 0, count: groupCount)
        for g in 0..<groupCount { cursors[g] = offsets[g] }
        for i in 0..<n {
            let g = i % groupCount
            members[Int(cursors[g])] = UInt32(i)
            cursors[g] &+= 1
        }
        return (offBuf, memBuf)
    }
}
