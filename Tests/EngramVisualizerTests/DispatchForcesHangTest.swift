import Foundation
import Testing
import simd
import CEngramSceneTypes
@preconcurrency import Metal
import EngramMetalShaders
import EngramSceneKit

/// Reproduces the EXACT app hang scenario:
/// 1. First encodeForcePass (no BH tree) — should complete fast
/// 2. Wait for BH tree to build async
/// 3. Second encodeForcePass (BH tree ready) — THIS is what hangs in the app
@Suite("DispatchForces Hang Reproduction")
struct DispatchForcesHangTest {

    static let device: MTLDevice? = MTLCreateSystemDefaultDevice()

    @Test("dispatchForces with concurrent GPU render work")
    @MainActor func testDispatchWithConcurrentRender() async throws {
        let device = try #require(Self.device)
        let library = try EngramMetalShaders.makeLibrary(device: device)
        let fc = try #require(MetalForceCompute(device: device, library: library))
        let forceEngine = ForceEngine(forceCompute: fc)
        let queue = try #require(device.makeCommandQueue())
        let renderQueue = try #require(device.makeCommandQueue())

        let n = 7798
        let x = (0..<n).map { _ in Float.random(in: -500...500) }
        let y = (0..<n).map { _ in Float.random(in: -500...500) }
        let z = (0..<n).map { _ in Float.random(in: -500...500) }
        let projectGroups = (0..<n).map { $0 % 15 }
        let topicGroups = (0..<n).map { $0 % 248 }
        let galaxyGroups = [Int](repeating: 0, count: n)

        var edges: [(Int, Int)] = []
        for _ in 0..<91283 {
            edges.append((Int.random(in: 0..<n), Int.random(in: 0..<n)))
        }

        let snapshot1 = ForceEngine.SimulationSnapshot(
            nodeCount: n, isSettled: false,
            posX: x, posY: y, posZ: z,
            projectGroups: projectGroups, topicGroups: topicGroups,
            galaxyGroups: galaxyGroups, galaxyCenters: [SIMD3<Float>.zero],
            edgeIndices: edges, topologyDirty: true,
            chargeStrength: 500, crossChargeMultiplier: 3.0,
            sameTopicChargeScale: 0.35, sameProjectChargeScale: 1.0,
            springLength: 240, crossProjectSpringLength: 400, springStrength: 0.0004,
            cohesionStrength: 0.0015, centroidRepulsion: 2500,
            topicCohesionStrength: 0.009, topicCentroidRepulsion: 3500,
            centerStrength: 0.006, center: .zero,
            alpha: 1.0, damping: 0.78, maxSpeed: 12.0
        )

        // First dispatch to build BH tree
        let _: ForceResult = await withCheckedContinuation { cont in
            forceEngine.encodeForcePass(
                queue: queue,
                snapshot: snapshot1
            ) { result in cont.resume(returning: result) }
        }
        try await Task.sleep(nanoseconds: 500_000_000)

        // Now simulate concurrent render work on a SEPARATE queue
        // while dispatching forces (BH tree ready)
        print("[test] Starting concurrent render + force dispatch")

        // Submit render-like compute work on the render queue continuously
        nonisolated(unsafe) var renderRunning = true
        let packNodePL: MTLComputePipelineState = {
            let fn = library.makeFunction(name: "pack_node_instances")!
            return try! device.makeComputePipelineState(function: fn)
        }()
        let dummyBuf = device.makeBuffer(length: n * 64, options: .storageModeShared)!
        let paramsBuf = device.makeBuffer(length: 64, options: .storageModeShared)!

        // Background render loop — submits work on renderQueue like the app does
        Task.detached { @Sendable in
            var frame = 0
            while renderRunning {
                guard let cmdBuf = renderQueue.makeCommandBuffer(),
                      let enc = cmdBuf.makeComputeCommandEncoder() else { break }
                enc.setComputePipelineState(packNodePL)
                enc.setBuffer(dummyBuf, offset: 0, index: 0)
                enc.setBuffer(dummyBuf, offset: 0, index: 1)
                enc.setBuffer(dummyBuf, offset: 0, index: 2)
                enc.setBuffer(dummyBuf, offset: 0, index: 3)
                enc.setBuffer(dummyBuf, offset: 0, index: 4)
                enc.setBuffer(paramsBuf, offset: 0, index: 5)
                enc.setBuffer(dummyBuf, offset: 0, index: 6)
                enc.dispatchThreadgroups(
                    MTLSize(width: (n + 255) / 256, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.endEncoding()
                cmdBuf.commit()
                await cmdBuf.completed()
                frame += 1
                if frame % 100 == 0 { print("[test] render frame \(frame)") }
            }
        }

        // Force dispatch with BH tree while render is running
        let snapshot2 = ForceEngine.SimulationSnapshot(
            nodeCount: n, isSettled: false,
            posX: x, posY: y, posZ: z,
            projectGroups: projectGroups, topicGroups: topicGroups,
            galaxyGroups: galaxyGroups, galaxyCenters: [SIMD3<Float>.zero],
            edgeIndices: edges, topologyDirty: false,
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
                snapshot: snapshot2
            ) { result in cont.resume(returning: result) }
        }
        renderRunning = false
        let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
        print("[test] Force dispatch with concurrent render: \(String(format: "%.1f", ms))ms")
        #expect(result.fx.count == n)
        #expect(ms < 2000, "Force dispatch hung with concurrent render: \(String(format: "%.0f", ms))ms")
    }

    @Test("Two consecutive dispatchForces calls — second with BH tree")
    @MainActor func testTwoConsecutiveDispatches() async throws {
        let device = try #require(Self.device)
        let library = try EngramMetalShaders.makeLibrary(device: device)
        let fc = try #require(MetalForceCompute(device: device, library: library))
        let forceEngine = ForceEngine(forceCompute: fc)
        let queue = try #require(device.makeCommandQueue())
        #expect(fc.isFullSimAvailable)

        let n = 7798
        let edgeCount = 91283
        let x = (0..<n).map { _ in Float.random(in: -500...500) }
        let y = (0..<n).map { _ in Float.random(in: -500...500) }
        let z = (0..<n).map { _ in Float.random(in: -500...500) }
        let projectGroups = (0..<n).map { $0 % 15 }
        let topicGroups = (0..<n).map { $0 % 248 }
        let galaxyGroups = [Int](repeating: 0, count: n)

        var edges: [(Int, Int)] = []
        edges.reserveCapacity(edgeCount)
        for _ in 0..<edgeCount {
            let s = Int.random(in: 0..<n)
            var t = Int.random(in: 0..<n)
            while t == s { t = Int.random(in: 0..<n) }
            edges.append((s, t))
        }

        // --- First dispatch (no BH tree — charge skipped, springs+centroids only) ---
        print("[test] Dispatch 1: no BH tree")
        let snapshot1 = ForceEngine.SimulationSnapshot(
            nodeCount: n, isSettled: false,
            posX: x, posY: y, posZ: z,
            projectGroups: projectGroups, topicGroups: topicGroups,
            galaxyGroups: galaxyGroups, galaxyCenters: [SIMD3<Float>.zero],
            edgeIndices: edges, topologyDirty: true,
            chargeStrength: 500, crossChargeMultiplier: 3.0,
            sameTopicChargeScale: 0.35, sameProjectChargeScale: 1.0,
            springLength: 240, crossProjectSpringLength: 400, springStrength: 0.0004,
            cohesionStrength: 0.0015, centroidRepulsion: 2500,
            topicCohesionStrength: 0.009, topicCentroidRepulsion: 3500,
            centerStrength: 0.006, center: .zero,
            alpha: 1.0, damping: 0.78, maxSpeed: 12.0
        )

        let start1 = CFAbsoluteTimeGetCurrent()
        let result1: ForceResult = await withCheckedContinuation { cont in
            forceEngine.encodeForcePass(
                queue: queue,
                snapshot: snapshot1
            ) { result in
                cont.resume(returning: result)
            }
        }
        let ms1 = (CFAbsoluteTimeGetCurrent() - start1) * 1000.0
        print("[test] Dispatch 1 done: \(String(format: "%.1f", ms1))ms, forces=\(result1.fx.count)")
        #expect(result1.fx.count == n)

        // --- Wait for async BH tree to build ---
        print("[test] Waiting for BH tree async build...")
        try await Task.sleep(nanoseconds: 500_000_000) // 500ms

        // --- Second dispatch (BH tree should be ready now) ---
        print("[test] Dispatch 2: BH tree should be ready")
        let snapshot2 = ForceEngine.SimulationSnapshot(
            nodeCount: n, isSettled: false,
            posX: x, posY: y, posZ: z,
            projectGroups: projectGroups, topicGroups: topicGroups,
            galaxyGroups: galaxyGroups, galaxyCenters: [SIMD3<Float>.zero],
            edgeIndices: edges, topologyDirty: false,
            chargeStrength: 500, crossChargeMultiplier: 3.0,
            sameTopicChargeScale: 0.35, sameProjectChargeScale: 1.0,
            springLength: 240, crossProjectSpringLength: 400, springStrength: 0.0004,
            cohesionStrength: 0.0015, centroidRepulsion: 2500,
            topicCohesionStrength: 0.009, topicCentroidRepulsion: 3500,
            centerStrength: 0.006, center: .zero,
            alpha: 1.0, damping: 0.78, maxSpeed: 12.0
        )

        let start2 = CFAbsoluteTimeGetCurrent()
        let result2: ForceResult = await withCheckedContinuation { cont in
            forceEngine.encodeForcePass(
                queue: queue,
                snapshot: snapshot2
            ) { result in
                cont.resume(returning: result)
            }
        }
        let ms2 = (CFAbsoluteTimeGetCurrent() - start2) * 1000.0
        print("[test] Dispatch 2 done: \(String(format: "%.1f", ms2))ms, forces=\(result2.fx.count)")
        #expect(result2.fx.count == n)
        #expect(ms2 < 2000, "Second dispatch (BH) took \(String(format: "%.0f", ms2))ms — GPU hang")
    }
}
