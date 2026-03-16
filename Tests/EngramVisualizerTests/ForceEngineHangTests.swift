import Foundation
import Testing
import simd
import CEngramSceneTypes
@preconcurrency import Metal
import EngramMetalShaders
import EngramSceneKit

/// Reproduces the production hang: 8K nodes, 91K edges, first encode
/// (no BH tree ready → charge skipped, springs run on full edge set).
/// The force command buffer must complete within frame budget to avoid
/// monopolizing the GPU and starving the drawable pool.
@Suite("ForceEngine Hang Guard")
struct ForceEngineHangTests {

    static let device: MTLDevice? = MTLCreateSystemDefaultDevice()

    /// Build a snapshot matching production scale (7798 nodes, 91K edges).
    private func productionSnapshot() -> ForceEngine.SimulationSnapshot {
        let n = 7798
        let edgeCount = 91_283
        let projectCount = 15
        let topicCount = 248

        let x = (0..<n).map { _ in Float.random(in: -500...500) }
        let y = (0..<n).map { _ in Float.random(in: -500...500) }
        let z = (0..<n).map { _ in Float.random(in: -500...500) }
        let projectGroups = (0..<n).map { $0 % projectCount }
        let topicGroups = (0..<n).map { $0 % topicCount }
        let galaxyGroups = [Int](repeating: 0, count: n)

        var edges: [(Int, Int)] = []
        edges.reserveCapacity(edgeCount)
        for _ in 0..<edgeCount {
            let s = Int.random(in: 0..<n)
            var t = Int.random(in: 0..<n)
            while t == s { t = Int.random(in: 0..<n) }
            edges.append((s, t))
        }

        return ForceEngine.SimulationSnapshot(
            nodeCount: n, isSettled: false,
            posX: x, posY: y, posZ: z,
            projectGroups: projectGroups, topicGroups: topicGroups,
            galaxyGroups: galaxyGroups, galaxyCenters: [.zero],
            edgeIndices: edges, topologyDirty: true,
            chargeStrength: 500, crossChargeMultiplier: 3.0,
            sameTopicChargeScale: 0.35, sameProjectChargeScale: 1.0,
            springLength: 240, crossProjectSpringLength: 400, springStrength: 0.0004,
            cohesionStrength: 0.0015, centroidRepulsion: 2500,
            topicCohesionStrength: 0.009, topicCentroidRepulsion: 3500,
            centerStrength: 0.006, center: .zero,
            alpha: 1.0, damping: 0.78, maxSpeed: 12.0
        )
    }

    @Test("Production-scale force encode completes within 200ms (catches GPU hang)")
    @MainActor func testForceEncodeDoesNotHang() async throws {
        let device = try #require(Self.device)
        let library = try EngramMetalShaders.makeLibrary(device: device)
        let queue = try #require(device.makeCommandQueue())
        let fc = try #require(MetalForceCompute(device: device, library: library))
        let engine = ForceEngine(forceCompute: fc)

        let snapshot = productionSnapshot()
        let start = CFAbsoluteTimeGetCurrent()

        let result: ForceResult = await withCheckedContinuation { cont in
            let didEncode = engine.encodeForcePass(queue: queue, snapshot: snapshot) { result in
                cont.resume(returning: result)
            }
            #expect(didEncode, "Should encode at production scale")
        }

        let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
        print("[engram:hang-guard] Production force encode: \(String(format: "%.1f", ms))ms n=\(snapshot.nodeCount) edges=\(snapshot.edgeIndices.count)")
        #expect(result.fx.count == snapshot.nodeCount)

        // Must complete within 200ms to leave GPU time for render
        // (at 60fps, 16.6ms budget; force should take <10ms, allowing margin)
        #expect(ms < 200, "Force encode took \(String(format: "%.0f", ms))ms — will hang drawable pool")
    }

    @Test("Force completion handler actually fires (catches inFlight stuck bug)")
    @MainActor func testCompletionHandlerFires() async throws {
        let device = try #require(Self.device)
        let library = try EngramMetalShaders.makeLibrary(device: device)
        let queue = try #require(device.makeCommandQueue())
        let fc = try #require(MetalForceCompute(device: device, library: library))
        let engine = ForceEngine(forceCompute: fc)

        let snapshot = productionSnapshot()

        // Dispatch forces
        var completionFired = false
        let didEncode = engine.encodeForcePass(queue: queue, snapshot: snapshot) { _ in
            completionFired = true
        }
        #expect(didEncode)
        #expect(engine.isInFlight, "Should be in flight immediately after encode")

        // Wait for completion with timeout
        let deadline = CFAbsoluteTimeGetCurrent() + 5.0  // 5 second timeout
        while !completionFired && CFAbsoluteTimeGetCurrent() < deadline {
            try await Task.sleep(nanoseconds: 10_000_000) // 10ms
        }

        #expect(completionFired, "Completion handler never fired — inFlight stuck forever")
        #expect(!engine.isInFlight, "inFlight should be cleared after completion")
    }

    @Test("Consecutive force dispatches don't accumulate (pipeline stays bounded)")
    @MainActor func testPipelineStaysBounded() async throws {
        let device = try #require(Self.device)
        let library = try EngramMetalShaders.makeLibrary(device: device)
        let queue = try #require(device.makeCommandQueue())
        let fc = try #require(MetalForceCompute(device: device, library: library))
        let engine = ForceEngine(forceCompute: fc)

        // Simulate 10 consecutive frames of force dispatch
        var times: [Double] = []
        for i in 0..<10 {
            // Use smaller scale to avoid test timeout but still meaningful
            let snapshot = ForceEngine.SimulationSnapshot(
                nodeCount: 2000, isSettled: false,
                posX: (0..<2000).map { _ in Float.random(in: -500...500) },
                posY: (0..<2000).map { _ in Float.random(in: -500...500) },
                posZ: (0..<2000).map { _ in Float.random(in: -500...500) },
                projectGroups: (0..<2000).map { $0 % 15 },
                topicGroups: (0..<2000).map { $0 % 100 },
                galaxyGroups: [Int](repeating: 0, count: 2000),
                galaxyCenters: [.zero],
                edgeIndices: (0..<5000).map { _ in (Int.random(in: 0..<2000), Int.random(in: 0..<2000)) },
                topologyDirty: i == 0,  // Only dirty on first frame
                chargeStrength: 500, crossChargeMultiplier: 3.0,
                sameTopicChargeScale: 0.35, sameProjectChargeScale: 1.0,
                springLength: 240, crossProjectSpringLength: 400, springStrength: 0.0004,
                cohesionStrength: 0.0015, centroidRepulsion: 2500,
                topicCohesionStrength: 0.009, topicCentroidRepulsion: 3500,
                centerStrength: 0.006, center: .zero,
                alpha: 1.0, damping: 0.78, maxSpeed: 12.0
            )

            let start = CFAbsoluteTimeGetCurrent()
            let _: ForceResult = await withCheckedContinuation { cont in
                engine.encodeForcePass(queue: queue, snapshot: snapshot) { result in
                    cont.resume(returning: result)
                }
            }
            times.append((CFAbsoluteTimeGetCurrent() - start) * 1000.0)
        }

        let avg = times.reduce(0, +) / Double(times.count)
        let max = times.max() ?? 0
        print("[engram:hang-guard] 10-frame pipeline: avg=\(String(format: "%.1f", avg))ms max=\(String(format: "%.1f", max))ms")
        print("[engram:hang-guard] per-frame: \(times.map { String(format: "%.0f", $0) }.joined(separator: ", "))ms")

        // No single frame should take > 500ms
        #expect(max < 500, "Frame \(times.firstIndex(of: max)!) took \(String(format: "%.0f", max))ms — will hang")
    }
}
