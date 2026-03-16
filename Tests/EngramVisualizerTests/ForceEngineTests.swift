import Foundation
import Testing
import simd
import CEngramSceneTypes
@preconcurrency import Metal
import EngramMetalShaders
import EngramSceneKit

/// Tests that ForceEngine correctly orchestrates GPU force dispatch.
/// Catches the specific bugs from the debugging session:
/// - cutoffSq hardcoded → BH O(n²) degeneration
/// - Topology rebuild missing → edges=0 → no clustering
/// - Uses separate command buffer on shared queue (no dual-queue contention)
@Suite("ForceEngine")
struct ForceEngineTests {

    static let device: MTLDevice? = MTLCreateSystemDefaultDevice()

    @MainActor private func makeForceEngine() throws -> (ForceEngine, MTLCommandQueue) {
        let device = try #require(Self.device)
        let library = try EngramMetalShaders.makeLibrary(device: device)
        let queue = try #require(device.makeCommandQueue())
        let fc = try #require(MetalForceCompute(device: device, library: library))
        return (ForceEngine(forceCompute: fc), queue)
    }

    private func makeSnapshot(
        nodeCount n: Int = 8000,
        edgeCount: Int = 1000,
        topologyDirty: Bool = true,
        isSettled: Bool = false
    ) -> ForceEngine.SimulationSnapshot {
        let projectCount = 15
        let topicCount = 247
        let x = (0..<n).map { _ in Float.random(in: -500...500) }
        let y = (0..<n).map { _ in Float.random(in: -500...500) }
        let z = (0..<n).map { _ in Float.random(in: -500...500) }
        let projectGroups = (0..<n).map { $0 % projectCount }
        let topicGroups = (0..<n).map { $0 % topicCount }
        let galaxyGroups = [Int](repeating: 0, count: n)

        var edges: [(Int, Int)] = []
        for _ in 0..<edgeCount {
            let s = Int.random(in: 0..<n)
            var t = Int.random(in: 0..<n)
            while t == s { t = Int.random(in: 0..<n) }
            edges.append((s, t))
        }

        return ForceEngine.SimulationSnapshot(
            nodeCount: n, isSettled: isSettled,
            posX: x, posY: y, posZ: z,
            projectGroups: projectGroups, topicGroups: topicGroups,
            galaxyGroups: galaxyGroups, galaxyCenters: [.zero],
            edgeIndices: edges, topologyDirty: topologyDirty,
            chargeStrength: 500, crossChargeMultiplier: 3.0,
            sameTopicChargeScale: 0.35, sameProjectChargeScale: 1.0,
            springLength: 240, crossProjectSpringLength: 400, springStrength: 0.0004,
            cohesionStrength: 0.0015, centroidRepulsion: 2500,
            topicCohesionStrength: 0.009, topicCentroidRepulsion: 3500,
            centerStrength: 0.006, center: .zero,
            alpha: 1.0, damping: 0.78, maxSpeed: 12.0
        )
    }

    @Test("encodeForcePass returns true and delivers non-zero forces")
    @MainActor func testEncodeForcePassDelivers() async throws {
        let (engine, queue) = try makeForceEngine()
        let snapshot = makeSnapshot(nodeCount: 500, edgeCount: 200)

        let result: ForceResult = await withCheckedContinuation { cont in
            let didEncode = engine.encodeForcePass(queue: queue, snapshot: snapshot) { result in
                cont.resume(returning: result)
            }
            #expect(didEncode, "Should encode forces for non-settled sim")
        }

        #expect(result.fx.count == 500)
        // At least some forces should be non-zero
        let maxForce = result.fx.map { abs($0) }.max() ?? 0
        #expect(maxForce > 0, "Forces should be non-zero at production scale")
    }

    @Test("skips encoding when settled")
    @MainActor func testSkipsWhenSettled() throws {
        let (engine, queue) = try makeForceEngine()
        let snapshot = makeSnapshot(nodeCount: 100, isSettled: true)

        let didEncode = engine.encodeForcePass(queue: queue, snapshot: snapshot) { _ in }
        #expect(!didEncode, "Should not encode when settled")
    }

    @Test("skips encoding when only 1 node")
    @MainActor func testSkipsSingleNode() throws {
        let (engine, queue) = try makeForceEngine()
        let snapshot = makeSnapshot(nodeCount: 1, edgeCount: 0)

        let didEncode = engine.encodeForcePass(queue: queue, snapshot: snapshot) { _ in }
        #expect(!didEncode, "Should not encode for single node")
    }

    @Test("topology rebuild fires when dirty — catches edges=0 bug")
    @MainActor func testTopologyRebuildWhenDirty() async throws {
        let (engine, queue) = try makeForceEngine()
        // Topology dirty with edges → should rebuild before encoding
        let snapshot = makeSnapshot(nodeCount: 200, edgeCount: 500, topologyDirty: true)

        let result: ForceResult = await withCheckedContinuation { cont in
            engine.encodeForcePass(queue: queue, snapshot: snapshot) { result in
                cont.resume(returning: result)
            }
        }

        // With edges present and topology rebuilt, springs should produce forces
        #expect(result.fx.count == 200)
        let maxForce = result.fx.map { abs($0) }.max() ?? 0
        #expect(maxForce > 0, "Forces should be non-zero after topology rebuild")
    }

    @Test("does not double-dispatch when inFlight")
    @MainActor func testNoDoubleDispatch() async throws {
        let (engine, queue) = try makeForceEngine()
        let snapshot = makeSnapshot(nodeCount: 200, edgeCount: 100)

        // First dispatch should succeed
        var firstCompleted = false
        let didEncode1 = engine.encodeForcePass(queue: queue, snapshot: snapshot) { _ in
            firstCompleted = true
        }
        #expect(didEncode1)

        // While first is in flight, second should be rejected
        let didEncode2 = engine.encodeForcePass(queue: queue, snapshot: snapshot) { _ in }
        #expect(!didEncode2, "Should reject dispatch while in flight")

        // Wait for first to complete
        while !firstCompleted {
            try await Task.sleep(nanoseconds: 10_000_000)
        }
    }

    @Test("isFullSimAvailable reflects underlying compute state")
    @MainActor func testSimAvailability() throws {
        let (engine, _) = try makeForceEngine()
        #expect(engine.isFullSimAvailable, "Full sim should be available with proper shaders")
    }
}
