import Foundation
import Testing
import simd
import CEngramSceneTypes
@preconcurrency import Metal
import EngramMetalShaders
import EngramSceneKit

/// Tests that the O(n²) brute-force charge kernel does NOT fire at 8K+ nodes.
/// This is the root cause of GPU starvation → 1-second drawable blocks → laptop crash.
/// At 8K nodes, brute force = 64M iterations per command buffer, starving the render queue.
@Suite("Brute Force Guard")
struct BruteForceGuardTests {

    static let device: MTLDevice? = MTLCreateSystemDefaultDevice()

    // MARK: - GPULog captures force compute decisions

    @Test("GPULog.configure + log produces output")
    func testGPULogWorks() throws {
        let tmpPath = NSTemporaryDirectory() + "gpulog-test-\(UUID().uuidString).log"
        GPULog.configure(path: tmpPath)
        GPULog.log("TEST MESSAGE")

        let contents = try String(contentsOfFile: tmpPath)
        #expect(contents.contains("TEST MESSAGE"), "GPULog should write to configured file")
        #expect(contents.contains("APP LAUNCH"), "GPULog.configure should write launch header")
        try? FileManager.default.removeItem(atPath: tmpPath)
    }

    // MARK: - Brute force guard at scale

    @Test("dispatchForces at 8K nodes: brute force must NOT fire")
    @MainActor func testBruteForceGuard8K() async throws {
        let device = try #require(Self.device)
        let library = try EngramMetalShaders.makeLibrary(device: device)
        let fc = try #require(MetalForceCompute(device: device, library: library))
        let forceEngine = ForceEngine(forceCompute: fc)
        let queue = try #require(device.makeCommandQueue())
        #expect(fc.isFullSimAvailable)

        // Configure GPULog to a temp file so we can read what happened
        let tmpPath = NSTemporaryDirectory() + "gpulog-guard-\(UUID().uuidString).log"
        GPULog.configure(path: tmpPath)

        let n = 8000
        let x = (0..<n).map { _ in Float.random(in: -500...500) }
        let y = (0..<n).map { _ in Float.random(in: -500...500) }
        let z = (0..<n).map { _ in Float.random(in: -500...500) }
        let projectGroups = (0..<n).map { $0 % 15 }
        let topicGroups = (0..<n).map { $0 % 247 }
        let galaxyGroups = [Int](repeating: 0, count: n)

        var edges: [(Int, Int)] = []
        for _ in 0..<100_000 {
            let s = Int.random(in: 0..<n)
            var t = Int.random(in: 0..<n)
            while t == s { t = Int.random(in: 0..<n) }
            edges.append((s, t))
        }

        let snapshot = ForceEngine.SimulationSnapshot(
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

        // Dispatch forces — this is the exact code path the app uses
        let _: ForceResult = await withCheckedContinuation { cont in
            forceEngine.encodeForcePass(
                queue: queue,
                snapshot: snapshot
            ) { result in
                cont.resume(returning: result)
            }
        }

        // Read the log and verify brute force did NOT fire
        let log = try String(contentsOfFile: tmpPath)
        print("[test:log] GPU log contents:\n\(log)")

        #expect(!log.contains("BRUTE charge"),
                "Brute force O(n²) must NOT fire at n=\(n) — this crashes the laptop")
        #expect(log.contains("DISPATCH n=\(n)"),
                "dispatchForces should log DISPATCH")
        // Either BH tree or SKIPPED is acceptable — brute force is the danger
        let hasBH = log.contains("BH charge")
        let hasSkipped = log.contains("SKIPPED charge")
        #expect(hasBH || hasSkipped,
                "At n=\(n), charge should use BH tree or be skipped (not brute force)")

        try? FileManager.default.removeItem(atPath: tmpPath)
    }

    // MARK: - Brute force IS allowed for small n

    @Test("dispatchForces at 1K nodes: brute force is OK")
    @MainActor func testBruteForceAllowed1K() async throws {
        let device = try #require(Self.device)
        let library = try EngramMetalShaders.makeLibrary(device: device)
        let fc = try #require(MetalForceCompute(device: device, library: library))
        let forceEngine = ForceEngine(forceCompute: fc)
        let queue = try #require(device.makeCommandQueue())

        let tmpPath = NSTemporaryDirectory() + "gpulog-small-\(UUID().uuidString).log"
        GPULog.configure(path: tmpPath)

        let n = 1000
        let x = (0..<n).map { _ in Float.random(in: -500...500) }
        let y = (0..<n).map { _ in Float.random(in: -500...500) }
        let z = (0..<n).map { _ in Float.random(in: -500...500) }
        let projectGroups = (0..<n).map { $0 % 5 }
        let topicGroups = (0..<n).map { $0 % 20 }
        let galaxyGroups = [Int](repeating: 0, count: n)

        let snapshot = ForceEngine.SimulationSnapshot(
            nodeCount: n, isSettled: false,
            posX: x, posY: y, posZ: z,
            projectGroups: projectGroups, topicGroups: topicGroups,
            galaxyGroups: galaxyGroups, galaxyCenters: [SIMD3<Float>.zero],
            edgeIndices: [], topologyDirty: true,
            chargeStrength: 500, crossChargeMultiplier: 3.0,
            sameTopicChargeScale: 0.35, sameProjectChargeScale: 1.0,
            springLength: 240, crossProjectSpringLength: 400, springStrength: 0.0004,
            cohesionStrength: 0.0015, centroidRepulsion: 2500,
            topicCohesionStrength: 0.009, topicCentroidRepulsion: 3500,
            centerStrength: 0.006, center: .zero,
            alpha: 1.0, damping: 0.78, maxSpeed: 12.0
        )

        let _: ForceResult = await withCheckedContinuation { cont in
            forceEngine.encodeForcePass(
                queue: queue,
                snapshot: snapshot
            ) { result in
                cont.resume(returning: result)
            }
        }

        let log = try String(contentsOfFile: tmpPath)
        print("[test:log] Small-n GPU log:\n\(log)")

        #expect(log.contains("BRUTE charge: n=\(n)"),
                "Brute force should be used at n=\(n) (safe for small counts)")
        #expect(log.contains("DISPATCH n=\(n)"))

        try? FileManager.default.removeItem(atPath: tmpPath)
    }
}
