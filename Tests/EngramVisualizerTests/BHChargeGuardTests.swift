import Foundation
import Testing
import simd
import CEngramSceneTypes
@preconcurrency import Metal
import EngramMetalShaders
import EngramSceneKit

/// Tests that the GPU Barnes-Hut charge kernel completes in reasonable time
/// at production scale. Previous bug: hardcoded cutoffSq=250000 with graph
/// span ~500 → every node within cutoff → BH degenerates to O(n²) → 1s GPU hang.
@Suite("BH Charge Performance Guard")
struct BHChargeGuardTests {

    static let device: MTLDevice? = MTLCreateSystemDefaultDevice()

    /// Dispatch forces repeatedly until BH charge fires (tree builds async).
    /// Returns (result, elapsed_ms, log_contents).
    @MainActor
    private func dispatchUntilBH(
        forceEngine: ForceEngine,
        queue: MTLCommandQueue,
        x: [Float], y: [Float], z: [Float],
        projectGroups: [Int], topicGroups: [Int],
        logPath: String,
        maxAttempts: Int = 20
    ) async throws -> (ForceResult, Double, String) {
        let n = x.count
        let galaxyGroups = [Int](repeating: 0, count: n)

        for attempt in 0..<maxAttempts {
            let snapshot = ForceEngine.SimulationSnapshot(
                nodeCount: n, isSettled: false,
                posX: x, posY: y, posZ: z,
                projectGroups: projectGroups, topicGroups: topicGroups,
                galaxyGroups: galaxyGroups, galaxyCenters: [SIMD3<Float>.zero],
                edgeIndices: [], topologyDirty: attempt == 0,
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
                ) { result in cont.resume(returning: result) }
            }
            let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000.0
            let log = (try? String(contentsOfFile: logPath)) ?? ""

            if log.contains("BH charge") {
                return (result, ms, log)
            }
            // Tree still building — yield to let async Task.detached complete
            await Task.yield()
        }
        let log = (try? String(contentsOfFile: logPath)) ?? ""
        throw BHTestError.treeNeverBuilt(log)
    }

    enum BHTestError: Error { case treeNeverBuilt(String) }

    // MARK: - BH kernel must complete at 8K nodes without hanging

    @Test("BH charge 8K nodes completes under 100ms")
    @MainActor func testBHCharge8KCompletes() async throws {
        let device = try #require(Self.device)
        let library = try EngramMetalShaders.makeLibrary(device: device)
        let fc = try #require(MetalForceCompute(device: device, library: library))
        let forceEngine = ForceEngine(forceCompute: fc)
        let queue = try #require(device.makeCommandQueue())
        #expect(fc.isFullSimAvailable)

        let tmpPath = NSTemporaryDirectory() + "gpulog-bh-\(UUID().uuidString).log"
        GPULog.configure(path: tmpPath)

        let n = 8000
        let x = (0..<n).map { _ in Float.random(in: -250...250) }
        let y = (0..<n).map { _ in Float.random(in: -250...250) }
        let z = (0..<n).map { _ in Float.random(in: -250...250) }
        let projectGroups = (0..<n).map { $0 % 15 }
        let topicGroups = (0..<n).map { $0 % 247 }

        let (result, ms, log) = try await dispatchUntilBH(
            forceEngine: forceEngine, queue: queue,
            x: x, y: y, z: z,
            projectGroups: projectGroups, topicGroups: topicGroups,
            logPath: tmpPath)

        print("[test:bh] BH charge 8K completed in \(String(format: "%.1f", ms))ms")
        print("[test:bh] Log:\n\(log)")

        #expect(result.fx.count == n)
        #expect(ms < 100, "BH charge took \(String(format: "%.0f", ms))ms — must be under 100ms to avoid GPU hang")

        let totalMag = result.fx.reduce(Float(0)) { $0 + abs($1) }
        #expect(totalMag > 0, "BH charge forces should be non-zero")

        try? FileManager.default.removeItem(atPath: tmpPath)
    }

    // MARK: - BH at 12K nodes (worst case from production logs)

    @Test("BH charge 12K nodes completes under 200ms")
    @MainActor func testBHCharge12KCompletes() async throws {
        let device = try #require(Self.device)
        let library = try EngramMetalShaders.makeLibrary(device: device)
        let fc = try #require(MetalForceCompute(device: device, library: library))
        let forceEngine = ForceEngine(forceCompute: fc)
        let queue = try #require(device.makeCommandQueue())

        let tmpPath = NSTemporaryDirectory() + "gpulog-bh12k-\(UUID().uuidString).log"
        GPULog.configure(path: tmpPath)

        let n = 12000
        let x = (0..<n).map { _ in Float.random(in: -250...250) }
        let y = (0..<n).map { _ in Float.random(in: -250...250) }
        let z = (0..<n).map { _ in Float.random(in: -250...250) }
        let projectGroups = (0..<n).map { $0 % 15 }
        let topicGroups = (0..<n).map { $0 % 247 }

        let (result, ms, _) = try await dispatchUntilBH(
            forceEngine: forceEngine, queue: queue,
            x: x, y: y, z: z,
            projectGroups: projectGroups, topicGroups: topicGroups,
            logPath: tmpPath)

        print("[test:bh] BH charge 12K completed in \(String(format: "%.1f", ms))ms")

        #expect(result.fx.count == n)
        #expect(ms < 200, "BH charge took \(String(format: "%.0f", ms))ms — must be under 200ms at 12K nodes")

        try? FileManager.default.removeItem(atPath: tmpPath)
    }
}
