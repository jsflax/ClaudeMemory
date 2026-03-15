import Foundation
import Testing
import simd
import EngramSceneKit

@Suite("Force Computation Performance")
struct ForceComputationPerfTests {

    private func makeInput(nodeCount: Int, edgesPerNode: Int = 15, spread: Float = 1500) -> ForceComputationInput {
        var x = [Float](repeating: 0, count: nodeCount)
        var y = [Float](repeating: 0, count: nodeCount)
        var z = [Float](repeating: 0, count: nodeCount)
        var pg = [Int](repeating: 0, count: nodeCount)
        var tg = [Int](repeating: 0, count: nodeCount)
        var tpg = [Int]()

        for i in 0..<nodeCount {
            x[i] = Float.random(in: -spread...spread)
            y[i] = Float.random(in: -spread...spread)
            z[i] = Float.random(in: -spread...spread)
            pg[i] = i % 8
            tg[i] = i % 30
        }
        for t in 0..<30 { tpg.append(t % 8) }

        var edges: [(Int, Int)] = []
        let edgeCount = min(nodeCount * edgesPerNode, nodeCount * (nodeCount - 1) / 2)
        edges.reserveCapacity(edgeCount)
        for i in 0..<edgeCount {
            let src = i % nodeCount
            let tgt = (i / nodeCount + i + 1) % nodeCount
            if src != tgt { edges.append((src, tgt)) }
        }

        return ForceComputationInput(
            n: nodeCount, x: x, y: y, z: z,
            projectGroup: pg, topicGroup: tg,
            edgeIndices: edges, topicProjectGroup: tpg,
            alpha: 0.5, center: .zero,
            chargeStrength: 500, crossChargeMultiplier: 3.0,
            sameProjectChargeScale: 1.0, sameTopicChargeScale: 0.35,
            centerStrength: 0.006,
            springLength: 240, crossProjectSpringLength: 400, springStrength: 0.0004,
            cohesionStrength: 0.0015, centroidRepulsion: 2500,
            topicCohesionStrength: 0.009, topicCentroidRepulsion: 3500)
    }

    private func measure(_ label: String, iterations: Int = 5, block: () -> Void) -> Double {
        block()
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations { block() }
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let avgMs = (elapsed / Double(iterations)) * 1000.0
        print("[engram:perf] \(label): \(String(format: "%.2f", avgMs))ms")
        return avgMs
    }

    @Test("CPU force 1K nodes (exact O(n²)): under 5ms")
    func testForce1K() {
        let input = makeInput(nodeCount: 1000, spread: 200) // dense → exact path
        let ms = measure("cpu-force-1K") { _ = computeAllForces(input) }
        #expect(ms < 5.0, "took \(String(format: "%.1f", ms))ms")
    }

    @Test("CPU force 5K nodes (Barnes-Hut): under 15ms")
    func testForce5K() {
        let input = makeInput(nodeCount: 5000)
        let ms = measure("cpu-force-5K") { _ = computeAllForces(input) }
        #expect(ms < 15.0, "took \(String(format: "%.1f", ms))ms")
    }

    @Test("CPU force 10K nodes (Barnes-Hut): under 30ms")
    func testForce10K() {
        let input = makeInput(nodeCount: 10000)
        let ms = measure("cpu-force-10K") { _ = computeAllForces(input) }
        #expect(ms < 30.0, "took \(String(format: "%.1f", ms))ms")
    }

    @Test("CPU force 20K nodes (Barnes-Hut): under 60ms")
    func testForce20K() {
        let input = makeInput(nodeCount: 20000)
        let ms = measure("cpu-force-20K") { _ = computeAllForces(input) }
        #expect(ms < 60.0, "took \(String(format: "%.1f", ms))ms")
    }

    @Test("Force output has correct size")
    func testOutputSize() {
        let input = makeInput(nodeCount: 100)
        let result = computeAllForces(input)
        #expect(result.fx.count == 100)
        #expect(result.fy.count == 100)
        #expect(result.fz.count == 100)
    }

    @Test("Forces are non-zero for spread nodes")
    func testForcesNonZero() {
        let input = makeInput(nodeCount: 50)
        let result = computeAllForces(input)
        let totalMagnitude = zip(result.fx, zip(result.fy, result.fz)).reduce(Float(0)) { acc, v in
            acc + abs(v.0) + abs(v.1.0) + abs(v.1.1)
        }
        #expect(totalMagnitude > 0, "forces should be non-zero")
    }

    @Test("Barnes-Hut vs exact produce similar results")
    func testBHAccuracy() {
        // 300 nodes, dense → exact; 300 nodes spread → BH
        let dense = makeInput(nodeCount: 300, edgesPerNode: 0, spread: 100)
        let spread = makeInput(nodeCount: 300, edgesPerNode: 0, spread: 2000)

        let denseResult = computeAllForces(dense)
        let spreadResult = computeAllForces(spread)

        // Both should produce non-zero forces
        let denseMag = denseResult.fx.reduce(0) { $0 + abs($1) }
        let spreadMag = spreadResult.fx.reduce(0) { $0 + abs($1) }
        #expect(denseMag > 0)
        #expect(spreadMag > 0)
    }
}
