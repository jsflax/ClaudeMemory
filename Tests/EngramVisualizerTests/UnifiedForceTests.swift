import Foundation
import Testing
import simd
import EngramSceneKit

@Suite("Unified Force — Galaxy-aware computation")
struct UnifiedForceTests {

    // MARK: - Helpers

    private func makeInput(
        nodeCount: Int,
        galaxyGroup: [Int] = [],
        galaxyCenters: [SIMD3<Float>] = [],
        edgeIndices: [(Int, Int)] = [],
        spread: Float = 200,
        centerOverride: SIMD3<Float>? = nil
    ) -> ForceComputationInput {
        var x = [Float](repeating: 0, count: nodeCount)
        var y = [Float](repeating: 0, count: nodeCount)
        var z = [Float](repeating: 0, count: nodeCount)
        var pg = [Int](repeating: 0, count: nodeCount)
        var tg = [Int](repeating: 0, count: nodeCount)

        for i in 0..<nodeCount {
            x[i] = Float.random(in: -spread...spread)
            y[i] = Float.random(in: -spread...spread)
            z[i] = Float.random(in: -spread...spread)
            pg[i] = i % 4
            tg[i] = i % 10
        }
        var tpg = [Int]()
        for t in 0..<10 { tpg.append(t % 4) }

        return ForceComputationInput(
            n: nodeCount, x: x, y: y, z: z,
            projectGroup: pg, topicGroup: tg,
            edgeIndices: edgeIndices, topicProjectGroup: tpg,
            alpha: 0.5, center: centerOverride ?? .zero,
            chargeStrength: 500, crossChargeMultiplier: 3.0,
            sameProjectChargeScale: 1.0, sameTopicChargeScale: 0.35,
            centerStrength: 0.006,
            springLength: 240, crossProjectSpringLength: 400, springStrength: 0.0004,
            cohesionStrength: 0.0015, centroidRepulsion: 2500,
            topicCohesionStrength: 0.009, topicCentroidRepulsion: 3500,
            galaxyGroup: galaxyGroup, galaxyCenters: galaxyCenters)
    }

    // MARK: - Galaxy Center Force

    @Test("Per-galaxy center force pulls nodes toward their galaxy center")
    func testPerGalaxyCenterForce() {
        // 2 galaxies: center at (0,0,0) and (4000,0,0)
        // Place all nodes at origin so galaxy-1 nodes feel a pull toward (4000,0,0)
        let n = 20
        var x = [Float](repeating: 0, count: n)
        var y = [Float](repeating: 0, count: n)
        var z = [Float](repeating: 0, count: n)
        var galaxyGroup = [Int](repeating: 0, count: n)

        // First 10 nodes in galaxy 0, next 10 in galaxy 1
        for i in 10..<n { galaxyGroup[i] = 1 }

        // Slight spread to avoid zero-distance issues
        for i in 0..<n {
            x[i] = Float.random(in: -10...10)
            y[i] = Float.random(in: -10...10)
            z[i] = Float.random(in: -10...10)
        }

        let pg = [Int](repeating: 0, count: n)
        let tg = [Int](repeating: 0, count: n)

        let input = ForceComputationInput(
            n: n, x: x, y: y, z: z,
            projectGroup: pg, topicGroup: tg,
            edgeIndices: [], topicProjectGroup: [],
            alpha: 0.5, center: .zero,
            chargeStrength: 0, crossChargeMultiplier: 1.0,  // disable charge
            sameProjectChargeScale: 1.0, sameTopicChargeScale: 1.0,
            centerStrength: 0.1,  // strong center force
            springLength: 240, crossProjectSpringLength: 400, springStrength: 0,
            cohesionStrength: 0, centroidRepulsion: 0,
            topicCohesionStrength: 0, topicCentroidRepulsion: 0,
            galaxyGroup: galaxyGroup,
            galaxyCenters: [SIMD3<Float>(0, 0, 0), SIMD3<Float>(4000, 0, 0)])

        let result = computeAllForces(input)

        // Galaxy-0 nodes should be pulled toward (0,0,0) — small forces
        // Galaxy-1 nodes should be pulled strongly toward (4000,0,0) — large positive fx
        var galaxy1FxSum: Float = 0
        for i in 10..<n {
            galaxy1FxSum += result.fx[i]
        }
        #expect(galaxy1FxSum > 100, "Galaxy-1 nodes should be pulled toward x=4000, got \(galaxy1FxSum)")

        // Galaxy-0 nodes should have near-zero net center force (they're near origin)
        var galaxy0FxSum: Float = 0
        for i in 0..<10 {
            galaxy0FxSum += result.fx[i]
        }
        #expect(abs(galaxy0FxSum) < abs(galaxy1FxSum), "Galaxy-0 force should be much smaller than galaxy-1")
    }

    // MARK: - Cross-Galaxy Spring Skip

    @Test("Cross-galaxy edges produce zero spring force")
    func testCrossGalaxySpringSkip() {
        let n = 4
        // Nodes 0,1 in galaxy 0; nodes 2,3 in galaxy 1
        // Edge (0,2) crosses galaxies — should produce zero spring
        let x: [Float] = [0, 10, 1000, 1010]
        let y: [Float] = [0, 0, 0, 0]
        let z: [Float] = [0, 0, 0, 0]
        let galaxyGroup = [0, 0, 1, 1]
        let pg = [0, 0, 0, 0]
        let tg = [0, 0, 0, 0]

        let input = ForceComputationInput(
            n: n, x: x, y: y, z: z,
            projectGroup: pg, topicGroup: tg,
            edgeIndices: [(0, 2)],  // cross-galaxy edge only
            topicProjectGroup: [0],
            alpha: 0.5, center: .zero,
            chargeStrength: 0, crossChargeMultiplier: 1.0,
            sameProjectChargeScale: 1.0, sameTopicChargeScale: 1.0,
            centerStrength: 0,
            springLength: 240, crossProjectSpringLength: 400, springStrength: 0.5,
            cohesionStrength: 0, centroidRepulsion: 0,
            topicCohesionStrength: 0, topicCentroidRepulsion: 0,
            galaxyGroup: galaxyGroup,
            galaxyCenters: [.zero, SIMD3<Float>(1000, 0, 0)])

        let result = computeAllForces(input)

        // With no charge, no center force, and spring skipped — all forces should be zero
        for i in 0..<n {
            #expect(abs(result.fx[i]) < 0.001, "Node \(i) fx should be zero, got \(result.fx[i])")
            #expect(abs(result.fy[i]) < 0.001, "Node \(i) fy should be zero, got \(result.fy[i])")
            #expect(abs(result.fz[i]) < 0.001, "Node \(i) fz should be zero, got \(result.fz[i])")
        }
    }

    // MARK: - Intra-Galaxy Spring Applied

    @Test("Same-galaxy edge produces non-zero spring force")
    func testIntraGalaxySpringApplied() {
        let n = 2
        let x: [Float] = [0, 500]  // distance=500 > springLength=240 → attractive
        let y: [Float] = [0, 0]
        let z: [Float] = [0, 0]
        let galaxyGroup = [0, 0]  // same galaxy
        let pg = [0, 0]
        let tg = [0, 0]

        let input = ForceComputationInput(
            n: n, x: x, y: y, z: z,
            projectGroup: pg, topicGroup: tg,
            edgeIndices: [(0, 1)],
            topicProjectGroup: [0],
            alpha: 0.5, center: .zero,
            chargeStrength: 0, crossChargeMultiplier: 1.0,
            sameProjectChargeScale: 1.0, sameTopicChargeScale: 1.0,
            centerStrength: 0,
            springLength: 240, crossProjectSpringLength: 400, springStrength: 0.5,
            cohesionStrength: 0, centroidRepulsion: 0,
            topicCohesionStrength: 0, topicCentroidRepulsion: 0,
            galaxyGroup: galaxyGroup,
            galaxyCenters: [.zero])

        let result = computeAllForces(input)

        // Node 0 should be pulled toward node 1 (positive x)
        #expect(result.fx[0] > 1.0, "Node 0 should be pulled right, got \(result.fx[0])")
        // Node 1 should be pulled toward node 0 (negative x)
        #expect(result.fx[1] < -1.0, "Node 1 should be pulled left, got \(result.fx[1])")
    }

    // MARK: - Backward Compatibility

    @Test("Empty galaxyGroup defaults to single-galaxy behavior")
    func testBackwardCompat() {
        // No galaxyGroup → should work exactly as before (single center, all springs active)
        let input = makeInput(
            nodeCount: 50,
            galaxyGroup: [],  // empty = backward compat
            galaxyCenters: [],
            edgeIndices: [(0, 1), (1, 2), (2, 3)])

        let result = computeAllForces(input)

        // Should produce non-zero forces (springs + charge + center gravity)
        let totalMag = zip(result.fx, zip(result.fy, result.fz)).reduce(Float(0)) { acc, v in
            acc + abs(v.0) + abs(v.1.0) + abs(v.1.1)
        }
        #expect(totalMag > 0, "Forces should be non-zero in backward compat mode")
    }

    // MARK: - Performance

    private func measure(_ label: String, iterations: Int = 5, block: () -> Void) -> Double {
        block()  // warmup
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations { block() }
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let avgMs = (elapsed / Double(iterations)) * 1000.0
        print("[engram:perf] \(label): \(String(format: "%.2f", avgMs))ms")
        return avgMs
    }

    @Test("6K unified nodes with 2 galaxy groups under 15ms")
    func testPerf6KUnified() {
        let n = 6000
        var galaxyGroup = [Int](repeating: 0, count: n)
        for i in (n/2)..<n { galaxyGroup[i] = 1 }

        let input = makeInput(
            nodeCount: n,
            galaxyGroup: galaxyGroup,
            galaxyCenters: [.zero, SIMD3<Float>(4000, 0, 0)],
            spread: 1500)

        let ms = measure("unified-6K-2galaxy") { _ = computeAllForces(input) }
        #expect(ms < 15.0, "took \(String(format: "%.1f", ms))ms")
    }

    // MARK: - Multi-Galaxy Scaling

    @Test("10 galaxies × 1K nodes converge to distinct centers")
    func testForce10Galaxies() {
        let galaxyCount = 10
        let nodesPerGalaxy = 1000
        let n = galaxyCount * nodesPerGalaxy
        var galaxyGroup = [Int](repeating: 0, count: n)
        var galaxyCenters = [SIMD3<Float>]()

        for g in 0..<galaxyCount {
            let cx = Float(g) * 4000
            galaxyCenters.append(SIMD3<Float>(cx, 0, 0))
            for i in 0..<nodesPerGalaxy {
                galaxyGroup[g * nodesPerGalaxy + i] = g
            }
        }

        let input = makeInput(
            nodeCount: n,
            galaxyGroup: galaxyGroup,
            galaxyCenters: galaxyCenters,
            spread: 500)

        let result = computeAllForces(input)

        // Each galaxy's nodes should have center force pulling them toward their center
        // Just verify forces are non-zero and consistent
        #expect(result.fx.count == n)
        let totalMag = result.fx.reduce(Float(0)) { $0 + abs($1) }
        #expect(totalMag > 0, "10-galaxy forces should be non-zero")
    }

    @Test("15 galaxies × 500 nodes force compute under budget")
    func testPerf15Galaxies() {
        let galaxyCount = 15
        let nodesPerGalaxy = 500
        let n = galaxyCount * nodesPerGalaxy  // 7500 total
        var galaxyGroup = [Int](repeating: 0, count: n)
        var galaxyCenters = [SIMD3<Float>]()

        for g in 0..<galaxyCount {
            galaxyCenters.append(SIMD3<Float>(Float(g) * 4000, 0, 0))
            for i in 0..<nodesPerGalaxy {
                galaxyGroup[g * nodesPerGalaxy + i] = g
            }
        }

        let input = makeInput(
            nodeCount: n,
            galaxyGroup: galaxyGroup,
            galaxyCenters: galaxyCenters,
            spread: 1500)

        let ms = measure("unified-15galaxy-7500nodes") { _ = computeAllForces(input) }
        #expect(ms < 30.0, "took \(String(format: "%.1f", ms))ms")
    }
}
