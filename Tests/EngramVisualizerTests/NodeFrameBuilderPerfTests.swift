import Foundation
import Testing
import simd
import CEngramSceneTypes
import EngramSceneKit

@Suite("NodeFrame Builder Performance")
struct NodeFrameBuilderPerfTests {

    /// Helper to build test data at a given scale using the parallel-array API.
    private func makeScaleInput(nodeCount: Int) -> SceneNodeFrameInput {
        let projects = ["Engram", "Lattice", "ClaudeMemory", "engram-server",
                        "sidescroller", "global", "LatticeCore", "SwiftLM"]

        var nodeIds: [UUID] = []
        var nodeProjects: [String] = []
        var nodeImportances: [Int] = []
        var nodePositions: [SIMD3<Float>] = []
        var hasPosition: [Bool] = []
        var nodeRadii: [Float] = []
        var nodeIsHub: [Bool] = []

        nodeIds.reserveCapacity(nodeCount)
        nodeProjects.reserveCapacity(nodeCount)
        nodeImportances.reserveCapacity(nodeCount)
        nodePositions.reserveCapacity(nodeCount)
        hasPosition.reserveCapacity(nodeCount)
        nodeRadii.reserveCapacity(nodeCount)
        nodeIsHub.reserveCapacity(nodeCount)

        for i in 0..<nodeCount {
            let id = UUID()
            let project = projects[i % projects.count]
            let importance = (i % 5) + 1
            let isHub = i % 50 == 0

            nodeIds.append(id)
            nodeProjects.append(project)
            nodeImportances.append(importance)
            nodePositions.append(SIMD3<Float>(Float.random(in: -500...500),
                                              Float.random(in: -500...500),
                                              Float.random(in: -500...500)))
            hasPosition.append(true)
            nodeIsHub.append(isHub)

            let baseR: Float = isHub ? 0.04 * 1.6 : 0.04
            nodeRadii.append(baseR * (1.0 + Float(importance - 1) * 0.08))
        }

        var projectColors: [String: SIMD3<Float>] = [:]
        for p in projects {
            projectColors[p] = SIMD3<Float>(Float.random(in: 0...1),
                                             Float.random(in: 0...1),
                                             Float.random(in: 0...1))
        }

        return SceneNodeFrameInput(
            nodeIds: nodeIds,
            nodeProjects: nodeProjects,
            nodeImportances: nodeImportances,
            nodePositions: nodePositions,
            hasPosition: hasPosition,
            nodeRadii: nodeRadii,
            nodeIsHub: nodeIsHub,
            projectColors: projectColors,
            selectedNodeIndex: nil,
            glowingNodes: [:],
            newNodes: [:],
            isSearchActive: false,
            searchMatchIndices: [],
            inspectingIntensity: [:],
            birthingElapsed: [:],
            nodeRadius: 0.04,
            cameraPosition: SIMD3<Float>(0, 0, 800)
        )
    }

    private func measure(_ label: String, iterations: Int = 10, block: () -> Void) -> Double {
        // Warm up
        block()
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations { block() }
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let avgMs = (elapsed / Double(iterations)) * 1000.0
        print("[engram:perf] \(label): avg=\(String(format: "%.2f", avgMs))ms (\(iterations) iterations)")
        return avgMs
    }

    @Test("8684 nodes — full build under 8ms")
    func testBuild8k() {
        let input = makeScaleInput(nodeCount: 8684)
        let ms = measure("buildSceneNodeFrame 8684") { _ = buildSceneNodeFrame(input) }
        #expect(ms < 8.0, "took \(String(format: "%.1f", ms))ms, budget 8ms")
    }

    @Test("20000 nodes — full build under 16ms")
    func testBuild20k() {
        let input = makeScaleInput(nodeCount: 20000)
        let ms = measure("buildSceneNodeFrame 20000") { _ = buildSceneNodeFrame(input) }
        #expect(ms < 16.0, "took \(String(format: "%.1f", ms))ms, budget 16ms")
    }

    @Test("8684 nodes — visual state functions are cheap")
    func testVisualStateCost() {
        let ms = measure("recallGlowIntensity 8684x", iterations: 100) {
            for i in 0..<8684 {
                _ = recallGlowIntensity(elapsed: Float(i % 5))
            }
        }
        #expect(ms < 1.0, "visual state took \(String(format: "%.2f", ms))ms")
    }

    @Test("8684 nodes — position array access cost")
    func testPositionAccessCost() {
        let input = makeScaleInput(nodeCount: 8684)
        let ms = measure("position access 8684", iterations: 20) {
            var sum: Float = 0
            for i in 0..<input.nodePositions.count {
                if input.hasPosition[i] {
                    sum += input.nodePositions[i].x
                }
            }
            _ = sum // prevent optimization
        }
        #expect(ms < 3.0, "position access took \(String(format: "%.2f", ms))ms")
    }

    @Test("20000 nodes — position array access cost")
    func testPositionAccess20k() {
        let input = makeScaleInput(nodeCount: 20000)
        let ms = measure("position access 20000", iterations: 20) {
            var sum: Float = 0
            for i in 0..<input.nodePositions.count {
                if input.hasPosition[i] {
                    sum += input.nodePositions[i].x
                }
            }
            _ = sum
        }
        #expect(ms < 6.0, "position access took \(String(format: "%.2f", ms))ms")
    }

    @Test("packNodeState encoding is trivial")
    func testPackNodeStateCost() {
        let ms = measure("packNodeState 20000x", iterations: 100) {
            var sum: Float = 0
            for i in 0..<20000 {
                sum += packNodeState(stateType: Float(i % 7), searchDimmed: i % 3 == 0, intensity: Float(i % 100) / 100.0)
            }
            _ = sum
        }
        #expect(ms < 0.5)
    }
}
