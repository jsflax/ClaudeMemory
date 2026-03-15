import Foundation
import Testing
import simd
import CEngramSceneTypes
import EngramSceneKit

/// Perf tests for every component of the render pipeline.
/// Budget: 16.67ms total for 60fps. Each component gets a slice.
///
/// Target breakdown for 10K nodes / 150K edges:
///   Node packing:  3ms
///   Edge packing:  2ms
///   Label packing: 2ms
///   Nebula:        0.5ms
///   Total builder: 7.5ms (leaves 9ms for sim + mascot + GPU)
@Suite("Render Pipeline Performance")
struct RenderPipelinePerfTests {

    // MARK: - Test Data Builders

    static let projects = ["Engram", "Lattice", "ClaudeMemory", "engram-server",
                           "sidescroller", "global", "LatticeCore", "SwiftLM"]

    struct TestGraph {
        let nodes: [(id: UUID, project: String, importance: Int)]
        let positions: [UUID: SIMD3<Float>]
        let hubs: Set<UUID>
        let nodeIndexMap: [UUID: UInt32]
        let nodeRadii: [UUID: Float]
        let nodeProjects: [UUID: String]
        let projectColors: [String: SIMD3<Float>]
        let edges: [(sourceId: UUID, targetId: UUID, relation: String)]
    }

    static func makeGraph(nodeCount: Int, edgesPerNode: Int = 15) -> TestGraph {
        var nodes: [(id: UUID, project: String, importance: Int)] = []
        var positions: [UUID: SIMD3<Float>] = [:]
        var hubs: Set<UUID> = []
        var nodeIndexMap: [UUID: UInt32] = [:]
        var nodeRadii: [UUID: Float] = [:]
        var nodeProjects: [UUID: String] = [:]

        nodes.reserveCapacity(nodeCount)
        for i in 0..<nodeCount {
            let id = UUID()
            let project = projects[i % projects.count]
            let importance = (i % 5) + 1
            nodes.append((id: id, project: project, importance: importance))
            positions[id] = SIMD3<Float>(
                Float.random(in: -500...500),
                Float.random(in: -500...500),
                Float.random(in: -500...500))
            nodeIndexMap[id] = UInt32(i)
            nodeProjects[id] = project
            let isHub = i % 50 == 0
            if isHub { hubs.insert(id) }
            let baseR: Float = isHub ? 0.04 * 1.6 : 0.04
            nodeRadii[id] = baseR * (1.0 + Float(importance - 1) * 0.08)
        }

        var projectColors: [String: SIMD3<Float>] = [:]
        for p in projects {
            projectColors[p] = SIMD3<Float>(Float.random(in: 0...1),
                                             Float.random(in: 0...1),
                                             Float.random(in: 0...1))
        }

        // Build edges
        var edges: [(sourceId: UUID, targetId: UUID, relation: String)] = []
        let edgeCount = min(nodeCount * edgesPerNode, nodeCount * (nodeCount - 1) / 2)
        edges.reserveCapacity(edgeCount)
        for i in 0..<edgeCount {
            let src = i % nodeCount
            let tgt = (i / nodeCount + i + 1) % nodeCount
            if src != tgt {
                edges.append((sourceId: nodes[src].id, targetId: nodes[tgt].id, relation: "relates_to"))
            }
        }

        return TestGraph(nodes: nodes, positions: positions, hubs: hubs,
                         nodeIndexMap: nodeIndexMap, nodeRadii: nodeRadii,
                         nodeProjects: nodeProjects, projectColors: projectColors,
                         edges: edges)
    }

    private func measure(_ label: String, iterations: Int = 10, block: () -> Void) -> Double {
        block() // warm up
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations { block() }
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let avgMs = (elapsed / Double(iterations)) * 1000.0
        print("[engram:perf] \(label): \(String(format: "%.2f", avgMs))ms")
        return avgMs
    }

    // MARK: - Node Packing

    @Test("Node packing 10K: under 3ms")
    func testNodePacking10K() {
        let g = Self.makeGraph(nodeCount: 10000)
        let input = SceneNodeFrameInput(
            nodes: g.nodes, positions: g.positions, hubs: g.hubs,
            projectColors: g.projectColors, selectedNode: nil,
            glowingNodes: [:], newNodes: [:],
            isSearchActive: false, searchMatchIds: [],
            inspectingIntensity: [:], birthingElapsed: [:],
            nodeRadius: 0.04, cameraPosition: SIMD3<Float>(0, 0, 800),
            nodeIndexMap: g.nodeIndexMap, nodeRadii: g.nodeRadii)
        let ms = measure("node-pack-10K") { _ = buildSceneNodeFrame(input) }
        #expect(ms < 3.0, "node packing took \(String(format: "%.1f", ms))ms, budget 3ms")
    }

    @Test("Node packing 20K: under 6ms")
    func testNodePacking20K() {
        let g = Self.makeGraph(nodeCount: 20000)
        let input = SceneNodeFrameInput(
            nodes: g.nodes, positions: g.positions, hubs: g.hubs,
            projectColors: g.projectColors, selectedNode: nil,
            glowingNodes: [:], newNodes: [:],
            isSearchActive: false, searchMatchIds: [],
            inspectingIntensity: [:], birthingElapsed: [:],
            nodeRadius: 0.04, cameraPosition: SIMD3<Float>(0, 0, 800),
            nodeIndexMap: g.nodeIndexMap, nodeRadii: g.nodeRadii)
        let ms = measure("node-pack-20K") { _ = buildSceneNodeFrame(input) }
        #expect(ms < 6.0, "node packing took \(String(format: "%.1f", ms))ms, budget 6ms")
    }

    // MARK: - Edge Packing

    @Test("Edge resolve 10K/150K (topology change, one-time): under 50ms")
    func testEdgeResolve10K() {
        let g = Self.makeGraph(nodeCount: 10000, edgesPerNode: 15)
        let ms = measure("edge-resolve-10K/150K", iterations: 5) {
            _ = resolveEdges(edges: g.edges, nodeIndexMap: g.nodeIndexMap,
                             nodeRadii: g.nodeRadii, nodeProjects: g.nodeProjects,
                             projectColors: g.projectColors, defaultEdgeRadius: 0.004)
        }
        #expect(ms < 50.0, "edge resolve took \(String(format: "%.1f", ms))ms, budget 50ms (one-time)")
    }

    @Test("Edge packing 10K nodes / 150K edges: under 2ms")
    func testEdgePacking10K() {
        let g = Self.makeGraph(nodeCount: 10000, edgesPerNode: 15)
        let resolved = resolveEdges(edges: g.edges, nodeIndexMap: g.nodeIndexMap,
                                     nodeRadii: g.nodeRadii, nodeProjects: g.nodeProjects,
                                     projectColors: g.projectColors, defaultEdgeRadius: 0.004)
        let input = SceneEdgeFrameInput(resolvedEdges: resolved, edgeRadius: 0.004,
                                         nodeCount: g.nodeIndexMap.count)
        let ms = measure("edge-pack-10K/150K") { _ = buildSceneEdgeFrame(input) }
        #expect(ms < 2.0, "edge packing took \(String(format: "%.1f", ms))ms, budget 2ms")
    }

    @Test("Edge packing 20K nodes / 300K edges: under 4ms")
    func testEdgePacking20K() {
        let g = Self.makeGraph(nodeCount: 20000, edgesPerNode: 15)
        let resolved = resolveEdges(edges: g.edges, nodeIndexMap: g.nodeIndexMap,
                                     nodeRadii: g.nodeRadii, nodeProjects: g.nodeProjects,
                                     projectColors: g.projectColors, defaultEdgeRadius: 0.004)
        let input = SceneEdgeFrameInput(resolvedEdges: resolved, edgeRadius: 0.004,
                                         nodeCount: g.nodeIndexMap.count)
        let ms = measure("edge-pack-20K/300K") { _ = buildSceneEdgeFrame(input) }
        #expect(ms < 4.0, "edge packing took \(String(format: "%.1f", ms))ms, budget 4ms")
    }

    // MARK: - Label Packing

    @Test("Label packing 10K nodes: under 2ms")
    func testLabelPacking10K() {
        let g = Self.makeGraph(nodeCount: 10000)
        let rect = AtlasRect(u0: 0, v0: 0, u1: 0.1, v1: 0.05)
        var atlasRects: [UUID: AtlasRect] = [:]
        var nodeLabels: [(id: UUID, position: SIMD3<Float>, isHub: Bool,
                          importance: Int, isSelected: Bool,
                          isSearchMatch: Bool, searchDimmed: Bool,
                          isExpandedChild: Bool)] = []
        for node in g.nodes {
            atlasRects[node.id] = rect
            nodeLabels.append((id: node.id, position: g.positions[node.id]!,
                               isHub: g.hubs.contains(node.id),
                               importance: node.importance, isSelected: false,
                               isSearchMatch: false, searchDimmed: false,
                               isExpandedChild: false))
        }
        var projCentroids: [String: (centroid: SIMD3<Float>, maxY: Float, count: Int)] = [:]
        for p in Self.projects {
            projCentroids[p] = (centroid: SIMD3<Float>(Float.random(in: -100...100), 50, 0), maxY: 100, count: 500)
        }
        var projRects: [String: AtlasRect] = [:]
        for p in Self.projects { projRects[p] = rect }

        let input = SceneLabelFrameInput(
            nodeLabels: nodeLabels, atlasRects: atlasRects,
            projectCentroids: projCentroids, projectLabelRects: projRects,
            projectColors: g.projectColors, galaxyLabels: [],
            galaxyLabelRects: [:], topicGroups: [], topicLabelRects: [:],
            scaleFactor: 1.0/200.0, nodeRadius: 0.04,
            aspectCorrection: 1.0)
        let ms = measure("label-pack-10K") { _ = buildSceneLabelFrame(input) }
        #expect(ms < 2.0, "label packing took \(String(format: "%.1f", ms))ms, budget 2ms")
    }

    // MARK: - Nebula Packing

    @Test("Nebula packing 64 projects: under 0.1ms")
    func testNebulaPacking() {
        var centroids: [String: (centroid: SIMD3<Float>, maxY: Float, count: Int)] = [:]
        var colors: [String: SIMD4<Float>] = [:]
        for i in 0..<64 {
            let name = "project_\(i)"
            centroids[name] = (centroid: SIMD3<Float>(Float(i) * 100, 0, 0), maxY: 50, count: 100)
            colors[name] = SIMD4<Float>(0.5, 0.5, 0.5, 0.25)
        }
        let ms = measure("nebula-pack-64", iterations: 100) {
            _ = buildNebulaGroups(projectCentroids: centroids, projectColors: colors, scaleFactor: 1.0/200.0)
        }
        #expect(ms < 0.1, "nebula packing took \(String(format: "%.3f", ms))ms, budget 0.1ms")
    }

    // MARK: - Dictionary Lookup Microbenchmarks

    @Test("UUID dict lookup 10K: under 1ms")
    func testUUIDLookup10K() {
        var dict: [UUID: SIMD3<Float>] = [:]
        var keys: [UUID] = []
        for _ in 0..<10000 {
            let id = UUID()
            keys.append(id)
            dict[id] = SIMD3<Float>(1, 2, 3)
        }
        let ms = measure("uuid-lookup-10K", iterations: 50) {
            var sum: Float = 0
            for k in keys { sum += dict[k]!.x }
            _ = sum
        }
        #expect(ms < 1.0, "UUID lookup took \(String(format: "%.2f", ms))ms")
    }

    @Test("UUID dict INSERT 10K: cost of rebuilding per frame")
    func testUUIDInsert10K() {
        var keys: [UUID] = []
        for _ in 0..<10000 { keys.append(UUID()) }
        let ms = measure("uuid-insert-10K", iterations: 20) {
            var dict: [UUID: UInt32] = [:]
            dict.reserveCapacity(10000)
            for (i, k) in keys.enumerated() { dict[k] = UInt32(i) }
        }
        print("[engram:perf] ^ this is what rebuilding nodeIndexMap every frame costs")
        #expect(ms < 3.0)
    }

    // MARK: - Full Pipeline (all builders)

    @Test("Full pipeline 10K nodes / 150K edges: under 7ms")
    func testFullPipeline10K() {
        let g = Self.makeGraph(nodeCount: 10000, edgesPerNode: 15)

        let nodeInput = SceneNodeFrameInput(
            nodes: g.nodes, positions: g.positions, hubs: g.hubs,
            projectColors: g.projectColors, selectedNode: nil,
            glowingNodes: [:], newNodes: [:],
            isSearchActive: false, searchMatchIds: [],
            inspectingIntensity: [:], birthingElapsed: [:],
            nodeRadius: 0.04, cameraPosition: SIMD3<Float>(0, 0, 800),
            nodeIndexMap: g.nodeIndexMap, nodeRadii: g.nodeRadii)

        let resolved = resolveEdges(edges: g.edges, nodeIndexMap: g.nodeIndexMap,
                                     nodeRadii: g.nodeRadii, nodeProjects: g.nodeProjects,
                                     projectColors: g.projectColors, defaultEdgeRadius: 0.004)
        let edgeInput = SceneEdgeFrameInput(resolvedEdges: resolved, edgeRadius: 0.004,
                                             nodeCount: g.nodeIndexMap.count)

        let ms = measure("full-pipeline-10K") {
            let nf = buildSceneNodeFrame(nodeInput)
            _ = buildSceneEdgeFrame(edgeInput)
            _ = buildNebulaGroups(projectCentroids: nf.projectCentroids,
                                  projectColors: [:], scaleFactor: 1.0/200.0)
        }
        #expect(ms < 7.0, "full pipeline took \(String(format: "%.1f", ms))ms, budget 7ms")
    }
}
