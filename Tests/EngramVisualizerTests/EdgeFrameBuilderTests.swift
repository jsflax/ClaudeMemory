import Foundation
import Testing
import simd
import CEngramSceneTypes
import EngramSceneKit

@Suite("SceneEdgeFrame Builder")
struct EdgeFrameBuilderTests {

    let defaultEdgeRadius: Float = 0.004

    private func resolveAndBuild(
        edges: [(sourceId: UUID, targetId: UUID, relation: String)] = [],
        nodeIndexMap: [UUID: UInt32] = [:],
        nodeRadii: [UUID: Float] = [:],
        nodeProjects: [UUID: String] = [:],
        projectColors: [String: SIMD3<Float>] = [:],
        selectedNode: UUID? = nil,
        isSearchActive: Bool = false,
        searchMatchIds: Set<UUID> = [],
        isSemanticMode: Bool = false,
        edgeRadius: Float = 0.004,
        interGalaxyConnections: [(from: SIMD3<Float>, to: SIMD3<Float>)] = []
    ) -> SceneEdgeFrame {
        let resolved = resolveEdges(
            edges: edges, nodeIndexMap: nodeIndexMap, nodeRadii: nodeRadii,
            nodeProjects: nodeProjects, projectColors: projectColors,
            defaultEdgeRadius: edgeRadius)

        let selectedIdx = selectedNode.flatMap { nodeIndexMap[$0] }

        var dimmedIndices = Set<UInt32>()
        if isSearchActive {
            for (id, idx) in nodeIndexMap where !searchMatchIds.contains(id) {
                dimmedIndices.insert(idx)
            }
        }

        let input = SceneEdgeFrameInput(
            resolvedEdges: resolved, selectedIdx: selectedIdx,
            isSearchActive: isSearchActive, searchDimmedSourceIdx: dimmedIndices,
            isSemanticMode: isSemanticMode, edgeRadius: edgeRadius,
            nodeCount: nodeIndexMap.count,
            interGalaxyConnections: interGalaxyConnections)
        return buildSceneEdgeFrame(input)
    }

    @Test("Empty edges produces empty frame")
    func testEmptyEdges() {
        let frame = resolveAndBuild()
        #expect(frame.count == 0)
        #expect(frame.descriptors.isEmpty)
    }

    @Test("Normal edge: state=0, 1.3x radius")
    func testNormalEdge() {
        let src = UUID(), tgt = UUID()
        let frame = resolveAndBuild(
            edges: [(sourceId: src, targetId: tgt, relation: "relates_to")],
            nodeIndexMap: [src: 0, tgt: 1],
            nodeProjects: [src: "p"],
            projectColors: ["p": SIMD3<Float>(1, 0, 0)]
        )
        #expect(frame.count == 1)
        let desc = frame.descriptors[0]
        #expect(desc.state == 0)
        #expect(abs(desc.baseRadius - defaultEdgeRadius * 1.3) < 0.0001)
    }

    @Test("Connected to selected: state=1, white, 2.5x radius")
    func testConnectedToSelected() {
        let src = UUID(), tgt = UUID()
        let frame = resolveAndBuild(
            edges: [(sourceId: src, targetId: tgt, relation: "r")],
            nodeIndexMap: [src: 0, tgt: 1],
            selectedNode: src
        )
        let desc = frame.descriptors[0]
        #expect(desc.state == 1)
        #expect(abs(desc.color.x - 1.0) < 0.001) // white
        #expect(abs(desc.baseRadius - defaultEdgeRadius * 2.5) < 0.0001)
    }

    @Test("Search dimmed: both endpoints non-matching → state=2")
    func testSearchDimmed() {
        let src = UUID(), tgt = UUID()
        let matchId = UUID()
        let frame = resolveAndBuild(
            edges: [(sourceId: src, targetId: tgt, relation: "r")],
            nodeIndexMap: [src: 0, tgt: 1, matchId: 2],
            isSearchActive: true,
            searchMatchIds: [matchId]
        )
        #expect(frame.descriptors[0].state == 2)
    }

    @Test("Semantic mode: non-selected → state=3")
    func testSemanticMode() {
        let src = UUID(), tgt = UUID()
        let frame = resolveAndBuild(
            edges: [(sourceId: src, targetId: tgt, relation: "r")],
            nodeIndexMap: [src: 0, tgt: 1],
            isSemanticMode: true
        )
        #expect(frame.descriptors[0].state == 3)
    }

    @Test("Missing node: edge referencing absent node is skipped")
    func testMissingNode() {
        let src = UUID(), tgt = UUID()
        let frame = resolveAndBuild(
            edges: [(sourceId: src, targetId: tgt, relation: "r")],
            nodeIndexMap: [src: 0] // tgt missing
        )
        #expect(frame.count == 0)
    }

    @Test("Inter-galaxy: dimmed state, 4x radius, virtual positions")
    func testInterGalaxy() {
        let from = SIMD3<Float>(1, 2, 3)
        let to = SIMD3<Float>(4, 5, 6)
        let frame = resolveAndBuild(
            interGalaxyConnections: [(from: from, to: to)]
        )
        #expect(frame.count == 1)
        #expect(frame.descriptors[0].state == 2)
        #expect(abs(frame.descriptors[0].baseRadius - defaultEdgeRadius * 4.0) < 0.0001)
        #expect(frame.interGalaxyPositions.count == 2)
        #expect(frame.interGalaxyPositions[0] == from)
        #expect(frame.interGalaxyPositions[1] == to)
    }
}
