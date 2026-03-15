import Foundation
import Testing
import simd
import CEngramSceneTypes
@testable import EngramSceneKit

@Suite("SceneNodeFrame Builder")
struct SceneNodeFrameBuilderTests {

    let defaultRadius: Float = 0.04

    private func makeInput(
        nodes: [(id: UUID, project: String, importance: Int)] = [],
        positions: [UUID: SIMD3<Float>] = [:],
        hubs: Set<UUID> = [],
        projectColors: [String: SIMD3<Float>] = [:],
        selectedNode: UUID? = nil,
        glowingNodes: [UUID: Float] = [:],
        newNodes: [UUID: Float] = [:],
        isSearchActive: Bool = false,
        searchMatchIds: Set<UUID> = [],
        inspectingIntensity: [UUID: Float] = [:],
        birthingElapsed: [UUID: Float] = [:],
        nodeRadius: Float = 0.04,
        cameraPosition: SIMD3<Float> = .zero
    ) -> SceneNodeFrameInput {
        SceneNodeFrameInput(
            nodes: nodes, positions: positions, hubs: hubs,
            projectColors: projectColors, selectedNode: selectedNode,
            glowingNodes: glowingNodes, newNodes: newNodes,
            isSearchActive: isSearchActive, searchMatchIds: searchMatchIds,
            inspectingIntensity: inspectingIntensity, birthingElapsed: birthingElapsed,
            nodeRadius: nodeRadius, cameraPosition: cameraPosition
        )
    }

    @Test("Empty nodes produces empty frame")
    func testEmptyNodes() {
        let frame = buildSceneNodeFrame(makeInput())
        #expect(frame.packInputs.isEmpty)
        #expect(frame.nodePositions.isEmpty)
    }

    @Test("Single node produces correct position and radius")
    func testSingleNode() {
        let id = UUID()
        let pos = SIMD3<Float>(1, 2, 3)
        let frame = buildSceneNodeFrame(makeInput(
            nodes: [(id: id, project: "test", importance: 1)],
            positions: [id: pos],
            projectColors: ["test": SIMD3<Float>(0.5, 0.5, 0.5)]
        ))
        #expect(frame.packInputs.count == 1)
        #expect(frame.nodePositions.count == 1)
        let pack = frame.packInputs[0]
        #expect(abs(pack.position.x - 1) < 0.001)
        #expect(abs(pack.position.y - 2) < 0.001)
        #expect(abs(pack.position.z - 3) < 0.001)
        #expect(abs(pack.baseRadius - defaultRadius) < 0.001)
        // state = 0 (normal), not dimmed, intensity = 0
        #expect(abs(pack.packedState - 0.0) < 0.001)
    }

    @Test("Hub nodes get 1.6x radius")
    func testHubRadius() {
        let id = UUID()
        let frame = buildSceneNodeFrame(makeInput(
            nodes: [(id: id, project: "p", importance: 1)],
            positions: [id: .zero],
            hubs: [id]
        ))
        let expected = defaultRadius * 1.6
        #expect(abs(frame.packInputs[0].baseRadius - expected) < 0.001)
    }

    @Test("Importance scales radius: importance=5 → 1.32x")
    func testImportanceScaling() {
        let id = UUID()
        let frame = buildSceneNodeFrame(makeInput(
            nodes: [(id: id, project: "p", importance: 5)],
            positions: [id: .zero]
        ))
        // 0.04 * (1.0 + (5-1) * 0.08) = 0.04 * 1.32 = 0.0528
        let expected: Float = 0.04 * 1.32
        #expect(abs(frame.packInputs[0].baseRadius - expected) < 0.001)
    }

    @Test("Selected node gets white color and state=1")
    func testSelectedNodeWhite() {
        let id = UUID()
        let frame = buildSceneNodeFrame(makeInput(
            nodes: [(id: id, project: "p", importance: 1)],
            positions: [id: .zero],
            projectColors: ["p": SIMD3<Float>(1, 0, 0)],
            selectedNode: id
        ))
        let pack = frame.packInputs[0]
        #expect(abs(pack.baseColor.x - 1.0) < 0.001)
        #expect(abs(pack.baseColor.y - 1.0) < 0.001)
        #expect(abs(pack.baseColor.z - 1.0) < 0.001)
        // state=1, not dimmed, intensity=0 → packedState = 1.0
        #expect(abs(pack.packedState - 1.0) < 0.001)
    }

    @Test("Recall glow: state=2, intensity encoded")
    func testRecallGlow() {
        let id = UUID()
        let frame = buildSceneNodeFrame(makeInput(
            nodes: [(id: id, project: "p", importance: 1)],
            positions: [id: .zero],
            glowingNodes: [id: 1.5] // in hold phase → intensity = 1.0
        ))
        let pack = frame.packInputs[0]
        // state=2, not dimmed, intensity=1.0 → 2.0 + 1.0*0.01 = 2.01
        #expect(abs(pack.packedState - 2.01) < 0.001)
    }

    @Test("Arrival glow: state=3")
    func testArrivalGlow() {
        let id = UUID()
        let frame = buildSceneNodeFrame(makeInput(
            nodes: [(id: id, project: "p", importance: 1)],
            positions: [id: .zero],
            newNodes: [id: 1.5] // in hold phase → intensity = 1.0
        ))
        let pack = frame.packInputs[0]
        // state=3, intensity=1.0 → 3.0 + 0.01 = 3.01
        #expect(abs(pack.packedState - 3.01) < 0.001)
    }

    @Test("Search dimming: non-match gets dimmed offset")
    func testSearchDimming() {
        let id = UUID()
        let matchId = UUID()
        let frame = buildSceneNodeFrame(makeInput(
            nodes: [(id: id, project: "p", importance: 1)],
            positions: [id: .zero],
            isSearchActive: true,
            searchMatchIds: [matchId] // id is NOT a match
        ))
        let pack = frame.packInputs[0]
        // state=0, dimmed=true, intensity=0 → 0 + 10.0 + 0 = 10.0
        #expect(abs(pack.packedState - 10.0) < 0.001)
    }

    @Test("Search match: state=4")
    func testSearchMatch() {
        let id = UUID()
        let frame = buildSceneNodeFrame(makeInput(
            nodes: [(id: id, project: "p", importance: 1)],
            positions: [id: .zero],
            isSearchActive: true,
            searchMatchIds: [id]
        ))
        let pack = frame.packInputs[0]
        // state=4, not dimmed (it's a match), intensity=0 → 4.0
        #expect(abs(pack.packedState - 4.0) < 0.001)
    }

    @Test("Birthing: state=6, intensity from elapsed")
    func testBirthing() {
        let id = UUID()
        let frame = buildSceneNodeFrame(makeInput(
            nodes: [(id: id, project: "p", importance: 1)],
            positions: [id: .zero],
            birthingElapsed: [id: 0.75] // 0.75/1.5 = 0.5
        ))
        let pack = frame.packInputs[0]
        // state=6, intensity=0.5 → 6.0 + 0.5*0.01 = 6.005
        #expect(abs(pack.packedState - 6.005) < 0.001)
    }

    @Test("Completed births: elapsed >= 1.5 appears in completedBirths")
    func testCompletedBirths() {
        let id = UUID()
        let frame = buildSceneNodeFrame(makeInput(
            nodes: [(id: id, project: "p", importance: 1)],
            positions: [id: .zero],
            birthingElapsed: [id: 2.0] // >= 1.5 → completed
        ))
        #expect(frame.completedBirths.contains(id))
    }

    @Test("Project centroids: 3 nodes same project → centroid at average")
    func testProjectCentroids() {
        let ids = (0..<3).map { _ in UUID() }
        let positions: [UUID: SIMD3<Float>] = [
            ids[0]: SIMD3<Float>(0, 0, 0),
            ids[1]: SIMD3<Float>(3, 6, 9),
            ids[2]: SIMD3<Float>(6, 3, 0),
        ]
        let frame = buildSceneNodeFrame(makeInput(
            nodes: ids.map { (id: $0, project: "proj", importance: 1) },
            positions: positions
        ))
        guard let centroid = frame.projectCentroids["proj"] else {
            Issue.record("Missing centroid for 'proj'")
            return
        }
        #expect(abs(centroid.centroid.x - 3.0) < 0.001)
        #expect(abs(centroid.centroid.y - 3.0) < 0.001)
        #expect(abs(centroid.centroid.z - 3.0) < 0.001)
        #expect(centroid.count == 3)
        #expect(abs(centroid.maxY - 6.0) < 0.001)
    }

    @Test("State priority: birthing > selected > searchMatch > dimmed")
    func testStatePriority() {
        let id = UUID()
        // Node is birthing AND selected — birthing should win
        let frame = buildSceneNodeFrame(makeInput(
            nodes: [(id: id, project: "p", importance: 1)],
            positions: [id: .zero],
            selectedNode: id,
            birthingElapsed: [id: 0.5]
        ))
        let state = floor(frame.packInputs[0].packedState)
        #expect(state == 6) // birthing state
    }

    @Test("Depth range from camera distance")
    func testDepthRange() {
        let ids = [UUID(), UUID()]
        let camPos = SIMD3<Float>(0, 0, 0)
        let frame = buildSceneNodeFrame(makeInput(
            nodes: ids.map { (id: $0, project: "p", importance: 1) },
            positions: [
                ids[0]: SIMD3<Float>(10, 0, 0),  // distance = 10
                ids[1]: SIMD3<Float>(0, 50, 0),  // distance = 50
            ],
            cameraPosition: camPos
        ))
        #expect(abs(frame.minDepth - 10.0) < 0.001)
        #expect(abs(frame.maxDepth - 50.0) < 0.001)
    }

    @Test("Missing position skips node")
    func testMissingPosition() {
        let id = UUID()
        let frame = buildSceneNodeFrame(makeInput(
            nodes: [(id: id, project: "p", importance: 1)],
            positions: [:] // no position for id
        ))
        #expect(frame.packInputs.isEmpty)
    }
}
