import Foundation
import Testing
import simd
import CEngramSceneTypes
import EngramSceneKit

@Suite("SceneNodeFrame Builder")
struct SceneNodeFrameBuilderTests {

    let defaultRadius: Float = 0.04

    private func makeInput(
        nodeIds: [UUID] = [],
        nodeProjects: [String] = [],
        nodeImportances: [Int] = [],
        nodePositions: [SIMD3<Float>] = [],
        hasPosition: [Bool] = [],
        nodeRadii: [Float] = [],
        nodeIsHub: [Bool] = [],
        projectColors: [String: SIMD3<Float>] = [:],
        selectedNodeIndex: Int? = nil,
        glowingNodes: [UUID: Float] = [:],
        newNodes: [UUID: Float] = [:],
        isSearchActive: Bool = false,
        searchMatchIndices: Set<Int> = [],
        inspectingIntensity: [UUID: Float] = [:],
        birthingElapsed: [UUID: Float] = [:],
        nodeRadius: Float = 0.04,
        cameraPosition: SIMD3<Float> = .zero
    ) -> SceneNodeFrameInput {
        SceneNodeFrameInput(
            nodeIds: nodeIds,
            nodeProjects: nodeProjects,
            nodeImportances: nodeImportances,
            nodePositions: nodePositions,
            hasPosition: hasPosition,
            nodeRadii: nodeRadii,
            nodeIsHub: nodeIsHub,
            projectColors: projectColors,
            selectedNodeIndex: selectedNodeIndex,
            glowingNodes: glowingNodes,
            newNodes: newNodes,
            isSearchActive: isSearchActive,
            searchMatchIndices: searchMatchIndices,
            inspectingIntensity: inspectingIntensity,
            birthingElapsed: birthingElapsed,
            nodeRadius: nodeRadius,
            cameraPosition: cameraPosition
        )
    }

    /// Convenience: build a single-node input from id, position, and options.
    private func makeSingleNodeInput(
        id: UUID = UUID(),
        position: SIMD3<Float> = .zero,
        hasPos: Bool = true,
        project: String = "p",
        importance: Int = 1,
        radius: Float? = nil,
        isHub: Bool = false,
        projectColors: [String: SIMD3<Float>] = [:],
        selectedNodeIndex: Int? = nil,
        glowingNodes: [UUID: Float] = [:],
        newNodes: [UUID: Float] = [:],
        isSearchActive: Bool = false,
        searchMatchIndices: Set<Int> = [],
        inspectingIntensity: [UUID: Float] = [:],
        birthingElapsed: [UUID: Float] = [:]
    ) -> SceneNodeFrameInput {
        let r = radius ?? (defaultRadius * (isHub ? 1.6 : 1.0) * (1.0 + Float(importance - 1) * 0.08))
        return makeInput(
            nodeIds: [id],
            nodeProjects: [project],
            nodeImportances: [importance],
            nodePositions: [position],
            hasPosition: [hasPos],
            nodeRadii: [r],
            nodeIsHub: [isHub],
            projectColors: projectColors,
            selectedNodeIndex: selectedNodeIndex,
            glowingNodes: glowingNodes,
            newNodes: newNodes,
            isSearchActive: isSearchActive,
            searchMatchIndices: searchMatchIndices,
            inspectingIntensity: inspectingIntensity,
            birthingElapsed: birthingElapsed
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
        let frame = buildSceneNodeFrame(makeSingleNodeInput(
            id: id,
            position: pos,
            project: "test",
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
        let expected = defaultRadius * 1.6
        let frame = buildSceneNodeFrame(makeSingleNodeInput(
            id: id,
            radius: expected,
            isHub: true
        ))
        #expect(abs(frame.packInputs[0].baseRadius - expected) < 0.001)
    }

    @Test("Importance scales radius: importance=5 → 1.32x")
    func testImportanceScaling() {
        let id = UUID()
        // 0.04 * (1.0 + (5-1) * 0.08) = 0.04 * 1.32 = 0.0528
        let expected: Float = 0.04 * 1.32
        let frame = buildSceneNodeFrame(makeSingleNodeInput(
            id: id,
            importance: 5,
            radius: expected
        ))
        #expect(abs(frame.packInputs[0].baseRadius - expected) < 0.001)
    }

    @Test("Selected node gets white color and state=1")
    func testSelectedNodeWhite() {
        let id = UUID()
        let frame = buildSceneNodeFrame(makeSingleNodeInput(
            id: id,
            project: "p",
            projectColors: ["p": SIMD3<Float>(1, 0, 0)],
            selectedNodeIndex: 0
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
        let frame = buildSceneNodeFrame(makeSingleNodeInput(
            id: id,
            glowingNodes: [id: 1.5] // in hold phase → intensity = 1.0
        ))
        let pack = frame.packInputs[0]
        // state=2, not dimmed, intensity=1.0 → 2.0 + 1.0*0.01 = 2.01
        #expect(abs(pack.packedState - 2.01) < 0.001)
    }

    @Test("Arrival glow: state=3")
    func testArrivalGlow() {
        let id = UUID()
        let frame = buildSceneNodeFrame(makeSingleNodeInput(
            id: id,
            newNodes: [id: 1.5] // in hold phase → intensity = 1.0
        ))
        let pack = frame.packInputs[0]
        // state=3, intensity=1.0 → 3.0 + 0.01 = 3.01
        #expect(abs(pack.packedState - 3.01) < 0.001)
    }

    @Test("Search dimming: non-match gets dimmed offset")
    func testSearchDimming() {
        let id = UUID()
        let frame = buildSceneNodeFrame(makeSingleNodeInput(
            id: id,
            isSearchActive: true,
            searchMatchIndices: [] // index 0 is NOT a match
        ))
        let pack = frame.packInputs[0]
        // state=0, dimmed=true, intensity=0 → 0 + 10.0 + 0 = 10.0
        #expect(abs(pack.packedState - 10.0) < 0.001)
    }

    @Test("Search match: state=4")
    func testSearchMatch() {
        let id = UUID()
        let frame = buildSceneNodeFrame(makeSingleNodeInput(
            id: id,
            isSearchActive: true,
            searchMatchIndices: [0] // index 0 IS a match
        ))
        let pack = frame.packInputs[0]
        // state=4, not dimmed (it's a match), intensity=0 → 4.0
        #expect(abs(pack.packedState - 4.0) < 0.001)
    }

    @Test("Birthing: state=6, intensity from elapsed")
    func testBirthing() {
        let id = UUID()
        let frame = buildSceneNodeFrame(makeSingleNodeInput(
            id: id,
            birthingElapsed: [id: 0.75] // 0.75/1.5 = 0.5
        ))
        let pack = frame.packInputs[0]
        // state=6, intensity=0.5 → 6.0 + 0.5*0.01 = 6.005
        #expect(abs(pack.packedState - 6.005) < 0.001)
    }

    @Test("Completed births: elapsed >= 1.5 appears in completedBirths")
    func testCompletedBirths() {
        let id = UUID()
        let frame = buildSceneNodeFrame(makeSingleNodeInput(
            id: id,
            birthingElapsed: [id: 2.0] // >= 1.5 → completed
        ))
        #expect(frame.completedBirths.contains(id))
    }

    @Test("Project centroids: 3 nodes same project → centroid at average")
    func testProjectCentroids() {
        let ids = (0..<3).map { _ in UUID() }
        let positions: [SIMD3<Float>] = [
            SIMD3<Float>(0, 0, 0),
            SIMD3<Float>(3, 6, 9),
            SIMD3<Float>(6, 3, 0),
        ]
        let frame = buildSceneNodeFrame(makeInput(
            nodeIds: ids,
            nodeProjects: ["proj", "proj", "proj"],
            nodeImportances: [1, 1, 1],
            nodePositions: positions,
            hasPosition: [true, true, true],
            nodeRadii: [0.04, 0.04, 0.04],
            nodeIsHub: [false, false, false]
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
        let frame = buildSceneNodeFrame(makeSingleNodeInput(
            id: id,
            selectedNodeIndex: 0,
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
            nodeIds: ids,
            nodeProjects: ["p", "p"],
            nodeImportances: [1, 1],
            nodePositions: [
                SIMD3<Float>(10, 0, 0),  // distance = 10
                SIMD3<Float>(0, 50, 0),  // distance = 50
            ],
            hasPosition: [true, true],
            nodeRadii: [0.04, 0.04],
            nodeIsHub: [false, false],
            cameraPosition: camPos
        ))
        #expect(abs(frame.minDepth - 10.0) < 0.001)
        #expect(abs(frame.maxDepth - 50.0) < 0.001)
    }

    @Test("Missing position skips node")
    func testMissingPosition() {
        let id = UUID()
        let frame = buildSceneNodeFrame(makeInput(
            nodeIds: [id],
            nodeProjects: ["p"],
            nodeImportances: [1],
            nodePositions: [.zero],
            hasPosition: [false], // no position for id
            nodeRadii: [0.04],
            nodeIsHub: [false]
        ))
        #expect(frame.packInputs.isEmpty)
    }
}
