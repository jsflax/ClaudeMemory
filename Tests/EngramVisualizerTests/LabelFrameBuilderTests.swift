import Foundation
import Testing
import simd
import CEngramSceneTypes
import EngramSceneKit

@Suite("SceneLabelFrame Builder")
struct SceneLabelFrameBuilderTests {

    let sf: Float = 1.0 / 200.0
    let nodeRadius: Float = 0.04
    let aspectCorrection: Float = 1.0

    private func makeInput(
        nodeLabels: [(id: UUID, position: SIMD3<Float>, isHub: Bool,
                       importance: Int, isSelected: Bool,
                       isSearchMatch: Bool, searchDimmed: Bool,
                       isExpandedChild: Bool)] = [],
        atlasRects: [UUID: AtlasRect] = [:],
        projectCentroids: [String: (centroid: SIMD3<Float>, maxY: Float, count: Int)] = [:],
        projectLabelRects: [String: AtlasRect] = [:],
        projectColors: [String: SIMD3<Float>] = [:],
        galaxyLabels: [(name: String, center: SIMD3<Float>)] = [],
        galaxyLabelRects: [String: AtlasRect] = [:],
        topicGroups: [(topic: String, project: String, memberPositions: [SIMD3<Float>])] = [],
        topicLabelRects: [String: AtlasRect] = [:],
        maxLabels: Int = 2048
    ) -> SceneLabelFrameInput {
        SceneLabelFrameInput(
            nodeLabels: nodeLabels, atlasRects: atlasRects,
            projectCentroids: projectCentroids, projectLabelRects: projectLabelRects,
            projectColors: projectColors, galaxyLabels: galaxyLabels,
            galaxyLabelRects: galaxyLabelRects, topicGroups: topicGroups,
            topicLabelRects: topicLabelRects,
            scaleFactor: sf, nodeRadius: nodeRadius,
            aspectCorrection: aspectCorrection, maxLabels: maxLabels
        )
    }

    @Test("Node label has correct anchor offset")
    func testNodeLabel() {
        let id = UUID()
        let pos = SIMD3<Float>(200, 400, 0) // scaled: (1, 2, 0)
        let rect = AtlasRect(u0: 0.0, v0: 0.0, u1: 0.5, v1: 0.1)
        let frame = buildSceneLabelFrame(makeInput(
            nodeLabels: [(id: id, position: pos, isHub: false,
                          importance: 1, isSelected: false,
                          isSearchMatch: false, searchDimmed: false,
                          isExpandedChild: false)],
            atlasRects: [id: rect]
        ))
        #expect(frame.count == 1)
        let inst = frame.instances[0]
        // anchor = pos * sf + (0, nodeRadius * 1.8, 0)
        #expect(abs(inst.anchor.x - 1.0) < 0.001)
        #expect(abs(inst.anchor.y - (2.0 + nodeRadius * 1.8)) < 0.001)
    }

    @Test("Selected label: flags bit 0, larger halfH")
    func testSelectedLabel() {
        let id = UUID()
        let rect = AtlasRect(u0: 0, v0: 0, u1: 1, v1: 1)
        let frame = buildSceneLabelFrame(makeInput(
            nodeLabels: [(id: id, position: .zero, isHub: false,
                          importance: 1, isSelected: true,
                          isSearchMatch: false, searchDimmed: false,
                          isExpandedChild: false)],
            atlasRects: [id: rect]
        ))
        let inst = frame.instances[0]
        #expect(inst.flags & 1 == 1)
        #expect(abs(inst.halfH - 0.025) < 0.001)
    }

    @Test("Search match label: flags bit 1")
    func testSearchMatchLabel() {
        let id = UUID()
        let rect = AtlasRect(u0: 0, v0: 0, u1: 1, v1: 1)
        let frame = buildSceneLabelFrame(makeInput(
            nodeLabels: [(id: id, position: .zero, isHub: false,
                          importance: 1, isSelected: false,
                          isSearchMatch: true, searchDimmed: false,
                          isExpandedChild: false)],
            atlasRects: [id: rect]
        ))
        #expect(frame.instances[0].flags & 2 == 2)
    }

    @Test("Project label positioned at centroid maxY + offset")
    func testProjectLabel() {
        let rect = AtlasRect(u0: 0, v0: 0, u1: 0.5, v1: 0.1)
        let frame = buildSceneLabelFrame(makeInput(
            projectCentroids: ["proj": (centroid: SIMD3<Float>(100, 50, 0), maxY: 200, count: 5)],
            projectLabelRects: ["proj": rect],
            projectColors: ["proj": SIMD3<Float>(0.5, 0.5, 1.0)]
        ))
        #expect(frame.count == 1)
        let inst = frame.instances[0]
        // labelY = 200 * sf + 0.15 = 1.0 + 0.15 = 1.15
        #expect(abs(inst.anchor.y - 1.15) < 0.01)
        #expect(abs(inst.halfH - 0.20) < 0.001)
    }

    @Test("Max labels cap at 2048")
    func testMaxLabels() {
        var labels: [(id: UUID, position: SIMD3<Float>, isHub: Bool,
                       importance: Int, isSelected: Bool,
                       isSearchMatch: Bool, searchDimmed: Bool,
                       isExpandedChild: Bool)] = []
        var rects: [UUID: AtlasRect] = [:]
        let rect = AtlasRect(u0: 0, v0: 0, u1: 1, v1: 1)
        for _ in 0..<3000 {
            let id = UUID()
            labels.append((id: id, position: .zero, isHub: false,
                           importance: 1, isSelected: false,
                           isSearchMatch: false, searchDimmed: false,
                           isExpandedChild: false))
            rects[id] = rect
        }
        let frame = buildSceneLabelFrame(makeInput(
            nodeLabels: labels, atlasRects: rects, maxLabels: 2048
        ))
        #expect(frame.count <= 2048)
    }
}
