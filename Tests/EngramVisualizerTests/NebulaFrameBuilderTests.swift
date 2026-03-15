import Foundation
import Testing
import simd
import CEngramSceneTypes
@testable import EngramSceneKit

@Suite("NebulaFrame Builder")
struct NebulaFrameBuilderTests {

    @Test("2 projects → 2 nebula groups")
    func testNebulaFromCentroids() {
        let centroids: [String: (centroid: SIMD3<Float>, maxY: Float, count: Int)] = [
            "project1": (centroid: SIMD3<Float>(100, 50, 0), maxY: 80, count: 10),
            "project2": (centroid: SIMD3<Float>(-100, -50, 0), maxY: 30, count: 5),
        ]
        let colors: [String: SIMD4<Float>] = [
            "project1": SIMD4<Float>(1, 0, 0, 0.25),
            "project2": SIMD4<Float>(0, 0, 1, 0.25),
        ]
        let groups = buildNebulaGroups(
            projectCentroids: centroids,
            projectColors: colors,
            scaleFactor: 1.0 / 200.0
        )
        #expect(groups.count == 2)
    }

    @Test("Projects with <2 members are skipped")
    func testMinMembers() {
        let centroids: [String: (centroid: SIMD3<Float>, maxY: Float, count: Int)] = [
            "solo": (centroid: .zero, maxY: 0, count: 1),
        ]
        let groups = buildNebulaGroups(
            projectCentroids: centroids,
            projectColors: [:],
            scaleFactor: 1.0 / 200.0
        )
        #expect(groups.isEmpty)
    }

    @Test("Nebula radius scales with member count")
    func testRadiusScaling() {
        let centroids: [String: (centroid: SIMD3<Float>, maxY: Float, count: Int)] = [
            "big": (centroid: .zero, maxY: 0, count: 100),
        ]
        let sf: Float = 1.0 / 200.0
        let groups = buildNebulaGroups(
            projectCentroids: centroids,
            projectColors: [:],
            scaleFactor: sf
        )
        #expect(groups.count == 1)
        // radius = max(30, 100*2) * sf = 200 * 0.005 = 1.0
        #expect(abs(groups[0].radius - 1.0) < 0.001)
    }
}
