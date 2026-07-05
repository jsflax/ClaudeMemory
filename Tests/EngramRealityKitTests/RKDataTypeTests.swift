import Testing
import Foundation
import simd
@testable import EngramRealityKit

@Suite("RKDataType Tests")
struct RKDataTypeTests {

    @Test("RKNodeSnapshot construction and field mapping")
    func nodeSnapshotFields() {
        let id = UUID()
        let node = RKNodeSnapshot(
            id: id,
            project: "TestProject",
            topic: "architecture",
            label: "my_node",
            importance: 3,
            isHub: true
        )
        #expect(node.id == id)
        #expect(node.project == "TestProject")
        #expect(node.topic == "architecture")
        #expect(node.label == "my_node")
        #expect(node.importance == 3)
        #expect(node.isHub == true)
    }

    @Test("RKEdgeSnapshot construction and field mapping")
    func edgeSnapshotFields() {
        let id = UUID()
        let src = UUID()
        let tgt = UUID()
        let edge = RKEdgeSnapshot(
            id: id,
            sourceId: src,
            targetId: tgt,
            relation: "part_of"
        )
        #expect(edge.id == id)
        #expect(edge.sourceId == src)
        #expect(edge.targetId == tgt)
        #expect(edge.relation == "part_of")
    }

    @Test("LODTier from distance")
    func lodTierClassification() {
        #expect(LODTier.from(distance: 100) == .near)
        #expect(LODTier.from(distance: 499) == .near)
        #expect(LODTier.from(distance: 500) == .mid)
        #expect(LODTier.from(distance: 1999) == .mid)
        #expect(LODTier.from(distance: 2000) == .far)
        #expect(LODTier.from(distance: 4999) == .far)
        #expect(LODTier.from(distance: 5000) == .culled)
        #expect(LODTier.from(distance: 10000) == .culled)
    }

    @Test("LODTier ordering")
    func lodTierOrdering() {
        #expect(LODTier.near < LODTier.mid)
        #expect(LODTier.mid < LODTier.far)
        #expect(LODTier.far < LODTier.culled)
    }

    @Test("VisibleSet empty initialization")
    func visibleSetEmpty() {
        let vs = VisibleSet()
        #expect(vs.nearNodes.isEmpty)
        #expect(vs.midNodes.isEmpty)
        #expect(vs.farNodes.isEmpty)
        #expect(vs.totalNodeCount == 0)
    }

    @Test("VisibleSet totalNodeCount")
    func visibleSetTotalCount() {
        let vs = VisibleSet(
            nearNodes: [0, 1, 2],
            midNodes: [3, 4],
            farNodes: [5]
        )
        #expect(vs.totalNodeCount == 6)
    }

    @Test("MascotBehavior idle states")
    func mascotBehaviorIdle() {
        let state = MascotBehavior.idle(.awake)
        #expect(state == .idle(.awake))
        #expect(state != .idle(.drowsy))
    }

    @Test("MascotJointAngles defaults")
    func jointAnglesDefaults() {
        let angles = MascotJointAngles()
        #expect(angles.leftArmPitch == 0)
        #expect(angles.rightArmPitch == 0)
        #expect(angles.eyePulse == 0)
        #expect(angles.bottomThruster == 0)
        #expect(angles.bodyBob == 0)
    }
}
