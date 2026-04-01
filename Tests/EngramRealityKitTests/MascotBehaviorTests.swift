import Testing
import Foundation
import simd
@testable import EngramRealityKit

@Suite("Mascot Behavior Tests")
struct MascotBehaviorTests {

    @Test("MascotBehavior idle state equality")
    func idleEquality() {
        let a = MascotBehavior.idle(.awake)
        let b = MascotBehavior.idle(.awake)
        let c = MascotBehavior.idle(.drowsy)
        #expect(a == b)
        #expect(a != c)
    }

    @Test("MascotBehavior patrol state")
    func patrolState() {
        let id = UUID()
        let pos = SIMD3<Float>(1, 2, 3)
        let state = MascotBehavior.patrol(targetId: id, targetPos: pos)
        if case .patrol(let targetId, let targetPos) = state {
            #expect(targetId == id)
            #expect(targetPos == pos)
        } else {
            Issue.record("Expected patrol state")
        }
    }

    @Test("MascotBehavior hover timer")
    func hoverTimer() {
        let id = UUID()
        let state = MascotBehavior.hover(nodeId: id, timer: 2.5)
        if case .hover(let nodeId, let timer) = state {
            #expect(nodeId == id)
            #expect(timer == 2.5)
        } else {
            Issue.record("Expected hover state")
        }
    }

    @Test("MascotBehavior conjure phase")
    func conjurePhase() {
        let id = UUID()
        let state = MascotBehavior.conjure(nodeId: id, phase: 1.5)
        #expect(state.isConjuring)
        #expect(!MascotBehavior.idle(.awake).isConjuring)
    }

    @Test("MascotTask priority ordering")
    func taskPriority() {
        let tasks: [RKMascotSystem.MascotTask] = [
            .idle,
            .patrol,
            .react(nodeId: UUID(), type: .headTurn),
            .delete(nodeId: UUID(), position: .zero),
            .create(nodeId: UUID(), position: .zero),
        ]

        let sorted = tasks.sorted(by: >)
        // create > delete > react > patrol > idle
        if case .create = sorted[0] {} else { Issue.record("Expected create first") }
        if case .delete = sorted[1] {} else { Issue.record("Expected delete second") }
        if case .react = sorted[2] {} else { Issue.record("Expected react third") }
        if case .patrol = sorted[3] {} else { Issue.record("Expected patrol fourth") }
        if case .idle = sorted[4] {} else { Issue.record("Expected idle fifth") }
    }

    @Test("MascotJointAngles custom values")
    func jointAnglesCustom() {
        let angles = MascotJointAngles(
            leftArmPitch: 0.5,
            rightArmPitch: -0.3,
            eyePulse: 1.0,
            bottomThruster: 0.8,
            bodyBob: 0.02
        )
        #expect(angles.leftArmPitch == 0.5)
        #expect(angles.rightArmPitch == -0.3)
        #expect(angles.eyePulse == 1.0)
        #expect(angles.bottomThruster == 0.8)
        #expect(angles.bodyBob == 0.02)
    }

    @Test("MascotBehavior idle sub-states")
    func idleSubStates() {
        let awake = MascotBehavior.IdleSubState.awake
        let drowsy = MascotBehavior.IdleSubState.drowsy
        let sleeping = MascotBehavior.IdleSubState.sleeping
        #expect(awake != drowsy)
        #expect(drowsy != sleeping)
        #expect(awake != sleeping)
    }

    @Test("MascotBehavior chatting state")
    func chattingState() {
        let state = MascotBehavior.chatting
        #expect(state == .chatting)
        #expect(!state.isConjuring)
    }
}
