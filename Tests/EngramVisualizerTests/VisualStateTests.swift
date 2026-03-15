import Foundation
import Testing
import simd
@testable import EngramSceneKit

// MARK: - Phase 1: Visual State Tests (TDD — written before implementation)

@Suite("Visual State")
struct VisualStateTests {

    // MARK: - Recall Glow

    @Test("Recall glow fade-in: cubic at 0.5s")
    func testRecallGlowFadeIn() {
        let result = recallGlowIntensity(elapsed: 0.5)
        // t = 0.5/1.0 = 0.5, cubic = 0.125
        #expect(abs(result - 0.125) < 0.001)
    }

    @Test("Recall glow hold: 1.0 at 1.5s")
    func testRecallGlowHold() {
        let result = recallGlowIntensity(elapsed: 1.5)
        #expect(abs(result - 1.0) < 0.001)
    }

    @Test("Recall glow fade-out: quadratic at 3.5s")
    func testRecallGlowFadeOut() {
        // elapsed=3.5, fadeIn=1.0, hold=1.5, fadeOut=2.0
        // past = 3.5 - 1.0 - 1.5 = 1.0, t = 1.0 - 1.0/2.0 = 0.5, quadratic = 0.25
        let result = recallGlowIntensity(elapsed: 3.5)
        #expect(abs(result - 0.25) < 0.001)
    }

    @Test("Recall glow expired: 0 at 5.0s")
    func testRecallGlowExpired() {
        let result = recallGlowIntensity(elapsed: 5.0)
        #expect(result == 0)
    }

    @Test("Recall glow at 0: 0")
    func testRecallGlowStart() {
        let result = recallGlowIntensity(elapsed: 0)
        #expect(result == 0)
    }

    // MARK: - Arrival Glow

    @Test("Arrival glow phases: fade-in 0.8s, hold 2.0s, fade-out 3.0s")
    func testArrivalPhases() {
        // During fade-in: t = 0.4/0.8 = 0.5, cubic = 0.125
        #expect(abs(arrivalGlowIntensity(elapsed: 0.4) - 0.125) < 0.001)

        // During hold (at 1.5s, which is within 0.8+2.0=2.8): 1.0
        #expect(abs(arrivalGlowIntensity(elapsed: 1.5) - 1.0) < 0.001)

        // During fade-out: elapsed=4.3, past hold = 4.3-0.8-2.0 = 1.5
        // t = 1.0 - 1.5/3.0 = 0.5, quadratic = 0.25
        #expect(abs(arrivalGlowIntensity(elapsed: 4.3) - 0.25) < 0.001)

        // Expired
        #expect(arrivalGlowIntensity(elapsed: 6.0) == 0)
    }

    // MARK: - Birth Intensity

    @Test("Birth intensity: linear 0→1 over 1.5s")
    func testBirthLinear() {
        #expect(abs(birthIntensity(elapsed: 0.75) - 0.5) < 0.001)
        #expect(abs(birthIntensity(elapsed: 1.5) - 1.0) < 0.001)
        // Clamped at 1.0
        #expect(abs(birthIntensity(elapsed: 2.0) - 1.0) < 0.001)
    }

    @Test("Birth intensity at 0: 0")
    func testBirthZero() {
        #expect(birthIntensity(elapsed: 0) == 0)
    }

    // MARK: - Pack Node State

    @Test("Pack node state encoding: state + intensity * 0.01")
    func testPackNodeStateEncoding() {
        let result = packNodeState(stateType: 2.0, searchDimmed: false, intensity: 0.5)
        // 2.0 + 0.0 + 0.5 * 0.01 = 2.005
        #expect(abs(result - 2.005) < 0.0001)
    }

    @Test("Pack node state dimmed: adds 10.0 offset")
    func testPackNodeStateDimmed() {
        let result = packNodeState(stateType: 0.0, searchDimmed: true, intensity: 0.0)
        #expect(abs(result - 10.0) < 0.0001)
    }

    @Test("Pack node state combined: state + dimmed + intensity")
    func testPackNodeStateCombined() {
        let result = packNodeState(stateType: 3.0, searchDimmed: true, intensity: 1.0)
        // 3.0 + 10.0 + 1.0 * 0.01 = 13.01
        #expect(abs(result - 13.01) < 0.0001)
    }
}
