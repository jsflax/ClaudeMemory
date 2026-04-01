import Testing
import Foundation
@testable import EngramRealityKit
import EngramSceneKit

@Suite("Visual State Encoding Tests")
struct VisualStateEncodingTests {

    @Test("packNodeState normal state")
    func normalState() {
        let packed = packNodeState(stateType: 0.0, searchDimmed: false, intensity: 0.0)
        #expect(packed == 0.0)
    }

    @Test("packNodeState selected with pulse")
    func selectedState() {
        let packed = packNodeState(stateType: 0.25, searchDimmed: false, intensity: 0.5)
        #expect(packed > 0.25)
        #expect(packed < 0.26)
    }

    @Test("packNodeState search dimmed adds 10.0")
    func searchDimmed() {
        let normal = packNodeState(stateType: 0.0, searchDimmed: false, intensity: 0.0)
        let dimmed = packNodeState(stateType: 0.0, searchDimmed: true, intensity: 0.0)
        #expect(dimmed - normal == 10.0)
    }

    @Test("recallGlowIntensity curve shape")
    func recallGlowCurve() {
        // Before start
        #expect(recallGlowIntensity(elapsed: -0.1) == 0)
        // Fade-in (0..1s) — cubic, starts from 0
        #expect(recallGlowIntensity(elapsed: 0) == 0)
        let mid = recallGlowIntensity(elapsed: 0.5)
        #expect(mid > 0 && mid < 0.5) // cubic is below linear
        // Peak
        #expect(recallGlowIntensity(elapsed: 1.0) == 1.0)
        // Hold (1..2.5s)
        #expect(recallGlowIntensity(elapsed: 1.5) == 1.0)
        #expect(recallGlowIntensity(elapsed: 2.4) == 1.0)
        // Fade-out (2.5..4.5s)
        let fadeout = recallGlowIntensity(elapsed: 3.0)
        #expect(fadeout > 0 && fadeout < 1.0)
        // End
        #expect(recallGlowIntensity(elapsed: 4.5) == 0)
        #expect(recallGlowIntensity(elapsed: 5.0) == 0)
    }

    @Test("arrivalGlowIntensity curve shape")
    func arrivalGlowCurve() {
        #expect(arrivalGlowIntensity(elapsed: -0.1) == 0)
        #expect(arrivalGlowIntensity(elapsed: 0) == 0)
        #expect(arrivalGlowIntensity(elapsed: 0.8) == 1.0)
        #expect(arrivalGlowIntensity(elapsed: 2.0) == 1.0)
        let fadeout = arrivalGlowIntensity(elapsed: 4.0)
        #expect(fadeout > 0 && fadeout < 1.0)
        #expect(arrivalGlowIntensity(elapsed: 5.8) == 0)
    }

    @Test("birthIntensity linear ramp")
    func birthIntensityRamp() {
        #expect(birthIntensity(elapsed: 0) == 0)
        #expect(abs(birthIntensity(elapsed: 0.75) - 0.5) < 0.01)
        #expect(birthIntensity(elapsed: 1.5) == 1.0)
        #expect(birthIntensity(elapsed: 3.0) == 1.0) // clamped
        #expect(birthIntensity(elapsed: -1.0) == 0)   // clamped
    }
}
