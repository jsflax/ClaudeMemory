import Foundation
import Testing
import SwiftUI
import simd
import EngramSceneKit

@Suite("Color Conversion")
struct ColorConversionTests {

    @Test("White color converts to (1, 1, 1)")
    func testWhite() {
        let result = colorToSIMD3(.white)
        #expect(abs(result.x - 1.0) < 0.01)
        #expect(abs(result.y - 1.0) < 0.01)
        #expect(abs(result.z - 1.0) < 0.01)
    }

    @Test("Custom sRGB red converts correctly")
    func testRed() {
        // SwiftUI .red is a display-P3 color, not sRGB (1,0,0).
        // Use an explicit sRGB color for deterministic testing.
        let srgbRed = Color(red: 1, green: 0, blue: 0)
        let result = colorToSIMD3(srgbRed)
        #expect(abs(result.x - 1.0) < 0.05)
        #expect(abs(result.y) < 0.05)
        #expect(abs(result.z) < 0.05)
    }

    @Test("Edge dimming applies 0.6x factor")
    func testEdgeDimming() {
        let nodeColor = SIMD3<Float>(1.0, 0.5, 0.2)
        let result = edgeColorSIMD3(nodeColor: nodeColor)
        #expect(abs(result.x - 0.6) < 0.001)
        #expect(abs(result.y - 0.3) < 0.001)
        #expect(abs(result.z - 0.12) < 0.001)
    }

    @Test("Nebula color boosts and clamps")
    func testNebulaColor() {
        let result = nebulaColorSIMD4(from: .white)
        // white (1,1,1) * 1.3 + 0.1 = 1.4 → clamped to 1.0
        #expect(abs(result.x - 1.0) < 0.01)
        #expect(abs(result.y - 1.0) < 0.01)
        #expect(abs(result.z - 1.0) < 0.01)
        #expect(abs(result.w - 0.25) < 0.01)
    }
}
