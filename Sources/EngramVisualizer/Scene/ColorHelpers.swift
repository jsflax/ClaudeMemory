import simd
import SwiftUI

/// Namespace for color conversion helpers used by MetalSceneManager.
/// Caches live on the caller; these functions are pure aside from the inout cache.
enum ColorHelpers {

    static func nodeColorFloat3(
        for project: String,
        colorMap: [String: Color],
        cache: inout [String: SIMD3<Float>]
    ) -> SIMD3<Float> {
        if let cached = cache[project] { return cached }
        let color = colorMap[project] ?? .gray
        let nsColor = NSColor(color).usingColorSpace(.sRGB) ?? NSColor(color)
        var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
        nsColor.getRed(&r, green: &g, blue: &b, alpha: &a)
        let c = SIMD3<Float>(Float(r), Float(g), Float(b))
        cache[project] = c
        return c
    }

    static func edgeColorFloat3(
        for project: String?,
        colorMap: [String: Color],
        cache: inout [String: SIMD3<Float>]
    ) -> SIMD3<Float> {
        let key = project ?? "__default"
        if let cached = cache[key] { return cached }
        if let project, let swiftColor = colorMap[project] {
            let nsColor = NSColor(swiftColor).usingColorSpace(.sRGB) ?? NSColor(swiftColor)
            var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
            nsColor.getRed(&r, green: &g, blue: &b, alpha: &a)
            let c = SIMD3<Float>(Float(r) * 0.6, Float(g) * 0.6, Float(b) * 0.6)
            cache[key] = c
            return c
        }
        let c = SIMD3<Float>(0.35, 0.35, 0.4)
        cache[key] = c
        return c
    }
}
