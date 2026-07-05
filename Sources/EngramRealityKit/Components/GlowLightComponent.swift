import RealityKit
import Foundation
import simd

/// Per-node point light for glowing/arriving/search-matched nodes.
public struct GlowLightComponent: Component {
    /// Node this light is tracking.
    public var nodeId: UUID
    /// Light intensity (0 = deactivated, pooled for reuse).
    public var intensity: Float
    /// Light color.
    public var color: SIMD3<Float>

    public init(nodeId: UUID, intensity: Float = 0, color: SIMD3<Float> = .one) {
        self.nodeId = nodeId
        self.intensity = intensity
        self.color = color
    }
}
