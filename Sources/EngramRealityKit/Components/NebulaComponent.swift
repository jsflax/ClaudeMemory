import RealityKit
import simd

/// Per-project nebula particle emitter configuration.
public struct NebulaComponent: Component {
    /// Project this nebula belongs to.
    public var project: String
    /// Nebula tint color (RGBA).
    public var color: SIMD4<Float>
    /// Cluster radius in world units.
    public var radius: Float
    /// Whether this nebula needs its emitter reconfigured.
    public var needsUpdate: Bool = true

    public init(project: String, color: SIMD4<Float>, radius: Float) {
        self.project = project
        self.color = color
        self.radius = radius
    }
}
