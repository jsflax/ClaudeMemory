import RealityKit

/// Marks the entity holding flow particle billboard quads along selected edges.
public struct FlowParticleComponent: Component {
    /// Number of active edges with flow particles.
    public var edgeCount: Int = 0

    public init() {}
}
