import RealityKit

/// Marks the entity holding the batched edge LowLevelMesh.
public struct EdgeBatchComponent: Component {
    /// Number of active edge instances.
    public var edgeCount: Int = 0
    /// Topology version at last mesh rebuild.
    public var topologyVersion: UInt64 = 0

    public init() {}
}
