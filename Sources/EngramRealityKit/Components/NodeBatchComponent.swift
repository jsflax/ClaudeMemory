import RealityKit

/// Marks the entity holding the batched node LowLevelMesh.
public struct NodeBatchComponent: Component {
    /// Number of active node instances.
    public var nodeCount: Int = 0
    /// Topology version at last mesh rebuild.
    public var topologyVersion: UInt64 = 0

    public init() {}
}
