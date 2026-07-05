import RealityKit

/// Marks the entity holding the batched label LowLevelMesh.
public struct LabelBatchComponent: Component {
    /// Number of active label instances.
    public var labelCount: Int = 0
    /// Atlas texture version — changes trigger atlas regeneration.
    public var atlasVersion: UInt64 = 0

    public init() {}
}
