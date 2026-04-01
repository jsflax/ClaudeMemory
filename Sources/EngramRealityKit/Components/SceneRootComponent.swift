import RealityKit

/// Marks the root entity of the Engram scene graph.
public struct SceneRootComponent: Component {
    /// Current topology version — changes trigger mesh rebuilds.
    public var topologyVersion: UInt64 = 0
    /// Accumulated animation time in seconds.
    public var animationTime: Float = 0

    public init() {}
}
