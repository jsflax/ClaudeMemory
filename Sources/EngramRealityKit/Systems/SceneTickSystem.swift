import simd

/// First system each frame — drives data provider tick and detects topology changes.
@MainActor
public final class SceneTickSystem {
    private var lastTopologyVersion: UInt64 = 0

    public init() {}

    /// Returns true if topology changed this frame.
    public func tick(dataProvider: SceneDataProvider, dt: Float) -> Bool {
        dataProvider.tick(dt: dt)
        let changed = dataProvider.topologyVersion != lastTopologyVersion
        if changed {
            lastTopologyVersion = dataProvider.topologyVersion
        }
        return changed
    }
}
