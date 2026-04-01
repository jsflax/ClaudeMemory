import RealityKit
import simd
import Foundation

/// Per-project mascot entity state.
public struct MascotComponent: Component {
    /// Project this mascot belongs to.
    public var project: String
    /// Current behavioral state.
    public var state: MascotBehavior
    /// World-space position.
    public var position: SIMD3<Float>
    /// Body yaw in radians.
    public var yaw: Float
    /// Joint angles for skeleton animation.
    public var jointAngles: MascotJointAngles
    /// Project tint color.
    public var tint: SIMD3<Float>
    /// Current node the mascot is visiting (if any).
    public var targetNodeId: UUID?
    /// Accumulated time for animation.
    public var animationTime: Float = 0
    /// Arcane circle visibility intensity (0–1).
    public var arcaneIntensity: Float = 0
    /// Conjure orb visibility intensity (0–1).
    public var conjureOrbIntensity: Float = 0
    /// Holo screen visibility intensity (0–1).
    public var holoIntensity: Float = 0

    public init(
        project: String,
        state: MascotBehavior = .idle(.awake),
        position: SIMD3<Float> = .zero,
        yaw: Float = 0,
        jointAngles: MascotJointAngles = .init(),
        tint: SIMD3<Float> = SIMD3<Float>(0, 0.8, 1.0)
    ) {
        self.project = project
        self.state = state
        self.position = position
        self.yaw = yaw
        self.jointAngles = jointAngles
        self.tint = tint
    }
}
