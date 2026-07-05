import RealityKit
import simd
import EngramSceneKit

/// Syncs RealityKit PerspectiveCamera transform from CameraProvider state.
@MainActor
public final class CameraSystem {

    public init() {}

    /// Update the RealityKit camera entity to match CameraState.
    public func update(camera: PerspectiveCamera, state: CameraState, scaleFactor: Float) {
        let pos = state.cameraPosition * scaleFactor
        let target = state.cameraTarget * scaleFactor

        // Build look-at transform
        let forward = normalize(target - pos)
        let worldUp = SIMD3<Float>(0, 1, 0)
        let right = normalize(cross(forward, worldUp))
        let up = cross(right, forward)

        // RealityKit camera looks along -Z in its local frame
        // So we construct a rotation that maps:
        //   local +X → right
        //   local +Y → up
        //   local -Z → forward (i.e., local +Z → -forward)
        var rotMatrix = simd_float3x3()
        rotMatrix.columns.0 = right
        rotMatrix.columns.1 = up
        rotMatrix.columns.2 = -forward

        camera.transform = Transform(
            scale: .one,
            rotation: simd_quatf(rotMatrix),
            translation: pos
        )

        // Set field of view
        camera.camera.fieldOfViewInDegrees = state.fovDegrees
    }
}
