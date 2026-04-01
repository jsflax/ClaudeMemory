import EngramSceneKit

/// Abstracts camera state for platform-agnostic rendering.
///
/// On macOS, CameraController provides orbit/pan/dolly.
/// On visionOS, the head IS the camera — isHeadTracked returns true.
@MainActor
public protocol CameraProvider: AnyObject {
    /// Current camera state snapshot.
    var cameraState: CameraState { get }
    /// Whether the camera is actively moving (user interaction or animation).
    var isMoving: Bool { get }
    /// True on visionOS where the camera tracks the user's head.
    var isHeadTracked: Bool { get }
}
