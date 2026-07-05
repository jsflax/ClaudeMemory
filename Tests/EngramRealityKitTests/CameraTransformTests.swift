import Testing
import Foundation
import simd
@testable import EngramRealityKit
import EngramSceneKit

@Suite("Camera Transform Tests")
struct CameraTransformTests {

    @Test("CameraState default values")
    func defaultValues() {
        let state = CameraState()
        #expect(state.azimuth == 0)
        #expect(abs(state.elevation - 0.3) < 0.001)
        #expect(state.cameraTarget == .zero)
        #expect(state.orbitRadius == 2.0)
        #expect(abs(state.scaleFactor - 0.005) < 0.0001)
        #expect(state.fovDegrees == 60)
    }

    @Test("CameraState camera position at azimuth 0")
    func cameraPositionAzimuth0() {
        let state = CameraState(azimuth: 0, elevation: 0, orbitRadius: 100)
        let pos = state.cameraPosition
        // At azimuth=0, elevation=0: camera is at (0, 0, 100)
        #expect(abs(pos.x) < 0.01)
        #expect(abs(pos.y) < 0.01)
        #expect(abs(pos.z - 100) < 0.01)
    }

    @Test("CameraState camera position at azimuth pi/2")
    func cameraPositionAzimuthHalfPi() {
        let state = CameraState(azimuth: Float.pi / 2, elevation: 0, orbitRadius: 100)
        let pos = state.cameraPosition
        // At azimuth=pi/2, elevation=0: camera is at (100, 0, 0)
        #expect(abs(pos.x - 100) < 0.1)
        #expect(abs(pos.y) < 0.01)
        #expect(abs(pos.z) < 0.1)
    }

    @Test("CameraState elevation moves camera up")
    func elevationMovesUp() {
        let state = CameraState(azimuth: 0, elevation: Float.pi / 4, orbitRadius: 100)
        let pos = state.cameraPosition
        #expect(pos.y > 50)  // elevation ~45° should have significant Y
    }

    @Test("CameraState forward direction points at target")
    func forwardPointsAtTarget() {
        let state = CameraState(
            azimuth: 0.5,
            elevation: 0.3,
            cameraTarget: SIMD3<Float>(10, 20, 30),
            orbitRadius: 200
        )
        let forward = state.forward
        // Forward should point from camera to target
        let expectedDir = normalize(state.cameraTarget - state.cameraPosition)
        #expect(abs(forward.x - expectedDir.x) < 0.01)
        #expect(abs(forward.y - expectedDir.y) < 0.01)
        #expect(abs(forward.z - expectedDir.z) < 0.01)
    }

    @Test("CameraState view matrix is valid")
    func viewMatrixValid() {
        let state = CameraState()
        let view = state.viewMatrix()
        // View matrix should not be identity (camera is offset)
        #expect(view[3][0] != 0 || view[3][1] != 0 || view[3][2] != 0)
    }

    @Test("CameraState projection matrix has correct FOV")
    func projectionMatrix() {
        let state = CameraState(fovDegrees: 60)
        let proj = state.projectionMatrix(aspect: 1.0)
        // Y scale should be 1/tan(30°) ≈ 1.732
        let expectedYScale = 1.0 / tan(Float.pi / 6)
        #expect(abs(proj[1][1] - expectedYScale) < 0.01)
    }
}
