import Foundation
import Testing
import simd
import EngramSceneKit

@Suite("CameraState")
struct CameraStateTests {

    @Test("cameraPosition orbits correctly around target")
    func testCameraPositionOrbit() {
        // Camera at azimuth=0, elevation=0 should be along +Z
        let cam = CameraState(azimuth: 0, elevation: 0, cameraTarget: .zero, orbitRadius: 100)
        let pos = cam.cameraPosition
        #expect(abs(pos.x) < 0.01, "x should be ~0 at azimuth=0")
        #expect(abs(pos.y) < 0.01, "y should be ~0 at elevation=0")
        #expect(abs(pos.z - 100) < 0.01, "z should be orbitRadius at azimuth=0, elevation=0")
    }

    @Test("cameraPosition at azimuth=pi/2 is along +X")
    func testCameraPositionAzimuthHalfPi() {
        let cam = CameraState(azimuth: .pi / 2, elevation: 0, cameraTarget: .zero, orbitRadius: 100)
        let pos = cam.cameraPosition
        #expect(abs(pos.x - 100) < 0.01)
        #expect(abs(pos.y) < 0.01)
        #expect(abs(pos.z) < 0.01)
    }

    @Test("cameraPosition with non-zero target offsets correctly")
    func testCameraPositionOffset() {
        let target = SIMD3<Float>(10, 20, 30)
        let cam = CameraState(azimuth: 0, elevation: 0, cameraTarget: target, orbitRadius: 50)
        let pos = cam.cameraPosition
        #expect(abs(pos.x - target.x) < 0.01)
        #expect(abs(pos.y - target.y) < 0.01)
        #expect(abs(pos.z - (target.z + 50)) < 0.01)
    }

    @Test("viewMatrix produces correct look direction")
    func testViewMatrix() {
        let cam = CameraState(azimuth: 0, elevation: 0, cameraTarget: .zero, orbitRadius: 100)
        let vm = cam.viewMatrix()
        // The view matrix should be non-zero and invertible
        let det = simd_determinant(vm)
        #expect(abs(det) > 0.0001, "View matrix should be invertible")
        // Camera at +Z looking at origin: lookAt negates forward → [2][2] = +1
        // Translation should place eye at negative Z in view space
        #expect(vm[3][2] < 0, "Eye should be translated to negative Z in view space")
    }

    @Test("projectionMatrix has correct FOV aspect")
    func testProjectionMatrix() {
        let cam = CameraState(fovDegrees: 60)
        let pm = cam.projectionMatrix(aspect: 16.0 / 9.0)
        // [1][1] = 1/tan(fov/2) for Y scale
        let expectedYScale = 1.0 / tan(Float(60.0 * .pi / 180.0) * 0.5)
        #expect(abs(pm[1][1] - expectedYScale) < 0.001)
        // [0][0] = yScale / aspect
        let expectedXScale = expectedYScale / (16.0 / 9.0)
        #expect(abs(pm[0][0] - expectedXScale) < 0.001)
    }

    @Test("lookAt produces standard view matrix")
    func testLookAt() {
        let vm = CameraState.lookAt(
            eye: SIMD3<Float>(0, 0, 5),
            center: SIMD3<Float>(0, 0, 0),
            up: SIMD3<Float>(0, 1, 0)
        )
        // Eye at (0,0,5) looking at origin: translation should move eye to origin in view space
        // vm[3][2] should be -5 (eye distance along Z)
        #expect(abs(vm[3][2] + 5) < 0.001)
    }

    @Test("perspectiveProjection near/far planes mapped correctly")
    func testPerspectiveProjection() {
        let pm = CameraState.perspectiveProjection(
            fovRadians: Float.pi / 3, aspect: 1.0, near: 0.1, far: 100.0
        )
        // [2][3] should be -1 (perspective divide)
        #expect(abs(pm[2][3] - (-1)) < 0.001)
        // [3][2] = -2*far*near / (far-near)
        let expected = -2.0 * 100.0 * 0.1 / (100.0 - 0.1)
        #expect(abs(pm[3][2] - Float(expected)) < 0.01)
    }

    @Test("forward/right/up form orthonormal basis")
    func testOrthonormalBasis() {
        let cam = CameraState(azimuth: 0.5, elevation: 0.2, orbitRadius: 100)
        let f = cam.forward
        let r = cam.right
        let u = cam.up
        // Should be approximately unit vectors
        #expect(abs(simd_length(f) - 1) < 0.001)
        #expect(abs(simd_length(r) - 1) < 0.001)
        #expect(abs(simd_length(u) - 1) < 0.001)
        // Should be approximately orthogonal
        #expect(abs(dot(f, r)) < 0.001)
        #expect(abs(dot(f, u)) < 0.001)
        #expect(abs(dot(r, u)) < 0.001)
    }
}
