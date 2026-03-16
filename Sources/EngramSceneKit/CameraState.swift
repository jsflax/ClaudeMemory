import simd

/// Pure-value camera state — all matrix math, no input handling or animation.
/// CameraController (app target) wraps this with @Observable, smoothing, and gestures.
public struct CameraState: Sendable {
    public var azimuth: Float
    public var elevation: Float
    public var cameraTarget: SIMD3<Float>
    public var orbitRadius: Float
    public var scaleFactor: Float
    public var fovDegrees: Float

    public init(
        azimuth: Float = 0,
        elevation: Float = 0.3,
        cameraTarget: SIMD3<Float> = .zero,
        orbitRadius: Float = 2.0,
        scaleFactor: Float = 1.0 / 200.0,
        fovDegrees: Float = 60
    ) {
        self.azimuth = azimuth
        self.elevation = elevation
        self.cameraTarget = cameraTarget
        self.orbitRadius = orbitRadius
        self.scaleFactor = scaleFactor
        self.fovDegrees = fovDegrees
    }

    // MARK: - Computed Directions

    public var cameraPosition: SIMD3<Float> {
        let x = orbitRadius * cos(elevation) * sin(azimuth)
        let y = orbitRadius * sin(elevation)
        let z = orbitRadius * cos(elevation) * cos(azimuth)
        return cameraTarget + SIMD3(x, y, z)
    }

    public var forward: SIMD3<Float> {
        normalize(cameraTarget - cameraPosition)
    }

    public var right: SIMD3<Float> {
        normalize(cross(forward, SIMD3<Float>(0, 1, 0)))
    }

    public var up: SIMD3<Float> {
        cross(right, forward)
    }

    // MARK: - Matrices

    public func viewMatrix() -> simd_float4x4 {
        let pos = cameraPosition * scaleFactor
        let target = cameraTarget * scaleFactor
        return Self.lookAt(eye: pos, center: target, up: SIMD3(0, 1, 0))
    }

    public func projectionMatrix(aspect: Float) -> simd_float4x4 {
        let fovRad = fovDegrees * .pi / 180.0
        return Self.perspectiveProjection(fovRadians: fovRad, aspect: aspect, near: 0.001, far: 100.0)
    }

    // MARK: - Matrix Helpers

    public static func lookAt(eye: SIMD3<Float>, center: SIMD3<Float>, up: SIMD3<Float>) -> simd_float4x4 {
        let f = normalize(center - eye)
        let s = normalize(cross(f, up))
        let u = cross(s, f)

        var result = matrix_identity_float4x4
        result[0][0] = s.x;  result[1][0] = s.y;  result[2][0] = s.z
        result[0][1] = u.x;  result[1][1] = u.y;  result[2][1] = u.z
        result[0][2] = -f.x; result[1][2] = -f.y; result[2][2] = -f.z
        result[3][0] = -dot(s, eye)
        result[3][1] = -dot(u, eye)
        result[3][2] = dot(f, eye)
        return result
    }

    public static func perspectiveProjection(fovRadians: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
        let yScale = 1 / tan(fovRadians * 0.5)
        let xScale = yScale / aspect
        let zRange = far - near

        var result = simd_float4x4(0)
        result[0][0] = xScale
        result[1][1] = yScale
        result[2][2] = -(far + near) / zRange
        result[2][3] = -1
        result[3][2] = -2 * far * near / zRange
        return result
    }
}
