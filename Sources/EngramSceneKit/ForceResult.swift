import simd

/// GPU/CPU force computation result — decoupled from ForceSimulation3D for testability.
/// Supports two modes:
/// - Forces: raw force vectors for CPU integration (fx/fy/fz)
/// - Positions: GPU-integrated positions (positions array) — forces already applied with alpha
public struct ForceResult: Sendable {
    public let fx: [Float], fy: [Float], fz: [Float]
    /// GPU-integrated positions. When non-nil, forces were already integrated on GPU
    /// (with alpha scaling) and these positions should replace CPU arrays directly.
    public let positions: [SIMD3<Float>]?

    /// Per-phase timing breakdown (milliseconds). Populated by encodeForcePassSync.
    public var cpuPrepMs: Double = 0
    public var gpuMs: Double = 0
    public var readbackMs: Double = 0
    public var algorithm: String = ""

    public init(fx: [Float], fy: [Float], fz: [Float]) {
        self.fx = fx; self.fy = fy; self.fz = fz
        self.positions = nil
    }

    public init(positions: [SIMD3<Float>]) {
        self.fx = []; self.fy = []; self.fz = []
        self.positions = positions
    }
}
