/// GPU/CPU force computation result — decoupled from ForceSimulation3D for testability.
public struct ForceResult: Sendable {
    public let fx: [Float], fy: [Float], fz: [Float]
    public init(fx: [Float], fy: [Float], fz: [Float]) {
        self.fx = fx; self.fy = fy; self.fz = fz
    }
}
