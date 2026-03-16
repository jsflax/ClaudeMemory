@preconcurrency import Metal
import CEngramSceneTypes
import simd
import os

/// Encapsulates GPU force dispatch orchestration.
/// Checks readiness, manages topology, creates separate command buffers
/// on the shared queue, and delivers results via callback.
///
/// Extracted from MetalGraphRenderer.drawFrame() lines 789-844.
/// Key invariants preserved:
/// - Separate command buffer for forces (GPU scheduler preemption)
/// - Same command queue as render (no dual-queue contention)
/// - Topology rebuild before first encode after dirty
@MainActor
public final class ForceEngine {
    public let forceCompute: MetalForceCompute

    public init(forceCompute: MetalForceCompute) {
        self.forceCompute = forceCompute
    }

    /// Whether the underlying force compute pipeline is ready.
    public var isFullSimAvailable: Bool { forceCompute.isFullSimAvailable }

    /// Whether a force dispatch is currently in flight on the GPU.
    public var isInFlight: Bool { forceCompute.inFlight }

    /// Snapshot of simulation state needed for force encoding.
    /// Avoids coupling ForceEngine to ForceSimulation3D.
    public struct SimulationSnapshot {
        public let nodeCount: Int
        public let isSettled: Bool
        public let posX: [Float]
        public let posY: [Float]
        public let posZ: [Float]
        public let projectGroups: [Int]
        public let topicGroups: [Int]
        public let galaxyGroups: [Int]
        public let galaxyCenters: [SIMD3<Float>]
        public let edgeIndices: [(Int, Int)]
        public let topologyDirty: Bool

        // Force parameters
        public let chargeStrength: Float
        public let crossChargeMultiplier: Float
        public let sameTopicChargeScale: Float
        public let sameProjectChargeScale: Float
        public let springLength: Float
        public let crossProjectSpringLength: Float
        public let springStrength: Float
        public let cohesionStrength: Float
        public let centroidRepulsion: Float
        public let topicCohesionStrength: Float
        public let topicCentroidRepulsion: Float
        public let centerStrength: Float
        public let center: SIMD3<Float>
        public let alpha: Float
        public let damping: Float
        public let maxSpeed: Float

        public init(
            nodeCount: Int, isSettled: Bool,
            posX: [Float], posY: [Float], posZ: [Float],
            projectGroups: [Int], topicGroups: [Int],
            galaxyGroups: [Int], galaxyCenters: [SIMD3<Float>],
            edgeIndices: [(Int, Int)], topologyDirty: Bool,
            chargeStrength: Float, crossChargeMultiplier: Float,
            sameTopicChargeScale: Float, sameProjectChargeScale: Float,
            springLength: Float, crossProjectSpringLength: Float, springStrength: Float,
            cohesionStrength: Float, centroidRepulsion: Float,
            topicCohesionStrength: Float, topicCentroidRepulsion: Float,
            centerStrength: Float, center: SIMD3<Float>,
            alpha: Float, damping: Float, maxSpeed: Float
        ) {
            self.nodeCount = nodeCount; self.isSettled = isSettled
            self.posX = posX; self.posY = posY; self.posZ = posZ
            self.projectGroups = projectGroups; self.topicGroups = topicGroups
            self.galaxyGroups = galaxyGroups; self.galaxyCenters = galaxyCenters
            self.edgeIndices = edgeIndices; self.topologyDirty = topologyDirty
            self.chargeStrength = chargeStrength
            self.crossChargeMultiplier = crossChargeMultiplier
            self.sameTopicChargeScale = sameTopicChargeScale
            self.sameProjectChargeScale = sameProjectChargeScale
            self.springLength = springLength
            self.crossProjectSpringLength = crossProjectSpringLength
            self.springStrength = springStrength
            self.cohesionStrength = cohesionStrength
            self.centroidRepulsion = centroidRepulsion
            self.topicCohesionStrength = topicCohesionStrength
            self.topicCentroidRepulsion = topicCentroidRepulsion
            self.centerStrength = centerStrength; self.center = center
            self.alpha = alpha; self.damping = damping; self.maxSpeed = maxSpeed
        }
    }

    /// Attempt to encode a GPU force pass.
    /// Returns true if forces were encoded and committed.
    ///
    /// - Parameters:
    ///   - queue: The shared command queue (same as render — avoids dual-queue contention)
    ///   - snapshot: Current simulation state
    ///   - onComplete: Callback with force results, dispatched to @MainActor
    @discardableResult
    public func encodeForcePass(
        queue: MTLCommandQueue,
        snapshot: SimulationSnapshot,
        onComplete: @MainActor @Sendable @escaping (ForceResult) -> Void
    ) -> Bool {
        guard !snapshot.isSettled,
              snapshot.nodeCount > 1,
              forceCompute.isFullSimAvailable,
              !forceCompute.inFlight else { return false }

        // Update topology if dirty
        if snapshot.topologyDirty {
            forceCompute.setTopologyDirty(edges: snapshot.edgeIndices)
        }
        if forceCompute.topologyDirtyForDispatch {
            forceCompute.rebuildTopology(
                nodeCount: snapshot.nodeCount,
                projectGroups: snapshot.projectGroups,
                topicGroups: snapshot.topicGroups)
        }

        // Create separate command buffer on same queue — GPU scheduler can
        // service WindowServer between force compute and render.
        guard let cmdBuf = queue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else { return false }

        forceCompute.inFlight = true

        forceCompute.encodeForces(
            encoder: encoder,
            x: snapshot.posX, y: snapshot.posY, z: snapshot.posZ,
            projectGroups: snapshot.projectGroups, topicGroups: snapshot.topicGroups,
            galaxyGroups: snapshot.galaxyGroups, galaxyCenters: snapshot.galaxyCenters,
            chargeStrength: snapshot.chargeStrength,
            crossChargeMultiplier: snapshot.crossChargeMultiplier,
            sameTopicChargeScale: snapshot.sameTopicChargeScale,
            sameProjectChargeScale: snapshot.sameProjectChargeScale,
            springLength: snapshot.springLength,
            crossProjectSpringLength: snapshot.crossProjectSpringLength,
            springStrength: snapshot.springStrength,
            cohesionStrength: snapshot.cohesionStrength,
            centroidRepulsion: snapshot.centroidRepulsion,
            topicCohesionStrength: snapshot.topicCohesionStrength,
            topicCentroidRepulsion: snapshot.topicCentroidRepulsion,
            centerStrength: snapshot.centerStrength, center: snapshot.center,
            alpha: snapshot.alpha, damping: snapshot.damping, maxSpeed: snapshot.maxSpeed)
        encoder.endEncoding()

        let fc = forceCompute
        cmdBuf.addCompletedHandler { @Sendable cb in
            print("[engram:force] CB complete status=\(cb.status.rawValue)")
            GPULog.log("FORCE CB status=\(cb.status.rawValue)")
            if cb.status == .error {
                GPULog.log("FORCE ERROR: \(cb.error?.localizedDescription ?? "unknown")")
                Task { @MainActor in fc.inFlight = false }
                return
            }
            if let forces = fc.readBackForces() {
                let result = ForceResult(fx: forces.fx, fy: forces.fy, fz: forces.fz)
                GPULog.log("DELIVERED n=\(forces.fx.count)")
                Task { @MainActor in
                    fc.inFlight = false
                    onComplete(result)
                }
            } else {
                GPULog.log("READBACK nil — inFlight stuck")
                Task { @MainActor in fc.inFlight = false }
            }
        }
        GPULog.log("FORCE DISPATCH n=\(snapshot.nodeCount) edges=\(snapshot.edgeIndices.count)")
        cmdBuf.commit()
        return true
    }
}
