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
    /// GPU simulation state for integration (position/velocity buffers).
    /// When set, encodeForcePass also encodes GPU integration and delivers positions.
    public let simState: GPUSimulationState?

    public init(forceCompute: MetalForceCompute, simState: GPUSimulationState? = nil) {
        self.forceCompute = forceCompute
        self.simState = simState
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
        public let topicProjectGroup: [Int]
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
        public let crossProjectSpringScale: Float
        public let cohesionStrength: Float
        public let centroidRepulsion: Float
        public let topicCohesionStrength: Float
        public let topicCentroidRepulsion: Float
        public let topicLeashStrength: Float
        public let centerStrength: Float
        public let center: SIMD3<Float>
        public let alpha: Float
        public let forceAge: Int
        public let damping: Float
        public let maxSpeed: Float

        public init(
            nodeCount: Int, isSettled: Bool,
            posX: [Float], posY: [Float], posZ: [Float],
            projectGroups: [Int], topicGroups: [Int],
            topicProjectGroup: [Int] = [],
            galaxyGroups: [Int], galaxyCenters: [SIMD3<Float>],
            edgeIndices: [(Int, Int)], topologyDirty: Bool,
            chargeStrength: Float, crossChargeMultiplier: Float,
            sameTopicChargeScale: Float, sameProjectChargeScale: Float,
            springLength: Float, crossProjectSpringLength: Float, springStrength: Float,
            crossProjectSpringScale: Float = 1.0,
            cohesionStrength: Float, centroidRepulsion: Float,
            topicCohesionStrength: Float, topicCentroidRepulsion: Float,
            topicLeashStrength: Float = 0.01,
            centerStrength: Float, center: SIMD3<Float>,
            alpha: Float, forceAge: Int = 2, damping: Float, maxSpeed: Float
        ) {
            self.nodeCount = nodeCount; self.isSettled = isSettled
            self.posX = posX; self.posY = posY; self.posZ = posZ
            self.projectGroups = projectGroups; self.topicGroups = topicGroups
            self.topicProjectGroup = topicProjectGroup
            self.galaxyGroups = galaxyGroups; self.galaxyCenters = galaxyCenters
            self.edgeIndices = edgeIndices; self.topologyDirty = topologyDirty
            self.chargeStrength = chargeStrength
            self.crossChargeMultiplier = crossChargeMultiplier
            self.sameTopicChargeScale = sameTopicChargeScale
            self.sameProjectChargeScale = sameProjectChargeScale
            self.springLength = springLength
            self.crossProjectSpringLength = crossProjectSpringLength
            self.springStrength = springStrength
            self.crossProjectSpringScale = crossProjectSpringScale
            self.cohesionStrength = cohesionStrength
            self.centroidRepulsion = centroidRepulsion
            self.topicCohesionStrength = topicCohesionStrength
            self.topicCentroidRepulsion = topicCentroidRepulsion
            self.topicLeashStrength = topicLeashStrength
            self.centerStrength = centerStrength; self.center = center
            self.alpha = alpha; self.forceAge = forceAge
            self.damping = damping; self.maxSpeed = maxSpeed
        }
    }

    /// Synchronous GPU force pass — encodes, waits, reads back, returns result.
    /// Use when the caller is already blocking (e.g. preview's waitUntilCompleted pattern).
    /// Forces are applied immediately, avoiding the 1-frame delay of the async variant.
    public func encodeForcePassSync(
        queue: MTLCommandQueue,
        snapshot: SimulationSnapshot
    ) -> ForceResult? {
        guard !snapshot.isSettled,
              snapshot.nodeCount > 1,
              forceCompute.isFullSimAvailable else { return nil }

        let t0 = CFAbsoluteTimeGetCurrent()

        if snapshot.topologyDirty {
            forceCompute.setTopologyDirty(edges: snapshot.edgeIndices)
        }
        if forceCompute.topologyDirtyForDispatch {
            forceCompute.rebuildTopology(
                nodeCount: snapshot.nodeCount,
                projectGroups: snapshot.projectGroups,
                topicGroups: snapshot.topicGroups,
                topicProjectGroup: snapshot.topicProjectGroup)
        }

        guard let cmdBuf = queue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else { return nil }

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
            crossProjectSpringScale: snapshot.crossProjectSpringScale,
            cohesionStrength: snapshot.cohesionStrength,
            centroidRepulsion: snapshot.centroidRepulsion,
            topicCohesionStrength: snapshot.topicCohesionStrength,
            topicCentroidRepulsion: snapshot.topicCentroidRepulsion,
            topicLeashStrength: snapshot.topicLeashStrength,
            centerStrength: snapshot.centerStrength, center: snapshot.center,
            alpha: snapshot.alpha, damping: snapshot.damping, maxSpeed: snapshot.maxSpeed)

        if let ss = simState, let forceBuf = forceCompute.outputForceBuffer {
            ss.uploadPositions(x: snapshot.posX, y: snapshot.posY, z: snapshot.posZ)
            ss.alpha = snapshot.alpha
            ss.encodeIntegration(encoder: encoder, forceBuffer: forceBuf,
                                 damping: snapshot.damping, maxSpeed: snapshot.maxSpeed,
                                 alpha: snapshot.alpha)
        }
        encoder.endEncoding()

        let t1 = CFAbsoluteTimeGetCurrent()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        let t2 = CFAbsoluteTimeGetCurrent()

        var result: ForceResult?
        if let ss = simState {
            ss.readBackPositions()
            result = ForceResult(positions: ss.cpuPositions)
        } else if let forces = forceCompute.readBackForces() {
            result = ForceResult(fx: forces.fx, fy: forces.fy, fz: forces.fz)
        }
        let t3 = CFAbsoluteTimeGetCurrent()

        result?.cpuPrepMs = (t1 - t0) * 1000
        result?.gpuMs = (t2 - t1) * 1000
        result?.readbackMs = (t3 - t2) * 1000
        result?.algorithm = forceCompute.lastChargeAlgorithm
        return result
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
    ) -> MTLCommandBuffer? {
        guard !snapshot.isSettled,
              snapshot.nodeCount > 1,
              forceCompute.isFullSimAvailable,
              !forceCompute.inFlight else { return nil }

        // Update topology if dirty
        if snapshot.topologyDirty {
            forceCompute.setTopologyDirty(edges: snapshot.edgeIndices)
        }
        if forceCompute.topologyDirtyForDispatch {
            forceCompute.rebuildTopology(
                nodeCount: snapshot.nodeCount,
                projectGroups: snapshot.projectGroups,
                topicGroups: snapshot.topicGroups,
                topicProjectGroup: snapshot.topicProjectGroup)
        }

        // Create separate command buffer on same queue — GPU scheduler can
        // service WindowServer between force compute and render.
        guard let cmdBuf = queue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else { return nil }

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
            crossProjectSpringScale: snapshot.crossProjectSpringScale,
            cohesionStrength: snapshot.cohesionStrength,
            centroidRepulsion: snapshot.centroidRepulsion,
            topicCohesionStrength: snapshot.topicCohesionStrength,
            topicCentroidRepulsion: snapshot.topicCentroidRepulsion,
            topicLeashStrength: snapshot.topicLeashStrength,
            centerStrength: snapshot.centerStrength, center: snapshot.center,
            alpha: snapshot.alpha, damping: snapshot.damping, maxSpeed: snapshot.maxSpeed)

        // Encode GPU integration if simState is available.
        let hasGPUIntegration: Bool
        if let ss = simState, let forceBuf = forceCompute.outputForceBuffer {
            ss.uploadPositions(x: snapshot.posX, y: snapshot.posY, z: snapshot.posZ)
            ss.alpha = snapshot.alpha
            ss.encodeIntegration(encoder: encoder, forceBuffer: forceBuf,
                                 damping: snapshot.damping, maxSpeed: snapshot.maxSpeed,
                                 alpha: snapshot.alpha)
            hasGPUIntegration = true
        } else {
            hasGPUIntegration = false
        }
        encoder.endEncoding()

        let fc = forceCompute
        let ss = simState
        cmdBuf.addCompletedHandler { @Sendable cb in
            print("[engram:force] CB complete status=\(cb.status.rawValue)")
            GPULog.log("FORCE CB status=\(cb.status.rawValue)")
            if cb.status == .error {
                GPULog.log("FORCE ERROR: \(cb.error?.localizedDescription ?? "unknown")")
                Task { @MainActor in fc.inFlight = false }
                return
            }

            if hasGPUIntegration, let ss {
                // GPU integration ran — read back positions (not forces).
                ss.readBackPositions()
                let positions = ss.cpuPositions
                GPULog.log("DELIVERED positions n=\(positions.count)")
                Task { @MainActor in
                    fc.inFlight = false
                    onComplete(ForceResult(positions: positions))
                }
            } else if let forces = fc.readBackForces() {
                // Fallback: deliver raw forces for CPU integration.
                let result = ForceResult(fx: forces.fx, fy: forces.fy, fz: forces.fz)
                GPULog.log("DELIVERED forces n=\(forces.fx.count)")
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
        return cmdBuf
    }
}
