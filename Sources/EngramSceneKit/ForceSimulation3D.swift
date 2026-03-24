import Foundation
import Accelerate
import simd
import os

private let forceSimLog = Logger(subsystem: "com.claudememory.visualizer", category: "ForceSimulation3D")

// GPU force computation via MetalForceCompute + ForceIntegrate.metal.

/// 3D force-directed graph layout engine. Same split architecture as ForceSimulation:
/// O(n²) force computation runs async on @ForceSimulatorActor, O(n) integration runs sync at 60fps.
@MainActor
public final class ForceSimulation3D {
    // SoA layout for cache-friendly iteration
    private var ids: [UUID] = []
    private var x: [Float] = []
    private var y: [Float] = []
    private var z: [Float] = []
    private var vx: [Float] = []
    private var vy: [Float] = []
    private var vz: [Float] = []
    private var pinned: [Bool] = []
    private var idToIndex: [UUID: Int] = [:]
    private var edgeIndices: [(Int, Int)] = []
    private var edgeIndexSet: Set<UInt64> = []  // packed (si << 32 | ti) for O(1) duplicate check
    private var projectGroup: [Int] = []
    private var topicGroup: [Int] = []
    private var topicProjectGroup: [Int] = []

    // Galaxy grouping — per-node galaxy index + per-galaxy center
    private(set) var galaxyGroup: [Int] = []
    private var galaxyGroupToIndex: [String: Int] = [:]
    private(set) var galaxyCenters: [SIMD3<Float>] = []

    // Reverse mappings for surgical insert (addNode)
    private var projectToGroup: [String: Int] = [:]
    private var topicKeyToGroup: [String: Int] = [:]

    // Force results now delivered via @MainActor callback (applyGPUForces).
    // No cross-thread lock needed.

    /// Public accessors for renderer's force encoding (read-only snapshots of sim state)
    public var posX: [Float] { x }
    public var posY: [Float] { y }
    public var posZ: [Float] { z }
    public var projectGroupPublic: [Int] { projectGroup }
    public var topicGroupPublic: [Int] { topicGroup }
    public var galaxyGroupPublic: [Int] { galaxyGroup }
    public var galaxyCentersPublic: [SIMD3<Float>] { galaxyCenters }
    public var edgeIndicesPublic: [(Int, Int)] { edgeIndices }
    public var topicProjectGroupPublic: [Int] { topicProjectGroup }
    public var nodeIds: [UUID] { ids }
    public var nodeCount: Int { ids.count }

    public private(set) var positions: [UUID: SIMD3<Float>] = [:]

    // Force parameters (restored from Metal-era tuning on main)
    public let springLength: Float = 240
    public let crossProjectSpringLength: Float = 400
    public let springStrength: Float = 0.0004
    public let crossProjectSpringScale: Float = 1.0
    public let chargeStrength: Float = 500
    public let crossChargeMultiplier: Float = 3.0
    public let sameProjectChargeScale: Float = 1.0
    public let sameTopicChargeScale: Float = 0.65
    public let centerStrength: Float = 0.006
    public let cohesionStrength: Float = 0.0015
    public let centroidRepulsion: Float = 5000
    public let topicCohesionStrength: Float = 0.009
    public let topicCentroidRepulsion: Float = 7000
    public let topicLeashStrength: Float = 0.01
    private let damping: Float = 0.78
    private let maxSpeed: Float = 12.0

    public private(set) var alpha: Float = 1.0
    private let alphaDecay: Float = 0.995
    private let alphaFloor: Float = 0.01
    private var topologyVersion: UInt64 = 0

    /// Set by addNode/addEdge, drained by tick(). Coalesces multiple topology changes
    /// into a single wake per tick instead of one per call.
    private var hasPendingTopologyChanges = false

    /// Set by topology changes, cleared after CSR rebuild in dispatchForceComputation.
    /// CSR (Compressed Sparse Row) structures are pre-built for GPU gather kernels.
    public var topologyDirtyForGPU = true

    public init() {}

    public var center: SIMD3<Float> = .zero
    public var isActive: Bool = true

    /// True when the simulation has converged (velocities near-zero for consecutive frames).
    /// When settled, tick() skips force dispatch and position sync to save CPU/GPU.
    public private(set) var isSettled = false
    private(set) var settledFrameCount = 0
    public private(set) var framesSinceWake = 0
    /// Frames spent at alpha floor. After 60 frames at floor, GPU force dispatch should stop
    /// to let velocities decay via damping (matches old maxPostAlphaDispatches behavior).
    private var framesAtAlphaFloor = 0
    private let maxPostAlphaDispatches = 60
    /// False after alpha floor + maxPostAlphaDispatches frames. GPU callers should check this.
    public var shouldDispatchForces: Bool { !isSettled && framesAtAlphaFloor < maxPostAlphaDispatches }
    /// Minimum frames after wake() before settle is allowed. Prevents premature settling
    /// when the first async force results haven't arrived yet.
    private let settleGuardFrames = 30  // ~0.5s at 60fps

    /// Max speed² from the last tick. Used by the Timer to throttle visual updates
    /// when nodes are barely moving (convergence tail).
    public private(set) var lastMaxSpeedSq: Float = 0

    // MARK: - Galaxy Management

    /// Register or update a galaxy's center position.
    public func setGalaxyCenter(_ galaxyId: String, _ center: SIMD3<Float>) {
        let idx: Int
        if let existing = galaxyGroupToIndex[galaxyId] {
            idx = existing
        } else {
            idx = galaxyCenters.count
            galaxyGroupToIndex[galaxyId] = idx
            galaxyCenters.append(center)
        }
        galaxyCenters[idx] = center
    }

    /// Resolve a galaxy ID to its group index, creating if needed.
    public func galaxyIndex(for galaxyId: String) -> Int {
        if let existing = galaxyGroupToIndex[galaxyId] { return existing }
        let idx = galaxyCenters.count
        galaxyGroupToIndex[galaxyId] = idx
        galaxyCenters.append(.zero)
        return idx
    }

    /// Change galaxy group for a set of nodes (e.g. migration between galaxies).
    public func changeGalaxyGroup(for nodeIds: Set<UUID>, to galaxyId: String) {
        let newGroup = galaxyIndex(for: galaxyId)
        // Lazily initialize galaxyGroup if updateGraph was used (which doesn't populate it)
        if galaxyGroup.count < ids.count {
            galaxyGroup.append(contentsOf: [Int](repeating: 0, count: ids.count - galaxyGroup.count))
        }
        for id in nodeIds {
            guard let i = idToIndex[id] else { continue }
            galaxyGroup[i] = newGroup
        }
        hasPendingTopologyChanges = true
        topologyDirtyForGPU = true
    }

    /// Wake the simulation from settled state (e.g. after topology change or user interaction).
    public func wake() {
        isSettled = false
        settledFrameCount = 0
        framesSinceWake = 0
        framesAtAlphaFloor = 0
        // Reset alpha so center gravity and other alpha-scaled forces are meaningful.
        // Without this, alpha decays to 0.01 after ~15s and never recovers — making
        // migration animations crawl because center force is 100x weaker.
        alpha = 1.0
    }

    // MARK: - Graph Management

    public func updateGraph(nodeIds: Set<UUID>, edges: [(UUID, UUID)],
                     projectForNode: [UUID: String] = [:], topicForNode: [UUID: String] = [:]) {
        // Remove nodes no longer in the graph
        var keep = [Bool](repeating: false, count: ids.count)
        for (i, id) in ids.enumerated() { keep[i] = nodeIds.contains(id) }

        var newIds: [UUID] = []
        var newX: [Float] = [], newY: [Float] = [], newZ: [Float] = []
        var newVx: [Float] = [], newVy: [Float] = [], newVz: [Float] = []
        var newPinned: [Bool] = []
        let cap = nodeIds.count
        newIds.reserveCapacity(cap)
        newX.reserveCapacity(cap); newY.reserveCapacity(cap); newZ.reserveCapacity(cap)
        newVx.reserveCapacity(cap); newVy.reserveCapacity(cap); newVz.reserveCapacity(cap)
        newPinned.reserveCapacity(cap)

        for i in 0..<ids.count where keep[i] {
            newIds.append(ids[i])
            newX.append(x[i]); newY.append(y[i]); newZ.append(z[i])
            newVx.append(vx[i]); newVy.append(vy[i]); newVz.append(vz[i])
            newPinned.append(pinned[i])
        }

        // Add new nodes — seed positions by project so same-project nodes start near
        // each other. Without this, random placement creates local minima that the
        // force simulation can't escape, causing inconsistent clustering quality.
        // Pre-build project index for seeded positioning
        var projGroupForSeed: [String: Int] = [:]
        for id in nodeIds {
            let proj = projectForNode[id] ?? ""
            if projGroupForSeed[proj] == nil { projGroupForSeed[proj] = projGroupForSeed.count }
        }
        let numProjects = max(projGroupForSeed.count, 1)
        for id in nodeIds where idToIndex[id] == nil {
            let proj = projectForNode[id] ?? ""
            let projIdx = projGroupForSeed[proj] ?? 0
            // Golden-angle spacing in azimuth gives maximally separated project directions
            let baseAngle = Float(projIdx) * 2.399963  // golden angle in radians
            let basePhi = Float(projIdx) * 0.8 - Float(numProjects) * 0.4  // spread in elevation
            let jitterAngle = Float.random(in: -0.4...0.4)
            let jitterPhi = Float.random(in: -0.3...0.3)
            let r = Float.random(in: 80...180)
            let angle = baseAngle + jitterAngle
            let phi = max(-.pi/2, min(.pi/2, basePhi + jitterPhi))
            newIds.append(id)
            newX.append(center.x + cos(angle) * cos(phi) * r)
            newY.append(center.y + sin(angle) * cos(phi) * r)
            newZ.append(center.z + sin(phi) * r)
            newVx.append(0); newVy.append(0); newVz.append(0)
            newPinned.append(false)
        }

        ids = newIds
        x = newX; y = newY; z = newZ
        vx = newVx; vy = newVy; vz = newVz
        pinned = newPinned

        // Rebuild index
        idToIndex.removeAll(keepingCapacity: true)
        for (i, id) in ids.enumerated() { idToIndex[id] = i }

        // Build project group indices
        var newProjectToGroup: [String: Int] = [:]
        projectGroup = ids.map { id in
            let proj = projectForNode[id] ?? ""
            if let g = newProjectToGroup[proj] { return g }
            let g = newProjectToGroup.count; newProjectToGroup[proj] = g; return g
        }
        self.projectToGroup = newProjectToGroup

        // Build topic group indices
        var newTopicKeyToGroup: [String: Int] = [:]
        var topicGroupToProjectGroup: [Int] = []
        topicGroup = ids.map { id in
            let proj = projectForNode[id] ?? ""
            let topic = topicForNode[id] ?? "general"
            let key = "\(proj)|\(topic)"
            if let g = newTopicKeyToGroup[key] { return g }
            let g = newTopicKeyToGroup.count; newTopicKeyToGroup[key] = g
            topicGroupToProjectGroup.append(newProjectToGroup[proj] ?? 0)
            return g
        }
        self.topicKeyToGroup = newTopicKeyToGroup
        topicProjectGroup = topicGroupToProjectGroup

        // Convert edges to index pairs
        edgeIndices = edges.compactMap { (src, tgt) in
            guard let si = idToIndex[src], let ti = idToIndex[tgt] else { return nil }
            return (si, ti)
        }
        edgeIndexSet = Set(edgeIndices.map { UInt64($0.0) << 32 | UInt64($0.1) })

        topologyVersion &+= 1
        topologyDirtyForGPU = true
        isSettled = false; settledFrameCount = 0
        rebuildPositions()
    }

    /// Surgically remove nodes without rebuilding the entire graph.
    /// Compacts SoA arrays and remaps edge indices.
    public func removeNodes(_ idsToRemove: Set<UUID>) {
        guard !idsToRemove.isEmpty else { return }
        let oldCount = ids.count

        // Build old→new index mapping
        var oldToNew = [Int](repeating: -1, count: oldCount)
        var newIdx = 0
        for i in 0..<oldCount {
            if !idsToRemove.contains(ids[i]) {
                oldToNew[i] = newIdx
                newIdx += 1
            }
        }

        // Compact SoA arrays
        let cap = newIdx
        var newIds: [UUID] = [];      newIds.reserveCapacity(cap)
        var newX: [Float] = [];       newX.reserveCapacity(cap)
        var newY: [Float] = [];       newY.reserveCapacity(cap)
        var newZ: [Float] = [];       newZ.reserveCapacity(cap)
        var newVx: [Float] = [];      newVx.reserveCapacity(cap)
        var newVy: [Float] = [];      newVy.reserveCapacity(cap)
        var newVz: [Float] = [];      newVz.reserveCapacity(cap)
        var newPinned: [Bool] = [];   newPinned.reserveCapacity(cap)
        var newProjGroup: [Int] = []; newProjGroup.reserveCapacity(cap)
        var newTopicGrp: [Int] = [];  newTopicGrp.reserveCapacity(cap)
        var newGalaxyGrp: [Int] = []; if !galaxyGroup.isEmpty { newGalaxyGrp.reserveCapacity(cap) }

        for i in 0..<oldCount where oldToNew[i] >= 0 {
            newIds.append(ids[i])
            newX.append(x[i]); newY.append(y[i]); newZ.append(z[i])
            newVx.append(vx[i]); newVy.append(vy[i]); newVz.append(vz[i])
            newPinned.append(pinned[i])
            newProjGroup.append(projectGroup[i])
            newTopicGrp.append(topicGroup[i])
            if !galaxyGroup.isEmpty { newGalaxyGrp.append(galaxyGroup[i]) }
        }

        ids = newIds; x = newX; y = newY; z = newZ
        vx = newVx; vy = newVy; vz = newVz; pinned = newPinned
        projectGroup = newProjGroup; topicGroup = newTopicGrp
        galaxyGroup = newGalaxyGrp

        // Rebuild index
        idToIndex.removeAll(keepingCapacity: true)
        for (i, id) in ids.enumerated() { idToIndex[id] = i }

        // Remap edge indices (drop edges that touch removed nodes)
        edgeIndices = edgeIndices.compactMap { (si, ti) in
            let nsi = oldToNew[si]; let nti = oldToNew[ti]
            guard nsi >= 0, nti >= 0 else { return nil }
            return (nsi, nti)
        }
        edgeIndexSet = Set(edgeIndices.map { UInt64($0.0) << 32 | UInt64($0.1) })

        // Remove from positions dict
        for id in idsToRemove { positions.removeValue(forKey: id) }

        hasPendingTopologyChanges = true
        topologyDirtyForGPU = true
    }

    /// Surgically insert a single node without rebuilding the entire graph.
    /// When the sim is settled, activates local wake instead of full wake.
    public func addNode(_ id: UUID, project: String, topic: String, galaxyId: String? = nil) {
        guard idToIndex[id] == nil else { return }

        let newIndex = ids.count

        // Smart positioning: place at project cluster centroid + jitter when settled,
        // random sphere position when sim is actively converging.
        // Seed near galaxy center when galaxyId is provided, so galaxies start separated.
        let galaxyCenter: SIMD3<Float>?
        if let gid = galaxyId, let gc = galaxyGroupToIndex[gid], gc < galaxyCenters.count {
            galaxyCenter = galaxyCenters[gc]
        } else {
            galaxyCenter = nil
        }
        let position: SIMD3<Float>
        if isSettled, let pg = projectToGroup[project] {
            position = projectCentroid(group: pg, jitter: 15.0)
        } else {
            position = randomSpherePosition(around: galaxyCenter)
        }

        ids.append(id)
        x.append(position.x); y.append(position.y); z.append(position.z)
        vx.append(0); vy.append(0); vz.append(0)
        pinned.append(false)
        idToIndex[id] = newIndex

        // Galaxy group
        if let gid = galaxyId {
            galaxyGroup.append(galaxyIndex(for: gid))
        } else if !galaxyGroupToIndex.isEmpty {
            // Default to first galaxy when galaxies exist but caller didn't specify
            galaxyGroup.append(0)
        }
        // When no galaxies registered, don't append (keep galaxyGroup empty = backward compat)

        // Project group
        let projGroup: Int
        if let g = projectToGroup[project] {
            projGroup = g
        } else {
            projGroup = projectToGroup.count
            projectToGroup[project] = projGroup
        }
        projectGroup.append(projGroup)

        // Topic group
        let topicKey = "\(project)|\(topic)"
        if let g = topicKeyToGroup[topicKey] {
            topicGroup.append(g)
        } else {
            let g = topicKeyToGroup.count
            topicKeyToGroup[topicKey] = g
            topicGroup.append(g)
            topicProjectGroup.append(projGroup)
        }

        positions[id] = position
        topologyDirtyForGPU = true
        hasPendingTopologyChanges = true
    }

    /// Surgically insert a single edge without rebuilding the entire graph.
    public func addEdge(from source: UUID, to target: UUID) {
        guard let si = idToIndex[source], let ti = idToIndex[target] else { return }
        let key = UInt64(si) << 32 | UInt64(ti)
        guard edgeIndexSet.insert(key).inserted else { return }
        edgeIndices.append((si, ti))
        topologyDirtyForGPU = true
        hasPendingTopologyChanges = true
    }

    /// Write external positions (e.g. from 2D positions + z jitter) into internal arrays.
    public func setPositions(_ positions: [UUID: SIMD3<Float>]) {
        for (id, point) in positions {
            guard let i = idToIndex[id] else { continue }
            x[i] = point.x; y[i] = point.y; z[i] = point.z
            vx[i] = 0; vy[i] = 0; vz[i] = 0
        }
        alpha = 0.5
        syncPositions()
    }

    // MARK: - Pinning (for node expansion)

    /// Pin a node so the force simulation won't move it.
    public func pin(_ id: UUID) {
        guard let i = idToIndex[id] else { return }
        pinned[i] = true
        vx[i] = 0; vy[i] = 0; vz[i] = 0
    }

    /// Unpin a node so the force simulation resumes moving it.
    public func unpin(_ id: UUID) {
        guard let i = idToIndex[id] else { return }
        pinned[i] = false
    }

    /// Set a node's position directly and zero its velocity.
    public func setPosition(_ id: UUID, to pos: SIMD3<Float>) {
        guard let i = idToIndex[id] else { return }
        x[i] = pos.x; y[i] = pos.y; z[i] = pos.z
        vx[i] = 0; vy[i] = 0; vz[i] = 0
        positions[id] = pos
    }

    // MARK: - Per-frame tick

    public func tick() {
        guard isActive else { return }
        let n = ids.count
        guard n > 1 else {
            if n == 1 { x[0] = center.x; y[0] = center.y; z[0] = center.z; syncPositions() }
            return
        }

        // Coalesce topology changes into a single wake per tick regardless of how many
        // addNode/addEdge calls arrived since the last tick.
        if hasPendingTopologyChanges {
            hasPendingTopologyChanges = false
            topologyVersion &+= 1
            isSettled = false
            settledFrameCount = 0
        }

        // Fast path: when settled, skip all work (no integration, no sync, no force dispatch)
        if isSettled { return }

        // GPU handles force computation + integration via ForceIntegrate.metal.
        // tick() only decays alpha, checks settle, and syncs positions.
        let tickStart = CFAbsoluteTimeGetCurrent()
        var maxSpeedSq: Float = 0

        var totalKineticEnergy: Float = 0
        // GPU integration delivered positions via applyGPUForces().
        // Measure velocity for settle detection (from last GPU delivery).
        for i in 0..<n {
            let speedSq = vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]
            maxSpeedSq = max(maxSpeedSq, speedSq)
            totalKineticEnergy += speedSq
        }

        // Adaptive alpha decay — larger graphs converge structurally sooner because
        // per-node forces are individually weaker (more spread out). Gentle ramp:
        // 0.995 at <1K, 0.994 at 5K, 0.993 at 10K+. Avoids hitting the alpha floor
        // too early (which leaves permanent residual structural forces).
        let nScale = min(1.0, Float(n) / 10000.0)
        let effectiveDecay = alphaDecay - 0.002 * nScale
        alpha = max(alpha * effectiveDecay, alphaFloor)
        if alpha <= alphaFloor { framesAtAlphaFloor += 1 }
        lastMaxSpeedSq = maxSpeedSq
        framesSinceWake += 1

        // Settle detection — uses both max speed (strict) and mean kinetic energy
        // (outlier-tolerant). At high node counts, one wiggling outlier shouldn't
        // block settling when 99.9% of nodes are stationary.
        let settleThreshold: Float = 0.05
        let relaxedThreshold: Float = 1.0
        let meanSpeedSq = totalKineticEnergy / max(Float(n), 1.0)
        if alpha < 0.08 {
            if maxSpeedSq < settleThreshold {
                // All nodes barely moving
                settledFrameCount += 1
                if settledFrameCount >= 30 {
                    isSettled = true
                    syncPositions()
                    return
                }
            } else if meanSpeedSq < 0.005 && maxSpeedSq < 5.0 {
                // Mean energy negligible — a few outliers still moving but graph is stable
                settledFrameCount += 1
                if settledFrameCount >= 45 {
                    isSettled = true
                    syncPositions()
                    return
                }
            } else if maxSpeedSq < relaxedThreshold {
                // Low energy but not zero — structural force residuals.
                // Force settle after 120 frames of sub-pixel motion.
                settledFrameCount += 1
                if settledFrameCount >= 120 {
                    isSettled = true
                    syncPositions()
                    return
                }
            } else {
                settledFrameCount = 0
            }
        }

        syncPositions()

        let totalTickMs = (CFAbsoluteTimeGetCurrent() - tickStart) * 1000.0
        if framesSinceWake % 60 == 1 || totalTickMs > 20 {
            print("[engram:sim] tick n=\(n) integrate=\(String(format: "%.1f", totalTickMs))ms maxSpeedSq=\(String(format: "%.4f", maxSpeedSq)) alpha=\(String(format: "%.4f", alpha)) settled=\(isSettled)")
        }
    }

    /// Apply GPU force results delivered via @MainActor callback.
    /// Called from MetalForceCompute's completion handler (dispatched to main actor).
    public func applyGPUForces(_ result: ForceResult) {
        // GPU-integrated positions: forces were already applied with alpha scaling on GPU.
        // Write positions directly into CPU arrays.
        if let gpuPositions = result.positions {
            guard gpuPositions.count == ids.count else { return }
            // Compute velocity from position delta (for settle detection).
            // GPU velocities live on GPU; this derives them from the position change.
            for i in 0..<ids.count {
                vx[i] = gpuPositions[i].x - x[i]
                vy[i] = gpuPositions[i].y - y[i]
                vz[i] = gpuPositions[i].z - z[i]
                x[i] = gpuPositions[i].x
                y[i] = gpuPositions[i].y
                z[i] = gpuPositions[i].z
            }
            syncPositions()
            return
        }

        // Raw forces fallback (unused in GPU-only path, but kept for test compatibility).
        guard result.fx.count == ids.count else { return }
        GPULog.log("APPLIED forces n=\(result.fx.count)")
    }

    // MARK: - Positioning helpers

    /// Compute project cluster centroid from existing node positions.
    private func projectCentroid(group: Int, jitter: Float) -> SIMD3<Float> {
        var sumX: Float = 0, sumY: Float = 0, sumZ: Float = 0, count: Float = 0
        for i in 0..<ids.count {
            if projectGroup[i] == group {
                sumX += x[i]; sumY += y[i]; sumZ += z[i]; count += 1
            }
        }
        if count > 0 {
            return SIMD3<Float>(
                sumX / count + .random(in: -jitter...jitter),
                sumY / count + .random(in: -jitter...jitter),
                sumZ / count + .random(in: -jitter...jitter)
            )
        }
        return randomSpherePosition()
    }

    /// Random position on a spherical shell around center.
    private func randomSpherePosition(around c: SIMD3<Float>? = nil) -> SIMD3<Float> {
        let o = c ?? center
        let angle = Float.random(in: 0...(2 * .pi))
        let phi = Float.random(in: -.pi/2...(.pi/2))
        let r = Float.random(in: 50...250)
        return SIMD3<Float>(
            o.x + cos(angle) * cos(phi) * r,
            o.y + sin(angle) * cos(phi) * r,
            o.z + sin(phi) * r
        )
    }


    // MARK: - Position sync

    private func rebuildPositions() {
        positions.removeAll(keepingCapacity: true)
        for i in 0..<ids.count {
            positions[ids[i]] = SIMD3(x[i], y[i], z[i])
        }
    }

    private func syncPositions() {
        for i in 0..<ids.count {
            positions[ids[i]] = SIMD3(x[i], y[i], z[i])
        }
    }
}
