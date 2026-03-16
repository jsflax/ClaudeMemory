import Foundation
import Accelerate
import simd
import os
import EngramSceneKit

private let forceSimLog = Logger(subsystem: "com.claudememory.visualizer", category: "ForceSimulation3D")

// Barnes-Hut octree code is in EngramSceneKit/ForceComputation.swift (public).

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

    // Stored per-node forces from last async computation
    private var storedFx: [Float] = []
    private var storedFy: [Float] = []
    private var storedFz: [Float] = []

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
    public var nodeCount: Int { ids.count }

    public private(set) var positions: [UUID: SIMD3<Float>] = [:]

    // Force parameters (tuned for 3D — matches JS version exactly)
    public let springLength: Float = 240
    public let crossProjectSpringLength: Float = 400
    public let springStrength: Float = 0.0004
    public let chargeStrength: Float = 500
    public let crossChargeMultiplier: Float = 3.0
    public let sameProjectChargeScale: Float = 1.0
    public let sameTopicChargeScale: Float = 0.35
    public let centerStrength: Float = 0.006
    public let cohesionStrength: Float = 0.0015
    public let centroidRepulsion: Float = 2500
    public let topicCohesionStrength: Float = 0.009
    public let topicCentroidRepulsion: Float = 3500
    private let damping: Float = 0.78
    private let maxSpeed: Float = 12.0

    public private(set) var alpha: Float = 1.0
    private let alphaDecay: Float = 0.995
    private let alphaFloor: Float = 0.01
    public private(set) var tickInFlight = false
    public private(set) var forceAge: Int = 100
    public private(set) var smoothedAttenuation: Float = 0.001
    private var topologyVersion: UInt64 = 0

    /// Set by addNode/addEdge, drained by tick(). Coalesces multiple topology changes
    /// into a single wake per tick instead of one per call.
    private var hasPendingTopologyChanges = false

    /// Set by topology changes, cleared after CSR rebuild in dispatchForceComputation.
    /// CSR (Compressed Sparse Row) structures are pre-built for GPU gather kernels.
    public var topologyDirtyForGPU = true
    /// When true, the renderer handles GPU force encoding — tick() skips CPU dispatch.
    public var useGPUForces = false

    // MARK: - Local wake state
    // When a node is inserted into a settled sim, we run a lightweight CPU-only
    // force computation on just the neighborhood (~50-100 nodes) instead of waking
    // the full O(n) simulation. This avoids the 300x per-frame cost cliff.

    /// Indices of nodes in the active local wake neighborhood.
    private var localWakeIndices: Set<Int> = []

    /// True when a local wake is active (node inserted while settled).
    public private(set) var isLocalWake = false

    /// Frames remaining in local wake. Counts down from localWakeMaxFrames.
    private var localWakeFramesRemaining = 0
    private let localWakeMaxFrames = 15  // ~250ms at 60fps

    /// Index of the newly inserted node that triggered the local wake.
    private var localWakeNewNodeIndex: Int? = nil

    /// Whether the neighborhood has been expanded (done once on first local tick).
    private var localWakeExpanded = false

    public init() {}

    public var center: SIMD3<Float> = .zero
    public var isActive: Bool = true

    /// True when the simulation has converged (velocities near-zero for consecutive frames).
    /// When settled, tick() skips force dispatch and position sync to save CPU/GPU.
    public private(set) var isSettled = false
    private(set) var settledFrameCount = 0
    public private(set) var framesSinceWake = 0
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
        for id in nodeIds {
            guard let i = idToIndex[id] else { continue }
            galaxyGroup[i] = newGroup
        }
        hasPendingTopologyChanges = true
        topologyDirtyForGPU = true
    }

    /// Wake the simulation from settled state (e.g. after topology change or user interaction).
    public func wake() {
        // Cancel any active local wake — full wake supersedes it
        clearLocalWake()
        isSettled = false
        settledFrameCount = 0
        framesSinceWake = 0
        forceAge = 2
        // Reset alpha so center gravity and other alpha-scaled forces are meaningful.
        // Without this, alpha decays to 0.01 after ~15s and never recovers — making
        // migration animations crawl because center force is 100x weaker.
        alpha = 1.0
        // Reset smoothedAttenuation so forces apply at meaningful strength immediately.
        // Without this, smoothedAttenuation ramps from near-zero at 4%/frame — taking
        // ~30 frames to reach useful values — causing newly added nodes to appear stuck.
        smoothedAttenuation = pow(damping, Float(forceAge))
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

        // Add new nodes
        for id in nodeIds where idToIndex[id] == nil {
            let angle = Float.random(in: 0...(2 * .pi))
            let phi = Float.random(in: -.pi/2...(.pi/2))
            let r = Float.random(in: 50...250)
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

        storedFx = [Float](repeating: 0, count: ids.count)
        storedFy = [Float](repeating: 0, count: ids.count)
        storedFz = [Float](repeating: 0, count: ids.count)

        topologyVersion &+= 1
        topologyDirtyForGPU = true
        isSettled = false; settledFrameCount = 0
        forceAge = 2
        rebuildPositions()
    }

    /// Surgically remove nodes without rebuilding the entire graph.
    /// Compacts SoA arrays and remaps edge indices.
    public func removeNodes(_ idsToRemove: Set<UUID>) {
        guard !idsToRemove.isEmpty else { return }
        clearLocalWake()  // indices become invalid after compaction
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

        // Compact stored forces
        storedFx = [Float](repeating: 0, count: cap)
        storedFy = [Float](repeating: 0, count: cap)
        storedFz = [Float](repeating: 0, count: cap)

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
        let position: SIMD3<Float>
        if (isSettled || isLocalWake), let pg = projectToGroup[project] {
            position = projectCentroid(group: pg, jitter: 15.0)
        } else {
            position = randomSpherePosition()
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

        storedFx.append(0); storedFy.append(0); storedFz.append(0)
        positions[id] = position
        topologyDirtyForGPU = true

        // Route to local wake or full wake
        if isSettled && !isLocalWake {
            // First insert into a settled sim → activate local wake
            isLocalWake = true
            localWakeFramesRemaining = localWakeMaxFrames
            localWakeNewNodeIndex = newIndex
            localWakeIndices = [newIndex]
            topologyVersion &+= 1  // so render pipeline picks up the new node
        } else if isLocalWake {
            // Additional insert during active local wake → extend the set
            localWakeIndices.insert(newIndex)
            topologyVersion &+= 1
        } else {
            // Sim already awake → normal full-wake path
            hasPendingTopologyChanges = true
        }
    }

    /// Surgically insert a single edge without rebuilding the entire graph.
    public func addEdge(from source: UUID, to target: UUID) {
        guard let si = idToIndex[source], let ti = idToIndex[target] else { return }
        let key = UInt64(si) << 32 | UInt64(ti)
        guard edgeIndexSet.insert(key).inserted else { return }
        edgeIndices.append((si, ti))
        topologyDirtyForGPU = true

        if isLocalWake {
            // Extend neighborhood to include edge endpoints
            localWakeIndices.insert(si)
            localWakeIndices.insert(ti)
        } else {
            hasPendingTopologyChanges = true
        }
    }

    /// Write external positions (e.g. from 2D positions + z jitter) into internal arrays.
    public func setPositions(_ positions: [UUID: SIMD3<Float>]) {
        clearLocalWake()
        for (id, point) in positions {
            guard let i = idToIndex[id] else { continue }
            x[i] = point.x; y[i] = point.y; z[i] = point.z
            vx[i] = 0; vy[i] = 0; vz[i] = 0
        }
        storedFx = [Float](repeating: 0, count: ids.count)
        storedFy = [Float](repeating: 0, count: ids.count)
        storedFz = [Float](repeating: 0, count: ids.count)
        forceAge = 5
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
            // Full topology change during local wake → escalate to full wake
            if isLocalWake { clearLocalWake() }
        }

        // Local wake path: lightweight CPU-only integration for neighborhood only.
        // Runs instead of full sim when a node was inserted while settled.
        if isLocalWake {
            tickLocalWake()
            return
        }

        // Fast path: when settled, skip all work (no integration, no sync, no force dispatch)
        if isSettled { return }

        // Force results arrive via applyGPUForces() callback (from GPU or CPU).
        // tick() just integrates with whatever stored forces are current.
        let forceStart = CFAbsoluteTimeGetCurrent()
        let forceMethod = useGPUForces ? "gpu" : "cpu"
        let forceMs = 0.0  // force computation is async, not measured here

        // Integrate with whatever forces we have (current or from previous frame)
        let integrateStart = CFAbsoluteTimeGetCurrent()
        let hasForcesComputed = storedFx.count == n
        var maxSpeedSq: Float = 0
        var totalKineticEnergy: Float = 0
        for i in 0..<n where !pinned[i] {
            if hasForcesComputed {
                vx[i] += storedFx[i]
                vy[i] += storedFy[i]
                vz[i] += storedFz[i]
            }
            vx[i] *= damping; vy[i] *= damping; vz[i] *= damping

            let speedSq = vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]
            maxSpeedSq = max(maxSpeedSq, speedSq)
            totalKineticEnergy += speedSq
            if speedSq > maxSpeed * maxSpeed {
                let scale = maxSpeed / sqrt(speedSq)
                vx[i] *= scale; vy[i] *= scale; vz[i] *= scale
            }

            x[i] += vx[i]; y[i] += vy[i]; z[i] += vz[i]
        }
        alpha = max(alpha * alphaDecay, alphaFloor)
        lastMaxSpeedSq = maxSpeedSq
        framesSinceWake += 1

        // Settle detection — simple threshold like JS version.
        // No scale-adaptive hacks needed since forces are synchronous (no micro-oscillation).
        let settleThreshold: Float = 0.01
        if hasForcesComputed
            && framesSinceWake >= settleGuardFrames
            && maxSpeedSq < settleThreshold {
            settledFrameCount += 1
            if settledFrameCount >= 30 {
                isSettled = true
                syncPositions()
                return
            }
        } else if maxSpeedSq >= settleThreshold {
            settledFrameCount = 0
        }

        syncPositions()

        // CPU fallback: only dispatch if GPU force encoding isn't active.
        // When useGPUForces is true, the renderer encodes forces into its command buffer.
        if !tickInFlight && !useGPUForces {
            tickInFlight = true
            let input = ForceComputationInput(
                n: n, x: x, y: y, z: z,
                projectGroup: projectGroup, topicGroup: topicGroup,
                edgeIndices: edgeIndices, topicProjectGroup: topicProjectGroup,
                alpha: alpha, center: center,
                chargeStrength: chargeStrength, crossChargeMultiplier: crossChargeMultiplier,
                sameProjectChargeScale: sameProjectChargeScale, sameTopicChargeScale: sameTopicChargeScale,
                centerStrength: centerStrength,
                springLength: springLength, crossProjectSpringLength: crossProjectSpringLength,
                springStrength: springStrength,
                cohesionStrength: cohesionStrength, centroidRepulsion: centroidRepulsion,
                topicCohesionStrength: topicCohesionStrength, topicCentroidRepulsion: topicCentroidRepulsion,
                galaxyGroup: galaxyGroup, galaxyCenters: galaxyCenters
            )
            Task.detached(priority: .userInitiated) {
                let result = computeAllForces(input)
                let forceResult = ForceResult(fx: result.fx, fy: result.fy, fz: result.fz)
                Task { @MainActor [weak self] in
                    self?.applyGPUForces(forceResult)
                }
            }
        }

        let integrateMs = (CFAbsoluteTimeGetCurrent() - integrateStart) * 1000.0
        let totalTickMs = (CFAbsoluteTimeGetCurrent() - forceStart) * 1000.0
        if framesSinceWake % 60 == 1 || totalTickMs > 20 {
            print("[engram:sim] tick n=\(n) method=\(forceMethod) force=\(String(format: "%.1f", forceMs))ms integrate=\(String(format: "%.1f", integrateMs))ms total=\(String(format: "%.1f", totalTickMs))ms maxSpeedSq=\(String(format: "%.4f", maxSpeedSq)) alpha=\(String(format: "%.4f", alpha)) settled=\(isSettled)")
        }
    }

    /// Apply GPU force results delivered via @MainActor callback.
    /// Called from MetalForceCompute's completion handler (dispatched to main actor).
    public func applyGPUForces(_ result: ForceResult) {
        guard result.fx.count == ids.count else { return }
        storedFx = result.fx
        storedFy = result.fy
        storedFz = result.fz
        tickInFlight = false
        GPULog.log("APPLIED n=\(result.fx.count)")
    }

    // Force computation now handled by:
    //   CPU: computeAllForces() in EngramSceneKit (dispatched from tick())
    //   GPU: MetalForceCompute.dispatchForces() on its own command queue

    // MARK: - Local wake (targeted force computation for single-node inserts)

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
    private func randomSpherePosition() -> SIMD3<Float> {
        let angle = Float.random(in: 0...(2 * .pi))
        let phi = Float.random(in: -.pi/2...(.pi/2))
        let r = Float.random(in: 50...250)
        return SIMD3<Float>(
            center.x + cos(angle) * cos(phi) * r,
            center.y + sin(angle) * cos(phi) * r,
            center.z + sin(phi) * r
        )
    }

    /// Expand local wake neighborhood to include same-project-group nodes, capped.
    private func expandLocalWakeNeighborhood() {
        let maxSize = 100
        guard let newIdx = localWakeNewNodeIndex else { return }
        let newProjGroup = projectGroup[newIdx]
        for i in 0..<ids.count {
            if projectGroup[i] == newProjGroup {
                localWakeIndices.insert(i)
            }
            if localWakeIndices.count >= maxSize { break }
        }
    }

    /// Lightweight per-frame tick for local wake: CPU-only forces on neighborhood.
    private func tickLocalWake() {
        // First tick: expand neighborhood to include project peers
        if !localWakeExpanded {
            expandLocalWakeNeighborhood()
            localWakeExpanded = true
        }

        let localIndices = Array(localWakeIndices)
        let k = localIndices.count
        guard k > 0 else { finishLocalWake(); return }

        // --- O(K²) charge repulsion among local nodes ---
        for ii in 0..<k {
            let i = localIndices[ii]
            guard !pinned[i] else { continue }
            for jj in (ii + 1)..<k {
                let j = localIndices[jj]
                let dx = x[i] - x[j], dy = y[i] - y[j], dz = z[i] - z[j]
                var distSq = dx * dx + dy * dy + dz * dz
                if distSq > 250_000 { continue }  // 500² cutoff
                if distSq < 1 { distSq = 1 }

                let charge: Float
                if projectGroup[i] != projectGroup[j] {
                    charge = chargeStrength * crossChargeMultiplier
                } else if topicGroup[i] == topicGroup[j] {
                    charge = chargeStrength * sameTopicChargeScale
                } else {
                    charge = chargeStrength * sameProjectChargeScale
                }

                let dist = sqrt(distSq)
                let forceMag = charge / distSq
                let ux = dx / dist, uy = dy / dist, uz = dz / dist
                let ffx = ux * forceMag, ffy = uy * forceMag, ffz = uz * forceMag
                vx[i] += ffx; vy[i] += ffy; vz[i] += ffz
                vx[j] -= ffx; vy[j] -= ffy; vz[j] -= ffz
            }
        }

        // --- Spring forces for edges touching the local set ---
        let localSet = localWakeIndices
        for (si, ti) in edgeIndices {
            guard localSet.contains(si) || localSet.contains(ti) else { continue }
            let dx = x[ti] - x[si], dy = y[ti] - y[si], dz = z[ti] - z[si]
            var d = sqrt(dx * dx + dy * dy + dz * dz)
            if d < 1 { d = 1 }
            let cross = projectGroup[si] != projectGroup[ti]
            let rest = cross ? crossProjectSpringLength : springLength
            let force = springStrength * (d - rest)
            let efx = (dx / d) * force, efy = (dy / d) * force, efz = (dz / d) * force
            if localSet.contains(si) { vx[si] += efx; vy[si] += efy; vz[si] += efz }
            if localSet.contains(ti) { vx[ti] -= efx; vy[ti] -= efy; vz[ti] -= efz }
        }

        // --- Cohesion: pull new node(s) toward project centroid ---
        if let newIdx = localWakeNewNodeIndex {
            let pg = projectGroup[newIdx]
            var sumX: Float = 0, sumY: Float = 0, sumZ: Float = 0, count: Float = 0
            for i in 0..<ids.count where projectGroup[i] == pg {
                sumX += x[i]; sumY += y[i]; sumZ += z[i]; count += 1
            }
            if count > 1 {
                let cx = sumX / count, cy = sumY / count, cz = sumZ / count
                // Stronger cohesion for fast convergence (3x normal)
                let str = cohesionStrength * 3.0
                let dx = cx - x[newIdx], dy = cy - y[newIdx], dz = cz - z[newIdx]
                vx[newIdx] += dx * str
                vy[newIdx] += dy * str
                vz[newIdx] += dz * str
            }
        }

        // --- Center gravity for local nodes ---
        for i in localIndices where !pinned[i] {
            vx[i] += (center.x - x[i]) * alphaFloor * centerStrength
            vy[i] += (center.y - y[i]) * alphaFloor * centerStrength
            vz[i] += (center.z - z[i]) * alphaFloor * centerStrength
        }

        // --- Integration: only local nodes ---
        var maxSpeedSq: Float = 0
        for i in localIndices where !pinned[i] {
            vx[i] *= damping; vy[i] *= damping; vz[i] *= damping
            let speedSq = vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]
            maxSpeedSq = max(maxSpeedSq, speedSq)
            if speedSq > maxSpeed * maxSpeed {
                let scale = maxSpeed / sqrt(speedSq)
                vx[i] *= scale; vy[i] *= scale; vz[i] *= scale
            }
            x[i] += vx[i]; y[i] += vy[i]; z[i] += vz[i]
        }

        lastMaxSpeedSq = maxSpeedSq

        // Sync only changed positions (O(K) instead of O(n))
        for i in localIndices {
            positions[ids[i]] = SIMD3(x[i], y[i], z[i])
        }

        // Settle: velocities converged or max frames reached
        localWakeFramesRemaining -= 1
        if maxSpeedSq < 0.01 || localWakeFramesRemaining <= 0 {
            finishLocalWake()
        }
    }

    /// Finish local wake — transition back to settled state.
    private func finishLocalWake() {
        clearLocalWake()
        isSettled = true
        lastMaxSpeedSq = 0
        // Final full sync for consistency
        syncPositions()
    }

    /// Reset all local wake state without changing isSettled.
    private func clearLocalWake() {
        isLocalWake = false
        localWakeIndices.removeAll()
        localWakeNewNodeIndex = nil
        localWakeExpanded = false
        localWakeFramesRemaining = 0
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
