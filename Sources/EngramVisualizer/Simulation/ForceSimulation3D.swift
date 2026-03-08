import Foundation
import Accelerate
import simd
import os

private let forceSimLog = Logger(subsystem: "com.claudememory.visualizer", category: "ForceSimulation3D")

/// 3D force-directed graph layout engine. Same split architecture as ForceSimulation:
/// O(n²) force computation runs async on @ForceSimulatorActor, O(n) integration runs sync at 60fps.
@MainActor
final class ForceSimulation3D {
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

    // Reverse mappings for surgical insert (addNode)
    private var projectToGroup: [String: Int] = [:]
    private var topicKeyToGroup: [String: Int] = [:]

    // Stored per-node forces from last async computation
    private var storedFx: [Float] = []
    private var storedFy: [Float] = []
    private var storedFz: [Float] = []

    /// Pending force result from async computation — written off-MainActor, read by tick().
    /// Using OSAllocatedUnfairLock for safe cross-isolation handoff without MainActor hop.
    private let pendingForces = OSAllocatedUnfairLock<ForceResult?>(initialState: nil)

    private(set) var positions: [UUID: SIMD3<Float>] = [:]

    // Force parameters (tuned for 3D — slightly stronger since space is larger)
    private let springLength: Float = 240
    private let crossProjectSpringLength: Float = 400
    private let springStrength: Float = 0.0004
    private let chargeStrength: Float = 500
    private let crossChargeMultiplier: Float = 3.0
    private let sameProjectChargeScale: Float = 1.0   // pre-GPU: no reduction for same-project different-topic
    private let sameTopicChargeScale: Float = 0.35     // pre-GPU value (was 0.25 post-GPU)
    private let centerStrength: Float = 0.006
    private let cohesionStrength: Float = 0.0015       // pre-GPU value (was 0.004 post-GPU)
    private let centroidRepulsion: Float = 2500         // pre-GPU value (was 4000 post-GPU)
    private let topicCohesionStrength: Float = 0.009    // pre-GPU value (was 0.005 post-GPU)
    private let topicCentroidRepulsion: Float = 3500    // pre-GPU value (was 6000 post-GPU)
    private let damping: Float = 0.78
    private let maxSpeed: Float = 12.0

    private(set) var alpha: Float = 1.0
    private let alphaDecay: Float = 0.995
    private let alphaFloor: Float = 0.01
    private var tickInFlight = false
    private var forceAge: Int = 100
    private var smoothedAttenuation: Float = 0.001
    private var topologyVersion: UInt64 = 0

    /// Set by addNode/addEdge, drained by tick(). Coalesces multiple topology changes
    /// into a single wake per tick instead of one per call.
    private var hasPendingTopologyChanges = false

    /// Metal compute for GPU-accelerated charge force calculation.
    private var metalForceCompute: MetalForceCompute? = MetalForceCompute()

    var center: SIMD3<Float> = .zero
    var isActive: Bool = true

    /// True when the simulation has converged (velocities near-zero for 30 consecutive frames).
    /// When settled, tick() skips force dispatch and position sync to save CPU/GPU.
    private(set) var isSettled = false
    private var settledFrameCount = 0

    /// Max speed² from the last tick. Used by the Timer to throttle visual updates
    /// when nodes are barely moving (convergence tail).
    private(set) var lastMaxSpeedSq: Float = 0

    /// Wake the simulation from settled state (e.g. after topology change or user interaction).
    func wake() {
        isSettled = false
        settledFrameCount = 0
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

    func updateGraph(nodeIds: Set<UUID>, edges: [(UUID, UUID)],
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
        isSettled = false; settledFrameCount = 0
        forceAge = 2
        rebuildPositions()
    }

    /// Surgically remove nodes without rebuilding the entire graph.
    /// Compacts SoA arrays and remaps edge indices.
    func removeNodes(_ idsToRemove: Set<UUID>) {
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

        for i in 0..<oldCount where oldToNew[i] >= 0 {
            newIds.append(ids[i])
            newX.append(x[i]); newY.append(y[i]); newZ.append(z[i])
            newVx.append(vx[i]); newVy.append(vy[i]); newVz.append(vz[i])
            newPinned.append(pinned[i])
            newProjGroup.append(projectGroup[i])
            newTopicGrp.append(topicGroup[i])
        }

        ids = newIds; x = newX; y = newY; z = newZ
        vx = newVx; vy = newVy; vz = newVz; pinned = newPinned
        projectGroup = newProjGroup; topicGroup = newTopicGrp

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
    }

    /// Surgically insert a single node without rebuilding the entire graph.
    func addNode(_ id: UUID, project: String, topic: String) {
        guard idToIndex[id] == nil else { return }
        let angle = Float.random(in: 0...(2 * .pi))
        let phi = Float.random(in: -.pi/2...(.pi/2))
        let r = Float.random(in: 50...250)
        ids.append(id)
        x.append(center.x + cos(angle) * cos(phi) * r)
        y.append(center.y + sin(angle) * cos(phi) * r)
        z.append(center.z + sin(phi) * r)
        vx.append(0); vy.append(0); vz.append(0)
        pinned.append(false)
        idToIndex[id] = ids.count - 1

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
        hasPendingTopologyChanges = true
        // Set only the new node's position — syncPositions() in the next tick() will update all.
        positions[id] = SIMD3(x.last!, y.last!, z.last!)
    }

    /// Surgically insert a single edge without rebuilding the entire graph.
    func addEdge(from source: UUID, to target: UUID) {
        guard let si = idToIndex[source], let ti = idToIndex[target] else { return }
        let key = UInt64(si) << 32 | UInt64(ti)
        guard edgeIndexSet.insert(key).inserted else { return }
        edgeIndices.append((si, ti))
        hasPendingTopologyChanges = true
    }

    /// Write external positions (e.g. from 2D positions + z jitter) into internal arrays.
    func setPositions(_ positions: [UUID: SIMD3<Float>]) {
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
    func pin(_ id: UUID) {
        guard let i = idToIndex[id] else { return }
        pinned[i] = true
        vx[i] = 0; vy[i] = 0; vz[i] = 0
    }

    /// Unpin a node so the force simulation resumes moving it.
    func unpin(_ id: UUID) {
        guard let i = idToIndex[id] else { return }
        pinned[i] = false
    }

    /// Set a node's position directly and zero its velocity.
    func setPosition(_ id: UUID, to pos: SIMD3<Float>) {
        guard let i = idToIndex[id] else { return }
        x[i] = pos.x; y[i] = pos.y; z[i] = pos.z
        vx[i] = 0; vy[i] = 0; vz[i] = 0
        positions[id] = pos
    }

    // MARK: - Per-frame tick

    func tick() {
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

        // Drain pending force results from async computation (lock-free MainActor handoff)
        if let result = pendingForces.withLock({ value -> ForceResult? in
            let r = value; value = nil; return r
        }) {
            // Blend old → new forces to avoid acceleration discontinuity.
            // Without blending, forceAge reset causes a ~64% jump in applied magnitude.
            let blend: Float = 0.65
            if storedFx.count == result.fx.count {
                for i in 0..<result.fx.count {
                    storedFx[i] = storedFx[i] * (1 - blend) + result.fx[i] * blend
                    storedFy[i] = storedFy[i] * (1 - blend) + result.fy[i] * blend
                    storedFz[i] = storedFz[i] * (1 - blend) + result.fz[i] * blend
                }
            } else {
                storedFx = result.fx; storedFy = result.fy; storedFz = result.fz
            }
            forceAge = 2
            tickInFlight = false
        }

        let hasForcesComputed = storedFx.count == n
        let rawAttenuation = pow(damping, Float(forceAge))
        // Blend=0.04: with off-MainActor dispatch, forces arrive every 1-2 frames,
        // so forceAge oscillates 2→3→2. Heavy smoothing flattens the sawtooth into
        // near-constant application. Settling still works because isSettled returns
        // early before this code runs, and forceAge grows unbounded when forces stop.
        smoothedAttenuation += (rawAttenuation - smoothedAttenuation) * 0.04
        let attenuation = smoothedAttenuation
        forceAge += 1

        var maxSpeedSq: Float = 0
        for i in 0..<n where !pinned[i] {
            if hasForcesComputed {
                vx[i] += storedFx[i] * attenuation
                vy[i] += storedFy[i] * attenuation
                vz[i] += storedFz[i] * attenuation
            }
            vx[i] *= damping; vy[i] *= damping; vz[i] *= damping

            let speedSq = vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]
            maxSpeedSq = max(maxSpeedSq, speedSq)
            if speedSq > maxSpeed * maxSpeed {
                let scale = maxSpeed / sqrt(speedSq)
                vx[i] *= scale; vy[i] *= scale; vz[i] *= scale
            }

            x[i] += vx[i]; y[i] += vy[i]; z[i] += vz[i]
        }
        alpha = max(alpha * alphaDecay, alphaFloor)
        lastMaxSpeedSq = maxSpeedSq

        // Settle detection: all nodes nearly stationary for 30 consecutive frames.
        if hasForcesComputed && maxSpeedSq < 0.01 {
            settledFrameCount += 1
            if settledFrameCount >= 30 {
                isSettled = true
                syncPositions()  // one final sync
                return
            }
        } else {
            settledFrameCount = 0
        }

        syncPositions()

        // Dispatch force computation whenever the pipeline is idle.
        if !tickInFlight {
            dispatchForceComputation()
        }
    }

    // MARK: - Async force computation

    private struct SimState: Sendable {
        let n: Int
        let x: [Float], y: [Float], z: [Float]
        let projectGroup: [Int], topicGroup: [Int]
        let edgeIndices: [(Int, Int)]
        let topicProjectGroup: [Int]
        let alpha: Float, center: SIMD3<Float>
        let chargeStrength: Float, crossChargeMultiplier: Float
        let sameProjectChargeScale: Float, sameTopicChargeScale: Float
        let centerStrength: Float
        let springLength: Float, crossProjectSpringLength: Float, springStrength: Float
        let cohesionStrength: Float, centroidRepulsion: Float
        let topicCohesionStrength: Float, topicCentroidRepulsion: Float
    }

    private struct ForceResult: Sendable {
        let fx: [Float], fy: [Float], fz: [Float]
    }

    private func dispatchForceComputation() {
        let n = ids.count
        guard n > 1 else { return }

        tickInFlight = true

        // Capture state needed for CPU forces (springs, cohesion, center)
        let state = SimState(
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
            topicCohesionStrength: topicCohesionStrength, topicCentroidRepulsion: topicCentroidRepulsion
        )

        // Try GPU path for charge forces (O(n²) bottleneck)
        if let metal = metalForceCompute, n >= 16 {
            let positions = (x: Array(x), y: Array(y), z: Array(z))
            let projGroups = Array(projectGroup)
            let topGroups = Array(topicGroup)

            // Run entire force pipeline off MainActor — MetalForceCompute is @unchecked Sendable
            // and Metal device/queue/pipeline are thread-safe. The tickInFlight flag ensures
            // at most one concurrent dispatch, so buffer access is safe.
            Task.detached(priority: .userInitiated) { [pendingForces] in
                async let chargeResult = metal.dispatchChargeForces(
                    positions: positions,
                    projectGroups: projGroups,
                    topicGroups: topGroups,
                    chargeStrength: state.chargeStrength,
                    crossChargeMultiplier: state.crossChargeMultiplier,
                    sameTopicChargeScale: state.sameTopicChargeScale,
                    sameProjectChargeScale: state.sameProjectChargeScale
                )
                // Run CPU forces concurrently with GPU charge forces
                let cpuForces = ForceSimulation3D.computeCPUOnlyForces(state)
                let charge = await chargeResult

                let nn = min(charge.fx.count, cpuForces.fx.count)
                var combinedFx = [Float](repeating: 0, count: nn)
                var combinedFy = [Float](repeating: 0, count: nn)
                var combinedFz = [Float](repeating: 0, count: nn)
                for i in 0..<nn {
                    combinedFx[i] = charge.fx[i] + cpuForces.fx[i]
                    combinedFy[i] = charge.fy[i] + cpuForces.fy[i]
                    combinedFz[i] = charge.fz[i] + cpuForces.fz[i]
                }
                let result = ForceResult(fx: combinedFx, fy: combinedFy, fz: combinedFz)
                pendingForces.withLock { $0 = result }
            }
            return
        }

        // CPU fallback path — should not be reached in production (Metal always available on macOS)
        forceSimLog.error("[ForceSimulation3D] ⚠️ CPU fallback path hit — Metal unavailable or n<16 (n=\(n))")
        Task.detached(priority: .userInitiated) { [pendingForces] in
            let result = await Self.computeForces(state)
            pendingForces.withLock { $0 = result }
        }
    }

    /// Compute CPU-only forces (spring attraction, topic/project cohesion, center gravity).
    /// These are O(n) or O(groups²) and not worth GPU-offloading.
    /// CPU-only forces (springs, cohesion, center) — nonisolated static so it can run off MainActor.
    /// Used by GPU path to combine with GPU-computed charge forces.
    private nonisolated static func computeCPUOnlyForces(_ s: SimState) -> ForceResult {
        let n = s.n
        let x = s.x, y = s.y, z = s.z
        let projectGroup = s.projectGroup, topicGroup = s.topicGroup
        var fx = [Float](repeating: 0, count: n)
        var fy = [Float](repeating: 0, count: n)
        var fz = [Float](repeating: 0, count: n)

        let hasProjects = !projectGroup.isEmpty
        let hasTopics = !topicGroup.isEmpty

        // Spring attraction
        for (si, ti) in s.edgeIndices {
            let dx = x[ti] - x[si], dy = y[ti] - y[si], dz = z[ti] - z[si]
            var d = sqrt(dx * dx + dy * dy + dz * dz)
            if d < 1 { d = 1 }
            let cross = hasProjects && projectGroup[si] != projectGroup[ti]
            let rest = cross ? s.crossProjectSpringLength : s.springLength
            let force = s.springStrength * (d - rest)
            let efx = (dx / d) * force, efy = (dy / d) * force, efz = (dz / d) * force
            fx[si] += efx; fy[si] += efy; fz[si] += efz
            fx[ti] -= efx; fy[ti] -= efy; fz[ti] -= efz
        }

        // Topic centroid forces
        let topicN = hasTopics ? ((topicGroup.max() ?? -1) + 1) : 0
        if topicN > 1 {
            var tSumX = [Float](repeating: 0, count: topicN)
            var tSumY = [Float](repeating: 0, count: topicN)
            var tSumZ = [Float](repeating: 0, count: topicN)
            var tCount = [Int](repeating: 0, count: topicN)
            for i in 0..<n {
                let g = topicGroup[i]
                tSumX[g] += x[i]; tSumY[g] += y[i]; tSumZ[g] += z[i]; tCount[g] += 1
            }
            for i in 0..<n {
                let g = topicGroup[i]; let cnt = Float(tCount[g])
                if cnt < 2 { continue }
                let cx = tSumX[g] / cnt, cy = tSumY[g] / cnt, cz = tSumZ[g] / cnt
                fx[i] += (cx - x[i]) * s.topicCohesionStrength
                fy[i] += (cy - y[i]) * s.topicCohesionStrength
                fz[i] += (cz - z[i]) * s.topicCohesionStrength
            }
            let tpg = s.topicProjectGroup
            for g1 in 0..<topicN {
                guard tCount[g1] > 0 else { continue }
                let c1x = tSumX[g1] / Float(tCount[g1])
                let c1y = tSumY[g1] / Float(tCount[g1])
                let c1z = tSumZ[g1] / Float(tCount[g1])
                for g2 in (g1 + 1)..<topicN {
                    guard tCount[g2] > 0 else { continue }
                    guard g1 < tpg.count && g2 < tpg.count, tpg[g1] == tpg[g2] else { continue }
                    let c2x = tSumX[g2] / Float(tCount[g2])
                    let c2y = tSumY[g2] / Float(tCount[g2])
                    let c2z = tSumZ[g2] / Float(tCount[g2])
                    var tdx = c1x - c2x, tdy = c1y - c2y, tdz = c1z - c2z
                    var tdist = sqrt(tdx * tdx + tdy * tdy + tdz * tdz)
                    if tdist < 1 { tdist = 1; tdx = .random(in: -1...1); tdy = .random(in: -1...1); tdz = .random(in: -1...1) }
                    let force = s.topicCentroidRepulsion / (tdist * tdist)
                    let tfx = (tdx / tdist) * force, tfy = (tdy / tdist) * force, tfz = (tdz / tdist) * force
                    let f1 = 1.0 / Float(tCount[g1]), f2 = 1.0 / Float(tCount[g2])
                    for i in 0..<n where topicGroup[i] == g1 {
                        fx[i] += tfx * f1; fy[i] += tfy * f1; fz[i] += tfz * f1
                    }
                    for i in 0..<n where topicGroup[i] == g2 {
                        fx[i] -= tfx * f2; fy[i] -= tfy * f2; fz[i] -= tfz * f2
                    }
                }
            }
        }

        // Project centroid forces (with non-linear cohesion for stray node handling)
        if hasProjects {
            let groupN = (projectGroup.max() ?? -1) + 1
            if groupN > 0 {
                var gSumX = [Float](repeating: 0, count: groupN)
                var gSumY = [Float](repeating: 0, count: groupN)
                var gSumZ = [Float](repeating: 0, count: groupN)
                var gCount = [Int](repeating: 0, count: groupN)
                for i in 0..<n {
                    let g = projectGroup[i]
                    gSumX[g] += x[i]; gSumY[g] += y[i]; gSumZ[g] += z[i]; gCount[g] += 1
                }

                // Compute per-group 75th percentile radius for non-linear cohesion
                var gRadii = [[Float]](repeating: [], count: groupN)
                for i in 0..<n {
                    let g = projectGroup[i]; let cnt = Float(gCount[g])
                    if cnt < 2 { continue }
                    let cx = gSumX[g] / cnt, cy = gSumY[g] / cnt, cz = gSumZ[g] / cnt
                    let dx = x[i] - cx, dy = y[i] - cy, dz = z[i] - cz
                    gRadii[g].append(sqrt(dx * dx + dy * dy + dz * dz))
                }
                var gRefR = [Float](repeating: 30.0, count: groupN)
                for g in 0..<groupN {
                    guard !gRadii[g].isEmpty else { continue }
                    gRadii[g].sort()
                    gRefR[g] = max(30.0, gRadii[g][gRadii[g].count * 3 / 4])
                }

                // Non-linear cohesion: quadratic ramp beyond the cluster's core radius
                let cohStr = s.cohesionStrength
                for i in 0..<n {
                    let g = projectGroup[i]; let cnt = Float(gCount[g])
                    if cnt < 2 { continue }
                    let cx = gSumX[g] / cnt, cy = gSumY[g] / cnt, cz = gSumZ[g] / cnt
                    let dx = cx - x[i], dy = cy - y[i], dz = cz - z[i]
                    let dist = sqrt(dx * dx + dy * dy + dz * dz)
                    let ratio = max(1.0, dist / gRefR[g])
                    let scale = ratio * ratio  // quadratic: 2x beyond edge → 4x force
                    fx[i] += dx * cohStr * scale
                    fy[i] += dy * cohStr * scale
                    fz[i] += dz * cohStr * scale
                }

                for g1 in 0..<groupN {
                    guard gCount[g1] > 0 else { continue }
                    let c1 = SIMD3<Float>(gSumX[g1], gSumY[g1], gSumZ[g1]) / Float(gCount[g1])
                    for g2 in (g1 + 1)..<groupN {
                        guard gCount[g2] > 0 else { continue }
                        let c2 = SIMD3<Float>(gSumX[g2], gSumY[g2], gSumZ[g2]) / Float(gCount[g2])
                        var delta = c1 - c2
                        var pdist = simd_length(delta)
                        if pdist < 1 { pdist = 1; delta = SIMD3(.random(in: -1...1), .random(in: -1...1), .random(in: -1...1)) }
                        let force = s.centroidRepulsion / (pdist * pdist)
                        let fVec = (delta / pdist) * force
                        let f1 = 1.0 / Float(gCount[g1]), f2 = 1.0 / Float(gCount[g2])
                        for i in 0..<n where projectGroup[i] == g1 {
                            fx[i] += fVec.x * f1; fy[i] += fVec.y * f1; fz[i] += fVec.z * f1
                        }
                        for i in 0..<n where projectGroup[i] == g2 {
                            fx[i] -= fVec.x * f2; fy[i] -= fVec.y * f2; fz[i] -= fVec.z * f2
                        }
                    }
                }
            }
        }

        // Center gravity
        let c = s.center
        for i in 0..<n {
            fx[i] += (c.x - x[i]) * s.alpha * s.centerStrength
            fy[i] += (c.y - y[i]) * s.alpha * s.centerStrength
            fz[i] += (c.z - z[i]) * s.alpha * s.centerStrength
        }

        return ForceResult(fx: fx, fy: fy, fz: fz)
    }

    @ForceSimulatorActor
    private static func computeForces(_ s: SimState) -> ForceResult {
        let n = s.n
        let x = s.x, y = s.y, z = s.z
        let projectGroup = s.projectGroup, topicGroup = s.topicGroup
        let alpha = s.alpha

        let hasProjects = !projectGroup.isEmpty
        let hasTopics = !topicGroup.isEmpty

        var fx = [Float](repeating: 0, count: n)
        var fy = [Float](repeating: 0, count: n)
        var fz = [Float](repeating: 0, count: n)

        // --- Charge repulsion (O(n²)) ---
        let chargeBase = s.chargeStrength
        let chargeCross = chargeBase * s.crossChargeMultiplier
        let chargeSameProject = chargeBase * s.sameProjectChargeScale
        let chargeSameTopic = chargeBase * s.sameTopicChargeScale
        let cutoffSq: Float = 500 * 500

        for i in 0..<n {
            let xi = x[i], yi = y[i], zi = z[i]
            let pg_i = hasProjects ? projectGroup[i] : 0
            let tg_i = hasTopics ? topicGroup[i] : -1

            for j in (i + 1)..<n {
                let dx = xi - x[j], dy = yi - y[j], dz = zi - z[j]
                var distSq = dx * dx + dy * dy + dz * dz
                if distSq > cutoffSq { continue }
                if distSq < 1 { distSq = 1 }

                let charge: Float
                if hasProjects && projectGroup[j] != pg_i {
                    charge = chargeCross
                } else if hasTopics && topicGroup[j] == tg_i {
                    charge = chargeSameTopic
                } else if hasProjects {
                    charge = chargeSameProject  // same project, different topic
                } else {
                    charge = chargeBase
                }

                let dist = sqrt(distSq)
                let forceMag = charge / distSq
                let ux = dx / dist, uy = dy / dist, uz = dz / dist
                let ffx = ux * forceMag, ffy = uy * forceMag, ffz = uz * forceMag
                fx[i] += ffx; fy[i] += ffy; fz[i] += ffz
                fx[j] -= ffx; fy[j] -= ffy; fz[j] -= ffz
            }
        }

        // --- Spring attraction ---
        let springStr = s.springStrength
        let springLen = s.springLength
        let crossSpringLen = s.crossProjectSpringLength
        for (si, ti) in s.edgeIndices {
            let dx = x[ti] - x[si], dy = y[ti] - y[si], dz = z[ti] - z[si]
            var d = sqrt(dx * dx + dy * dy + dz * dz)
            if d < 1 { d = 1 }
            let cross = hasProjects && projectGroup[si] != projectGroup[ti]
            let rest = cross ? crossSpringLen : springLen
            let force = springStr * (d - rest)
            let efx = (dx / d) * force, efy = (dy / d) * force, efz = (dz / d) * force
            fx[si] += efx; fy[si] += efy; fz[si] += efz
            fx[ti] -= efx; fy[ti] -= efy; fz[ti] -= efz
        }

        // --- Topic centroid forces ---
        let topicN = (topicGroup.max() ?? -1) + 1
        if topicN > 1 {
            var tSumX = [Float](repeating: 0, count: topicN)
            var tSumY = [Float](repeating: 0, count: topicN)
            var tSumZ = [Float](repeating: 0, count: topicN)
            var tCount = [Int](repeating: 0, count: topicN)
            for i in 0..<n {
                let g = topicGroup[i]
                tSumX[g] += x[i]; tSumY[g] += y[i]; tSumZ[g] += z[i]; tCount[g] += 1
            }
            let topicCoh = s.topicCohesionStrength
            for i in 0..<n {
                let g = topicGroup[i]; let cnt = Float(tCount[g])
                if cnt < 2 { continue }
                let cx = tSumX[g] / cnt, cy = tSumY[g] / cnt, cz = tSumZ[g] / cnt
                fx[i] += (cx - x[i]) * topicCoh
                fy[i] += (cy - y[i]) * topicCoh
                fz[i] += (cz - z[i]) * topicCoh
            }
            let tpg = s.topicProjectGroup
            let topicCentRep = s.topicCentroidRepulsion
            for g1 in 0..<topicN {
                guard tCount[g1] > 0 else { continue }
                let c1x = tSumX[g1] / Float(tCount[g1])
                let c1y = tSumY[g1] / Float(tCount[g1])
                let c1z = tSumZ[g1] / Float(tCount[g1])
                for g2 in (g1 + 1)..<topicN {
                    guard tCount[g2] > 0 else { continue }
                    guard g1 < tpg.count && g2 < tpg.count, tpg[g1] == tpg[g2] else { continue }
                    let c2x = tSumX[g2] / Float(tCount[g2])
                    let c2y = tSumY[g2] / Float(tCount[g2])
                    let c2z = tSumZ[g2] / Float(tCount[g2])
                    var tdx = c1x - c2x, tdy = c1y - c2y, tdz = c1z - c2z
                    var tdist = sqrt(tdx * tdx + tdy * tdy + tdz * tdz)
                    if tdist < 1 { tdist = 1; tdx = .random(in: -1...1); tdy = .random(in: -1...1); tdz = .random(in: -1...1) }
                    let force = topicCentRep / (tdist * tdist)
                    let tfx = (tdx / tdist) * force, tfy = (tdy / tdist) * force, tfz = (tdz / tdist) * force
                    let f1 = 1.0 / Float(tCount[g1]), f2 = 1.0 / Float(tCount[g2])
                    for i in 0..<n where topicGroup[i] == g1 {
                        fx[i] += tfx * f1; fy[i] += tfy * f1; fz[i] += tfz * f1
                    }
                    for i in 0..<n where topicGroup[i] == g2 {
                        fx[i] -= tfx * f2; fy[i] -= tfy * f2; fz[i] -= tfz * f2
                    }
                }
            }
        }

        // --- Project centroid forces (with non-linear cohesion) ---
        if hasProjects {
            let groupN = (projectGroup.max() ?? -1) + 1
            if groupN > 0 {
                var gSumX = [Float](repeating: 0, count: groupN)
                var gSumY = [Float](repeating: 0, count: groupN)
                var gSumZ = [Float](repeating: 0, count: groupN)
                var gCount = [Int](repeating: 0, count: groupN)
                for i in 0..<n {
                    let g = projectGroup[i]
                    gSumX[g] += x[i]; gSumY[g] += y[i]; gSumZ[g] += z[i]; gCount[g] += 1
                }

                // Compute per-group 75th percentile radius for non-linear cohesion
                var gRadii = [[Float]](repeating: [], count: groupN)
                for i in 0..<n {
                    let g = projectGroup[i]; let cnt = Float(gCount[g])
                    if cnt < 2 { continue }
                    let cx = gSumX[g] / cnt, cy = gSumY[g] / cnt, cz = gSumZ[g] / cnt
                    let dx = x[i] - cx, dy = y[i] - cy, dz = z[i] - cz
                    gRadii[g].append(sqrt(dx * dx + dy * dy + dz * dz))
                }
                var gRefR = [Float](repeating: 30.0, count: groupN)
                for g in 0..<groupN {
                    guard !gRadii[g].isEmpty else { continue }
                    gRadii[g].sort()
                    gRefR[g] = max(30.0, gRadii[g][gRadii[g].count * 3 / 4])
                }

                // Non-linear cohesion: quadratic ramp beyond the cluster's core radius
                let cohStr = s.cohesionStrength
                for i in 0..<n {
                    let g = projectGroup[i]; let cnt = Float(gCount[g])
                    if cnt < 2 { continue }
                    let cx = gSumX[g] / cnt, cy = gSumY[g] / cnt, cz = gSumZ[g] / cnt
                    let dx = cx - x[i], dy = cy - y[i], dz = cz - z[i]
                    let dist = sqrt(dx * dx + dy * dy + dz * dz)
                    let ratio = max(1.0, dist / gRefR[g])
                    let scale = ratio * ratio  // quadratic: 2x beyond edge → 4x force
                    fx[i] += dx * cohStr * scale
                    fy[i] += dy * cohStr * scale
                    fz[i] += dz * cohStr * scale
                }

                let centRep = s.centroidRepulsion
                for g1 in 0..<groupN {
                    guard gCount[g1] > 0 else { continue }
                    let c1 = SIMD3<Float>(gSumX[g1], gSumY[g1], gSumZ[g1]) / Float(gCount[g1])
                    for g2 in (g1 + 1)..<groupN {
                        guard gCount[g2] > 0 else { continue }
                        let c2 = SIMD3<Float>(gSumX[g2], gSumY[g2], gSumZ[g2]) / Float(gCount[g2])
                        var delta = c1 - c2
                        var pdist = simd_length(delta)
                        if pdist < 1 { pdist = 1; delta = SIMD3(.random(in: -1...1), .random(in: -1...1), .random(in: -1...1)) }
                        let force = centRep / (pdist * pdist)
                        let fVec = (delta / pdist) * force
                        let f1 = 1.0 / Float(gCount[g1]), f2 = 1.0 / Float(gCount[g2])
                        for i in 0..<n where projectGroup[i] == g1 {
                            fx[i] += fVec.x * f1; fy[i] += fVec.y * f1; fz[i] += fVec.z * f1
                        }
                        for i in 0..<n where projectGroup[i] == g2 {
                            fx[i] -= fVec.x * f2; fy[i] -= fVec.y * f2; fz[i] -= fVec.z * f2
                        }
                    }
                }
            }
        }

        // --- Center gravity ---
        let centerStr = s.centerStrength
        let c = s.center
        for i in 0..<n {
            fx[i] += (c.x - x[i]) * alpha * centerStr
            fy[i] += (c.y - y[i]) * alpha * centerStr
            fz[i] += (c.z - z[i]) * alpha * centerStr
        }

        return ForceResult(fx: fx, fy: fy, fz: fz)
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
