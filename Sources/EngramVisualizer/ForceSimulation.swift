import Foundation
import Accelerate

@globalActor struct ForceSimulatorActor {
    static let shared: ActorType = .init()

    actor ActorType {}
}

/// Force-directed graph layout engine with drag support.
/// Split architecture: O(n²) force computation runs async, O(n) integration runs sync every frame.
/// This ensures positions update at 60fps while expensive forces are 1-2 frames stale at most.
@MainActor
final class ForceSimulation: ObservableObject {
    // SoA layout for cache-friendly iteration
    private var ids: [UUID] = []
    private var x: [CGFloat] = []
    private var y: [CGFloat] = []
    private var vx: [CGFloat] = []
    private var vy: [CGFloat] = []
    private var pinned: [Bool] = []
    private var idToIndex: [UUID: Int] = [:]
    private var edgeIndices: [(Int, Int)] = []
    private var projectGroup: [Int] = []  // group index per node (same project = same group)
    private var topicGroup: [Int] = []    // group index per node (same project+topic = same group)
    private var topicProjectGroup: [Int] = []  // which project group each topic group belongs to
    private var prevTopicGroup: [Int] = []     // previous tick's topic groups, for change detection

    // Stored per-node forces from last async computation.
    // Applied every frame in tick() — forces change slowly so 1-2 frame staleness is imperceptible.
    private var storedFx: [CGFloat] = []
    private var storedFy: [CGFloat] = []

    private(set) var positions: [UUID: CGPoint] = [:]

    private let springLength: CGFloat = 240
    private let crossProjectSpringLength: CGFloat = 400
    private let springStrength: CGFloat = 0.0004
    private let chargeStrength: CGFloat = 400
    private let sameTopicChargeScale: CGFloat = 0.35
    private let centerStrength: CGFloat = 0.006
    private let cohesionStrength: CGFloat = 0.0015
    private let centroidRepulsion: CGFloat = 2000
    private let topicCohesionStrength: CGFloat = 0.009
    private let topicCentroidRepulsion: CGFloat = 3000
    private let damping: CGFloat = 0.78
    private let maxSpeed: CGFloat = 12.0

    private var alpha: CGFloat = 1.0
    private let alphaDecay: CGFloat = 0.997
    private let alphaFloor: CGFloat = 0.01  // never stops — always at live equilibrium
    private var tickInFlight = false
    private var forceAge: Int = 100  // frames since last force computation; starts high so zero-forces aren't amplified
    private var topologyVersion: UInt64 = 0

    /// Set by addNode/addEdge, drained by tick(). Coalesces multiple topology changes
    /// between frames into a single alpha bump instead of one per call.
    private var hasPendingTopologyChanges = false

    var center: CGPoint = CGPoint(x: 400, y: 300)
    var isActive: Bool = true

    // MARK: - Dragging

    func pinNode(_ id: UUID, at position: CGPoint) {
        guard let i = idToIndex[id] else { return }
        x[i] = position.x
        y[i] = position.y
        vx[i] = 0
        vy[i] = 0
        pinned[i] = true
        syncPositions()
    }

    func unpinNode(_ id: UUID) {
        guard let i = idToIndex[id] else { return }
        pinned[i] = false

        // Rubber-band snap: compute velocity impulse toward connected-neighbor centroid
        var homeX: CGFloat = 0, homeY: CGFloat = 0, count: CGFloat = 0
        for (si, ti) in edgeIndices {
            if si == i { homeX += x[ti]; homeY += y[ti]; count += 1 }
            else if ti == i { homeX += x[si]; homeY += y[si]; count += 1 }
        }
        if count > 0 {
            homeX /= count; homeY /= count
            let dx = homeX - x[i], dy = homeY - y[i]
            let dist = sqrt(dx * dx + dy * dy)
            if dist > 1 {
                // Target ~80% of distance in initial impulse: v₀ / (1-damping) ≈ 0.8*dist
                let snapSpeed = min(dist * 0.2, 50)
                vx[i] = (dx / dist) * snapSpeed
                vy[i] = (dy / dist) * snapSpeed
            }
        }

        forceAge = 0
        if !tickInFlight {
            dispatchForceComputation()
        }
    }

    // Per-node metadata for surgical operations
    private var nodeProject: [UUID: String] = [:]
    private var nodeTopic: [UUID: String] = [:]
    private var projectToGroupIdx: [String: Int] = [:]
    private var topicKeyToGroupIdx: [String: Int] = [:]

    // MARK: - Surgical node/edge operations

    func addNode(_ id: UUID, project: String, topic: String) {
        guard idToIndex[id] == nil else { return }

        let idx = ids.count
        ids.append(id)
        idToIndex[id] = idx

        // Spawn near project centroid
        var spawnX = center.x
        var spawnY = center.y
        if let projGroupIdx = projectToGroupIdx[project] {
            var sumX: CGFloat = 0, sumY: CGFloat = 0, count: CGFloat = 0
            for i in 0..<projectGroup.count where projectGroup[i] == projGroupIdx {
                sumX += x[i]; sumY += y[i]; count += 1
            }
            if count > 0 {
                let angle = CGFloat.random(in: 0...(2 * .pi))
                let r = CGFloat.random(in: 20...60)
                spawnX = sumX / count + cos(angle) * r
                spawnY = sumY / count + sin(angle) * r
            }
        } else {
            let angle = CGFloat.random(in: 0...(2 * .pi))
            let r = CGFloat.random(in: 50...250)
            spawnX = center.x + cos(angle) * r
            spawnY = center.y + sin(angle) * r
        }

        x.append(spawnX); y.append(spawnY)
        vx.append(0); vy.append(0)
        pinned.append(false)
        storedFx.append(0); storedFy.append(0)

        // Project group
        let projGroup: Int
        if let g = projectToGroupIdx[project] { projGroup = g }
        else { projGroup = projectToGroupIdx.count; projectToGroupIdx[project] = projGroup }
        projectGroup.append(projGroup)

        // Topic group
        let topicKey = "\(project)|\(topic)"
        let topGroup: Int
        if let g = topicKeyToGroupIdx[topicKey] { topGroup = g }
        else {
            topGroup = topicKeyToGroupIdx.count
            topicKeyToGroupIdx[topicKey] = topGroup
            topicProjectGroup.append(projGroup)
        }
        topicGroup.append(topGroup)

        nodeProject[id] = project
        nodeTopic[id] = topic
        prevTopicGroup = topicGroup

        hasPendingTopologyChanges = true
        rebuildPositions()
    }

    func removeNode(_ id: UUID) {
        guard let idx = idToIndex[id] else { return }
        let lastIdx = ids.count - 1

        // Remove edges involving this node
        edgeIndices.removeAll { $0.0 == idx || $0.1 == idx }

        if idx != lastIdx {
            let lastId = ids[lastIdx]
            ids[idx] = lastId
            x[idx] = x[lastIdx]; y[idx] = y[lastIdx]
            vx[idx] = vx[lastIdx]; vy[idx] = vy[lastIdx]
            pinned[idx] = pinned[lastIdx]
            projectGroup[idx] = projectGroup[lastIdx]
            topicGroup[idx] = topicGroup[lastIdx]
            storedFx[idx] = storedFx[lastIdx]; storedFy[idx] = storedFy[lastIdx]
            idToIndex[lastId] = idx

            // Patch edge indices: lastIdx → idx
            for i in 0..<edgeIndices.count {
                if edgeIndices[i].0 == lastIdx { edgeIndices[i].0 = idx }
                if edgeIndices[i].1 == lastIdx { edgeIndices[i].1 = idx }
            }
        }

        ids.removeLast(); x.removeLast(); y.removeLast()
        vx.removeLast(); vy.removeLast(); pinned.removeLast()
        projectGroup.removeLast(); topicGroup.removeLast()
        storedFx.removeLast(); storedFy.removeLast()
        idToIndex.removeValue(forKey: id)
        positions.removeValue(forKey: id)
        nodeProject.removeValue(forKey: id)
        nodeTopic.removeValue(forKey: id)
        prevTopicGroup = topicGroup

        hasPendingTopologyChanges = true
        rebuildPositions()
    }

    func addEdge(from sourceId: UUID, to targetId: UUID) {
        guard let si = idToIndex[sourceId], let ti = idToIndex[targetId] else { return }
        if edgeIndices.contains(where: { $0 == (si, ti) }) { return }
        edgeIndices.append((si, ti))
        hasPendingTopologyChanges = true
    }

    func removeEdge(from sourceId: UUID, to targetId: UUID) {
        guard let si = idToIndex[sourceId], let ti = idToIndex[targetId] else { return }
        edgeIndices.removeAll { $0 == (si, ti) }
        hasPendingTopologyChanges = true
    }

    // MARK: - Full graph reconciliation

    func updateGraph(nodeIds: Set<UUID>, edges: [(UUID, UUID)], projectForNode: [UUID: String] = [:], topicForNode: [UUID: String] = [:]) {
        // Remove nodes no longer in the graph
        var keep = [Bool](repeating: false, count: ids.count)
        for (i, id) in ids.enumerated() {
            keep[i] = nodeIds.contains(id)
        }

        // Compact arrays, preserving only kept nodes
        var newIds: [UUID] = []
        var newX: [CGFloat] = []
        var newY: [CGFloat] = []
        var newVx: [CGFloat] = []
        var newVy: [CGFloat] = []
        var newPinned: [Bool] = []
        let cap = nodeIds.count
        newIds.reserveCapacity(cap)
        newX.reserveCapacity(cap)
        newY.reserveCapacity(cap)
        newVx.reserveCapacity(cap)
        newVy.reserveCapacity(cap)
        newPinned.reserveCapacity(cap)

        for i in 0..<ids.count where keep[i] {
            newIds.append(ids[i])
            newX.append(x[i])
            newY.append(y[i])
            newVx.append(vx[i])
            newVy.append(vy[i])
            newPinned.append(pinned[i])
        }

        // Add new nodes — spawn near their project cluster's centroid so they don't fly across the screen
        var topologyChanged = newIds.count != ids.count

        // Pre-compute project centroids from existing (kept) nodes
        var projSumX: [String: CGFloat] = [:]
        var projSumY: [String: CGFloat] = [:]
        var projCount: [String: Int] = [:]
        for i in 0..<newIds.count {
            let proj = projectForNode[newIds[i]] ?? ""
            projSumX[proj, default: 0] += newX[i]
            projSumY[proj, default: 0] += newY[i]
            projCount[proj, default: 0] += 1
        }

        for id in nodeIds where idToIndex[id] == nil {
            let proj = projectForNode[id] ?? ""
            let spawnX: CGFloat
            let spawnY: CGFloat
            if let count = projCount[proj], count > 0 {
                // Spawn near project centroid with small jitter
                let cx = projSumX[proj]! / CGFloat(count)
                let cy = projSumY[proj]! / CGFloat(count)
                let angle = CGFloat.random(in: 0...(2 * .pi))
                let r = CGFloat.random(in: 20...60)
                spawnX = cx + cos(angle) * r
                spawnY = cy + sin(angle) * r
            } else {
                // No existing cluster — fall back to center with random offset
                let angle = CGFloat.random(in: 0...(2 * .pi))
                let r = CGFloat.random(in: 50...250)
                spawnX = center.x + cos(angle) * r
                spawnY = center.y + sin(angle) * r
            }
            newIds.append(id)
            newX.append(spawnX)
            newY.append(spawnY)
            newVx.append(0)
            newVy.append(0)
            newPinned.append(false)
            topologyChanged = true

            // Update centroid so subsequent new nodes in same project cluster near each other
            projSumX[proj, default: 0] += spawnX
            projSumY[proj, default: 0] += spawnY
            projCount[proj, default: 0] += 1
        }

        ids = newIds
        x = newX
        y = newY
        vx = newVx
        vy = newVy
        pinned = newPinned

        // Rebuild index
        idToIndex.removeAll(keepingCapacity: true)
        for (i, id) in ids.enumerated() {
            idToIndex[id] = i
        }

        // Build project group indices for inter-project repulsion
        var projectToGroup: [String: Int] = [:]
        projectGroup = ids.map { id in
            let proj = projectForNode[id] ?? ""
            if let g = projectToGroup[proj] { return g }
            let g = projectToGroup.count
            projectToGroup[proj] = g
            return g
        }

        // Build topic group indices for intra-project topic clustering
        // Key = "project|topic", so same topic in different projects = different groups
        var topicKeyToGroup: [String: Int] = [:]
        var topicGroupToProjectGroup: [Int] = []
        topicGroup = ids.map { id in
            let proj = projectForNode[id] ?? ""
            let topic = topicForNode[id] ?? "general"
            let key = "\(proj)|\(topic)"
            if let g = topicKeyToGroup[key] { return g }
            let g = topicKeyToGroup.count
            topicKeyToGroup[key] = g
            topicGroupToProjectGroup.append(projectToGroup[proj] ?? 0)
            return g
        }
        topicProjectGroup = topicGroupToProjectGroup

        // Reheat simulation if topology or topic groupings changed
        let topicGroupsChanged = topicGroup != prevTopicGroup
        if topologyChanged || topicGroupsChanged {
            alpha = max(alpha, topologyChanged ? 0.3 : 0.03)
            prevTopicGroup = topicGroup
        }

        // Convert edges to index pairs
        edgeIndices = edges.compactMap { (src, tgt) in
            guard let si = idToIndex[src], let ti = idToIndex[tgt] else { return nil }
            return (si, ti)
        }

        // Sync per-node metadata for surgical operations
        nodeProject = projectForNode
        nodeTopic = topicForNode
        projectToGroupIdx = projectToGroup
        topicKeyToGroupIdx = topicKeyToGroup

        // Reset stored forces — will be recomputed on next async dispatch
        storedFx = [CGFloat](repeating: 0, count: ids.count)
        storedFy = [CGFloat](repeating: 0, count: ids.count)

        topologyVersion &+= 1
        rebuildPositions()
    }

    // MARK: - Per-frame tick: sync O(n) integration + async force dispatch

    /// Write external positions (e.g. from t-SNE) into internal SoA arrays.
    /// Force layout resumes from these positions when switching back.
    func setPositions(_ positions: [UUID: CGPoint]) {
        for (id, point) in positions {
            guard let i = idToIndex[id] else { continue }
            x[i] = point.x
            y[i] = point.y
            vx[i] = 0
            vy[i] = 0
        }
        // Reset stored forces so simulation doesn't jump on first tick
        storedFx = [CGFloat](repeating: 0, count: ids.count)
        storedFy = [CGFloat](repeating: 0, count: ids.count)
        forceAge = 100
        alpha = 0.05
        syncPositions()
    }

    /// Called every frame by the timer. Applies stored forces, integrates positions (sync O(n)),
    /// then dispatches async O(n²) force recomputation when the pipeline is idle.
    func tick() {
        guard isActive else { return }

        // Coalesce topology changes: single alpha bump per tick regardless of how many
        // addNode/addEdge calls arrived since the last tick.
        if hasPendingTopologyChanges {
            hasPendingTopologyChanges = false
            alpha = max(alpha, 0.05)
            topologyVersion &+= 1
        }

        let n = ids.count
        guard n > 1 else {
            if n == 1 { x[0] = center.x; y[0] = center.y; syncPositions() }
            return
        }

        let hasForcesComputed = storedFx.count == n
        let attenuation = pow(damping, CGFloat(forceAge))
        forceAge += 1

        for i in 0..<n where !pinned[i] {
            if hasForcesComputed {
                vx[i] += storedFx[i] * attenuation
                vy[i] += storedFy[i] * attenuation
            }
            vx[i] *= damping
            vy[i] *= damping

            let speedSq = vx[i] * vx[i] + vy[i] * vy[i]
            if speedSq > maxSpeed * maxSpeed {
                let scale = maxSpeed / sqrt(speedSq)
                vx[i] *= scale
                vy[i] *= scale
            }

            x[i] += vx[i]
            y[i] += vy[i]
        }
        alpha = max(alpha * alphaDecay, alphaFloor)
        syncPositions()

        if !tickInFlight {
            dispatchForceComputation()
        }
    }

    // MARK: - Async force computation

    private struct SimState: Sendable {
        let n: Int
        let x: [CGFloat], y: [CGFloat]
        let projectGroup: [Int], topicGroup: [Int]
        let edgeIndices: [(Int, Int)]
        let topicProjectGroup: [Int]
        let alpha: CGFloat, center: CGPoint
        let chargeStrength: CGFloat, sameTopicChargeScale: CGFloat
        let centerStrength: CGFloat
        let springLength: CGFloat, crossProjectSpringLength: CGFloat, springStrength: CGFloat
        let cohesionStrength: CGFloat, centroidRepulsion: CGFloat
        let topicCohesionStrength: CGFloat, topicCentroidRepulsion: CGFloat
    }

    private struct ForceResult: Sendable {
        let fx: [CGFloat]
        let fy: [CGFloat]
    }

    private func dispatchForceComputation() {
        let n = ids.count
        guard n > 1 else { return }

        tickInFlight = true
        let version = topologyVersion

        let state = SimState(
            n: n, x: x, y: y,
            projectGroup: projectGroup,
            topicGroup: topicGroup, edgeIndices: edgeIndices,
            topicProjectGroup: topicProjectGroup,
            alpha: alpha, center: center,
            chargeStrength: chargeStrength, sameTopicChargeScale: sameTopicChargeScale,
            centerStrength: centerStrength,
            springLength: springLength, crossProjectSpringLength: crossProjectSpringLength,
            springStrength: springStrength,
            cohesionStrength: cohesionStrength, centroidRepulsion: centroidRepulsion,
            topicCohesionStrength: topicCohesionStrength, topicCentroidRepulsion: topicCentroidRepulsion
        )

        Task.detached(priority: .userInitiated) {
            let result = await Self.computeForces(state)
            await MainActor.run { [self] in
                guard topologyVersion == version else {
                    tickInFlight = false
                    return
                }
                storedFx = result.fx
                storedFy = result.fy
                forceAge = 2  // start at 2 so attenuation ≈ 0.72, never full-strength arrival
                tickInFlight = false
            }
        }
    }

    /// Pure force computation — runs on background thread via @ForceSimulatorActor.
    /// Returns per-node force vectors (NOT integrated positions/velocities).
    /// Charge repulsion vectorized via Accelerate/vDSP; other forces remain scalar.
    @ForceSimulatorActor
    private static func computeForces(_ s: SimState) -> ForceResult {
        let n = s.n
        let x = s.x.map(Double.init)
        let y = s.y.map(Double.init)
        let projectGroup = s.projectGroup, topicGroup = s.topicGroup
        let alpha = Double(s.alpha)

        let hasProjects = !projectGroup.isEmpty
        let hasTopics = !topicGroup.isEmpty
        let vn = vDSP_Length(n)

        // Per-node force accumulators
        var fx = [Double](repeating: 0, count: n)
        var fy = [Double](repeating: 0, count: n)

        // --- Charge repulsion (O(n²), vectorized per-node via vDSP) ---
        // Distance cutoff: no charge beyond 500px — centroid forces handle long-range.
        let chargeBase = Double(s.chargeStrength)
        let chargeCross = chargeBase * 3.0
        let chargeSameTopic = chargeBase * Double(s.sameTopicChargeScale)
        let cutoffSq = 500.0 * 500.0

        // Scratch buffers (reused each iteration)
        var dx = [Double](repeating: 0, count: n)
        var dy = [Double](repeating: 0, count: n)
        var distSq = [Double](repeating: 0, count: n)
        var dist = [Double](repeating: 0, count: n)
        var charge = [Double](repeating: chargeBase, count: n)
        var forceMag = [Double](repeating: 0, count: n)
        var nodeFx = [Double](repeating: 0, count: n)
        var nodeFy = [Double](repeating: 0, count: n)
        var scratch = [Double](repeating: 0, count: n)

        // Compute charge forces for ALL nodes (including pinned) so stored forces
        // are ready when a node is unpinned — enables instant snap-back on drag release.
        for i in 0..<n {
            var xi = x[i], yi = y[i]

            // dx = x[i] - x[:], dy = y[i] - y[:]
            vDSP_vfillD(&xi, &scratch, 1, vn)
            vDSP_vsubD(x, 1, scratch, 1, &dx, 1, vn)
            vDSP_vfillD(&yi, &scratch, 1, vn)
            vDSP_vsubD(y, 1, scratch, 1, &dy, 1, vn)

            // distSq = dx² + dy², clamped to min 1.0
            vDSP_vsqD(dx, 1, &distSq, 1, vn)
            vDSP_vsqD(dy, 1, &scratch, 1, vn)
            vDSP_vaddD(distSq, 1, scratch, 1, &distSq, 1, vn)
            var one = 1.0
            vDSP_vthrD(distSq, 1, &one, &distSq, 1, vn)

            // dist = sqrt(distSq)
            var n32 = Int32(n)
            vvsqrt(&dist, distSq, &n32)

            // Charge scale per j (project/topic dependent)
            if hasProjects || hasTopics {
                let pg_i = hasProjects ? projectGroup[i] : 0
                let tg_i = hasTopics ? topicGroup[i] : -1
                for j in 0..<n {
                    let cross = hasProjects && projectGroup[j] != pg_i
                    if cross {
                        charge[j] = chargeCross
                    } else if hasTopics && topicGroup[j] == tg_i {
                        charge[j] = chargeSameTopic
                    } else {
                        charge[j] = chargeBase
                    }
                }
            }

            // forceMag = charge / distSq, zeroed beyond cutoff
            // No alpha scaling — charge is a local force (500px cutoff + inverse-square),
            // so it stays active at steady state for responsive drag interactions.
            vDSP_vdivD(distSq, 1, charge, 1, &forceMag, 1, vn)
            for j in 0..<n where distSq[j] > cutoffSq {
                forceMag[j] = 0
            }

            // nodeFx = (dx / dist) * forceMag, nodeFy = (dy / dist) * forceMag
            vDSP_vdivD(dist, 1, dx, 1, &nodeFx, 1, vn)
            vDSP_vmulD(nodeFx, 1, forceMag, 1, &nodeFx, 1, vn)
            vDSP_vdivD(dist, 1, dy, 1, &nodeFy, 1, vn)
            vDSP_vmulD(nodeFy, 1, forceMag, 1, &nodeFy, 1, vn)

            // Accumulate total force on node i
            var sumFx = 0.0, sumFy = 0.0
            vDSP_sveD(nodeFx, 1, &sumFx, vn)
            vDSP_sveD(nodeFy, 1, &sumFy, vn)
            fx[i] += sumFx
            fy[i] += sumFy
        }

        // --- Spring attraction (O(edges), scalar) ---
        let springStr = Double(s.springStrength)
        let springLen = Double(s.springLength)
        let crossSpringLen = Double(s.crossProjectSpringLength)
        for (si, ti) in s.edgeIndices {
            let edx = x[ti] - x[si], edy = y[ti] - y[si]
            var d = sqrt(edx * edx + edy * edy)
            if d < 1 { d = 1 }
            let cross = hasProjects && projectGroup[si] != projectGroup[ti]
            let rest = cross ? crossSpringLen : springLen
            let force = springStr * (d - rest)
            let efx = (edx / d) * force, efy = (edy / d) * force
            fx[si] += efx; fy[si] += efy
            fx[ti] -= efx; fy[ti] -= efy
        }

        // --- Topic centroid forces (scalar) ---
        let topicN = (topicGroup.max() ?? -1) + 1
        if topicN > 1 {
            var tSumX = [Double](repeating: 0, count: topicN)
            var tSumY = [Double](repeating: 0, count: topicN)
            var tCount = [Int](repeating: 0, count: topicN)
            for i in 0..<n {
                let g = topicGroup[i]
                tSumX[g] += x[i]; tSumY[g] += y[i]; tCount[g] += 1
            }
            let topicCoh = Double(s.topicCohesionStrength)
            for i in 0..<n {
                let g = topicGroup[i]
                let cnt = Double(tCount[g])
                if cnt < 2 { continue }
                let cx = tSumX[g] / cnt, cy = tSumY[g] / cnt
                fx[i] += (cx - x[i]) * topicCoh
                fy[i] += (cy - y[i]) * topicCoh
            }
            let tpg = s.topicProjectGroup
            let topicCentRep = Double(s.topicCentroidRepulsion)
            for g1 in 0..<topicN {
                guard tCount[g1] > 0 else { continue }
                let c1x = tSumX[g1] / Double(tCount[g1])
                let c1y = tSumY[g1] / Double(tCount[g1])
                for g2 in (g1 + 1)..<topicN {
                    guard tCount[g2] > 0 else { continue }
                    guard g1 < tpg.count && g2 < tpg.count, tpg[g1] == tpg[g2] else { continue }
                    let c2x = tSumX[g2] / Double(tCount[g2])
                    let c2y = tSumY[g2] / Double(tCount[g2])
                    var tdx = c1x - c2x, tdy = c1y - c2y
                    var tdist = sqrt(tdx * tdx + tdy * tdy)
                    if tdist < 1 { tdist = 1; tdx = .random(in: -1...1); tdy = .random(in: -1...1) }
                    let force = topicCentRep / (tdist * tdist)
                    let tfx = (tdx / tdist) * force, tfy = (tdy / tdist) * force
                    let f1 = 1.0 / Double(tCount[g1])
                    let f2 = 1.0 / Double(tCount[g2])
                    for i in 0..<n where topicGroup[i] == g1 {
                        fx[i] += tfx * f1; fy[i] += tfy * f1
                    }
                    for i in 0..<n where topicGroup[i] == g2 {
                        fx[i] -= tfx * f2; fy[i] -= tfy * f2
                    }
                }
            }
        }

        // --- Project centroid forces (scalar) ---
        if hasProjects {
            let groupN = (projectGroup.max() ?? -1) + 1
            if groupN > 0 {
                var groupSumX = [Double](repeating: 0, count: groupN)
                var groupSumY = [Double](repeating: 0, count: groupN)
                var groupCount = [Int](repeating: 0, count: groupN)
                for i in 0..<n {
                    let g = projectGroup[i]
                    groupSumX[g] += x[i]; groupSumY[g] += y[i]; groupCount[g] += 1
                }
                let cohStr = Double(s.cohesionStrength)
                for i in 0..<n {
                    let g = projectGroup[i]
                    let cnt = Double(groupCount[g])
                    if cnt < 2 { continue }
                    let cx = groupSumX[g] / cnt, cy = groupSumY[g] / cnt
                    fx[i] += (cx - x[i]) * cohStr
                    fy[i] += (cy - y[i]) * cohStr
                }
                let centRep = Double(s.centroidRepulsion)
                for g1 in 0..<groupN {
                    guard groupCount[g1] > 0 else { continue }
                    let c1x = groupSumX[g1] / Double(groupCount[g1])
                    let c1y = groupSumY[g1] / Double(groupCount[g1])
                    for g2 in (g1 + 1)..<groupN {
                        guard groupCount[g2] > 0 else { continue }
                        let c2x = groupSumX[g2] / Double(groupCount[g2])
                        let c2y = groupSumY[g2] / Double(groupCount[g2])
                        var pdx = c1x - c2x, pdy = c1y - c2y
                        var pdist = sqrt(pdx * pdx + pdy * pdy)
                        if pdist < 1 { pdist = 1; pdx = .random(in: -1...1); pdy = .random(in: -1...1) }
                        let force = centRep / (pdist * pdist)
                        let pfx = (pdx / pdist) * force, pfy = (pdy / pdist) * force
                        let f1 = 1.0 / Double(groupCount[g1])
                        let f2 = 1.0 / Double(groupCount[g2])
                        for i in 0..<n where projectGroup[i] == g1 {
                            fx[i] += pfx * f1; fy[i] += pfy * f1
                        }
                        for i in 0..<n where projectGroup[i] == g2 {
                            fx[i] -= pfx * f2; fy[i] -= pfy * f2
                        }
                    }
                }
            }
        }

        // --- Center gravity ---
        let centerStr = Double(s.centerStrength)
        let cx = Double(s.center.x), cy = Double(s.center.y)
        for i in 0..<n {
            fx[i] += (cx - x[i]) * alpha * centerStr
            fy[i] += (cy - y[i]) * alpha * centerStr
        }

        // Damping and position integration are done per-frame in tick()
        return ForceResult(fx: fx.map { CGFloat($0) }, fy: fy.map { CGFloat($0) })
    }

    /// Full rebuild — only needed after topology changes (add/remove node, updateGraph)
    private func rebuildPositions() {
        positions.removeAll(keepingCapacity: true)
        for i in 0..<ids.count {
            positions[ids[i]] = CGPoint(x: x[i], y: y[i])
        }
    }

    /// In-place update — same key set, just overwrite values.
    private func syncPositions() {
        for i in 0..<ids.count {
            positions[ids[i]] = CGPoint(x: x[i], y: y[i])
        }
    }
}
