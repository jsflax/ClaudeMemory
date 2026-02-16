import Foundation

@globalActor struct ForceSimulatorActor {
    static let shared: ActorType = .init()
    
    actor ActorType {}
}

/// Force-directed graph layout engine with drag support.
/// No internal timer — driven externally by TimelineView calling `tick()` each frame.
/// Uses contiguous arrays for cache-friendly O(n²) force computation.
@MainActor
final class ForceSimulation {
    // SoA layout for cache-friendly iteration
    private var ids: [Int64] = []
    private var x: [CGFloat] = []
    private var y: [CGFloat] = []
    private var vx: [CGFloat] = []
    private var vy: [CGFloat] = []
    private var pinned: [Bool] = []
    private var idToIndex: [Int64: Int] = [:]
    private var edgeIndices: [(Int, Int)] = []
    private var projectGroup: [Int] = []  // group index per node (same project = same group)
    private var topicGroup: [Int] = []    // group index per node (same project+topic = same group)
    private var topicProjectGroup: [Int] = []  // which project group each topic group belongs to
    private var prevTopicGroup: [Int] = []     // previous tick's topic groups, for change detection

    private(set) var positions: [Int64: CGPoint] = [:]

    private let springLength: CGFloat = 240
    private let crossProjectSpringLength: CGFloat = 400
    private let springStrength: CGFloat = 0.003
    private let chargeStrength: CGFloat = 3000
    private let sameTopicChargeScale: CGFloat = 0.35  // same-topic nodes repel much less
    private let centerStrength: CGFloat = 0.006
    private let cohesionStrength: CGFloat = 0.012
    private let centroidRepulsion: CGFloat = 10000
    private let topicCohesionStrength: CGFloat = 0.07
    private let topicCentroidRepulsion: CGFloat = 20000
    private let damping: CGFloat = 0.85
    private let minAlpha: CGFloat = 0.001

    private var alpha: CGFloat = 1.0
    private var alphaDecay: CGFloat = 0.997
    private var maxSpeed: CGFloat = 0
    private var tickInFlight = false
    private var topologyVersion: UInt64 = 0

    var center: CGPoint = CGPoint(x: 400, y: 300)
    /// Settled when alpha is negligible OR nodes have stopped moving perceptibly
    var isActive: Bool { alpha > minAlpha && maxSpeed > 0.5 }

    // MARK: - Dragging

    func pinNode(_ id: Int64, at position: CGPoint) {
        guard let i = idToIndex[id] else { return }
        x[i] = position.x
        y[i] = position.y
        vx[i] = 0
        vy[i] = 0
        pinned[i] = true
        alpha = max(alpha, 0.3)

        // Synchronous local repulsion — immediately push nearby nodes so drag feels reactive.
        // Without this, nearby nodes only move when the next async tick completes (~50ms).
        let n = ids.count
        let hasProjects = !projectGroup.isEmpty
        let hasTopics = !topicGroup.isEmpty
        for j in 0..<n where j != i && !pinned[j] {
            var dx = x[j] - position.x
            var dy = y[j] - position.y
            let distSq = dx * dx + dy * dy
            guard distSq < 300 * 300 else { continue }
            var dist = sqrt(distSq)
            if dist < 1 { dist = 1; dx = .random(in: -1...1); dy = .random(in: -1...1) }

            let crossProject = hasProjects && projectGroup[i] != projectGroup[j]
            let sameTopic = hasTopics && topicGroup[i] == topicGroup[j]
            let charge: CGFloat
            if crossProject { charge = chargeStrength * 3.0 }
            else if sameTopic { charge = chargeStrength * sameTopicChargeScale }
            else { charge = chargeStrength }

            let force = 0.3 * charge / (dist * dist)
            let fx = (dx / dist) * force
            let fy = (dy / dist) * force
            // Apply to velocity (sustained effect picked up by next tick)
            vx[j] += fx; vy[j] += fy
            // Direct position nudge for instant visual feedback
            x[j] += fx * 0.4; y[j] += fy * 0.4
        }

        syncPositions()
    }

    func unpinNode(_ id: Int64) {
        guard let i = idToIndex[id] else { return }
        pinned[i] = false
    }

    // Per-node metadata for surgical operations
    private var nodeProject: [Int64: String] = [:]
    private var nodeTopic: [Int64: String] = [:]
    private var projectToGroupIdx: [String: Int] = [:]
    private var topicKeyToGroupIdx: [String: Int] = [:]

    // MARK: - Surgical node/edge operations

    func addNode(_ id: Int64, project: String, topic: String) {
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

        alpha = max(alpha, 0.4)
        topologyVersion &+= 1
        rebuildPositions()
    }

    func removeNode(_ id: Int64) {
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
        idToIndex.removeValue(forKey: id)
        positions.removeValue(forKey: id)
        nodeProject.removeValue(forKey: id)
        nodeTopic.removeValue(forKey: id)
        prevTopicGroup = topicGroup

        alpha = max(alpha, 0.4)
        topologyVersion &+= 1
        rebuildPositions()
    }

    func addEdge(from sourceId: Int64, to targetId: Int64) {
        guard let si = idToIndex[sourceId], let ti = idToIndex[targetId] else { return }
        if edgeIndices.contains(where: { $0 == (si, ti) }) { return }
        edgeIndices.append((si, ti))
        alpha = max(alpha, 0.3)
        topologyVersion &+= 1
    }

    func removeEdge(from sourceId: Int64, to targetId: Int64) {
        guard let si = idToIndex[sourceId], let ti = idToIndex[targetId] else { return }
        edgeIndices.removeAll { $0 == (si, ti) }
        alpha = max(alpha, 0.2)
        topologyVersion &+= 1
    }

    // MARK: - Full graph reconciliation

    func updateGraph(nodeIds: Set<Int64>, edges: [(Int64, Int64)], projectForNode: [Int64: String] = [:], topicForNode: [Int64: String] = [:]) {
        // Remove nodes no longer in the graph
        var keep = [Bool](repeating: false, count: ids.count)
        for (i, id) in ids.enumerated() {
            keep[i] = nodeIds.contains(id)
        }

        // Compact arrays, preserving only kept nodes
        var newIds: [Int64] = []
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
            alpha = max(alpha, topologyChanged ? 0.4 : 0.3)
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

        topologyVersion &+= 1
        rebuildPositions()
    }

    // MARK: - Simulation tick (O(n²) computation offloaded to background thread)

    private struct SimState: Sendable {
        let n: Int
        let x: [CGFloat], y: [CGFloat], vx: [CGFloat], vy: [CGFloat]
        let pinned: [Bool]
        let projectGroup: [Int], topicGroup: [Int]
        let edgeIndices: [(Int, Int)]
        let topicProjectGroup: [Int]
        let alpha: CGFloat, center: CGPoint
        let chargeStrength: CGFloat, sameTopicChargeScale: CGFloat
        let springLength: CGFloat, crossProjectSpringLength: CGFloat, springStrength: CGFloat
        let centerStrength: CGFloat, cohesionStrength: CGFloat, centroidRepulsion: CGFloat
        let topicCohesionStrength: CGFloat, topicCentroidRepulsion: CGFloat
        let damping: CGFloat, alphaDecay: CGFloat
    }

    private struct SimResult: Sendable {
        let x: [CGFloat], y: [CGFloat], vx: [CGFloat], vy: [CGFloat]
        let alpha: CGFloat, maxSpeed: CGFloat
    }

    func tick() {
        guard alpha > minAlpha, !tickInFlight else { return }

        let n = ids.count
        guard n > 1 else {
            if n == 1 { x[0] = center.x; y[0] = center.y }
            syncPositions()
            return
        }

        tickInFlight = true
        let version = topologyVersion

        let state = SimState(
            n: n, x: x, y: y, vx: vx, vy: vy,
            pinned: pinned, projectGroup: projectGroup,
            topicGroup: topicGroup, edgeIndices: edgeIndices,
            topicProjectGroup: topicProjectGroup,
            alpha: alpha, center: center,
            chargeStrength: chargeStrength, sameTopicChargeScale: sameTopicChargeScale,
            springLength: springLength, crossProjectSpringLength: crossProjectSpringLength,
            springStrength: springStrength, centerStrength: centerStrength,
            cohesionStrength: cohesionStrength, centroidRepulsion: centroidRepulsion,
            topicCohesionStrength: topicCohesionStrength, topicCentroidRepulsion: topicCentroidRepulsion,
            damping: damping, alphaDecay: alphaDecay
        )

        Task {
            let result = await Task.detached(priority: .userInitiated) { @ForceSimulatorActor in
                Self.computeForces(state)
            }.value
            guard self.topologyVersion == version else {
                self.tickInFlight = false
                return
            }
            // Preserve pinned positions (may have been updated via pinNode during computation)
            var pinnedState: [(Int, CGFloat, CGFloat)] = []
            for i in 0..<self.ids.count where self.pinned[i] {
                pinnedState.append((i, self.x[i], self.y[i]))
            }
            self.x = result.x; self.y = result.y
            self.vx = result.vx; self.vy = result.vy
            self.alpha = result.alpha; self.maxSpeed = result.maxSpeed
            for (i, px, py) in pinnedState {
                self.x[i] = px; self.y[i] = py
                self.vx[i] = 0; self.vy[i] = 0
            }
            self.syncPositions()
            self.tickInFlight = false
            // Eagerly dispatch next tick to keep pipeline full (don't wait for next timer fire)
            self.tick()
        }
    }

    /// Pure force computation — runs on background thread, no actor isolation.
    nonisolated private static func computeForces(_ s: SimState) -> SimResult {
        let n = s.n
        actor SendableArray<T> {
            let array: [T]
            init(array: [T]) {
                self.array = array
            }
            subscript(_ idx: Int) -> T {
                get {
                    array[idx]
                }
            }
        }
        
        var x = s.x, y = s.y, vx = s.vx, vy = s.vy
        let pinned = s.pinned
        let projectGroup = s.projectGroup, topicGroup = s.topicGroup
        let alpha = s.alpha

        // Charge repulsion (all pairs)
        let hasProjects = !projectGroup.isEmpty
        let hasTopics = !topicGroup.isEmpty
        
        for i in 0..<n {
            let xi = x[i], yi = y[i]
            for j in (i + 1)..<n {
                var dx = xi - x[j]
                var dy = yi - y[j]
                var dist = sqrt(dx * dx + dy * dy)
                if dist < 1 { dist = 1; dx = .random(in: -1...1); dy = .random(in: -1...1) }

                let crossProject = hasProjects && projectGroup[i] != projectGroup[j]
                let sameTopic = hasTopics && topicGroup[i] == topicGroup[j]
                let charge: CGFloat
                if crossProject {
                    charge = s.chargeStrength * 3.0
                } else if sameTopic {
                    charge = s.chargeStrength * s.sameTopicChargeScale
                } else {
                    charge = s.chargeStrength
                }
                let force = alpha * charge / (dist * dist)
                let fx = (dx / dist) * force, fy = (dy / dist) * force

                if !pinned[i] { vx[i] += fx; vy[i] += fy }
                if !pinned[j] { vx[j] -= fx; vy[j] -= fy }
            }
        }

        // Spring attraction (edges)
        for (si, ti) in s.edgeIndices {
            let dx = x[ti] - x[si], dy = y[ti] - y[si]
            var dist = sqrt(dx * dx + dy * dy)
            if dist < 1 { dist = 1 }
            let crossProject = hasProjects && projectGroup[si] != projectGroup[ti]
            let rest = crossProject ? s.crossProjectSpringLength : s.springLength
            let force = alpha * s.springStrength * (dist - rest)
            let fx = (dx / dist) * force, fy = (dy / dist) * force
            if !pinned[si] { vx[si] += fx; vy[si] += fy }
            if !pinned[ti] { vx[ti] -= fx; vy[ti] -= fy }
        }

        // Topic centroid forces
        let topicN = (topicGroup.max() ?? -1) + 1
        if topicN > 1 {
            var tSumX = [CGFloat](repeating: 0, count: topicN)
            var tSumY = [CGFloat](repeating: 0, count: topicN)
            var tCount = [Int](repeating: 0, count: topicN)
            for i in 0..<n {
                let g = topicGroup[i]
                tSumX[g] += x[i]; tSumY[g] += y[i]; tCount[g] += 1
            }
            for i in 0..<n where !pinned[i] {
                let g = topicGroup[i]
                let cnt = CGFloat(tCount[g])
                if cnt < 2 { continue }
                let cx = tSumX[g] / cnt, cy = tSumY[g] / cnt
                vx[i] += (cx - x[i]) * alpha * s.topicCohesionStrength
                vy[i] += (cy - y[i]) * alpha * s.topicCohesionStrength
            }
            let tpg = s.topicProjectGroup
            for g1 in 0..<topicN {
                guard tCount[g1] > 0 else { continue }
                let c1x = tSumX[g1] / CGFloat(tCount[g1])
                let c1y = tSumY[g1] / CGFloat(tCount[g1])
                for g2 in (g1 + 1)..<topicN {
                    guard tCount[g2] > 0 else { continue }
                    guard g1 < tpg.count && g2 < tpg.count, tpg[g1] == tpg[g2] else { continue }
                    let c2x = tSumX[g2] / CGFloat(tCount[g2])
                    let c2y = tSumY[g2] / CGFloat(tCount[g2])
                    var dx = c1x - c2x, dy = c1y - c2y
                    var dist = sqrt(dx * dx + dy * dy)
                    if dist < 1 { dist = 1; dx = .random(in: -1...1); dy = .random(in: -1...1) }
                    let force = alpha * s.topicCentroidRepulsion / (dist * dist)
                    let fx = (dx / dist) * force, fy = (dy / dist) * force
                    let f1 = 1.0 / CGFloat(tCount[g1])
                    let f2 = 1.0 / CGFloat(tCount[g2])
                    for i in 0..<n where topicGroup[i] == g1 && !pinned[i] {
                        vx[i] += fx * f1; vy[i] += fy * f1
                    }
                    for i in 0..<n where topicGroup[i] == g2 && !pinned[i] {
                        vx[i] -= fx * f2; vy[i] -= fy * f2
                    }
                }
            }
        }

        // Project centroid forces
        if hasProjects {
            let groupN = (projectGroup.max() ?? -1) + 1
            if groupN > 0 {
                var groupSumX = [CGFloat](repeating: 0, count: groupN)
                var groupSumY = [CGFloat](repeating: 0, count: groupN)
                var groupCount = [Int](repeating: 0, count: groupN)
                for i in 0..<n {
                    let g = projectGroup[i]
                    groupSumX[g] += x[i]; groupSumY[g] += y[i]; groupCount[g] += 1
                }
                for i in 0..<n where !pinned[i] {
                    let g = projectGroup[i]
                    let cnt = CGFloat(groupCount[g])
                    if cnt < 2 { continue }
                    let cx = groupSumX[g] / cnt, cy = groupSumY[g] / cnt
                    vx[i] += (cx - x[i]) * alpha * s.cohesionStrength
                    vy[i] += (cy - y[i]) * alpha * s.cohesionStrength
                }
                for g1 in 0..<groupN {
                    guard groupCount[g1] > 0 else { continue }
                    let c1x = groupSumX[g1] / CGFloat(groupCount[g1])
                    let c1y = groupSumY[g1] / CGFloat(groupCount[g1])
                    for g2 in (g1 + 1)..<groupN {
                        guard groupCount[g2] > 0 else { continue }
                        let c2x = groupSumX[g2] / CGFloat(groupCount[g2])
                        let c2y = groupSumY[g2] / CGFloat(groupCount[g2])
                        var dx = c1x - c2x, dy = c1y - c2y
                        var dist = sqrt(dx * dx + dy * dy)
                        if dist < 1 { dist = 1; dx = .random(in: -1...1); dy = .random(in: -1...1) }
                        let force = alpha * s.centroidRepulsion / (dist * dist)
                        let fx = (dx / dist) * force, fy = (dy / dist) * force
                        let f1 = 1.0 / CGFloat(groupCount[g1])
                        let f2 = 1.0 / CGFloat(groupCount[g2])
                        for i in 0..<n where projectGroup[i] == g1 && !pinned[i] {
                            vx[i] += fx * f1; vy[i] += fy * f1
                        }
                        for i in 0..<n where projectGroup[i] == g2 && !pinned[i] {
                            vx[i] -= fx * f2; vy[i] -= fy * f2
                        }
                    }
                }
            }
        }

        // Center gravity + integrate
        var ms: CGFloat = 0
        for i in 0..<n where !pinned[i] {
            vx[i] += (s.center.x - x[i]) * alpha * s.centerStrength
            vy[i] += (s.center.y - y[i]) * alpha * s.centerStrength
            vx[i] *= s.damping; vy[i] *= s.damping
            x[i] += vx[i]; y[i] += vy[i]
            let speed = vx[i] * vx[i] + vy[i] * vy[i]
            if speed > ms { ms = speed }
        }

        return SimResult(x: x, y: y, vx: vx, vy: vy,
                         alpha: alpha * s.alphaDecay, maxSpeed: sqrt(ms))
    }

    /// Full rebuild — only needed after topology changes (add/remove node, updateGraph)
    private func rebuildPositions() {
        positions.removeAll(keepingCapacity: true)
        for i in 0..<ids.count {
            positions[ids[i]] = CGPoint(x: x[i], y: y[i])
        }
    }

    /// In-place update — same key set, just overwrite values. No hashing overhead.
    private func syncPositions() {
        for i in 0..<ids.count {
            positions[ids[i]] = CGPoint(x: x[i], y: y[i])
        }
    }
}
