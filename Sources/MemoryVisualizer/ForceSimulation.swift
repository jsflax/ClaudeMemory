import Foundation

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

    var center: CGPoint = CGPoint(x: 400, y: 300)
    var isActive: Bool { alpha > minAlpha }

    // MARK: - Dragging

    func pinNode(_ id: Int64, at position: CGPoint) {
        guard let i = idToIndex[id] else { return }
        x[i] = position.x
        y[i] = position.y
        vx[i] = 0
        vy[i] = 0
        pinned[i] = true
        alpha = max(alpha, 0.3)
        syncPositions()
    }

    func unpinNode(_ id: Int64) {
        guard let i = idToIndex[id] else { return }
        pinned[i] = false
    }

    // MARK: - Graph updates

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

        syncPositions()
    }

    // MARK: - Simulation tick (called by TimelineView each frame)

    func tick() {
        guard alpha > minAlpha else { return }

        let n = ids.count
        guard n > 1 else {
            if n == 1 { x[0] = center.x; y[0] = center.y }
            syncPositions()
            return
        }

        // Charge repulsion (all pairs)
        // Same topic: reduced charge → nodes sit closer
        // Same project, different topic: normal charge
        // Different project: 3x charge → projects spread far apart
        let hasProjects = !projectGroup.isEmpty
        let hasTopics = !topicGroup.isEmpty
        for i in 0..<n {
            for j in (i + 1)..<n {
                var dx = x[i] - x[j]
                var dy = y[i] - y[j]
                var dist = sqrt(dx * dx + dy * dy)
                if dist < 1 { dist = 1; dx = .random(in: -1...1); dy = .random(in: -1...1) }

                let crossProject = hasProjects && projectGroup[i] != projectGroup[j]
                let sameTopic = hasTopics && topicGroup[i] == topicGroup[j]
                let charge: CGFloat
                if crossProject {
                    charge = chargeStrength * 3.0
                } else if sameTopic {
                    charge = chargeStrength * sameTopicChargeScale
                } else {
                    charge = chargeStrength
                }
                let force = alpha * charge / (dist * dist)
                let fx = (dx / dist) * force, fy = (dy / dist) * force

                if !pinned[i] { vx[i] += fx; vy[i] += fy }
                if !pinned[j] { vx[j] -= fx; vy[j] -= fy }
            }
        }

        // Spring attraction (edges) — longer rest length for cross-project edges
        for (si, ti) in edgeIndices {
            let dx = x[ti] - x[si], dy = y[ti] - y[si]
            var dist = sqrt(dx * dx + dy * dy)
            if dist < 1 { dist = 1 }
            let crossProject = hasProjects && projectGroup[si] != projectGroup[ti]
            let rest = crossProject ? crossProjectSpringLength : springLength
            let force = alpha * springStrength * (dist - rest)
            let fx = (dx / dist) * force, fy = (dy / dist) * force
            if !pinned[si] { vx[si] += fx; vy[si] += fy }
            if !pinned[ti] { vx[ti] -= fx; vy[ti] -= fy }
        }

        // Topic centroid forces: cohesion within topic + repulsion between topics in same project
        let topicN = (topicGroup.max() ?? -1) + 1
        if topicN > 1 {
            var tSumX = [CGFloat](repeating: 0, count: topicN)
            var tSumY = [CGFloat](repeating: 0, count: topicN)
            var tCount = [Int](repeating: 0, count: topicN)
            for i in 0..<n {
                let g = topicGroup[i]
                tSumX[g] += x[i]; tSumY[g] += y[i]; tCount[g] += 1
            }

            // Topic cohesion: pull toward topic centroid (stronger than project cohesion)
            for i in 0..<n where !pinned[i] {
                let g = topicGroup[i]
                let cnt = CGFloat(tCount[g])
                if cnt < 2 { continue }
                let cx = tSumX[g] / cnt, cy = tSumY[g] / cnt
                vx[i] += (cx - x[i]) * alpha * topicCohesionStrength
                vy[i] += (cy - y[i]) * alpha * topicCohesionStrength
            }

            // Topic centroid repulsion: push apart topics within the same project
            for g1 in 0..<topicN {
                guard tCount[g1] > 0 else { continue }
                let c1x = tSumX[g1] / CGFloat(tCount[g1])
                let c1y = tSumY[g1] / CGFloat(tCount[g1])
                for g2 in (g1 + 1)..<topicN {
                    guard tCount[g2] > 0 else { continue }
                    // Only repel topics in the same project
                    guard g1 < topicProjectGroup.count && g2 < topicProjectGroup.count,
                          topicProjectGroup[g1] == topicProjectGroup[g2] else { continue }
                    let c2x = tSumX[g2] / CGFloat(tCount[g2])
                    let c2y = tSumY[g2] / CGFloat(tCount[g2])
                    var dx = c1x - c2x, dy = c1y - c2y
                    var dist = sqrt(dx * dx + dy * dy)
                    if dist < 1 { dist = 1; dx = .random(in: -1...1); dy = .random(in: -1...1) }
                    let force = alpha * topicCentroidRepulsion / (dist * dist)
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

        // Project centroid forces: cohesion (pull toward own centroid) + centroid repulsion (push clusters apart)
        if hasProjects {
            // Compute centroids per project group
            var groupSumX: [CGFloat] = []
            var groupSumY: [CGFloat] = []
            var groupCount: [Int] = []
            let groupN = (projectGroup.max() ?? -1) + 1
            if groupN > 0 {
                groupSumX = [CGFloat](repeating: 0, count: groupN)
                groupSumY = [CGFloat](repeating: 0, count: groupN)
                groupCount = [Int](repeating: 0, count: groupN)
                for i in 0..<n {
                    let g = projectGroup[i]
                    groupSumX[g] += x[i]
                    groupSumY[g] += y[i]
                    groupCount[g] += 1
                }

                // Cohesion: pull each node gently toward its project centroid
                for i in 0..<n where !pinned[i] {
                    let g = projectGroup[i]
                    let cnt = CGFloat(groupCount[g])
                    if cnt < 2 { continue }
                    let cx = groupSumX[g] / cnt, cy = groupSumY[g] / cnt
                    vx[i] += (cx - x[i]) * alpha * cohesionStrength
                    vy[i] += (cy - y[i]) * alpha * cohesionStrength
                }

                // Centroid repulsion: push project centroids apart
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
                        let force = alpha * centroidRepulsion / (dist * dist)
                        let fx = (dx / dist) * force, fy = (dy / dist) * force
                        // Distribute force to all nodes in each group
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
        for i in 0..<n where !pinned[i] {
            vx[i] += (center.x - x[i]) * alpha * centerStrength
            vy[i] += (center.y - y[i]) * alpha * centerStrength
            vx[i] *= damping; vy[i] *= damping
            x[i] += vx[i]; y[i] += vy[i]
        }

        alpha *= alphaDecay
        syncPositions()
    }

    private func syncPositions() {
        // Reuse existing dictionary capacity
        positions.removeAll(keepingCapacity: true)
        for i in 0..<ids.count {
            positions[ids[i]] = CGPoint(x: x[i], y: y[i])
        }
    }
}
