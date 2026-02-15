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

    private(set) var positions: [Int64: CGPoint] = [:]

    private let springLength: CGFloat = 200
    private let crossProjectSpringLength: CGFloat = 350
    private let springStrength: CGFloat = 0.003
    private let chargeStrength: CGFloat = 3000
    private let centerStrength: CGFloat = 0.008
    private let cohesionStrength: CGFloat = 0.015
    private let centroidRepulsion: CGFloat = 8000
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

    func updateGraph(nodeIds: Set<Int64>, edges: [(Int64, Int64)], projectForNode: [Int64: String] = [:]) {
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

        // Add new nodes
        var topologyChanged = newIds.count != ids.count
        for id in nodeIds where idToIndex[id] == nil {
            let angle = CGFloat.random(in: 0...(2 * .pi))
            let r = CGFloat.random(in: 50...250)
            newIds.append(id)
            newX.append(center.x + cos(angle) * r)
            newY.append(center.y + sin(angle) * r)
            newVx.append(0)
            newVy.append(0)
            newPinned.append(false)
            topologyChanged = true
        }

        if topologyChanged { alpha = 1.0 }

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

        // Charge repulsion (all pairs) — extra repulsion between different projects
        let hasProjects = !projectGroup.isEmpty
        for i in 0..<n {
            for j in (i + 1)..<n {
                var dx = x[i] - x[j]
                var dy = y[i] - y[j]
                var dist = sqrt(dx * dx + dy * dy)
                if dist < 1 { dist = 1; dx = .random(in: -1...1); dy = .random(in: -1...1) }

                let crossProject = hasProjects && projectGroup[i] != projectGroup[j]
                let charge = crossProject ? chargeStrength * 3.0 : chargeStrength
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
