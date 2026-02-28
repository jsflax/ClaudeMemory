import SwiftUI
import simd

// MARK: - Clustering & Void Detection

extension EmbeddingProjection {

    // MARK: - Knowledge Void Detection

    /// Detect sparse regions in the projected 2D space between clusters.
    /// Uses nearest-node distance on a grid — a void is a region far from any node
    /// but surrounded by populated areas (between clusters, not at the edges).
    func detectVoids(nodeTopics: [UUID: String], nodeProjects: [UUID: String]) {
        guard state == .ready, detectionPositions.count >= 5 else {
            knowledgeVoids = []
            return
        }

        let points = Array(detectionPositions.values)

        // Compute bounding box
        var bMinX = CGFloat.greatestFiniteMagnitude, bMinY = CGFloat.greatestFiniteMagnitude
        var bMaxX = -CGFloat.greatestFiniteMagnitude, bMaxY = -CGFloat.greatestFiniteMagnitude
        for p in points {
            bMinX = min(bMinX, p.x); bMinY = min(bMinY, p.y)
            bMaxX = max(bMaxX, p.x); bMaxY = max(bMaxY, p.y)
        }
        let padding: CGFloat = 30
        bMinX -= padding; bMinY -= padding
        bMaxX += padding; bMaxY += padding
        let width = bMaxX - bMinX
        let height = bMaxY - bMinY
        guard width > 0, height > 0 else { knowledgeVoids = []; return }

        // For each cell in a 30x30 grid, compute distance to nearest node
        let gridSize = 30
        let cellW = width / CGFloat(gridSize)
        let cellH = height / CGFloat(gridSize)

        var minDist = [CGFloat](repeating: .greatestFiniteMagnitude, count: gridSize * gridSize)
        for gy in 0..<gridSize {
            for gx in 0..<gridSize {
                let cx = bMinX + (CGFloat(gx) + 0.5) * cellW
                let cy = bMinY + (CGFloat(gy) + 0.5) * cellH
                for p in points {
                    let d = hypot(cx - p.x, cy - p.y)
                    let idx = gy * gridSize + gx
                    if d < minDist[idx] { minDist[idx] = d }
                }
            }
        }

        // Threshold: cells with min-distance above the 70th percentile are "empty"
        let sortedDists = minDist.sorted()
        let emptyThreshold = sortedDists[Int(Double(sortedDists.count) * 0.70)]
        // "Populated" threshold: cells close to nodes (below 30th percentile)
        let populatedThreshold = sortedDists[Int(Double(sortedDists.count) * 0.30)]

        // Mark void cells: empty AND within 3 cells of a populated cell
        var isVoid = [Bool](repeating: false, count: gridSize * gridSize)
        for gy in 1..<(gridSize - 1) {
            for gx in 1..<(gridSize - 1) {
                let idx = gy * gridSize + gx
                guard minDist[idx] > emptyThreshold else { continue }

                // Check 3-cell radius for a populated cell
                var nearPopulated = false
                let radius = 3
                outer: for dy in -radius...radius {
                    for dx in -radius...radius {
                        let ny = gy + dy, nx = gx + dx
                        guard ny >= 0, ny < gridSize, nx >= 0, nx < gridSize else { continue }
                        if minDist[ny * gridSize + nx] < populatedThreshold {
                            nearPopulated = true
                            break outer
                        }
                    }
                }
                if nearPopulated { isVoid[idx] = true }
            }
        }

        // Flood-fill adjacent void cells into regions
        var visited = [Bool](repeating: false, count: gridSize * gridSize)
        var voids: [KnowledgeVoid] = []

        for startIdx in 0..<(gridSize * gridSize) {
            guard isVoid[startIdx], !visited[startIdx] else { continue }

            var queue = [startIdx]
            var region: [Int] = []
            visited[startIdx] = true

            while !queue.isEmpty {
                let idx = queue.removeFirst()
                region.append(idx)
                let gy = idx / gridSize, gx = idx % gridSize
                let neighbors = [
                    gy > 0 ? (gy - 1) * gridSize + gx : -1,
                    gy < gridSize - 1 ? (gy + 1) * gridSize + gx : -1,
                    gx > 0 ? gy * gridSize + (gx - 1) : -1,
                    gx < gridSize - 1 ? gy * gridSize + (gx + 1) : -1
                ]
                for ni in neighbors where ni >= 0 && !visited[ni] && isVoid[ni] {
                    visited[ni] = true
                    queue.append(ni)
                }
            }

            guard region.count >= 2 else { continue }

            // Compute centroid and radius
            var cx: CGFloat = 0, cy: CGFloat = 0
            for idx in region {
                let gy = idx / gridSize, gx = idx % gridSize
                cx += bMinX + (CGFloat(gx) + 0.5) * cellW
                cy += bMinY + (CGFloat(gy) + 0.5) * cellH
            }
            cx /= CGFloat(region.count)
            cy /= CGFloat(region.count)
            let radius = sqrt(CGFloat(region.count) * cellW * cellH / .pi)

            // Find nearest topics
            let voidCenter = CGPoint(x: cx, y: cy)
            var topicDistances: [(topic: String, dist: CGFloat)] = []
            for (id, pos) in detectionPositions {
                guard let topic = nodeTopics[id], topic != "general" else { continue }
                let dx = pos.x - voidCenter.x, dy = pos.y - voidCenter.y
                topicDistances.append((topic: topic, dist: hypot(dx, dy)))
            }
            topicDistances.sort { $0.dist < $1.dist }
            var seen = Set<String>()
            let nearestTopics = topicDistances.prefix(10).compactMap { item -> String? in
                seen.insert(item.topic).inserted ? item.topic : nil
            }
            let uniqueTopics = Array(nearestTopics.prefix(3))

            // Sparsity based on average min-distance in the region
            let maxMinDist = sortedDists.last ?? 1
            let avgMinDist = region.reduce(CGFloat(0)) { $0 + minDist[$1] } / CGFloat(region.count)
            let sparsity = min(avgMinDist / maxMinDist, 1.0)

            voids.append(KnowledgeVoid(
                position: voidCenter,
                radius: max(radius, 40),
                nearestTopics: uniqueTopics,
                sparsity: min(max(sparsity, 0), 1)
            ))
        }

        knowledgeVoids = voids
    }

    // MARK: - Semantic Cluster Detection

    /// Detect spatial clusters in the projected 2D space using hierarchical k-means.
    /// Groups nearby nodes and labels clusters by dominant topics/projects.
    /// Large clusters are sub-divided into sub-clusters for finer-grained visual grouping.
    func detectClusters(nodeTopics: [UUID: String], nodeProjects: [UUID: String], nodeLabels: [UUID: String]) {
        guard state == .ready, detectionPositions.count >= 3 else {
            semanticClusters = []
            return
        }

        let ids = Array(detectionPositions.keys)
        let positions = ids.map { detectionPositions[$0]! }
        let n = ids.count

        // K-means clustering with farthest-first initialization (deterministic).
        let targetK = max(3, min(15, Int(sqrt(Double(n) / 3.0))))
        let minClusterSize = 3
        let clusterGroups = kMeans(positions: positions, k: targetK, minClusterSize: minClusterSize)

        // Build cluster structs with sub-clusters
        var clusters: [SemanticCluster] = []
        for memberIndices in clusterGroups {
            guard memberIndices.count >= minClusterSize else { continue }

            let cluster = buildCluster(
                memberIndices: memberIndices, ids: ids, positions: positions,
                nodeTopics: nodeTopics, nodeProjects: nodeProjects, hullPadding: 20
            )

            // Sub-cluster large clusters (>= 8 nodes, split into 2-3 sub-clusters)
            if memberIndices.count >= 8 {
                let subK = max(2, min(4, memberIndices.count / 6))
                let subPositions = memberIndices.map { positions[$0] }
                let subGroups = kMeans(positions: subPositions, k: subK, minClusterSize: 3)

                var subs: [SemanticCluster] = []
                for subMemberIndices in subGroups where subMemberIndices.count >= 3 {
                    // Map sub-indices back to original indices
                    let originalIndices = subMemberIndices.map { memberIndices[$0] }
                    let sub = buildCluster(
                        memberIndices: originalIndices, ids: ids, positions: positions,
                        nodeTopics: nodeTopics, nodeProjects: nodeProjects, hullPadding: 10
                    )
                    subs.append(sub)
                }
                // Only show sub-clusters if we got at least 2
                var clusterWithSubs = cluster
                if subs.count >= 2 {
                    clusterWithSubs.subClusters = subs.sorted { $0.nodeIds.count > $1.nodeIds.count }
                }
                clusters.append(clusterWithSubs)
            } else {
                clusters.append(cluster)
            }
        }

        semanticClusters = clusters.sorted { $0.nodeIds.count > $1.nodeIds.count }
    }

    /// Build a SemanticCluster from a set of member indices into the ids/positions arrays.
    func buildCluster(
        memberIndices: [Int], ids: [UUID], positions: [CGPoint],
        nodeTopics: [UUID: String], nodeProjects: [UUID: String], hullPadding: CGFloat
    ) -> SemanticCluster {
        let memberIds = memberIndices.map { ids[$0] }
        let memberPositions = memberIndices.map { positions[$0] }

        let cx = memberPositions.map(\.x).reduce(0, +) / CGFloat(memberPositions.count)
        let cy = memberPositions.map(\.y).reduce(0, +) / CGFloat(memberPositions.count)

        let hull: [CGPoint]
        if memberPositions.count >= 3 {
            hull = ConvexHull.expand(ConvexHull.compute(points: memberPositions), by: hullPadding)
        } else {
            hull = []
        }

        var topicCounts: [String: Int] = [:]
        var projectCounts: [String: Int] = [:]
        for id in memberIds {
            if let topic = nodeTopics[id], topic != "general", topic != "episode" {
                topicCounts[topic, default: 0] += 1
            }
            if let project = nodeProjects[id] {
                projectCounts[project, default: 0] += 1
            }
        }

        let topTopics = topicCounts.sorted { $0.value > $1.value }.prefix(3).map(\.key)
        let projectBreakdown = projectCounts.sorted { $0.value > $1.value }.map { (project: $0.key, count: $0.value) }

        let label: String
        if topTopics.isEmpty {
            label = projectBreakdown.first?.project ?? "cluster"
        } else {
            label = topTopics.joined(separator: " · ")
        }

        return SemanticCluster(
            nodeIds: memberIds,
            centroid: CGPoint(x: cx, y: cy),
            hullPoints: hull,
            label: label,
            projectBreakdown: projectBreakdown
        )
    }

    // MARK: - Helpers

    /// K-means clustering on 2D positions. Returns array of (clusterIndex, memberIndices).
    /// Uses farthest-first initialization (deterministic) and Lloyd's algorithm.
    func kMeans(positions: [CGPoint], k: Int, minClusterSize: Int) -> [[Int]] {
        let n = positions.count
        guard n >= k, k >= 2 else { return [(0..<n).map { $0 }] }

        // Farthest-first initialization
        var centroids: [CGPoint] = []
        let globalCx = positions.map(\.x).reduce(0, +) / CGFloat(n)
        let globalCy = positions.map(\.y).reduce(0, +) / CGFloat(n)

        var bestStart = 0
        var bestStartDist: CGFloat = .greatestFiniteMagnitude
        for i in 0..<n {
            let d = hypot(positions[i].x - globalCx, positions[i].y - globalCy)
            if d < bestStartDist { bestStartDist = d; bestStart = i }
        }
        centroids.append(positions[bestStart])

        for _ in 1..<k {
            var farthestIdx = 0
            var farthestDist: CGFloat = 0
            for i in 0..<n {
                var minD: CGFloat = .greatestFiniteMagnitude
                for c in centroids {
                    let d = hypot(positions[i].x - c.x, positions[i].y - c.y)
                    minD = min(minD, d)
                }
                if minD > farthestDist { farthestDist = minD; farthestIdx = i }
            }
            centroids.append(positions[farthestIdx])
        }

        // Lloyd's algorithm
        var assignments = [Int](repeating: 0, count: n)
        for _ in 0..<30 {
            var changed = false
            for i in 0..<n {
                var bestDist: CGFloat = .greatestFiniteMagnitude
                var bestC = 0
                for c in 0..<k {
                    let dx = positions[i].x - centroids[c].x
                    let dy = positions[i].y - centroids[c].y
                    let d = dx * dx + dy * dy
                    if d < bestDist { bestDist = d; bestC = c }
                }
                if assignments[i] != bestC { changed = true }
                assignments[i] = bestC
            }
            if !changed { break }

            for c in 0..<k {
                let members = (0..<n).filter { assignments[$0] == c }
                guard !members.isEmpty else { continue }
                let cx = members.map { positions[$0].x }.reduce(0, +) / CGFloat(members.count)
                let cy = members.map { positions[$0].y }.reduce(0, +) / CGFloat(members.count)
                centroids[c] = CGPoint(x: cx, y: cy)
            }
        }

        // Merge small clusters into nearest neighbor
        for c in 0..<k {
            let memberCount = (0..<n).filter { assignments[$0] == c }.count
            guard memberCount > 0, memberCount < minClusterSize else { continue }
            var bestOther = -1
            var bestDist: CGFloat = .greatestFiniteMagnitude
            for other in 0..<k where other != c {
                let otherCount = (0..<n).filter { assignments[$0] == other }.count
                guard otherCount >= minClusterSize else { continue }
                let d = hypot(centroids[c].x - centroids[other].x, centroids[c].y - centroids[other].y)
                if d < bestDist { bestDist = d; bestOther = other }
            }
            if bestOther >= 0 {
                for i in 0..<n where assignments[i] == c { assignments[i] = bestOther }
            }
        }

        // Collect clusters
        let usedLabels = Set(assignments).sorted()
        return usedLabels.map { c in (0..<n).filter { assignments[$0] == c } }
    }

    func standardDeviation(_ values: [CGFloat]) -> CGFloat {
        let n = CGFloat(values.count)
        guard n > 1 else { return 1 }
        let mean = values.reduce(0, +) / n
        let variance = values.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / (n - 1)
        return sqrt(variance)
    }

    // MARK: - 3D Semantic Cluster Detection

    /// Detect spatial clusters in the projected 3D space using k-means.
    func detectClusters3D(nodeTopics: [UUID: String], nodeProjects: [UUID: String], nodeLabels: [UUID: String]) {
        let positions3D = projectedPositions3D
        guard state == .ready, positions3D.count >= 3 else {
            semanticClusters3D = []
            return
        }

        let ids = Array(positions3D.keys)
        let positions = ids.map { positions3D[$0]! }
        let n = ids.count

        let targetK = max(3, min(15, Int(sqrt(Double(n) / 3.0))))
        let minClusterSize = 3
        let clusterGroups = kMeans3D(positions: positions, k: targetK, minClusterSize: minClusterSize)

        var clusters: [SemanticCluster3D] = []
        for memberIndices in clusterGroups {
            guard memberIndices.count >= minClusterSize else { continue }

            let memberIds = memberIndices.map { ids[$0] }
            let memberPositions = memberIndices.map { positions[$0] }

            // Centroid
            var sum = SIMD3<Float>.zero
            for p in memberPositions { sum += p }
            let centroid = sum / Float(memberPositions.count)

            // Bounding radius
            var maxDist: Float = 0
            for p in memberPositions {
                let d = simd_length(p - centroid)
                if d > maxDist { maxDist = d }
            }

            // Label
            var topicCounts: [String: Int] = [:]
            var projectCounts: [String: Int] = [:]
            for id in memberIds {
                if let topic = nodeTopics[id], topic != "general", topic != "episode" {
                    topicCounts[topic, default: 0] += 1
                }
                if let project = nodeProjects[id] {
                    projectCounts[project, default: 0] += 1
                }
            }
            let topTopics = topicCounts.sorted { $0.value > $1.value }.prefix(3).map(\.key)
            let projectBreakdown = projectCounts.sorted { $0.value > $1.value }.map { (project: $0.key, count: $0.value) }
            let label: String
            if topTopics.isEmpty {
                label = projectBreakdown.first?.project ?? "cluster"
            } else {
                label = topTopics.joined(separator: " · ")
            }

            clusters.append(SemanticCluster3D(
                nodeIds: memberIds,
                centroid: centroid,
                boundingRadius: maxDist,
                label: label,
                projectBreakdown: projectBreakdown
            ))
        }

        semanticClusters3D = clusters.sorted { $0.nodeIds.count > $1.nodeIds.count }
    }

    /// K-means clustering on 3D positions. Same algorithm as 2D kMeans but using SIMD3<Float>.
    func kMeans3D(positions: [SIMD3<Float>], k: Int, minClusterSize: Int) -> [[Int]] {
        let n = positions.count
        guard n >= k, k >= 2 else { return [(0..<n).map { $0 }] }

        // Farthest-first initialization
        var centroids: [SIMD3<Float>] = []
        var sum = SIMD3<Float>.zero
        for p in positions { sum += p }
        let globalCentroid = sum / Float(n)

        var bestStart = 0
        var bestStartDist: Float = .greatestFiniteMagnitude
        for i in 0..<n {
            let d = simd_length(positions[i] - globalCentroid)
            if d < bestStartDist { bestStartDist = d; bestStart = i }
        }
        centroids.append(positions[bestStart])

        for _ in 1..<k {
            var farthestIdx = 0
            var farthestDist: Float = 0
            for i in 0..<n {
                var minD: Float = .greatestFiniteMagnitude
                for c in centroids {
                    let d = simd_length(positions[i] - c)
                    minD = min(minD, d)
                }
                if minD > farthestDist { farthestDist = minD; farthestIdx = i }
            }
            centroids.append(positions[farthestIdx])
        }

        // Lloyd's algorithm
        var assignments = [Int](repeating: 0, count: n)
        for _ in 0..<30 {
            var changed = false
            for i in 0..<n {
                var bestDist: Float = .greatestFiniteMagnitude
                var bestC = 0
                for c in 0..<k {
                    let delta = positions[i] - centroids[c]
                    let d = simd_length_squared(delta)
                    if d < bestDist { bestDist = d; bestC = c }
                }
                if assignments[i] != bestC { changed = true }
                assignments[i] = bestC
            }
            if !changed { break }

            for c in 0..<k {
                let members = (0..<n).filter { assignments[$0] == c }
                guard !members.isEmpty else { continue }
                var s = SIMD3<Float>.zero
                for m in members { s += positions[m] }
                centroids[c] = s / Float(members.count)
            }
        }

        // Merge small clusters into nearest neighbor
        for c in 0..<k {
            let memberCount = (0..<n).filter { assignments[$0] == c }.count
            guard memberCount > 0, memberCount < minClusterSize else { continue }
            var bestOther = -1
            var bestDist: Float = .greatestFiniteMagnitude
            for other in 0..<k where other != c {
                let otherCount = (0..<n).filter { assignments[$0] == other }.count
                guard otherCount >= minClusterSize else { continue }
                let d = simd_length(centroids[c] - centroids[other])
                if d < bestDist { bestDist = d; bestOther = other }
            }
            if bestOther >= 0 {
                for i in 0..<n where assignments[i] == c { assignments[i] = bestOther }
            }
        }

        let usedLabels = Set(assignments).sorted()
        return usedLabels.map { c in (0..<n).filter { assignments[$0] == c } }
    }
}
