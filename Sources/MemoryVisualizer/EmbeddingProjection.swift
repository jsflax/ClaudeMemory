import SwiftUI
import Lattice
import ClaudeMemoryLib

// MARK: - Data Types

enum LayoutMode: String, CaseIterable, Equatable {
    case forceDirected = "Force"
    case embedding = "Semantic"
}

enum ProjectionState: Equatable {
    case idle
    case loadingEmbeddings
    case computing(progress: Double)
    case ready
    case failed(String)
}

struct KnowledgeVoid: Identifiable {
    let id = UUID()
    let position: CGPoint
    let radius: CGFloat
    let nearestTopics: [String]
    let sparsity: CGFloat  // 0..1
}

struct SemanticCluster: Identifiable {
    let id = UUID()
    let nodeIds: [Int64]
    let centroid: CGPoint
    let hullPoints: [CGPoint]  // convex hull for drawing
    let label: String          // dominant topics/projects
    let projectBreakdown: [(project: String, count: Int)]
    var subClusters: [SemanticCluster] = []
}

// MARK: - Orchestrator

@Observable
@MainActor
final class EmbeddingProjection {
    private(set) var state: ProjectionState = .idle
    private(set) var projectedPositions: [Int64: CGPoint] = [:]
    private(set) var knowledgeVoids: [KnowledgeVoid] = []
    private(set) var semanticClusters: [SemanticCluster] = []

    /// Raw 384-dim embeddings, cached from Lattice
    private var embeddings: [Int64: [Float]] = [:]

    /// Track which node IDs were projected so we know when to re-project
    private var projectedNodeIds: Set<Int64> = []

    /// Target positions from the latest t-SNE emission — frame-level interpolation
    /// lerps projectedPositions toward these each frame for smooth animation.
    private var targetPositions: [Int64: CGPoint] = [:]

    /// Authoritative positions for cluster/void detection. Uses targetPositions (the true
    /// final t-SNE result) when available, so hulls are computed at the correct locations
    /// even while projectedPositions is still lerping toward them.
    private var detectionPositions: [Int64: CGPoint] {
        targetPositions.isEmpty ? projectedPositions : targetPositions
    }

    /// Whether the projection is stale (topology changed since last computation)
    var isStale: Bool {
        guard state == .ready else { return false }
        return false  // Staleness tracked externally via GraphView
    }

    // MARK: - Embedding Loading

    func loadEmbeddings(for nodeIds: Set<Int64>, from lattice: Lattice) {
        state = .loadingEmbeddings
        var loaded = 0
        for id in nodeIds {
            if embeddings[id] != nil { loaded += 1; continue }
            guard let memory = lattice.object(Memory.self, primaryKey: id) else { continue }
            let elements = memory.embedding.elements
            guard !elements.isEmpty else { continue }
            embeddings[id] = elements
            loaded += 1
        }
        // Prune stale entries
        let staleIds = Set(embeddings.keys).subtracting(nodeIds)
        for id in staleIds { embeddings.removeValue(forKey: id) }
    }

    // MARK: - Projection

    func computeProjection(
        nodeIds: Set<Int64>, center: CGPoint, spread: CGFloat,
        initialPositions: [Int64: CGPoint] = [:]
    ) async {
        // Collect embeddings for nodes that have them
        var ids: [Int64] = []
        var embeddingArrays: [[Float]] = []
        var noEmbeddingIds: [Int64] = []

        for id in nodeIds {
            if let emb = embeddings[id], !emb.isEmpty {
                ids.append(id)
                embeddingArrays.append(emb)
            } else {
                noEmbeddingIds.append(id)
            }
        }

        guard ids.count >= 2 else {
            state = .failed("Need at least 2 memories with embeddings")
            return
        }

        // Clear stale clusters/voids from previous run so they don't flash
        // while the new projection computes. They'll be re-detected after convergence.
        semanticClusters = []
        knowledgeVoids = []

        // Set initial projected positions (force layout) for progressive display
        if !initialPositions.isEmpty {
            projectedPositions = initialPositions
        }

        state = .computing(progress: 0)

        let perplexity = min(30.0, Double(ids.count - 1) / 3.0)

        // Build t-SNE initial positions from force layout (for smooth visual transition)
        let tsneInitialPositions: [(x: Double, y: Double)]?
        if !initialPositions.isEmpty {
            tsneInitialPositions = ids.map { id in
                let p = initialPositions[id] ?? CGPoint(x: center.x, y: center.y)
                return (x: Double(p.x), y: Double(p.y))
            }
        } else {
            tsneInitialPositions = nil
        }

        let input = TSNEKernel.Input(
            embeddings: embeddingArrays,
            ids: ids,
            perplexity: max(1.0, perplexity),
            maxIterations: 1000,
            initialPositions: tsneInitialPositions
        )

        let progressHandler: @Sendable (Double) -> Void = { [weak self] progress in
            Task { @MainActor [weak self] in
                self?.state = .computing(progress: progress)
            }
        }

        targetPositions = [:]

        // Intermediate position handler — updates targetPositions with per-emission tight bbox.
        // Frame-level interpolation in tickAnimation() smoothly lerps projectedPositions toward
        // these targets each frame, dampening any bbox-induced coordinate shifts between emissions.
        let capturedNoEmbeddingIds = noEmbeddingIds
        let capturedCenter = center
        let capturedSpread = spread
        let positionsHandler: @Sendable ([(id: Int64, x: Double, y: Double)]) -> Void = { [weak self] rawPositions in
            Task { @MainActor [weak self] in
                guard let self else { return }
                // Ignore stale emissions that arrive after computation finished
                guard case .computing = self.state else { return }
                self.targetPositions = Self.scaleToWorld(
                    rawPositions, center: capturedCenter, spread: capturedSpread,
                    noEmbeddingIds: capturedNoEmbeddingIds
                )
            }
        }

        let result = await Task.detached(priority: .userInitiated) {
            await TSNEKernel.compute(input, progress: progressHandler, onPositions: positionsHandler)
        }.value

        // Final scaling with the converged positions
        guard !result.positions.isEmpty else {
            state = .failed("t-SNE produced no positions")
            return
        }

        // Set final positions as the lerp target — tickAnimation() will smoothly
        // converge projectedPositions toward them. Never assign projectedPositions
        // directly to avoid a one-frame snap (the lerp is always lagging the target).
        targetPositions = Self.scaleToWorld(
            result.positions, center: center, spread: spread,
            noEmbeddingIds: noEmbeddingIds
        )
        projectedNodeIds = nodeIds
        state = .ready
    }

    /// Scale raw t-SNE positions to world coordinates.
    private static func scaleToWorld(
        _ rawPositions: [(id: Int64, x: Double, y: Double)],
        center: CGPoint, spread: CGFloat,
        noEmbeddingIds: [Int64]
    ) -> [Int64: CGPoint] {
        var minX = Double.greatestFiniteMagnitude, minY = Double.greatestFiniteMagnitude
        var maxX = -Double.greatestFiniteMagnitude, maxY = -Double.greatestFiniteMagnitude
        for p in rawPositions {
            minX = min(minX, p.x); minY = min(minY, p.y)
            maxX = max(maxX, p.x); maxY = max(maxY, p.y)
        }
        let rangeX = max(maxX - minX, 1)
        let rangeY = max(maxY - minY, 1)

        var positions: [Int64: CGPoint] = [:]
        for p in rawPositions {
            let nx = (p.x - minX) / rangeX - 0.5
            let ny = (p.y - minY) / rangeY - 0.5
            positions[p.id] = CGPoint(
                x: center.x + CGFloat(nx) * spread,
                y: center.y + CGFloat(ny) * spread
            )
        }

        if !noEmbeddingIds.isEmpty {
            let edgeRadius = spread * 0.6
            let angleStep = 2 * CGFloat.pi / CGFloat(noEmbeddingIds.count)
            for (i, id) in noEmbeddingIds.enumerated() {
                let angle = angleStep * CGFloat(i)
                positions[id] = CGPoint(
                    x: center.x + cos(angle) * edgeRadius,
                    y: center.y + sin(angle) * edgeRadius
                )
            }
        }

        return positions
    }

    // MARK: - Frame-Level Animation

    /// Lerps projectedPositions toward targetPositions each frame.
    /// Call at 60fps from the Canvas timer for smooth progressive animation.
    /// Automatically stops when positions have converged (max delta < 0.5px)
    /// to avoid unnecessary @Observable mutations that cause indefinite CPU load.
    func tickAnimation() {
        guard !targetPositions.isEmpty else { return }
        let alpha: CGFloat = 0.15  // 15% per frame at 60fps ≈ smooth ~100ms convergence
        var updated = projectedPositions
        var maxDelta: CGFloat = 0
        for (id, target) in targetPositions {
            let current = updated[id] ?? target
            let dx = target.x - current.x
            let dy = target.y - current.y
            maxDelta = max(maxDelta, abs(dx), abs(dy))
            updated[id] = CGPoint(x: current.x + dx * alpha, y: current.y + dy * alpha)
        }
        projectedPositions = updated

        // Once converged, snap to exact targets and stop ticking.
        // This prevents indefinite @Observable mutations at 60fps.
        if maxDelta < 0.5 {
            projectedPositions = targetPositions
            targetPositions = [:]
        }
    }

    // MARK: - Knowledge Void Detection

    /// Detect sparse regions in the projected 2D space between clusters.
    /// Uses nearest-node distance on a grid — a void is a region far from any node
    /// but surrounded by populated areas (between clusters, not at the edges).
    func detectVoids(nodeTopics: [Int64: String], nodeProjects: [Int64: String]) {
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
    func detectClusters(nodeTopics: [Int64: String], nodeProjects: [Int64: String], nodeLabels: [Int64: String]) {
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
    private func buildCluster(
        memberIndices: [Int], ids: [Int64], positions: [CGPoint],
        nodeTopics: [Int64: String], nodeProjects: [Int64: String], hullPadding: CGFloat
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

    // MARK: - Invalidation

    func invalidate() {
        state = .idle
        projectedPositions = [:]
        targetPositions = [:]
        knowledgeVoids = []
        semanticClusters = []
        projectedNodeIds = []
    }

    func markStale() {
        // Don't clear positions — keep showing old projection with a stale badge
        if state == .ready {
            // State stays .ready but GraphView will check topology version
        }
    }

    // MARK: - Helpers

    /// K-means clustering on 2D positions. Returns array of (clusterIndex, memberIndices).
    /// Uses farthest-first initialization (deterministic) and Lloyd's algorithm.
    private func kMeans(positions: [CGPoint], k: Int, minClusterSize: Int) -> [[Int]] {
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

    private func standardDeviation(_ values: [CGFloat]) -> CGFloat {
        let n = CGFloat(values.count)
        guard n > 1 else { return 1 }
        let mean = values.reduce(0, +) / n
        let variance = values.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / (n - 1)
        return sqrt(variance)
    }
}
