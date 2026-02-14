import Lattice
import MCP
import Foundation

// MARK: - Consolidation

extension MemoryTools {

    // MARK: - find_clusters

    func handleFindClusters(_ args: [String: Value]?) async throws -> CallTool.Result {
        let a = try args.decode(FindClustersArgs.self)
        let minSize = a.minClusterSize?.value ?? 3
        let threshold = Double(a.distanceThreshold?.value ?? 30) / 100.0
        let maxClusters = a.maxClusters?.value ?? 10

        // Load all non-expired memories matching filters
        let now = Date()
        var baseQuery = lattice.objects(Memory.self).where { $0.expiresAt > now }
        if let project = a.project {
            baseQuery = baseQuery.where { $0.project == project }
        }
        if let topic = a.topic {
            baseQuery = baseQuery.where { $0.topic == topic }
        }
        let memories = baseQuery.snapshot()

        guard memories.count >= minSize else {
            return CallTool.Result(content: [.text("No clusters found. Only \(memories.count) memories match the filters (minimum cluster size: \(minSize)).")], isError: false)
        }

        // Index memories by ID for lookup
        let memoryMap: [Int64: Memory] = Dictionary(
            uniqueKeysWithValues: memories.compactMap { m in
                guard let pk = m.primaryKey, !m.embedding.isEmpty else { return nil }
                return (pk, m)
            }
        )
        let validIds = Set(memoryMap.keys)

        guard validIds.count >= minSize else {
            return CallTool.Result(content: [.text("No clusters found. Only \(validIds.count) memories have embeddings (minimum cluster size: \(minSize)).")], isError: false)
        }

        // Build neighbor map using Lattice vector search per memory.
        // For each memory, run .nearest() to find neighbors within the distance threshold.
        // This delegates distance computation to sqlite-vec instead of pulling embeddings into memory.
        var neighborMap: [Int64: [Int64]] = [:]
        var distanceCache: [Int64: [Int64: Double]] = [:]

        for (memId, mem) in memoryMap {
            let matches = baseQuery
                .nearest(to: mem.embedding, on: \.embedding, limit: memories.count, distance: .cosine)

            var neighbors: [Int64] = []
            var dists: [Int64: Double] = [:]
            for match in matches {
                let nId = match.object.primaryKey!
                guard nId != memId, validIds.contains(nId) else { continue }
                let dist = match.distances["embedding"] ?? 1.0
                if dist <= threshold {
                    neighbors.append(nId)
                    dists[nId] = dist
                }
            }
            neighborMap[memId] = neighbors
            distanceCache[memId] = dists
        }

        // Greedy clustering: pick memory with most unassigned neighbors as seed
        var assigned = Set<Int64>()
        var clusters: [[Int64]] = []

        while clusters.count < maxClusters {
            var bestSeed: Int64? = nil
            var bestCount = 0
            for memId in validIds where !assigned.contains(memId) {
                let unassigned = (neighborMap[memId] ?? []).filter { !assigned.contains($0) }.count
                if unassigned > bestCount {
                    bestCount = unassigned
                    bestSeed = memId
                }
            }

            guard let seed = bestSeed else { break }

            // Build cluster: seed + unassigned neighbors sorted by distance to seed
            var cluster = [seed]
            assigned.insert(seed)
            let neighbors = (neighborMap[seed] ?? []).filter { !assigned.contains($0) }
            let sorted = neighbors.sorted { a, b in
                (distanceCache[seed]?[a] ?? 1.0) < (distanceCache[seed]?[b] ?? 1.0)
            }
            for n in sorted where !assigned.contains(n) {
                cluster.append(n)
                assigned.insert(n)
            }

            if cluster.count >= minSize {
                clusters.append(cluster)
            } else {
                // Not enough for a cluster — unassign all except seed to avoid infinite loop
                for id in cluster where id != seed {
                    assigned.remove(id)
                }
            }
        }

        guard !clusters.isEmpty else {
            return CallTool.Result(content: [.text("No clusters found. Memories are too dissimilar at distance threshold \(Int(threshold * 100)) (try increasing distance_threshold).")], isError: false)
        }

        // Format output
        var output = "Found \(clusters.count) cluster(s):\n"
        for (i, cluster) in clusters.enumerated() {
            // Compute average pairwise similarity from cached distances
            var totalSim: Double = 0
            var pairCount = 0
            for a in cluster {
                for b in cluster where a < b {
                    if let dist = distanceCache[a]?[b] ?? distanceCache[b]?[a] {
                        totalSim += 1.0 - dist
                        pairCount += 1
                    }
                }
            }
            let avgSim = pairCount > 0 ? totalSim / Double(pairCount) : 0

            let topics = cluster.compactMap { memoryMap[$0]?.topic }
            let majorityTopic = Dictionary(grouping: topics, by: { $0 })
                .max(by: { $0.value.count < $1.value.count })?.key ?? "general"

            output += "\n## Cluster \(i + 1): \(majorityTopic) (\(cluster.count) memories)\n"
            output += "Avg. similarity: \(String(format: "%.2f", avgSim))\n\n"
            for memId in cluster {
                let content = memoryMap[memId]?.content ?? ""
                let preview = String(content.prefix(120))
                output += "[id:\(memId)] \(preview)\n"
            }
            let idList = cluster.map { String($0) }.joined(separator: ",")
            output += "\nSuggested action: consolidate --ids \(idList)\n"
        }

        return CallTool.Result(content: [.text(output)], isError: false)
    }

    // MARK: - consolidate

    func handleConsolidate(_ args: [String: Value]?) async throws -> CallTool.Result {
        let a = try args.decode(ConsolidateArgs.self)
        guard !a.content.isEmpty else {
            throw MCPError.invalidParams("'content' is required")
        }
        let ids = a.ids.map { Int64($0.value) }
        guard ids.count >= 2 else {
            throw MCPError.invalidParams("'ids' must contain at least 2 memory IDs to consolidate")
        }
        let importance = min(max(a.importance?.value ?? 3, 1), 5)

        // Fetch all memories by IDs
        var sources: [Memory] = []
        for id in ids {
            guard let mem = lattice.objects(Memory.self).where({ $0.primaryKey == id }).first else {
                return CallTool.Result(content: [.text("Memory with id \(id) not found.")], isError: true)
            }
            sources.append(mem)
        }

        // Determine project and topic
        let project = a.project ?? sources[0].project
        let topic = a.topic ?? {
            let grouped = Dictionary(grouping: sources, by: { $0.topic })
            return grouped.max(by: { $0.value.count < $1.value.count })?.key ?? sources[0].topic
        }()

        // Generate embedding for summary
        var embeddingVec = Vector<Float>([])
        if let floats = try await embedder.embed(text: a.content) {
            embeddingVec = Vector<Float>(floats)
        }

        // Create summary memory
        let summary = Memory(
            content: a.content,
            topic: topic,
            project: project,
            source: "consolidation",
            embedding: embeddingVec,
            importance: importance
        )
        lattice.add(summary)
        let summaryId = summary.primaryKey!

        // Deprioritize originals (importance → 0) and create summarized_by edges
        for source in sources {
            source.importance = 0
            let edge = Edge(sourceId: source.primaryKey!, targetId: summaryId, relation: "summarized_by")
            lattice.add(edge)
        }

        let preview = String(a.content.prefix(120))
        log("Consolidated \(ids.count) memories into [id:\(summaryId)]")
        return CallTool.Result(
            content: [.text("""
                Created summary [id:\(summaryId)] (project: \(project), topic: \(topic), importance: \(importance))
                Deprioritized \(sources.count) original memories (importance → 0)
                Created \(sources.count) 'summarized_by' edges

                Summary: \(preview)
                """)],
            isError: false
        )
    }
}
