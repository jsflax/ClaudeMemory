import Lattice
import MCP
import Foundation

// MARK: - Clustering Algorithm

/// Compute semantic clusters from memories using embedding similarity + Jaccard term overlap.
/// Uses Lattice `.nearest()` for vector search and greedy seed-based clustering.
/// Returns clusters (arrays of memory primary keys) and pairwise distance cache.
public func findMemoryClusters(
    in lattice: Lattice,
    project: String? = nil,
    topic: String? = nil,
    distanceThreshold: Double = 0.15,
    jaccardThreshold: Double = 0.2,
    minClusterSize: Int = 2,
    maxClusters: Int = 10,
    neighborLimit: Int? = nil
) -> (clusters: [[Int64]], distances: [Int64: [Int64: Double]]) {
    let now = Date()
    var baseQuery = lattice.objects(Memory.self).where { $0.expiresAt > now }
    if let project { baseQuery = baseQuery.where { $0.project == project } }
    if let topic { baseQuery = baseQuery.where { $0.topic == topic } }
    let memories = baseQuery.snapshot()

    guard memories.count >= minClusterSize else { return ([], [:]) }

    let memoryMap: [Int64: Memory] = Dictionary(
        uniqueKeysWithValues: memories.compactMap { m in
            guard let pk = m.primaryKey, !m.embedding.isEmpty else { return nil }
            return (pk, m)
        }
    )
    let validIds = Set(memoryMap.keys)
    guard validIds.count >= minClusterSize else { return ([], [:]) }

    // Build neighbor map using Lattice vector search per memory.
    // For each memory, run .nearest() to find neighbors within the distance threshold
    // AND Jaccard term overlap >= threshold. This prevents same-project memories about different
    // subsystems from clustering together just because they share project vocabulary.
    var neighborMap: [Int64: [Int64]] = [:]
    var distanceCache: [Int64: [Int64: Double]] = [:]

    for (memId, mem) in memoryMap {
        let matches = baseQuery
            .nearest(to: mem.embedding, on: \.embedding, limit: neighborLimit ?? memories.count, distance: .cosine)

        var neighbors: [Int64] = []
        var dists: [Int64: Double] = [:]
        for match in matches {
            guard let nId = match.object.primaryKey else { continue }
            guard nId != memId, validIds.contains(nId) else { continue }
            let dist = match.distances["embedding"] ?? 1.0
            if dist <= distanceThreshold {
                let jaccard = jaccardSimilarity(mem.content, match.object.content)
                guard jaccard >= jaccardThreshold else { continue }
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

        if cluster.count >= minClusterSize {
            clusters.append(cluster)
        } else {
            for id in cluster where id != seed {
                assigned.remove(id)
            }
        }
    }

    return (clusters, distanceCache)
}

// MARK: - MCP Tool Handlers

extension MemoryTools {

    // MARK: - find_clusters

    func handleFindClusters(_ args: [String: Value]?) async throws -> CallTool.Result {
        let a = try args.decode(FindClustersArgs.self)
        let minSize = a.minClusterSize?.value ?? 3
        let threshold = Double(a.distanceThreshold?.value ?? 15) / 100.0
        let maxClusters = a.maxClusters?.value ?? 10

        let result = findMemoryClusters(
            in: lattice,
            project: a.project,
            topic: a.topic,
            distanceThreshold: threshold,
            jaccardThreshold: 0.2,
            minClusterSize: minSize,
            maxClusters: maxClusters
        )

        guard !result.clusters.isEmpty else {
            return CallTool.Result(content: [.text("No clusters found. Memories are too dissimilar at distance threshold \(Int(threshold * 100)) (try increasing distance_threshold).")], isError: false)
        }

        // Fetch memory data for formatting
        var memoryMap: [Int64: Memory] = [:]
        for id in Set(result.clusters.flatMap({ $0 })) {
            if let mem = lattice.objects(Memory.self).where({ $0.primaryKey == id }).first {
                memoryMap[id] = mem
            }
        }

        // Format output
        var output = "Found \(result.clusters.count) cluster(s):\n"
        for (i, cluster) in result.clusters.enumerated() {
            // Compute average pairwise similarity from cached distances
            var totalSim: Double = 0
            var pairCount = 0
            var distances: [Double] = []
            for a in cluster {
                for b in cluster where a < b {
                    if let dist = result.distances[a]?[b] ?? result.distances[b]?[a] {
                        totalSim += 1.0 - dist
                        pairCount += 1
                        distances.append(dist)
                    }
                }
            }
            let avgSim = pairCount > 0 ? totalSim / Double(pairCount) : 0

            let meanDist = distances.isEmpty ? 0.0 : distances.reduce(0, +) / Double(distances.count)
            let variance = distances.isEmpty ? 0.0 : distances.map { ($0 - meanDist) * ($0 - meanDist) }.reduce(0, +) / Double(distances.count)
            let stdev = sqrt(variance)

            let topics = cluster.compactMap { memoryMap[$0]?.topic }
            let uniqueTopics = Set(topics)
            let majorityTopic = Dictionary(grouping: topics, by: { $0 })
                .max(by: { $0.value.count < $1.value.count })?.key ?? "general"

            // Redundancy assessment
            let assessment: String
            if avgSim >= 0.95 && stdev <= 0.03 {
                assessment = "⚠️ Likely redundant — good candidates for consolidation"
            } else {
                assessment = "ℹ️ Topically similar but may cover distinct aspects — review content carefully before consolidating"
            }

            output += "\n## Cluster \(i + 1): \(majorityTopic) (\(cluster.count) memories, \(uniqueTopics.count) topic(s))\n"
            output += "Avg. similarity: \(String(format: "%.2f", avgSim)) | Distance stdev: \(String(format: "%.4f", stdev))\n"
            output += "\(assessment)\n\n"
            for memId in cluster {
                let content = memoryMap[memId]?.content ?? ""
                let preview = String(content.prefix(120))
                output += "[id:\(memId)] \(preview)\n"
            }
            let idList = cluster.map { String($0) }.joined(separator: ",")
            output += "\nIDs: \(idList)\n"
        }

        return CallTool.Result(content: [.text(output)], isError: false)
    }

    // MARK: - consolidate

    func handleConsolidate(_ args: [String: Value]?) async throws -> CallTool.Result {
        let a = try args.decode(ConsolidateArgs.self)
        guard !a.content.isEmpty else {
            throw MCPError.invalidParams("'content' is required")
        }
        let ids = a.ids.values.map { Int64($0) }
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
        guard let floats = try await embedder.embed(text: a.content) else {
            throw MCPError.internalError("Embedding model unavailable — cannot consolidate memories without a vector. Check that the CoreML model is bundled correctly.")
        }
        let embeddingVec = Vector<Float>(floats)

        // Create summary memory
        let consolidateTargetDB = writeLattice(for: project)
        let summary = Memory(
            content: a.content,
            topic: topic,
            project: project,
            source: "consolidation",
            embedding: embeddingVec,
            importance: importance
        )
        consolidateTargetDB.add(summary)

        guard let summaryId = summary.primaryKey else {
            throw MCPError.internalError("Failed to persist summary — primaryKey is nil after add()")
        }

        // Deprioritize originals (importance → 0) and create summarized_by edges
        guard let summaryGlobalId = summary.__globalId else {
            throw MCPError.internalError("Failed to get summary globalId after persist")
        }
        for source in sources {
            source.importance = 0
            guard let sourceGlobalId = source.__globalId else { continue }
            let edge = Edge(sourceGlobalId: sourceGlobalId, targetGlobalId: summaryGlobalId, relation: .summarizedBy)
            consolidateTargetDB.add(edge)
        }

        let preview = String(a.content.prefix(120))
        log("Consolidated \(ids.count) memories into [id:\(summaryId)]")
        incrementCrudCounter()
        resetMaintenanceBaseline()
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
