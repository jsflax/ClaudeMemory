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
    distanceThreshold: Double = 0.547,  // L2 equivalent of cosine 0.15
    jaccardThreshold: Double = 0.2,
    minClusterSize: Int = 2,
    maxClusters: Int = 10,
    neighborLimit: Int? = nil,
    excludeForeign: Bool = false,
    selfUserId: UUID? = nil
) -> (clusters: [[UUID]], distances: [UUID: [UUID: Double]]) {
    let now = Date()
    var baseQuery = lattice.objects(Memory.self).distinct(by: \.globalId).where { $0.expiresAt > now && $0.deletedAt == nil }
    if let project { baseQuery = baseQuery.where { $0.project == project } }
    if let topic { baseQuery = baseQuery.where { $0.topic == topic } }
    var memories = baseQuery.snapshot()
    // Maintenance guard: the maintenance subprocess must never receive
    // foreign-authored content in cluster listings (its prompts are a
    // Bash-capable injection target). Handlers pass their exclusion policy
    // through; nil-author rows are the user's own legacy rows.
    if excludeForeign {
        memories = memories.filter { $0.authorUserId == nil || $0.authorUserId == selfUserId }
    }

    guard memories.count >= minClusterSize else { return ([], [:]) }

    let memoryMap: [UUID: Memory] = Dictionary(
        uniqueKeysWithValues: memories.compactMap { m in
            guard let gid = m.globalId, !m.embedding.isEmpty else { return nil }
            return (gid, m)
        }
    )
    let validIds = Set(memoryMap.keys)
    guard validIds.count >= minClusterSize else { return ([], [:]) }

    // Build neighbor map using Lattice vector search per memory.
    // For each memory, run .nearest() to find neighbors within the distance threshold
    // AND Jaccard term overlap >= threshold. This prevents same-project memories about different
    // subsystems from clustering together just because they share project vocabulary.
    var neighborMap: [UUID: [UUID]] = [:]
    var distanceCache: [UUID: [UUID: Double]] = [:]

    for (memId, mem) in memoryMap {
        let matches = baseQuery
            .nearest(to: mem.embedding, on: \.embedding, limit: neighborLimit ?? memories.count, distance: .l2)

        var neighbors: [UUID] = []
        var dists: [UUID: Double] = [:]
        for match in matches {
            guard let nId = match.object.globalId else { continue }
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
    var assigned = Set<UUID>()
    var clusters: [[UUID]] = []

    while clusters.count < maxClusters {
        var bestSeed: UUID? = nil
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
        // Convert user-facing cosine percentage to L2 distance: L2 = sqrt(2 * cosine)
        let cosineThreshold = Double(a.distanceThreshold?.value ?? 15) / 100.0
        let threshold = sqrt(2.0 * cosineThreshold)
        let maxClusters = a.maxClusters?.value ?? 10

        let db = readLattice(for: a.project)
        let result = findMemoryClusters(
            in: db,
            project: a.project,
            topic: a.topic,
            distanceThreshold: threshold,
            jaccardThreshold: 0.2,
            minClusterSize: minSize,
            maxClusters: maxClusters,
            neighborLimit: 50,
            excludeForeign: excludeForeignAuthored,
            selfUserId: currentUserId
        )

        guard !result.clusters.isEmpty else {
            return CallTool.Result(content: [.text("No clusters found. Memories are too dissimilar at distance threshold \(Int(threshold * 100)) (try increasing distance_threshold).")], isError: false)
        }

        // Fetch memory data for formatting
        var memoryMap: [UUID: Memory] = [:]
        for gid in Set(result.clusters.flatMap({ $0 })) {
            if let mem = db.objects(Memory.self).where({ $0.globalId == gid }).first {
                memoryMap[gid] = mem
            }
        }

        // Format output
        var output = "Found \(result.clusters.count) cluster(s):\n"
        for (i, cluster) in result.clusters.enumerated() {
            // Compute average pairwise similarity from cached distances
            var totalSim: Double = 0
            var pairCount = 0
            var distances: [Double] = []
            for (ai, a) in cluster.enumerated() {
                for b in cluster[(ai + 1)...] {
                    if let dist = result.distances[a]?[b] ?? result.distances[b]?[a] {
                        // Convert L2 distance to cosine similarity for normalized vectors
                        totalSim += 1.0 - (dist * dist) / 2.0
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
            let selfId = currentUserId
            for memGid in cluster {
                let mem = memoryMap[memGid]
                let content = mem?.content ?? ""
                let preview = String(content.prefix(120))
                // Foreign-author label — the consolidation decision needs to
                // know whose memories a cluster contains (demoting them is
                // group-wide; the consolidate handler enforces force: true).
                let badge = (mem?.authorUserId != nil && mem?.authorUserId != selfId)
                    ? " [by:\(GroupDirectory.badgeName(for: mem?.authorUserId))]" : ""
                output += "[id:\(memGid.uuidString)]\(badge) \(preview)\n"
            }
            let idList = cluster.map { $0.uuidString }.joined(separator: ",")
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
        let ids = a.ids.values
        guard ids.count >= 2 else {
            throw MCPError.invalidParams("'ids' must contain at least 2 memory IDs to consolidate")
        }
        let importance = min(max(a.importance?.value ?? 3, 1), 5)

        // Fetch all memories by globalId
        var sources: [(memory: Memory, lattice: Lattice)] = []
        for gid in ids {
            guard let found = findMemory(id: gid) else {
                return CallTool.Result(content: [.text("Memory with id \(gid.uuidString) not found.")], isError: true)
            }
            sources.append(found)
        }

        // Hard backstop (not just agent-prompt guidance): consolidating a
        // cluster with FOREIGN-authored members demotes teammates' memories
        // group-wide. Require an explicit force acknowledgment. Tombstoned
        // members are never consolidated.
        if sources.contains(where: { $0.memory.deletedAt != nil }) {
            return CallTool.Result(
                content: [.text("One or more memories are tombstoned — undelete them first or drop them from the cluster.")],
                isError: true
            )
        }
        let selfId = currentUserId
        let foreignMembers = sources.filter { $0.memory.authorUserId != nil && $0.memory.authorUserId != selfId }
        if !foreignMembers.isEmpty && a.force != true {
            let names = foreignMembers
                .map { GroupDirectory.badgeName(for: $0.memory.authorUserId) }
            let uniqueNames = Array(Set(names)).sorted().joined(separator: ", ")
            return CallTool.Result(
                content: [.text("This cluster contains \(foreignMembers.count) memory(ies) written by teammates (\(uniqueNames)) — consolidating demotes THEIR memories for the whole group. Prefer connect(relation: \"relates_to\") to link without demoting, or pass force: true to proceed deliberately.")],
                isError: true
            )
        }

        // Determine project and topic
        let project = a.project ?? sources[0].memory.project
        let topic = a.topic ?? {
            let grouped = Dictionary(grouping: sources, by: { $0.memory.topic })
            return grouped.max(by: { $0.value.count < $1.value.count })?.key ?? sources[0].memory.topic
        }()

        // Generate embedding for summary
        guard let floats = try await embedder.embed(text: a.content) else {
            throw MCPError.internalError("Embedding model unavailable — cannot consolidate memories without a vector. Check that the CoreML model is bundled correctly.")
        }
        let embeddingVec = Vector<Float>(floats)

        // Create summary memory (authored by the runner)
        let summary = Memory(
            content: a.content,
            topic: topic,
            project: project,
            source: "consolidation",
            embedding: embeddingVec,
            importance: importance,
            authorUserId: currentUserId,
            modifiedAt: Date()
        )
        try localLattice.add(summary)

        guard let summaryGlobalId = summary.globalId else {
            throw MCPError.internalError("Failed to get summary globalId after persist")
        }

        // Deprioritize originals (importance → 0) and create summarized_by edges
        nonisolated(unsafe) let sourceMemories = sources.map(\.memory)
        let sourceGlobalIds = sourceMemories.compactMap(\.globalId)
        try localLattice.transaction {
            for mem in sourceMemories {
                mem.importance = 0
            }
            for sourceGlobalId in sourceGlobalIds {
                let edge = Edge(sourceGlobalId: sourceGlobalId, targetGlobalId: summaryGlobalId, relation: .summarizedBy, authorUserId: currentUserId)
                try localLattice.add(edge)
            }
        }

        let preview = String(a.content.prefix(120))
        log("Consolidated \(ids.count) memories into [id:\(summaryGlobalId.uuidString)]")
        return CallTool.Result(
            content: [.text("""
                Created summary [id:\(summaryGlobalId.uuidString)] (project: \(project), topic: \(topic), importance: \(importance))
                Deprioritized \(sources.count) original memories (importance → 0)
                Created \(sources.count) 'summarized_by' edges

                Summary: \(preview)
                """)],
            isError: false
        )
    }
}
