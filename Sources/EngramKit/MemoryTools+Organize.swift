import Lattice
import MCP
import Foundation

// MARK: - Organize & Community Detection

extension MemoryTools {

    // MARK: - detect_communities

    /// Uses label propagation on the knowledge graph to find natural communities.
    /// Read-only — shows detected groups for Claude to review.
    func handleDetectCommunities(_ args: [String: Value]?) async throws -> CallTool.Result {
        let a = try args.decode(DetectCommunitiesArgs.self)
        let project = a.project

        // Fetch all non-expired, non-episode memories for the project
        let allMemories = lattice.objects(Memory.self)
            .where { $0.project == project && $0.expiresAt > Date() && $0.topic != "episode" }
            .snapshot()

        guard allMemories.count >= 3 else {
            return CallTool.Result(
                content: [.text("Not enough memories in '\(project)' to detect communities (found \(allMemories.count), need at least 3).")],
                isError: false
            )
        }

        // Build memory globalId set and lookup
        let memoryGlobalIds = Set(allMemories.compactMap(\.__globalId))
        let memoryMapByGlobalId: [UUID: Memory] = Dictionary(
            uniqueKeysWithValues: allMemories.compactMap { m in
                guard let gid = m.__globalId else { return nil }
                return (gid, m)
            }
        )

        // Build undirected adjacency from all edges between these memories (keyed by globalId)
        var adjacency: [UUID: Set<UUID>] = [:]
        for gid in memoryGlobalIds { adjacency[gid] = [] }

        let edges = lattice.objects(Edge.self).snapshot()
        for edge in edges {
            guard memoryGlobalIds.contains(edge.sourceGlobalId) && memoryGlobalIds.contains(edge.targetGlobalId) else { continue }
            adjacency[edge.sourceGlobalId, default: []].insert(edge.targetGlobalId)
            adjacency[edge.targetGlobalId, default: []].insert(edge.sourceGlobalId)
        }

        // Run label propagation
        let communities = labelPropagation(adjacency: adjacency)

        // Filter to communities with 2+ members
        let minSize = a.minSize?.value ?? 2
        let significantCommunities = communities.filter { $0.count >= minSize }

        guard !significantCommunities.isEmpty else {
            return CallTool.Result(
                content: [.text("No communities with \(minSize)+ members found in '\(project)'. Memories may not be connected — use `connect` to create edges between related memories first.")],
                isError: false
            )
        }

        // Format output
        var output = "Found \(significantCommunities.count) community/communities in '\(project)':\n"

        for (i, community) in significantCommunities.enumerated() {
            let sorted = community.sorted(by: { $0.uuidString < $1.uuidString })
            output += "\n## Community \(i + 1) (\(sorted.count) memories)"
            for memGlobalId in sorted {
                let mem = memoryMapByGlobalId[memGlobalId]
                let topic = mem?.topic ?? "general"
                let content = mem?.content ?? ""
                let preview = String(content.prefix(150))
                let displayId = mem?.primaryKey.map(String.init) ?? "?"
                output += "\n  [id:\(displayId)] [\(topic)] \(preview)"
            }
            output += "\n"
        }

        // Show isolated nodes count
        let isolatedCount = memoryGlobalIds.count - significantCommunities.reduce(0) { $0 + $1.count }
        if isolatedCount > 0 {
            output += "\n(\(isolatedCount) memories not in any community — they have no edges or are in groups smaller than \(minSize))\n"
        }

        return CallTool.Result(content: [.text(output)], isError: false)
    }

    // MARK: - organize

    /// Simple action: takes memory IDs + label, updates their topics, creates hub, links via part_of.
    func handleOrganize(_ args: [String: Value]?) async throws -> CallTool.Result {
        let a = try args.decode(OrganizeArgs.self)
        let ids = a.ids.values.map { Int64($0) }
        let label = a.label

        guard !ids.isEmpty else {
            return CallTool.Result(content: [.text("No memory IDs provided.")], isError: true)
        }
        guard !label.isEmpty else {
            return CallTool.Result(content: [.text("Label cannot be empty.")], isError: true)
        }

        // Verify memories exist and determine project
        var memories: [Int64: Memory] = [:]
        for id in ids {
            guard let mem = lattice.objects(Memory.self).where({ $0.primaryKey == Int64(id) }).first else {
                return CallTool.Result(content: [.text("Memory [id:\(id)] not found.")], isError: true)
            }
            memories[Int64(id)] = mem
        }

        let project = a.project ?? memories[Int64(ids[0])]?.project ?? "global"

        // Create hub memory
        let hubContent = a.summary ?? "Hub: \(label)"
        var hubEmbedding = Vector<Float>([])
        if let floats = try await embedder.embed(text: hubContent) {
            hubEmbedding = Vector<Float>(floats)
        }
        let hub = Memory(
            content: hubContent,
            topic: label,
            project: project,
            source: "organize",
            embedding: hubEmbedding
        )
        localLattice.add(hub)

        guard let hubId = hub.primaryKey else {
            return CallTool.Result(content: [.text("Failed to create hub memory.")], isError: true)
        }

        // Link each memory to hub and update its topic
        guard let hubGlobalId = hub.__globalId else {
            return CallTool.Result(content: [.text("Failed to get hub globalId.")], isError: true)
        }
        for id in ids {
            if let mem = memories[Int64(id)], let memGid = mem.__globalId {
                let edge = Edge(sourceGlobalId: memGid, targetGlobalId: hubGlobalId, relation: .partOf)
                localLattice.add(edge)
                mem.topic = label
            }
        }

        log("Organized \(ids.count) memories under '\(label)' → hub [id:\(hubId)]")

        var output = "Organized \(ids.count) memories under '\(label)':\n"
        output += "  Hub: [id:\(hubId)]\n"
        output += "  Topic updated to '\(label)' on all \(ids.count) memories\n"
        output += "  part_of edges created from each memory → hub"

        return CallTool.Result(content: [.text(output)], isError: false)
    }
}

// MARK: - Label Propagation

/// Detect communities in an undirected graph via label propagation.
/// Each node starts with its own label, then iteratively adopts the most common
/// label among its neighbors. Converges when no labels change.
/// Returns communities as arrays of node globalIds (sorted by size descending).
public func labelPropagation(adjacency: [UUID: Set<UUID>], maxIterations: Int = 10) -> [[UUID]] {
    var labels: [UUID: UUID] = [:]
    for id in adjacency.keys {
        labels[id] = id
    }

    for _ in 0..<maxIterations {
        var newLabels = labels
        var changed = false

        for id in adjacency.keys {
            guard let neighbors = adjacency[id], !neighbors.isEmpty else { continue }

            var labelCounts: [UUID: Int] = [:]
            for neighbor in neighbors {
                guard let neighborLabel = labels[neighbor] else { continue }
                labelCounts[neighborLabel, default: 0] += 1
            }

            guard let bestLabel = labelCounts.max(by: {
                $0.value != $1.value ? $0.value < $1.value : $0.key.uuidString > $1.key.uuidString
            })?.key else { continue }

            if newLabels[id] != bestLabel {
                newLabels[id] = bestLabel
                changed = true
            }
        }

        labels = newLabels
        if !changed { break }
    }

    var communities: [UUID: [UUID]] = [:]
    for (id, label) in labels {
        communities[label, default: []].append(id)
    }

    return communities.values
        .sorted { $0.count > $1.count }
}
