import Lattice
import MCP
import Foundation

// MARK: - CRUD + Search + Timeline

extension MemoryTools {

    // MARK: - remember

    func handleRemember(_ args: [String: Value]?) async throws -> CallTool.Result {
        let a = try args.decode(RememberArgs.self)
        guard !a.content.isEmpty else {
            throw MCPError.invalidParams("'content' is required")
        }
        let content = a.content
        let topic = a.topic ?? "general"
        let project = a.project ?? "global"
        let source = a.source ?? ""
        let expiresAt: Date
        if let days = a.expiresInDays?.value, days > 0 {
            expiresAt = Date().addingTimeInterval(Double(days) * 86400)
        } else {
            expiresAt = .distantFuture
        }
        let importance: Int
        if let imp = a.importance?.value {
            guard (1...5).contains(imp) else {
                throw MCPError.invalidParams("'importance' must be between 1 and 5, got \(imp)")
            }
            importance = imp
        } else {
            importance = 0
        }

        var embeddingVec = Vector<Float>([])
        if let floats = try await embedder.embed(text: content) {
            embeddingVec = Vector<Float>(floats)
        }

        // Conflict detection: check for near-duplicates before storing
        // Requires BOTH close embedding distance AND high term overlap (Jaccard >= 0.4)
        // to avoid false positives from topically similar but informationally distinct memories
        if a.force != true && !embeddingVec.isEmpty {
            let candidates = lattice.objects(Memory.self)
                .where { $0.expiresAt > Date() }
                .where { $0.project == project || $0.project == "global" }
                .nearest(to: embeddingVec, on: \.embedding, limit: 5, distance: .cosine)

            let conflicts = candidates.filter { match in
                let sameProject = match.object.project == project
                let threshold = sameProject ? 0.12 : 0.05
                guard match.distance < threshold else { return false }
                return jaccardSimilarity(content, match.object.content) >= 0.4
            }
            if !conflicts.isEmpty {
                var warning = "⚠️ Near-duplicate memory detected. The new memory was NOT stored.\n\nExisting similar memories:"
                for match in conflicts {
                    let m = match.object
                    let dist = String(format: "%.3f", match.distance)
                    let jaccard = String(format: "%.0f%%", jaccardSimilarity(content, m.content) * 100)
                    warning += "\n  [id:\(m.primaryKey!)] (distance: \(dist), term overlap: \(jaccard)) \(m.content)"
                }
                warning += "\n\nTo resolve:"
                warning += "\n  - Use `update(id: N, ...)` to modify the existing memory"
                warning += "\n  - Use `remember(..., force: true)` to keep both"
                warning += "\n  - Use `forget(id: N)` to remove the old one, then `remember` the new one"
                log("Conflict detected for: \(content.prefix(80))")
                return CallTool.Result(content: [.text(warning)], isError: false)
            }
        }

        // Auto-episode: create or reuse episode for this session
        let episodeId: Int64
        if let active = activeEpisodeId {
            let gap = Date().timeIntervalSince(lastMemoryTime)
            if gap > 1800 { // 30 minutes
                endActiveEpisode()
                episodeId = createAutoEpisode(project: project)
            } else if !isExplicitEpisode, let ep = lattice.objects(Episode.self).where({ $0.primaryKey == active }).first, ep.project != project {
                // Auto-episode project mismatch — start a new one for this project
                endActiveEpisode()
                episodeId = createAutoEpisode(project: project)
            } else {
                episodeId = active
            }
        } else {
            episodeId = createAutoEpisode(project: project)
        }
        lastMemoryTime = Date()

        let memory = Memory(content: content, topic: topic, project: project, source: source, embedding: embeddingVec, expiresAt: expiresAt, importance: importance, episodeId: episodeId)
        lattice.add(memory)

        // Auto-create part_of edge when parent_id is provided
        var parentNote = ""
        if let parentId = a.parentId?.value {
            let pid = Int64(parentId)
            guard lattice.objects(Memory.self).where({ $0.primaryKey == pid }).first != nil else {
                throw MCPError.invalidParams("parent_id \(parentId) not found")
            }
            let edge = Edge(sourceId: memory.primaryKey!, targetId: pid, relation: "part_of")
            lattice.add(edge)
            parentNote = ", parent: \(parentId)"
            log("Auto-created part_of edge: \(memory.primaryKey!) -> \(parentId)")
        }

        let expiresNote = expiresAt == .distantFuture ? "" : ", expires: \(Self.dateFormatter.string(from: expiresAt))"
        let importanceNote = importance > 0 ? ", importance: \(importance)" : ""
        log("Stored memory [\(project)/\(topic)]: \(content.prefix(80))")

        var response = "Stored memory (id: \(memory.primaryKey!), project: \(project), topic: \(topic)\(parentNote)\(expiresNote)\(importanceNote)): \(content.prefix(100))\(content.count > 100 ? "..." : "")"

        // Nudge toward atomic memories when content is complex
        let lines = content.components(separatedBy: "\n")
        let headers = lines.filter { $0.hasPrefix("## ") || $0.hasPrefix("### ") }
        if content.count > 1000 && headers.count >= 2 {
            let names = headers.map {
                $0.drop(while: { $0 == "#" || $0 == " " })
            }
            response += "\n\n💡 This memory has \(headers.count) sections (\(names.joined(separator: ", "))). "
            response += "Consider storing each section as a child memory with `parent_id: \(memory.primaryKey!)` for more precise recall."
        }

        return CallTool.Result(
            content: [.text(response)],
            isError: false
        )
    }

    // MARK: - recall

    func handleRecall(_ args: [String: Value]?) async throws -> CallTool.Result {
        let a = try args.decode(RecallArgs.self)
        guard !a.query.isEmpty else {
            throw MCPError.invalidParams("'query' is required")
        }
        let query = a.query
        let projectFilter = a.project
        let topicFilter = a.topic
        let limit = a.limit?.value ?? 10
        let depth = min(a.depth?.value ?? 0, 3)
        let hasTemporalFilter = a.since != nil || a.before != nil

        // Build base query — always filter out expired memories
        var results = lattice.objects(Memory.self)
            .where { $0.expiresAt > Date() }

        // Topic is still a hard filter (it's a narrow constraint)
        if let topicFilter {
            results = results.where { $0.topic == topicFilter }
        }

        // Temporal filters on createdAt
        if let sinceStr = a.since {
            guard let sinceDate = parseTemporalDate(sinceStr) else {
                throw MCPError.invalidParams("Could not parse 'since' date: \(sinceStr)")
            }
            results = results.where { $0.createdAt >= sinceDate }
        }
        if let beforeStr = a.before {
            guard let beforeDate = parseTemporalDate(beforeStr) else {
                throw MCPError.invalidParams("Could not parse 'before' date: \(beforeStr)")
            }
            results = results.where { $0.createdAt <= beforeDate }
        }

        // FTS5 query: use anyOf so any matching term qualifies
        let ftsQuery: TextQuery = ._anyOf(query.split(separator: " ").map(String.init))

        // Semantic search with vector similarity
        if let queryEmbedding = try await embedder.embed(text: query) {
            // Soft project boost: fetch wider net, then re-rank
            let fetchLimit = projectFilter != nil ? limit * 3 : limit
            let embedding = Vector<Float>(queryEmbedding)

            // Hybrid: FTS5 (any term) intersected with vector similarity
            let nearest = results
                .matching(ftsQuery, on: \.content)
                .nearest(to: embedding, on: \.embedding, limit: fetchLimit, distance: .cosine)

            if nearest.isEmpty {
                return CallTool.Result(content: [.text("No memories found.")], isError: false)
            }

            // Apply soft project boosting and reinforcement scoring on distances
            let now = Date()
            let boosted: [(object: Memory, distance: Double, ftsRank: Double?)] = nearest.map { match in
                let m = match.object
                let cosine = match.distances["embedding"] ?? match.distance

                // Project boost
                let projectBoost: Double
                if let projectFilter {
                    if m.project == projectFilter {
                        projectBoost = 0.7   // same-project: strong boost
                    } else if m.project == "global" {
                        projectBoost = 0.85  // global: moderate boost
                    } else {
                        projectBoost = 1.0   // other-project: no boost
                    }
                } else {
                    projectBoost = 1.0       // no project filter: no boost
                }

                // Frequency boost — log-scaled accessCount (caps at 15% reduction)
                let frequencyBoost = 1.0 - min(log2(1.0 + Double(m.accessCount)) * 0.04, 0.15)

                // Importance boost — explicit 1-5 rating (caps at 20% reduction)
                let importanceBoost = m.importance > 0 ? 1.0 - Double(m.importance - 1) * 0.05 : 1.0

                // Recency boost — exponential decay from lastAccessedAt (10% max for just-accessed)
                let daysSinceAccess = now.timeIntervalSince(m.lastAccessedAt) / 86400.0
                let recencyBoost = 1.0 - 0.1 * exp(-daysSinceAccess / 30.0)

                let distance = cosine * projectBoost * frequencyBoost * importanceBoost * recencyBoost
                return (object: m, distance: distance, ftsRank: match.distances["content"])
            }

            // Re-sort by boosted distance and take top `limit`
            let sorted = boosted.sorted { $0.distance < $1.distance }
            let topResults = Array(sorted.prefix(limit))

            if topResults.isEmpty {
                return CallTool.Result(content: [.text("No memories found.")], isError: false)
            }

            // Filter out outliers: use adaptive threshold based on the result cluster
            let distances = topResults.map(\.distance)
            let p75 = distances[distances.count * 3 / 4]
            let threshold = p75 * 1.2
            let filtered = topResults.filter { $0.distance <= threshold }

            // Bump lastAccessedAt and accessCount on recalled memories
            let accessNow = Date()
            for match in filtered {
                match.object.lastAccessedAt = accessNow
                match.object.accessCount += 1
            }

            let lines = filtered.map { match in
                let m = match.object
                let dist = String(format: "%.3f", match.distance)
                let ftsInfo = match.ftsRank.map { ", fts5: \(String(format: "%.3f", $0))" } ?? ""
                let impInfo = m.importance > 0 ? ", importance: \(m.importance)" : ""
                let expires = m.expiresAt == .distantFuture ? "" : ", expires: \(Self.dateFormatter.string(from: m.expiresAt))"
                let created = hasTemporalFilter ? ", created: \(Self.dateFormatter.string(from: m.createdAt))" : ""
                return "[id:\(m.primaryKey!)] [\(m.project)/\(m.topic)] (relevance: \(dist)\(ftsInfo)\(impInfo)\(expires)\(created)) \(m.content)"
            }

            var output = lines.joined(separator: "\n\n")

            // Graph traversal when depth > 0
            if depth > 0 {
                let recalledIds = Set(filtered.map { $0.object.primaryKey! })
                let connected = traverseGraph(from: recalledIds, depth: depth, excludeIds: recalledIds)
                if !connected.isEmpty {
                    output += "\n\n--- Connected (graph traversal, depth: \(depth)) ---"
                    let connNow = Date()
                    // Track all known IDs (recalled + connected so far) for edge lookup at depth>1
                    var knownIds = recalledIds
                    for mem in connected {
                        mem.lastAccessedAt = connNow
                        mem.accessCount += 1

                        let expires = mem.expiresAt == .distantFuture ? "" : ", expires: \(Self.dateFormatter.string(from: mem.expiresAt))"
                        let memId = mem.primaryKey!

                        // Look up the edge relation connecting this memory to any known memory
                        var edgeInfo = ""
                        if let edge = lattice.objects(Edge.self).where({ $0.targetId == memId }).first(where: { knownIds.contains($0.sourceId) }) {
                            edgeInfo = " <--[\(edge.relation)]-- [id:\(edge.sourceId)]"
                        } else if let edge = lattice.objects(Edge.self).where({ $0.sourceId == memId }).first(where: { knownIds.contains($0.targetId) }) {
                            edgeInfo = " --[\(edge.relation)]--> [id:\(edge.targetId)]"
                        }
                        knownIds.insert(memId)

                        // Small memories shown in full; large ones get a compact preview
                        if mem.content.count <= 500 {
                            output += "\n\n[id:\(memId)] [\(mem.project)/\(mem.topic)]\(expires)\(edgeInfo) \(mem.content)"
                        } else {
                            let firstLine = mem.content.split(separator: "\n", maxSplits: 1).first.map(String.init) ?? mem.content
                            let preview = String(firstLine.prefix(120))
                            let charCount = mem.content.count
                            let sectionCount = mem.content.components(separatedBy: "\n").filter { $0.hasPrefix("## ") || $0.hasPrefix("### ") }.count
                            let sizeInfo = sectionCount > 0 ? "\(sectionCount) sections, \(charCount) chars" : "\(charCount) chars"
                            output += "\n\n[id:\(memId)] [\(mem.project)/\(mem.topic)] (\(sizeInfo)\(expires))\(edgeInfo) \(preview)\(charCount > 120 ? "..." : "")"
                        }
                    }
                }
            }

            return CallTool.Result(content: [.text(output)], isError: false)
        } else {
            // Degraded mode: FTS5 full-text search (no embedding model loaded)
            if let projectFilter {
                results = results.where { $0.project == projectFilter || $0.project == "global" }
            }
            let ftsResults = results.matching(ftsQuery, on: \.content, limit: limit)

            var lines: [String] = []
            for match in ftsResults {
                let m = match.object
                let ftsInfo = match.distances["content"].map { " (fts5: \(String(format: "%.3f", $0)))" } ?? ""
                let expires = m.expiresAt == .distantFuture ? "" : ", expires: \(Self.dateFormatter.string(from: m.expiresAt))"
                let created = hasTemporalFilter ? ", created: \(Self.dateFormatter.string(from: m.createdAt))" : ""
                lines.append("[id:\(m.primaryKey!)] [\(m.project)/\(m.topic)]\(ftsInfo)\(expires)\(created) \(m.content)")
            }
            if lines.isEmpty {
                return CallTool.Result(content: [.text("No memories found.")], isError: false)
            }
            return CallTool.Result(content: [.text(lines.joined(separator: "\n\n"))], isError: false)
        }
    }

    // MARK: - forget

    func handleForget(_ args: [String: Value]?) throws -> CallTool.Result {
        let a = try args.decode(ForgetArgs.self)

        // Delete by ID takes priority
        if let id = a.id?.value {
            let id64 = Int64(id)
            let matches = lattice.objects(Memory.self).where { $0.primaryKey == id64 }
            guard let mem = matches.first else {
                return CallTool.Result(content: [.text("Memory with id \(id) not found.")], isError: true)
            }
            let summary = mem.content.prefix(80)

            // Cascade: delete edges referencing this memory
            let edgeCount = deleteEdgesForMemories([id64])

            lattice.delete(Memory.self, where: { $0.primaryKey == id64 })
            let edgeNote = edgeCount > 0 ? " Removed \(edgeCount) edge(s)." : ""
            log("Deleted memory [id:\(id)]: \(summary)")
            return CallTool.Result(
                content: [.text("Deleted memory (id: \(id), project: \(mem.project), topic: \(mem.topic)): \(summary)\(edgeNote)")],
                isError: false
            )
        }

        switch (a.topic, a.project) {
        case let (topic?, project?):
            let query = lattice.objects(Memory.self).where { $0.topic == topic && $0.project == project }
            let memoryIds = query.compactMap(\.primaryKey)
            let edgeCount = deleteEdgesForMemories(memoryIds)
            let count = memoryIds.count
            lattice.delete(Memory.self, where: { $0.topic == topic && $0.project == project })
            let edgeNote = edgeCount > 0 ? " Removed \(edgeCount) edge(s)." : ""
            return CallTool.Result(
                content: [.text("Deleted \(count) memories (project: \(project), topic: \(topic)).\(edgeNote)")],
                isError: false
            )
        case let (topic?, nil):
            let query = lattice.objects(Memory.self).where { $0.topic == topic }
            let memoryIds = query.compactMap(\.primaryKey)
            let edgeCount = deleteEdgesForMemories(memoryIds)
            let count = memoryIds.count
            lattice.delete(Memory.self, where: { $0.topic == topic })
            let edgeNote = edgeCount > 0 ? " Removed \(edgeCount) edge(s)." : ""
            return CallTool.Result(
                content: [.text("Deleted \(count) memories with topic '\(topic)'.\(edgeNote)")],
                isError: false
            )
        case let (nil, project?):
            let query = lattice.objects(Memory.self).where { $0.project == project }
            let memoryIds = query.compactMap(\.primaryKey)
            let edgeCount = deleteEdgesForMemories(memoryIds)
            let count = memoryIds.count
            lattice.delete(Memory.self, where: { $0.project == project })
            let edgeNote = edgeCount > 0 ? " Removed \(edgeCount) edge(s)." : ""
            return CallTool.Result(
                content: [.text("Deleted \(count) memories for project '\(project)'.\(edgeNote)")],
                isError: false
            )
        case (nil, nil):
            throw MCPError.invalidParams("Specify 'id', 'topic', or 'project'. Refusing to delete all memories without an explicit filter.")
        }
    }

    // MARK: - update

    func handleUpdate(_ args: [String: Value]?) async throws -> CallTool.Result {
        let a = try args.decode(UpdateArgs.self)

        // 1. Validate targeting — at least id or query required
        guard a.id != nil || (a.query != nil && !a.query!.isEmpty) else {
            throw MCPError.invalidParams("Provide 'id' or 'query' to target a memory.")
        }

        // 2. Validate at least one edit field
        let hasContentEdit = a.content != nil || a.append != nil || a.prepend != nil || a.find != nil
        let hasMetadataEdit = a.topic != nil || a.source != nil || a.expiresInDays != nil || a.importance != nil
        guard hasContentEdit || hasMetadataEdit else {
            throw MCPError.invalidParams("Provide at least one edit: content, append, prepend, find+replace, topic, source, or expires_in_days.")
        }

        // 3. Validate content edit modes are mutually exclusive
        let contentModes = [a.content != nil, a.append != nil, a.prepend != nil, a.find != nil]
        if contentModes.filter({ $0 }).count > 1 {
            throw MCPError.invalidParams("Content edit modes are mutually exclusive: use only one of content, append, prepend, or find+replace.")
        }

        // 4. find requires replace (replace can be empty string)
        if a.find != nil && a.replace == nil {
            throw MCPError.invalidParams("'find' requires 'replace' (can be empty string to delete matched text).")
        }

        // 5. Locate memory
        let mem: Memory
        if let id = a.id?.value {
            let id64 = Int64(id)
            let matches = lattice.objects(Memory.self).where { $0.primaryKey == id64 }
            guard let found = matches.first else {
                return CallTool.Result(content: [.text("Memory with id \(id) not found.")], isError: true)
            }
            mem = found
        } else {
            let query = a.query!
            var results = lattice.objects(Memory.self)
            if let projectFilter = a.project {
                results = results.where { $0.project == projectFilter }
            }
            guard let queryEmbedding = try await embedder.embed(text: query) else {
                throw MCPError.internalError("Failed to generate embedding for query")
            }
            let nearest = results
                .nearest(to: Vector<Float>(queryEmbedding), on: \.embedding, limit: 1, distance: .cosine)
            guard let match = nearest.first else {
                return CallTool.Result(content: [.text("No matching memory found to update.")], isError: false)
            }
            mem = match.object
        }

        // 6. Apply content edits
        let oldContent = mem.content
        var contentChanged = false

        if let content = a.content {
            mem.content = content
            contentChanged = true
        } else if let append = a.append {
            mem.content += "\n" + append
            contentChanged = true
        } else if let prepend = a.prepend {
            mem.content = prepend + "\n" + mem.content
            contentChanged = true
        } else if let find = a.find {
            let replace = a.replace!
            guard mem.content.contains(find) else {
                return CallTool.Result(
                    content: [.text("Find pattern not found in memory content.\nPattern: \(find)\nContent: \(mem.content)")],
                    isError: true
                )
            }
            mem.content = mem.content.replacingOccurrences(of: find, with: replace)
            contentChanged = true
        }

        // 7. Apply metadata edits
        var changes: [String] = []
        if contentChanged {
            changes.append("content: \(oldContent.prefix(60))... → \(mem.content.prefix(60))...")
        }

        if let topic = a.topic {
            let old = mem.topic
            mem.topic = topic
            changes.append("topic: \(old) → \(topic)")
        }
        if let source = a.source {
            let old = mem.source
            mem.source = source
            changes.append("source: \(old) → \(source)")
        }
        if let days = a.expiresInDays?.value {
            let oldExpires = mem.expiresAt == .distantFuture ? "permanent" : Self.dateFormatter.string(from: mem.expiresAt)
            if days == 0 {
                mem.expiresAt = .distantFuture
                changes.append("expires: \(oldExpires) → permanent")
            } else {
                mem.expiresAt = Date().addingTimeInterval(Double(days) * 86400)
                changes.append("expires: \(oldExpires) → \(Self.dateFormatter.string(from: mem.expiresAt))")
            }
        }
        if let imp = a.importance?.value {
            guard (0...5).contains(imp) else {
                throw MCPError.invalidParams("'importance' must be between 0 and 5, got \(imp)")
            }
            let old = mem.importance
            mem.importance = imp
            changes.append("importance: \(old) → \(imp)")
        }

        // 8. Re-embed only if content changed
        if contentChanged {
            if let newEmbedding = try await embedder.embed(text: mem.content) {
                mem.embedding = Vector<Float>(newEmbedding)
            }
        }

        // 9. Update lastAccessedAt
        mem.lastAccessedAt = Date()

        log("Updated memory [id:\(mem.primaryKey!)] [\(mem.project)/\(mem.topic)]: \(changes.joined(separator: ", "))")
        return CallTool.Result(
            content: [.text("Updated memory (id: \(mem.primaryKey!), project: \(mem.project), topic: \(mem.topic)).\nChanges:\n\(changes.joined(separator: "\n"))")],
            isError: false
        )
    }

    // MARK: - merge

    func handleMerge(_ args: [String: Value]?) async throws -> CallTool.Result {
        let a = try args.decode(MergeArgs.self)
        guard !a.content.isEmpty else {
            throw MCPError.invalidParams("'content' is required")
        }
        let ids = a.ids.values.map { Int64($0) }
        guard ids.count >= 2 else {
            throw MCPError.invalidParams("'ids' must contain at least 2 memory IDs to merge")
        }
        let content = a.content

        // Fetch the source memories
        var sources: [Memory] = []
        for id in ids {
            let matches = lattice.objects(Memory.self).where { $0.primaryKey == id }
            guard let mem = matches.first else {
                return CallTool.Result(content: [.text("Memory with id \(id) not found.")], isError: true)
            }
            sources.append(mem)
        }

        // Use first source for defaults
        let topic = a.topic ?? sources[0].topic
        let project = a.project ?? sources[0].project

        // Embed the merged content
        var embeddingVec = Vector<Float>([])
        if let floats = try await embedder.embed(text: content) {
            embeddingVec = Vector<Float>(floats)
        }

        // Create merged memory
        let merged = Memory(content: content, topic: topic, project: project, source: "merged", embedding: embeddingVec)
        lattice.add(merged)

        // Collect old content summaries before deleting
        let oldSummaries = sources.map { "[id:\($0.primaryKey!)] \($0.content.prefix(60))" }

        // Clean up edges referencing source memories
        let edgeCount = deleteEdgesForMemories(ids)

        // Delete originals
        for id in ids {
            lattice.delete(Memory.self, where: { $0.primaryKey == id })
        }

        let edgeNote = edgeCount > 0 ? " Removed \(edgeCount) edge(s) from source memories." : ""
        log("Merged \(ids.count) memories into [id:\(merged.primaryKey!)]")
        return CallTool.Result(
            content: [.text("Merged \(ids.count) memories into new memory (id: \(merged.primaryKey!), project: \(project), topic: \(topic)).\(edgeNote)\n\nDeleted:\n\(oldSummaries.joined(separator: "\n"))\n\nNew:\n\(content)")],
            isError: false
        )
    }

    // MARK: - stats

    func handleStats(_ args: [String: Value]?) throws -> CallTool.Result {
        let a = try args.decode(StatsArgs.self)
        let projectFilter = a.project

        var base = lattice.objects(Memory.self)
        if let projectFilter {
            base = base.where { $0.project == projectFilter }
        }

        let total = base.count
        if total == 0 {
            return CallTool.Result(content: [.text("No memories stored.")], isError: false)
        }

        var lines: [String] = []
        lines.append("Total memories: \(total)")

        // Per-project breakdown
        if projectFilter == nil {
            lines.append("\nBy project:")
            let grouped = base.group(by: \.project)
            var projectLines: [String] = []
            for mem in grouped {
                let count = lattice.count(Memory.self, where: { $0.project == mem.project })
                projectLines.append("  \(mem.project): \(count)")
            }
            projectLines.sort()
            lines.append(contentsOf: projectLines)
        }

        // Per-topic breakdown
        lines.append("\nBy topic:")
        let topicGrouped = base.group(by: \.topic)
        var topicLines: [String] = []
        for mem in topicGrouped {
            var countQuery = lattice.objects(Memory.self).where { $0.topic == mem.topic }
            if let projectFilter {
                countQuery = countQuery.where { $0.project == projectFilter }
            }
            topicLines.append("  \(mem.topic): \(countQuery.count)")
        }
        topicLines.sort()
        lines.append(contentsOf: topicLines)

        return CallTool.Result(content: [.text(lines.joined(separator: "\n"))], isError: false)
    }

    // MARK: - list_topics

    func handleListTopics(_ args: [String: Value]?) throws -> CallTool.Result {
        let a = try args.decode(ListTopicsArgs.self)
        let projectFilter = a.project

        var base = lattice.objects(Memory.self)
        if let projectFilter {
            base = base.where { $0.project == projectFilter }
        }

        let grouped = base.group(by: \.topic)
        if grouped.endIndex == 0 {
            return CallTool.Result(content: [.text("No memories stored.")], isError: false)
        }

        var lines: [String] = []
        for memory in grouped {
            var countQuery = lattice.objects(Memory.self).where { $0.topic == memory.topic }
            if let projectFilter {
                countQuery = countQuery.where { $0.project == projectFilter }
            }
            let count = countQuery.count
            lines.append("\(memory.topic): \(count) memories")
        }
        lines.sort()

        let header = projectFilter.map { "Topics for project '\($0)':" } ?? "All topics:"
        return CallTool.Result(
            content: [.text(header + "\n" + lines.joined(separator: "\n"))],
            isError: false
        )
    }

    // MARK: - timeline

    func handleTimeline(_ args: [String: Value]?) throws -> CallTool.Result {
        let a = try args.decode(TimelineArgs.self)
        let groupBy = a.groupBy ?? "day"
        let limit = a.limit?.value ?? 50

        guard ["day", "week", "month"].contains(groupBy) else {
            throw MCPError.invalidParams("'group_by' must be 'day', 'week', or 'month', got '\(groupBy)'")
        }

        // Build query with filters
        var results = lattice.objects(Memory.self)
            .where { $0.expiresAt > Date() }

        if let project = a.project {
            results = results.where { $0.project == project }
        }
        if let topic = a.topic {
            results = results.where { $0.topic == topic }
        }
        if let sinceStr = a.since {
            guard let sinceDate = parseTemporalDate(sinceStr) else {
                throw MCPError.invalidParams("Could not parse 'since' date: \(sinceStr)")
            }
            results = results.where { $0.createdAt >= sinceDate }
        }
        if let beforeStr = a.before {
            guard let beforeDate = parseTemporalDate(beforeStr) else {
                throw MCPError.invalidParams("Could not parse 'before' date: \(beforeStr)")
            }
            results = results.where { $0.createdAt <= beforeDate }
        }

        // Sort by createdAt descending in SQL, apply limit
        let limited = results
            .sortedBy(.init(\.createdAt, order: .reverse))
            .snapshot(limit: Int64(limit))

        if limited.isEmpty {
            return CallTool.Result(content: [.text("No memories found.")], isError: false)
        }

        // Group by calendar period
        let cal = Calendar.current
        var groups: [(key: Date, memories: [Memory])] = []
        var currentGroupDate: Date?
        var currentGroup: [Memory] = []

        for mem in limited {
            let periodStart: Date
            switch groupBy {
            case "week":
                periodStart = cal.dateInterval(of: .weekOfYear, for: mem.createdAt)?.start ?? cal.startOfDay(for: mem.createdAt)
            case "month":
                periodStart = cal.dateInterval(of: .month, for: mem.createdAt)?.start ?? cal.startOfDay(for: mem.createdAt)
            default: // "day"
                periodStart = cal.startOfDay(for: mem.createdAt)
            }

            if periodStart == currentGroupDate {
                currentGroup.append(mem)
            } else {
                if let key = currentGroupDate, !currentGroup.isEmpty {
                    groups.append((key: key, memories: currentGroup))
                }
                currentGroupDate = periodStart
                currentGroup = [mem]
            }
        }
        if let key = currentGroupDate, !currentGroup.isEmpty {
            groups.append((key: key, memories: currentGroup))
        }

        // Format output with date headers
        let headerFormatter = DateFormatter()
        switch groupBy {
        case "week":
            headerFormatter.dateFormat = "'Week of' MMM d, yyyy"
        case "month":
            headerFormatter.dateFormat = "MMMM yyyy"
        default:
            headerFormatter.dateFormat = "MMM d, yyyy"
        }

        var output = ""
        for (i, group) in groups.enumerated() {
            if i > 0 { output += "\n\n" }
            let countLabel = group.memories.count == 1 ? "1 memory" : "\(group.memories.count) memories"
            output += "## \(headerFormatter.string(from: group.key)) (\(countLabel))"
            for mem in group.memories {
                let impInfo = mem.importance > 0 ? " [importance: \(mem.importance)]" : ""
                output += "\n[id:\(mem.primaryKey!)] [\(mem.project)/\(mem.topic)]\(impInfo) \(mem.content)"
            }
        }

        return CallTool.Result(content: [.text(output)], isError: false)
    }
}
