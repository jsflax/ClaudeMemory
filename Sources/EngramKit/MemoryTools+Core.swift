import Lattice
import MCP
import Foundation
import NaturalLanguage

// MARK: - Direct Access (for hooks)

extension MemoryTools {
    /// Perform a recall and return the text result directly, without MCP framing.
    /// Used by the hooks binary to inject recalled memories as context.
    public func directRecall(query: String, project: String?, depth: Int = 1, limit: Int = 5) async throws -> String? {
        var args: [String: Value] = ["query": .string(query)]
        if let project {
            args["project"] = .string(project)
        }
        args["depth"] = .int(depth)
        args["limit"] = .int(limit)
        let result = try await handleRecall(args)
        // Extract text from the CallTool.Result
        guard let textContent = result.content.first else { return nil }
        switch textContent {
        case .text(let text, _, _):
            return text == "No memories found." ? nil : text
        default:
            return nil
        }
    }
}

// MARK: - Content Word Extraction

extension MemoryTools {
    /// POS tags to drop from queries — function words that add noise to FTS5 search.
    /// Language-agnostic: NLTagger assigns these tags regardless of input language.
    private static let dropTags: Set<NLTag> = [
        .determiner, .pronoun, .preposition, .conjunction, .particle,
    ]

    /// Extract content words from a query using NLTagger POS tagging.
    /// Drops determiners, pronouns, prepositions, conjunctions, and particles.
    /// Keeps nouns, verbs (including auxiliaries), adjectives, adverbs, and
    /// unknown/technical terms. Auxiliary verbs that slip through are acceptable —
    /// the vector search handles ranking regardless, and POS filtering catches
    /// the highest-noise function words across all languages.
    ///
    /// Falls back to all whitespace-split words if no content words survive.
    public static func extractContentWords(from query: String) -> [String] {
        guard !query.isEmpty else { return [] }

        let tagger = NLTagger(tagSchemes: [.lexicalClass])
        tagger.string = query

        var contentWords: [String] = []
        tagger.enumerateTags(
            in: query.startIndex..<query.endIndex,
            unit: .word,
            scheme: .lexicalClass
        ) { tag, range in
            // Skip tokens with no tag (whitespace, punctuation)
            guard let tag else { return true }
            // Drop known function-word POS tags
            if Self.dropTags.contains(tag) { return true }
            // Keep everything else: nouns, verbs, adjectives, adverbs, unknown terms
            let word = String(query[range]).trimmingCharacters(in: .whitespaces)
            if !word.isEmpty { contentWords.append(word) }
            return true
        }

        // Fall back to raw split if NLTagger stripped everything
        if contentWords.isEmpty {
            return query.split(separator: " ").map(String.init)
        }
        return contentWords
    }
}

// MARK: - CRUD Operation Tracking

extension MemoryTools {
    /// Increment the CRUD operation counter in HookState.
    /// Called after successful remember, forget, update, merge, consolidate.
    func incrementCrudCounter() {
        let current = Int(lattice.objects(HookState.self)
            .where { $0.key == .crudOperationCount }
            .first?.value ?? "0") ?? 0
        let newCount = current + 1
        if let existing = lattice.objects(HookState.self).where({ $0.key == .crudOperationCount }).first {
            existing.value = String(newCount)
            existing.updatedAt = Date()
        } else {
            lattice.add(HookState(key: .crudOperationCount, value: String(newCount)))
        }
    }

    /// Reset the maintenance baseline to the current CRUD op count.
    /// Called after organize/consolidate — the actual maintenance actions.
    func resetMaintenanceBaseline() {
        let opCount = lattice.objects(HookState.self)
            .where { $0.key == .crudOperationCount }
            .first?.value ?? "0"
        if let existing = lattice.objects(HookState.self).where({ $0.key == .maintenanceLastOpCount }).first {
            existing.value = opCount
            existing.updatedAt = Date()
        } else {
            lattice.add(HookState(key: .maintenanceLastOpCount, value: opCount))
        }
    }
}

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

        guard let floats = try await embedder.embed(text: content) else {
            throw MCPError.internalError("Embedding model unavailable — cannot store memory without a vector. Check that the CoreML model is bundled correctly.")
        }
        let embeddingVec = Vector<Float>(floats)

        // Conflict detection + auto-connect candidate gathering
        // Search is done outside the force guard so auto-connect candidates survive force=true
        var autoConnectCandidates: [(object: Memory, distance: Double)] = []

        if !embeddingVec.isEmpty && topic != "episode" {
            let candidates = lattice.objects(Memory.self)
                .where { $0.expiresAt > Date() && $0.topic != "episode" }
                .where { $0.project == project || $0.project == "global" }
                .nearest(to: embeddingVec, on: \.embedding, limit: 8, distance: .cosine)

            // Conflict detection (unchanged behavior, just nested under force guard)
            if a.force != true {
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
                        guard let mId = m.primaryKey else { continue }
                        let dist = String(format: "%.3f", match.distance)
                        let jaccard = String(format: "%.0f%%", jaccardSimilarity(content, m.content) * 100)
                        warning += "\n  [id:\(mId)] (distance: \(dist), term overlap: \(jaccard)) \(m.content)"
                    }
                    warning += "\n\nTo resolve:"
                    warning += "\n  - Use `update(id: N, ...)` to modify the existing memory"
                    warning += "\n  - Use `remember(..., force: true)` to keep both"
                    warning += "\n  - Use `forget(id: N)` to remove the old one, then `remember` the new one"
                    log("Conflict detected for: \(content.prefix(80))")
                    return CallTool.Result(content: [.text(warning)], isError: false)
                }
            }

            // Gather auto-connect candidates (beyond conflict zone, within relatedness threshold)
            autoConnectCandidates = candidates.compactMap { match in
                let sameProject = match.object.project == project
                let conflictThreshold = sameProject ? 0.12 : 0.05
                guard match.distance >= conflictThreshold && match.distance < 0.20 else { return nil }
                return (object: match.object, distance: match.distance)
            }
            autoConnectCandidates.sort { $0.distance < $1.distance }
            autoConnectCandidates = Array(autoConnectCandidates.prefix(3))
        }

        // Episode: end stale episodes on 30-min gap
        if activeEpisodeId != nil {
            let gap = Date().timeIntervalSince(lastMemoryTime)
            if gap > 1800 { endActiveEpisode() }
        }
        lastMemoryTime = Date()

        let isPrivate = a.isPrivate ?? false

        let memory = Memory(content: content, topic: topic, project: project, source: source, embedding: embeddingVec, expiresAt: expiresAt, importance: importance, isPrivate: isPrivate)
        lattice.add(memory)

        guard let memoryId = memory.primaryKey else {
            throw MCPError.internalError("Failed to persist memory — primaryKey is nil after add()")
        }

        // Auto-create part_of edge when parent_id is provided
        var parentNote = ""
        if let parentId = a.parentId?.value {
            let pid = Int64(parentId)
            guard lattice.objects(Memory.self).where({ $0.primaryKey == pid }).first != nil else {
                throw MCPError.invalidParams("parent_id \(parentId) not found")
            }
            let edge = Edge(sourceId: memoryId, targetId: pid, relation: "part_of")
            lattice.add(edge)
            parentNote = ", parent: \(parentId)"
            log("Auto-created part_of edge: \(memoryId) -> \(parentId)")
        }

        // Link to active episode via part_of edge
        if let epId = activeEpisodeId {
            let edge = Edge(sourceId: memoryId, targetId: epId, relation: "part_of")
            lattice.add(edge)
            log("Linked memory \(memoryId) to episode \(epId)")
        }

        // Auto-connect: create relates_to edges to semantically similar memories
        var autoLinkedIds: [Int64] = []
        let parentIdValue: Int64? = a.parentId.map { Int64($0.value) }

        for candidate in autoConnectCandidates {
            guard let targetId = candidate.object.primaryKey else { continue }
            // Skip if already linked as parent or episode
            if let pid = parentIdValue, targetId == pid { continue }
            if let epId = activeEpisodeId, targetId == epId { continue }
            // Dedup: check both directions
            let hasEdge = lattice.objects(Edge.self)
                .where { ($0.sourceId == memoryId && $0.targetId == targetId && $0.relation == "relates_to")
                      || ($0.sourceId == targetId && $0.targetId == memoryId && $0.relation == "relates_to") }
                .first != nil
            guard !hasEdge else { continue }

            let edge = Edge(sourceId: memoryId, targetId: targetId, relation: "relates_to")
            lattice.add(edge)
            autoLinkedIds.append(targetId)
            log("Auto-connected [\(memoryId)] --[relates_to]--> [\(targetId)] (distance: \(String(format: "%.3f", candidate.distance)))")
        }

        // Cross-project hub linking: if content mentions another project by name, link to its hub
        if topic != "episode" {
            let allProjects = Set(
                lattice.objects(Memory.self)
                    .snapshot()
                    .map(\.project)
            ).subtracting([project, "global"])

            for otherProject in allProjects {
                guard otherProject.count >= 3 else { continue }

                let pattern = "\\b\(NSRegularExpression.escapedPattern(for: otherProject))\\b"
                guard let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive),
                      regex.firstMatch(in: content, range: NSRange(content.startIndex..., in: content)) != nil
                else { continue }

                // Find hub: the memory with the most incoming part_of edges
                let projectMemories = lattice.objects(Memory.self)
                    .where { $0.project == otherProject && $0.topic != "episode" }
                    .snapshot()

                var bestHub: (id: Int64, count: Int)? = nil
                for mem in projectMemories {
                    guard let memId = mem.primaryKey else { continue }
                    let incomingCount = lattice.objects(Edge.self)
                        .where { $0.targetId == memId && $0.relation == "part_of" }
                        .count
                    if incomingCount > 0 && (bestHub == nil || incomingCount > bestHub!.count) {
                        bestHub = (id: memId, count: incomingCount)
                    }
                }

                guard let hub = bestHub else { continue }

                let alreadyLinked = lattice.objects(Edge.self)
                    .where { ($0.sourceId == memoryId && $0.targetId == hub.id && $0.relation == "relates_to")
                          || ($0.sourceId == hub.id && $0.targetId == memoryId && $0.relation == "relates_to") }
                    .first != nil
                guard !alreadyLinked else { continue }

                let edge = Edge(sourceId: memoryId, targetId: hub.id, relation: "relates_to")
                lattice.add(edge)
                autoLinkedIds.append(hub.id)
                log("Cross-project link [\(memoryId)] --[relates_to]--> [\(hub.id)] (project '\(otherProject)' mentioned in content)")
            }
        }

        // Incremental topic/hub inference from auto-connect neighbors
        if !autoConnectCandidates.isEmpty {
            var hubCounts: [Int64: Int] = [:]
            var topicCounts: [String: Int] = [:]

            for candidate in autoConnectCandidates {
                guard let candidateId = candidate.object.primaryKey else { continue }
                // Find hubs this neighbor belongs to (outgoing part_of to non-episode memory)
                for edge in lattice.objects(Edge.self)
                    .where({ $0.sourceId == candidateId && $0.relation == "part_of" }) {
                    if lattice.objects(Memory.self)
                        .where({ $0.primaryKey == edge.targetId && $0.topic != "episode" }).first != nil {
                        hubCounts[edge.targetId, default: 0] += 1
                    }
                }
                // Check if this neighbor IS a hub (has incoming part_of edges)
                if candidate.object.topic != "episode" {
                    let hasIncoming = lattice.objects(Edge.self)
                        .where { $0.targetId == candidateId && $0.relation == "part_of" }
                        .first != nil
                    if hasIncoming {
                        hubCounts[candidateId, default: 0] += 1
                    }
                }
                // Count non-generic topics
                let t = candidate.object.topic
                if t != "general" && t != "episode" {
                    topicCounts[t, default: 0] += 1
                }
            }

            // Auto-link to hub if >= 2 neighbors share one
            if let (hubId, count) = hubCounts.max(by: { $0.value < $1.value }),
               count >= 2,
               parentIdValue.map({ $0 != hubId }) ?? true {
                let alreadyLinked = lattice.objects(Edge.self)
                    .where { $0.sourceId == memoryId && $0.targetId == hubId && $0.relation == "part_of" }
                    .first != nil
                if !alreadyLinked {
                    let edge = Edge(sourceId: memoryId, targetId: hubId, relation: "part_of")
                    lattice.add(edge)
                    log("Auto-organized [\(memoryId)] into hub [\(hubId)]")
                }
            }

            // Inherit topic if neighbors agree and Claude used "general"
            if topic == "general",
               let (consensusTopic, count) = topicCounts.max(by: { $0.value < $1.value }),
               count >= 2 {
                memory.topic = consensusTopic
                log("Auto-inferred topic '\(consensusTopic)' for [\(memoryId)]")
            }
        }

        let expiresNote = expiresAt == .distantFuture ? "" : ", expires: \(Self.dateFormatter.string(from: expiresAt))"
        let importanceNote = importance > 0 ? ", importance: \(importance)" : ""
        let privateNote = isPrivate ? ", private: true" : ""
        let autoLinkNote = autoLinkedIds.isEmpty ? "" : ", auto-linked to \(autoLinkedIds.map { "[id:\($0)]" }.joined(separator: ", "))"
        log("Stored memory [\(project)/\(topic)]: \(content.prefix(80))")

        var response = "Stored memory (id: \(memoryId), project: \(project), topic: \(topic)\(parentNote)\(expiresNote)\(importanceNote)\(privateNote)\(autoLinkNote)): \(content.prefix(100))\(content.count > 100 ? "..." : "")"

        // Nudge toward atomic memories when content is complex
        let lines = content.components(separatedBy: "\n")
        let headers = lines.filter { $0.hasPrefix("## ") || $0.hasPrefix("### ") }
        if content.count > 1000 && headers.count >= 2 {
            let names = headers.map {
                $0.drop(while: { $0 == "#" || $0 == " " })
            }
            response += "\n\n💡 This memory has \(headers.count) sections (\(names.joined(separator: ", "))). "
            response += "Consider storing each section as a child memory with `parent_id: \(memoryId)` for more precise recall."
        }

        incrementCrudCounter()
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

        // FTS5 query: extract content words via NLTagger, fall back to raw split
        let contentWords = Self.extractContentWords(from: query)
        let ftsTerms = contentWords.isEmpty ? query.split(separator: " ").map(String.init) : contentWords
        let ftsQuery: TextQuery = ._anyOf(ftsTerms)

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

                // Staleness penalty — never-accessed memories older than 14 days rank lower (up to 20%)
                let daysSinceCreation = now.timeIntervalSince(m.createdAt) / 86400.0
                let stalenessPenalty: Double = (m.accessCount == 0 && daysSinceCreation > 14.0)
                    ? 1.0 + min((daysSinceCreation - 14.0) / 180.0 * 0.20, 0.20)
                    : 1.0

                let distance = cosine * projectBoost * frequencyBoost * importanceBoost * recencyBoost * stalenessPenalty
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

            let lines = filtered.compactMap { match -> String? in
                let m = match.object
                guard let mId = m.primaryKey else { return nil }
                let dist = String(format: "%.3f", match.distance)
                let ftsInfo = match.ftsRank.map { ", fts5: \(String(format: "%.3f", $0))" } ?? ""
                let impInfo = m.importance > 0 ? ", importance: \(m.importance)" : ""
                let expires = m.expiresAt == .distantFuture ? "" : ", expires: \(Self.dateFormatter.string(from: m.expiresAt))"
                let created = hasTemporalFilter ? ", created: \(Self.dateFormatter.string(from: m.createdAt))" : ""
                return "[id:\(mId)] [\(m.project)/\(m.topic)] (distance: \(dist)\(ftsInfo)\(impInfo)\(expires)\(created)) \(m.content)"
            }

            var output = lines.joined(separator: "\n\n")

            // Knowledge gap detection — signal when recall results are weak
            let avgDistance = filtered.map(\.distance).reduce(0, +) / Double(max(filtered.count, 1))
            if avgDistance > 0.07 {
                output = "⚠️ Weak recall (avg distance: \(String(format: "%.3f", avgDistance)), count: \(filtered.count)). Results may not be closely related to the query.\n\n" + output
            }

            // Graph traversal when depth > 0
            if depth > 0 {
                let recalledIds = Set(filtered.compactMap { $0.object.primaryKey })
                let allConnected = traverseGraph(from: recalledIds, depth: depth, excludeIds: recalledIds)
                // Filter connected memories by relevance to query.
                // Structural edges (part_of, derived_from, supersedes) always pass through.
                // Loose edges (relates_to, contradicts) require semantic proximity.
                let queryVec = Vector<Float>(queryEmbedding)
                let structuralRelations: Set<String> = ["part_of", "derived_from", "supersedes"]
                let connected = allConnected.filter { mem in
                    guard let memId = mem.memory.primaryKey else { return false }
                    // Check if any edge connecting this memory to a recalled memory is structural
                    let hasStructuralEdge = lattice.objects(Edge.self)
                        .where { ($0.sourceId == memId || $0.targetId == memId) }
                        .contains { structuralRelations.contains($0.relation) }
                    if hasStructuralEdge { return true }
                    // Loose edges: filter by cosine distance to query
                    guard mem.memory.embedding.dimensions > 0 else { return false }
                    return Double(mem.memory.embedding.cosineDistance(to: queryVec)) <= 0.15
                }
                if !connected.isEmpty {
                    output += "\n\n--- Connected (graph traversal, depth: \(depth)) ---"
                    let connNow = Date()
                    // Track all known IDs (recalled + connected so far) for edge lookup at depth>1
                    var knownIds = recalledIds
                    for mem in connected {
                        let m = mem.memory
                        guard let memId = m.primaryKey else { continue }
                        m.lastAccessedAt = connNow
                        m.accessCount += 1

                        let expires = m.expiresAt == .distantFuture ? "" : ", expires: \(Self.dateFormatter.string(from: m.expiresAt))"

                        // Look up the edge relation connecting this memory to any known memory
                        var edgeInfo = ""
                        if let edge = lattice.objects(Edge.self).where({ $0.targetId == memId }).first(where: { knownIds.contains($0.sourceId) }) {
                            edgeInfo = " <--[\(edge.relation)]-- [id:\(edge.sourceId)]"
                        } else if let edge = lattice.objects(Edge.self).where({ $0.sourceId == memId }).first(where: { knownIds.contains($0.targetId) }) {
                            edgeInfo = " --[\(edge.relation)]--> [id:\(edge.targetId)]"
                        }
                        knownIds.insert(memId)

                        // Small memories shown in full; large ones get a compact preview
                        if m.content.count <= 500 {
                            output += "\n\n[id:\(memId)] [\(m.project)/\(m.topic)]\(expires)\(edgeInfo) \(m.content)"
                        } else {
                            let firstLine = m.content.split(separator: "\n", maxSplits: 1).first.map(String.init) ?? m.content
                            let preview = String(firstLine.prefix(120))
                            let charCount = m.content.count
                            let sectionCount = m.content.components(separatedBy: "\n").filter { $0.hasPrefix("## ") || $0.hasPrefix("### ") }.count
                            let sizeInfo = sectionCount > 0 ? "\(sectionCount) sections, \(charCount) chars" : "\(charCount) chars"
                            output += "\n\n[id:\(memId)] [\(m.project)/\(m.topic)] (\(sizeInfo)\(expires))\(edgeInfo) \(preview)\(charCount > 120 ? "..." : "")"
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
                guard let mId = m.primaryKey else { continue }
                lines.append("[id:\(mId)] [\(m.project)/\(m.topic)]\(ftsInfo)\(expires)\(created) \(m.content)")
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
            incrementCrudCounter()
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
            incrementCrudCounter()
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
            incrementCrudCounter()
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
            incrementCrudCounter()
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
        let hasMetadataEdit = a.setProject != nil || a.topic != nil || a.source != nil || a.expiresInDays != nil || a.importance != nil || a.isPrivate != nil
        guard hasContentEdit || hasMetadataEdit else {
            throw MCPError.invalidParams("Provide at least one edit: content, append, prepend, find+replace, set_project, topic, source, or expires_in_days.")
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

        if let project = a.setProject {
            let old = mem.project
            mem.project = project
            changes.append("project: \(old) → \(project)")
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
        if let priv = a.isPrivate {
            let old = mem.isPrivate
            mem.isPrivate = priv
            changes.append("private: \(old) → \(priv)")
        }

        // 8. Re-embed only if content changed
        if contentChanged {
            if let newEmbedding = try await embedder.embed(text: mem.content) {
                mem.embedding = Vector<Float>(newEmbedding)
            }
        }

        // 9. Update lastAccessedAt
        mem.lastAccessedAt = Date()

        let memId = mem.primaryKey.map(String.init) ?? "unknown"
        log("Updated memory [id:\(memId)] [\(mem.project)/\(mem.topic)]: \(changes.joined(separator: ", "))")
        // Count as CRUD op unless only the topic changed
        let hasNonTopicChange = contentChanged || a.setProject != nil || a.source != nil || a.expiresInDays != nil || a.importance != nil || a.isPrivate != nil
        if hasNonTopicChange {
            incrementCrudCounter()
        }
        return CallTool.Result(
            content: [.text("Updated memory (id: \(memId), project: \(mem.project), topic: \(mem.topic)).\nChanges:\n\(changes.joined(separator: "\n"))")],
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
        guard let floats = try await embedder.embed(text: content) else {
            throw MCPError.internalError("Embedding model unavailable — cannot merge memories without a vector. Check that the CoreML model is bundled correctly.")
        }
        let embeddingVec = Vector<Float>(floats)

        // Create merged memory
        let merged = Memory(content: content, topic: topic, project: project, source: "merged", embedding: embeddingVec)
        lattice.add(merged)

        guard let mergedId = merged.primaryKey else {
            throw MCPError.internalError("Failed to persist merged memory — primaryKey is nil after add()")
        }

        // Collect old content summaries before deleting
        let oldSummaries = sources.map { "[id:\($0.primaryKey.map(String.init) ?? "?")] \($0.content.prefix(60))" }

        // Clean up edges referencing source memories
        let edgeCount = deleteEdgesForMemories(ids)

        // Delete originals
        for id in ids {
            lattice.delete(Memory.self, where: { $0.primaryKey == id })
        }

        let edgeNote = edgeCount > 0 ? " Removed \(edgeCount) edge(s) from source memories." : ""
        log("Merged \(ids.count) memories into [id:\(mergedId)]")
        incrementCrudCounter()
        return CallTool.Result(
            content: [.text("Merged \(ids.count) memories into new memory (id: \(mergedId), project: \(project), topic: \(topic)).\(edgeNote)\n\nDeleted:\n\(oldSummaries.joined(separator: "\n"))\n\nNew:\n\(content)")],
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
                let memId = mem.primaryKey.map(String.init) ?? "?"
                output += "\n[id:\(memId)] [\(mem.project)/\(mem.topic)]\(impInfo) \(mem.content)"
            }
        }

        return CallTool.Result(content: [.text(output)], isError: false)
    }
}
