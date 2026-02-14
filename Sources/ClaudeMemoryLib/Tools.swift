import Lattice
import MCP
import Foundation

public func log(_ message: String) {
    FileHandle.standardError.write(Data("[claude-memory] \(message)\n".utf8))
}

// MARK: - Codable Argument Decoding

extension Optional where Wrapped == [String: MCP.Value] {
    func decode<T: Decodable>(_ type: T.Type) throws -> T {
        let data = try JSONEncoder().encode(self ?? [:])
        return try JSONDecoder().decode(type, from: data)
    }
}

struct FlexibleInt: Decodable {
    let value: Int
    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let int = try? container.decode(Int.self) {
            value = int
        } else if let string = try? container.decode(String.self), let int = Int(string) {
            value = int
        } else {
            throw DecodingError.typeMismatch(
                Int.self,
                DecodingError.Context(codingPath: decoder.codingPath, debugDescription: "Expected an integer or string-encoded integer")
            )
        }
    }
}

private struct RememberArgs: Decodable {
    let content: String
    let topic: String?
    let project: String?
    let source: String?
    let expiresInDays: FlexibleInt?
    let force: Bool?
    let importance: FlexibleInt?

    enum CodingKeys: String, CodingKey {
        case content, topic, project, source, force, importance
        case expiresInDays = "expires_in_days"
    }
}

private struct RecallArgs: Decodable {
    let query: String
    let project: String?
    let topic: String?
    let limit: FlexibleInt?
    let depth: FlexibleInt?
}

private struct ForgetArgs: Decodable {
    let id: FlexibleInt?
    let topic: String?
    let project: String?
}

private struct UpdateArgs: Decodable {
    // Targeting (one required)
    let id: FlexibleInt?
    let query: String?
    let project: String?

    // Content editing (all optional, mutually exclusive)
    let content: String?
    let append: String?
    let prepend: String?
    let find: String?
    let replace: String?

    // Field-level metadata updates (all optional)
    let topic: String?
    let source: String?
    let expiresInDays: FlexibleInt?
    let importance: FlexibleInt?

    enum CodingKeys: String, CodingKey {
        case id, query, project, content, append, prepend, find, replace, topic, source, importance
        case expiresInDays = "expires_in_days"
    }
}

private struct MergeArgs: Decodable {
    let ids: [FlexibleInt]
    let content: String
    let topic: String?
    let project: String?
}

private struct StatsArgs: Decodable {
    let project: String?
}

private struct ListTopicsArgs: Decodable {
    let project: String?
}

private struct ConnectArgs: Decodable {
    let from: FlexibleInt
    let to: FlexibleInt
    let relation: String
}

private struct DisconnectArgs: Decodable {
    let id: FlexibleInt?
    let from: FlexibleInt?
    let to: FlexibleInt?
    let relation: String?
}

private struct GraphArgs: Decodable {
    let id: FlexibleInt
    let depth: FlexibleInt?
}

/// Valid relation types for knowledge graph edges.
private let validRelations: Set<String> = [
    "relates_to", "contradicts", "supersedes", "derived_from", "part_of",
]

/// Implements the MCP tool handlers for the ClaudeMemory server.
///
/// Tools:
/// - **remember**: Store a memory with semantic embedding for later recall.
/// - **recall**: Search memories by semantic similarity, with soft project boosting and optional graph traversal.
/// - **forget**: Delete memories by ID, topic, project (cascades edges).
/// - **list_topics**: List all memory topics with counts, optionally filtered by project.
/// - **update**: Edit existing memories by ID or similarity.
/// - **stats**: Overview of the memory database.
/// - **merge**: Consolidate multiple memories into one (cleans up source edges).
/// - **connect**: Create a directed edge between two memories.
/// - **disconnect**: Remove edges by ID or by (from, to) pair.
/// - **graph**: View a memory's neighborhood in the knowledge graph.
public actor MemoryTools {
    let lattice: Lattice
    let embedder: EmbeddingService

    static let dateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateStyle = .medium
        f.timeStyle = .none
        return f
    }()

    public init(lattice: Lattice, embedder: EmbeddingService) {
        self.lattice = lattice
        self.embedder = embedder
    }

    // MARK: - Tool Definitions

    public var definitions: [Tool] {
        [
            Tool(
                name: "remember",
                description: """
                    Store a memory for later recall. Use this to save important context, \
                    preferences, patterns, decisions, or debugging insights. Memories are \
                    embedded as vectors for semantic search.
                    """,
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "content": .object([
                            "type": .string("string"),
                            "description": .string("The memory text to store. Be specific and self-contained — this should make sense without surrounding context."),
                        ]),
                        "topic": .object([
                            "type": .string("string"),
                            "description": .string("Category: 'preferences', 'architecture', 'debugging', 'patterns', 'conventions', or any custom topic. Defaults to 'general'."),
                        ]),
                        "project": .object([
                            "type": .string("string"),
                            "description": .string("Project name this memory belongs to (e.g., 'Lattice', 'MyApp'). Use 'global' for cross-project knowledge like user preferences. Defaults to 'global'."),
                        ]),
                        "source": .object([
                            "type": .string("string"),
                            "description": .string("Where this memory came from: 'conversation', 'code-review', 'debugging-session', or a file path."),
                        ]),
                        "expires_in_days": .object([
                            "type": .string("integer"),
                            "description": .string("Number of days until this memory expires. Expired memories are automatically filtered from recall. Omit for permanent memories. Use for temporal context like 'currently working on X' or 'PR #42 needs review'."),
                        ]),
                        "force": .object([
                            "type": .string("boolean"),
                            "description": .string("Skip conflict detection and store even if near-duplicates exist. Use after reviewing a conflict warning."),
                        ]),
                        "importance": .object([
                            "type": .string("integer"),
                            "description": .string("Importance rating (1-5). Higher values boost this memory in recall ranking. Omit or 0 for default."),
                        ]),
                    ]),
                    "required": .array([.string("content")]),
                    "additionalProperties": .bool(false),
                ])
            ),
            Tool(
                name: "recall",
                description: """
                    Search memories by semantic similarity. Returns the most relevant stored \
                    memories for a query. Project is a soft ranking signal — same-project and \
                    global memories rank higher, but cross-project results still surface if \
                    semantically relevant. Use depth > 0 to follow knowledge graph edges.
                    """,
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "query": .object([
                            "type": .string("string"),
                            "description": .string("Natural language search query to find relevant memories."),
                        ]),
                        "project": .object([
                            "type": .string("string"),
                            "description": .string("Project to boost in ranking. Same-project and global memories rank higher, but cross-project results still appear if relevant. If omitted, no project boosting."),
                        ]),
                        "topic": .object([
                            "type": .string("string"),
                            "description": .string("Optional topic filter to narrow search (e.g., 'debugging', 'architecture')."),
                        ]),
                        "limit": .object([
                            "type": .string("integer"),
                            "description": .string("Maximum number of results (default 10)."),
                        ]),
                        "depth": .object([
                            "type": .string("integer"),
                            "description": .string("Graph traversal depth (0 = no traversal, 1 = follow direct edges, max 3). Default 0."),
                        ]),
                    ]),
                    "required": .array([.string("query")]),
                    "additionalProperties": .bool(false),
                ])
            ),
            Tool(
                name: "forget",
                description: "Delete stored memories. Can delete by ID (from recall output), or filter by topic, project, or delete all. Automatically removes associated knowledge graph edges.",
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "id": .object([
                            "type": .string("integer"),
                            "description": .string("Delete a specific memory by ID (from recall output [id:N] prefix). Takes priority over topic/project filters."),
                        ]),
                        "topic": .object([
                            "type": .string("string"),
                            "description": .string("Delete only memories with this topic."),
                        ]),
                        "project": .object([
                            "type": .string("string"),
                            "description": .string("Delete only memories for this project."),
                        ]),
                    ]),
                    "additionalProperties": .bool(false),
                ])
            ),
            Tool(
                name: "list_topics",
                description: "List all memory topics with counts, optionally filtered by project.",
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "project": .object([
                            "type": .string("string"),
                            "description": .string("Optional project filter. If omitted, lists topics across all projects."),
                        ]),
                    ]),
                    "additionalProperties": .bool(false),
                ])
            ),
            Tool(
                name: "update",
                description: """
                    Update an existing memory. Target by ID (preferred) or semantic similarity. \
                    Supports full content replacement, append, prepend, find-and-replace, and \
                    field-level metadata updates. Content edit modes are mutually exclusive. \
                    Metadata updates (topic, source, expires_in_days) can combine with any content edit.
                    """,
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "id": .object([
                            "type": .string("integer"),
                            "description": .string("Target memory by ID (from recall output [id:N] prefix). Preferred over query."),
                        ]),
                        "query": .object([
                            "type": .string("string"),
                            "description": .string("Find memory by semantic similarity. Used when ID is not known."),
                        ]),
                        "project": .object([
                            "type": .string("string"),
                            "description": .string("Project scope for similarity search (only used with query, not id)."),
                        ]),
                        "content": .object([
                            "type": .string("string"),
                            "description": .string("Full content replacement."),
                        ]),
                        "append": .object([
                            "type": .string("string"),
                            "description": .string("Text to append to existing content."),
                        ]),
                        "prepend": .object([
                            "type": .string("string"),
                            "description": .string("Text to prepend to existing content."),
                        ]),
                        "find": .object([
                            "type": .string("string"),
                            "description": .string("Text to find in content (used with replace)."),
                        ]),
                        "replace": .object([
                            "type": .string("string"),
                            "description": .string("Replacement text (used with find). Can be empty to delete matched text."),
                        ]),
                        "topic": .object([
                            "type": .string("string"),
                            "description": .string("Update the memory's topic."),
                        ]),
                        "source": .object([
                            "type": .string("string"),
                            "description": .string("Update the memory's source."),
                        ]),
                        "expires_in_days": .object([
                            "type": .string("integer"),
                            "description": .string("Update expiration. 0 = make permanent, >0 = expire in N days from now."),
                        ]),
                        "importance": .object([
                            "type": .string("integer"),
                            "description": .string("Update importance rating (1-5). Higher values boost this memory in recall ranking. 0 to clear."),
                        ]),
                    ]),
                    "additionalProperties": .bool(false),
                ])
            ),
            Tool(
                name: "stats",
                description: "Get a quick overview of the memory database: total count, per-project and per-topic breakdowns.",
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "project": .object([
                            "type": .string("string"),
                            "description": .string("Optional project filter. If omitted, shows stats across all projects."),
                        ]),
                    ]),
                    "additionalProperties": .bool(false),
                ])
            ),
            Tool(
                name: "merge",
                description: """
                    Merge multiple memories into one. Use after recalling related memories to \
                    consolidate fragments into a single, well-organized memory. The original \
                    memories are deleted and replaced with the merged content. Associated \
                    knowledge graph edges are cleaned up automatically.
                    """,
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "ids": .object([
                            "type": .string("array"),
                            "items": .object(["type": .string("integer")]),
                            "description": .string("Array of memory IDs to merge (from recall output)."),
                        ]),
                        "content": .object([
                            "type": .string("string"),
                            "description": .string("The merged/consolidated content to replace the originals."),
                        ]),
                        "topic": .object([
                            "type": .string("string"),
                            "description": .string("Topic for the merged memory. Defaults to the topic of the first source memory."),
                        ]),
                        "project": .object([
                            "type": .string("string"),
                            "description": .string("Project for the merged memory. Defaults to the project of the first source memory."),
                        ]),
                    ]),
                    "required": .array([.string("ids"), .string("content")]),
                    "additionalProperties": .bool(false),
                ])
            ),
            Tool(
                name: "connect",
                description: """
                    Create a directed edge between two memories in the knowledge graph. \
                    Edges represent typed relationships. Duplicate edges (same from, to, relation) \
                    are idempotent.
                    """,
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "from": .object([
                            "type": .string("integer"),
                            "description": .string("Source memory ID."),
                        ]),
                        "to": .object([
                            "type": .string("integer"),
                            "description": .string("Target memory ID."),
                        ]),
                        "relation": .object([
                            "type": .string("string"),
                            "enum": .array([
                                .string("relates_to"),
                                .string("contradicts"),
                                .string("supersedes"),
                                .string("derived_from"),
                                .string("part_of"),
                            ]),
                            "description": .string("Relationship type: 'relates_to', 'contradicts', 'supersedes', 'derived_from', 'part_of'."),
                        ]),
                    ]),
                    "required": .array([.string("from"), .string("to"), .string("relation")]),
                    "additionalProperties": .bool(false),
                ])
            ),
            Tool(
                name: "disconnect",
                description: """
                    Remove edges from the knowledge graph. Target by edge ID, or by \
                    (from, to) memory pair with optional relation filter.
                    """,
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "id": .object([
                            "type": .string("integer"),
                            "description": .string("Delete a specific edge by its ID."),
                        ]),
                        "from": .object([
                            "type": .string("integer"),
                            "description": .string("Source memory ID (used with 'to')."),
                        ]),
                        "to": .object([
                            "type": .string("integer"),
                            "description": .string("Target memory ID (used with 'from')."),
                        ]),
                        "relation": .object([
                            "type": .string("string"),
                            "description": .string("Optional relation filter when using from+to targeting."),
                        ]),
                    ]),
                    "additionalProperties": .bool(false),
                ])
            ),
            Tool(
                name: "graph",
                description: """
                    View a memory's neighborhood in the knowledge graph. Shows the memory \
                    and its connections (edges) up to a specified depth.
                    """,
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "id": .object([
                            "type": .string("integer"),
                            "description": .string("Memory ID to explore from."),
                        ]),
                        "depth": .object([
                            "type": .string("integer"),
                            "description": .string("How many hops to traverse (default 1, max 3)."),
                        ]),
                    ]),
                    "required": .array([.string("id")]),
                    "additionalProperties": .bool(false),
                ])
            ),
        ]
    }

    // MARK: - Dispatch

    public func handle(_ params: CallTool.Parameters) async throws -> CallTool.Result {
        switch params.name {
        case "remember":
            return try await handleRemember(params.arguments)
        case "recall":
            return try await handleRecall(params.arguments)
        case "forget":
            return try handleForget(params.arguments)
        case "list_topics":
            return try handleListTopics(params.arguments)
        case "update":
            return try await handleUpdate(params.arguments)
        case "stats":
            return try handleStats(params.arguments)
        case "merge":
            return try await handleMerge(params.arguments)
        case "connect":
            return try handleConnect(params.arguments)
        case "disconnect":
            return try handleDisconnect(params.arguments)
        case "graph":
            return try handleGraph(params.arguments)
        default:
            throw MCPError.invalidParams("Unknown tool: \(params.name)")
        }
    }

    // MARK: - remember

    private func handleRemember(_ args: [String: Value]?) async throws -> CallTool.Result {
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
        // Uses tiered thresholds: 0.12 for same-project, 0.05 for cross-scope (global↔project)
        if a.force != true && !embeddingVec.isEmpty {
            let candidates = lattice.objects(Memory.self)
                .where { $0.expiresAt > Date() }
                .where { $0.project == project || $0.project == "global" }
                .nearest(to: embeddingVec, on: \.embedding, limit: 5, distance: .cosine)

            let conflicts = candidates.filter { match in
                let sameProject = match.object.project == project
                let threshold = sameProject ? 0.12 : 0.05
                return match.distance < threshold
            }
            if !conflicts.isEmpty {
                var warning = "⚠️ Near-duplicate memory detected. The new memory was NOT stored.\n\nExisting similar memories:"
                for match in conflicts {
                    let m = match.object
                    let dist = String(format: "%.3f", match.distance)
                    warning += "\n  [id:\(m.primaryKey!)] (distance: \(dist)) \(m.content)"
                }
                warning += "\n\nTo resolve:"
                warning += "\n  - Use `update(id: N, ...)` to modify the existing memory"
                warning += "\n  - Use `remember(..., force: true)` to keep both"
                warning += "\n  - Use `forget(id: N)` to remove the old one, then `remember` the new one"
                log("Conflict detected for: \(content.prefix(80))")
                return CallTool.Result(content: [.text(warning)], isError: false)
            }
        }

        let memory = Memory(content: content, topic: topic, project: project, source: source, embedding: embeddingVec, expiresAt: expiresAt, importance: importance)
        lattice.add(memory)

        let expiresNote = expiresAt == .distantFuture ? "" : ", expires: \(Self.dateFormatter.string(from: expiresAt))"
        let importanceNote = importance > 0 ? ", importance: \(importance)" : ""
        log("Stored memory [\(project)/\(topic)]: \(content.prefix(80))")
        return CallTool.Result(
            content: [.text("Stored memory (id: \(memory.primaryKey!), project: \(project), topic: \(topic)\(expiresNote)\(importanceNote)): \(content.prefix(100))\(content.count > 100 ? "..." : "")")],
            isError: false
        )
    }

    // MARK: - recall

    private func handleRecall(_ args: [String: Value]?) async throws -> CallTool.Result {
        let a = try args.decode(RecallArgs.self)
        guard !a.query.isEmpty else {
            throw MCPError.invalidParams("'query' is required")
        }
        let query = a.query
        let projectFilter = a.project
        let topicFilter = a.topic
        let limit = a.limit?.value ?? 10
        let depth = min(a.depth?.value ?? 0, 3)

        // Build base query — always filter out expired memories
        var results = lattice.objects(Memory.self)
            .where { $0.expiresAt > Date() }

        // Topic is still a hard filter (it's a narrow constraint)
        if let topicFilter {
            results = results.where { $0.topic == topicFilter }
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
                return "[id:\(m.primaryKey!)] [\(m.project)/\(m.topic)] (relevance: \(dist)\(ftsInfo)\(impInfo)\(expires)) \(m.content)"
            }

            var output = lines.joined(separator: "\n\n")

            // Graph traversal when depth > 0
            if depth > 0 {
                let recalledIds = Set(filtered.map { $0.object.primaryKey! })
                let connected = traverseGraph(from: recalledIds, depth: depth, excludeIds: recalledIds)
                if !connected.isEmpty {
                    output += "\n\n--- Connected (graph traversal, depth: \(depth)) ---"
                    let connNow = Date()
                    for mem in connected {
                        mem.lastAccessedAt = connNow
                        mem.accessCount += 1
                        let expires = mem.expiresAt == .distantFuture ? "" : ", expires: \(Self.dateFormatter.string(from: mem.expiresAt))"
                        output += "\n\n[id:\(mem.primaryKey!)] [\(mem.project)/\(mem.topic)]\(expires) \(mem.content)"
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
                lines.append("[id:\(m.primaryKey!)] [\(m.project)/\(m.topic)]\(ftsInfo)\(expires) \(m.content)")
            }
            if lines.isEmpty {
                return CallTool.Result(content: [.text("No memories found.")], isError: false)
            }
            return CallTool.Result(content: [.text(lines.joined(separator: "\n\n"))], isError: false)
        }
    }

    // MARK: - forget

    private func handleForget(_ args: [String: Value]?) throws -> CallTool.Result {
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
            let memoryIds = collectMemoryIds { $0.topic == topic && $0.project == project }
            let edgeCount = deleteEdgesForMemories(memoryIds)
            let count = memoryIds.count
            lattice.delete(Memory.self, where: { $0.topic == topic && $0.project == project })
            let edgeNote = edgeCount > 0 ? " Removed \(edgeCount) edge(s)." : ""
            return CallTool.Result(
                content: [.text("Deleted \(count) memories (project: \(project), topic: \(topic)).\(edgeNote)")],
                isError: false
            )
        case let (topic?, nil):
            let memoryIds = collectMemoryIds { $0.topic == topic }
            let edgeCount = deleteEdgesForMemories(memoryIds)
            let count = memoryIds.count
            lattice.delete(Memory.self, where: { $0.topic == topic })
            let edgeNote = edgeCount > 0 ? " Removed \(edgeCount) edge(s)." : ""
            return CallTool.Result(
                content: [.text("Deleted \(count) memories with topic '\(topic)'.\(edgeNote)")],
                isError: false
            )
        case let (nil, project?):
            let memoryIds = collectMemoryIds { $0.project == project }
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

    private func handleUpdate(_ args: [String: Value]?) async throws -> CallTool.Result {
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

    private func handleMerge(_ args: [String: Value]?) async throws -> CallTool.Result {
        let a = try args.decode(MergeArgs.self)
        guard !a.content.isEmpty else {
            throw MCPError.invalidParams("'content' is required")
        }
        let ids = a.ids.map { Int64($0.value) }
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

    // MARK: - connect

    private func handleConnect(_ args: [String: Value]?) throws -> CallTool.Result {
        let a = try args.decode(ConnectArgs.self)
        let fromId = Int64(a.from.value)
        let toId = Int64(a.to.value)
        let relation = a.relation

        // Validate relation
        guard validRelations.contains(relation) else {
            throw MCPError.invalidParams("Invalid relation '\(relation)'. Must be one of: \(validRelations.sorted().joined(separator: ", "))")
        }

        // Validate both memories exist
        guard lattice.objects(Memory.self).where({ $0.primaryKey == fromId }).first != nil else {
            return CallTool.Result(content: [.text("Memory with id \(fromId) not found.")], isError: true)
        }
        guard lattice.objects(Memory.self).where({ $0.primaryKey == toId }).first != nil else {
            return CallTool.Result(content: [.text("Memory with id \(toId) not found.")], isError: true)
        }

        // Check for duplicate edge
        let existing = lattice.objects(Edge.self)
            .where { $0.sourceId == fromId && $0.targetId == toId && $0.relation == relation }
        if let edge = existing.first {
            return CallTool.Result(
                content: [.text("Edge already exists (edge id: \(edge.primaryKey!), \(fromId) --[\(relation)]--> \(toId)).")],
                isError: false
            )
        }

        // Create edge
        let edge = Edge(sourceId: fromId, targetId: toId, relation: relation)
        lattice.add(edge)

        log("Connected [id:\(fromId)] --[\(relation)]--> [id:\(toId)]")
        return CallTool.Result(
            content: [.text("Connected (edge id: \(edge.primaryKey!)) [id:\(fromId)] --[\(relation)]--> [id:\(toId)]")],
            isError: false
        )
    }

    // MARK: - disconnect

    private func handleDisconnect(_ args: [String: Value]?) throws -> CallTool.Result {
        let a = try args.decode(DisconnectArgs.self)

        // By edge ID
        if let id = a.id?.value {
            let id64 = Int64(id)
            let matches = lattice.objects(Edge.self).where { $0.primaryKey == id64 }
            guard matches.first != nil else {
                return CallTool.Result(content: [.text("Edge with id \(id) not found.")], isError: true)
            }
            lattice.delete(Edge.self, where: { $0.primaryKey == id64 })
            log("Disconnected edge [id:\(id)]")
            return CallTool.Result(
                content: [.text("Deleted edge (id: \(id)).")],
                isError: false
            )
        }

        // By from + to
        guard let from = a.from?.value, let to = a.to?.value else {
            throw MCPError.invalidParams("Provide 'id' or both 'from' and 'to' to target edges.")
        }
        let fromId = Int64(from)
        let toId = Int64(to)

        var query = lattice.objects(Edge.self)
            .where { $0.sourceId == fromId && $0.targetId == toId }
        if let relation = a.relation {
            query = query.where { $0.relation == relation }
        }

        let count = query.count
        if count == 0 {
            return CallTool.Result(content: [.text("No edges found from \(fromId) to \(toId).")], isError: false)
        }

        if let relation = a.relation {
            lattice.delete(Edge.self, where: { $0.sourceId == fromId && $0.targetId == toId && $0.relation == relation })
        } else {
            lattice.delete(Edge.self, where: { $0.sourceId == fromId && $0.targetId == toId })
        }

        log("Disconnected \(count) edge(s) from [id:\(fromId)] to [id:\(toId)]")
        return CallTool.Result(
            content: [.text("Deleted \(count) edge(s) from [id:\(fromId)] to [id:\(toId)].")],
            isError: false
        )
    }

    // MARK: - graph

    private func handleGraph(_ args: [String: Value]?) throws -> CallTool.Result {
        let a = try args.decode(GraphArgs.self)
        let memId = Int64(a.id.value)
        let depth = min(a.depth?.value ?? 1, 3)

        // Validate memory exists
        guard let rootMem = lattice.objects(Memory.self).where({ $0.primaryKey == memId }).first else {
            return CallTool.Result(content: [.text("Memory with id \(memId) not found.")], isError: true)
        }

        // BFS traversal
        var visited = Set<Int64>([memId])
        var frontier = Set<Int64>([memId])
        var allEdges: [(edge: Edge, depth: Int)] = []

        for d in 1...depth {
            var nextFrontier = Set<Int64>()
            for nodeId in frontier {
                // Outgoing edges
                let outgoing = lattice.objects(Edge.self).where { $0.sourceId == nodeId }
                for edge in outgoing {
                    allEdges.append((edge: edge, depth: d))
                    if !visited.contains(edge.targetId) {
                        visited.insert(edge.targetId)
                        nextFrontier.insert(edge.targetId)
                    }
                }
                // Incoming edges
                let incoming = lattice.objects(Edge.self).where { $0.targetId == nodeId }
                for edge in incoming {
                    allEdges.append((edge: edge, depth: d))
                    if !visited.contains(edge.sourceId) {
                        visited.insert(edge.sourceId)
                        nextFrontier.insert(edge.sourceId)
                    }
                }
            }
            frontier = nextFrontier
            if frontier.isEmpty { break }
        }

        // Deduplicate edges by primary key
        var seenEdgeIds = Set<Int64>()
        var uniqueEdges: [Edge] = []
        for (edge, _) in allEdges {
            let edgeId = edge.primaryKey!
            if !seenEdgeIds.contains(edgeId) {
                seenEdgeIds.insert(edgeId)
                uniqueEdges.append(edge)
            }
        }

        // Format output
        var output = "[id:\(memId)] \(rootMem.content)"

        if uniqueEdges.isEmpty {
            output += "\n\nNo connections."
        } else {
            output += "\n\nConnections:"
            for edge in uniqueEdges {
                if edge.sourceId == memId {
                    // Outgoing
                    let targetContent = lattice.objects(Memory.self)
                        .where { $0.primaryKey == edge.targetId }.first?.content ?? "(deleted)"
                    output += "\n  --[\(edge.relation)]--> [id:\(edge.targetId)] \(targetContent.prefix(80))"
                } else if edge.targetId == memId {
                    // Incoming
                    let sourceContent = lattice.objects(Memory.self)
                        .where { $0.primaryKey == edge.sourceId }.first?.content ?? "(deleted)"
                    output += "\n  <--[\(edge.relation)]-- [id:\(edge.sourceId)] \(sourceContent.prefix(80))"
                } else {
                    // Edge between two non-root nodes (deeper traversal)
                    let sourceContent = lattice.objects(Memory.self)
                        .where { $0.primaryKey == edge.sourceId }.first?.content ?? "(deleted)"
                    let targetContent = lattice.objects(Memory.self)
                        .where { $0.primaryKey == edge.targetId }.first?.content ?? "(deleted)"
                    output += "\n  [id:\(edge.sourceId)] \(sourceContent.prefix(40))... --[\(edge.relation)]--> [id:\(edge.targetId)] \(targetContent.prefix(40))..."
                }
            }
        }

        return CallTool.Result(content: [.text(output)], isError: false)
    }

    // MARK: - stats

    private func handleStats(_ args: [String: Value]?) throws -> CallTool.Result {
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

    private func handleListTopics(_ args: [String: Value]?) throws -> CallTool.Result {
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

    // MARK: - Graph Helpers

    /// Collect IDs of memories matching a predicate.
    private func collectMemoryIds(_ predicate: (Memory) -> Bool) -> [Int64] {
        lattice.objects(Memory.self).filter(predicate).compactMap(\.primaryKey)
    }

    /// Delete all edges where sourceId or targetId is in the given set. Returns count deleted.
    @discardableResult
    private func deleteEdgesForMemories(_ ids: [Int64]) -> Int {
        var total = 0
        for id in ids {
            let count = lattice.count(Edge.self, where: { $0.sourceId == id || $0.targetId == id })
            if count > 0 {
                lattice.delete(Edge.self, where: { $0.sourceId == id || $0.targetId == id })
                total += count
            }
        }
        return total
    }

    /// BFS graph traversal from a set of starting memory IDs, returning connected memories.
    private func traverseGraph(from startIds: Set<Int64>, depth: Int, excludeIds: Set<Int64>) -> [Memory] {
        var visited = excludeIds
        var frontier = startIds
        var result: [Memory] = []
        let now = Date()

        for _ in 1...depth {
            var nextFrontier = Set<Int64>()
            for nodeId in frontier {
                // Outgoing edges
                for edge in lattice.objects(Edge.self).where({ $0.sourceId == nodeId }) {
                    if !visited.contains(edge.targetId) {
                        visited.insert(edge.targetId)
                        nextFrontier.insert(edge.targetId)
                    }
                }
                // Incoming edges
                for edge in lattice.objects(Edge.self).where({ $0.targetId == nodeId }) {
                    if !visited.contains(edge.sourceId) {
                        visited.insert(edge.sourceId)
                        nextFrontier.insert(edge.sourceId)
                    }
                }
            }
            // Fetch the memories for the next frontier
            for id in nextFrontier {
                if let mem = lattice.objects(Memory.self).where({ $0.primaryKey == id }).first {
                    // Skip expired
                    if mem.expiresAt > now {
                        result.append(mem)
                    }
                }
            }
            frontier = nextFrontier
            if frontier.isEmpty { break }
        }

        return result
    }
}
