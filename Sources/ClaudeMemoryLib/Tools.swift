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

    enum CodingKeys: String, CodingKey {
        case content, topic, project, source
        case expiresInDays = "expires_in_days"
    }
}

private struct RecallArgs: Decodable {
    let query: String
    let project: String?
    let topic: String?
    let limit: FlexibleInt?
}

private struct ForgetArgs: Decodable {
    let id: FlexibleInt?
    let topic: String?
    let project: String?
}

private struct UpdateArgs: Decodable {
    let query: String
    let newContent: String
    let project: String?

    enum CodingKeys: String, CodingKey {
        case query, project
        case newContent = "new_content"
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

/// Implements the MCP tool handlers for the ClaudeMemory server.
///
/// Tools:
/// - **remember**: Store a memory with semantic embedding for later recall.
/// - **recall**: Search memories by semantic similarity, scoped by project.
/// - **forget**: Delete memories by ID, topic, project, or all.
/// - **list_topics**: List all memory topics with counts, optionally filtered by project.
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
                    ]),
                    "required": .array([.string("content")]),
                    "additionalProperties": .bool(false),
                ])
            ),
            Tool(
                name: "recall",
                description: """
                    Search memories by semantic similarity. Returns the most relevant stored \
                    memories for a query. Results include both project-specific and global memories.
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
                            "description": .string("Project to search within. Results include both project-specific AND global memories. If omitted, searches all memories."),
                        ]),
                        "topic": .object([
                            "type": .string("string"),
                            "description": .string("Optional topic filter to narrow search (e.g., 'debugging', 'architecture')."),
                        ]),
                        "limit": .object([
                            "type": .string("integer"),
                            "description": .string("Maximum number of results (default 10)."),
                        ]),
                    ]),
                    "required": .array([.string("query")]),
                    "additionalProperties": .bool(false),
                ])
            ),
            Tool(
                name: "forget",
                description: "Delete stored memories. Can delete by ID (from recall output), or filter by topic, project, or delete all.",
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
                    Find a memory by semantic similarity and replace its content. Useful for \
                    correcting or refining an existing memory without creating a duplicate.
                    """,
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "query": .object([
                            "type": .string("string"),
                            "description": .string("Search query to find the memory to update."),
                        ]),
                        "new_content": .object([
                            "type": .string("string"),
                            "description": .string("Replacement content for the memory."),
                        ]),
                        "project": .object([
                            "type": .string("string"),
                            "description": .string("Optional project scope for the search."),
                        ]),
                    ]),
                    "required": .array([.string("query"), .string("new_content")]),
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
                    memories are deleted and replaced with the merged content.
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

        var embeddingVec = Vector<Float>([])
        if let floats = try await embedder.embed(text: content) {
            embeddingVec = Vector<Float>(floats)
        }

        let memory = Memory(content: content, topic: topic, project: project, source: source, embedding: embeddingVec, expiresAt: expiresAt)
        lattice.add(memory)

        let expiresNote = expiresAt == .distantFuture ? "" : ", expires: \(Self.dateFormatter.string(from: expiresAt))"
        log("Stored memory [\(project)/\(topic)]: \(content.prefix(80))")
        return CallTool.Result(
            content: [.text("Stored memory (id: \(memory.primaryKey!), project: \(project), topic: \(topic)\(expiresNote)): \(content.prefix(100))\(content.count > 100 ? "..." : "")")],
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

        // Build base query — if project specified, include both project-specific AND global memories
        // Always filter out expired memories
        var results = lattice.objects(Memory.self)
            .where { $0.expiresAt > Date() }
        if let projectFilter {
            results = results.where { $0.project == projectFilter || $0.project == "global" }
        }
        if let topicFilter {
            results = results.where { $0.topic == topicFilter }
        }

        // Semantic search with vector similarity
        if let queryEmbedding = try await embedder.embed(text: query) {
            let nearest = results
                .nearest(to: Vector<Float>(queryEmbedding), on: \.embedding, limit: limit, distance: .cosine)

            if nearest.isEmpty {
                return CallTool.Result(content: [.text("No memories found.")], isError: false)
            }

            // Filter out outliers: use adaptive threshold based on the result cluster
            // Takes the 75th percentile distance and adds 20% margin
            let distances = nearest.map(\.distance)
            let p75 = distances[distances.count * 3 / 4]
            let threshold = p75 * 1.2
            let filtered = nearest.filter { $0.distance <= threshold }

            // Bump lastAccessedAt on recalled memories
            let now = Date()
            for match in filtered {
                match.object.lastAccessedAt = now
            }

            let lines = filtered.map { match in
                let m = match.object
                let dist = String(format: "%.3f", match.distance)
                let expires = m.expiresAt == .distantFuture ? "" : ", expires: \(Self.dateFormatter.string(from: m.expiresAt))"
                return "[id:\(m.primaryKey!)] [\(m.project)/\(m.topic)] (relevance: \(dist)\(expires)) \(m.content)"
            }
            return CallTool.Result(content: [.text(lines.joined(separator: "\n\n"))], isError: false)
        } else {
            // Degraded mode: text search with SQL LIKE (no embedding model loaded)
            let filtered = results.where { $0.content.contains(query) }

            var lines: [String] = []
            for m in filtered.prefix(limit) {
                let expires = m.expiresAt == .distantFuture ? "" : ", expires: \(Self.dateFormatter.string(from: m.expiresAt))"
                lines.append("[id:\(m.primaryKey!)] [\(m.project)/\(m.topic)]\(expires) \(m.content)")
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
            lattice.delete(Memory.self, where: { $0.primaryKey == id64 })
            log("Deleted memory [id:\(id)]: \(summary)")
            return CallTool.Result(
                content: [.text("Deleted memory (id: \(id), project: \(mem.project), topic: \(mem.topic)): \(summary)")],
                isError: false
            )
        }

        switch (a.topic, a.project) {
        case let (topic?, project?):
            let count = lattice.count(Memory.self, where: { $0.topic == topic && $0.project == project })
            lattice.delete(Memory.self, where: { $0.topic == topic && $0.project == project })
            return CallTool.Result(
                content: [.text("Deleted \(count) memories (project: \(project), topic: \(topic)).")],
                isError: false
            )
        case let (topic?, nil):
            let count = lattice.count(Memory.self, where: { $0.topic == topic })
            lattice.delete(Memory.self, where: { $0.topic == topic })
            return CallTool.Result(
                content: [.text("Deleted \(count) memories with topic '\(topic)'.")],
                isError: false
            )
        case let (nil, project?):
            let count = lattice.count(Memory.self, where: { $0.project == project })
            lattice.delete(Memory.self, where: { $0.project == project })
            return CallTool.Result(
                content: [.text("Deleted \(count) memories for project '\(project)'.")],
                isError: false
            )
        case (nil, nil):
            throw MCPError.invalidParams("Specify 'id', 'topic', or 'project'. Refusing to delete all memories without an explicit filter.")
        }
    }

    // MARK: - update

    private func handleUpdate(_ args: [String: Value]?) async throws -> CallTool.Result {
        let a = try args.decode(UpdateArgs.self)
        guard !a.query.isEmpty else {
            throw MCPError.invalidParams("'query' is required")
        }
        guard !a.newContent.isEmpty else {
            throw MCPError.invalidParams("'new_content' is required")
        }
        let query = a.query
        let newContent = a.newContent
        let projectFilter = a.project

        var results = lattice.objects(Memory.self)
        if let projectFilter {
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

        let mem = match.object
        let oldContent = mem.content

        // Re-embed with new content
        if let newEmbedding = try await embedder.embed(text: newContent) {
            mem.embedding = Vector<Float>(newEmbedding)
        }
        mem.content = newContent
        mem.lastAccessedAt = Date()

        log("Updated memory [\(mem.project)/\(mem.topic)]: \(newContent.prefix(80))")
        return CallTool.Result(
            content: [.text("Updated memory (project: \(mem.project), topic: \(mem.topic)).\nOld: \(oldContent)\nNew: \(newContent)")],
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

        // Delete originals
        for id in ids {
            lattice.delete(Memory.self, where: { $0.primaryKey == id })
        }

        log("Merged \(ids.count) memories into [id:\(merged.primaryKey!)]")
        return CallTool.Result(
            content: [.text("Merged \(ids.count) memories into new memory (id: \(merged.primaryKey!), project: \(project), topic: \(topic)).\n\nDeleted:\n\(oldSummaries.joined(separator: "\n"))\n\nNew:\n\(content)")],
            isError: false
        )
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
}
