import Lattice
import MCP
import Foundation

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

    /// Tracks the currently active episode for this session (auto or explicit).
    var activeEpisodeId: Int64? = nil
    /// Whether the active episode was explicitly created via begin_episode (vs auto-created).
    var isExplicitEpisode: Bool = false
    /// Tracks when the last memory was stored, for auto-episode gap detection.
    var lastMemoryTime: Date = .distantPast

    static let dateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateStyle = .medium
        f.timeStyle = .none
        return f
    }()

    static let isoDateFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd"
        return f
    }()

    public init(lattice: Lattice, embedder: EmbeddingService) {
        self.lattice = lattice
        self.embedder = embedder
    }

    #if DEBUG
    /// Test-only: set lastMemoryTime to simulate time gaps for auto-episode.
    package func setLastMemoryTime(_ date: Date) { lastMemoryTime = date }
    #endif

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
                        "since": .object([
                            "type": .string("string"),
                            "description": .string("Only return memories created after this date. Accepts ISO 8601 (\"2024-01-15\"), relative shorthand (\"7d\", \"2w\", \"3m\"), or natural language (\"last week\", \"yesterday\")."),
                        ]),
                        "before": .object([
                            "type": .string("string"),
                            "description": .string("Only return memories created before this date. Same formats as since."),
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
                                .string("summarized_by"),
                            ]),
                            "description": .string("Relationship type: 'relates_to', 'contradicts', 'supersedes', 'derived_from', 'part_of', 'summarized_by'."),
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
            Tool(
                name: "timeline",
                description: """
                    View memories in chronological order, grouped by time period. \
                    Useful for browsing what was learned recently or during a specific \
                    time range. No semantic search — purely chronological.
                    """,
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "project": .object([
                            "type": .string("string"),
                            "description": .string("Filter to memories from this project."),
                        ]),
                        "topic": .object([
                            "type": .string("string"),
                            "description": .string("Filter to memories with this topic."),
                        ]),
                        "since": .object([
                            "type": .string("string"),
                            "description": .string("Only show memories created after this date. Accepts ISO 8601 (\"2024-01-15\"), relative shorthand (\"7d\", \"2w\", \"3m\"), or natural language (\"last week\", \"yesterday\")."),
                        ]),
                        "before": .object([
                            "type": .string("string"),
                            "description": .string("Only show memories created before this date. Same formats as since."),
                        ]),
                        "group_by": .object([
                            "type": .string("string"),
                            "enum": .array([.string("day"), .string("week"), .string("month")]),
                            "description": .string("Group memories by time period: 'day' (default), 'week', or 'month'."),
                        ]),
                        "limit": .object([
                            "type": .string("integer"),
                            "description": .string("Maximum number of memories to return (default 50)."),
                        ]),
                    ]),
                    "additionalProperties": .bool(false),
                ])
            ),
            Tool(
                name: "checkpoint",
                description: """
                    Create or update a task checkpoint. Use this to save work-in-progress \
                    state so you can resume later. On create, provide at least a title. \
                    On update, provide task_id and any fields to change.
                    """,
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "task_id": .object([
                            "type": .string("integer"),
                            "description": .string("ID of an existing task to update. Omit to create a new task."),
                        ]),
                        "title": .object([
                            "type": .string("string"),
                            "description": .string("Short title describing the task. Required when creating a new task."),
                        ]),
                        "status": .object([
                            "type": .string("string"),
                            "enum": .array([.string("active"), .string("paused"), .string("completed")]),
                            "description": .string("Task status: 'active' (in progress), 'paused' (shelved), 'completed' (done)."),
                        ]),
                        "project": .object([
                            "type": .string("string"),
                            "description": .string("Project this task belongs to. Defaults to 'global'."),
                        ]),
                        "plan": .object([
                            "type": .string("string"),
                            "description": .string("The planned approach or steps."),
                        ]),
                        "progress": .object([
                            "type": .string("string"),
                            "description": .string("What has been accomplished so far."),
                        ]),
                        "context": .object([
                            "type": .string("string"),
                            "description": .string("Free-form context: file paths, decisions, blockers, notes."),
                        ]),
                    ]),
                    "additionalProperties": .bool(false),
                ])
            ),
            Tool(
                name: "resume",
                description: """
                    Load a task checkpoint by ID and mark it active. Returns the full \
                    task state (plan, progress, context) so you can pick up where you left off.
                    """,
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "task_id": .object([
                            "type": .string("integer"),
                            "description": .string("ID of the task to resume (from list_tasks or checkpoint output)."),
                        ]),
                    ]),
                    "required": .array([.string("task_id")]),
                    "additionalProperties": .bool(false),
                ])
            ),
            Tool(
                name: "list_tasks",
                description: """
                    List task checkpoints, optionally filtered by project and/or status. \
                    By default shows active and paused tasks (not completed).
                    """,
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "project": .object([
                            "type": .string("string"),
                            "description": .string("Filter to tasks in this project."),
                        ]),
                        "status": .object([
                            "type": .string("string"),
                            "enum": .array([.string("active"), .string("paused"), .string("completed")]),
                            "description": .string("Filter by status. If omitted, shows active and paused (not completed)."),
                        ]),
                        "limit": .object([
                            "type": .string("integer"),
                            "description": .string("Maximum number of tasks to return (default 20)."),
                        ]),
                    ]),
                    "additionalProperties": .bool(false),
                ])
            ),
            Tool(
                name: "begin_episode",
                description: """
                    Start a new episode to group related memories into a narrative session. \
                    Memories created after this call are automatically associated with the episode \
                    until it is ended. If an auto-created episode is active, it will be closed first.
                    """,
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "title": .object([
                            "type": .string("string"),
                            "description": .string("Title for the episode. If omitted, auto-generates 'Session: {date}'."),
                        ]),
                        "project": .object([
                            "type": .string("string"),
                            "description": .string("Project this episode belongs to. Defaults to 'global'."),
                        ]),
                    ]),
                    "additionalProperties": .bool(false),
                ])
            ),
            Tool(
                name: "end_episode",
                description: """
                    End an active episode. Optionally provide a summary of what happened. \
                    If no episode_id is given, ends the currently active episode.
                    """,
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "episode_id": .object([
                            "type": .string("integer"),
                            "description": .string("ID of the episode to end. If omitted, ends the currently active episode."),
                        ]),
                        "summary": .object([
                            "type": .string("string"),
                            "description": .string("A summary of what happened in this episode."),
                        ]),
                    ]),
                    "additionalProperties": .bool(false),
                ])
            ),
            Tool(
                name: "recall_episode",
                description: """
                    Retrieve an episode and its memories in chronological order. \
                    Use this to replay what happened during a session.
                    """,
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "episode_id": .object([
                            "type": .string("integer"),
                            "description": .string("ID of the episode to recall (from list_episodes or begin_episode output)."),
                        ]),
                        "limit": .object([
                            "type": .string("integer"),
                            "description": .string("Maximum number of memories to return (default 200)."),
                        ]),
                    ]),
                    "required": .array([.string("episode_id")]),
                    "additionalProperties": .bool(false),
                ])
            ),
            Tool(
                name: "list_episodes",
                description: """
                    List episodes, optionally filtered by project and/or status. \
                    Shows episodes sorted by most recent first.
                    """,
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "project": .object([
                            "type": .string("string"),
                            "description": .string("Filter to episodes in this project."),
                        ]),
                        "status": .object([
                            "type": .string("string"),
                            "enum": .array([.string("active"), .string("ended")]),
                            "description": .string("Filter by status: 'active' or 'ended'. If omitted, shows all."),
                        ]),
                        "limit": .object([
                            "type": .string("integer"),
                            "description": .string("Maximum number of episodes to return (default 20)."),
                        ]),
                    ]),
                    "additionalProperties": .bool(false),
                ])
            ),
            Tool(
                name: "find_clusters",
                description: """
                    Find groups of semantically similar memories that could be consolidated \
                    into summaries. Returns clusters with their member memories for review \
                    before consolidation.
                    """,
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "project": .object([
                            "type": .string("string"),
                            "description": .string("Scope search to memories in this project."),
                        ]),
                        "topic": .object([
                            "type": .string("string"),
                            "description": .string("Scope search to memories with this topic."),
                        ]),
                        "min_cluster_size": .object([
                            "type": .string("integer"),
                            "description": .string("Minimum memories to form a cluster (default 3)."),
                        ]),
                        "distance_threshold": .object([
                            "type": .string("integer"),
                            "description": .string("Max cosine distance × 100 for cluster membership (default 30, i.e. 0.30). Lower = tighter clusters."),
                        ]),
                        "max_clusters": .object([
                            "type": .string("integer"),
                            "description": .string("Maximum clusters to return (default 10)."),
                        ]),
                    ]),
                    "additionalProperties": .bool(false),
                ])
            ),
            Tool(
                name: "consolidate",
                description: """
                    Create a summary memory from a set of related memories. Originals are \
                    deprioritized (importance set to 0) and linked to the summary via \
                    'summarized_by' edges. The summary gets an embedding for future recall.
                    """,
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "ids": .object([
                            "type": .string("array"),
                            "items": .object(["type": .string("integer")]),
                            "description": .string("Memory IDs to consolidate (from recall or find_clusters output)."),
                        ]),
                        "content": .object([
                            "type": .string("string"),
                            "description": .string("Summary content written by you, capturing the essential knowledge from the memories."),
                        ]),
                        "topic": .object([
                            "type": .string("string"),
                            "description": .string("Topic for the summary. Defaults to the most common topic among the originals."),
                        ]),
                        "project": .object([
                            "type": .string("string"),
                            "description": .string("Project for the summary. Defaults to the project of the first original."),
                        ]),
                        "importance": .object([
                            "type": .string("integer"),
                            "description": .string("Importance of the summary memory (1-5, default 3)."),
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
        case "connect":
            return try handleConnect(params.arguments)
        case "disconnect":
            return try handleDisconnect(params.arguments)
        case "graph":
            return try handleGraph(params.arguments)
        case "timeline":
            return try handleTimeline(params.arguments)
        case "checkpoint":
            return try handleCheckpoint(params.arguments)
        case "resume":
            return try handleResume(params.arguments)
        case "list_tasks":
            return try handleListTasks(params.arguments)
        case "begin_episode":
            return try handleBeginEpisode(params.arguments)
        case "end_episode":
            return try handleEndEpisode(params.arguments)
        case "recall_episode":
            return try handleRecallEpisode(params.arguments)
        case "list_episodes":
            return try handleListEpisodes(params.arguments)
        case "find_clusters":
            return try await handleFindClusters(params.arguments)
        case "consolidate":
            return try await handleConsolidate(params.arguments)
        default:
            throw MCPError.invalidParams("Unknown tool: \(params.name)")
        }
    }
}
