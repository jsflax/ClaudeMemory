import Lattice
import MCP
import Foundation

/// Implements the MCP tool handlers for the Engram memory server.
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
    /// Local database — all writes go here. Reads go here for local-only projects.
    package let localLattice: Lattice
    /// Synced database (memory-synced.sqlite) — reads go here for synced projects.
    /// Contains cross-device data relayed by the sync daemon. Nil if no synced DB exists.
    package let syncedLattice: Lattice?
    package let embedder: EmbeddingService

    /// Tracks the currently active episode globalId for this session.
    var activeEpisodeId: UUID? = nil
    /// Tracks when the last memory was stored, for episode gap detection.
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

    /// Resolves sendable references inside the actor's isolation domain.
    public init(localRef: LatticeThreadSafeReference, syncedRef: LatticeThreadSafeReference?, embedder: EmbeddingService) {
        guard let local = localRef.resolve() else {
            fatalError("Failed to resolve local lattice reference")
        }
        self.localLattice = local
        self.syncedLattice = syncedRef?.resolve()
        self.embedder = embedder
    }

    /// Returns the right Lattice for reading based on the project's sync policy.
    /// For synced projects, returns a UNION ALL view spanning both local and synced DBs
    /// so we get local memories + cross-device data in one query.
    /// For local projects, returns localLattice directly.
    /// Cached attached Lattice — created once, reused across recalls.
    private var _attachedLattice: Lattice?

    package func readLattice(for project: String?) -> Lattice {
        log("[readLattice] enter, project=\(project ?? "nil"), syncedLattice=\(syncedLattice != nil)")
        guard let syncedLattice, let project else {
            log("[readLattice] returning localLattice (no sync or no project)")
            return localLattice
        }
        let isSynced: Bool
        log("[readLattice] querying SyncConfig for project=\(project)")
        if let config = localLattice.objects(SyncConfig.self)
            .where({ $0.project == project }).first {
            isSynced = config.policy == .sync
            log("[readLattice] found config, isSynced=\(isSynced)")
        } else if let fallback = localLattice.objects(SyncConfig.self)
            .where({ $0.project == "_default" }).first {
            isSynced = fallback.policy == .sync
            log("[readLattice] using default config, isSynced=\(isSynced)")
        } else {
            isSynced = false
            log("[readLattice] no config, isSynced=false")
        }
        log("[readLattice] about to return, isSynced=\(isSynced)")
        if isSynced {
            if _attachedLattice == nil {
                _attachedLattice = localLattice.attaching(lattice: syncedLattice)
            }
            log("[readLattice] returning cached attached lattice")
            return _attachedLattice!
        }
        log("[readLattice] returning localLattice")
        return localLattice
    }

    /// The signed-in user's id for authorUserId stamping — from the
    /// daemon-authored groups.json (never the Keychain; short-lived
    /// processes can't reliably read it). Nil when signed out — writers
    /// stamp nil and the daemon-start backfill sweep repairs later.
    var currentUserId: UUID? { GroupDirectory.currentUserId() }

    /// Whether this memory is (or would be) shared with any group: not
    /// private, and its project is exposed to at least one group. Drives
    /// tombstone-vs-hard-delete — hard deletes must never propagate into a
    /// shared graph (they LWW-replicate to every member).
    func isGroupShared(_ mem: Memory) -> Bool {
        guard !mem.isPrivate else { return false }
        guard let config = localLattice.objects(SyncConfig.self)
            .where({ $0.project == mem.project }).first else { return false }
        return !config.exposedGroups.isEmpty
    }

    /// Tombstones a memory in place (soft delete): hidden from every read
    /// path, recoverable via `update(undelete: true)`, propagates as a
    /// field-delta UPDATE that survives LWW.
    func tombstone(_ mem: Memory, in lattice: Lattice) {
        lattice.transaction {
            mem.deletedAt = Date()
            mem.deletedBy = currentUserId
            mem.modifiedAt = Date()
        }
    }

    /// Finds a memory by globalId, trying localLattice first then syncedLattice.
    /// Returns the memory and which lattice it was found in.
    func findMemory(id: UUID) -> (memory: Memory, lattice: Lattice)? {
        if let mem = localLattice.objects(Memory.self)
            .where({ $0.__globalId == id }).first {
            return (mem, localLattice)
        }
        if let syncedLattice,
           let mem = syncedLattice.objects(Memory.self)
            .where({ $0.__globalId == id }).first {
            return (mem, syncedLattice)
        }
        return nil
    }

    /// Finds an edge by globalId, trying localLattice first then syncedLattice.
    func findEdge(id: UUID) -> (edge: Edge, lattice: Lattice)? {
        if let edge = localLattice.objects(Edge.self)
            .where({ $0.__globalId == id }).first {
            return (edge, localLattice)
        }
        if let syncedLattice,
           let edge = syncedLattice.objects(Edge.self)
            .where({ $0.__globalId == id }).first {
            return (edge, syncedLattice)
        }
        return nil
    }

    #if DEBUG
    /// Test-only: set lastMemoryTime to simulate time gaps for episode expiry.
    package func setLastMemoryTime(_ date: Date) { lastMemoryTime = date }
    #endif

    // MARK: - Tool Definitions

    public var definitions: [Tool] {
        [
            Tool(
                name: "remember",
                description: """
                    Store a memory for later recall. Keep each memory atomic and focused — \
                    one concept or fact per memory. For complex topics, create a brief hub \
                    memory first, then store details as child memories using `parent_id` to \
                    automatically link them via `part_of` edges. This enables precise recall \
                    and targeted updates. Memories are embedded as vectors for semantic search.
                    """,
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "content": .object([
                            "type": .string("string"),
                            "description": .string("The memory text to store. Keep it focused on a single concept — if you have multiple sections or topics, store them as separate memories using parent_id."),
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
                        "parent_id": .object([
                            "type": .string("string"),
                            "description": .string("UUID of a parent/hub memory. Automatically creates a part_of edge from this memory to the parent. Use to build hierarchies: store a brief hub, then add detail memories as children."),
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
                        "is_private": .object([
                            "type": .string("boolean"),
                            "description": .string("Mark this memory as private. Private memories sync to your cloud backup but are excluded from group shared graphs. Default: false."),
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
                            "type": .string("string"),
                            "description": .string("Delete a specific memory by UUID (from recall output [id:UUID] prefix). Takes priority over topic/project filters."),
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
                    Metadata updates (set_project, topic, source, expires_in_days) can combine with any content edit.
                    """,
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "id": .object([
                            "type": .string("string"),
                            "description": .string("Target memory by UUID (from recall output [id:UUID] prefix). Preferred over query."),
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
                        "set_project": .object([
                            "type": .string("string"),
                            "description": .string("Move the memory to a different project scope."),
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
                        "is_private": .object([
                            "type": .string("boolean"),
                            "description": .string("Update the memory's private flag. Private memories sync to your cloud backup but are excluded from group shared graphs. AUTHOR-ONLY on group-shared memories, and flipping to true RETRACTS the memory from the group — the group's copy (including teammates' edits) is removed for all members."),
                        ]),
                        "undelete": .object([
                            "type": .string("boolean"),
                            "description": .string("Restore a tombstoned (soft-deleted) memory — clears the tombstone for all group members."),
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
                            "items": .object(["type": .string("string")]),
                            "description": .string("Array of memory UUIDs to merge (from recall output)."),
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
                            "type": .string("string"),
                            "description": .string("Source memory UUID."),
                        ]),
                        "to": .object([
                            "type": .string("string"),
                            "description": .string("Target memory UUID."),
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
                            "type": .string("string"),
                            "description": .string("Delete a specific edge by its UUID."),
                        ]),
                        "from": .object([
                            "type": .string("string"),
                            "description": .string("Source memory UUID (used with 'to')."),
                        ]),
                        "to": .object([
                            "type": .string("string"),
                            "description": .string("Target memory UUID (used with 'from')."),
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
                            "type": .string("string"),
                            "description": .string("Memory UUID to explore from."),
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
                    Memories created after this call are linked to the episode via part_of edges \
                    until it is ended. If another episode is active, it will be closed first.
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
                    End an active episode. Optionally provide a summary of what happened \
                    (appended to the episode memory). If no episode_id is given, ends the \
                    currently active episode. To delete an episode, use forget(id:).
                    """,
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "episode_id": .object([
                            "type": .string("string"),
                            "description": .string("UUID of the episode to end. If omitted, ends the currently active episode."),
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
                    Retrieve an episode and its linked memories in chronological order. \
                    Use this to replay what happened during a focused work session.
                    """,
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "episode_id": .object([
                            "type": .string("string"),
                            "description": .string("UUID of the episode to recall (from list_episodes or begin_episode output)."),
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
                    List episodes, optionally filtered by project. \
                    Shows episodes sorted by most recent first. \
                    Episodes are memories with topic 'episode' — use forget(id:) to delete them.
                    """,
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "project": .object([
                            "type": .string("string"),
                            "description": .string("Filter to episodes in this project."),
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
                            "description": .string("Max cosine distance × 100 for cluster membership (default 15, i.e. 0.15). Lower = tighter clusters."),
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
                name: "detect_communities",
                description: """
                    Detect natural communities in a project's knowledge graph using label \
                    propagation. Read-only — shows which memories cluster together based on \
                    their edge connections. Useful for understanding memory structure before \
                    organizing. Memories need edges (from auto-connect or manual connect) to \
                    form communities.
                    """,
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "project": .object([
                            "type": .string("string"),
                            "description": .string("Project to analyze."),
                        ]),
                        "min_size": .object([
                            "type": .string("integer"),
                            "description": .string("Minimum community size to report (default 2)."),
                        ]),
                    ]),
                    "required": .array([.string("project")]),
                    "additionalProperties": .bool(false),
                ])
            ),
            Tool(
                name: "organize",
                description: """
                    Label a group of memories with a topic and create a hub. Takes memory IDs \
                    and a label — updates each memory's topic to the label, creates a hub memory, \
                    and links all members to the hub via part_of edges.
                    """,
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "ids": .object([
                            "type": .string("array"),
                            "items": .object(["type": .string("string")]),
                            "description": .string("Memory UUIDs to organize under this label."),
                        ]),
                        "label": .object([
                            "type": .string("string"),
                            "description": .string("Topic label for the group (e.g., 'editor-rendering', 'AI-pipeline')."),
                        ]),
                        "project": .object([
                            "type": .string("string"),
                            "description": .string("Project for the hub memory. If omitted, inferred from the first memory."),
                        ]),
                        "summary": .object([
                            "type": .string("string"),
                            "description": .string("Brief summary for the hub memory content. A short natural-language description of what the group covers improves recall. If omitted, defaults to 'Hub: {label}'."),
                        ]),
                    ]),
                    "required": .array([.string("ids"), .string("label")]),
                    "additionalProperties": .bool(false),
                ])
            ),
            Tool(
                name: "consolidate",
                description: """
                    Create a summary memory from a set of related memories. Originals are \
                    deprioritized (importance set to 0) and linked to the summary via \
                    'summarized_by' edges. The summary gets an embedding for future recall. \
                    Only consolidate memories with genuinely overlapping content — not \
                    memories that are merely topically related. Separate memories for \
                    distinct concerns give better recall precision.
                    """,
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([
                        "ids": .object([
                            "type": .string("array"),
                            "items": .object(["type": .string("string")]),
                            "description": .string("Memory UUIDs to consolidate (from recall or find_clusters output)."),
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
            Tool(
                name: "vacuum",
                description: """
                    Run database maintenance: checkpoint the WAL, vacuum the vector index \
                    to reclaim space from deleted entries, and compact the database. \
                    Use periodically or after large batch deletes/consolidations.
                    """,
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([:]),
                    "additionalProperties": .bool(false),
                ])
            ),
            Tool(
                name: "train_vectors",
                description: """
                    Build or rebuild the IVF vector search index. This trains k-means \
                    centroids for fast approximate nearest neighbor search. Run once after \
                    upgrading, or after large bulk inserts. Queries work without training \
                    (brute-force fallback) but are slower. Training takes ~1-5 seconds \
                    for 10K vectors.
                    """,
                inputSchema: .object([
                    "type": .string("object"),
                    "properties": .object([:]),
                    "additionalProperties": .bool(false),
                ])
            ),
        ]
    }

    // MARK: - Dispatch

    public func handle(_ params: CallTool.Parameters) async throws -> CallTool.Result {
        log("Tool call: \(params.name)")
        let result: CallTool.Result
        do {
            result = try await dispatch(params)
        } catch {
            log("Tool \(params.name) failed: \(error)")
            throw error
        }
        log("Tool \(params.name) completed")
        return result
    }

    private func dispatch(_ params: CallTool.Parameters) async throws -> CallTool.Result {
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
            return try await handleBeginEpisode(params.arguments)
        case "end_episode":
            return try handleEndEpisode(params.arguments)
        case "recall_episode":
            return try handleRecallEpisode(params.arguments)
        case "list_episodes":
            return try handleListEpisodes(params.arguments)
        case "find_clusters":
            return try await handleFindClusters(params.arguments)
        case "detect_communities":
            return try await handleDetectCommunities(params.arguments)
        case "organize":
            return try await handleOrganize(params.arguments)
        case "consolidate":
            return try await handleConsolidate(params.arguments)
        case "vacuum":
            return try handleVacuum()
        case "train_vectors":
            return try handleTrainVectors()
        default:
            throw MCPError.invalidParams("Unknown tool: \(params.name)")
        }
    }
}
