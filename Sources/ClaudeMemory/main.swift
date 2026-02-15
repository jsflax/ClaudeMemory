import ClaudeMemoryLib
import Lattice
import MCP
import Foundation

// MARK: - Configuration

/// Override the bundled embedding model with a custom path (optional).
let modelPath = ProcessInfo.processInfo.environment["CLAUDE_MEMORY_MODEL"]

/// Database location. Defaults to ~/.claude/memory.sqlite
let dbPath = ProcessInfo.processInfo.environment["CLAUDE_MEMORY_DB"]
    ?? NSHomeDirectory() + "/.claude/memory.sqlite"

// Ensure parent directory exists
let dbDir = (dbPath as NSString).deletingLastPathComponent
do {
    try FileManager.default.createDirectory(atPath: dbDir, withIntermediateDirectories: true)
} catch {
    log("Failed to create database directory at \(dbDir): \(error)")
    exit(1)
}

// MARK: - Init Lattice

let lattice: Lattice
do {
    lattice = try Lattice(Memory.self, Edge.self, Checkpoint.self, HookState.self, configuration: .init(fileURL: URL(fileURLWithPath: dbPath)))
} catch {
    log("Failed to initialize database at \(dbPath): \(error)")
    exit(1)
}
log("Database at \(dbPath)")

// MARK: - Init Embedding Service

let embedder = EmbeddingService(modelPath: modelPath)
await embedder.startLoading()

// MARK: - MCP Server

let tools = MemoryTools(lattice: lattice, embedder: embedder)

let server = Server(
    name: "memory",
    version: "1.0.0",
    instructions: """
        You have access to a persistent semantic memory system. Use it proactively — don't wait \
        to be asked.

        ## When to remember
        - User states a preference or convention ("I prefer tabs", "always use guard let")
        - You discover a non-obvious pattern, architecture decision, or debugging insight
        - A mistake is made that's worth avoiding next time
        - Key file paths, project structure, or integration details that took effort to find
        - Do NOT store trivial or easily re-discoverable facts (standard library APIs, etc.)

        ## Structuring memories
        Keep each memory **atomic** — one concept per memory. If you write content with multiple \
        sections or topics, the server will nudge you to decompose it into separate memories.

        For complex topics, create a brief **hub memory** first, then store details as children \
        using `parent_id` to automatically create `part_of` edges. This enables precise recall \
        and targeted updates.

        Key `remember` parameters:
        - **parent_id**: ID of a parent memory. Creates a `part_of` edge automatically. Use to \
        build hierarchies (hub + detail children).
        - **importance** (1-5): Boosts recall ranking by up to 20%. Use for critical knowledge \
        that should surface consistently. Omit for default priority.
        - **source**: Where the memory came from — `conversation`, `code-review`, \
        `debugging-session`, or a file path. Helps with provenance tracking.
        - **expires_in_days**: Auto-expire temporal context (see "Keeping memories clean").

        ## When to recall
        - At the START of every conversation: recall with the current project name to load context
        - Before making architectural decisions: check if prior decisions exist
        - Before exploring or researching any codebase or topic: check what you already know first
        - When the user references something from a past session
        - When you're unsure about a convention or preference

        ## Project vs Global
        - **Project-scoped** (e.g., project: "Lattice"): architecture, file paths, patterns, \
        decisions specific to that codebase. Recall with a project filter returns these + global.
        - **Global** (project: "global"): user preferences, cross-project conventions, workflow \
        preferences, tool configurations. Always included in project-scoped recalls.
        - When in doubt, use the project scope — it's better to be specific than to pollute global.

        ## Conflict detection
        `remember` checks for near-duplicates using both embedding distance AND term overlap \
        (Jaccard similarity). A memory is only blocked if it's close in embedding space (cosine \
        distance < 0.12 same-project, < 0.05 cross-scope) AND shares 40%+ of its terms with an \
        existing memory. This avoids false positives from topically similar but distinct memories. \
        If blocked, you'll see the existing memory and suggestions. To resolve:
        - Use `update` to modify the existing memory
        - Use `remember` with `force: true` to keep both (e.g., if they're related but distinct)
        - Use `forget` to remove the old one, then `remember` the new one

        ## Keeping memories clean
        - Use **update** (by id or similarity) to refine existing memories instead of \
        creating duplicates. Prefer targeting by `id` from recall output. Supports partial \
        edits: `append`, `prepend`, `find`+`replace`, and metadata-only updates (`topic`, \
        `source`, `importance`, `expires_in_days`) without rewriting full content
        - Use **organize** to re-topic multiple memories at once. Pass `ids` and a `label` \
        — updates all their topics, creates a hub memory, and links members via part_of edges. \
        **Always use organize instead of calling update in a loop to change topics.** \
        Pick descriptive labels (e.g., "editor-rendering", "AI-pipeline", "characters").
        - Use **merge** when you notice multiple memories about the same topic — consolidate \
        fragments into one well-written memory
        - Set **expires_in_days** for temporal context: "currently working on X", \
        "PR #42 needs review", "blocked on API migration". These auto-expire from recall results.
        - Use **forget** to remove memories that are wrong or no longer relevant

        ## Topics
        Use consistent topic names: "preferences", "architecture", "debugging", "patterns", \
        "conventions", "workflow", "dependencies". Custom topics are fine for project-specific \
        categories. When a generic topic grows large enough to benefit from subtopics, \
        break it up using **organize**.

        ## Recall output
        Each recalled memory includes an `[id:N]` prefix. Use these IDs for precise update, \
        merge, and forget operations. Expiring memories also show their expiration date.

        ## Discovery & analytics
        - **stats**: Memory counts grouped by project and topic. Use to understand memory \
        distribution.
        - **list_topics**: All topics with their memory counts. Use to check topic consistency.
        - **timeline**: Chronological view of memories grouped by day, week, or month. Pure \
        chronological — no semantic search. Use to review what was stored over a time period.

        ## Knowledge Graph
        Memories can be connected with directed edges to form a knowledge graph. Use this to \
        represent relationships between ideas, track contradictions, and enable graph-based discovery.

        **Tools:**
        - **connect**(from, to, relation): Create a directed edge between two memories. \
        `from` and `to` are memory IDs (integers, NOT `from_id`/`to_id`). \
        Relation types: `relates_to`, `contradicts`, `supersedes`, `derived_from`, `part_of`, \
        `summarized_by`. Duplicate edges are idempotent.
        - **disconnect**(id | from+to): Remove an edge by edge `id`, or by `from` and `to` \
        memory IDs. Optional `relation` filter.
        - **graph**(id, depth?): View a memory's neighborhood — shows connected memories up to \
        a given depth (default 1, max 3).
        - **recall** with `depth`: Set `depth: 1` (or up to 3) to follow edges from recalled \
        memories and surface connected knowledge. Default is 0 (no traversal).

        **When to connect — do this proactively:**
        - After **remember**: recall related memories and connect them. A new decision? \
        `supersedes` the old one. A detail about a system? `part_of` the overview. \
        Related context? `relates_to`.
        - When you discover a **contradiction**: connect with `contradicts` rather than \
        forgetting one — keeps both on record.
        - When a memory was **derived** from another (e.g., a summary, a conclusion): \
        `derived_from`.

        **When to use depth in recall:**
        - At conversation start, use `depth: 1` to get richer context from the graph.
        - When investigating a topic that may have non-obvious connections.

        **Automatic behavior:**
        - Edges are cleaned up when memories are deleted (forget) or merged.
        - Project is a soft ranking signal in recall — same-project memories rank higher, but \
        cross-project results still surface if semantically relevant.

        ## Task Continuity
        Use **checkpoint**, **resume**, and **list_tasks** to save and restore work-in-progress \
        state across sessions. Tasks are identified by `[task:N]` IDs (distinct from memory `[id:N]`).

        **When to checkpoint:**
        - At the END of a session when work is unfinished — save plan, progress, and context
        - After completing a significant milestone within a task
        - When the user asks to pause or switch tasks
        - Before any operation that might lose context (e.g., switching projects)

        **When to resume:**
        - At the START of a conversation: use `list_tasks` to check for active/paused work
        - When the user says "continue", "pick up where we left off", or references a previous task
        - After loading task state, use the plan/progress/context to orient yourself

        **Best practices:**
        - Keep **plan** structured (numbered steps, checkboxes)
        - Keep **progress** updated with what's done and what's next
        - Put file paths, key decisions, and blockers in **context**
        - Mark tasks **completed** when done, **paused** when shelving
        - Use project scoping to organize tasks by codebase

        ## Episodic Memory
        Use **begin_episode**, **end_episode**, **recall_episode**, and **list_episodes** to group \
        memories into narrative sessions. Episodes answer "what happened during X?" rather than \
        individual facts. Episodes are **opt-in** — memories are not automatically grouped.

        **When to begin an episode:**
        - Debugging sessions — chasing a bug across multiple files/hypotheses
        - Multi-step feature implementations — where context builds over many remembers
        - Code reviews or refactors with a clear narrative arc
        - Any focused work where the user might later ask "what happened when we did X?"
        - Do NOT begin episodes for routine Q&A, single-memory interactions, or trivial tasks

        **When to end an episode:**
        - When the focused work session wraps up (fix found, feature done, review complete)
        - Always provide a **summary** — this is the main value for future recall
        - Episodes auto-end after a 30-minute gap between remembers

        **When to recall an episode:**
        - When the user references a past session ("what happened when we debugged X?", \
        "continue where we left off")
        - Use `list_episodes` to find the right episode ID, then `recall_episode` for details

        **Best practices:**
        - Give episodes descriptive titles ("Debugging auth token expiry", not "Session")
        - Summaries should capture: what was attempted, what worked, what was decided
        - Episodes use `[episode:N]` IDs (distinct from `[id:N]` and `[task:N]`)

        ## Memory Consolidation
        Use **find_clusters** and **consolidate** to clean up redundant memories.

        **Workflow:**
        1. `find_clusters` with optional project/topic filters to discover similar memory groups
        2. Review the clusters — each shows member memories and suggested consolidation
        3. Write a concise summary that captures the essential knowledge from the cluster
        4. `consolidate` with the memory IDs and your written summary

        **What happens on consolidate:**
        - A new summary memory is created with your content and an embedding
        - Original memories get importance set to 0 (they still exist but rank lower in recall)
        - `summarized_by` edges link originals to the summary for graph traversal

        **When to consolidate:**
        - When `recall` returns many similar memories on the same topic
        - During periodic maintenance of a project's memory space
        - When you notice fragmented knowledge that would be better as one coherent memory

        **When NOT to consolidate:**
        - When memories cover different subsystems/concerns of the same project — even if semantically similar, separate memories give better recall precision
        - When memories are long, detailed references for distinct topics (e.g., "CI/CD" vs "recall algorithm")
        - Semantic similarity ≠ redundancy. Two memories about the same project's architecture can be very similar in embedding space while covering completely different things

        ## Auto-Organization
        Memories are automatically organized when possible:
        - **At remember-time**: if your new memory auto-connects to neighbors that share a hub, \
        it's automatically linked to that hub. If neighbors agree on a topic and you used the \
        default, their topic is inherited.
        - **detect_communities**: Discover natural groups in the knowledge graph via label \
        propagation. Read-only — shows which memories cluster together based on edge connections.
        - **organize**(ids, label): Batch re-topic memories. Use this instead of calling update \
        in a loop. Also creates a hub memory and part_of edges for graph structure.
        """,
    capabilities: .init(tools: .init(listChanged: false))
)

await server.withMethodHandler(ListTools.self) { _ in
    await ListTools.Result(tools: tools.definitions)
}

await server.withMethodHandler(CallTool.self) { params in
    do {
        return try await tools.handle(params)
    } catch let error as MCPError {
        throw error // MCP errors are expected — let the server return them to the client
    } catch {
        log("Unexpected error handling tool '\(params.name)': \(error)")
        return CallTool.Result(
            content: [.text("Internal error: \(error.localizedDescription)")],
            isError: true
        )
    }
}

let transport = StdioTransport()
do {
    try await server.start(transport: transport)
    log("Server started")
} catch {
    log("Server transport error: \(error)")
    exit(1)
}

// Keep alive
await server.waitUntilCompleted()
