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
try FileManager.default.createDirectory(atPath: dbDir, withIntermediateDirectories: true)

// MARK: - Init Lattice

let lattice = try Lattice(Memory.self, Edge.self, Checkpoint.self, Episode.self, configuration: .init(fileURL: URL(fileURLWithPath: dbPath)))
log("Database at \(dbPath)")

// MARK: - Init Embedding Service

let embedder = EmbeddingService(modelPath: modelPath)
await embedder.load()

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

        ## When to recall
        - At the START of every conversation: recall with the current project name to load context
        - Before making architectural decisions: check if prior decisions exist
        - When the user references something from a past session
        - When you're unsure about a convention or preference

        ## Project vs Global
        - **Project-scoped** (e.g., project: "Lattice"): architecture, file paths, patterns, \
        decisions specific to that codebase. Recall with a project filter returns these + global.
        - **Global** (project: "global"): user preferences, cross-project conventions, workflow \
        preferences, tool configurations. Always included in project-scoped recalls.
        - When in doubt, use the project scope — it's better to be specific than to pollute global.

        ## Conflict detection
        `remember` automatically checks for near-duplicate memories before storing. \
        Same-project matches use cosine distance < 0.12; cross-scope matches (global↔project) \
        use a tighter threshold of 0.05. If a near-duplicate is found, the memory is NOT \
        stored — instead you'll see the existing memory and suggestions. To resolve:
        - Use `update` to modify the existing memory
        - Use `remember` with `force: true` to keep both (e.g., if they're related but distinct)
        - Use `forget` to remove the old one, then `remember` the new one

        ## Keeping memories clean
        - Use **update** (by id or similarity) to refine existing memories instead of \
        creating duplicates. Prefer targeting by `id` from recall output. Supports partial \
        edits: `append`, `prepend`, `find`+`replace`, and metadata-only updates (`topic`, \
        `source`, `expires_in_days`) without rewriting full content
        - Use **merge** when you notice multiple memories about the same topic — consolidate \
        fragments into one well-written memory
        - Set **expires_in_days** for temporal context: "currently working on X", \
        "PR #42 needs review", "blocked on API migration". These auto-expire from recall results.
        - Use **forget** to remove memories that are wrong or no longer relevant

        ## Topics
        Use consistent topic names: "preferences", "architecture", "debugging", "patterns", \
        "conventions", "workflow", "dependencies". Custom topics are fine for project-specific \
        categories.

        ## Recall output
        Each recalled memory includes an `[id:N]` prefix. Use these IDs for precise update, \
        merge, and forget operations. Expiring memories also show their expiration date.

        ## Knowledge Graph
        Memories can be connected with directed edges to form a knowledge graph. Use this to \
        represent relationships between ideas, track contradictions, and enable graph-based discovery.

        **Tools:**
        - **connect**: Create an edge between two memories. Relation types: `relates_to`, \
        `contradicts`, `supersedes`, `derived_from`, `part_of`, `summarized_by`. Duplicate edges \
        are idempotent.
        - **disconnect**: Remove edges by edge ID or by (from, to) memory pair.
        - **graph**: View a memory's neighborhood — shows connected memories up to a given depth.
        - **recall with depth**: Set `depth: 1` (or up to 3) to follow edges from recalled \
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
        individual facts.

        **Auto-episode**: You don't have to explicitly begin episodes. The first `remember` in a \
        session auto-creates an episode. If there's a >30 minute gap between remembers, a new \
        episode starts automatically. Explicit `begin_episode` overrides auto-episodes.

        **When to begin an episode explicitly:**
        - Starting a focused work session (debugging, feature implementation, code review)
        - When you want a descriptive title instead of "Session: {date}"

        **When to end an episode:**
        - At the end of a focused work session
        - Provide a summary of what was accomplished for future recall
        - Episodes auto-end when a new one starts (explicitly or via time gap)

        **When to recall an episode:**
        - When the user references a past session ("what happened when we debugged X?")
        - Use `list_episodes` to find the right episode ID, then `recall_episode` for details

        **Best practices:**
        - Let auto-episodes handle casual sessions
        - Use explicit episodes for important work sessions
        - Provide summaries at end_episode for quick future reference
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
        """,
    capabilities: .init(tools: .init(listChanged: false))
)

await server.withMethodHandler(ListTools.self) { _ in
    await ListTools.Result(tools: tools.definitions)
}

await server.withMethodHandler(CallTool.self) { params in
    try await tools.handle(params)
}

let transport = StdioTransport()
try await server.start(transport: transport)
log("Server started")

// Keep alive
await server.waitUntilCompleted()
