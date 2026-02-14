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

let lattice = try Lattice(Memory.self, Edge.self, configuration: .init(fileURL: URL(fileURLWithPath: dbPath)))
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
        `contradicts`, `supersedes`, `derived_from`, `part_of`. Duplicate edges are idempotent.
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
