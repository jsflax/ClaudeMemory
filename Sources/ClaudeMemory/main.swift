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

let lattice = try Lattice(Memory.self, configuration: .init(fileURL: URL(fileURLWithPath: dbPath)))
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

        ## Keeping memories clean
        - Use **recall** before **remember** to check if similar knowledge already exists
        - Use **update** (by similarity or by id from recall) to refine existing memories \
        instead of creating duplicates
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
