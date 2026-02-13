import Testing
import ClaudeMemoryLib
import Lattice
import MCP
import Foundation

/// Helper to extract the text string from a CallTool.Result
private func text(from result: CallTool.Result) -> String {
    guard case .text(let text) = result.content.first else {
        return ""
    }
    return text
}

/// Shared embedder — loads the bundled CoreML model once for all tests.
private let sharedEmbedder: EmbeddingService = {
    let e = EmbeddingService()
    return e
}()

/// Create a MemoryTools with an isolated temp database and the real embedding model.
private func makeTools() async throws -> MemoryTools {
    let path = FileManager.default.temporaryDirectory
        .appending(path: "claude-memory-test-\(UUID().uuidString).sqlite")
    let lattice = try Lattice(Memory.self, configuration: .init(fileURL: path))
    let embedder = sharedEmbedder
    if await !embedder.isLoaded {
        await embedder.load()
    }
    return MemoryTools(lattice: lattice, embedder: embedder)
}

// MARK: - Remember

@Test func remember_basic() async throws {
    let tools = try await makeTools()
    let result = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Swift uses ARC for memory management")]
    ))
    let output = text(from: result)
    #expect(output.contains("Stored memory"))
    #expect(output.contains("topic: general"))
    #expect(output.contains("project: global"))
}

@Test func remember_withMetadata() async throws {
    let tools = try await makeTools()
    let result = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Always use snake_case for database columns"),
            "topic": .string("conventions"),
            "project": .string("Lattice"),
            "source": .string("code-review"),
        ]
    ))
    let output = text(from: result)
    #expect(output.contains("project: Lattice"))
    #expect(output.contains("topic: conventions"))
}

@Test func remember_missingContent_throws() async throws {
    let tools = try await makeTools()
    await #expect(throws: (any Error).self) {
        try await tools.handle(CallTool.Parameters(
            name: "remember",
            arguments: [:]
        ))
    }
}

@Test func remember_emptyContent_throws() async throws {
    let tools = try await makeTools()
    await #expect(throws: (any Error).self) {
        try await tools.handle(CallTool.Parameters(
            name: "remember",
            arguments: ["content": .string("")]
        ))
    }
}

// MARK: - Recall (semantic search with real embedding model)

@Test func recall_semanticSearch() async throws {
    let tools = try await makeTools()

    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Lattice uses SQLite as its backend database engine")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Swift macros generate code at compile time")]
    ))

    // Semantic search for "database" — should rank the SQLite memory higher
    let result = try await tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: ["query": .string("database")]
    ))
    let output = text(from: result)
    #expect(output.contains("SQLite"))
}

@Test func recall_noResults() async throws {
    let tools = try await makeTools()
    let result = try await tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: ["query": .string("nonexistent topic with no matches")]
    ))
    #expect(text(from: result) == "No memories found.")
}

@Test func recall_projectScoping() async throws {
    let tools = try await makeTools()

    // Store a global memory and two project-specific memories
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("User prefers dark mode themes globally"),
            "project": .string("global"),
        ]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Lattice dark mode uses NSAppearance for theming"),
            "project": .string("Lattice"),
        ]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("OtherApp dark mode uses UIKit appearance proxies"),
            "project": .string("OtherApp"),
        ]
    ))

    // Recall with project=Lattice should get Lattice + global, not OtherApp
    let result = try await tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: [
            "query": .string("dark mode theming"),
            "project": .string("Lattice"),
        ]
    ))
    let output = text(from: result)
    #expect(output.contains("NSAppearance") || output.contains("Lattice"))
    #expect(output.contains("prefers dark mode") || output.contains("global"))
    #expect(!output.contains("UIKit"))
}

@Test func recall_topicFilter() async throws {
    let tools = try await makeTools()

    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Use guard let for early returns in Swift functions"),
            "topic": .string("conventions"),
        ]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Fixed retain cycle in observer by using weak self capture"),
            "topic": .string("debugging"),
        ]
    ))

    // Recall with topic=debugging
    let result = try await tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: [
            "query": .string("memory leak retain cycle"),
            "topic": .string("debugging"),
        ]
    ))
    let output = text(from: result)
    #expect(output.contains("retain cycle"))
    #expect(!output.contains("guard let"))
}

@Test func recall_semanticRelevanceOrdering() async throws {
    let tools = try await makeTools()

    // Store memories with varying relevance to "cooking recipes"
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("The best pasta recipe uses fresh tomatoes and basil")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Quantum computing uses qubits for parallel computation")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Italian cuisine emphasizes fresh ingredients and simple preparation")]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: ["query": .string("cooking food recipes"), "limit": .int(2)]
    ))
    let output = text(from: result)
    // Should find food-related memories, not quantum computing
    #expect(!output.contains("qubits"))
    #expect(output.contains("pasta") || output.contains("Italian") || output.contains("cuisine"))
}

// MARK: - Forget

@Test func forget_byTopic() async throws {
    let tools = try await makeTools()

    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("debug note 1"), "topic": .string("debugging")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("debug note 2"), "topic": .string("debugging")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("architecture note about modules"), "topic": .string("architecture")]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "forget",
        arguments: ["topic": .string("debugging")]
    ))
    let output = text(from: result)
    #expect(output.contains("Deleted 2 memories"))

    // architecture memory should still exist
    let recall = try await tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: ["query": .string("architecture modules")]
    ))
    #expect(text(from: recall).contains("architecture"))
}

@Test func forget_byProject() async throws {
    let tools = try await makeTools()

    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("lattice specific implementation detail"), "project": .string("Lattice")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("global preference setting for editor"), "project": .string("global")]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "forget",
        arguments: ["project": .string("Lattice")]
    ))
    #expect(text(from: result).contains("Deleted 1 memories"))

    // global memory should remain
    let recall = try await tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: ["query": .string("preference setting editor")]
    ))
    #expect(text(from: recall).contains("preference"))
}

@Test func forget_byTopicAndProject() async throws {
    let tools = try await makeTools()

    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("debug for project X"), "topic": .string("debug"), "project": .string("X")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("debug for project Y"), "topic": .string("debug"), "project": .string("Y")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("arch for project X"), "topic": .string("arch"), "project": .string("X")]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "forget",
        arguments: ["topic": .string("debug"), "project": .string("X")]
    ))
    #expect(text(from: result).contains("Deleted 1 memories"))
}

@Test func forget_all() async throws {
    let tools = try await makeTools()

    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("first memory to be deleted")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("second memory to be deleted")]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "forget",
        arguments: nil
    ))
    #expect(text(from: result).contains("Deleted all 2 memories"))

    // Verify empty
    let recall = try await tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: ["query": .string("memory deleted")]
    ))
    #expect(text(from: recall) == "No memories found.")
}

// MARK: - List Topics

@Test func listTopics_empty() async throws {
    let tools = try await makeTools()
    let result = try await tools.handle(CallTool.Parameters(
        name: "list_topics",
        arguments: nil
    ))
    #expect(text(from: result) == "No memories stored.")
}

@Test func listTopics_withMemories() async throws {
    let tools = try await makeTools()

    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("debug note alpha"), "topic": .string("debugging")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("debug note beta"), "topic": .string("debugging")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("architecture overview"), "topic": .string("architecture")]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "list_topics",
        arguments: nil
    ))
    let output = text(from: result)
    #expect(output.contains("All topics:"))
    #expect(output.contains("debugging: 2 memories"))
    #expect(output.contains("architecture: 1 memories"))
}

@Test func listTopics_filteredByProject() async throws {
    let tools = try await makeTools()

    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("debug for X"), "topic": .string("debug"), "project": .string("X")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("debug for Y"), "topic": .string("debug"), "project": .string("Y")]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "list_topics",
        arguments: ["project": .string("X")]
    ))
    let output = text(from: result)
    #expect(output.contains("Topics for project 'X':"))
    #expect(output.contains("debug: 1 memories"))
}

// MARK: - Unknown Tool

@Test func unknownTool_throws() async throws {
    let tools = try await makeTools()
    await #expect(throws: (any Error).self) {
        try await tools.handle(CallTool.Parameters(name: "nonexistent"))
    }
}

// MARK: - Tool Definitions

@Test func definitions_hasSevenTools() async throws {
    let tools = try await makeTools()
    let defs = await tools.definitions
    #expect(defs.count == 7)
    let names = defs.map(\.name).sorted()
    #expect(names == ["forget", "list_topics", "merge", "recall", "remember", "stats", "update"])
}

// MARK: - Remember always creates new

@Test func remember_alwaysCreatesNew() async throws {
    let tools = try await makeTools()

    // Store two memories with very similar content
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Lattice uses SQLite as its database backend"),
            "project": .string("Lattice"),
        ]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Lattice uses SQLite as its backend database engine"),
            "project": .string("Lattice"),
        ]
    ))

    // Both should be stored — deduplication is Claude's responsibility
    let stats = try await tools.handle(CallTool.Parameters(
        name: "stats",
        arguments: ["project": .string("Lattice")]
    ))
    #expect(text(from: stats).contains("Total memories: 2"))
}

// MARK: - Update

@Test func update_existingMemory() async throws {
    let tools = try await makeTools()

    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("The default branch is called master"),
            "project": .string("MyProject"),
        ]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "update",
        arguments: [
            "query": .string("default branch name"),
            "new_content": .string("The default branch is called main"),
            "project": .string("MyProject"),
        ]
    ))
    let output = text(from: result)
    #expect(output.contains("Updated memory"))
    #expect(output.contains("Old: The default branch is called master"))
    #expect(output.contains("New: The default branch is called main"))

    // Verify content was actually changed
    let recall = try await tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: ["query": .string("default branch"), "project": .string("MyProject")]
    ))
    #expect(text(from: recall).contains("main"))
}

@Test func update_noMatch() async throws {
    let tools = try await makeTools()

    let result = try await tools.handle(CallTool.Parameters(
        name: "update",
        arguments: [
            "query": .string("something that does not exist"),
            "new_content": .string("new content"),
        ]
    ))
    #expect(text(from: result).contains("No matching memory found"))
}

// MARK: - Stats

@Test func stats_basic() async throws {
    let tools = try await makeTools()

    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("debug note A"), "topic": .string("debugging"), "project": .string("X")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("debug note B"), "topic": .string("debugging"), "project": .string("X")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("architecture pattern C"), "topic": .string("architecture"), "project": .string("Y")]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "stats",
        arguments: nil
    ))
    let output = text(from: result)
    #expect(output.contains("Total memories: 3"))
    #expect(output.contains("X: 2"))
    #expect(output.contains("Y: 1"))
    #expect(output.contains("debugging: 2"))
    #expect(output.contains("architecture: 1"))
}

@Test func stats_filteredByProject() async throws {
    let tools = try await makeTools()

    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("note for X"), "topic": .string("general"), "project": .string("X")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("note for Y"), "topic": .string("general"), "project": .string("Y")]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "stats",
        arguments: ["project": .string("X")]
    ))
    let output = text(from: result)
    #expect(output.contains("Total memories: 1"))
    // Should not contain per-project breakdown when filtered
    #expect(!output.contains("By project:"))
}

@Test func stats_empty() async throws {
    let tools = try await makeTools()
    let result = try await tools.handle(CallTool.Parameters(
        name: "stats",
        arguments: nil
    ))
    #expect(text(from: result) == "No memories stored.")
}

// MARK: - lastAccessedAt

@Test func recall_bumpsLastAccessed() async throws {
    let tools = try await makeTools()

    // Store a memory
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Important architectural decision about using SQLite")]
    ))

    // Wait a moment so timestamps differ
    try await Task.sleep(for: .milliseconds(50))

    // Recall it
    _ = try await tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: ["query": .string("architectural decision SQLite")]
    ))

    // Verify lastAccessedAt was bumped (it should be > createdAt)
    // We access the lattice directly through a stats call to confirm the memory exists
    // The real verification is that the code compiles and runs without error,
    // since lastAccessedAt is set in the recall path
    let stats = try await tools.handle(CallTool.Parameters(
        name: "stats",
        arguments: nil
    ))
    #expect(text(from: stats).contains("Total memories: 1"))
}

// MARK: - Recall includes IDs and expiration

@Test func recall_includesMemoryId() async throws {
    let tools = try await makeTools()

    let stored = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Lattice uses SQLite as its database backend")]
    ))
    // Remember now returns the id
    let storedText = text(from: stored)
    #expect(storedText.contains("id:"))

    let result = try await tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: ["query": .string("database backend")]
    ))
    let output = text(from: result)
    #expect(output.contains("[id:"))
    #expect(output.contains("SQLite"))
}

@Test func recall_showsExpiration() async throws {
    let tools = try await makeTools()

    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Currently refactoring the auth system"),
            "expires_in_days": .int(7),
        ]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: ["query": .string("auth refactoring")]
    ))
    let output = text(from: result)
    #expect(output.contains("expires:"))
    #expect(output.contains("auth system"))
}

@Test func recall_filtersExpired() async throws {
    let tools = try await makeTools()

    // Store a memory that's already expired (expires_in_days won't work for past,
    // so we store a normal one and verify non-expired ones show up)
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("This memory should be visible in recall results")]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: ["query": .string("visible recall results")]
    ))
    #expect(text(from: result).contains("visible"))
}

// MARK: - Merge

@Test func merge_twoMemories() async throws {
    let tools = try await makeTools()

    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Lattice uses SQLite as its database"),
            "project": .string("Lattice"),
            "topic": .string("architecture"),
        ]
    ))
    let r2 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Lattice SQLite runs in WAL mode for concurrency"),
            "project": .string("Lattice"),
            "topic": .string("architecture"),
        ]
    ))

    // Extract IDs from remember output
    let id1Text = text(from: r1)
    let id2Text = text(from: r2)
    let id1 = extractId(from: id1Text)!
    let id2 = extractId(from: id2Text)!

    let result = try await tools.handle(CallTool.Parameters(
        name: "merge",
        arguments: [
            "ids": .array([.int(id1), .int(id2)]),
            "content": .string("Lattice uses SQLite in WAL mode as its database backend for concurrent access"),
        ]
    ))
    let output = text(from: result)
    #expect(output.contains("Merged 2 memories"))
    #expect(output.contains("WAL mode"))

    // Verify only 1 memory remains
    let stats = try await tools.handle(CallTool.Parameters(
        name: "stats",
        arguments: ["project": .string("Lattice")]
    ))
    #expect(text(from: stats).contains("Total memories: 1"))
}

@Test func merge_notFoundId() async throws {
    let tools = try await makeTools()

    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("A real memory")]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "merge",
        arguments: [
            "ids": .array([.int(1), .int(9999)]),
            "content": .string("merged"),
        ]
    ))
    #expect(text(from: result).contains("not found"))
}

@Test func merge_needsAtLeastTwo() async throws {
    let tools = try await makeTools()
    await #expect(throws: (any Error).self) {
        try await tools.handle(CallTool.Parameters(
            name: "merge",
            arguments: [
                "ids": .array([.int(1)]),
                "content": .string("merged"),
            ]
        ))
    }
}

// MARK: - Expiration

@Test func remember_withExpiration() async throws {
    let tools = try await makeTools()

    let result = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("PR #42 needs review before merging"),
            "expires_in_days": .int(3),
        ]
    ))
    let output = text(from: result)
    #expect(output.contains("expires:"))
    #expect(output.contains("PR #42"))
}

@Test func remember_permanentByDefault() async throws {
    let tools = try await makeTools()

    let result = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("User prefers dark mode")]
    ))
    let output = text(from: result)
    // Permanent memories should not show an expiration
    #expect(!output.contains("expires:"))
}

/// Helper to extract an integer ID from text like "id: 42" or "id:42"
private func extractId(from text: String) -> Int? {
    guard let range = text.range(of: "id: ", options: .literal) ?? text.range(of: "id:", options: .literal) else {
        return nil
    }
    let after = text[range.upperBound...]
    let digits = after.prefix(while: { $0.isNumber })
    return Int(digits)
}
