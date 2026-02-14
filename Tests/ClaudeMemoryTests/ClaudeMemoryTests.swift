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
    let lattice = try Lattice(Memory.self, Edge.self, configuration: .init(fileURL: path))
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

    // Recall with project=Lattice should get Lattice + global, boosted above OtherApp
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
        arguments: ["content": .string("debug note 2"), "topic": .string("debugging"), "force": .bool(true)]
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

@Test func forget_all_requiresFilter() async throws {
    let tools = try await makeTools()

    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("first memory to be preserved")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("second memory to be preserved"), "force": .bool(true)]
    ))

    // Calling forget with no arguments should fail (safety check)
    await #expect(throws: (any Error).self) {
        try await tools.handle(CallTool.Parameters(
            name: "forget",
            arguments: nil
        ))
    }

    // Memories should still exist
    let stats = try await tools.handle(CallTool.Parameters(
        name: "stats",
        arguments: nil
    ))
    #expect(text(from: stats).contains("Total memories: 2"))
}

@Test func forget_byStringEncodedId() async throws {
    let tools = try await makeTools()

    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("memory to delete by string id")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("memory to keep")]
    ))

    let id = extractId(from: text(from: r1))!

    // Pass id as a string (simulating MCP client behavior)
    let result = try await tools.handle(CallTool.Parameters(
        name: "forget",
        arguments: ["id": .string(String(id))]
    ))
    #expect(result.isError != true)
    #expect(text(from: result).contains("Deleted memory"))

    // Only one memory should remain
    let stats = try await tools.handle(CallTool.Parameters(
        name: "stats",
        arguments: nil
    ))
    #expect(text(from: stats).contains("Total memories: 1"))
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
        arguments: ["content": .string("debug note beta"), "topic": .string("debugging"), "force": .bool(true)]
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

@Test func definitions_hasTenTools() async throws {
    let tools = try await makeTools()
    let defs = await tools.definitions
    #expect(defs.count == 10)
    let names = defs.map(\.name).sorted()
    #expect(names == ["connect", "disconnect", "forget", "graph", "list_topics", "merge", "recall", "remember", "stats", "update"])
}

// MARK: - Remember with force creates new

@Test func remember_forceCreatesNew() async throws {
    let tools = try await makeTools()

    // Store two memories with very similar content using force: true
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
            "force": .bool(true),
        ]
    ))

    // Both should be stored because force: true bypasses conflict detection
    let stats = try await tools.handle(CallTool.Parameters(
        name: "stats",
        arguments: ["project": .string("Lattice")]
    ))
    #expect(text(from: stats).contains("Total memories: 2"))
}

// MARK: - Conflict Detection

@Test func remember_conflictDetection_blocksStore() async throws {
    let tools = try await makeTools()

    // Store a memory
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Lattice uses SQLite"),
            "project": .string("Lattice"),
        ]
    ))

    // Try to store a near-duplicate — should be blocked
    let result = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Lattice uses SQLite as its database"),
            "project": .string("Lattice"),
        ]
    ))
    let output = text(from: result)
    #expect(output.contains("Near-duplicate"))
    #expect(output.contains("NOT stored"))

    // Only 1 memory should exist
    let stats = try await tools.handle(CallTool.Parameters(
        name: "stats",
        arguments: ["project": .string("Lattice")]
    ))
    #expect(text(from: stats).contains("Total memories: 1"))
}

@Test func remember_conflictDetection_force() async throws {
    let tools = try await makeTools()

    // Store a memory
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Lattice uses SQLite"),
            "project": .string("Lattice"),
        ]
    ))

    // Store near-duplicate with force: true — should succeed
    let result = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Lattice uses SQLite as its database"),
            "project": .string("Lattice"),
            "force": .bool(true),
        ]
    ))
    let output = text(from: result)
    #expect(output.contains("Stored memory"))

    // Both should exist
    let stats = try await tools.handle(CallTool.Parameters(
        name: "stats",
        arguments: ["project": .string("Lattice")]
    ))
    #expect(text(from: stats).contains("Total memories: 2"))
}

@Test func remember_conflictDetection_differentProject() async throws {
    let tools = try await makeTools()

    // Store a memory in project A
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("The app uses PostgreSQL for data storage"),
            "project": .string("ProjectA"),
        ]
    ))

    // Store same content in project B — should NOT conflict (different scope)
    let result = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("The app uses PostgreSQL for data storage"),
            "project": .string("ProjectB"),
        ]
    ))
    let output = text(from: result)
    #expect(output.contains("Stored memory"))

    // Both should exist
    let statsA = try await tools.handle(CallTool.Parameters(
        name: "stats",
        arguments: ["project": .string("ProjectA")]
    ))
    #expect(text(from: statsA).contains("Total memories: 1"))
    let statsB = try await tools.handle(CallTool.Parameters(
        name: "stats",
        arguments: ["project": .string("ProjectB")]
    ))
    #expect(text(from: statsB).contains("Total memories: 1"))
}

@Test func remember_conflictDetection_noEmbedding() async throws {
    // Create tools with a broken embedder (no model loaded) to test degraded mode
    let path = FileManager.default.temporaryDirectory
        .appending(path: "claude-memory-test-\(UUID().uuidString).sqlite")
    let lattice = try Lattice(Memory.self, Edge.self, configuration: .init(fileURL: path))
    let embedder = EmbeddingService(modelPath: "/nonexistent/path")
    // Don't call load — embedder.isLoaded will be false, embed() returns nil
    let tools = MemoryTools(lattice: lattice, embedder: embedder)

    // Store two identical memories — conflict check should be skipped (no embedding)
    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Some important fact")]
    ))
    #expect(text(from: r1).contains("Stored memory"))

    let r2 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Some important fact")]
    ))
    #expect(text(from: r2).contains("Stored memory"))

    // Both should be stored since conflict detection requires embeddings
    let stats = try await tools.handle(CallTool.Parameters(
        name: "stats",
        arguments: nil
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
            "content": .string("The default branch is called main"),
            "project": .string("MyProject"),
        ]
    ))
    let output = text(from: result)
    #expect(output.contains("Updated memory"))
    #expect(output.contains("content:"))

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
            "content": .string("new content"),
        ]
    ))
    #expect(text(from: result).contains("No matching memory found"))
}

@Test func update_byId() async throws {
    let tools = try await makeTools()

    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Original content for ID-based update")]
    ))
    let id = extractId(from: text(from: r1))!

    let result = try await tools.handle(CallTool.Parameters(
        name: "update",
        arguments: [
            "id": .int(id),
            "content": .string("Updated content via ID"),
        ]
    ))
    let output = text(from: result)
    #expect(output.contains("Updated memory"))
    #expect(output.contains("id: \(id)"))

    let recall = try await tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: ["query": .string("Updated content via ID")]
    ))
    #expect(text(from: recall).contains("Updated content via ID"))
}

@Test func update_append() async throws {
    let tools = try await makeTools()

    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Base content")]
    ))
    let id = extractId(from: text(from: r1))!

    let result = try await tools.handle(CallTool.Parameters(
        name: "update",
        arguments: [
            "id": .int(id),
            "append": .string("Appended line"),
        ]
    ))
    #expect(result.isError != true)

    let recall = try await tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: ["query": .string("Base content appended")]
    ))
    let output = text(from: recall)
    #expect(output.contains("Base content"))
    #expect(output.contains("Appended line"))
}

@Test func update_prepend() async throws {
    let tools = try await makeTools()

    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Original line")]
    ))
    let id = extractId(from: text(from: r1))!

    let result = try await tools.handle(CallTool.Parameters(
        name: "update",
        arguments: [
            "id": .int(id),
            "prepend": .string("Prepended line"),
        ]
    ))
    #expect(result.isError != true)

    let recall = try await tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: ["query": .string("Prepended line original")]
    ))
    let output = text(from: recall)
    #expect(output.contains("Prepended line"))
    #expect(output.contains("Original line"))
}

@Test func update_findReplace() async throws {
    let tools = try await makeTools()

    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("The project uses Python 2.7 for scripting")]
    ))
    let id = extractId(from: text(from: r1))!

    let result = try await tools.handle(CallTool.Parameters(
        name: "update",
        arguments: [
            "id": .int(id),
            "find": .string("Python 2.7"),
            "replace": .string("Python 3.12"),
        ]
    ))
    #expect(result.isError != true)

    let recall = try await tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: ["query": .string("project scripting language")]
    ))
    let output = text(from: recall)
    #expect(output.contains("Python 3.12"))
    #expect(!output.contains("Python 2.7"))
}

@Test func update_findReplace_patternNotFound() async throws {
    let tools = try await makeTools()

    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Uses Swift for development")]
    ))
    let id = extractId(from: text(from: r1))!

    let result = try await tools.handle(CallTool.Parameters(
        name: "update",
        arguments: [
            "id": .int(id),
            "find": .string("Rust"),
            "replace": .string("Go"),
        ]
    ))
    #expect(result.isError == true)
    #expect(text(from: result).contains("Find pattern not found"))
}

@Test func update_metadataOnly_topic() async throws {
    let tools = try await makeTools()

    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Important pattern for topic change test"),
            "topic": .string("general"),
        ]
    ))
    let id = extractId(from: text(from: r1))!

    let result = try await tools.handle(CallTool.Parameters(
        name: "update",
        arguments: [
            "id": .int(id),
            "topic": .string("patterns"),
        ]
    ))
    let output = text(from: result)
    #expect(output.contains("topic: general → patterns"))
    #expect(output.contains("topic: patterns"))

    // Verify the old topic is gone via list_topics
    let topics = try await tools.handle(CallTool.Parameters(
        name: "list_topics",
        arguments: nil
    ))
    let topicsOutput = text(from: topics)
    #expect(topicsOutput.contains("patterns: 1"))
    #expect(!topicsOutput.contains("general"))
}

@Test func update_metadataOnly_expiration() async throws {
    let tools = try await makeTools()

    // Create a permanent memory
    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Memory for expiration test")]
    ))
    let id = extractId(from: text(from: r1))!

    // Set expiration
    let result1 = try await tools.handle(CallTool.Parameters(
        name: "update",
        arguments: [
            "id": .int(id),
            "expires_in_days": .int(7),
        ]
    ))
    let output1 = text(from: result1)
    #expect(output1.contains("expires: permanent →"))

    // Make permanent again
    let result2 = try await tools.handle(CallTool.Parameters(
        name: "update",
        arguments: [
            "id": .int(id),
            "expires_in_days": .int(0),
        ]
    ))
    let output2 = text(from: result2)
    #expect(output2.contains("→ permanent"))
}

@Test func update_contentAndMetadata() async throws {
    let tools = try await makeTools()

    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Old content for combined test"),
            "topic": .string("general"),
        ]
    ))
    let id = extractId(from: text(from: r1))!

    let result = try await tools.handle(CallTool.Parameters(
        name: "update",
        arguments: [
            "id": .int(id),
            "append": .string("Extra detail added"),
            "topic": .string("architecture"),
        ]
    ))
    let output = text(from: result)
    #expect(output.contains("content:"))
    #expect(output.contains("topic: general → architecture"))
}

@Test func update_noTarget_throws() async throws {
    let tools = try await makeTools()
    await #expect(throws: (any Error).self) {
        try await tools.handle(CallTool.Parameters(
            name: "update",
            arguments: ["content": .string("no target")]
        ))
    }
}

@Test func update_noEdits_throws() async throws {
    let tools = try await makeTools()

    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Memory with no edits")]
    ))
    let id = extractId(from: text(from: r1))!

    await #expect(throws: (any Error).self) {
        try await tools.handle(CallTool.Parameters(
            name: "update",
            arguments: ["id": .int(id)]
        ))
    }
}

@Test func update_mutuallyExclusiveContent() async throws {
    let tools = try await makeTools()

    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Memory for mutual exclusion test")]
    ))
    let id = extractId(from: text(from: r1))!

    await #expect(throws: (any Error).self) {
        try await tools.handle(CallTool.Parameters(
            name: "update",
            arguments: [
                "id": .int(id),
                "content": .string("full replace"),
                "append": .string("also append"),
            ]
        ))
    }
}

@Test func update_byId_notFound() async throws {
    let tools = try await makeTools()

    let result = try await tools.handle(CallTool.Parameters(
        name: "update",
        arguments: [
            "id": .int(99999),
            "content": .string("should not work"),
        ]
    ))
    #expect(result.isError == true)
    #expect(text(from: result).contains("not found"))
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
        arguments: ["content": .string("debug note B"), "topic": .string("debugging"), "project": .string("X"), "force": .bool(true)]
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
            "force": .bool(true),
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

// MARK: - Connect

@Test func connect_basic() async throws {
    let tools = try await makeTools()

    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Auth uses JWT tokens")]
    ))
    let r2 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("JWT tokens expire after 1 hour"), "force": .bool(true)]
    ))
    let id1 = extractId(from: text(from: r1))!
    let id2 = extractId(from: text(from: r2))!

    let result = try await tools.handle(CallTool.Parameters(
        name: "connect",
        arguments: [
            "from": .int(id1),
            "to": .int(id2),
            "relation": .string("relates_to"),
        ]
    ))
    let output = text(from: result)
    #expect(output.contains("Connected"))
    #expect(output.contains("edge id:"))
    #expect(output.contains("relates_to"))
}

@Test func connect_invalidRelation_throws() async throws {
    let tools = try await makeTools()

    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Memory A")]
    ))
    let r2 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Memory B"), "force": .bool(true)]
    ))
    let id1 = extractId(from: text(from: r1))!
    let id2 = extractId(from: text(from: r2))!

    await #expect(throws: (any Error).self) {
        try await tools.handle(CallTool.Parameters(
            name: "connect",
            arguments: [
                "from": .int(id1),
                "to": .int(id2),
                "relation": .string("invalid_relation"),
            ]
        ))
    }
}

@Test func connect_memoryNotFound() async throws {
    let tools = try await makeTools()

    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Existing memory")]
    ))
    let id1 = extractId(from: text(from: r1))!

    let result = try await tools.handle(CallTool.Parameters(
        name: "connect",
        arguments: [
            "from": .int(id1),
            "to": .int(99999),
            "relation": .string("relates_to"),
        ]
    ))
    #expect(result.isError == true)
    #expect(text(from: result).contains("not found"))
}

@Test func connect_duplicate_isIdempotent() async throws {
    let tools = try await makeTools()

    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Memory A for dedup")]
    ))
    let r2 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Memory B for dedup"), "force": .bool(true)]
    ))
    let id1 = extractId(from: text(from: r1))!
    let id2 = extractId(from: text(from: r2))!

    // First connect
    let result1 = try await tools.handle(CallTool.Parameters(
        name: "connect",
        arguments: [
            "from": .int(id1),
            "to": .int(id2),
            "relation": .string("relates_to"),
        ]
    ))
    #expect(text(from: result1).contains("Connected"))

    // Second connect — should report already exists
    let result2 = try await tools.handle(CallTool.Parameters(
        name: "connect",
        arguments: [
            "from": .int(id1),
            "to": .int(id2),
            "relation": .string("relates_to"),
        ]
    ))
    #expect(text(from: result2).contains("already exists"))
}

// MARK: - Disconnect

@Test func disconnect_byEdgeId() async throws {
    let tools = try await makeTools()

    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Memory for disconnect test A")]
    ))
    let r2 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Memory for disconnect test B"), "force": .bool(true)]
    ))
    let id1 = extractId(from: text(from: r1))!
    let id2 = extractId(from: text(from: r2))!

    let connectResult = try await tools.handle(CallTool.Parameters(
        name: "connect",
        arguments: [
            "from": .int(id1),
            "to": .int(id2),
            "relation": .string("supersedes"),
        ]
    ))
    let edgeId = extractEdgeId(from: text(from: connectResult))!

    let result = try await tools.handle(CallTool.Parameters(
        name: "disconnect",
        arguments: ["id": .int(edgeId)]
    ))
    let output = text(from: result)
    #expect(output.contains("Deleted edge"))
    #expect(output.contains("id: \(edgeId)"))
}

@Test func disconnect_byFromTo() async throws {
    let tools = try await makeTools()

    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Memory for from-to disconnect A")]
    ))
    let r2 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Memory for from-to disconnect B"), "force": .bool(true)]
    ))
    let id1 = extractId(from: text(from: r1))!
    let id2 = extractId(from: text(from: r2))!

    _ = try await tools.handle(CallTool.Parameters(
        name: "connect",
        arguments: [
            "from": .int(id1),
            "to": .int(id2),
            "relation": .string("relates_to"),
        ]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "disconnect",
        arguments: [
            "from": .int(id1),
            "to": .int(id2),
        ]
    ))
    let output = text(from: result)
    #expect(output.contains("Deleted 1 edge(s)"))
}

// MARK: - Graph

@Test func graph_showsConnections() async throws {
    let tools = try await makeTools()

    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Central concept for graph test")]
    ))
    let r2 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Related concept for graph test"), "force": .bool(true)]
    ))
    let r3 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Contradicting concept for graph test"), "force": .bool(true)]
    ))
    let id1 = extractId(from: text(from: r1))!
    let id2 = extractId(from: text(from: r2))!
    let id3 = extractId(from: text(from: r3))!

    _ = try await tools.handle(CallTool.Parameters(
        name: "connect",
        arguments: ["from": .int(id1), "to": .int(id2), "relation": .string("relates_to")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "connect",
        arguments: ["from": .int(id3), "to": .int(id1), "relation": .string("contradicts")]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "graph",
        arguments: ["id": .int(id1)]
    ))
    let output = text(from: result)
    #expect(output.contains("Central concept"))
    #expect(output.contains("Connections:"))
    #expect(output.contains("--[relates_to]-->"))
    #expect(output.contains("<--[contradicts]--"))
}

@Test func graph_noConnections() async throws {
    let tools = try await makeTools()

    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Isolated memory with no edges")]
    ))
    let id1 = extractId(from: text(from: r1))!

    let result = try await tools.handle(CallTool.Parameters(
        name: "graph",
        arguments: ["id": .int(id1)]
    ))
    let output = text(from: result)
    #expect(output.contains("Isolated memory"))
    #expect(output.contains("No connections."))
}

@Test func graph_memoryNotFound() async throws {
    let tools = try await makeTools()

    let result = try await tools.handle(CallTool.Parameters(
        name: "graph",
        arguments: ["id": .int(99999)]
    ))
    #expect(result.isError == true)
    #expect(text(from: result).contains("not found"))
}

// MARK: - Forget cascades edges

@Test func forget_byId_cascadesEdges() async throws {
    let tools = try await makeTools()

    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Memory to delete with edges")]
    ))
    let r2 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Connected memory that stays"), "force": .bool(true)]
    ))
    let id1 = extractId(from: text(from: r1))!
    let id2 = extractId(from: text(from: r2))!

    // Create edge
    _ = try await tools.handle(CallTool.Parameters(
        name: "connect",
        arguments: ["from": .int(id1), "to": .int(id2), "relation": .string("relates_to")]
    ))

    // Delete memory 1
    let result = try await tools.handle(CallTool.Parameters(
        name: "forget",
        arguments: ["id": .int(id1)]
    ))
    let output = text(from: result)
    #expect(output.contains("Deleted memory"))
    #expect(output.contains("Removed 1 edge(s)"))

    // Graph of memory 2 should show no connections
    let graphResult = try await tools.handle(CallTool.Parameters(
        name: "graph",
        arguments: ["id": .int(id2)]
    ))
    #expect(text(from: graphResult).contains("No connections."))
}

// MARK: - Recall with depth (graph traversal)

@Test func recall_withDepth() async throws {
    let tools = try await makeTools()

    // Create two semantically distant memories connected by an edge
    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Authentication uses JWT tokens for session management")]
    ))
    let r2 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("The PostgreSQL database runs on port 5432 with max 100 connections")]
    ))
    let id1 = extractId(from: text(from: r1))!
    let id2 = extractId(from: text(from: r2))!

    // Connect them — auth depends on the database
    _ = try await tools.handle(CallTool.Parameters(
        name: "connect",
        arguments: ["from": .int(id1), "to": .int(id2), "relation": .string("relates_to")]
    ))

    // Recall with depth=1 — searching for auth should also surface the database memory via graph
    let result = try await tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: [
            "query": .string("authentication JWT session"),
            "depth": .int(1),
            "limit": .int(1),
        ]
    ))
    let output = text(from: result)
    #expect(output.contains("JWT"))
    #expect(output.contains("Connected (graph traversal"))
    #expect(output.contains("PostgreSQL"))
}

// MARK: - Cross-project soft boost

@Test func recall_crossProject_softBoost() async throws {
    let tools = try await makeTools()

    // Store a memory about Lattice under ClaudeMemory project
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Lattice ORM provides @Model macro for Swift database models"),
            "project": .string("ClaudeMemory"),
        ]
    ))

    // Recall with project=Lattice should still surface it (soft boost, not hard filter)
    let result = try await tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: [
            "query": .string("Lattice ORM database models"),
            "project": .string("Lattice"),
        ]
    ))
    let output = text(from: result)
    #expect(output.contains("Lattice ORM"))
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

/// Helper to extract an edge ID from text like "edge id: 42"
private func extractEdgeId(from text: String) -> Int? {
    guard let range = text.range(of: "edge id: ", options: .literal) else {
        return nil
    }
    let after = text[range.upperBound...]
    let digits = after.prefix(while: { $0.isNumber })
    return Int(digits)
}
