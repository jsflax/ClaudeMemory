import Testing
import EngramKit
import Lattice
import MCP
import Foundation

// MARK: - Find Clusters

@Test func findClusters_basic() async throws {
    let tools = try await makeTools()

    // Store 5 similar memories about Swift concurrency
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Swift concurrency uses async/await for asynchronous programming"), "force": .bool(true)]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Swift async/await simplifies asynchronous code execution"), "force": .bool(true)]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Swift structured concurrency with async let and task groups"), "force": .bool(true)]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Concurrency in Swift is handled through async functions and actors"), "force": .bool(true)]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Swift async programming uses structured concurrency patterns"), "force": .bool(true)]
    ))

    // Store 2 dissimilar memories
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("The best Italian pasta recipe uses fresh tomatoes and basil"), "force": .bool(true)]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Quantum computing uses qubits for parallel computation"), "force": .bool(true)]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "find_clusters",
        arguments: ["min_cluster_size": .int(3), "distance_threshold": .int(15)]
    ))
    let output = text(from: result)
    #expect(output.contains("cluster"))
    #expect(output.contains("Cluster 1"))
    #expect(output.contains("IDs:"))
    #expect(!output.contains("Suggested action"))
    // Redundancy assessment should be present
    #expect(output.contains("redundant") || output.contains("distinct aspects"))
    // At threshold=15, pasta/quantum should not cluster with Swift concurrency
    #expect(!output.contains("pasta"))
    #expect(!output.contains("quantum"))
}

@Test func findClusters_filterByProject() async throws {
    let tools = try await makeTools()

    // Store memories in project A
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("ProjectA uses React for frontend development"), "project": .string("ProjectA"), "force": .bool(true)]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("ProjectA frontend is built with React components"), "project": .string("ProjectA"), "force": .bool(true)]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("ProjectA React frontend architecture patterns"), "project": .string("ProjectA"), "force": .bool(true)]
    ))

    // Store memories in project B
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("ProjectB uses Vue for frontend development"), "project": .string("ProjectB"), "force": .bool(true)]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("ProjectB frontend is built with Vue components"), "project": .string("ProjectB"), "force": .bool(true)]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("ProjectB Vue frontend architecture patterns"), "project": .string("ProjectB"), "force": .bool(true)]
    ))

    // Filter to project A only
    let result = try await tools.handle(CallTool.Parameters(
        name: "find_clusters",
        arguments: ["project": .string("ProjectA"), "min_cluster_size": .int(3), "distance_threshold": .int(50)]
    ))
    let output = text(from: result)
    #expect(output.contains("React") || output.contains("ProjectA"))
    #expect(!output.contains("Vue"))
}

@Test func findClusters_filterByTopic() async throws {
    let tools = try await makeTools()

    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Debug memory leaks using Instruments"), "topic": .string("debugging"), "force": .bool(true)]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Debug retain cycles with memory graph debugger"), "topic": .string("debugging"), "force": .bool(true)]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Debugging memory issues in Xcode Instruments"), "topic": .string("debugging"), "force": .bool(true)]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Use guard let for early returns"), "topic": .string("conventions"), "force": .bool(true)]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "find_clusters",
        arguments: ["topic": .string("debugging"), "min_cluster_size": .int(3), "distance_threshold": .int(50)]
    ))
    let output = text(from: result)
    #expect(output.contains("Cluster 1"))
    #expect(!output.contains("guard let"))
}

@Test func findClusters_minClusterSize() async throws {
    let tools = try await makeTools()

    // Store 3 similar memories
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Swift uses ARC for memory management"), "force": .bool(true)]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("ARC automatically manages memory in Swift"), "force": .bool(true)]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Automatic Reference Counting handles Swift memory"), "force": .bool(true)]
    ))

    // With min_cluster_size=4, the group of 3 should not qualify
    let result = try await tools.handle(CallTool.Parameters(
        name: "find_clusters",
        arguments: ["min_cluster_size": .int(4), "distance_threshold": .int(50)]
    ))
    let output = text(from: result)
    #expect(output.contains("No clusters found"))
}

@Test func findClusters_noResults() async throws {
    let tools = try await makeTools()

    // Store 3 completely dissimilar memories
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("The best pasta recipe uses fresh tomatoes"), "force": .bool(true)]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Quantum computing uses qubits for computation"), "force": .bool(true)]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Ancient Egyptian pyramids were built as tombs"), "force": .bool(true)]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "find_clusters",
        arguments: ["distance_threshold": .int(5)]
    ))
    let output = text(from: result)
    #expect(output.contains("No clusters found"))
}

@Test func findClusters_showsRedundancyAssessment() async throws {
    let tools = try await makeTools()

    // Store nearly identical memories (should trigger "Likely redundant")
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Swift uses ARC for memory management"), "topic": .string("memory"), "force": .bool(true)]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Swift uses ARC for memory management automatically"), "topic": .string("memory"), "force": .bool(true)]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Swift uses ARC for automatic memory management"), "topic": .string("memory"), "force": .bool(true)]
    ))

    let redundantResult = try await tools.handle(CallTool.Parameters(
        name: "find_clusters",
        arguments: ["min_cluster_size": .int(3), "distance_threshold": .int(30)]
    ))
    let redundantOutput = text(from: redundantResult)
    // Nearly identical memories should show high similarity
    #expect(redundantOutput.contains("Cluster 1"))
    #expect(redundantOutput.contains("Distance stdev"))
    #expect(redundantOutput.contains("topic(s)"))
    #expect(redundantOutput.contains("IDs:"))
}

@Test func findClusters_showsDistinctAspectsForDiverseCluster() async throws {
    let tools = try await makeTools()

    // Store topically related but distinct memories (different topics)
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Our app uses React for the frontend UI components"), "topic": .string("frontend"), "force": .bool(true)]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Our app uses Express.js for the backend API server"), "topic": .string("backend"), "force": .bool(true)]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Our app uses PostgreSQL for the database layer"), "topic": .string("database"), "force": .bool(true)]
    ))

    let diverseResult = try await tools.handle(CallTool.Parameters(
        name: "find_clusters",
        arguments: ["min_cluster_size": .int(3), "distance_threshold": .int(50)]
    ))
    let diverseOutput = text(from: diverseResult)
    // If these cluster together, they should show "distinct aspects" due to different topics
    if diverseOutput.contains("Cluster 1") {
        #expect(diverseOutput.contains("distinct aspects"))
    }
}

// MARK: - Consolidate

@Test func consolidate_basic() async throws {
    let tools = try await makeTools()

    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Swift uses ARC for memory management"), "topic": .string("architecture"), "project": .string("MyApp"), "force": .bool(true)]
    ))
    let id1 = extractId(from: text(from: r1))!

    let r2 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("ARC automatically manages memory in Swift"), "topic": .string("architecture"), "project": .string("MyApp"), "force": .bool(true)]
    ))
    let id2 = extractId(from: text(from: r2))!

    let r3 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Automatic Reference Counting handles Swift memory"), "topic": .string("architecture"), "project": .string("MyApp"), "force": .bool(true)]
    ))
    let id3 = extractId(from: text(from: r3))!

    let result = try await tools.handle(CallTool.Parameters(
        name: "consolidate",
        arguments: [
            "ids": .array([.int(id1), .int(id2), .int(id3)]),
            "content": .string("Swift uses Automatic Reference Counting (ARC) for memory management, automatically tracking and freeing unused objects."),
        ]
    ))
    let output = text(from: result)
    #expect(output.contains("Created summary"))
    #expect(output.contains("importance: 3"))
    #expect(output.contains("Deprioritized 3"))
    #expect(output.contains("summarized_by"))
}

@Test func consolidate_inheritsMetadata() async throws {
    let tools = try await makeTools()

    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Lattice uses SQLite as backend"), "topic": .string("architecture"), "project": .string("Lattice"), "force": .bool(true)]
    ))
    let id1 = extractId(from: text(from: r1))!

    let r2 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Lattice ORM wraps SQLite with Swift types"), "topic": .string("architecture"), "project": .string("Lattice"), "force": .bool(true)]
    ))
    let id2 = extractId(from: text(from: r2))!

    // No topic or project provided — should inherit from originals
    let result = try await tools.handle(CallTool.Parameters(
        name: "consolidate",
        arguments: [
            "ids": .array([.int(id1), .int(id2)]),
            "content": .string("Lattice is a Swift ORM that uses SQLite as its backend database."),
        ]
    ))
    let output = text(from: result)
    #expect(output.contains("project: Lattice"))
    #expect(output.contains("topic: architecture"))
}

@Test func consolidate_customImportance() async throws {
    let tools = try await makeTools()

    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Memory A for importance test"), "force": .bool(true)]
    ))
    let id1 = extractId(from: text(from: r1))!

    let r2 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Memory B for importance test"), "force": .bool(true)]
    ))
    let id2 = extractId(from: text(from: r2))!

    let result = try await tools.handle(CallTool.Parameters(
        name: "consolidate",
        arguments: [
            "ids": .array([.int(id1), .int(id2)]),
            "content": .string("Combined importance test summary"),
            "importance": .int(5),
        ]
    ))
    let output = text(from: result)
    #expect(output.contains("importance: 5"))
}

@Test func consolidate_tooFewIds() async throws {
    let tools = try await makeTools()

    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Single memory for consolidate test")]
    ))
    let id1 = extractId(from: text(from: r1))!

    await #expect(throws: (any Error).self) {
        try await tools.handle(CallTool.Parameters(
            name: "consolidate",
            arguments: [
                "ids": .array([.int(id1)]),
                "content": .string("Summary"),
            ]
        ))
    }
}

@Test func consolidate_notFound() async throws {
    let tools = try await makeTools()

    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Existing memory for not-found test")]
    ))
    let id1 = extractId(from: text(from: r1))!

    let result = try await tools.handle(CallTool.Parameters(
        name: "consolidate",
        arguments: [
            "ids": .array([.int(id1), .int(99999)]),
            "content": .string("Summary"),
        ]
    ))
    #expect(result.isError == true)
    #expect(text(from: result).contains("not found"))
}

@Test func consolidate_edgesTraversable() async throws {
    let tools = try await makeTools()

    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Edge test memory alpha"), "force": .bool(true)]
    ))
    let id1 = extractId(from: text(from: r1))!

    let r2 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Edge test memory beta"), "force": .bool(true)]
    ))
    let id2 = extractId(from: text(from: r2))!

    let consolidateResult = try await tools.handle(CallTool.Parameters(
        name: "consolidate",
        arguments: [
            "ids": .array([.int(id1), .int(id2)]),
            "content": .string("Combined edge test summary"),
        ]
    ))
    let consolidateOutput = text(from: consolidateResult)
    // Extract summary ID from "Created summary [id:N]"
    guard let idRange = consolidateOutput.range(of: "[id:") else {
        Issue.record("No [id: found in consolidate output")
        return
    }
    let afterId = consolidateOutput[idRange.upperBound...]
    let summaryIdStr = afterId.prefix(while: { $0.isNumber })
    let summaryId = Int(summaryIdStr)!

    // Use graph on one of the originals — should show summarized_by edge to the summary
    let graphResult = try await tools.handle(CallTool.Parameters(
        name: "graph",
        arguments: ["id": .int(id1)]
    ))
    let graphOutput = text(from: graphResult)
    #expect(graphOutput.contains("summarized_by"))
    #expect(graphOutput.contains("\(summaryId)"))
}

@Test func consolidate_deprioritizesInRecall() async throws {
    let tools = try await makeTools()

    // Store 2 memories about the same topic with high importance
    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Rust borrow checker enforces memory safety at compile time"),
            "importance": .int(4), "force": .bool(true),
        ]
    ))
    let id1 = extractId(from: text(from: r1))!

    let r2 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Rust ownership model prevents data races through the borrow checker"),
            "importance": .int(4), "force": .bool(true),
        ]
    ))
    let id2 = extractId(from: text(from: r2))!

    // Consolidate — originals get importance=0, summary gets importance=3
    _ = try await tools.handle(CallTool.Parameters(
        name: "consolidate",
        arguments: [
            "ids": .array([.int(id1), .int(id2)]),
            "content": .string("Rust uses an ownership model with a borrow checker to enforce memory safety and prevent data races at compile time."),
        ]
    ))

    // Recall — the summary should appear in results
    let recallResult = try await tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: ["query": .string("Rust borrow checker memory safety")]
    ))
    let recallOutput = text(from: recallResult)
    #expect(recallOutput.contains("ownership model with a borrow checker"))

    // Verify originals were actually deprioritized (importance → 0) by reading back via update
    let update1 = try await tools.handle(CallTool.Parameters(
        name: "update",
        arguments: ["id": .int(id1), "importance": .int(0)]
    ))
    // "importance: 0 → 0" confirms it was already 0 from consolidation
    #expect(text(from: update1).contains("importance: 0 → 0"))

    let update2 = try await tools.handle(CallTool.Parameters(
        name: "update",
        arguments: ["id": .int(id2), "importance": .int(0)]
    ))
    #expect(text(from: update2).contains("importance: 0 → 0"))
}

// MARK: - FlexibleIntArray (comma-separated string)

@Test func consolidate_idsAsCommaSeparatedString() async throws {
    let tools = try await makeTools()

    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Flexible array test memory alpha"), "force": .bool(true)]
    ))
    let id1 = extractId(from: text(from: r1))!

    let r2 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Flexible array test memory beta"), "force": .bool(true)]
    ))
    let id2 = extractId(from: text(from: r2))!

    // Pass ids as a comma-separated string instead of a JSON array
    let result = try await tools.handle(CallTool.Parameters(
        name: "consolidate",
        arguments: [
            "ids": .string("\(id1), \(id2)"),
            "content": .string("Combined flexible array test summary"),
        ]
    ))
    let output = text(from: result)
    #expect(output.contains("Created summary"))
    #expect(output.contains("Deprioritized 2"))
}

@Test func merge_idsAsCommaSeparatedString() async throws {
    let tools = try await makeTools()

    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Merge flex test alpha"), "force": .bool(true)]
    ))
    let id1 = extractId(from: text(from: r1))!

    let r2 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Merge flex test beta"), "force": .bool(true)]
    ))
    let id2 = extractId(from: text(from: r2))!

    // Pass ids as a comma-separated string
    let result = try await tools.handle(CallTool.Parameters(
        name: "merge",
        arguments: [
            "ids": .string("\(id1), \(id2)"),
            "content": .string("Combined merge flex test"),
        ]
    ))
    let output = text(from: result)
    #expect(output.contains("Merged"))
}
