import Testing
import EngramKit
import Lattice
import MCP
import Foundation

// MARK: - detect_communities Tests

@Test func detectCommunities_findsTwoCliques() async throws {
    let tools = try await makeTools()

    // Group 1: networking (3 memories, connected)
    var netIds: [Int] = []
    for i in 1...3 {
        let r = try await tools.handle(CallTool.Parameters(
            name: "remember",
            arguments: [
                "content": .string("HTTP networking layer \(i): URLSession handles request routing with retry logic and timeout configuration"),
                "project": .string("DetTest"),
                "force": .bool(true),
            ]
        ))
        netIds.append(extractId(from: text(from: r))!)
    }
    for i in 0..<netIds.count {
        for j in (i+1)..<netIds.count {
            _ = try await tools.handle(CallTool.Parameters(
                name: "connect",
                arguments: ["from": .int(netIds[i]), "to": .int(netIds[j]), "relation": .string("relates_to")]
            ))
        }
    }

    // Group 2: database (3 memories, connected)
    var dbIds: [Int] = []
    for i in 1...3 {
        let r = try await tools.handle(CallTool.Parameters(
            name: "remember",
            arguments: [
                "content": .string("SQLite database schema \(i): table migration handles column additions and index creation"),
                "project": .string("DetTest"),
                "force": .bool(true),
            ]
        ))
        dbIds.append(extractId(from: text(from: r))!)
    }
    for i in 0..<dbIds.count {
        for j in (i+1)..<dbIds.count {
            _ = try await tools.handle(CallTool.Parameters(
                name: "connect",
                arguments: ["from": .int(dbIds[i]), "to": .int(dbIds[j]), "relation": .string("relates_to")]
            ))
        }
    }

    let result = try await tools.handle(CallTool.Parameters(
        name: "detect_communities",
        arguments: ["project": .string("DetTest")]
    ))
    let output = text(from: result)
    #expect(output.contains("Community"))
    #expect(output.contains("[id:"))
    // Should find 2 communities
    #expect(output.contains("Community 1") && output.contains("Community 2"))
}

@Test func detectCommunities_noEdges_noResults() async throws {
    let tools = try await makeTools()

    // Store 3 isolated memories (no edges)
    for i in 1...3 {
        _ = try await tools.handle(CallTool.Parameters(
            name: "remember",
            arguments: [
                "content": .string("Isolated memory \(i): unrelated topic about different things"),
                "project": .string("IsoDetTest"),
                "force": .bool(true),
            ]
        ))
    }

    let result = try await tools.handle(CallTool.Parameters(
        name: "detect_communities",
        arguments: ["project": .string("IsoDetTest")]
    ))
    let output = text(from: result)
    // No edges means no communities
    #expect(output.contains("No communities") || output.contains("not in any community"))
}

@Test func detectCommunities_tooFewMemories() async throws {
    let tools = try await makeTools()

    let result = try await tools.handle(CallTool.Parameters(
        name: "detect_communities",
        arguments: ["project": .string("EmptyProject")]
    ))
    let output = text(from: result)
    #expect(output.contains("Not enough"))
    #expect(result.isError != true)
}

// MARK: - organize Tests

@Test func organize_createsHubAndLinksMemories() async throws {
    let tools = try await makeTools()

    // Store 3 memories
    var memIds: [Int] = []
    for i in 1...3 {
        let r = try await tools.handle(CallTool.Parameters(
            name: "remember",
            arguments: [
                "content": .string("Compiler optimization pass \(i): dead code elimination and constant folding"),
                "project": .string("OrgTest"),
                "force": .bool(true),
            ]
        ))
        memIds.append(extractId(from: text(from: r))!)
    }

    // Organize them under a label
    let result = try await tools.handle(CallTool.Parameters(
        name: "organize",
        arguments: [
            "ids": .array(memIds.map { Value.int($0) }),
            "label": .string("compiler-optimization"),
        ]
    ))
    let output = text(from: result)
    #expect(output.contains("Organized"))
    #expect(output.contains("compiler-optimization"))
    #expect(output.contains("Hub:"))
    #expect(output.contains("part_of"))

    // Verify hub was created
    let recall = try await tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: ["query": .string("Hub compiler optimization"), "project": .string("OrgTest"), "limit": .int(10)]
    ))
    let recallOutput = text(from: recall)
    #expect(recallOutput.contains("Hub:"))

    // Verify part_of edges exist
    let graph = try await tools.handle(CallTool.Parameters(
        name: "graph",
        arguments: ["id": .int(memIds[0])]
    ))
    let graphOutput = text(from: graph)
    #expect(graphOutput.contains("part_of"))
}

@Test func organize_updatesTopics() async throws {
    let tools = try await makeTools()

    // Store memories with default topic
    var memIds: [Int] = []
    for i in 1...2 {
        let r = try await tools.handle(CallTool.Parameters(
            name: "remember",
            arguments: [
                "content": .string("UI rendering detail \(i): draw calls and frame timing"),
                "project": .string("TopicTest"),
                "force": .bool(true),
            ]
        ))
        memIds.append(extractId(from: text(from: r))!)
    }

    // Organize with a label
    _ = try await tools.handle(CallTool.Parameters(
        name: "organize",
        arguments: [
            "ids": .array(memIds.map { Value.int($0) }),
            "label": .string("rendering"),
        ]
    ))

    // Recall and check that topic was updated
    let recall = try await tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: ["query": .string("UI rendering"), "project": .string("TopicTest"), "topic": .string("rendering"), "limit": .int(10)]
    ))
    let recallOutput = text(from: recall)
    // Should find the memories under the new topic
    #expect(recallOutput.contains("rendering"))
}

@Test func organize_invalidMemoryId_returnsError() async throws {
    let tools = try await makeTools()

    let result = try await tools.handle(CallTool.Parameters(
        name: "organize",
        arguments: [
            "ids": .array([.int(99999)]),
            "label": .string("nonexistent"),
        ]
    ))
    #expect(result.isError == true)
    #expect(text(from: result).contains("not found"))
}

@Test func organize_emptyIds_returnsError() async throws {
    let tools = try await makeTools()

    let result = try await tools.handle(CallTool.Parameters(
        name: "organize",
        arguments: [
            "ids": .array([]),
            "label": .string("test"),
        ]
    ))
    #expect(result.isError == true)
}

@Test func organize_infersProjectFromMemories() async throws {
    let tools = try await makeTools()

    let r = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Auth token validation logic"),
            "project": .string("InferTest"),
            "force": .bool(true),
        ]
    ))
    let memId = extractId(from: text(from: r))!

    // Organize without specifying project — should infer from memory
    let result = try await tools.handle(CallTool.Parameters(
        name: "organize",
        arguments: [
            "ids": .array([.int(memId)]),
            "label": .string("auth"),
        ]
    ))
    let output = text(from: result)
    #expect(output.contains("Organized"))

    // Verify hub is in the right project
    let recall = try await tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: ["query": .string("Hub auth"), "project": .string("InferTest"), "limit": .int(5)]
    ))
    #expect(text(from: recall).contains("Hub: auth"))
}

// MARK: - Label Propagation Unit Tests

@Test func labelPropagation_findsCommunities() {
    // Two disconnected cliques of 3
    let adjacency: [Int64: Set<Int64>] = [
        1: [2, 3],
        2: [1, 3],
        3: [1, 2],
        4: [5, 6],
        5: [4, 6],
        6: [4, 5],
    ]

    let communities = labelPropagation(adjacency: adjacency)

    #expect(communities.count == 2)
    let sizes = communities.map(\.count).sorted()
    #expect(sizes == [3, 3])

    let c1 = Set(communities[0])
    let c2 = Set(communities[1])
    #expect((c1 == [1, 2, 3] && c2 == [4, 5, 6]) || (c1 == [4, 5, 6] && c2 == [1, 2, 3]))
}

@Test func labelPropagation_bridgeEdge() {
    // Two dense cliques connected by a single bridge edge
    let adjacency: [Int64: Set<Int64>] = [
        1: [2, 3, 4],
        2: [1, 3, 4],
        3: [1, 2, 4],
        4: [1, 2, 3, 5],
        5: [4, 6, 7, 8],
        6: [5, 7, 8],
        7: [5, 6, 8],
        8: [5, 6, 7],
    ]

    let communities = labelPropagation(adjacency: adjacency)

    #expect(communities.count == 2)
    let sizes = communities.map(\.count).sorted()
    #expect(sizes == [4, 4])
}
