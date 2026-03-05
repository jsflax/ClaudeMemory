import Testing
import EngramKit
import Lattice
import MCP
import Foundation

// MARK: - detect_communities Tests

@Test func detectCommunities_findsTwoCliques() async throws {
    let tools = try await makeTools()

    // Group 1: networking (3 memories, connected)
    var netIds: [String] = []
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
                arguments: ["from": .string(netIds[i]), "to": .string(netIds[j]), "relation": .string("relates_to")]
            ))
        }
    }

    // Group 2: database (3 memories, connected)
    var dbIds: [String] = []
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
                arguments: ["from": .string(dbIds[i]), "to": .string(dbIds[j]), "relation": .string("relates_to")]
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
    var memIds: [String] = []
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
            "ids": .array(memIds.map { Value.string($0) }),
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
        arguments: ["id": .string(memIds[0])]
    ))
    let graphOutput = text(from: graph)
    #expect(graphOutput.contains("part_of"))
}

@Test func organize_updatesTopics() async throws {
    let tools = try await makeTools()

    // Store memories with default topic
    var memIds: [String] = []
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
            "ids": .array(memIds.map { Value.string($0) }),
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
            "ids": .array([.string(UUID().uuidString)]),
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
            "ids": .array([.string(memId)]),
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
    let u1 = UUID(), u2 = UUID(), u3 = UUID(), u4 = UUID(), u5 = UUID(), u6 = UUID()
    let adjacency: [UUID: Set<UUID>] = [
        u1: [u2, u3],
        u2: [u1, u3],
        u3: [u1, u2],
        u4: [u5, u6],
        u5: [u4, u6],
        u6: [u4, u5],
    ]

    let communities = labelPropagation(adjacency: adjacency)

    #expect(communities.count == 2)
    let sizes: [Int] = communities.map(\.count).sorted()
    #expect(sizes == [3, 3])

    let c1 = Set(communities[0])
    let c2 = Set(communities[1])
    #expect((c1 == [u1, u2, u3] && c2 == [u4, u5, u6]) || (c1 == [u4, u5, u6] && c2 == [u1, u2, u3]))
}

@Test func labelPropagation_bridgeEdge() {
    // Two dense cliques connected by a single bridge edge
    let u1 = UUID(), u2 = UUID(), u3 = UUID(), u4 = UUID()
    let u5 = UUID(), u6 = UUID(), u7 = UUID(), u8 = UUID()
    let adjacency: [UUID: Set<UUID>] = [
        u1: [u2, u3, u4],
        u2: [u1, u3, u4],
        u3: [u1, u2, u4],
        u4: [u1, u2, u3, u5],
        u5: [u4, u6, u7, u8],
        u6: [u5, u7, u8],
        u7: [u5, u6, u8],
        u8: [u5, u6, u7],
    ]

    let communities = labelPropagation(adjacency: adjacency)

    #expect(communities.count == 2)
    let sizes: [Int] = communities.map(\.count).sorted()
    #expect(sizes == [4, 4])
}
