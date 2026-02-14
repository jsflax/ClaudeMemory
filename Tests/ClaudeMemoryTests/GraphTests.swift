import Testing
import ClaudeMemoryLib
import Lattice
import MCP
import Foundation

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

    let result1 = try await tools.handle(CallTool.Parameters(
        name: "connect",
        arguments: [
            "from": .int(id1),
            "to": .int(id2),
            "relation": .string("relates_to"),
        ]
    ))
    #expect(text(from: result1).contains("Connected"))

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

    _ = try await tools.handle(CallTool.Parameters(
        name: "connect",
        arguments: ["from": .int(id1), "to": .int(id2), "relation": .string("relates_to")]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "forget",
        arguments: ["id": .int(id1)]
    ))
    let output = text(from: result)
    #expect(output.contains("Deleted memory"))
    #expect(output.contains("Removed 1 edge(s)"))

    let graphResult = try await tools.handle(CallTool.Parameters(
        name: "graph",
        arguments: ["id": .int(id2)]
    ))
    #expect(text(from: graphResult).contains("No connections."))
}
