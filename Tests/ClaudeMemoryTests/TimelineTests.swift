import Testing
import ClaudeMemoryLib
import Lattice
import MCP
import Foundation

// MARK: - Timeline

@Test func timeline_basic() async throws {
    let tools = try await makeTools()

    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("First timeline memory"),
            "project": .string("TestProject"),
        ]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Second timeline memory"),
            "project": .string("TestProject"),
            "force": .bool(true),
        ]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "timeline",
        arguments: ["project": .string("TestProject")]
    ))
    let output = text(from: result)
    #expect(output.contains("##"))
    #expect(output.contains("2 memories"))
    #expect(output.contains("First timeline memory"))
    #expect(output.contains("Second timeline memory"))
}

@Test func timeline_groupByWeek() async throws {
    let tools = try await makeTools()

    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Memory for week grouping test")]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "timeline",
        arguments: ["group_by": .string("week")]
    ))
    let output = text(from: result)
    #expect(output.contains("Week of"))
    #expect(output.contains("week grouping test"))
}

@Test func timeline_groupByMonth() async throws {
    let tools = try await makeTools()

    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Memory for month grouping test")]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "timeline",
        arguments: ["group_by": .string("month")]
    ))
    let output = text(from: result)
    #expect(output.contains("month grouping test"))
    // Month format: "February 2026"
    #expect(output.contains("##"))
}

@Test func timeline_projectFilter() async throws {
    let tools = try await makeTools()

    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Alpha project memory"),
            "project": .string("Alpha"),
        ]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Beta project memory"),
            "project": .string("Beta"),
        ]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "timeline",
        arguments: ["project": .string("Alpha")]
    ))
    let output = text(from: result)
    #expect(output.contains("Alpha project memory"))
    #expect(!output.contains("Beta project memory"))
}

@Test func timeline_sinceFilter() async throws {
    let tools = try await makeTools()

    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Recent memory for timeline since")]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "timeline",
        arguments: ["since": .string("today")]
    ))
    let output = text(from: result)
    #expect(output.contains("Recent memory for timeline since"))
}

@Test func timeline_empty() async throws {
    let tools = try await makeTools()

    let result = try await tools.handle(CallTool.Parameters(
        name: "timeline",
        arguments: ["project": .string("NonexistentProject")]
    ))
    #expect(text(from: result) == "No memories found.")
}

@Test func timeline_invalidGroupBy_throws() async throws {
    let tools = try await makeTools()
    await #expect(throws: (any Error).self) {
        try await tools.handle(CallTool.Parameters(
            name: "timeline",
            arguments: ["group_by": .string("hour")]
        ))
    }
}
