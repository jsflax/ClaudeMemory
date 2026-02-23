import Testing
import EngramKit
import Lattice
import MCP
import Foundation

// MARK: - isPrivate Field

@Test func remember_private_storesFlag() async throws {
    let tools = try await makeTools()
    let result = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Secret API key rotation schedule"),
            "is_private": .bool(true),
        ]
    ))
    let output = text(from: result)
    #expect(output.contains("Stored memory"))
    #expect(output.contains("private: true"))
}

@Test func remember_defaultNotPrivate() async throws {
    let tools = try await makeTools()
    let result = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Public knowledge about architecture")]
    ))
    let output = text(from: result)
    #expect(output.contains("Stored memory"))
    #expect(!output.contains("private:"))
}

@Test func remember_explicitNotPrivate() async throws {
    let tools = try await makeTools()
    let result = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Explicitly non-private memory"),
            "is_private": .bool(false),
        ]
    ))
    let output = text(from: result)
    #expect(output.contains("Stored memory"))
    #expect(!output.contains("private:"))
}

@Test func update_togglePrivateOn() async throws {
    let tools = try await makeTools()

    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Memory to make private later")]
    ))
    let id = extractId(from: text(from: r1))!

    let result = try await tools.handle(CallTool.Parameters(
        name: "update",
        arguments: [
            "id": .int(id),
            "is_private": .bool(true),
        ]
    ))
    let output = text(from: result)
    #expect(output.contains("private: false → true"))
}

@Test func update_togglePrivateOff() async throws {
    let tools = try await makeTools()

    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Private memory to make public"),
            "is_private": .bool(true),
        ]
    ))
    let id = extractId(from: text(from: r1))!

    let result = try await tools.handle(CallTool.Parameters(
        name: "update",
        arguments: [
            "id": .int(id),
            "is_private": .bool(false),
        ]
    ))
    let output = text(from: result)
    #expect(output.contains("private: true → false"))
}

@Test func remember_privateWithMetadata() async throws {
    let tools = try await makeTools()
    let result = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Personal credential rotation for staging env"),
            "topic": .string("secrets"),
            "project": .string("MyApp"),
            "is_private": .bool(true),
            "importance": .int(3),
        ]
    ))
    let output = text(from: result)
    #expect(output.contains("Stored memory"))
    #expect(output.contains("project: MyApp"))
    #expect(output.contains("topic: secrets"))
    #expect(output.contains("private: true"))
    #expect(output.contains("importance: 3"))
}
