import Testing
import ClaudeMemoryLib
import Lattice
import MCP
import Foundation

// MARK: - Checkpoint Create

@Test func checkpoint_createBasic() async throws {
    let tools = try await makeTools()
    let result = try await tools.handle(CallTool.Parameters(
        name: "checkpoint",
        arguments: ["title": .string("Implement auth flow")]
    ))
    let output = text(from: result)
    #expect(output.contains("Created task"))
    #expect(output.contains("status: active"))
    #expect(output.contains("project: global"))
    #expect(output.contains("Implement auth flow"))
}

@Test func checkpoint_createWithAllFields() async throws {
    let tools = try await makeTools()
    let result = try await tools.handle(CallTool.Parameters(
        name: "checkpoint",
        arguments: [
            "title": .string("Add dark mode"),
            "status": .string("paused"),
            "project": .string("MyApp"),
            "plan": .string("1. Add theme provider\n2. Update components"),
            "progress": .string("Theme provider done"),
            "context": .string("See src/theme.ts"),
        ]
    ))
    let output = text(from: result)
    #expect(output.contains("Created task"))
    #expect(output.contains("status: paused"))
    #expect(output.contains("project: MyApp"))
}

@Test func checkpoint_createMissingTitle_throws() async throws {
    let tools = try await makeTools()
    await #expect(throws: (any Error).self) {
        try await tools.handle(CallTool.Parameters(
            name: "checkpoint",
            arguments: ["plan": .string("some plan")]
        ))
    }
}

@Test func checkpoint_createEmptyTitle_throws() async throws {
    let tools = try await makeTools()
    await #expect(throws: (any Error).self) {
        try await tools.handle(CallTool.Parameters(
            name: "checkpoint",
            arguments: ["title": .string("")]
        ))
    }
}

@Test func checkpoint_createInvalidStatus_throws() async throws {
    let tools = try await makeTools()
    await #expect(throws: (any Error).self) {
        try await tools.handle(CallTool.Parameters(
            name: "checkpoint",
            arguments: [
                "title": .string("Test"),
                "status": .string("invalid"),
            ]
        ))
    }
}

// MARK: - Checkpoint Update

@Test func checkpoint_updateExisting() async throws {
    let tools = try await makeTools()
    let r1 = try await tools.handle(CallTool.Parameters(
        name: "checkpoint",
        arguments: ["title": .string("Build feature X")]
    ))
    let taskId = extractTaskId(from: text(from: r1))!

    let result = try await tools.handle(CallTool.Parameters(
        name: "checkpoint",
        arguments: [
            "task_id": .int(taskId),
            "progress": .string("Step 1 complete"),
        ]
    ))
    let output = text(from: result)
    #expect(output.contains("Updated task"))
    #expect(output.contains("progress"))
}

@Test func checkpoint_updatePlanAndContext() async throws {
    let tools = try await makeTools()
    let r1 = try await tools.handle(CallTool.Parameters(
        name: "checkpoint",
        arguments: ["title": .string("Refactor DB layer")]
    ))
    let taskId = extractTaskId(from: text(from: r1))!

    _ = try await tools.handle(CallTool.Parameters(
        name: "checkpoint",
        arguments: [
            "task_id": .int(taskId),
            "plan": .string("1. Extract interface\n2. Swap implementation"),
            "context": .string("Key file: src/db.ts"),
        ]
    ))

    // Resume to verify persistence
    let resumed = try await tools.handle(CallTool.Parameters(
        name: "resume",
        arguments: ["task_id": .int(taskId)]
    ))
    let output = text(from: resumed)
    #expect(output.contains("Extract interface"))
    #expect(output.contains("src/db.ts"))
}

@Test func checkpoint_updateStatus() async throws {
    let tools = try await makeTools()
    let r1 = try await tools.handle(CallTool.Parameters(
        name: "checkpoint",
        arguments: ["title": .string("Task for status change")]
    ))
    let taskId = extractTaskId(from: text(from: r1))!

    let result = try await tools.handle(CallTool.Parameters(
        name: "checkpoint",
        arguments: [
            "task_id": .int(taskId),
            "status": .string("paused"),
        ]
    ))
    let output = text(from: result)
    #expect(output.contains("status: active → paused"))
}

@Test func checkpoint_reopenCompleted() async throws {
    let tools = try await makeTools()
    let r1 = try await tools.handle(CallTool.Parameters(
        name: "checkpoint",
        arguments: [
            "title": .string("Task to reopen"),
            "status": .string("completed"),
        ]
    ))
    let taskId = extractTaskId(from: text(from: r1))!

    let result = try await tools.handle(CallTool.Parameters(
        name: "checkpoint",
        arguments: [
            "task_id": .int(taskId),
            "status": .string("active"),
        ]
    ))
    let output = text(from: result)
    #expect(output.contains("(re-opened)"))
}

@Test func checkpoint_updateNotFound() async throws {
    let tools = try await makeTools()
    let result = try await tools.handle(CallTool.Parameters(
        name: "checkpoint",
        arguments: [
            "task_id": .int(99999),
            "progress": .string("something"),
        ]
    ))
    #expect(result.isError == true)
    #expect(text(from: result).contains("not found"))
}

@Test func checkpoint_updateStringEncodedId() async throws {
    let tools = try await makeTools()
    let r1 = try await tools.handle(CallTool.Parameters(
        name: "checkpoint",
        arguments: ["title": .string("String ID test")]
    ))
    let taskId = extractTaskId(from: text(from: r1))!

    // Pass task_id as string (FlexibleInt should handle it)
    let result = try await tools.handle(CallTool.Parameters(
        name: "checkpoint",
        arguments: [
            "task_id": .string(String(taskId)),
            "progress": .string("Updated via string ID"),
        ]
    ))
    #expect(result.isError != true)
    #expect(text(from: result).contains("Updated task"))
}

// MARK: - Resume

@Test func resume_basic() async throws {
    let tools = try await makeTools()
    let r1 = try await tools.handle(CallTool.Parameters(
        name: "checkpoint",
        arguments: [
            "title": .string("Full resume test"),
            "plan": .string("Step 1\nStep 2"),
            "progress": .string("Step 1 done"),
            "context": .string("files: a.ts, b.ts"),
        ]
    ))
    let taskId = extractTaskId(from: text(from: r1))!

    let result = try await tools.handle(CallTool.Parameters(
        name: "resume",
        arguments: ["task_id": .int(taskId)]
    ))
    let output = text(from: result)
    #expect(output.contains("## Resuming: Full resume test"))
    #expect(output.contains("### Plan"))
    #expect(output.contains("Step 1"))
    #expect(output.contains("### Progress"))
    #expect(output.contains("Step 1 done"))
    #expect(output.contains("### Context"))
    #expect(output.contains("a.ts"))
}

@Test func resume_fromPaused() async throws {
    let tools = try await makeTools()
    let r1 = try await tools.handle(CallTool.Parameters(
        name: "checkpoint",
        arguments: [
            "title": .string("Paused task"),
            "status": .string("paused"),
        ]
    ))
    let taskId = extractTaskId(from: text(from: r1))!

    let result = try await tools.handle(CallTool.Parameters(
        name: "resume",
        arguments: ["task_id": .int(taskId)]
    ))
    let output = text(from: result)
    #expect(output.contains("paused → active"))
}

@Test func resume_alreadyActive() async throws {
    let tools = try await makeTools()
    let r1 = try await tools.handle(CallTool.Parameters(
        name: "checkpoint",
        arguments: ["title": .string("Already active task")]
    ))
    let taskId = extractTaskId(from: text(from: r1))!

    let result = try await tools.handle(CallTool.Parameters(
        name: "resume",
        arguments: ["task_id": .int(taskId)]
    ))
    let output = text(from: result)
    #expect(output.contains("active → active"))
    #expect(result.isError != true)
}

@Test func resume_notFound() async throws {
    let tools = try await makeTools()
    let result = try await tools.handle(CallTool.Parameters(
        name: "resume",
        arguments: ["task_id": .int(99999)]
    ))
    #expect(result.isError == true)
    #expect(text(from: result).contains("not found"))
}

// MARK: - List Tasks

@Test func listTasks_defaultShowsActiveAndPaused() async throws {
    let tools = try await makeTools()

    _ = try await tools.handle(CallTool.Parameters(
        name: "checkpoint",
        arguments: ["title": .string("Active task"), "status": .string("active")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "checkpoint",
        arguments: ["title": .string("Paused task"), "status": .string("paused")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "checkpoint",
        arguments: ["title": .string("Completed task"), "status": .string("completed")]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "list_tasks",
        arguments: nil
    ))
    let output = text(from: result)
    #expect(output.contains("Active task"))
    #expect(output.contains("Paused task"))
    #expect(!output.contains("Completed task"))
}

@Test func listTasks_filterByStatus() async throws {
    let tools = try await makeTools()

    _ = try await tools.handle(CallTool.Parameters(
        name: "checkpoint",
        arguments: ["title": .string("Active one"), "status": .string("active")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "checkpoint",
        arguments: ["title": .string("Done one"), "status": .string("completed")]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "list_tasks",
        arguments: ["status": .string("completed")]
    ))
    let output = text(from: result)
    #expect(output.contains("Done one"))
    #expect(!output.contains("Active one"))
}

@Test func listTasks_filterByProject() async throws {
    let tools = try await makeTools()

    _ = try await tools.handle(CallTool.Parameters(
        name: "checkpoint",
        arguments: ["title": .string("Alpha task"), "project": .string("Alpha")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "checkpoint",
        arguments: ["title": .string("Beta task"), "project": .string("Beta")]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "list_tasks",
        arguments: ["project": .string("Alpha")]
    ))
    let output = text(from: result)
    #expect(output.contains("Alpha task"))
    #expect(!output.contains("Beta task"))
}

@Test func listTasks_empty() async throws {
    let tools = try await makeTools()
    let result = try await tools.handle(CallTool.Parameters(
        name: "list_tasks",
        arguments: nil
    ))
    #expect(text(from: result) == "No tasks found.")
}

@Test func listTasks_orderByMostRecentCheckpoint() async throws {
    let tools = try await makeTools()

    let r1 = try await tools.handle(CallTool.Parameters(
        name: "checkpoint",
        arguments: ["title": .string("Older task")]
    ))
    let olderId = extractTaskId(from: text(from: r1))!

    // Small delay to ensure different timestamps
    try await Task.sleep(for: .milliseconds(50))

    _ = try await tools.handle(CallTool.Parameters(
        name: "checkpoint",
        arguments: ["title": .string("Newer task")]
    ))

    // Update the older task to make it most recent
    try await Task.sleep(for: .milliseconds(50))
    _ = try await tools.handle(CallTool.Parameters(
        name: "checkpoint",
        arguments: [
            "task_id": .int(olderId),
            "progress": .string("Just updated"),
        ]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "list_tasks",
        arguments: nil
    ))
    let output = text(from: result)
    // Older task was updated last, so it should appear first
    let olderRange = output.range(of: "Older task")!
    let newerRange = output.range(of: "Newer task")!
    #expect(olderRange.lowerBound < newerRange.lowerBound)
}

@Test func listTasks_invalidStatus_throws() async throws {
    let tools = try await makeTools()
    await #expect(throws: (any Error).self) {
        try await tools.handle(CallTool.Parameters(
            name: "list_tasks",
            arguments: ["status": .string("invalid")]
        ))
    }
}
