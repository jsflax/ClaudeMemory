import Testing
import ClaudeMemoryLib
import Lattice
import MCP
import Foundation

// MARK: - Begin Episode

@Test func beginEpisode_basic() async throws {
    let tools = try await makeTools()
    let result = try await tools.handle(CallTool.Parameters(
        name: "begin_episode",
        arguments: [
            "title": .string("Debugging race condition"),
            "project": .string("MyApp"),
        ]
    ))
    let output = text(from: result)
    #expect(output.contains("Created episode"))
    #expect(output.contains("Debugging race condition"))
    #expect(output.contains("project: MyApp"))
}

@Test func beginEpisode_autoTitle() async throws {
    let tools = try await makeTools()
    let result = try await tools.handle(CallTool.Parameters(
        name: "begin_episode",
        arguments: nil
    ))
    let output = text(from: result)
    #expect(output.contains("Created episode"))
    #expect(output.contains("Session:"))
}

@Test func beginEpisode_closesAutoEpisode() async throws {
    let tools = try await makeTools()

    // Create an auto-episode by remembering
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Auto-episode memory")]
    ))

    // Explicitly begin an episode — should close the auto one
    let result = try await tools.handle(CallTool.Parameters(
        name: "begin_episode",
        arguments: ["title": .string("Explicit episode")]
    ))
    let output = text(from: result)
    #expect(output.contains("Explicit episode"))

    // List episodes — should see both (auto=ended, explicit=active)
    let list = try await tools.handle(CallTool.Parameters(
        name: "list_episodes",
        arguments: nil
    ))
    let listOutput = text(from: list)
    #expect(listOutput.contains("[ended]"))
    #expect(listOutput.contains("[active]"))
    #expect(listOutput.contains("Explicit episode"))
}

// MARK: - End Episode

@Test func endEpisode_basic() async throws {
    let tools = try await makeTools()
    let r1 = try await tools.handle(CallTool.Parameters(
        name: "begin_episode",
        arguments: ["title": .string("Session to end")]
    ))
    let epId = extractEpisodeId(from: text(from: r1))!

    let result = try await tools.handle(CallTool.Parameters(
        name: "end_episode",
        arguments: nil
    ))
    let output = text(from: result)
    #expect(output.contains("Ended episode"))
    #expect(output.contains("episode:\(epId)"))

    // Verify it's ended via list
    let list = try await tools.handle(CallTool.Parameters(
        name: "list_episodes",
        arguments: ["status": .string("ended")]
    ))
    #expect(text(from: list).contains("Session to end"))
}

@Test func endEpisode_withSummary() async throws {
    let tools = try await makeTools()
    _ = try await tools.handle(CallTool.Parameters(
        name: "begin_episode",
        arguments: ["title": .string("Summarized session")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Found the bug in auth module")]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "end_episode",
        arguments: ["summary": .string("Fixed auth bug by correcting token expiry")]
    ))
    let output = text(from: result)
    #expect(output.contains("Ended episode"))
    #expect(output.contains("summary: Fixed auth bug"))
}

@Test func endEpisode_byId() async throws {
    let tools = try await makeTools()
    let r1 = try await tools.handle(CallTool.Parameters(
        name: "begin_episode",
        arguments: ["title": .string("Episode to end by ID")]
    ))
    let epId = extractEpisodeId(from: text(from: r1))!

    let result = try await tools.handle(CallTool.Parameters(
        name: "end_episode",
        arguments: ["episode_id": .int(epId)]
    ))
    #expect(text(from: result).contains("Ended episode"))
}

@Test func endEpisode_noActive_throws() async throws {
    let tools = try await makeTools()
    await #expect(throws: (any Error).self) {
        try await tools.handle(CallTool.Parameters(
            name: "end_episode",
            arguments: nil
        ))
    }
}

@Test func endEpisode_notFound() async throws {
    let tools = try await makeTools()
    let result = try await tools.handle(CallTool.Parameters(
        name: "end_episode",
        arguments: ["episode_id": .int(99999)]
    ))
    #expect(result.isError == true)
    #expect(text(from: result).contains("not found"))
}

// MARK: - Auto-Episode

@Test func autoEpisode_firstRemember() async throws {
    let tools = try await makeTools()

    // Remember without begin_episode — should auto-create an episode
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Auto-episode first memory")]
    ))

    let list = try await tools.handle(CallTool.Parameters(
        name: "list_episodes",
        arguments: nil
    ))
    let output = text(from: list)
    #expect(output.contains("Session:"))
    #expect(output.contains("[active]"))
    #expect(output.contains("1 memories"))
}

@Test func autoEpisode_subsequentRemember() async throws {
    let tools = try await makeTools()

    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("First memory in auto-episode")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Second memory in same auto-episode"), "force": .bool(true)]
    ))

    // Should still be one episode with 2 memories
    let list = try await tools.handle(CallTool.Parameters(
        name: "list_episodes",
        arguments: nil
    ))
    let output = text(from: list)
    #expect(output.contains("2 memories"))

    // Count episodes — should be exactly 1 line
    let lines = output.split(separator: "\n")
    #expect(lines.count == 1)
}

@Test func autoEpisode_gapCreatesNew() async throws {
    let tools = try await makeTools()

    // First remember — creates auto-episode
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Memory before gap")]
    ))

    // Simulate >30 min gap by setting lastMemoryTime far in the past
    await tools.setLastMemoryTime(Date().addingTimeInterval(-3600))

    // Second remember — should create a new episode
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Memory after gap"), "force": .bool(true)]
    ))

    // Should have 2 episodes now
    let list = try await tools.handle(CallTool.Parameters(
        name: "list_episodes",
        arguments: nil
    ))
    let output = text(from: list)
    let lines = output.split(separator: "\n")
    #expect(lines.count == 2)

    // The older one should be ended, the newer one active
    #expect(output.contains("[ended]"))
    #expect(output.contains("[active]"))
}

@Test func autoEpisode_explicitOverrides() async throws {
    let tools = try await makeTools()

    // Auto-create episode via remember
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Auto memory")]
    ))

    // Explicitly begin — auto should be ended
    _ = try await tools.handle(CallTool.Parameters(
        name: "begin_episode",
        arguments: ["title": .string("Explicit override")]
    ))

    // New remember should go into the explicit episode
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Explicit memory"), "force": .bool(true)]
    ))

    let list = try await tools.handle(CallTool.Parameters(
        name: "list_episodes",
        arguments: nil
    ))
    let output = text(from: list)
    #expect(output.contains("[ended]"))
    #expect(output.contains("Explicit override"))
    #expect(output.contains("[active]"))
}

// MARK: - Recall Episode

@Test func recallEpisode_basic() async throws {
    let tools = try await makeTools()
    let r1 = try await tools.handle(CallTool.Parameters(
        name: "begin_episode",
        arguments: ["title": .string("Recall test episode")]
    ))
    let epId = extractEpisodeId(from: text(from: r1))!

    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("First recall test memory")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Second recall test memory"), "force": .bool(true)]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Third recall test memory"), "force": .bool(true)]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "recall_episode",
        arguments: ["episode_id": .int(epId)]
    ))
    let output = text(from: result)
    #expect(output.contains("## Episode: Recall test episode"))
    #expect(output.contains("Memories (3)"))
    #expect(output.contains("First recall test memory"))
    #expect(output.contains("Second recall test memory"))
    #expect(output.contains("Third recall test memory"))

    // Verify chronological order
    let firstRange = output.range(of: "First recall test")!
    let secondRange = output.range(of: "Second recall test")!
    let thirdRange = output.range(of: "Third recall test")!
    #expect(firstRange.lowerBound < secondRange.lowerBound)
    #expect(secondRange.lowerBound < thirdRange.lowerBound)
}

@Test func recallEpisode_withSummary() async throws {
    let tools = try await makeTools()
    let r1 = try await tools.handle(CallTool.Parameters(
        name: "begin_episode",
        arguments: ["title": .string("Summary recall test")]
    ))
    let epId = extractEpisodeId(from: text(from: r1))!

    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Memory in summarized episode")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "end_episode",
        arguments: ["summary": .string("We did important things")]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "recall_episode",
        arguments: ["episode_id": .int(epId)]
    ))
    let output = text(from: result)
    #expect(output.contains("### Summary"))
    #expect(output.contains("We did important things"))
}

@Test func recallEpisode_empty() async throws {
    let tools = try await makeTools()
    let r1 = try await tools.handle(CallTool.Parameters(
        name: "begin_episode",
        arguments: ["title": .string("Empty episode")]
    ))
    let epId = extractEpisodeId(from: text(from: r1))!

    let result = try await tools.handle(CallTool.Parameters(
        name: "recall_episode",
        arguments: ["episode_id": .int(epId)]
    ))
    let output = text(from: result)
    #expect(output.contains("No memories in this episode"))
}

@Test func recallEpisode_notFound() async throws {
    let tools = try await makeTools()
    let result = try await tools.handle(CallTool.Parameters(
        name: "recall_episode",
        arguments: ["episode_id": .int(99999)]
    ))
    #expect(result.isError == true)
    #expect(text(from: result).contains("not found"))
}

// MARK: - List Episodes

@Test func listEpisodes_basic() async throws {
    let tools = try await makeTools()

    _ = try await tools.handle(CallTool.Parameters(
        name: "begin_episode",
        arguments: ["title": .string("Episode Alpha")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "end_episode",
        arguments: nil
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "begin_episode",
        arguments: ["title": .string("Episode Beta")]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "list_episodes",
        arguments: nil
    ))
    let output = text(from: result)
    #expect(output.contains("Episode Alpha"))
    #expect(output.contains("Episode Beta"))
}

@Test func listEpisodes_filterByProject() async throws {
    let tools = try await makeTools()

    _ = try await tools.handle(CallTool.Parameters(
        name: "begin_episode",
        arguments: ["title": .string("ProjectA episode"), "project": .string("ProjectA")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "end_episode",
        arguments: nil
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "begin_episode",
        arguments: ["title": .string("ProjectB episode"), "project": .string("ProjectB")]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "list_episodes",
        arguments: ["project": .string("ProjectA")]
    ))
    let output = text(from: result)
    #expect(output.contains("ProjectA episode"))
    #expect(!output.contains("ProjectB episode"))
}

@Test func listEpisodes_filterByStatus() async throws {
    let tools = try await makeTools()

    _ = try await tools.handle(CallTool.Parameters(
        name: "begin_episode",
        arguments: ["title": .string("Ended episode")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "end_episode",
        arguments: nil
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "begin_episode",
        arguments: ["title": .string("Active episode")]
    ))

    let endedResult = try await tools.handle(CallTool.Parameters(
        name: "list_episodes",
        arguments: ["status": .string("ended")]
    ))
    let endedOutput = text(from: endedResult)
    #expect(endedOutput.contains("Ended episode"))
    #expect(!endedOutput.contains("Active episode"))

    let activeResult = try await tools.handle(CallTool.Parameters(
        name: "list_episodes",
        arguments: ["status": .string("active")]
    ))
    let activeOutput = text(from: activeResult)
    #expect(activeOutput.contains("Active episode"))
    #expect(!activeOutput.contains("Ended episode"))
}

@Test func listEpisodes_empty() async throws {
    let tools = try await makeTools()
    let result = try await tools.handle(CallTool.Parameters(
        name: "list_episodes",
        arguments: nil
    ))
    #expect(text(from: result) == "No episodes found.")
}

@Test func listEpisodes_showsMemoryCount() async throws {
    let tools = try await makeTools()

    _ = try await tools.handle(CallTool.Parameters(
        name: "begin_episode",
        arguments: ["title": .string("Counted episode")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Memory one in counted episode")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Memory two in counted episode"), "force": .bool(true)]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "list_episodes",
        arguments: nil
    ))
    #expect(text(from: result).contains("2 memories"))
}

@Test func listEpisodes_orderByMostRecent() async throws {
    let tools = try await makeTools()

    _ = try await tools.handle(CallTool.Parameters(
        name: "begin_episode",
        arguments: ["title": .string("Older episode")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "end_episode",
        arguments: nil
    ))

    try await Task.sleep(for: .milliseconds(50))

    _ = try await tools.handle(CallTool.Parameters(
        name: "begin_episode",
        arguments: ["title": .string("Newer episode")]
    ))

    let result = try await tools.handle(CallTool.Parameters(
        name: "list_episodes",
        arguments: nil
    ))
    let output = text(from: result)
    // Newer episode should appear first (sorted by startedAt desc)
    let newerRange = output.range(of: "Newer episode")!
    let olderRange = output.range(of: "Older episode")!
    #expect(newerRange.lowerBound < olderRange.lowerBound)
}

// MARK: - Cross-Project Auto-Episode

@Test func autoEpisode_crossProject_startsNewEpisode() async throws {
    let tools = try await makeTools()

    // Remember for project X — creates auto-episode with project X
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Memory for project X"), "project": .string("X")]
    ))

    // Remember for project Y — should start a new auto-episode for Y
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Memory for project Y"), "project": .string("Y")]
    ))

    // Should have 2 episodes, one per project
    let list = try await tools.handle(CallTool.Parameters(
        name: "list_episodes",
        arguments: nil
    ))
    let output = text(from: list)
    let lines = output.split(separator: "\n")
    #expect(lines.count == 2)

    // First episode (X) should be ended, second (Y) active
    #expect(output.contains("[ended]"))
    #expect(output.contains("[active]"))

    // Filter by project to verify scoping
    let xList = try await tools.handle(CallTool.Parameters(
        name: "list_episodes",
        arguments: ["project": .string("X")]
    ))
    #expect(text(from: xList).contains("1 memories"))

    let yList = try await tools.handle(CallTool.Parameters(
        name: "list_episodes",
        arguments: ["project": .string("Y")]
    ))
    #expect(text(from: yList).contains("1 memories"))
}

@Test func autoEpisode_crossProject_sameProjectReuses() async throws {
    let tools = try await makeTools()

    // Two remembers for the same project should reuse the episode
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("First X memory"), "project": .string("X")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Second X memory"), "project": .string("X"), "force": .bool(true)]
    ))

    let list = try await tools.handle(CallTool.Parameters(
        name: "list_episodes",
        arguments: nil
    ))
    let output = text(from: list)
    let lines = output.split(separator: "\n")
    #expect(lines.count == 1)
    #expect(output.contains("2 memories"))
}

@Test func autoEpisode_crossProject_backAndForth() async throws {
    let tools = try await makeTools()

    // X → Y → X should create 3 episodes (force: true to isolate from conflict detection)
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("First X memory"), "project": .string("X")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Y memory"), "project": .string("Y"), "force": .bool(true)]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Back to X memory"), "project": .string("X"), "force": .bool(true)]
    ))

    let list = try await tools.handle(CallTool.Parameters(
        name: "list_episodes",
        arguments: nil
    ))
    let output = text(from: list)
    let lines = output.split(separator: "\n")
    #expect(lines.count == 3)
}

@Test func explicitEpisode_crossProject_doesNotSplit() async throws {
    let tools = try await makeTools()

    // Explicit episode should NOT split on project change
    _ = try await tools.handle(CallTool.Parameters(
        name: "begin_episode",
        arguments: ["title": .string("Multi-project session"), "project": .string("X")]
    ))

    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("X memory"), "project": .string("X")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Y memory"), "project": .string("Y"), "force": .bool(true)]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Z memory"), "project": .string("Z"), "force": .bool(true)]
    ))

    // Should still be one episode with all 3 memories
    let list = try await tools.handle(CallTool.Parameters(
        name: "list_episodes",
        arguments: nil
    ))
    let output = text(from: list)
    let lines = output.split(separator: "\n")
    #expect(lines.count == 1)
    #expect(output.contains("3 memories"))
    #expect(output.contains("Multi-project session"))
}

// MARK: - End Already-Ended Episode

@Test func endEpisode_alreadyEnded_returnsInfo() async throws {
    let tools = try await makeTools()

    let r1 = try await tools.handle(CallTool.Parameters(
        name: "begin_episode",
        arguments: ["title": .string("Already ended test")]
    ))
    let epId = extractEpisodeId(from: text(from: r1))!

    // End it once
    _ = try await tools.handle(CallTool.Parameters(
        name: "end_episode",
        arguments: nil
    ))

    // End it again by ID — should get informational response, not error
    let result = try await tools.handle(CallTool.Parameters(
        name: "end_episode",
        arguments: ["episode_id": .int(epId)]
    ))
    let output = text(from: result)
    #expect(result.isError != true)
    #expect(output.contains("already ended"))
}

// MARK: - Recall Episode Limit

@Test func recallEpisode_withLimit() async throws {
    let tools = try await makeTools()

    let r1 = try await tools.handle(CallTool.Parameters(
        name: "begin_episode",
        arguments: ["title": .string("Limited recall test")]
    ))
    let epId = extractEpisodeId(from: text(from: r1))!

    // Add 5 memories
    for i in 1...5 {
        _ = try await tools.handle(CallTool.Parameters(
            name: "remember",
            arguments: ["content": .string("Memory number \(i) for limit test"), "force": .bool(true)]
        ))
    }

    // Recall with limit=2 — should show 2 of 5 with truncation notice
    let result = try await tools.handle(CallTool.Parameters(
        name: "recall_episode",
        arguments: ["episode_id": .int(epId), "limit": .int(2)]
    ))
    let output = text(from: result)
    #expect(output.contains("2 of 5"))
    #expect(output.contains("Showing 2 of 5"))
    #expect(output.contains("Memory number 1"))
    #expect(output.contains("Memory number 2"))
    #expect(!output.contains("Memory number 3"))
}

// MARK: - Explicit → Explicit Episode

@Test func beginEpisode_closesExplicitEpisode() async throws {
    let tools = try await makeTools()

    // Begin first explicit episode
    _ = try await tools.handle(CallTool.Parameters(
        name: "begin_episode",
        arguments: ["title": .string("First explicit")]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Memory in first explicit")]
    ))

    // Begin second explicit — should close the first
    _ = try await tools.handle(CallTool.Parameters(
        name: "begin_episode",
        arguments: ["title": .string("Second explicit")]
    ))

    let list = try await tools.handle(CallTool.Parameters(
        name: "list_episodes",
        arguments: nil
    ))
    let output = text(from: list)
    #expect(output.contains("[ended]"))
    #expect(output.contains("[active]"))
    #expect(output.contains("First explicit"))
    #expect(output.contains("Second explicit"))

    // Verify first is ended, second is active
    let endedList = try await tools.handle(CallTool.Parameters(
        name: "list_episodes",
        arguments: ["status": .string("ended")]
    ))
    #expect(text(from: endedList).contains("First explicit"))
    #expect(!text(from: endedList).contains("Second explicit"))

    let activeList = try await tools.handle(CallTool.Parameters(
        name: "list_episodes",
        arguments: ["status": .string("active")]
    ))
    #expect(text(from: activeList).contains("Second explicit"))
    #expect(!text(from: activeList).contains("First explicit"))
}
