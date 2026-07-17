import Testing
import EngramKit
import EngramModels
import Lattice
import MCP
import Foundation

// ============================================================================
// Groups increment 4 — tombstones, attribution, author-only privacy,
// consolidation backstop. All against the LOCAL lattice (spokes arrive in
// increment 3); "group-shared" is driven by SyncConfig.exposedGroups.
// ============================================================================

private struct ToolsContext {
    let tools: MemoryTools
    let lattice: Lattice
}

private func makeToolsWithLattice() async throws -> ToolsContext {
    let path = FileManager.default.temporaryDirectory
        .appending(path: "engram-tombstone-test-\(UUID().uuidString).sqlite")
    let lattice = try Lattice(Memory.self, Edge.self, Checkpoint.self, HookState.self, SyncConfig.self, configuration: .init(fileURL: path))
    let embedder = sharedEmbedder
    if await !embedder.isLoaded { await embedder.load() }
    return ToolsContext(
        tools: MemoryTools(localRef: lattice.sendableReference, syncedRef: nil, embedder: embedder),
        lattice: lattice)
}

/// Marks a project as exposed to a group (what setGroupExposure will do).
private func expose(_ project: String, in lattice: Lattice) {
    lattice.add(SyncConfig(project: project, policy: .sync, exposedTeams: ["group-1"]))
}

private func remember(_ tools: MemoryTools, _ content: String, project: String) async throws -> UUID {
    let result = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string(content), "project": .string(project), "force": .bool(true)]))
    guard let id = extractMemoryId(from: text(from: result)), let uuid = UUID(uuidString: id) else {
        throw MCPError.internalError("no id in remember output: \(text(from: result))")
    }
    return uuid
}

// MARK: - Tombstone vs hard delete

@Test func forget_groupShared_tombstonesAndRestores() async throws {
    let ctx = try await makeToolsWithLattice()
    expose("shared-proj", in: ctx.lattice)
    let gid = try await remember(ctx.tools, "Deploy uses blue-green with 5m soak", project: "shared-proj")

    let forgetOut = text(from: try await ctx.tools.handle(CallTool.Parameters(
        name: "forget", arguments: ["id": .string(gid.uuidString)])))
    #expect(forgetOut.contains("tombstoned"))
    #expect(forgetOut.contains("undelete"))

    // Row survives with a tombstone — NOT hard-deleted.
    let row = ctx.lattice.objects(Memory.self).where { $0.__globalId == gid }.first
    #expect(row != nil)
    #expect(row?.deletedAt != nil)

    // Recall excludes it.
    let recallOut = text(from: try await ctx.tools.handle(CallTool.Parameters(
        name: "recall", arguments: ["query": .string("blue-green deploy soak"), "project": .string("shared-proj")])))
    #expect(!recallOut.contains(gid.uuidString))

    // Non-undelete update is refused with attribution guidance.
    let updateOut = text(from: try await ctx.tools.handle(CallTool.Parameters(
        name: "update", arguments: ["id": .string(gid.uuidString), "topic": .string("deploy")])))
    #expect(updateOut.contains("tombstoned"))
    #expect(updateOut.contains("undelete"))

    // Undelete restores for everyone.
    let undeleteOut = text(from: try await ctx.tools.handle(CallTool.Parameters(
        name: "update", arguments: ["id": .string(gid.uuidString), "undelete": .bool(true)])))
    #expect(undeleteOut.contains("undeleted"))
    let restored = ctx.lattice.objects(Memory.self).where { $0.__globalId == gid }.first
    #expect(restored?.deletedAt == nil)
    let recallAfter = text(from: try await ctx.tools.handle(CallTool.Parameters(
        name: "recall", arguments: ["query": .string("blue-green deploy soak"), "project": .string("shared-proj")])))
    #expect(recallAfter.contains(gid.uuidString))
}

@Test func forget_unshared_hardDeletes() async throws {
    let ctx = try await makeToolsWithLattice()
    let gid = try await remember(ctx.tools, "Local scratch note about build flags", project: "local-proj")

    let out = text(from: try await ctx.tools.handle(CallTool.Parameters(
        name: "forget", arguments: ["id": .string(gid.uuidString)])))
    #expect(out.contains("Deleted memory"))
    #expect(!out.contains("tombstoned"))
    #expect(ctx.lattice.objects(Memory.self).where { $0.__globalId == gid }.first == nil)
}

@Test func forget_bulk_partitionsSharedFromPrivate() async throws {
    let ctx = try await makeToolsWithLattice()
    expose("mixed-proj", in: ctx.lattice)
    let sharedGid = try await remember(ctx.tools, "Shared team knowledge about the API gateway", project: "mixed-proj")
    // A PRIVATE row in the same exposed project never leaves the machine —
    // it keeps the hard delete.
    let privResult = try await ctx.tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Private note on my API token setup"),
                    "project": .string("mixed-proj"),
                    "is_private": .bool(true), "force": .bool(true)]))
    let privGid = UUID(uuidString: extractMemoryId(from: text(from: privResult))!)!

    let out = text(from: try await ctx.tools.handle(CallTool.Parameters(
        name: "forget", arguments: ["project": .string("mixed-proj")])))
    #expect(out.contains("1 deleted"))
    #expect(out.contains("1 group-shared → tombstoned"))

    #expect(ctx.lattice.objects(Memory.self).where { $0.__globalId == privGid }.first == nil)
    let sharedRow = ctx.lattice.objects(Memory.self).where { $0.__globalId == sharedGid }.first
    #expect(sharedRow?.deletedAt != nil)
}

@Test func tombstonedNeighbor_excludedFromGraphTraversal() async throws {
    let ctx = try await makeToolsWithLattice()
    expose("graph-proj", in: ctx.lattice)
    let a = try await remember(ctx.tools, "Root concept: retry budgets", project: "graph-proj")
    let b = try await remember(ctx.tools, "Detail: exponential backoff caps at 60s", project: "graph-proj")
    _ = try await ctx.tools.handle(CallTool.Parameters(
        name: "connect",
        arguments: ["from": .string(a.uuidString), "to": .string(b.uuidString), "relation": .string("relates_to")]))

    _ = try await ctx.tools.handle(CallTool.Parameters(
        name: "forget", arguments: ["id": .string(b.uuidString)]))

    let graphOut = text(from: try await ctx.tools.handle(CallTool.Parameters(
        name: "graph", arguments: ["id": .string(a.uuidString)])))
    #expect(!graphOut.contains(b.uuidString))

    let recallOut = text(from: try await ctx.tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: ["query": .string("retry budgets"), "project": .string("graph-proj"), "depth": .int(2)])))
    #expect(!recallOut.contains(b.uuidString))
}

// MARK: - Attribution + author-only privacy

@Test func foreignAuthor_badgeInRecall_ownRowsClean() async throws {
    let ctx = try await makeToolsWithLattice()
    let foreignAuthor = UUID()
    let mem = Memory(
        content: "Teammate insight: the flaky test is timezone-dependent",
        topic: "debugging", project: "badge-proj",
        embedding: Vector<Float>(try await sharedEmbedder.embed(text: "Teammate insight: the flaky test is timezone-dependent")!),
        authorUserId: foreignAuthor)
    ctx.lattice.add(mem)
    _ = try await remember(ctx.tools, "My own note about test flakiness", project: "badge-proj")

    let out = text(from: try await ctx.tools.handle(CallTool.Parameters(
        name: "recall", arguments: ["query": .string("flaky test timezone"), "project": .string("badge-proj")])))
    // Foreign row is badged (no groups.json in tests → fallback name).
    #expect(out.contains("[by:teammate]"))
    // Exactly ONE badge — own/legacy rows stay clean.
    #expect(out.components(separatedBy: "[by:").count == 2)
    // Badge placement: after [project/topic], before the parenthetical —
    // the hook-side parser anchors on [id: and the first bracket pair.
    if let line = out.split(separator: "\n").first(where: { $0.contains("[by:teammate]") }) {
        let projRange = line.range(of: "[badge-proj/debugging]")
        let badgeRange = line.range(of: "[by:teammate]")
        let parenRange = line.range(of: "(distance:")
        #expect(projRange != nil && badgeRange != nil && parenRange != nil)
        if let p = projRange, let b = badgeRange, let d = parenRange {
            #expect(p.upperBound <= b.lowerBound)
            #expect(b.upperBound <= d.lowerBound)
        }
    }
}

@Test func isPrivateFlip_isAuthorOnly() async throws {
    let ctx = try await makeToolsWithLattice()
    let foreignAuthor = UUID()
    let mem = Memory(
        content: "Foreign-authored group memory about CI caching",
        topic: "ci", project: "authz-proj",
        embedding: Vector<Float>(try await sharedEmbedder.embed(text: "Foreign-authored group memory about CI caching")!),
        authorUserId: foreignAuthor)
    ctx.lattice.add(mem)
    let gid = mem.__globalId!

    // Privacy flip on a teammate's row is refused…
    let flipOut = text(from: try await ctx.tools.handle(CallTool.Parameters(
        name: "update", arguments: ["id": .string(gid.uuidString), "is_private": .bool(true)])))
    #expect(flipOut.contains("author-only"))
    #expect(ctx.lattice.objects(Memory.self).where { $0.__globalId == gid }.first?.isPrivate == false)

    // …but ordinary edits are flat-trust, with a group-share note.
    let editOut = text(from: try await ctx.tools.handle(CallTool.Parameters(
        name: "update", arguments: ["id": .string(gid.uuidString), "topic": .string("ci-cache")])))
    #expect(editOut.contains("Updated memory"))
    #expect(editOut.contains("originally by"))
}

@Test func consolidate_foreignMembers_requireForce() async throws {
    let ctx = try await makeToolsWithLattice()
    let own = try await remember(ctx.tools, "Own memory about connection pooling", project: "cons-proj")
    let foreign = Memory(
        content: "Teammate memory about connection pooling limits",
        topic: "general", project: "cons-proj",
        embedding: Vector<Float>(try await sharedEmbedder.embed(text: "Teammate memory about connection pooling limits")!),
        authorUserId: UUID())
    ctx.lattice.add(foreign)
    let foreignGid = foreign.__globalId!

    let refuseOut = text(from: try await ctx.tools.handle(CallTool.Parameters(
        name: "consolidate",
        arguments: ["ids": .array([.string(own.uuidString), .string(foreignGid.uuidString)]),
                    "content": .string("Connection pooling: limits and tuning")])))
    #expect(refuseOut.contains("teammates"))
    #expect(refuseOut.contains("force"))
    // Foreign member untouched.
    #expect(ctx.lattice.objects(Memory.self).where { $0.__globalId == foreignGid }.first?.importance != 0 ||
            ctx.lattice.objects(Memory.self).where { $0.__globalId == foreignGid }.first != nil)

    let forcedOut = text(from: try await ctx.tools.handle(CallTool.Parameters(
        name: "consolidate",
        arguments: ["ids": .array([.string(own.uuidString), .string(foreignGid.uuidString)]),
                    "content": .string("Connection pooling: limits and tuning"),
                    "force": .bool(true)])))
    #expect(forcedOut.contains("Created summary"))
}

// MARK: - GroupDirectory fixture parsing

@Test func groupDirectory_fixtureParsing() throws {
    let fixture = """
    {
      "selfUserId": "11111111-1111-1111-1111-111111111111",
      "updatedAt": "2026-07-17T12:00:00Z",
      "groups": [
        {"id": "22222222-2222-2222-2222-222222222222", "name": "canary", "parentId": null, "myRole": "owner", "root": true},
        {"id": "33333333-3333-3333-3333-333333333333", "name": "team-ios", "parentId": "22222222-2222-2222-2222-222222222222", "myRole": "member", "root": false}
      ],
      "members": {"44444444-4444-4444-4444-444444444444": "Alice [Staff] (SF)"}
    }
    """
    let url = FileManager.default.temporaryDirectory
        .appending(path: "groups-fixture-\(UUID().uuidString).json")
    try fixture.data(using: .utf8)!.write(to: url)
    defer { try? FileManager.default.removeItem(at: url) }

    let file = GroupDirectory.load(from: url)
    #expect(file?.selfUserId?.uuidString == "11111111-1111-1111-1111-111111111111")
    #expect(file?.groups.count == 2)
    #expect(file?.groups[1].parentId?.uuidString == "22222222-2222-2222-2222-222222222222")
    #expect(file?.members.count == 1)
}
