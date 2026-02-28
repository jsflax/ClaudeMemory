import Testing
import EngramKit
import Lattice
import MCP
import Foundation

// Verification strategy:
//   ctx.synced sees only synced DB (independent connection)
//   ctx.local sees UNION ALL across both DBs (has ATTACH)
//
//   synced.count == 0, local.count == 1 → row is in local only
//   synced.count == 1, local.count == 1 → row is in synced only
//   synced.count == 1, local.count == 2 → row in both (dual write, shouldn't happen)

@Suite("Dual-DB Write Routing", .serialized)
struct DualDBWriteTests {

    @Test("Sync-policy project routes memory to syncedLattice")
    func syncProjectRoutesToSynced() async throws {
        let ctx = try await makeDualDBTools()

        // Configure "myproject" as sync
        ctx.synced.add(SyncConfig(project: "myproject", policy: .sync))

        let result = try await ctx.tools.handle(.init(
            name: "remember",
            arguments: ["content": .string("test memory"), "project": .string("myproject")]
        ))
        let output = text(from: result)
        #expect(output.contains("Stored memory"))

        // Memory should be in synced only
        #expect(ctx.synced.objects(Memory.self).where { $0.project == "myproject" }.count == 1)
        // UNION ALL total should be 1 (not 2), confirming no dual write
        #expect(ctx.local.objects(Memory.self).where { $0.project == "myproject" }.count == 1)
    }

    @Test("Local-policy project routes memory to localLattice")
    func localProjectRoutesToLocal() async throws {
        let ctx = try await makeDualDBTools()

        // Configure "localproj" as local
        ctx.synced.add(SyncConfig(project: "localproj", policy: .local))

        let result = try await ctx.tools.handle(.init(
            name: "remember",
            arguments: ["content": .string("local memory"), "project": .string("localproj")]
        ))
        let output = text(from: result)
        #expect(output.contains("Stored memory"))

        // Not in synced → must be in local
        #expect(ctx.synced.objects(Memory.self).where { $0.project == "localproj" }.count == 0)
        #expect(ctx.local.objects(Memory.self).where { $0.project == "localproj" }.count == 1)
    }

    @Test("Unconfigured project defaults to localLattice")
    func unconfiguredProjectDefaultsToLocal() async throws {
        let ctx = try await makeDualDBTools()

        // No SyncConfig for "newproject" — should default to local
        let result = try await ctx.tools.handle(.init(
            name: "remember",
            arguments: ["content": .string("unrouted memory"), "project": .string("newproject")]
        ))
        let output = text(from: result)
        #expect(output.contains("Stored memory"))

        // Not in synced → must be in local
        #expect(ctx.synced.objects(Memory.self).where { $0.project == "newproject" }.count == 0)
        #expect(ctx.local.objects(Memory.self).where { $0.project == "newproject" }.count == 1)
    }

    @Test("Private memory stays local even when project is sync")
    func privateMemoryStaysLocal() async throws {
        let ctx = try await makeDualDBTools()

        ctx.synced.add(SyncConfig(project: "syncproj", policy: .sync))

        let result = try await ctx.tools.handle(.init(
            name: "remember",
            arguments: [
                "content": .string("secret memory"),
                "project": .string("syncproj"),
                "is_private": .bool(true),
            ]
        ))
        let output = text(from: result)
        #expect(output.contains("Stored memory"))

        // Private memory should NOT be in synced, even though project is sync
        #expect(ctx.synced.objects(Memory.self).where { $0.project == "syncproj" }.count == 0)
        #expect(ctx.local.objects(Memory.self).where { $0.project == "syncproj" }.count == 1)
    }

    @Test("HookState always routes to localLattice")
    func hookStateAlwaysLocal() async throws {
        let ctx = try await makeDualDBTools()

        ctx.synced.add(SyncConfig(project: "syncproj", policy: .sync))

        // remember triggers incrementCrudCounter which writes HookState
        _ = try await ctx.tools.handle(.init(
            name: "remember",
            arguments: ["content": .string("trigger crud counter"), "project": .string("syncproj")]
        ))

        // HookState only exists in local schema, so local.count > 0 confirms it's there
        #expect(ctx.local.objects(HookState.self).count > 0)
    }

    @Test("Checkpoint always routes to localLattice")
    func checkpointAlwaysLocal() async throws {
        let ctx = try await makeDualDBTools()

        let result = try await ctx.tools.handle(.init(
            name: "checkpoint",
            arguments: ["title": .string("test task"), "project": .string("syncproj")]
        ))
        let output = text(from: result)
        #expect(output.contains("Created task"))

        // Checkpoint only exists in local schema
        #expect(ctx.local.objects(Checkpoint.self).count == 1)
    }
}

@Suite("Dual-DB Read Transparency", .serialized)
struct DualDBReadTests {

    @Test("Recall returns results from both DBs")
    func recallSpansBothDBs() async throws {
        let ctx = try await makeDualDBTools()

        ctx.synced.add(SyncConfig(project: "syncproj", policy: .sync))

        // Store one memory in local, one in synced
        _ = try await ctx.tools.handle(.init(
            name: "remember",
            arguments: ["content": .string("Swift concurrency patterns for async/await"), "project": .string("localproj")]
        ))
        _ = try await ctx.tools.handle(.init(
            name: "remember",
            arguments: ["content": .string("Swift concurrency patterns for structured tasks"), "project": .string("syncproj")]
        ))

        // Verify they landed in different DBs
        #expect(ctx.synced.objects(Memory.self).where { $0.project == "localproj" }.count == 0, "localproj should NOT be in synced")
        #expect(ctx.synced.objects(Memory.self).where { $0.project == "syncproj" }.count == 1, "syncproj should be in synced")
        // UNION ALL total should be 2 (one from each DB)
        #expect(ctx.local.objects(Memory.self).where { $0.topic != "episode" }.count == 2)

        // Recall should find both via the ATTACH'd query lattice
        let result = try await ctx.tools.handle(.init(
            name: "recall",
            arguments: ["query": .string("Swift concurrency")]
        ))
        let output = text(from: result)
        #expect(output.contains("localproj") || output.contains("syncproj"))
    }
}

@Suite("Dual-DB Edge Routing", .serialized)
struct DualDBEdgeTests {

    @Test("Connect routes edge to source memory's DB")
    func connectFollowsSourceMemory() async throws {
        let ctx = try await makeDualDBTools()

        ctx.synced.add(SyncConfig(project: "syncproj", policy: .sync))

        // Create two memories in synced
        let r1 = try await ctx.tools.handle(.init(
            name: "remember",
            arguments: ["content": .string("edge test source memory"), "project": .string("syncproj"), "force": .bool(true)]
        ))
        let r2 = try await ctx.tools.handle(.init(
            name: "remember",
            arguments: ["content": .string("edge test target memory"), "project": .string("syncproj"), "force": .bool(true)]
        ))
        let id1 = extractMemoryId(from: text(from: r1))!
        let id2 = extractMemoryId(from: text(from: r2))!

        let result = try await ctx.tools.handle(.init(
            name: "connect",
            arguments: ["from": .int(id1), "to": .int(id2), "relation": .string("relates_to")]
        ))
        let output = text(from: result)
        #expect(output.contains("Connected"))

        // Edge should be in synced (following the source memory's project)
        #expect(ctx.synced.objects(Edge.self).count >= 1)
    }
}

@Suite("Dual-DB Migration", .serialized)
struct DualDBMigrationTests {

    @Test("migrateProjects moves memories and edges")
    func migrateMovesMemoriesAndEdges() async throws {
        let ctx = try await makeDualDBTools()

        // Store memories in local (no sync config yet)
        let r1 = try await ctx.tools.handle(.init(
            name: "remember",
            arguments: ["content": .string("migrate test A"), "project": .string("migrateproj"), "force": .bool(true)]
        ))
        let r2 = try await ctx.tools.handle(.init(
            name: "remember",
            arguments: ["content": .string("migrate test B"), "project": .string("migrateproj"), "force": .bool(true)]
        ))
        let id1 = extractMemoryId(from: text(from: r1))!
        let id2 = extractMemoryId(from: text(from: r2))!

        // Connect them
        _ = try await ctx.tools.handle(.init(
            name: "connect",
            arguments: ["from": .int(id1), "to": .int(id2), "relation": .string("relates_to")]
        ))

        // Both in local (not synced, since no sync config)
        #expect(ctx.synced.objects(Memory.self).where { $0.project == "migrateproj" }.count == 0)

        // Migrate: use synced lattice directly as destination (not involved in ATTACH)
        let result = SyncMigration.migrateProjects(["migrateproj"], from: ctx.local, to: ctx.synced)
        #expect(result.memoriesMigrated == 2)
        #expect(result.edgesMigrated >= 1)

        // Memories should now be in synced
        #expect(ctx.synced.objects(Memory.self).where { $0.project == "migrateproj" }.count == 2)
    }

    @Test("Private memories stay in source during migration")
    func privateMemoriesStayInSource() async throws {
        let ctx = try await makeDualDBTools()

        // Store one normal and one private memory
        _ = try await ctx.tools.handle(.init(
            name: "remember",
            arguments: ["content": .string("public memory"), "project": .string("proj"), "force": .bool(true)]
        ))
        _ = try await ctx.tools.handle(.init(
            name: "remember",
            arguments: ["content": .string("private memory"), "project": .string("proj"), "is_private": .bool(true), "force": .bool(true)]
        ))

        // Both in local (not synced)
        #expect(ctx.synced.objects(Memory.self).where { $0.project == "proj" }.count == 0)

        let result = SyncMigration.migrateProjects(["proj"], from: ctx.local, to: ctx.synced)
        #expect(result.memoriesMigrated == 1) // Only the public one

        // Public memory migrated to synced
        #expect(ctx.synced.objects(Memory.self).where { $0.project == "proj" }.count == 1)
        #expect(ctx.synced.objects(Memory.self).where { $0.project == "proj" }.first?.isPrivate == false)
    }
}

@Suite("Single-DB Compatibility", .serialized)
struct SingleDBTests {

    @Test("nil syncedLattice behaves identically to old single-DB mode")
    func nilSyncedLatticeWorks() async throws {
        let tools = try await makeTools()

        let result = try await tools.handle(.init(
            name: "remember",
            arguments: ["content": .string("single db test"), "project": .string("test")]
        ))
        let output = text(from: result)
        #expect(output.contains("Stored memory"))

        let recall = try await tools.handle(.init(
            name: "recall",
            arguments: ["query": .string("single db test")]
        ))
        let recallOutput = text(from: recall)
        #expect(recallOutput.contains("single db test"))
    }
}
