import Testing
import EngramKit
import Lattice
import MCP
import Foundation

// MARK: - Read Routing

@Test func dualDB_localProject_readsFromLocal() async throws {
    let ctx = try await makeDualDBTools()

    // Store a memory via the tools (always writes to localLattice)
    let r = try await ctx.tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Local project memory about auth tokens"),
            "project": .string("LocalProj"),
        ]
    ))
    let id = extractId(from: text(from: r))!

    // No SyncConfig → project is local by default
    // Recall should find it in localLattice
    let recall = try await ctx.tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: ["query": .string("auth tokens"), "project": .string("LocalProj")]
    ))
    let output = text(from: recall)
    #expect(output.contains("auth tokens"))
    #expect(output.contains("[id:\(id)]"))
}

@Test func dualDB_syncedProject_readsFromSynced() async throws {
    let ctx = try await makeDualDBTools()

    // Configure "SyncedProj" as synced
    try ctx.localLattice.add(SyncConfig(project: "SyncedProj", policy: .sync))

    // Manually insert a memory into the synced database (simulating cross-device data)
    let embedder = sharedEmbedder
    if await !embedder.isLoaded { await embedder.load() }
    let crossDeviceMemory = Memory(
        content: "Cross-device memory about deployment pipeline",
        topic: "devops",
        project: "SyncedProj",
        embedding: try await Vector(embedder.embed(text: "Cross-device memory about deployment pipeline")!),
        importance: 3
    )
    try ctx.syncedLattice.add(crossDeviceMemory)

    // Recall should route to syncedLattice and find the cross-device memory
    let recall = try await ctx.tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: ["query": .string("deployment pipeline"), "project": .string("SyncedProj")]
    ))
    let output = text(from: recall)
    #expect(output.contains("deployment pipeline"))
}

@Test func dualDB_defaultFallback_routesSynced() async throws {
    let ctx = try await makeDualDBTools()

    // Set _default policy to sync (no per-project override)
    try ctx.localLattice.add(SyncConfig(project: "_default", policy: .sync))

    // Insert a memory directly into synced DB
    let embedder = sharedEmbedder
    if await !embedder.isLoaded { await embedder.load() }
    let mem = Memory(
        content: "Default-synced memory about caching strategy",
        topic: "architecture",
        project: "AnyProject",
        embedding: try await Vector(embedder.embed(text: "Default-synced memory about caching strategy")!),
        importance: 3
    )
    try ctx.syncedLattice.add(mem)

    // Recall for "AnyProject" should route to syncedLattice via _default fallback
    let recall = try await ctx.tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: ["query": .string("caching strategy"), "project": .string("AnyProject")]
    ))
    #expect(text(from: recall).contains("caching strategy"))
}

@Test func dualDB_explicitLocalOverridesDefault() async throws {
    let ctx = try await makeDualDBTools()

    // Default is sync, but "LocalProj" is explicitly local
    try ctx.localLattice.add(SyncConfig(project: "_default", policy: .sync))
    try ctx.localLattice.add(SyncConfig(project: "LocalProj", policy: .local))

    // Store via tools → goes to localLattice
    _ = try await ctx.tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Explicitly local project memory"),
            "project": .string("LocalProj"),
        ]
    ))

    // Recall should read from localLattice (explicit .local overrides _default .sync)
    let recall = try await ctx.tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: ["query": .string("explicitly local"), "project": .string("LocalProj")]
    ))
    #expect(text(from: recall).contains("Explicitly local"))
}

// MARK: - Write Routing

@Test func dualDB_writesAlwaysGoToLocal() async throws {
    let ctx = try await makeDualDBTools()

    // Configure project as synced
    try ctx.localLattice.add(SyncConfig(project: "SyncedProj", policy: .sync))

    // Store a memory — should go to localLattice even though project is synced
    _ = try await ctx.tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("New memory written to local regardless of sync config"),
            "project": .string("SyncedProj"),
        ]
    ))

    // Verify it's in localLattice
    let localMemories = ctx.localLattice.objects(Memory.self)
        .where({ $0.project == "SyncedProj" })
    #expect(localMemories.count >= 1)
    #expect(localMemories.first?.content.contains("written to local") == true)

    // Verify it's NOT in syncedLattice
    let syncedMemories = ctx.syncedLattice.objects(Memory.self)
        .where({ $0.project == "SyncedProj" })
    #expect(syncedMemories.count == 0)
}

// MARK: - findMemory (by UUID)

@Test func dualDB_findMemory_prefersLocal() async throws {
    let ctx = try await makeDualDBTools()

    // Store via tools → goes to localLattice
    let r = try await ctx.tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Memory findable by UUID")]
    ))
    let id = extractId(from: text(from: r))!

    // Forget should find it in localLattice
    let forget = try await ctx.tools.handle(CallTool.Parameters(
        name: "forget",
        arguments: ["id": .string(id)]
    ))
    #expect(text(from: forget).contains("Deleted"))
}

@Test func dualDB_findMemory_fallsBackToSynced() async throws {
    let ctx = try await makeDualDBTools()

    // Insert a memory directly into syncedLattice (simulating cross-device)
    let embedder = sharedEmbedder
    if await !embedder.isLoaded { await embedder.load() }
    let mem = Memory(
        content: "Cross-device memory only in synced DB",
        project: "SyncedProj",
        embedding: try await Vector(embedder.embed(text: "Cross-device memory only in synced DB")!)
    )
    try ctx.syncedLattice.add(mem)
    let globalId = mem.globalId!.uuidString

    // Graph lookup should find it in syncedLattice via fallback
    let graph = try await ctx.tools.handle(CallTool.Parameters(
        name: "graph",
        arguments: ["id": .string(globalId)]
    ))
    let output = text(from: graph)
    #expect(output.contains("Cross-device memory"))
}

// MARK: - Stats & List Topics Routing

@Test func dualDB_stats_routesByProject() async throws {
    let ctx = try await makeDualDBTools()

    try ctx.localLattice.add(SyncConfig(project: "SyncedProj", policy: .sync))

    // Add 2 memories to syncedLattice
    let embedder = sharedEmbedder
    if await !embedder.isLoaded { await embedder.load() }

    for i in 1...2 {
        let m = Memory(
            content: "Synced stat memory \(i)",
            topic: "testing",
            project: "SyncedProj",
            embedding: try await Vector(embedder.embed(text: "Synced stat memory \(i)")!)
        )
        try ctx.syncedLattice.add(m)
    }

    // Add 1 memory to localLattice in a different project
    _ = try await ctx.tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Local stat memory"),
            "project": .string("LocalProj"),
        ]
    ))

    // Stats for SyncedProj should show 2 (from syncedLattice)
    let syncedStats = try await ctx.tools.handle(CallTool.Parameters(
        name: "stats",
        arguments: ["project": .string("SyncedProj")]
    ))
    #expect(text(from: syncedStats).contains("2"))

    // Stats for LocalProj should show 1 (from localLattice)
    let localStats = try await ctx.tools.handle(CallTool.Parameters(
        name: "stats",
        arguments: ["project": .string("LocalProj")]
    ))
    #expect(text(from: localStats).contains("1"))
}

@Test func dualDB_listTopics_routesByProject() async throws {
    let ctx = try await makeDualDBTools()

    try ctx.localLattice.add(SyncConfig(project: "SyncedProj", policy: .sync))

    // Add a memory with topic "cross-device-arch" to syncedLattice
    let embedder = sharedEmbedder
    if await !embedder.isLoaded { await embedder.load() }

    let m = Memory(
        content: "Architecture decision from another device",
        topic: "cross-device-arch",
        project: "SyncedProj",
        embedding: try await Vector(embedder.embed(text: "Architecture decision from another device")!)
    )
    try ctx.syncedLattice.add(m)

    // list_topics for SyncedProj should find the topic from syncedLattice
    let topics = try await ctx.tools.handle(CallTool.Parameters(
        name: "list_topics",
        arguments: ["project": .string("SyncedProj")]
    ))
    #expect(text(from: topics).contains("cross-device-arch"))
}

// MARK: - No Synced DB (nil)

@Test func dualDB_nilSynced_routesEverythingToLocal() async throws {
    // Use the standard single-DB makeTools (syncedRef: nil)
    let tools = try await makeTools()

    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Memory in single-DB mode"),
            "project": .string("AnyProject"),
        ]
    ))

    let recall = try await tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: ["query": .string("single-DB mode"), "project": .string("AnyProject")]
    ))
    #expect(text(from: recall).contains("single-DB mode"))
}

// MARK: - Nil project routes to local

@Test func dualDB_nilProject_readsFromLocal() async throws {
    let ctx = try await makeDualDBTools()

    // Even with _default sync, nil project → localLattice
    try ctx.localLattice.add(SyncConfig(project: "_default", policy: .sync))

    _ = try await ctx.tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Unscoped memory without project")]
    ))

    // Recall without project filter should find it in localLattice
    let recall = try await ctx.tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: ["query": .string("unscoped memory without project")]
    ))
    #expect(text(from: recall).contains("Unscoped memory"))
}

// MARK: - Graph operations across DBs

@Test func dualDB_connect_writesEdgeToLocal() async throws {
    let ctx = try await makeDualDBTools()

    // Create two memories (both in localLattice)
    let r1 = try await ctx.tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Memory A for dual-DB edge test")]
    ))
    let r2 = try await ctx.tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Memory B for dual-DB edge test"), "force": .bool(true)]
    ))
    let id1 = extractId(from: text(from: r1))!
    let id2 = extractId(from: text(from: r2))!

    // Connect them
    let connect = try await ctx.tools.handle(CallTool.Parameters(
        name: "connect",
        arguments: [
            "from": .string(id1),
            "to": .string(id2),
            "relation": .string("relates_to"),
        ]
    ))
    #expect(text(from: connect).contains("Connected"))

    // Edge should be in localLattice
    let localEdges = ctx.localLattice.objects(Edge.self)
    #expect(localEdges.count >= 1)

    // No edges in syncedLattice
    let syncedEdges = ctx.syncedLattice.objects(Edge.self)
    #expect(syncedEdges.count == 0)
}

// MARK: - Forget across DBs

@Test func dualDB_forget_deletesFromSyncedDB() async throws {
    let ctx = try await makeDualDBTools()

    // Insert memory directly into syncedLattice
    let embedder = sharedEmbedder
    if await !embedder.isLoaded { await embedder.load() }
    let mem = Memory(
        content: "Cross-device memory to forget",
        project: "SyncedProj",
        embedding: try await Vector(embedder.embed(text: "Cross-device memory to forget")!)
    )
    try ctx.syncedLattice.add(mem)
    let globalId = mem.globalId!.uuidString

    // Forget should find and delete it from syncedLattice
    let forget = try await ctx.tools.handle(CallTool.Parameters(
        name: "forget",
        arguments: ["id": .string(globalId)]
    ))
    #expect(text(from: forget).contains("Deleted"))

    // Verify it's gone from syncedLattice
    let remaining = ctx.syncedLattice.objects(Memory.self)
        .where({ $0.globalId == mem.globalId })
    #expect(remaining.count == 0)
}

// MARK: - Overlapping Subset (deduplication via .distinct)

/// Tests the real-world scenario where local and synced DBs have overlapping data:
/// - 2 memories exist in BOTH DBs (same globalId, simulating relayed data)
/// - 1 memory exists only in localLattice (just written, not yet relayed)
/// - 1 memory exists only in syncedLattice (from another device, not yet pulled)
/// Recall on a synced project should return all 4 unique memories, deduplicating the 2 shared ones.
@Test func dualDB_overlappingSubset_deduplicatesOnRecall() async throws {
    let ctx = try await makeDualDBTools()
    try ctx.localLattice.add(SyncConfig(project: "SharedProj", policy: .sync))

    let embedder = sharedEmbedder
    if await !embedder.isLoaded { await embedder.load() }

    // --- Shared memory 1: exists in BOTH DBs with same globalId ---
    let shared1 = Memory(
        content: "Shared memory alpha about API design patterns",
        topic: "architecture",
        project: "SharedProj",
        embedding: try await Vector(embedder.embed(text: "Shared memory alpha about API design patterns")!),
        importance: 3
    )
    try ctx.localLattice.add(shared1)
    let gid1 = shared1.globalId!
    let syncedCopy1 = Memory(
        content: "Shared memory alpha about API design patterns",
        topic: "architecture",
        project: "SharedProj",
        embedding: try await Vector(embedder.embed(text: "Shared memory alpha about API design patterns")!),
        importance: 3
    )
    try ctx.syncedLattice.add(syncedCopy1, preservingGlobalId: gid1)

    // --- Shared memory 2: exists in BOTH DBs with same globalId ---
    let shared2 = Memory(
        content: "Shared memory beta about database indexing strategies",
        topic: "architecture",
        project: "SharedProj",
        embedding: try await Vector(embedder.embed(text: "Shared memory beta about database indexing strategies")!),
        importance: 3
    )
    try ctx.localLattice.add(shared2)
    let gid2 = shared2.globalId!
    let syncedCopy2 = Memory(
        content: "Shared memory beta about database indexing strategies",
        topic: "architecture",
        project: "SharedProj",
        embedding: try await Vector(embedder.embed(text: "Shared memory beta about database indexing strategies")!),
        importance: 3
    )
    try ctx.syncedLattice.add(syncedCopy2, preservingGlobalId: gid2)

    // --- Local-only memory: just written, not yet relayed to synced DB ---
    let localOnly = Memory(
        content: "Local only memory about pending refactoring work",
        topic: "architecture",
        project: "SharedProj",
        embedding: try await Vector(embedder.embed(text: "Local only memory about pending refactoring work")!),
        importance: 3
    )
    try ctx.localLattice.add(localOnly)
    let gidLocal = localOnly.globalId!

    // --- Synced-only memory: from another device, not in local DB ---
    let syncedOnly = Memory(
        content: "Synced only memory about deployment automation from laptop",
        topic: "architecture",
        project: "SharedProj",
        embedding: try await Vector(embedder.embed(text: "Synced only memory about deployment automation from laptop")!),
        importance: 3
    )
    try ctx.syncedLattice.add(syncedOnly)
    let gidSynced = syncedOnly.globalId!

    // Raw counts: local has 3, synced has 3, but 2 are shared → 4 unique
    #expect(ctx.localLattice.objects(Memory.self).count == 3)
    #expect(ctx.syncedLattice.objects(Memory.self).count == 3)

    // Recall should return all 4 unique memories (2 shared deduped + 1 local-only + 1 synced-only)
    let recall = try await ctx.tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: [
            "query": .string("architecture patterns design indexing refactoring deployment"),
            "project": .string("SharedProj"),
            "limit": .int(10),
        ]
    ))
    let output = text(from: recall)

    // All 4 unique globalIds should appear exactly once
    #expect(output.contains(gid1.uuidString))
    #expect(output.contains(gid2.uuidString))
    #expect(output.contains(gidLocal.uuidString))
    #expect(output.contains(gidSynced.uuidString))

    // Each shared memory should appear once (not twice)
    let gid1Count = output.components(separatedBy: gid1.uuidString).count - 1
    let gid2Count = output.components(separatedBy: gid2.uuidString).count - 1
    #expect(gid1Count == 1, "Shared memory 1 should appear exactly once, not \(gid1Count)")
    #expect(gid2Count == 1, "Shared memory 2 should appear exactly once, not \(gid2Count)")
}

/// Tests that stats correctly count deduplicated memories across overlapping DBs.
@Test func dualDB_overlappingSubset_statsCountCorrectly() async throws {
    let ctx = try await makeDualDBTools()
    try ctx.localLattice.add(SyncConfig(project: "SharedProj", policy: .sync))

    let embedder = sharedEmbedder
    if await !embedder.isLoaded { await embedder.load() }

    // 1 shared memory (same globalId in both DBs)
    let shared = Memory(
        content: "Shared stats memory",
        topic: "testing",
        project: "SharedProj",
        embedding: try await Vector(embedder.embed(text: "Shared stats memory")!),
        importance: 3
    )
    try ctx.localLattice.add(shared)
    let gid = shared.globalId!
    let syncedCopy = Memory(
        content: "Shared stats memory",
        topic: "testing",
        project: "SharedProj",
        embedding: try await Vector(embedder.embed(text: "Shared stats memory")!),
        importance: 3
    )
    try ctx.syncedLattice.add(syncedCopy, preservingGlobalId: gid)

    // 1 local-only
    let localOnly = Memory(
        content: "Local only stats memory",
        topic: "testing",
        project: "SharedProj",
        embedding: try await Vector(embedder.embed(text: "Local only stats memory")!),
        importance: 3
    )
    try ctx.localLattice.add(localOnly)

    // 1 synced-only
    let syncedOnly = Memory(
        content: "Synced only stats memory",
        topic: "testing",
        project: "SharedProj",
        embedding: try await Vector(embedder.embed(text: "Synced only stats memory")!),
        importance: 3
    )
    try ctx.syncedLattice.add(syncedOnly)

    // Raw: local=2, synced=2, unique=3
    // Stats should report 3, not 4
    let stats = try await ctx.tools.handle(CallTool.Parameters(
        name: "stats",
        arguments: ["project": .string("SharedProj")]
    ))
    let output = text(from: stats)
    #expect(output.contains("3"), "Stats should show 3 unique memories, got: \(output)")
}

/// Tests that list_topics correctly deduplicates when the same memory with the same globalId
/// appears in both DBs under the same topic.
@Test func dualDB_overlappingSubset_listTopicsDeduplicates() async throws {
    let ctx = try await makeDualDBTools()
    try ctx.localLattice.add(SyncConfig(project: "SharedProj", policy: .sync))

    let embedder = sharedEmbedder
    if await !embedder.isLoaded { await embedder.load() }

    // 2 memories in topic "debugging": 1 shared (same globalId), 1 synced-only
    let shared = Memory(
        content: "Debug shared memory",
        topic: "debugging",
        project: "SharedProj",
        embedding: try await Vector(embedder.embed(text: "Debug shared memory")!),
        importance: 3
    )
    try ctx.localLattice.add(shared)
    let gid = shared.globalId!
    try ctx.syncedLattice.add(Memory(
        content: "Debug shared memory",
        topic: "debugging",
        project: "SharedProj",
        embedding: try await Vector(embedder.embed(text: "Debug shared memory")!),
        importance: 3
    ), preservingGlobalId: gid)

    let syncedOnly = Memory(
        content: "Debug synced-only memory",
        topic: "debugging",
        project: "SharedProj",
        embedding: try await Vector(embedder.embed(text: "Debug synced-only memory")!),
        importance: 3
    )
    try ctx.syncedLattice.add(syncedOnly)

    // list_topics should show debugging: 2 (not 3)
    let topics = try await ctx.tools.handle(CallTool.Parameters(
        name: "list_topics",
        arguments: ["project": .string("SharedProj")]
    ))
    let output = text(from: topics)
    #expect(output.contains("debugging"))
    #expect(output.contains("2"), "debugging topic should have 2 unique memories, got: \(output)")
}
