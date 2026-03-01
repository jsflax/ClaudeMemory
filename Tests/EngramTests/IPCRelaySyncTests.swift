import Testing
import EngramKit
import Lattice
import MCP
import Foundation

// MARK: - IPC Relay Sync Tests
//
// These tests verify the Lattice IPC relay pattern that replaces the old
// migration-based sync. The architecture:
//
//   MCP server writes to memory.db
//       ↓ (cross-process notification)
//   hubLattice (2nd connection to memory.db, IPC target + sync filter)
//       ↓ (IPC relay, filtered)
//   syncedLattice (memory_synced.db, IPC target)
//       ↓ (WSS — not tested here)
//   cloud

@Suite("IPC Relay Sync", .serialized)
struct IPCRelaySyncTests {

    /// Create an observer config from an IPC config by stripping ipcTargets.
    /// Opens a plain DB connection (no new IPC endpoints) to the same file.
    private func observerConfig(from config: Lattice.Configuration) -> Lattice.Configuration {
        var c = config
        c.ipcTargets = nil
        return c
    }

    /// Wait for a specific table+operation to arrive on a Lattice DB via changeStream.
    /// Opens a read-only view (no IPC targets) to avoid creating new IPC connections.
    private func waitForChange(
        on config: Lattice.Configuration,
        table: String,
        operation: AuditLog.Operation,
        count: Int = 1
    ) async -> Task<Void, any Error> {
        let readConfig = observerConfig(from: config)
        var task: Task<Void, any Error>!
        await withCheckedContinuation { continuation in
            task = Task.detached {
                let db = try await Lattice(Memory.self, Edge.self, SyncConfig.self, configuration: readConfig)
                let stream = db.changeStream
                continuation.resume()
                var seen = 0
                for await changes in stream {
                    let resolved = changes.compactMap { $0.resolve(on: db) }
                    seen += resolved.filter({ $0.tableName == table && $0.operation == operation }).count
                    if seen >= count {
                        return
                    }
                }
            }
        }
        return task
    }

    // MARK: - Test 1: Basic IPC relay (memory.db → synced.db)

    @Test(.timeLimit(.minutes(1)))
    func ipcRelay_memoryAppearsOnSyncedSide() async throws {
        let channel = "engram-relay-\(UUID().uuidString.prefix(8))"

        let hubURL = FileManager.default.temporaryDirectory
            .appending(path: "relay-hub-\(UUID().uuidString).sqlite")
        let syncedURL = FileManager.default.temporaryDirectory
            .appending(path: "relay-synced-\(UUID().uuidString).sqlite")

        defer {
            try? Lattice.delete(for: .init(fileURL: hubURL))
            try? Lattice.delete(for: .init(fileURL: syncedURL))
        }

        // Hub: memory.db with IPC target (no filter — sync everything)
        var hubConfig = Lattice.Configuration(fileURL: hubURL)
        hubConfig.ipcTargets = [.init(channel: channel)]
        let hub = try Lattice(Memory.self, Edge.self, SyncConfig.self, configuration: hubConfig)

        try await Task.sleep(for: .milliseconds(100))

        // Synced: memory_synced.db with same IPC channel
        var syncedConfig = Lattice.Configuration(fileURL: syncedURL)
        syncedConfig.ipcTargets = [.init(channel: channel)]
        let synced = try Lattice(Memory.self, Edge.self, SyncConfig.self, configuration: syncedConfig)

        try await Task.sleep(for: .milliseconds(200))

        let task = await waitForChange(on: syncedConfig, table: "Memory", operation: .insert)

        // Write a memory on the hub side (simulates MCP server write)
        hub.add(Memory(
            content: "Test memory for IPC relay",
            topic: "testing",
            project: "Engram",
            embedding: Vector<Float>([Float](repeating: 0.1, count: 384))
        ))

        try await task.value

        let syncedMemories = synced.objects(Memory.self)
        #expect(syncedMemories.count == 1)
        #expect(syncedMemories.first?.content == "Test memory for IPC relay")
        #expect(syncedMemories.first?.project == "Engram")
    }

    // MARK: - Test 2: Filtered IPC relay (only synced projects replicate)

    @Test(.timeLimit(.minutes(1)))
    func ipcRelay_filterBlocksUnsyncedProjects() async throws {
        let channel = "engram-filt-\(UUID().uuidString.prefix(8))"

        let hubURL = FileManager.default.temporaryDirectory
            .appending(path: "relay-filt-hub-\(UUID().uuidString).sqlite")
        let syncedURL = FileManager.default.temporaryDirectory
            .appending(path: "relay-filt-synced-\(UUID().uuidString).sqlite")

        defer {
            try? Lattice.delete(for: .init(fileURL: hubURL))
            try? Lattice.delete(for: .init(fileURL: syncedURL))
        }

        // Build a sync filter: only non-private memories from "SyncedProject"
        var filter = Lattice.SyncFilter()
        filter.include(Memory.self) { mem in
            mem.isPrivate == false && mem.project == "SyncedProject"
        }
        filter.include(Edge.self)
        filter.include(SyncConfig.self)

        var hubConfig = Lattice.Configuration(fileURL: hubURL)
        hubConfig.ipcTargets = [.init(channel: channel, syncFilter: filter)]
        let hub = try Lattice(Memory.self, Edge.self, SyncConfig.self, configuration: hubConfig)

        try await Task.sleep(for: .milliseconds(100))

        var syncedConfig = Lattice.Configuration(fileURL: syncedURL)
        syncedConfig.ipcTargets = [.init(channel: channel)]
        let synced = try Lattice(Memory.self, Edge.self, SyncConfig.self, configuration: syncedConfig)

        try await Task.sleep(for: .milliseconds(200))

        let task = await waitForChange(on: syncedConfig, table: "Memory", operation: .insert)

        // Write a synced-project memory (should replicate)
        hub.add(Memory(
            content: "This should sync",
            topic: "testing",
            project: "SyncedProject",
            embedding: Vector<Float>([Float](repeating: 0.1, count: 384))
        ))

        // Write a local-only memory (should NOT replicate)
        hub.add(Memory(
            content: "This stays local",
            topic: "testing",
            project: "LocalProject",
            embedding: Vector<Float>([Float](repeating: 0.2, count: 384))
        ))

        // Wait for the synced-project insert
        try await task.value

        // Give extra time to confirm no straggler arrives
        try await Task.sleep(for: .milliseconds(500))

        // Only the synced-project memory should appear
        let syncedMemories = synced.objects(Memory.self)
        #expect(syncedMemories.count == 1)
        #expect(syncedMemories.first?.content == "This should sync")
    }

    // MARK: - Test 3: Private memories blocked by filter

    @Test(.timeLimit(.minutes(1)))
    func ipcRelay_privateMemoriesBlocked() async throws {
        let channel = "engram-priv-\(UUID().uuidString.prefix(8))"

        let hubURL = FileManager.default.temporaryDirectory
            .appending(path: "relay-priv-hub-\(UUID().uuidString).sqlite")
        let syncedURL = FileManager.default.temporaryDirectory
            .appending(path: "relay-priv-synced-\(UUID().uuidString).sqlite")

        defer {
            try? Lattice.delete(for: .init(fileURL: hubURL))
            try? Lattice.delete(for: .init(fileURL: syncedURL))
        }

        var filter = Lattice.SyncFilter()
        filter.include(Memory.self) { mem in
            mem.isPrivate == false && mem.project == "MyProject"
        }

        var hubConfig = Lattice.Configuration(fileURL: hubURL)
        hubConfig.ipcTargets = [.init(channel: channel, syncFilter: filter)]
        let hub = try Lattice(Memory.self, Edge.self, SyncConfig.self, configuration: hubConfig)

        try await Task.sleep(for: .milliseconds(100))

        var syncedConfig = Lattice.Configuration(fileURL: syncedURL)
        syncedConfig.ipcTargets = [.init(channel: channel)]
        let synced = try Lattice(Memory.self, Edge.self, SyncConfig.self, configuration: syncedConfig)

        try await Task.sleep(for: .milliseconds(200))

        let task = await waitForChange(on: syncedConfig, table: "Memory", operation: .insert)

        // Public memory — should sync
        hub.add(Memory(
            content: "Public knowledge",
            topic: "testing",
            project: "MyProject",
            embedding: Vector<Float>([Float](repeating: 0.1, count: 384)),
            isPrivate: false
        ))

        // Private memory — should NOT sync
        hub.add(Memory(
            content: "Secret stuff",
            topic: "testing",
            project: "MyProject",
            embedding: Vector<Float>([Float](repeating: 0.3, count: 384)),
            isPrivate: true
        ))

        try await task.value
        try await Task.sleep(for: .milliseconds(500))

        let syncedMemories = synced.objects(Memory.self)
        #expect(syncedMemories.count == 1)
        #expect(syncedMemories.first?.content == "Public knowledge")
    }

    // MARK: - Test 4: Bidirectional relay (synced → hub flows back)

    @Test(.timeLimit(.minutes(1)))
    func ipcRelay_bidirectionalSync() async throws {
        let channel = "engram-bidi-\(UUID().uuidString.prefix(8))"

        let hubURL = FileManager.default.temporaryDirectory
            .appending(path: "relay-bidi-hub-\(UUID().uuidString).sqlite")
        let syncedURL = FileManager.default.temporaryDirectory
            .appending(path: "relay-bidi-synced-\(UUID().uuidString).sqlite")

        defer {
            try? Lattice.delete(for: .init(fileURL: hubURL))
            try? Lattice.delete(for: .init(fileURL: syncedURL))
        }

        var hubConfig = Lattice.Configuration(fileURL: hubURL)
        hubConfig.ipcTargets = [.init(channel: channel)]
        let hub = try Lattice(Memory.self, Edge.self, SyncConfig.self, configuration: hubConfig)

        try await Task.sleep(for: .milliseconds(100))

        var syncedConfig = Lattice.Configuration(fileURL: syncedURL)
        syncedConfig.ipcTargets = [.init(channel: channel)]
        let synced = try Lattice(Memory.self, Edge.self, SyncConfig.self, configuration: syncedConfig)

        try await Task.sleep(for: .milliseconds(200))

        // Phase 1: Hub → Synced
        let task1 = await waitForChange(on: syncedConfig, table: "Memory", operation: .insert)
        hub.add(Memory(
            content: "From MCP server",
            project: "Engram",
            embedding: Vector<Float>([Float](repeating: 0.1, count: 384))
        ))
        try await task1.value
        #expect(synced.objects(Memory.self).count == 1)

        // Phase 2: Synced → Hub (simulates cloud download arriving via WSS)
        let task2 = await waitForChange(on: hubConfig, table: "Memory", operation: .insert)
        synced.add(Memory(
            content: "From cloud",
            project: "Engram",
            embedding: Vector<Float>([Float](repeating: 0.2, count: 384))
        ))
        try await task2.value
        try await Task.sleep(for: .milliseconds(500))

        #expect(hub.objects(Memory.self).count == 2)
        let contents = hub.objects(Memory.self).snapshot().map(\.content).sorted()
        #expect(contents == ["From MCP server", "From cloud"])
    }

    // MARK: - Test 5: Edge replication alongside memories

    @Test(.timeLimit(.minutes(1)))
    func ipcRelay_edgesReplicateWithMemories() async throws {
        let channel = "engram-edge-\(UUID().uuidString.prefix(8))"

        let hubURL = FileManager.default.temporaryDirectory
            .appending(path: "relay-edge-hub-\(UUID().uuidString).sqlite")
        let syncedURL = FileManager.default.temporaryDirectory
            .appending(path: "relay-edge-synced-\(UUID().uuidString).sqlite")

        defer {
            try? Lattice.delete(for: .init(fileURL: hubURL))
            try? Lattice.delete(for: .init(fileURL: syncedURL))
        }

        var filter = Lattice.SyncFilter()
        filter.include(Memory.self) { $0.isPrivate == false }
        filter.include(Edge.self)

        var hubConfig = Lattice.Configuration(fileURL: hubURL)
        hubConfig.ipcTargets = [.init(channel: channel, syncFilter: filter)]
        let hub = try Lattice(Memory.self, Edge.self, SyncConfig.self, configuration: hubConfig)

        try await Task.sleep(for: .milliseconds(100))

        var syncedConfig = Lattice.Configuration(fileURL: syncedURL)
        syncedConfig.ipcTargets = [.init(channel: channel)]
        let synced = try Lattice(Memory.self, Edge.self, SyncConfig.self, configuration: syncedConfig)

        try await Task.sleep(for: .milliseconds(200))

        // Add two memories, wait for both to arrive
        let memTask = await waitForChange(on: syncedConfig, table: "Memory", operation: .insert, count: 2)

        let m1 = Memory(
            content: "Architecture decision",
            project: "Engram",
            embedding: Vector<Float>([Float](repeating: 0.1, count: 384))
        )
        let m2 = Memory(
            content: "Implementation detail",
            project: "Engram",
            embedding: Vector<Float>([Float](repeating: 0.2, count: 384))
        )
        hub.add(m1)
        hub.add(m2)

        try await memTask.value
        #expect(synced.objects(Memory.self).count == 2)

        // Add an edge connecting them
        let edgeTask = await waitForChange(on: syncedConfig, table: "Edge", operation: .insert)
        let edge = Edge(sourceGlobalId: m1.__globalId!, targetGlobalId: m2.__globalId!, relation: .partOf)
        hub.add(edge)

        try await edgeTask.value

        #expect(synced.objects(Memory.self).count == 2)
        #expect(synced.objects(Edge.self).count == 1)

        let syncedEdge = synced.objects(Edge.self).first!
        #expect(syncedEdge.relation == .partOf)
        #expect(syncedEdge.sourceGlobalId == m1.__globalId)
        #expect(syncedEdge.targetGlobalId == m2.__globalId)
    }

    // MARK: - Test 6: Dynamic filter update via updateSyncFilter

    @Test(.timeLimit(.minutes(1)))
    func ipcRelay_dynamicFilterUpdate() async throws {
        let channel = "engram-dynfilt-\(UUID().uuidString.prefix(8))"

        let hubURL = FileManager.default.temporaryDirectory
            .appending(path: "relay-dyn-hub-\(UUID().uuidString).sqlite")
        let syncedURL = FileManager.default.temporaryDirectory
            .appending(path: "relay-dyn-synced-\(UUID().uuidString).sqlite")

        defer {
            try? Lattice.delete(for: .init(fileURL: hubURL))
            try? Lattice.delete(for: .init(fileURL: syncedURL))
        }

        // Start with filter for "ProjectA" only
        var filter = Lattice.SyncFilter()
        filter.include(Memory.self) { $0.project == "ProjectA" && $0.isPrivate == false }
        filter.include(Edge.self)
        filter.include(SyncConfig.self)

        var hubConfig = Lattice.Configuration(fileURL: hubURL)
        hubConfig.ipcTargets = [.init(channel: channel, syncFilter: filter)]
        let hub = try Lattice(Memory.self, Edge.self, SyncConfig.self, configuration: hubConfig)

        try await Task.sleep(for: .milliseconds(100))

        var syncedConfig = Lattice.Configuration(fileURL: syncedURL)
        syncedConfig.ipcTargets = [.init(channel: channel)]
        let synced = try Lattice(Memory.self, Edge.self, SyncConfig.self, configuration: syncedConfig)

        try await Task.sleep(for: .milliseconds(200))

        // Wait for ProjectA insert
        let task1 = await waitForChange(on: syncedConfig, table: "Memory", operation: .insert)

        // Add memories for both projects
        hub.add(Memory(
            content: "ProjectA memory",
            project: "ProjectA",
            embedding: Vector<Float>([Float](repeating: 0.1, count: 384))
        ))
        hub.add(Memory(
            content: "ProjectB memory",
            project: "ProjectB",
            embedding: Vector<Float>([Float](repeating: 0.2, count: 384))
        ))

        // Wait for ProjectA to arrive
        try await task1.value
        try await Task.sleep(for: .milliseconds(500))

        // Only ProjectA should have synced
        #expect(synced.objects(Memory.self).count == 1)
        #expect(synced.objects(Memory.self).first?.content == "ProjectA memory")

        // Update filter to include both projects (simulates toggleProject)
        var newFilter = Lattice.SyncFilter()
        newFilter.include(Memory.self) { mem in
            (mem.project == "ProjectA" || mem.project == "ProjectB") && mem.isPrivate == false
        }
        newFilter.include(Edge.self)
        newFilter.include(SyncConfig.self)

        // Wait for the reconciliation insert
        let task2 = await waitForChange(on: syncedConfig, table: "Memory", operation: .insert)

        hub.updateSyncFilter(newFilter)

        // reconcile_sync_filter should catch up ProjectB
        try await task2.value

        #expect(synced.objects(Memory.self).count == 2)
        let contents = synced.objects(Memory.self).snapshot().map(\.content).sorted()
        #expect(contents == ["ProjectA memory", "ProjectB memory"])
    }
}
