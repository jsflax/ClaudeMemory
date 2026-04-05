import Testing
import EngramKit
import EngramModels
import Lattice
import MCP
import Foundation

/// Reproduce the MCP recall crash: repeated attaching + writes on production DBs.
@Test(.timeLimit(.minutes(1)))
func attachRace_productionDB_repeatedRecall() async throws {
    let localPath = URL(fileURLWithPath: NSHomeDirectory())
        .appending(path: ".claude/memory.sqlite")
    let syncedPath = URL(fileURLWithPath: NSHomeDirectory())
        .appending(path: ".claude/sync/memory-synced.sqlite")
    guard FileManager.default.fileExists(atPath: localPath.path(percentEncoded: false)),
          FileManager.default.fileExists(atPath: syncedPath.path(percentEncoded: false)) else { return }

    let localLattice = try Lattice(Memory.self, Edge.self, Checkpoint.self, HookState.self, SyncConfig.self,
                                   configuration: .init(fileURL: localPath, migration: engramMigrations))
    let syncedLattice = try Lattice(Memory.self, Edge.self, SyncConfig.self,
                                    configuration: .init(fileURL: syncedPath, migration: engramMigrations))

    let embedder = sharedEmbedder
    if await !embedder.isLoaded { await embedder.load() }

    let tools = MemoryTools(
        localRef: localLattice.sendableReference,
        syncedRef: syncedLattice.sendableReference,
        embedder: embedder
    )

    // Rapid-fire recall — each creates a temporary attached Lattice
    for i in 0..<10 {
        let result = try await tools.handle(CallTool.Parameters(
            name: "recall",
            arguments: [
                "query": .string("Lattice architecture and patterns"),
                "project": .string("Lattice"),
                "limit": .int(3),
            ]
        ))
        let output = text(from: result)
        print("Recall \(i): \(output.prefix(80))...")
    }
}
