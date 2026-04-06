import Testing
import EngramKit
import EngramModels
import Lattice
import MCP
import Foundation

/// Test that distances are correct from attached vs non-attached recalls.
@Test func distance_attachedVsLocal() async throws {
    let localPath = URL(fileURLWithPath: NSHomeDirectory())
        .appending(path: ".claude/memory.sqlite")
    let syncedPath = URL(fileURLWithPath: NSHomeDirectory())
        .appending(path: ".claude/sync/memory-synced.sqlite")
    guard FileManager.default.fileExists(atPath: localPath.path(percentEncoded: false)),
          FileManager.default.fileExists(atPath: syncedPath.path(percentEncoded: false)) else { return }

    let localLattice = try Lattice(Memory.self, Edge.self, Checkpoint.self, HookState.self, SyncConfig.self,
                                   configuration: .init(fileURL: localPath, migration: engramMigrations))

    let dims = 384
    let query = Vector<Float>((0..<dims).map { _ in Float.random(in: -1...1) })

    // Local-only nearest (no attaching)
    let localNearest = localLattice.objects(Memory.self)
        .nearest(to: query, on: \.embedding, limit: 5)
    let localSnapshot = localNearest.snapshot()

    print("=== Local-only distances ===")
    for (i, m) in localSnapshot.enumerated() {
        print("  \(i): distance=\(m.distance), content=\(m.object.content.prefix(40))")
    }

    // Attached nearest (local + synced UNION ALL)
    let syncedLattice = try Lattice(Memory.self, Edge.self, SyncConfig.self,
                                    configuration: .init(fileURL: syncedPath, migration: engramMigrations))
    let combined = localLattice.attaching(lattice: syncedLattice)
    let attachedNearest = combined.objects(Memory.self)
        .nearest(to: query, on: \.embedding, limit: 5)
    let attachedSnapshot = attachedNearest.snapshot()

    print("=== Attached distances ===")
    for (i, m) in attachedSnapshot.enumerated() {
        print("  \(i): distance=\(m.distance), content=\(m.object.content.prefix(40))")
    }

    // Verify local distances are non-zero
    for m in localSnapshot {
        #expect(m.distance > 0, "Local distance should be > 0, got \(m.distance)")
    }

    // Verify attached distances are non-zero
    for m in attachedSnapshot {
        #expect(m.distance > 0, "Attached distance should be > 0, got \(m.distance)")
    }
}
