import ArgumentParser
import EngramKit
import EngramModels
import Foundation
import Lattice

/// Manual embedding-space migration — the same sweep the daemon runs at
/// startup, pointable at any database (calibration copies, spokes, repair).
struct MigrateEmbeddingsCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "migrate-embeddings",
        abstract: "Re-embed stored memories with the current bundled model (space v\(EmbeddingSpace.currentVersion)).",
        discussion: """
        The original bundled model was CLS-pooled — a defective vector space.
        This sweep re-embeds every memory with the fixed mean-pooled model
        and rebuilds the vector index. Idempotent; the version marker is
        advanced only after a complete pass. Run against the live hub it
        relays updates through the daemon's sync automatically.
        """
    )

    @Option(name: .long, help: "Database path (default: ~/.claude/memory.sqlite)")
    var db: String?

    @Flag(name: .long, help: "Run even if the version marker is current.")
    var force: Bool = false

    @Flag(name: .long, help: "Group-spoke mode: [Memory, Edge, GroupProjectMap] schema, no version marker (spokes must not gain hub tables — the relay denies them).")
    var spoke: Bool = false

    func run() async throws {
        let path = db ?? NSHomeDirectory() + "/.claude/memory.sqlite"
        guard FileManager.default.fileExists(atPath: path) else {
            throw ValidationError("No database at \(path)")
        }
        let isSpoke = spoke || (path as NSString).lastPathComponent.hasPrefix("group-")
        let lattice: Lattice
        if isSpoke {
            lattice = try Lattice(
                Memory.self, Edge.self, GroupProjectMap.self,
                configuration: .init(fileURL: URL(fileURLWithPath: path),
                                     migration: engramMigrations)
            )
        } else {
            lattice = try Lattice(
                Memory.self, Edge.self, Checkpoint.self,
                HookState.self, SessionState.self, SyncConfig.self,
                configuration: .init(fileURL: URL(fileURLWithPath: path),
                                     migration: engramMigrations)
            )
        }
        let report = try await EmbeddingMigration.runIfNeeded(
            on: lattice, force: force || isSpoke, useMarker: !isSpoke
        ) { print($0) }
        if report.skipped {
            print("Already at space v\(EmbeddingSpace.currentVersion) — nothing to do (use --force to re-run).")
        } else {
            print("Done: \(report.reembedded) re-embedded, \(report.failed) failed.")
        }
        lattice.checkpoint()
        lattice.close()
    }
}
