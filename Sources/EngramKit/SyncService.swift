import EngramModels
import Foundation
import Lattice

/// Data-layer sync logic extracted from the Visualizer's SyncManager.
/// Used by both the Visualizer and the standalone sync daemon.
public enum SyncService {

    /// Build a SyncFilter from SyncConfig rows in the given database.
    /// Includes non-private memories for projects with `.sync` policy,
    /// edges where both endpoints match, and all SyncConfig rows.
    public static func buildSyncFilter(from lattice: Lattice) -> Lattice.SyncFilter {
        var projects: [String] = []
        for config in lattice.objects(SyncConfig.self).where({ $0.policy == .sync }) {
            projects.append(config.project)
        }
        let syncedProjects = projects

        var filter = Lattice.SyncFilter()

        if !syncedProjects.isEmpty {
            let memoryPredicate: @Sendable (Query<Memory>) -> Query<Bool> = { mem in
                return !mem.isPrivate && mem.project.in(syncedProjects)
            }
            filter.include(Memory.self, where: memoryPredicate)

            // Edges: only sync edges where both endpoints are synced memories
            filter.include(Edge.self) { edge in
                edge.sourceGlobalId.in(\Memory.globalId, where: memoryPredicate)
                    && edge.targetGlobalId.in(\Memory.globalId, where: memoryPredicate)
            }
        }
        // SyncConfig: replicate sync preferences across devices
        filter.include(SyncConfig.self)

        return filter
    }

    /// Compact history before sync to reduce AuditLog entries.
    /// Uses slot-aware compaction — only deletes entries all synchronizers have confirmed.
    public static func compactBeforeSync(_ lattice: Lattice) {
        lattice.compactHistory()
        lattice.checkpoint()
    }

    /// Path to the synced database, in a daemon-owned subdirectory.
    /// The `sync/` directory is created with `700` permissions so only the owner
    /// can list/delete its contents, protecting against accidental deletion.
    public static func syncedDbPath(claudeDir: String) -> String {
        let syncDir = claudeDir + "/sync"
        let fm = FileManager.default
        if !fm.fileExists(atPath: syncDir) {
            try? fm.createDirectory(atPath: syncDir, withIntermediateDirectories: true)
            // 700 = owner rwx only
            try? fm.setAttributes([.posixPermissions: 0o700], ofItemAtPath: syncDir)
        }
        return syncDir + "/memory-synced.sqlite"
    }

    /// Open the synced lattice (relay endpoint) with WSS + IPC.
    ///
    /// - Parameters:
    ///   - claudeDir: Path to ~/.claude.
    ///   - authToken: WSS authorization token.
    ///   - wssEndpoint: WebSocket sync server URL.
    ///   - channel: IPC channel name (default: "engram-sync").
    /// - Returns: The opened synced Lattice, or nil on failure.
    public static func openSyncedLattice(
        claudeDir: String,
        authToken: String,
        wssEndpoint: URL,
        channel: String = "engram-sync"
    ) -> Lattice? {
        let dbPath = syncedDbPath(claudeDir: claudeDir)
        var syncedConfig = Lattice.Configuration(
            fileURL: URL(fileURLWithPath: dbPath),
            authorizationToken: authToken,
            wssEndpoint: wssEndpoint,
            migration: engramMigrations
        )
        syncedConfig.ipcTargets = [.init(channel: channel)]
        return try? Lattice(
            Memory.self, Edge.self, SyncConfig.self,
            configuration: syncedConfig
        )
    }
}
