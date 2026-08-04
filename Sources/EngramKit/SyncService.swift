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

    // MARK: - Groups (increment 3)

    /// IPC channel feeding one group's spoke. One channel per membership —
    /// they are independently filtered, which is precisely what Stage A1's
    /// per-sync_id sync set made safe.
    public static func groupChannel(_ groupId: UUID) -> String {
        "engram-group-\(groupId.uuidString)"
    }

    /// Daemon-owned spoke database for a group.
    public static func groupDbPath(claudeDir: String, groupId: UUID) -> String {
        // Reuse syncedDbPath's directory creation (700).
        let syncDir = (syncedDbPath(claudeDir: claudeDir) as NSString).deletingLastPathComponent
        return syncDir + "/group-\(groupId.uuidString).sqlite"
    }

    /// A revoked spoke is renamed rather than deleted: MCP discovery stops
    /// attaching it immediately, and the file survives for forensics.
    public static func revokedGroupDbPath(claudeDir: String, groupId: UUID) -> String {
        groupDbPath(claudeDir: claudeDir, groupId: groupId) + ".revoked"
    }

    /// HUB→SPOKE filter for one group channel: non-private memories of the
    /// projects this member exposed to THAT group, plus edges whose
    /// endpoints both qualify.
    ///
    /// Deliberately excludes SyncConfig (device preferences are personal,
    /// not group data) and GroupProjectMap (map rows are written straight
    /// into the spoke — see `MemoryTools`/`SyncManager`; the hub does not
    /// even carry that table).
    ///
    /// An empty exposure set yields a filter that matches NOTHING rather
    /// than everything — the difference between "this group gets nothing
    /// yet" and "this group gets your entire database".
    public static func buildGroupSyncFilter(from lattice: Lattice, groupId: UUID) -> Lattice.SyncFilter {
        let gid = groupId.uuidString
        var exposed: [String] = []
        for config in lattice.objects(SyncConfig.self) where config.exposedGroups.contains(gid) {
            exposed.append(config.project)
        }
        let exposedProjects = exposed

        var filter = Lattice.SyncFilter()
        let memoryPredicate: @Sendable (Query<Memory>) -> Query<Bool> = { mem in
            // `.in([])` is false for every row — the fail-closed empty case.
            !mem.isPrivate && mem.project.in(exposedProjects)
        }
        filter.include(Memory.self, where: memoryPredicate)
        filter.include(Edge.self) { edge in
            edge.sourceGlobalId.in(\Memory.globalId, where: memoryPredicate)
                && edge.targetGlobalId.in(\Memory.globalId, where: memoryPredicate)
        }
        return filter
    }

    /// SPOKE→HUB upload filter — the cross-group leak firewall.
    ///
    /// The hub relays indiscriminately across its channels, so without this
    /// a teammate's row arriving in group A's spoke would flow up into the
    /// hub and back down into every OTHER group whose filter matches its
    /// project. Only rows authored by ME may travel spoke→hub, which keeps
    /// teammates' edits to MY memories converging into my personal copy
    /// while their own memories stay in the group they belong to.
    ///
    /// Memory-only by design: an Edge written in the spoke may reference
    /// memories that never transit the hub, so relaying it up would create
    /// a dangling edge. Edges flow hub→spoke only. GroupProjectMap likewise
    /// never leaves the spoke.
    public static func buildSpokeToHubFilter(me: UUID) -> Lattice.SyncFilter {
        var filter = Lattice.SyncFilter()
        filter.include(Memory.self) { mem in mem.authorUserId == me }
        // Edge/GroupProjectMap intentionally absent → never uploaded.
        return filter
    }

    /// Open one group's spoke: WSS to that group's relay channel, IPC back
    /// to the hub with the authored-by-me firewall.
    ///
    /// - Parameter selfUserId: nil (signed out / directory not yet written)
    ///   makes the spoke→hub filter match NOTHING — safer than uploading
    ///   unattributed rows that could never converge back.
    public static func openGroupLattice(
        claudeDir: String,
        groupId: UUID,
        authToken: String,
        wssBase: URL,
        selfUserId: UUID?
    ) -> Lattice? {
        let dbPath = groupDbPath(claudeDir: claudeDir, groupId: groupId)
        let endpoint = wssBase
            .appendingPathComponent("sync")
            .appendingPathComponent("group")
            .appendingPathComponent(groupId.uuidString)

        var config = Lattice.Configuration(
            fileURL: URL(fileURLWithPath: dbPath),
            authorizationToken: authToken,
            wssEndpoint: endpoint,
            migration: engramMigrations
        )
        // Filter at CONFIGURATION time, not after open: a spoke that
        // connects before its filter lands would run unfiltered, which on a
        // group channel is a cross-group leak window (Stage A2).
        config.ipcTargets = [.init(
            channel: groupChannel(groupId),
            syncFilter: buildSpokeToHubFilter(
                me: selfUserId ?? UUID(uuidString: "00000000-0000-0000-0000-000000000000")!),
            // Stage A4: un-exposing a project must NOT delete rows the
            // group already has — narrowing here is bookkeeping-only and
            // the spoke stays a full mirror (reads are membership-scoped).
            narrowingEmitsRemovals: false)]
        return try? Lattice(
            Memory.self, Edge.self, GroupProjectMap.self,
            configuration: config
        )
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
