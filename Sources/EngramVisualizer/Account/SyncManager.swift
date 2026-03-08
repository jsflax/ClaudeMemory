import Combine
import EngramKit
import EngramModels
import Foundation
import Lattice
import Observation

/// Manages cloud sync observation in the Visualizer.
///
/// The sync daemon (`memory-sync`) owns the WSS connection and IPC relay.
/// The Visualizer opens both databases as plain read-only observers:
/// - `localLattice`: memory.db — the app's primary Lattice
/// - `syncedLattice`: memory-synced.db — opened plain for cross-process progress observation
///
/// Sync progress is observed cross-process via Lattice's AuditLog-based fallback.
@Observable
@MainActor
final class SyncManager {
    /// Fires once when sync connects (syncedLattice becomes non-nil).
    let didConnect = PassthroughSubject<Void, Never>()
    /// The app's primary Lattice (memory.db).
    var localLattice: Lattice?
    /// Path to the primary database file.
    var dbPath: String?

    /// memory-synced.db opened as a plain Lattice (no WSS, no IPC) for observation.
    /// Exposed (internal) for GalaxyRegistry to create a synced galaxy.
    private(set) var syncedLattice: Lattice?

    var teamLattices: [String: Lattice] = [:]  // teamId → Lattice (Phase 2)
    var statusMessage: String?

    /// Whether sync is configured (daemon may or may not be running).
    var isSyncing: Bool { syncedLattice != nil }

    /// IPC sync progress (memory.db → synced.db via daemon's IPC relay).
    var ipcProgress: Lattice.SyncProgress?

    /// WSS sync progress (synced.db → cloud via daemon's WebSocket).
    /// Cross-process: derived from AuditLog observation.
    var wssProgress: Lattice.SyncProgress?

    // MARK: - Sync Lifecycle

    /// Set up sync observation. The daemon owns the actual WSS + IPC connections.
    /// The Visualizer just opens the synced DB for progress observation and
    /// starts the daemon if it isn't already running.
    func connectSync(wssEndpoint: URL, authToken: String) {
        guard let dbPath, let localLattice else { return }

        // Open synced DB as a plain Lattice (no WSS, no IPC) for observation.
        // The synced DB lives in the daemon-owned sync/ directory.
        let claudeDir = (dbPath as NSString).deletingLastPathComponent
        let syncedDbPath = SyncService.syncedDbPath(claudeDir: claudeDir)
        syncedLattice = try? Lattice(
            Memory.self, Edge.self, SyncConfig.self,
            configuration: .init(
                fileURL: URL(fileURLWithPath: syncedDbPath),
                migration: engramMigrations
            )
        )

        statusMessage = "Connected to sync server"
        wireSyncProgress()
        CLIInstaller.startDaemon()
        didConnect.send()
    }

    /// Tear down sync: stop daemon, delete synced DB so next sign-in starts fresh.
    func disconnectSync() {
        CLIInstaller.stopDaemon()
        syncedLattice = nil

        // Delete the synced DB (closes connections, removes DB + WAL + SHM)
        if let dbPath {
            let claudeDir = (dbPath as NSString).deletingLastPathComponent
            let syncedDbPath = SyncService.syncedDbPath(claudeDir: claudeDir)
            let config = Lattice.Configuration(fileURL: URL(fileURLWithPath: syncedDbPath))
            try? Lattice.delete(for: config)
        }

        // Clear sync state on local DB so next sign-in does a full re-sync.
        // Without this, the local DB thinks rows are already synced to a DB that no longer exists.
        localLattice?.updateSyncFilter(nil)

        statusMessage = nil
        ipcProgress = nil
        wssProgress = nil
    }

    // MARK: - Sync Progress

    private func wireSyncProgress() {
        // IPC progress: cross-process observation of memory.db's IPC relay
        localLattice?.onSyncProgress { [weak self] progress in
            Task { @MainActor [weak self] in
                self?.ipcProgress = progress
            }
        }
        // WSS progress: cross-process observation of synced.db's WSS upload
        syncedLattice?.onSyncProgress { [weak self] progress in
            Task { @MainActor [weak self] in
                self?.wssProgress = progress
            }
        }
    }

    // MARK: - Sync Policy

    /// Current sync policy for a project (defaults to `.local`).
    func syncPolicy(for project: String) -> SyncConfig.Policy {
        guard let localLattice else { return .local }
        if let config = localLattice.objects(SyncConfig.self)
            .where({ $0.project == project }).first {
            return config.policy
        }
        return .local
    }

    /// Toggle a project's sync policy and update the IPC relay filter.
    func toggleProject(_ project: String) {
        guard let localLattice else { return }

        let current = syncPolicy(for: project)
        let newPolicy: SyncConfig.Policy = current == .sync ? .local : .sync

        // Update SyncConfig row in memory.db
        if let existing = localLattice.objects(SyncConfig.self)
            .where({ $0.project == project }).first {
            existing.policy = newPolicy
            existing.updatedAt = Date()
        } else {
            localLattice.add(SyncConfig(project: project, policy: newPolicy))
        }

        // Rebuild filter and push to localLattice — Lattice's reconcile_sync_filter handles catch-up
        let filter = SyncService.buildSyncFilter(from: localLattice)
        localLattice.updateSyncFilter(filter)

        statusMessage = newPolicy == .sync
            ? "Syncing \(project)"
            : "Stopped syncing \(project)"

        if newPolicy == .sync {
            Self.spawnReconciliationAgent(project: project)
        }
    }

    /// Project names currently configured for sync. Used by GalaxyRegistry to build
    /// the local galaxy's node filter (complement: exclude synced non-private memories).
    var syncedProjectNames: Set<String> {
        guard let localLattice else { return [] }
        var result = Set<String>()
        for config in localLattice.objects(SyncConfig.self).where({ $0.policy == .sync }) {
            result.insert(config.project)
        }
        return result
    }

    // MARK: - Reconciliation Subprocess

    private static let reconciliationLogPath = NSHomeDirectory() + "/.claude/sync-reconciliation.log"

    private static let reconciliationSystemPrompt: String = loadAgentSystemPrompt(
        name: "sync-reconciliation",
        fallback: """
        You are a sync reconciliation agent. Reconcile duplicate and conflicting memories after cross-device sync.
        1. Run find_clusters(project, distance_threshold: 12, min_cluster_size: 2). If none, exit.
        2. For each cluster: recall full content, then consolidate true duplicates, connect contradictions, link related.
        3. Run a global pass with find_clusters(project: "global").
        4. Report what changed.
        Safety: always consolidate (never forget/merge), always recall before consolidating, never auto-resolve contradictions.
        """
    )

    private static func spawnReconciliationAgent(project: String) {
        let prompt = """
        Reconcile memories for project "\(project)" after sync migration.

        Follow your system prompt workflow: assess with stats + find_clusters, reconcile each cluster, \
        global pass, then report. If no clusters are found, exit immediately.
        """

        do {
            try spawnClaudeSubprocess(
                prompt: prompt,
                systemPrompt: reconciliationSystemPrompt,
                allowedTools: "mcp__memory__*",
                model: "sonnet",
                envGuard: (key: "CLAUDE_MEMORY_SYNC_RECONCILIATION", value: "1"),
                logPath: reconciliationLogPath
            )
        } catch {
            // Best-effort — don't block the UI for reconciliation failures
        }
    }
}
