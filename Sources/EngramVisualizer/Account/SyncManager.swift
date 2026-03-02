import EngramKit
import Foundation
import Lattice
import Observation

/// Manages cloud sync via Lattice IPC relay.
///
/// Architecture:
/// - `localLattice`: the app's primary Lattice (memory.db) with IPC target + sync filter
/// - `syncedLattice`: Lattice on memory_synced.db with IPC target + WSS credentials
///
/// Write path: MCP → memory.db → localLattice (IPC, filtered) → syncedLattice → WSS → cloud
/// Read path: cloud → WSS → syncedLattice → IPC → localLattice → MCP sees new data
@Observable
@MainActor
final class SyncManager {
    /// The app's primary Lattice (memory.db), initialized with IPC target.
    var localLattice: Lattice?
    /// Path to the primary database file.
    var dbPath: String?

    /// memory_synced.db with IPC target + WSS — cloud endpoint.
    /// Exposed (internal) for GalaxyRegistry to create a synced galaxy.
    private(set) var syncedLattice: Lattice?

    var teamLattices: [String: Lattice] = [:]  // teamId → Lattice (Phase 2)
    var statusMessage: String?

    /// Whether the IPC relay sync is active.
    var isSyncing: Bool { syncedLattice != nil }

    /// Current sync progress (updated via callback from synchronizer thread).
    var syncProgressPending: Int = 0
    var syncProgressTotal: Int = 0
    var syncProgressAcked: Int = 0
    var syncProgressReceived: Int = 0
    var syncUploadFraction: Double { syncProgressTotal > 0 ? Double(syncProgressAcked) / Double(syncProgressTotal) : 1.0 }
    var isSyncUploading: Bool { syncProgressPending > 0 }

    // MARK: - Pre-Sync Compaction

    /// Compact history before first sync to reduce AuditLog entries.
    /// e.g. 76k entries → ~1.7k after compaction.
    func compactBeforeSync() {
        localLattice?.compactHistory()
        localLattice?.vacuum()
        localLattice?.checkpoint()
    }

    // MARK: - Sync Lifecycle

    /// Set up IPC relay sync with cloud credentials.
    /// Called when the user signs in and has an active subscription.
    ///
    /// localLattice is already initialized with an IPC target (channel only).
    /// This method builds the sync filter, pushes it to localLattice, and
    /// creates syncedLattice on the other end of the IPC channel.
    func connectSync(wssEndpoint: URL, authToken: String) {
        guard let dbPath, let localLattice else { return }

        let syncedDbPath = (dbPath as NSString).deletingPathExtension + "-synced.sqlite"
        #if TEST_SYNC_CHANNEL
        let channel = ProcessInfo.processInfo.environment["TEST_SYNC_CHANNEL"] ?? "engram-sync"
        #else
        let channel = "engram-sync"
        #endif

        // Build and push sync filter to localLattice (which owns the IPC target)
        let filter = buildSyncFilter(from: localLattice)
        localLattice.updateSyncFilter(filter)

        // Synced: memory_synced.db with IPC (no filter) + WSS
        var syncedConfig = Lattice.Configuration(
            fileURL: URL(fileURLWithPath: syncedDbPath),
            authorizationToken: authToken,
            wssEndpoint: wssEndpoint,
            migration: engramMigrations
        )
        syncedConfig.ipcTargets = [.init(channel: channel)]
        syncedLattice = try? Lattice(
            Memory.self, Edge.self, SyncConfig.self,
            configuration: syncedConfig
        )

        statusMessage = "Connected to sync server"
        wireSyncProgress()
    }

    /// Tear down IPC relay and WSS connection.
    func disconnectSync() {
        syncedLattice = nil
        statusMessage = nil
        syncProgressPending = 0
        syncProgressTotal = 0
        syncProgressAcked = 0
        syncProgressReceived = 0
    }

    // MARK: - Sync Progress

    private func wireSyncProgress() {
        syncedLattice?.onSyncProgress { [weak self] progress in
            Task { @MainActor [weak self] in
                guard let self else { return }
                self.syncProgressPending = progress.pendingUpload
                self.syncProgressTotal = progress.totalUpload
                self.syncProgressAcked = progress.acked
                self.syncProgressReceived = progress.received
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
        let filter = buildSyncFilter(from: localLattice)
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

    // MARK: - Sync Filter Construction

    /// Build a SyncFilter from SyncConfig rows in the local database.
    /// Includes non-private memories for projects with `.sync` policy,
    /// all edges, and all SyncConfig rows.
    private func buildSyncFilter(from lattice: Lattice) -> Lattice.SyncFilter {
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
                edge.sourceGlobalId.in(\Memory.__globalId, where: memoryPredicate)
                    && edge.targetGlobalId.in(\Memory.__globalId, where: memoryPredicate)
            }
        }
        // SyncConfig: replicate sync preferences across devices
        filter.include(SyncConfig.self)

        return filter
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
