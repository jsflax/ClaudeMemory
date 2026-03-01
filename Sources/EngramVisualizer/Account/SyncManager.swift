import EngramKit
import Foundation
import Lattice
import Observation

/// Manages cloud sync via Lattice IPC relay.
///
/// Architecture:
/// - `hubLattice`: second Lattice instance on memory.db with IPC target + sync filter
/// - `syncedLattice`: Lattice on memory_synced.db with IPC target + WSS credentials
///
/// Write path: MCP → memory.db → xproc notify → hubLattice (IPC, filtered) → syncedLattice → WSS → cloud
/// Read path: cloud → WSS → syncedLattice → IPC → hubLattice → xproc notify → MCP sees new data
@Observable
@MainActor
final class SyncManager {
    /// The app's primary Lattice (memory.db). Set during configuration.
    var localLattice: Lattice?
    /// Path to the primary database file.
    var dbPath: String?

    /// Second instance of memory.db with IPC target — drives filtered sync to syncedLattice.
    private var hubLattice: Lattice?
    /// memory_synced.db with IPC target + WSS — cloud endpoint.
    private var syncedLattice: Lattice?

    var teamLattices: [String: Lattice] = [:]  // teamId → Lattice (Phase 2)
    var statusMessage: String?

    /// Whether the IPC relay sync is active.
    var isSyncing: Bool { hubLattice != nil && syncedLattice != nil }

    // MARK: - Sync Lifecycle

    /// Set up IPC relay sync with cloud credentials.
    /// Called when the user signs in and has an active subscription.
    func connectSync(wssEndpoint: URL, authToken: String) {
        guard let dbPath, let localLattice else { return }

        let syncedDbPath = (dbPath as NSString).deletingPathExtension + "-synced.sqlite"
        let channel = "engram-sync"
        let filter = buildSyncFilter(from: localLattice)

        // Hub: second connection to memory.db with IPC + sync filter
        var hubConfig = Lattice.Configuration(
            fileURL: URL(fileURLWithPath: dbPath),
            migration: engramMigrations
        )
        hubConfig.ipcTargets = [.init(channel: channel, syncFilter: filter)]
        hubLattice = try? Lattice(
            Memory.self, Edge.self, SyncConfig.self,
            configuration: hubConfig
        )

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
    }

    /// Tear down IPC relay and WSS connection.
    func disconnectSync() {
        hubLattice = nil
        syncedLattice = nil
        statusMessage = nil
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

        // Rebuild filter and push to hub — Lattice's reconcile_sync_filter handles catch-up
        let filter = buildSyncFilter(from: localLattice)
        hubLattice?.updateSyncFilter(filter)

        statusMessage = newPolicy == .sync
            ? "Syncing \(project)"
            : "Stopped syncing \(project)"

        if newPolicy == .sync {
            Self.spawnReconciliationAgent(project: project)
        }
    }

    // MARK: - Sync Filter Construction

    /// Build a SyncFilter from SyncConfig rows in the local database.
    /// Includes non-private memories for projects with `.sync` policy,
    /// all edges, and all SyncConfig rows.
    private func buildSyncFilter(from lattice: Lattice) -> Lattice.SyncFilter {
        let syncedProjects = lattice.objects(SyncConfig.self)
            .where { $0.policy == .sync }
            .snapshot()
            .map(\.project)

        var filter = Lattice.SyncFilter()

        if !syncedProjects.isEmpty {
            filter.include(Memory.self) { mem in
                var match = mem.project == syncedProjects[0]
                for p in syncedProjects.dropFirst() { match = match || mem.project == p }
                return mem.isPrivate == false && match
            }
        }

        // Edges: sync all (dangling refs are harmless, cleaned on next reconciliation)
        filter.include(Edge.self)
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
