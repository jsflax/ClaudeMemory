import EngramKit
import Foundation
import Lattice
import Observation

/// Dedicated manager for sync configuration and project-level migration.
/// Owns all Lattice references needed for sync — AccountService stays auth-only.
@Observable
@MainActor
final class SyncManager {
    var localLattice: Lattice?
    var syncedLattice: Lattice?
    var teamLattices: [String: Lattice] = [:]  // teamId → Lattice (Phase 2)
    var dbPath: String?
    var statusMessage: String?

    /// Whether the synced lattice has an active WebSocket connection.
    var isSyncing: Bool { syncedLattice != nil }

    /// Create the synced lattice with WebSocket credentials and start syncing.
    /// Called when the user signs in and has an active subscription.
    func connectSync(wssEndpoint: URL, authToken: String) {
        guard let dbPath else { return }
        let syncedDbPath = (dbPath as NSString).deletingPathExtension + "-synced.sqlite"
        syncedLattice = try! Lattice(
            Memory.self, Edge.self, SyncConfig.self,
            configuration: .init(
                fileURL: URL(fileURLWithPath: syncedDbPath),
                authorizationToken: authToken,
                wssEndpoint: wssEndpoint,
                migration: engramMigrations
            )
        )
        statusMessage = "Connected to sync server"
    }

    /// Tear down the synced lattice. Stops the WebSocket connection.
    func disconnectSync() {
        syncedLattice = nil
        statusMessage = nil
    }

    /// All distinct project names across both local and synced databases.
    /// Uses SQL-level `.group(by:)` — no `.snapshot()`.
    func allProjects() -> [String] {
        guard let localLattice else { return [] }
        var projects = Set<String>()
        for mem in localLattice.objects(Memory.self).group(by: \.project) {
            projects.insert(mem.project)
        }
        if let syncedLattice {
            for mem in syncedLattice.objects(Memory.self).group(by: \.project) {
                projects.insert(mem.project)
            }
        }
        return projects.sorted()
    }

    /// Memory count for a single project across both databases.
    /// Uses SQL-level `.where { }.count` — no `.snapshot()`.
    func memoryCount(for project: String) -> Int {
        guard let localLattice else { return 0 }
        var total = localLattice.objects(Memory.self).where { $0.project == project }.count
        if let syncedLattice {
            total += syncedLattice.objects(Memory.self).where { $0.project == project }.count
        }
        return total
    }

    /// Current sync policy for a project (defaults to `.local`).
    func syncPolicy(for project: String) -> SyncConfig.Policy {
        guard let localLattice else { return .local }
        if let config = localLattice.objects(SyncConfig.self)
            .where({ $0.project == project }).first {
            return config.policy
        }
        return .local
    }

    /// Toggle a single project's sync policy and migrate immediately.
    func toggleProject(_ project: String) {
        guard let localLattice, let syncedLattice else { return }

        let current = syncPolicy(for: project)
        let newPolicy: SyncConfig.Policy = current == .sync ? .local : .sync

        // Update SyncConfig row
        if let existing = localLattice.objects(SyncConfig.self)
            .where({ $0.project == project }).first {
            existing.policy = newPolicy
            existing.updatedAt = Date()
        } else {
            localLattice.add(SyncConfig(project: project, policy: newPolicy))
        }

        // Migrate memories
        let result: SyncMigration.Result
        if newPolicy == .sync {
            result = SyncMigration.migrateProjects([project], from: localLattice, to: syncedLattice)
        } else {
            result = SyncMigration.migrateProjects([project], from: syncedLattice, to: localLattice)
        }

        if result.memoriesMigrated > 0 {
            statusMessage = "Moved \(result.memoriesMigrated) memories"

            if newPolicy == .sync {
                Self.spawnReconciliationAgent(project: project)
            }
        }
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
