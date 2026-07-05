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
    
    /// Path to the primary database file.
    var dbPath: String?

    actor Actor {
        @MainActor weak var parent: SyncManager?
        init(parent: SyncManager) {
            self.parent = parent
        }
        
        /// The app's primary Lattice (memory.db).
        var localLattice: Lattice?
        /// memory-synced.db opened as a plain Lattice (no WSS, no IPC) for observation.
        /// Exposed (internal) for GalaxyRegistry to create a synced galaxy.
        private(set) var syncedLattice: Lattice?
        /// Path to the primary database file.
        var dbPath: String?
        
        @MainActor var localLatticeRef: LatticeThreadSafeReference?
        
        func setLocalLattice(_ ref: LatticeThreadSafeReference?) {
            localLattice = ref?.resolve()
            dbPath = localLattice?.configuration.fileURL.path()
            Task { @MainActor in localLatticeRef = ref }
        }
        
        @MainActor var syncedLatticeRef: LatticeThreadSafeReference?
        
        // MARK: - Sync Lifecycle

        /// Set up sync observation. The daemon owns the actual WSS + IPC connections.
        /// The Visualizer just opens the synced DB for progress observation and
        /// starts the daemon if it isn't already running.
        func connectSync(wssEndpoint: URL, authToken: String) {
            guard let dbPath, let localLattice else { return }
            print("connecting to sync")
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

            wireSyncProgress()
            CLIInstaller.startDaemon()
            
            Task { @MainActor [ref = syncedLattice?.sendableReference] in
                syncedLatticeRef = ref
                parent?.statusMessage = "Connected to sync server"
                parent?.didConnect.send()
            }
        }

        /// Tear down sync: stop daemon, delete synced DB so next sign-in starts fresh.
        func disconnectSync() {
            CLIInstaller.stopDaemon()
            syncedLattice = nil

            // Delete the synced DB (closes connections, removes DB + WAL + SHM),
            // but ONLY after the daemon has actually released it. stopDaemon()
            // signals the daemon; it isn't synchronous. Deleting while the
            // daemon still holds the DB open serves the deleted inode and
            // leaves -wal/-shm orphans. Probe the daemon's process flock: if
            // we can take it, the daemon is gone; if not within the timeout,
            // SKIP deletion rather than corrupt an in-use DB.
            if let dbPath {
                let claudeDir = (dbPath as NSString).deletingLastPathComponent
                let syncedDbPath = SyncService.syncedDbPath(claudeDir: claudeDir)
                if Self.waitForDaemonExit(claudeDir: claudeDir, timeout: 5.0) {
                    let config = Lattice.Configuration(fileURL: URL(fileURLWithPath: syncedDbPath))
                    try? Lattice.delete(for: config)
                } else {
                    NSLog("[SyncManager] daemon still holds sync lock after 5s — skipping synced-DB deletion")
                }
            }

            // Clear sync state on local DB so next sign-in does a full re-sync.
            // Without this, the local DB thinks rows are already synced to a DB that no longer exists.
            localLattice?.updateSyncFilter(nil)
            
            Task { @MainActor in
                parent?.statusMessage = nil
                parent?.ipcProgress = nil
                parent?.wssProgress = nil
            }
        }
        
        
        
        /// Poll the daemon's process flock until it's free (daemon exited) or
        /// the timeout elapses. Non-destructive: acquiring the lock proves the
        /// daemon released it; we immediately release again. Returns false if
        /// the daemon is still holding it — the caller must not delete the DB.
        static func waitForDaemonExit(claudeDir: String, timeout: TimeInterval) -> Bool {
            let lockPath = claudeDir + "/engram-sync.lock"
            let deadline = Date().addingTimeInterval(timeout)
            repeat {
                let fd = open(lockPath, O_RDWR)
                if fd < 0 { return true }  // no lock file → no daemon
                let got = flock(fd, LOCK_EX | LOCK_NB) == 0
                if got { flock(fd, LOCK_UN) }
                close(fd)
                if got { return true }
                Thread.sleep(forTimeInterval: 0.1)
            } while Date() < deadline
            return false
        }

        // MARK: - Sync Progress

        private func wireSyncProgress() {
            // IPC progress: cross-process observation of memory.db's IPC relay
            localLattice?.onSyncProgress { [weak self] progress in
                Task { @MainActor [weak self] in
                    self?.parent?.ipcProgress = progress
                }
            }
            // WSS progress: cross-process observation of synced.db's WSS upload
            syncedLattice?.onSyncProgress { [weak self] progress in
                Task { @MainActor [weak self] in
                    self?.parent?.wssProgress = progress
                }
            }
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
            
            Task { @MainActor in
                parent?.statusMessage = newPolicy == .sync
                ? "Syncing \(project)"
                : "Stopped syncing \(project)"
                
                if newPolicy == .sync {
                    SyncManager.spawnReconciliationAgent(project: project)
                }
            }
        }
    }
    
    var actor: Actor!
    init() {
        actor = Actor(parent: self)
    }
    
    func initialize(lattice: Lattice) {
        self.dbPath = lattice.configuration.fileURL.path()
        Task { [ref = lattice.sendableReference] in
            await actor.setLocalLattice(ref)
        }
    }
    
    func connectSync(wssEndpoint: URL, authToken: String) {
        Task {
            await actor.connectSync(wssEndpoint: wssEndpoint, authToken: authToken)
        }
    }
    func disconnectSync() {
        Task {
            await actor.disconnectSync()
        }
    }
    var teamLattices: [String: Lattice] = [:]  // teamId → Lattice (Phase 2)
    var statusMessage: String?

    /// Whether sync is configured (daemon may or may not be running).
    var isSyncing: Bool { actor.syncedLatticeRef != nil }

    /// IPC sync progress (memory.db → synced.db via daemon's IPC relay).
    var ipcProgress: Lattice.SyncProgress?

    /// WSS sync progress (synced.db → cloud via daemon's WebSocket).
    /// Cross-process: derived from AuditLog observation.
    var wssProgress: Lattice.SyncProgress?

    /// Daemon health read from ~/.claude/sync-daemon-status.json (written by
    /// memory-sync on every state transition). Progress rows alone can't show
    /// connection failures: "nothing pending" looks identical to "the WSS has
    /// been rejected with 401 for three months" — which happened.
    struct DaemonHealth {
        let state: String        // connected/disconnected/error/waiting_for_auth/...
        let detail: String?
        let lastSyncAt: String?
        let updatedAt: Date?
        var isHealthy: Bool { state == "connected" || state == "starting" }
        var isStale: Bool {
            guard let updatedAt else { return true }
            return Date().timeIntervalSince(updatedAt) > 180
        }
    }
    var daemonHealth: DaemonHealth?

    func refreshDaemonHealth() {
        let path = NSHomeDirectory() + "/.claude/sync-daemon-status.json"
        guard let data = FileManager.default.contents(atPath: path),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let state = obj["state"] as? String else {
            daemonHealth = nil  // old daemon binary or daemon not running
            return
        }
        let iso = ISO8601DateFormatter()
        daemonHealth = DaemonHealth(
            state: state,
            detail: obj["detail"] as? String,
            lastSyncAt: obj["lastSyncAt"] as? String,
            updatedAt: (obj["updatedAt"] as? String).flatMap { iso.date(from: $0) }
        )
    }

    // MARK: - Sync Policy

    /// Current sync policy for a project (defaults to `.local`).
    func syncPolicy(for project: String) -> SyncConfig.Policy {
        guard let localLattice = actor.localLatticeRef?.resolve() else { return .local }
        if let config = localLattice.objects(SyncConfig.self)
            .where({ $0.project == project }).first {
            return config.policy
        }
        return .local
    }

    /// Toggle a project's sync policy and update the IPC relay filter.
    func toggleProject(_ project: String) {
        Task {
            await actor.toggleProject(project)
        }
    }

    /// Project names currently configured for sync. Used by GalaxyRegistry to build
    /// the local galaxy's node filter (complement: exclude synced non-private memories).
    var syncedProjectNames: Set<String> {
        guard let localLattice = actor.localLatticeRef?.resolve() else {
            return []
        }
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
