import ArgumentParser
import EngramKit
import EngramModels
import Foundation
import Lattice

@main
struct EngramDaemon: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "memory-sync",
        abstract: "Engram sync daemon — relays memory changes to the cloud via WSS.",
        subcommands: [ExposeCommand.self, GroupsCommand.self,
                      WhoamiCommand.self, AcceptInviteCommand.self,
                      CompactServerHistoryCommand.self,
                      MigrateEmbeddingsCommand.self]
    )

    @Option(name: .long, help: "Lattice log level: off, error, warning, info, debug")
    var logLevel: String?

    @Option(name: .long, help: "Sync server endpoint (default: https://engramdb.io)")
    var endpoint: String?

    @Flag(name: .long, help: "Nuclear compaction: wipe all sync state and re-sync from scratch")
    var nuclearCompact: Bool = false

    func run() async throws {
        if let logLevel {
            switch logLevel.lowercased() {
            case "debug": Lattice.setLogLevel(.debug)
            case "info": Lattice.setLogLevel(.info)
            case "warn", "warning": Lattice.setLogLevel(.warn)
            case "error": Lattice.setLogLevel(.error)
            case "off": Lattice.setLogLevel(.off)
            default: log("Unknown log level '\(logLevel)', using default")
            }
        }

        let claudeDir = NSHomeDirectory() + "/.claude"

        // 1. Acquire process-level flock — only one daemon instance at a time
        let lockPath = claudeDir + "/engram-sync.lock"
        let lockFd = open(lockPath, O_CREAT | O_RDWR, 0o600)
        guard lockFd >= 0 else {
            log("Failed to open lock file: \(lockPath)")
            throw ExitCode.failure
        }
        guard flock(lockFd, LOCK_EX | LOCK_NB) == 0 else {
            log("Another sync daemon is already running, exiting.")
            close(lockFd)
            throw ExitCode.success
        }

        // 2. Read auth credentials — with retry. The daemon is commonly
        // launched (by launchd, at login) BEFORE the user signs in via the
        // Visualizer, so an immediate success-exit here meant the daemon
        // stayed dead until the next login even after sign-in. Poll the
        // keychain for up to 10 minutes (20 × 30s) before giving up.
        var credentials: SyncCredentials?
        let credRetryLimit = 20
        for attempt in 0..<credRetryLimit {
            if let c = readCredentials(claudeDir: claudeDir, endpointOverride: endpoint) {
                credentials = c
                break
            }
            if attempt == 0 {
                DaemonStatus.write(state: "waiting_for_auth",
                                   detail: "No credentials yet — sign in via the Visualizer.")
                log("No auth credentials found — polling keychain (up to 10 min)...")
            }
            sleep(30)
        }
        guard let credentials else {
            log("No auth credentials after \(credRetryLimit) attempts. Exiting; relaunch after sign-in.")
            DaemonStatus.write(state: "stopped", detail: "No credentials after 10 min.")
            flock(lockFd, LOCK_UN)
            close(lockFd)
            throw ExitCode.success  // Exit cleanly — launchd won't restart on successful exit
        }
        log("Auth loaded (endpoint: \(credentials.endpoint))")
        DaemonStatus.write(state: "starting", detail: "Credentials loaded.")

        // 3. Open localLattice on memory.sqlite
        let dbPath = claudeDir + "/memory.sqlite"
        let channel = "engram-sync"

        // Compact BEFORE enabling IPC — forceCompactHistory resets replication
        // slot cursors in the DB, but an already-initialized IPC synchronizer
        // caches the old cursor in memory and ignores the reset.
        // Open a plain Lattice (no IPC), compact, then close() to evict from
        // the LatticeCache so the real open (with IPC) gets a fresh instance.
        if nuclearCompact {
            log("Nuclear compaction: wiping sync state...")
            let compactConfig = Lattice.Configuration(
                fileURL: URL(fileURLWithPath: dbPath),
                migration: engramMigrations
            )
            do {
                let compactLattice = try Lattice(
                    Memory.self, Edge.self, Checkpoint.self,
                    HookState.self, SessionState.self, SyncConfig.self,
                    configuration: compactConfig
                )
                compactLattice.forceCompactHistory()
                compactLattice.vacuum()
                compactLattice.checkpoint()
                compactLattice.close()
                log("Nuclear compaction complete")
            } catch {
                log("Failed to open database for compaction: \(error)")
                flock(lockFd, LOCK_UN)
                close(lockFd)
                throw ExitCode.failure
            }
        }

        // Group memberships FIRST: `ipcTargets` is configuration-time only,
        // so the hub must know every channel before it opens. A fetch
        // failure falls back to the last-written groups.json (offline) and
        // finally to personal-only — corrected at the next membership poll.
        var memberships: [GroupDirectorySync.Membership] = []
        var selfUserId: UUID?
        // "I asked the server and it told me" vs "I could not ask". ONLY the
        // former may drive spoke retirement: retireDepartedSpokes reads an
        // empty set as "this user is in no groups" and quarantines every
        // spoke + wipes its hub channel state. A 401 on an expired token
        // would otherwise destroy every group's local mirror (and the
        // GroupProjectMap rows that live nowhere else) and force a full
        // re-download once the token refreshes.
        var membershipsAuthoritative = false
        do {
            let snapshot = try await GroupDirectorySync.fetch(
                endpoint: credentials.endpoint, token: credentials.token)
            try? GroupDirectorySync.write(snapshot, claudeDir: claudeDir)
            memberships = snapshot.groups
            selfUserId = snapshot.selfUserId
            membershipsAuthoritative = true
            log("Groups: \(memberships.count) membership(s) fetched")
        } catch GroupDirectorySync.FetchError.unauthorized {
            // Same treatment as any other failure: a rejected token says
            // nothing about membership. Fall back to the cached directory so
            // an expired token degrades to "stale groups" rather than "no
            // groups" — the poll corrects it once auth recovers.
            let cached = GroupDirectory.load(from: URL(fileURLWithPath:
                (SyncService.syncedDbPath(claudeDir: claudeDir) as NSString)
                    .deletingLastPathComponent + "/groups.json"))
            memberships = (cached?.groups ?? []).map {
                GroupDirectorySync.Membership(id: $0.id, name: $0.name, parentId: $0.parentId,
                                              myRole: $0.myRole, root: $0.root ?? false)
            }
            selfUserId = cached?.selfUserId
            log("Groups: unauthorized — keeping cached directory (\(memberships.count) group(s)); NOT retiring spokes")
        } catch {
            let cached = GroupDirectory.load(from: URL(fileURLWithPath:
                (SyncService.syncedDbPath(claudeDir: claudeDir) as NSString)
                    .deletingLastPathComponent + "/groups.json"))
            memberships = (cached?.groups ?? []).map {
                GroupDirectorySync.Membership(id: $0.id, name: $0.name, parentId: $0.parentId,
                                              myRole: $0.myRole, root: $0.root ?? false)
            }
            selfUserId = cached?.selfUserId
            log("Groups: fetch failed (\(error)) — using cached directory (\(memberships.count) group(s))")
        }

        // Personal entitlement decides whether the PERSONAL spoke may open.
        // Group seats are billed separately, so a user with no personal
        // subscription but a paid group seat is a first-class persona:
        // their group sync must work while /sync (personal) stays closed —
        // opening it would loop on 402 and paint a false error state.
        // Unknown (server unreachable) is treated as ENABLED so a transient
        // outage never silently downgrades a paying user to group-only.
        let personalSyncEnabled = await GroupDirectorySync.fetchPersonalSyncEnabled(
            endpoint: credentials.endpoint, token: credentials.token) ?? true

        if !personalSyncEnabled && memberships.isEmpty {
            // Nothing to sync — but do NOT exit: launchd will not restart a
            // clean exit, and the user may subscribe or accept an invite at
            // any moment. Idle and re-check on the same cadence as the
            // membership poll.
            log("No personal subscription and no group memberships — idling until entitlements change")
            DaemonStatus.write(state: "idle",
                               detail: "No active subscription or group membership.")
            while true {
                sleep(300)
                let groupsNow = (try? await GroupDirectorySync.fetch(
                    endpoint: credentials.endpoint, token: credentials.token))?.groups ?? []
                let personalNow = await GroupDirectorySync.fetchPersonalSyncEnabled(
                    endpoint: credentials.endpoint, token: credentials.token) ?? false
                if personalNow || !groupsNow.isEmpty {
                    log("Entitlements changed (personal: \(personalNow), groups: \(groupsNow.count)) — restarting")
                    flock(lockFd, LOCK_UN)
                    close(lockFd)
                    Darwin.exit(1)   // launchd restarts on non-zero
                }
            }
        }
        if !personalSyncEnabled {
            log("Personal sync disabled (no active personal subscription) — running GROUP-ONLY with \(memberships.count) spoke(s)")
        }

        // Now open with IPC — replication slot cursor will be 0 after nuclear compact
        var localConfig = Lattice.Configuration(
            fileURL: URL(fileURLWithPath: dbPath),
            migration: engramMigrations
        )
        // One IPC target per sync destination: the personal spoke plus one
        // per group. Every filter is set HERE, at configuration time — a
        // spoke that connects before its filter arrives would run
        // unfiltered, which on a group channel is a cross-group leak
        // (Stage A2). Independent filters on one database is exactly what
        // Stage A1's per-sync_id sync set made correct.
        var hubTargets: [Lattice.IPCSyncTarget] = [
            .init(channel: channel, syncFilter: SyncService.buildSyncFilter(from: hubFilterSource(dbPath)))
        ]
        for membership in memberships {
            hubTargets.append(.init(
                channel: SyncService.groupChannel(membership.id),
                syncFilter: SyncService.buildGroupSyncFilter(
                    from: hubFilterSource(dbPath), groupId: membership.id),
                // Stage A4: un-exposing must not delete what the group has.
                narrowingEmitsRemovals: false))
        }
        localConfig.ipcTargets = hubTargets

        let localLattice: Lattice
        do {
            localLattice = try Lattice(
                Memory.self, Edge.self, Checkpoint.self,
                HookState.self, SessionState.self, SyncConfig.self,
                configuration: localConfig
            )
        } catch {
            log("Failed to open local database: \(error)")
            flock(lockFd, LOCK_UN)
            close(lockFd)
            throw ExitCode.failure
        }

        // 4. Normal compaction (non-nuclear) — safe with IPC already active
        if !nuclearCompact {
            log("Compacting history before sync...")
            SyncService.compactBeforeSync(localLattice)
        }

        // 4b. Embedding-space migration (v2 mean-pooling, Aug 2026): re-embed
        //     stored vectors with the fixed model before serving. The hub is
        //     open WITH ipcTargets, so updated vectors relay to spokes/server
        //     like any field update. Cheap no-op when the marker is current;
        //     a failure logs and continues — recall degrades (mixed spaces)
        //     but sync must not be held hostage by a model-load problem.
        do {
            let report = try await EmbeddingMigration.runIfNeeded(on: localLattice, log: log)
            if report.reembedded > 0 {
                DaemonStatus.write(state: "starting",
                                   detail: "Re-embedded \(report.reembedded) memories (space v\(EmbeddingSpace.currentVersion)).")
            }
        } catch {
            log("Embedding migration failed (will retry next start): \(error)")
        }

        // 5. Filters were applied at configuration time (above). Refresh
        //    them per channel now that the hub is open with real data —
        //    per-channel, never the fan-to-all overload, which would
        //    overwrite every group's filter with one of them (Stage A2).
        localLattice.updateSyncFilter(SyncService.buildSyncFilter(from: localLattice),
                                      forChannel: channel)
        for membership in memberships {
            localLattice.updateSyncFilter(
                SyncService.buildGroupSyncFilter(from: localLattice, groupId: membership.id),
                forChannel: SyncService.groupChannel(membership.id))
        }
        log("Sync filters applied (personal + \(memberships.count) group channel(s))")

        // 6. Open syncedLattice with WSS + IPC (in daemon-owned sync/ directory).
        //    Skipped entirely for the group-only persona — see above.
        let syncedDbPath = SyncService.syncedDbPath(claudeDir: claudeDir)
        var syncedLattice: Lattice?
        if personalSyncEnabled {
            guard let opened = SyncService.openSyncedLattice(
                claudeDir: claudeDir,
                authToken: credentials.token,
                wssEndpoint: credentials.wssEndpoint,
                channel: channel
            ) else {
                log("Failed to open synced database")
                flock(lockFd, LOCK_UN)
                close(lockFd)
                throw ExitCode.failure
            }
            syncedLattice = opened
            log("Synced lattice opened (WSS: \(credentials.wssEndpoint), db: \(syncedDbPath))")
        }

        // 6b. Group spokes — one per membership. Held for the daemon's
        //     lifetime; a dropped reference would tear down the sync
        //     threads that make the spoke a live mirror.
        var groupSpokes: [UUID: Lattice] = [:]
        // What the BILLING gate decided at startup, per group. The poll
        // compares against this rather than against "is a spoke running",
        // because a spoke can be absent for reasons that have nothing to do
        // with billing (a corrupt or locked spoke file, a failed migration)
        // — and conflating the two turns one unopenable spoke into a
        // permanent 5-minute restart loop that also tears down personal sync.
        var startupPermitted: [UUID: Bool] = [:]
        for membership in memberships {
            // Billing gate BEFORE connecting: a lapsed group subscription
            // makes the relay answer 402 at the WebSocket upgrade, which
            // reaches the client only as an untyped error string and would
            // reconnect-loop forever. Ask the server instead, and publish
            // `payment_required` — a first-class state the sidebar renders
            // as a billing banner, never as a red failure.
            let billing = await GroupDirectorySync.fetchGroupSubscriptionStatus(
                endpoint: credentials.endpoint, token: credentials.token, groupId: membership.id)
            startupPermitted[membership.id] = GroupDirectorySync.groupSyncPermitted(status: billing)
            guard GroupDirectorySync.groupSyncPermitted(status: billing) else {
                log("Group spoke SKIPPED (billing \(billing ?? "unknown")): \(membership.name)")
                DaemonStatus.writeSpoke(groupId: membership.id, name: membership.name,
                                        state: "payment_required",
                                        detail: "Team sync paused — subscription \(billing ?? "inactive")")
                continue
            }
            guard let spoke = SyncService.openGroupLattice(
                claudeDir: claudeDir,
                groupId: membership.id,
                authToken: credentials.token,
                wssBase: credentials.httpBase,
                selfUserId: selfUserId
            ) else {
                log("Group spoke FAILED to open: \(membership.name) (\(membership.id))")
                DaemonStatus.writeSpoke(groupId: membership.id, name: membership.name,
                                        state: "error", detail: "could not open spoke database")
                continue
            }
            groupSpokes[membership.id] = spoke
            // decision 13 backstop: an exposure recorded while the spoke did
            // not exist yet (visualizer toggle or `memory-sync expose` before
            // the daemon's first multi-spoke run) has its GroupProjectMap row
            // written HERE, when the spoke first opens. Idempotent upsert.
            if let me = selfUserId {
                let gidString = membership.id.uuidString
                for config in localLattice.objects(SyncConfig.self).snapshot()
                    where config.exposedGroups.contains(gidString) {
                    GroupProjectMapWriter.upsert(into: spoke, me: me,
                                                 localProject: config.project) { log($0) }
                }
            }
            DaemonStatus.writeSpoke(groupId: membership.id, name: membership.name,
                                    state: billing == "past_due" ? "past_due" : "connected",
                                    detail: billing == "past_due"
                                        ? "Subscription past due — sync continues during the grace period"
                                        : nil)
            log("Group spoke opened: \(membership.name) → group-\(membership.id.uuidString).sqlite")
        }
        // Retire spokes for groups we are no longer in: quarantine by rename
        // so MCP discovery stops attaching them immediately (the file stays
        // for forensics), and drop the channel's sync bookkeeping so its
        // stale confirmations can't collapse entries the live channels still
        // need (Stage A6).
        //
        // Gated on an AUTHORITATIVE membership list. Retiring on a cached or
        // failed fetch would let one bad token wipe every group mirror.
        if membershipsAuthoritative {
            retireDepartedSpokes(claudeDir: claudeDir,
                                 activeGroupIds: Set(memberships.map(\.id)),
                                 hub: localLattice)
        } else {
            log("Skipping spoke retirement — membership list is cached, not authoritative")
        }

        // 7. Observe SyncConfig changes — rebuild filter dynamically
        // A SyncConfig write can change the personal policy OR a project's
        // group exposure — both live in the same row, so every channel is
        // rebuilt, each against its own filter.
        let observedGroupIds = memberships.map(\.id)
        let syncConfigObserver = localLattice.observe(SyncConfig.self) { _ in
            localLattice.updateSyncFilter(SyncService.buildSyncFilter(from: localLattice),
                                          forChannel: channel)
            for groupId in observedGroupIds {
                localLattice.updateSyncFilter(
                    SyncService.buildGroupSyncFilter(from: localLattice, groupId: groupId),
                    forChannel: SyncService.groupChannel(groupId))
            }
            log("Sync filters rebuilt (SyncConfig changed)")
        }

        // 8. Wire sync error, state, and progress logging
        // Lattice 1.x: the callback registrations became AsyncStreams.
        // Consume them in detached tasks that live as long as the daemon.
        // Sealed-query failures (lattice 1.4.1) log rather than pass
        // silently — a degraded read under memory pressure should be
        // findable in the daemon log.
        localLattice.onQueryError { message in
            log("hub query failed (degraded to empty result): \(message)")
        }
        syncedLattice?.onQueryError { message in
            log("synced query failed (degraded to empty result): \(message)")
        }
        let ipcProgressStream = localLattice.syncProgressStream
        Task.detached {
            for await progress in ipcProgressStream {
                log("IPC relay: acked=\(progress.acked) total=\(progress.totalUpload) pending=\(progress.pendingUpload) uploading=\(progress.isUploading)")
            }
        }

        if let syncedLattice {
        syncedLattice.onSyncError { error in
            log("WSS error: \(error)")
            DaemonStatus.write(state: "error", detail: "\(error)")
        }

        syncedLattice.onSyncStateChange { connected in
            log("WSS \(connected ? "connected" : "disconnected")")
            DaemonStatus.write(state: connected ? "connected" : "disconnected", detail: nil)
        }

        let wssProgressStream = syncedLattice.syncProgressStream
        Task.detached {
            for await progress in wssProgressStream {
                if progress.isUploading {
                    log("WSS upload: \(progress.acked)/\(progress.totalUpload) (\(progress.pendingUpload) pending)")
                }
                // Record last activity + pending depth for the health file. A
                // fully-acked idle progress event is the "caught up" signal.
                DaemonStatus.write(state: "connected",
                                   detail: nil,
                                   pendingUpload: progress.pendingUpload,
                                   didSync: progress.acked > 0 && progress.pendingUpload == 0)
            }
        }
        } else {
            // Group-only: there is no personal WSS to report on, so publish
            // a state the UI can render as healthy rather than leaving the
            // health file stuck on "starting" forever.
            DaemonStatus.write(state: "connected",
                               detail: "Group sync only (no personal subscription)")
        }

        // 8b. Membership poll — group changes rebuild the channel set via
        //     a launchd-backed restart (ipcTargets are config-time only).
        startMembershipPoll(claudeDir: claudeDir,
                            credentials: credentials,
                            knownGroupIds: Set(memberships.map(\.id)),
                            startupPermitted: startupPermitted,
                            lockFd: lockFd)

        // Handle graceful shutdown
        signal(SIGTERM, SIG_IGN)
        let termSource = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
        termSource.setEventHandler {
            log("SIGTERM received, shutting down...")
            DaemonStatus.write(state: "stopped", detail: "SIGTERM")
            flock(lockFd, LOCK_UN)
            close(lockFd)
            Darwin.exit(0)
        }
        termSource.resume()

        signal(SIGINT, SIG_IGN)
        let intSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
        intSource.setEventHandler {
            log("SIGINT received, shutting down...")
            flock(lockFd, LOCK_UN)
            close(lockFd)
            Darwin.exit(0)
        }
        intSource.resume()

        // 9. File watchdog — detect if synced DB is deleted and restart
        let watchdogFd = Darwin.open(syncedDbPath, O_RDONLY)
        if watchdogFd >= 0 {
            let watchdog = DispatchSource.makeFileSystemObjectSource(
                fileDescriptor: watchdogFd,
                eventMask: [.delete, .rename],
                queue: .main
            )
            watchdog.setEventHandler {
                let events = watchdog.data
                if events.contains(.delete) || events.contains(.rename) {
                    log("CRITICAL: Synced database was deleted/renamed — restarting daemon")
                    DaemonStatus.write(state: "restarting", detail: "synced DB deleted")
                    Darwin.close(watchdogFd)
                    flock(lockFd, LOCK_UN)
                    Darwin.close(lockFd)
                    // Exit with failure so launchd restarts us
                    Darwin.exit(1)
                }
            }
            watchdog.setCancelHandler {
                Darwin.close(watchdogFd)
            }
            watchdog.resume()
        } else {
            log("Warning: Could not open synced DB for file monitoring")
        }

        log("Sync daemon running (PID \(ProcessInfo.processInfo.processIdentifier))")
        DaemonStatus.write(state: "running",
                           detail: "PID \(ProcessInfo.processInfo.processIdentifier)")

        // 10. Sit forever — launchd manages lifecycle.
        //
        // Parked ASYNCHRONOUSLY, not with dispatchMain(). `run()` on an
        // @main AsyncParsableCommand executes on a cooperative-pool thread,
        // and dispatch_main() from a non-main thread is a hard client crash
        // (DISPATCH_CLIENT_CRASH "dispatch_main() must be called on the main
        // thread") — measured here as SIGTRAP/exit 133, which under launchd's
        // KeepAlive(SuccessfulExit: false) is a permanent restart loop that
        // kills the SyncConfig observer microseconds after registering it
        // (so `memory-sync expose` would silently never take effect).
        //
        // The Swift async-main runtime already drains the dispatch main queue
        // on the real main thread, so the `.main`-queue DispatchSources above
        // (SIGTERM handler, synced-DB watchdog) keep firing — verified with a
        // standalone repro before this change.
        while true {
            try await Task.sleep(for: .seconds(3600))
            withExtendedLifetime(syncConfigObserver) {}
        }
    }
}

// MARK: - Daemon Status File

/// Health surface for the Visualizer / user: a small JSON file the daemon
/// updates on every state transition. Content-based (not just "is the process
/// alive") so the UI can distinguish waiting-for-auth from connected-and-idle
/// from error. Best-effort — a failed write never affects sync.
enum DaemonStatus {
    private static let path = NSHomeDirectory() + "/.claude/sync-daemon-status.json"
    // lastSyncAt is sticky across writes that don't themselves sync.
    // Guarded by `lock` on every access, so the unsafe global is sound.
    nonisolated(unsafe) private static var lastSyncAt: String?
    private static let lock = NSLock()

    /// Per-group spoke state, surfaced to the visualizer's sidebar.
    /// `payment_required` is a first-class state, distinct from `error`:
    /// the UI renders it as a billing banner (with a checkout link for
    /// owners), never as a red failure.
    nonisolated(unsafe) private static var spokes: [String: [String: Any]] = [:]

    static func writeSpoke(groupId: UUID, name: String, state: String, detail: String? = nil) {
        lock.lock()
        var entry: [String: Any] = ["name": name, "state": state]
        if let detail { entry["detail"] = detail }
        spokes[groupId.uuidString] = entry
        lock.unlock()
        // Re-publish so the change lands in the file immediately; the
        // top-level state is unchanged by a per-spoke update.
        write(state: lastKnownState ?? "connected", detail: lastKnownDetail)
    }

    nonisolated(unsafe) private static var lastKnownState: String?
    nonisolated(unsafe) private static var lastKnownDetail: String?

    static func write(state: String,
                      detail: String?,
                      pendingUpload: Int? = nil,
                      didSync: Bool = false) {
        lock.lock(); defer { lock.unlock() }
        let now = ISO8601DateFormatter().string(from: Date())
        if didSync { lastSyncAt = now }
        lastKnownState = state
        lastKnownDetail = detail
        var obj: [String: Any] = [
            "state": state,
            "updatedAt": now,
            "pid": ProcessInfo.processInfo.processIdentifier,
        ]
        if let detail { obj["detail"] = detail }
        if let pendingUpload { obj["pendingUpload"] = pendingUpload }
        if let lastSyncAt { obj["lastSyncAt"] = lastSyncAt }
        if !spokes.isEmpty { obj["groups"] = spokes }
        guard let data = try? JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted]) else { return }
        try? data.write(to: URL(fileURLWithPath: path), options: [.atomic])
    }
}

// MARK: - Credential Loading

struct SyncCredentials {
    let token: String
    let endpoint: String
    /// ws(s):// base — group channels append /sync/group/<id>.
    var httpBase: URL {
        let ws = endpoint
            .replacingOccurrences(of: "https://", with: "wss://")
            .replacingOccurrences(of: "http://", with: "ws://")
        return URL(string: ws)!
    }
    var wssEndpoint: URL {
        let ws = endpoint
            .replacingOccurrences(of: "https://", with: "wss://")
            .replacingOccurrences(of: "http://", with: "ws://")
        return URL(string: "\(ws)/sync")!
    }
}

func readCredentials(claudeDir: String, endpointOverride: String?) -> SyncCredentials? {
    guard let token = keychainLoad(service: "io.engram.app", account: "auth_token") else {
        return nil
    }
    let endpoint = endpointOverride ?? "https://engramdb.io"
    return SyncCredentials(token: token, endpoint: endpoint)
}

/// Minimal Keychain read (no dependency on KeychainHelper from Visualizer)
func keychainLoad(service: String, account: String) -> String? {
    let query: [String: Any] = [
        kSecClass as String: kSecClassGenericPassword,
        kSecAttrService as String: service,
        kSecAttrAccount as String: account,
        kSecReturnData as String: true,
        kSecMatchLimit as String: kSecMatchLimitOne,
    ]
    var result: AnyObject?
    let status = SecItemCopyMatching(query as CFDictionary, &result)
    guard status == errSecSuccess, let data = result as? Data else { return nil }
    return String(data: data, encoding: .utf8)
}

// MARK: - Group spoke lifecycle

/// Opens the hub read-only-ish for filter construction BEFORE the real
/// (IPC-carrying) open. Filters must exist at configuration time, but the
/// SyncConfig rows that define them live in the very database being opened —
/// so read them through a separate plain handle first.
private func hubFilterSource(_ dbPath: String) -> Lattice {
    // Plain open: no IPC, no WSS. Same file, so the row set is identical.
    (try? Lattice(
        Memory.self, Edge.self, Checkpoint.self,
        HookState.self, SessionState.self, SyncConfig.self,
        configuration: .init(fileURL: URL(fileURLWithPath: dbPath),
                             migration: engramMigrations)
    )) ?? {
        // Unreachable in practice (the caller opens the same file moments
        // later and exits on failure); an in-memory stand-in keeps filter
        // construction total rather than crashing the daemon.
        try! Lattice(Memory.self, Edge.self, SyncConfig.self,
                     configuration: .init(storage: .memory()))
    }()
}

/// Quarantine spokes for groups this user is no longer in, and retire their
/// hub channel state.
///
/// Rename rather than delete: MCP discovery globs `group-*.sqlite`, so the
/// rename stops attachment at once (combined with readLattice's per-call
/// liveness check, a kicked member's live sessions stop serving group data
/// at the next recall) while the file survives for forensics.
private func retireDepartedSpokes(claudeDir: String, activeGroupIds: Set<UUID>, hub: Lattice) {
    let syncDir = (SyncService.syncedDbPath(claudeDir: claudeDir) as NSString)
        .deletingLastPathComponent
    guard let entries = try? FileManager.default.contentsOfDirectory(atPath: syncDir) else { return }
    for entry in entries where entry.hasPrefix("group-") && entry.hasSuffix(".sqlite") {
        let idString = String(entry.dropFirst("group-".count).dropLast(".sqlite".count))
        guard let groupId = UUID(uuidString: idString), !activeGroupIds.contains(groupId) else {
            continue
        }
        // Stage A6: a dead channel's stale sync_state/sync_set rows can let
        // an audit entry collapse before a LIVE channel relayed it (silent
        // relay loss), and its stale pending rows pin entries forever.
        hub.removeSyncChannelState(ipcChannel: SyncService.groupChannel(groupId))
        let from = syncDir + "/" + entry
        let to = SyncService.revokedGroupDbPath(claudeDir: claudeDir, groupId: groupId)
        try? FileManager.default.removeItem(atPath: to)
        do {
            try FileManager.default.moveItem(atPath: from, toPath: to)
            // WAL/SHM siblings follow the database.
            for suffix in ["-wal", "-shm"] {
                try? FileManager.default.moveItem(atPath: from + suffix, toPath: to + suffix)
            }
            log("Group spoke retired (membership ended): \(entry) → \(to as NSString).lastPathComponent")
        } catch {
            log("Group spoke retire FAILED for \(entry): \(error)")
        }
    }
}

/// Poll memberships; relaunch when the set changes.
///
/// `ipcTargets` are configuration-time only in the core, so a membership
/// change cannot be applied in place — the daemon writes the refreshed
/// directory, then exits NON-zero so launchd restarts it with the new
/// channel set. Exiting is safe: the hub is a durable queue, and the
/// restart re-reads every filter from SyncConfig.
private func startMembershipPoll(
    claudeDir: String,
    credentials: SyncCredentials,
    knownGroupIds: Set<UUID>,
    startupPermitted: [UUID: Bool],
    lockFd: Int32
) {
    Task.detached {
        while true {
            // Wake early when the visualizer requests a refresh after an
            // interactive membership change.
            var slept = 0
            while slept < 300 {
                try? await Task.sleep(for: .seconds(5))
                slept += 5
                if GroupDirectorySync.consumeRefreshRequest(claudeDir: claudeDir) {
                    log("Groups: refresh requested by the app")
                    break
                }
            }

            do {
                let snapshot = try await GroupDirectorySync.fetch(
                    endpoint: credentials.endpoint, token: credentials.token)
                try? GroupDirectorySync.write(snapshot, claudeDir: claudeDir)
                // Billing can change without membership changing — a lapse
                // must pause the spoke and a payment must resume it. Both
                // require the restart path (spokes are opened at startup).
                for membership in snapshot.groups {
                    let billing = await GroupDirectorySync.fetchGroupSubscriptionStatus(
                        endpoint: credentials.endpoint, token: credentials.token,
                        groupId: membership.id)
                    let permitted = GroupDirectorySync.groupSyncPermitted(status: billing)
                    // Compare against the DECISION made at startup, not
                    // against whether a spoke happens to be running: a group
                    // whose spoke failed to open for a non-billing reason
                    // would otherwise look like a billing transition on every
                    // single poll and restart the daemon forever. A group
                    // this poll has never seen (just joined) has no startup
                    // decision — the membership check below handles it.
                    guard let wasPermitted = startupPermitted[membership.id] else { continue }
                    if permitted != wasPermitted {
                        log("Group billing changed for \(membership.name) (\(billing ?? "unknown")) — restarting")
                        DaemonStatus.write(state: "restarting", detail: "group billing changed")
                        flock(lockFd, LOCK_UN)
                        close(lockFd)
                        Darwin.exit(1)
                    }
                    if !permitted {
                        DaemonStatus.writeSpoke(groupId: membership.id, name: membership.name,
                                                state: "payment_required",
                                                detail: "Team sync paused — subscription \(billing ?? "inactive")")
                    }
                }

                let current = Set(snapshot.groups.map(\.id))
                if current != knownGroupIds {
                    let added = current.subtracting(knownGroupIds).count
                    let removed = knownGroupIds.subtracting(current).count
                    log("Groups changed (+\(added)/-\(removed)) — restarting to rebuild IPC targets")
                    DaemonStatus.write(state: "restarting", detail: "group membership changed")
                    flock(lockFd, LOCK_UN)
                    close(lockFd)
                    // Non-zero: launchd's KeepAlive(SuccessfulExit:false)
                    // restarts us; a clean exit would NOT come back.
                    Darwin.exit(1)
                }
            } catch GroupDirectorySync.FetchError.unauthorized {
                log("Groups: poll unauthorized — leaving membership set unchanged")
            } catch {
                log("Groups: poll failed (\(error)) — retrying next cycle")
            }
        }
    }
}

// MARK: - Logging

private func log(_ message: String) {
    let ts = ISO8601DateFormatter().string(from: Date())
    let line = "[\(ts)] \(message)\n"
    fputs(line, stderr)

    // Also append to log file
    let logPath = NSHomeDirectory() + "/.claude/sync-daemon.log"
    if let fh = FileHandle(forWritingAtPath: logPath) {
        defer { fh.closeFile() }
        fh.seekToEndOfFile()
        fh.write(Data(line.utf8))
    } else {
        FileManager.default.createFile(atPath: logPath, contents: Data(line.utf8))
    }
}
