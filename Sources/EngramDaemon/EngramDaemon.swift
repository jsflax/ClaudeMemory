import ArgumentParser
import EngramKit
import EngramModels
import Foundation
import Lattice

@main
struct EngramDaemon: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "memory-sync",
        abstract: "Engram sync daemon — relays memory changes to the cloud via WSS."
    )

    @Option(name: .long, help: "Lattice log level: off, error, warning, info, debug")
    var logLevel: String?

    @Option(name: .long, help: "Sync server endpoint (default: https://engramdb.io)")
    var endpoint: String?

    @Flag(name: .long, help: "Nuclear compaction: wipe all sync state and re-sync from scratch")
    var nuclearCompact: Bool = false

    func run() throws {
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

        // 2. Read auth credentials
        guard let credentials = readCredentials(claudeDir: claudeDir, endpointOverride: endpoint) else {
            log("No auth credentials found. Sign in via the Visualizer first.")
            flock(lockFd, LOCK_UN)
            close(lockFd)
            throw ExitCode.success  // Exit cleanly — launchd won't restart on successful exit
        }
        log("Auth loaded (endpoint: \(credentials.endpoint))")

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

        // Now open with IPC — replication slot cursor will be 0 after nuclear compact
        var localConfig = Lattice.Configuration(
            fileURL: URL(fileURLWithPath: dbPath),
            migration: engramMigrations
        )
        localConfig.ipcTargets = [.init(channel: channel)]

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

        // 5. Build and push sync filter
        let filter = SyncService.buildSyncFilter(from: localLattice)
        localLattice.updateSyncFilter(filter)
        log("Sync filter applied")

        // 6. Open syncedLattice with WSS + IPC (in daemon-owned sync/ directory)
        let syncedDbPath = SyncService.syncedDbPath(claudeDir: claudeDir)
        guard let syncedLattice = SyncService.openSyncedLattice(
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
        log("Synced lattice opened (WSS: \(credentials.wssEndpoint), db: \(syncedDbPath))")

        // 7. Observe SyncConfig changes — rebuild filter dynamically
        let syncConfigObserver = localLattice.observe(SyncConfig.self) { _ in
            let newFilter = SyncService.buildSyncFilter(from: localLattice)
            localLattice.updateSyncFilter(newFilter)
            log("Sync filter rebuilt (SyncConfig changed)")
        }

        // 8. Wire sync error, state, and progress logging
        localLattice.onSyncProgress { progress in
            log("IPC relay: acked=\(progress.acked) total=\(progress.totalUpload) pending=\(progress.pendingUpload) uploading=\(progress.isUploading)")
        }

        syncedLattice.onSyncError { error in
            log("WSS error: \(error)")
        }

        syncedLattice.onSyncStateChange { connected in
            log("WSS \(connected ? "connected" : "disconnected")")
        }

        syncedLattice.onSyncProgress { progress in
            if progress.isUploading {
                log("WSS upload: \(progress.acked)/\(progress.totalUpload) (\(progress.pendingUpload) pending)")
            }
        }

        // Handle graceful shutdown
        signal(SIGTERM, SIG_IGN)
        let termSource = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
        termSource.setEventHandler {
            log("SIGTERM received, shutting down...")
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

        // 10. Sit forever — launchd manages lifecycle.
        // withExtendedLifetime keeps the observer token (and other locals)
        // alive across the never-returning dispatchMain().
        withExtendedLifetime(syncConfigObserver) {
            dispatchMain()
        }
    }
}

// MARK: - Credential Loading

private struct SyncCredentials {
    let token: String
    let endpoint: String
    var wssEndpoint: URL {
        let ws = endpoint
            .replacingOccurrences(of: "https://", with: "wss://")
            .replacingOccurrences(of: "http://", with: "ws://")
        return URL(string: "\(ws)/sync")!
    }
}

private func readCredentials(claudeDir: String, endpointOverride: String?) -> SyncCredentials? {
    guard let token = keychainLoad(service: "io.engram.app", account: "auth_token") else {
        return nil
    }
    let endpoint = endpointOverride ?? "https://engramdb.io"
    return SyncCredentials(token: token, endpoint: endpoint)
}

/// Minimal Keychain read (no dependency on KeychainHelper from Visualizer)
private func keychainLoad(service: String, account: String) -> String? {
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
