import ArgumentParser
import EngramKit
import EngramModels
import Foundation
import Lattice

/// One-time repair for the Aug 2026 audit-log explosion: compacts the
/// bloated AuditLog (4.69M rows on a 28K-memory hub), purges dead per-entry
/// sync bookkeeping, normalizes TEXT timestamps minted by the old apply
/// path, and truncates the multi-GB WALs — on every Engram database file on
/// the machine (hub, personal mirror, group spokes).
///
/// Requires the FIXED core (latticecore >= 1.3.0): floor-keyed compaction,
/// download-cursor preservation, and the empty-pass floor advance. Running
/// this against an old binary would compact nothing (group slots at
/// confirmed=0 vetoed everything) — the guard is implicit: this command
/// ships in the same binary as the fixed core.
///
/// QUIESCED by contract: refuses to run while the daemon holds its lock,
/// Engram.app is running, or any other process holds the hub open. Repair
/// under live readers/writers would race the compaction transaction and the
/// WAL truncate would silently fail — the half-fixed state is worse than
/// deferring.
struct RepairAuditCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "repair-audit",
        abstract: "One-time audit-log repair: compact history, purge dead sync state, truncate WALs."
    )

    @Flag(name: .long, help: "Report what would happen without modifying anything")
    var dryRun: Bool = false

    @Flag(name: .long, help: "Skip the exclusive-access check (repair may be partial under live readers)")
    var force: Bool = false

    /// Rehearse on a COPY before touching the live databases — the plan's
    /// first verification step, and impossible without an explicit root
    /// (NSHomeDirectory ignores $HOME here).
    @Option(name: .long, help: "Engram data directory (default: ~/.claude)")
    var claudeDirOverride: String?

    func run() async throws {
        let claudeDir = claudeDirOverride ?? (NSHomeDirectory() + "/.claude")
        let hubPath = claudeDir + "/memory.sqlite"

        // ---- Quiesce gate -------------------------------------------------
        // Daemon: its flock is authoritative.
        let lockPath = claudeDir + "/engram-sync.lock"
        let lockFd = open(lockPath, O_CREAT | O_RDWR, 0o600)
        defer { if lockFd >= 0 { close(lockFd) } }
        if lockFd >= 0, flock(lockFd, LOCK_EX | LOCK_NB) != 0 {
            print("✗ The sync daemon is running. Stop it first:")
            print("    launchctl bootout gui/$(id -u)/io.engram.sync")
            throw ExitCode.failure
        }
        // Any other holder of the hub (MCP servers from open Claude Code
        // sessions, Engram.app). lsof is the honest check — a repair from
        // inside a Claude session would otherwise always pass.
        let holders = Self.otherHolders(of: hubPath)
        if !holders.isEmpty && !force {
            print("✗ Other processes hold the memory database (close them or re-run with --force):")
            for h in holders { print("    \(h)") }
            print("  Likely: open Claude Code sessions (their memory MCP servers) or Engram.app.")
            throw ExitCode.failure
        }

        // ---- The four database files -------------------------------------
        var targets: [(path: String, kind: Kind)] = [(hubPath, .hub)]
        let syncedPath = SyncService.syncedDbPath(claudeDir: claudeDir)
        if FileManager.default.fileExists(atPath: syncedPath) {
            targets.append((syncedPath, .synced))
        }
        for spoke in SyncService.discoverGroupSpokes(claudeDir: claudeDir) {
            targets.append((spoke.path, .spoke))
        }

        for (path, kind) in targets {
            try await repair(path: path, kind: kind)
        }
        print(dryRun ? "Dry run complete." : "Repair complete. Re-enable the daemon with:")
        if !dryRun {
            print("    launchctl enable gui/$(id -u)/io.engram.sync && launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/io.engram.sync.plist")
        }
    }

    private enum Kind { case hub, synced, spoke }

    private func repair(path: String, kind: Kind) async throws {
        let name = (path as NSString).lastPathComponent
        let auditBefore = Self.sqliteScalar(path, "SELECT COUNT(*) FROM AuditLog") ?? -1
        let stateBefore = Self.sqliteScalar(path, "SELECT COUNT(*) FROM _lattice_sync_state") ?? -1
        let walBefore = Self.fileSize(path + "-wal")
        print("\(name): \(auditBefore) audit rows, \(stateBefore) sync-state rows, WAL \(Self.human(walBefore))")
        if dryRun { return }

        // Backup: db + wal side-by-side; rollback = move them back.
        let stamp = ISO8601DateFormatter().string(from: Date())
            .replacingOccurrences(of: ":", with: "-")
        for suffix in ["", "-wal"] where FileManager.default.fileExists(atPath: path + suffix) {
            try FileManager.default.copyItem(
                atPath: path + suffix,
                toPath: path + suffix + ".repair-backup-\(stamp)")
        }

        // TEXT-timestamp normalization (old apply path bound ISO strings
        // into the REAL column). sqlite3 CLI: the Swift wrapper deliberately
        // has no raw-SQL surface, and this is a one-time repair on a
        // quiesced file. unixepoch() parses the ISO forms directly.
        _ = Self.sqlite3(path,
            "UPDATE AuditLog SET timestamp = unixepoch(timestamp) WHERE typeof(timestamp) = 'text'")

        // Compaction under the fixed rules (floor-keyed, cursor-preserving,
        // transactional) + WAL truncate. The remaining rows are the genuine
        // pending backlog above each channel's floor — they drain through
        // the fixed uploader once the daemon reconnects, and the daemon's
        // hourly compaction sweeps them as floors advance.
        let lattice: Lattice
        switch kind {
        case .hub:
            lattice = try Lattice(Memory.self, Edge.self, Checkpoint.self,
                                  HookState.self, SessionState.self, SyncConfig.self,
                                  configuration: .init(fileURL: URL(fileURLWithPath: path),
                                                       migration: engramMigrations))
        case .synced:
            lattice = try Lattice(Memory.self, Edge.self, SyncConfig.self,
                                  configuration: .init(fileURL: URL(fileURLWithPath: path),
                                                       migration: engramMigrations))
        case .spoke:
            lattice = try Lattice(Memory.self, Edge.self, GroupProjectMap.self,
                                  configuration: .init(fileURL: URL(fileURLWithPath: path),
                                                       migration: engramMigrations))
        }
        let deleted = lattice.compactHistory()
        lattice.vacuum()
        lattice.checkpoint()  // quiesced: the TRUNCATE lands
        lattice.close()

        let auditAfter = Self.sqliteScalar(path, "SELECT COUNT(*) FROM AuditLog") ?? -1
        let walAfter = Self.fileSize(path + "-wal")
        print("  → compacted \(deleted) rows; \(auditAfter) remain (pending backlog); WAL \(Self.human(walAfter))")
    }

    // MARK: - Helpers (quiesced-file introspection)

    private static func otherHolders(of path: String) -> [String] {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/sbin/lsof")
        p.arguments = ["-t", path, path + "-wal"]
        let out = Pipe()
        p.standardOutput = out
        p.standardError = Pipe()
        try? p.run()
        p.waitUntilExit()
        let text = String(data: out.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        let me = ProcessInfo.processInfo.processIdentifier
        return Set(text.split(separator: "\n").compactMap { Int32($0) })
            .filter { $0 != me }
            .map { pid in
                let ps = Process()
                ps.executableURL = URL(fileURLWithPath: "/bin/ps")
                ps.arguments = ["-o", "comm=", "-p", "\(pid)"]
                let o = Pipe(); ps.standardOutput = o
                try? ps.run(); ps.waitUntilExit()
                let comm = String(data: o.fileHandleForReading.readDataToEndOfFile(),
                                  encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "?"
                return "pid \(pid) (\(comm))"
            }
            .sorted()
    }

    private static func sqlite3(_ path: String, _ sql: String) -> Bool {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/sqlite3")
        p.arguments = [path, sql]
        p.standardOutput = Pipe(); p.standardError = Pipe()
        try? p.run()
        p.waitUntilExit()
        return p.terminationStatus == 0
    }

    private static func sqliteScalar(_ path: String, _ sql: String) -> Int64? {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/sqlite3")
        p.arguments = ["-readonly", path, sql]
        let out = Pipe()
        p.standardOutput = out; p.standardError = Pipe()
        try? p.run()
        p.waitUntilExit()
        let s = String(data: out.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return Int64(s)
    }

    private static func fileSize(_ path: String) -> Int64 {
        ((try? FileManager.default.attributesOfItem(atPath: path))?[.size] as? Int64) ?? 0
    }

    private static func human(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }
}
