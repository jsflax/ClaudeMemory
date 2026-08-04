import ArgumentParser
import EngramKit
import EngramModels
import Foundation
import Lattice

/// `memory-sync expose <project> <groupId>` — the CLI-only exposure path.
///
/// Sign-in is Visualizer-only today, so the headless bootstrap (and the
/// two-account E2E demo) needs a way to share a project with a group
/// without the UI. This writes the same `SyncConfig.exposedGroups` row the
/// sidebar's checklist writes.
///
/// It must go through Lattice rather than `sqlite3`: a raw SQL write
/// bypasses the audit log, so the daemon's IPC relay would never see the
/// change and the project would silently never sync.
struct ExposeCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "expose",
        abstract: "Share a local project with a group (or stop sharing it).",
        discussion: """
        Writes SyncConfig.exposedGroups in ~/.claude/memory.sqlite, which the
        running daemon observes to rebuild that group's channel filter.

        Un-sharing stops FUTURE sharing only — memories the group already
        received stay with the group (retract an individual memory by
        marking it private).
        """
    )

    @Argument(help: "Local project name (as it appears in your memories).")
    var project: String

    @Argument(help: "Group UUID (see `memory-sync groups`).")
    var groupId: String

    @Flag(name: .long, help: "Stop sharing this project with the group.")
    var unshare: Bool = false

    func run() async throws {
        guard let gid = UUID(uuidString: groupId) else {
            throw ValidationError("'\(groupId)' is not a valid group UUID")
        }
        let claudeDir = NSHomeDirectory() + "/.claude"
        let dbPath = claudeDir + "/memory.sqlite"
        guard FileManager.default.fileExists(atPath: dbPath) else {
            throw ValidationError("No memory database at \(dbPath)")
        }

        // Plain open (no IPC/WSS): the running daemon owns the relay and
        // picks this write up through its SyncConfig observer.
        let lattice = try Lattice(
            Memory.self, Edge.self, Checkpoint.self,
            HookState.self, SessionState.self, SyncConfig.self,
            configuration: .init(fileURL: URL(fileURLWithPath: dbPath),
                                 migration: engramMigrations))

        let existing = lattice.objects(SyncConfig.self)
            .where { $0.project == project }.first
        var groups = existing?.exposedGroups ?? []
        if unshare {
            groups.remove(gid.uuidString)
        } else {
            groups.insert(gid.uuidString)
        }

        if let existing {
            existing.exposedGroups = groups
            existing.updatedAt = Date()
        } else {
            try lattice.add(SyncConfig(project: project, policy: .local,
                                       exposedTeams: groups))
        }

        // Attribute the project's rows so they can cross the spoke→hub
        // firewall (which admits only authored-by-me) and render with a
        // name rather than [by:unknown] for teammates. The daemon's
        // start-up sweep is the recurring backstop; doing it here means the
        // very first share is attributed too.
        if !unshare, let me = GroupDirectory.currentUserId() {
            var stamped = 0
            for memory in lattice.objects(Memory.self)
                .where({ $0.project == project && $0.authorUserId == nil })
                .snapshot() {
                memory.authorUserId = me
                stamped += 1
            }
            if stamped > 0 { print("Attributed \(stamped) previously-unattributed memories to you.") }
        }

        let action = unshare ? "No longer sharing" : "Sharing"
        print("\(action) '\(project)' with group \(gid.uuidString).")
        if !unshare {
            print("The daemon will relay matching memories on its group channel.")
        } else {
            print("Memories already shared remain with the group.")
        }
    }
}

/// `memory-sync groups` — list memberships from the daemon-written
/// directory, so the CLI flow can discover group ids without the app.
struct GroupsCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "groups",
        abstract: "List the groups you belong to (from the daemon's directory)."
    )

    func run() async throws {
        guard let file = GroupDirectory.load() else {
            print("No group directory yet (~/.claude/sync/groups.json).")
            print("Sign in and let the sync daemon run once.")
            return
        }
        if file.groups.isEmpty {
            print("You are not a member of any groups.")
            return
        }
        if let selfId = file.selfUserId {
            print("You: \(selfId.uuidString)")
        }
        for group in file.groups.sorted(by: { $0.name < $1.name }) {
            let role = group.myRole.map { " [\($0)]" } ?? ""
            let kind = (group.root ?? false) ? "org" : "team"
            print("\(group.id.uuidString)  \(group.name)\(role)  (\(kind))")
        }
    }
}
