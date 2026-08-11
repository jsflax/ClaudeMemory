import ArgumentParser
import EngramKit
import Foundation

// Account-scoped maintenance subcommands. These live in memory-sync — not a
// separate tool — because this binary is the ONE process with a Keychain ACL
// for the auth token: the secret never leaves it, and every command operates
// strictly on the signed-in user's own account.

/// Shared plumbing: token + a JSON call helper.
private func requireCredentials(endpoint: String?) throws -> SyncCredentials {
    let claudeDir = NSHomeDirectory() + "/.claude"
    switch readCredentials(claudeDir: claudeDir, endpointOverride: endpoint,
                           timeout: keychainInteractiveDeadline) {
    case .found(let creds):
        return creds
    case .notFound:
        throw ValidationError("No auth token in the Keychain — sign in via the Engram app first.")
    case .timedOut:
        // Distinct from "no token": the item is there, securityd just won't
        // hand it over to a binary whose signature the ACL doesn't accept.
        // Say so — a bare "not signed in" sends people to re-authenticate
        // over and over against a wall.
        throw ValidationError(keychainBlockedDiagnostic())
    }
}

private func call(
    _ method: String, _ path: String, creds: SyncCredentials, body: [String: Any]? = nil
) async throws -> (status: Int, json: Any?) {
    var request = URLRequest(url: URL(string: creds.endpoint + path)!)
    request.httpMethod = method
    request.setValue("Bearer \(creds.token)", forHTTPHeaderField: "Authorization")
    if let body {
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
    }
    request.timeoutInterval = 120   // compaction over millions of rows takes a while
    let (data, response) = try await URLSession.shared.data(for: request)
    let status = (response as? HTTPURLResponse)?.statusCode ?? 0
    return (status, try? JSONSerialization.jsonObject(with: data))
}

/// `memory-sync whoami` — the signed-in identity, from the server.
struct WhoamiCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "whoami",
        abstract: "Show the signed-in account (from the server, not the cache)."
    )
    @Option(name: .long) var endpoint: String?

    func run() async throws {
        let creds = try requireCredentials(endpoint: endpoint)
        let (status, json) = try await call("GET", "/me", creds: creds)
        guard status == 200, let dict = json as? [String: Any] else {
            throw ValidationError("GET /me failed (HTTP \(status)) — token may be expired.")
        }
        print("email: \(dict["email"] as? String ?? "?")")
        print("id:    \(dict["id"] as? String ?? "?")")
        if let name = dict["fullName"] as? String { print("name:  \(name)") }
    }
}

/// `memory-sync accept-invite <token>` — join a group from the CLI, then
/// poke the daemon so the new spoke opens without waiting for the poll.
struct AcceptInviteCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "accept-invite",
        abstract: "Accept a group invite by its token (from the invite link)."
    )
    @Argument(help: "The invite token — the part after token= in the invite URL.")
    var token: String
    @Option(name: .long) var endpoint: String?

    func run() async throws {
        let creds = try requireCredentials(endpoint: endpoint)
        let (status, json) = try await call("POST", "/invites/accept", creds: creds,
                                            body: ["token": token])
        guard status == 200, let dict = json as? [String: Any] else {
            let reason = (json as? [String: Any])?["reason"] as? String ?? "HTTP \(status)"
            throw ValidationError("Accept failed: \(reason)")
        }
        print("Joined \(dict["groupName"] as? String ?? "group") as \(dict["role"] as? String ?? "member").")
        // The daemon polls memberships every 5 minutes; the marker file asks
        // it to refresh now, so the group spoke opens within seconds.
        let syncDir = NSHomeDirectory() + "/.claude/sync"
        FileManager.default.createFile(atPath: syncDir + "/groups-refresh-requested",
                                       contents: Data())
        print("Daemon poked — the group's memories will start syncing shortly.")
    }
}

/// `memory-sync compact-server-history` — collapse this account's
/// server-side audit history to a state snapshot.
struct CompactServerHistoryCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "compact-server-history",
        abstract: "Collapse this account's server-side sync history to a snapshot.",
        discussion: """
        The relay replays HISTORY to catching-up clients, so catch-up cost
        grows with every edit ever made — not with how many memories exist.
        Compaction makes it proportional to the data again.

        Afterwards every device re-syncs from the (now small) snapshot;
        memories themselves are untouched. All this account's clients must
        be on a current build (LatticeCore 1.1.0+).
        """
    )
    @Option(name: .long) var endpoint: String?

    func run() async throws {
        let creds = try requireCredentials(endpoint: endpoint)
        print("Compacting server-side history (this can take a minute on a large account)…")
        let (status, json) = try await call("POST", "/me/compact-history", creds: creds)
        guard status == 200, let dict = json as? [String: Any] else {
            let reason = (json as? [String: Any])?["reason"] as? String ?? "HTTP \(status)"
            throw ValidationError("Compaction failed: \(reason)")
        }
        let before = dict["entriesBefore"] as? Int ?? 0
        let after = dict["entriesAfter"] as? Int ?? 0
        let factor = after > 0 ? String(format: "%.1fx", Double(before) / Double(after)) : "∞"
        print("Done: \(before) audit entries -> \(after) (catch-up \(factor) smaller).")
        print("Devices will reconnect and re-sync from the snapshot automatically.")
    }
}
