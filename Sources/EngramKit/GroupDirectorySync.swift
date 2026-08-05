import Foundation
import EngramModels

/// Fetches the membership directory from the server and writes
/// `~/.claude/sync/groups.json`.
///
/// The DAEMON is the sole writer of that file (plan §3): short-lived
/// processes — the MCP server, every hook — need `authorUserId → name` and
/// the signed-in user's id without network or Keychain access, and
/// extending Keychain ACLs to every hook process is exactly where macOS
/// prompts or silently fails. The file is 0600; a user id is not a secret.
public enum GroupDirectorySync {

    public struct Membership: Sendable, Equatable {
        public let id: UUID
        public let name: String
        public let parentId: UUID?
        public let myRole: String?
        public let root: Bool
        /// True when this graph is granted ONLY via a team link (no
        /// direct/subtree membership). Additive — absent on old servers.
        public let viaLink: Bool?
        /// The team(s) whose links grant this graph.
        public let linkedFrom: [UUID]?

        public init(id: UUID, name: String, parentId: UUID?, myRole: String?,
                    root: Bool, viaLink: Bool? = nil, linkedFrom: [UUID]? = nil) {
            self.id = id
            self.name = name
            self.parentId = parentId
            self.myRole = myRole
            self.root = root
            self.viaLink = viaLink
            self.linkedFrom = linkedFrom
        }
    }

    public struct Snapshot: Sendable {
        public let selfUserId: UUID?
        public let groups: [Membership]
        /// userId → display name, flattened across every group.
        public let members: [String: String]

        public init(selfUserId: UUID?, groups: [Membership], members: [String: String]) {
            self.selfUserId = selfUserId
            self.groups = groups
            self.members = members
        }
    }

    public enum FetchError: Error {
        /// Auth rejected (401/403): the token is bad, not the network.
        case unauthorized
        case transport(String)
        case decoding(String)
    }

    // Server DTOs (mirror engram-server's GroupController).
    private struct GroupSummaryDTO: Decodable {
        let id: UUID
        let name: String
        let parentId: UUID?
        let myRole: String?
        let root: Bool
        let viaLink: Bool?
        let linkedFrom: [UUID]?
    }
    private struct MemberInfoDTO: Decodable {
        let userId: UUID
        let displayName: String
    }
    private struct ProfileDTO: Decodable {
        let id: UUID
    }

    private static func decoder() -> JSONDecoder {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601   // Vapor's default coder
        return d
    }

    private static func get(_ url: URL, token: String) async throws -> Data {
        var request = URLRequest(url: url)
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        request.timeoutInterval = 30
        let (data, response): (Data, URLResponse)
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            throw FetchError.transport(error.localizedDescription)
        }
        guard let http = response as? HTTPURLResponse else {
            throw FetchError.transport("no HTTP response")
        }
        if http.statusCode == 401 || http.statusCode == 403 {
            throw FetchError.unauthorized
        }
        guard (200..<300).contains(http.statusCode) else {
            throw FetchError.transport("HTTP \(http.statusCode)")
        }
        return data
    }

    /// Fetch memberships + the flattened member directory.
    ///
    /// A per-group member fetch that fails is SKIPPED rather than failing
    /// the whole snapshot: a directory missing one group's names degrades
    /// author badges to "teammate", while an exception here would leave the
    /// daemon with no directory at all.
    public static func fetch(endpoint: String, token: String) async throws -> Snapshot {
        guard let base = URL(string: endpoint) else {
            throw FetchError.transport("bad endpoint \(endpoint)")
        }
        let groupsData = try await get(base.appendingPathComponent("groups"), token: token)
        let dtos: [GroupSummaryDTO]
        do {
            dtos = try decoder().decode([GroupSummaryDTO].self, from: groupsData)
        } catch {
            throw FetchError.decoding("\(error)")
        }

        var selfId: UUID?
        if let meData = try? await get(base.appendingPathComponent("me"), token: token) {
            selfId = (try? decoder().decode(ProfileDTO.self, from: meData))?.id
        }

        var members: [String: String] = [:]
        for dto in dtos {
            let url = base.appendingPathComponent("groups")
                .appendingPathComponent(dto.id.uuidString)
                .appendingPathComponent("members")
            guard let data = try? await get(url, token: token),
                  let infos = try? decoder().decode([MemberInfoDTO].self, from: data) else {
                continue
            }
            for info in infos {
                members[info.userId.uuidString] = info.displayName
            }
        }

        return Snapshot(
            selfUserId: selfId,
            groups: dtos.map {
                Membership(id: $0.id, name: $0.name, parentId: $0.parentId,
                           myRole: $0.myRole, root: $0.root,
                           viaLink: $0.viaLink, linkedFrom: $0.linkedFrom)
            },
            members: members)
    }

    /// Whether the PERSONAL subscription entitles personal cloud sync.
    ///
    /// Lives here because it shares the authenticated-GET plumbing, and
    /// because the daemon needs both answers — personal entitlement and
    /// group memberships — to decide what it is allowed to run. A user with
    /// no personal subscription but a paid group seat is a first-class
    /// persona: group sync must work while the personal WSS stays closed
    /// (the server gates `/sync` behind the personal subscription, so
    /// opening it would loop on 402 and paint a false error state).
    ///
    /// Unreachable server ⇒ nil: "unknown", which callers treat as
    /// "keep whatever you were doing" rather than tearing sync down on a
    /// transient network failure.
    public static func fetchPersonalSyncEnabled(endpoint: String, token: String) async -> Bool? {
        guard let base = URL(string: endpoint) else { return nil }
        let url = base.appendingPathComponent("subscriptions").appendingPathComponent("status")
        guard let data = try? await get(url, token: token) else { return nil }
        struct StatusDTO: Decodable { let status: String }
        guard let dto = try? decoder().decode(StatusDTO.self, from: data) else { return nil }
        return dto.status == "active" || dto.status == "trialing"
    }

    /// Billing state of one group's BILLING ROOT, as the server reports it.
    ///
    /// Polled rather than inferred from transport errors: a lapsed
    /// subscription surfaces at the relay as a 402 during the WebSocket
    /// upgrade, which reaches the client only as an untyped error string.
    /// The server already publishes the truth, so ask it.
    ///
    /// Returns nil when unreachable/undecidable — callers keep their
    /// current state rather than reacting to a network blip.
    public static func fetchGroupSubscriptionStatus(
        endpoint: String, token: String, groupId: UUID
    ) async -> String? {
        guard let base = URL(string: endpoint) else { return nil }
        let url = base.appendingPathComponent("groups")
            .appendingPathComponent(groupId.uuidString)
            .appendingPathComponent("subscription")
        guard let data = try? await get(url, token: token) else { return nil }
        struct StatusDTO: Decodable { let status: String }
        return (try? decoder().decode(StatusDTO.self, from: data))?.status
    }

    /// Whether a reported group-subscription status permits sync.
    /// `past_due` DOES permit it — the server grants a grace window and
    /// sends a warning header rather than cutting members off mid-cycle.
    public static func groupSyncPermitted(status: String?) -> Bool {
        guard let status else { return true }   // unknown ⇒ don't tear down
        return status == "active" || status == "trialing" || status == "past_due"
    }

    /// Write `groups.json` atomically at 0600. Atomic because short-lived
    /// readers (hooks) can open it at any instant; a torn write would make
    /// every author render as "unknown" for that session.
    public static func write(_ snapshot: Snapshot, claudeDir: String) throws {
        let syncDir = (SyncService.syncedDbPath(claudeDir: claudeDir) as NSString)
            .deletingLastPathComponent
        let path = syncDir + "/groups.json"

        var groupsJSON: [[String: Any]] = []
        for group in snapshot.groups {
            var entry: [String: Any] = [
                "id": group.id.uuidString,
                "name": group.name,
                "root": group.root,
            ]
            if let parentId = group.parentId { entry["parentId"] = parentId.uuidString }
            if let role = group.myRole { entry["myRole"] = role }
            if group.viaLink == true { entry["viaLink"] = true }
            if let linkedFrom = group.linkedFrom, !linkedFrom.isEmpty {
                entry["linkedFrom"] = linkedFrom.map(\.uuidString)
            }
            groupsJSON.append(entry)
        }
        var root: [String: Any] = [
            "updatedAt": ISO8601DateFormatter().string(from: Date()),
            "groups": groupsJSON,
            "members": snapshot.members,
        ]
        if let selfId = snapshot.selfUserId { root["selfUserId"] = selfId.uuidString }

        let data = try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
        let tmp = path + ".tmp"
        try data.write(to: URL(fileURLWithPath: tmp), options: .atomic)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: tmp)
        _ = try FileManager.default.replaceItemAt(
            URL(fileURLWithPath: path), withItemAt: URL(fileURLWithPath: tmp))
    }

    /// Marker the visualizer drops to request an out-of-cycle refresh after
    /// an interactive membership change (it never writes groups.json itself).
    public static func consumeRefreshRequest(claudeDir: String) -> Bool {
        let syncDir = (SyncService.syncedDbPath(claudeDir: claudeDir) as NSString)
            .deletingLastPathComponent
        let marker = syncDir + "/groups-refresh-requested"
        guard FileManager.default.fileExists(atPath: marker) else { return false }
        try? FileManager.default.removeItem(atPath: marker)
        return true
    }
}
