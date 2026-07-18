import Foundation
import Observation

/// REST client + cached state for the Groups feature. Mirrors
/// AccountService's bearer/URLSession/errorMessage pattern.
///
/// Panel routing state (`selectedGroupId`, `showCreateGroup`) lives here so
/// the sidebar (which opens panels) and GraphView (which renders them as
/// ZStack overlays) share it without binding plumbing.
///
/// This service never writes `~/.claude/sync/groups.json` — the daemon is
/// its sole writer. After membership changes it drops a marker file asking
/// the daemon to re-fetch on its next poll.
@Observable
@MainActor
final class GroupService {
    private let account: AccountService

    init(account: AccountService) {
        self.account = account
    }

    // MARK: - State

    /// Visible groups, cached across refresh failures (degraded state shows
    /// stale data + retry rather than an empty section).
    private(set) var groups: [GroupSummary] = []
    /// Pending invites addressed to the signed-in user's email.
    private(set) var incomingInvites: [IncomingInvite] = []
    private(set) var membersByGroup: [UUID: [MemberInfo]] = [:]
    private(set) var outboundInvitesByGroup: [UUID: [InviteInfo]] = [:]
    private(set) var projectsByGroup: [UUID: [GroupProjectInfo]] = [:]
    private(set) var isLoading = false
    private(set) var hasLoadedOnce = false
    /// Set when the last refresh failed; cached `groups` remain displayed.
    private(set) var lastRefreshError: String?
    var errorMessage: String?

    // Panel routing (GraphView overlays read these).
    var selectedGroupId: UUID?
    var showCreateGroup = false
    /// Pre-selected parent when creating a sub-group from a detail panel.
    var createGroupParentId: UUID?

    // MARK: - DTOs (mirror engram-server GroupController)

    enum GroupRole: String, CaseIterable, Comparable {
        case member, admin, owner
        private var rank: Int {
            switch self {
            case .member: 0
            case .admin: 1
            case .owner: 2
            }
        }
        static func < (lhs: GroupRole, rhs: GroupRole) -> Bool { lhs.rank < rhs.rank }
    }

    struct GroupSummary: Decodable, Identifiable, Hashable {
        let id: UUID
        let name: String
        let parentId: UUID?
        let myRole: String?
        let root: Bool
        let memberCount: Int
    }

    struct MemberInfo: Decodable, Identifiable, Hashable {
        let userId: UUID
        let displayName: String
        let email: String?
        let profilePictureUrl: String?
        let role: String
        let direct: Bool
        let lastSyncAt: Date?
        var id: UUID { userId }
    }

    struct InviteInfo: Decodable, Identifiable, Hashable {
        let id: UUID
        let groupId: UUID
        let email: String
        let role: String
        let status: String
        let expiresAt: Date
        let url: String
    }

    struct IncomingInvite: Decodable, Identifiable, Hashable {
        let id: UUID
        let groupId: UUID
        let groupName: String
        let inviterName: String
        let role: String
        let expiresAt: Date
        let token: String
    }

    /// Server `GroupProject` rows arrive as Fluent-encoded JSON (snake_case
    /// field keys); only what the UI needs is decoded.
    struct GroupProjectInfo: Decodable, Identifiable, Hashable {
        let id: UUID
        let name: String
        enum CodingKeys: String, CodingKey {
            case id, name
        }
    }

    struct AcceptResponse: Decodable {
        let groupId: UUID
        let groupName: String
        let role: String
    }

    // MARK: - Derived accessors

    func group(_ id: UUID) -> GroupSummary? {
        groups.first { $0.id == id }
    }

    func myRole(in group: GroupSummary) -> GroupRole? {
        group.myRole.flatMap(GroupRole.init(rawValue:))
    }

    func canAdmin(_ group: GroupSummary) -> Bool {
        (myRole(in: group) ?? .member) >= .admin
    }

    func children(of id: UUID) -> [GroupSummary] {
        groups.filter { $0.parentId == id }.sorted { $0.name < $1.name }
    }

    /// Groups ordered as a forest: roots by name, children DFS-indented.
    /// A child whose ancestors aren't visible (downward-admin scenarios)
    /// surfaces as its own root.
    func orderedGroups() -> [(group: GroupSummary, depth: Int)] {
        let visibleIds = Set(groups.map(\.id))
        let roots = groups
            .filter { $0.parentId == nil || !visibleIds.contains($0.parentId!) }
            .sorted { $0.name < $1.name }
        var result: [(GroupSummary, Int)] = []
        func visit(_ group: GroupSummary, depth: Int) {
            result.append((group, depth))
            guard depth < 10 else { return }
            for child in children(of: group.id) {
                visit(child, depth: depth + 1)
            }
        }
        for root in roots {
            visit(root, depth: 0)
        }
        return result
    }

    /// Sign-out teardown: drop every cache (member emails, invite URLs
    /// carrying live tokens, registries) and close any open panels — cached
    /// group data must not survive into a signed-out session or a different
    /// account on the same machine.
    func reset() {
        groups = []
        incomingInvites = []
        membersByGroup = [:]
        outboundInvitesByGroup = [:]
        projectsByGroup = [:]
        lastRefreshError = nil
        errorMessage = nil
        hasLoadedOnce = false
        selectedGroupId = nil
        showCreateGroup = false
        createGroupParentId = nil
    }

    // MARK: - Refresh

    func refresh() async {
        guard account.isSignedIn else {
            groups = []
            incomingInvites = []
            return
        }
        isLoading = true
        defer {
            isLoading = false
            hasLoadedOnce = true
        }
        do {
            let groupData = try await send("GET", "/groups")
            groups = try Self.decoder.decode([GroupSummary].self, from: groupData)
            lastRefreshError = nil
        } catch {
            lastRefreshError = Self.describe(error)
        }
        do {
            let inviteData = try await send("GET", "/invites")
            incomingInvites = try Self.decoder.decode([IncomingInvite].self, from: inviteData)
        } catch {
            // Older servers lack GET /invites; the section just stays empty.
        }
    }

    /// Members + outbound invites + project registry for the detail panel.
    /// Outbound invites are admin-only server-side — a 403 for a plain
    /// member is expected, not an error.
    func loadGroupDetail(_ id: UUID) async {
        async let membersTask = try? send("GET", "/groups/\(id.uuidString)/members")
        async let invitesTask = try? send("GET", "/groups/\(id.uuidString)/invites")
        async let projectsTask = try? send("GET", "/groups/\(id.uuidString)/projects")
        if let data = await membersTask,
           let members = try? Self.decoder.decode([MemberInfo].self, from: data) {
            membersByGroup[id] = members
        }
        if let data = await invitesTask,
           let invites = try? Self.decoder.decode([InviteInfo].self, from: data) {
            outboundInvitesByGroup[id] = invites
        }
        if let data = await projectsTask,
           let projects = try? Self.decoder.decode([GroupProjectInfo].self, from: data) {
            projectsByGroup[id] = projects
        }
    }

    // MARK: - Group CRUD

    @discardableResult
    func createGroup(name: String, parentId: UUID?) async -> GroupSummary? {
        await mutate {
            let body = try Self.encoder.encode(["name": name.trimmingCharacters(in: .whitespacesAndNewlines),
                                                "parentId": parentId?.uuidString])
            let data = try await self.send("POST", "/groups", body: body)
            let created = try Self.decoder.decode(GroupSummary.self, from: data)
            await self.refresh()
            return created
        }
    }

    @discardableResult
    func renameGroup(_ id: UUID, to name: String) async -> Bool {
        await mutate {
            let body = try Self.encoder.encode(["name": name])
            _ = try await self.send("PATCH", "/groups/\(id.uuidString)", body: body)
            await self.refresh()
            return true
        } ?? false
    }

    @discardableResult
    func deleteGroup(_ id: UUID) async -> Bool {
        await mutate {
            _ = try await self.send("DELETE", "/groups/\(id.uuidString)")
            if self.selectedGroupId == id { self.selectedGroupId = nil }
            await self.refresh()
            self.requestDaemonDirectoryRefresh()
            return true
        } ?? false
    }

    // MARK: - Members

    @discardableResult
    func leaveGroup(_ id: UUID) async -> Bool {
        guard let me = account.userProfile?.id else {
            errorMessage = "Profile not loaded yet — try again in a moment."
            return false
        }
        return await removeMember(groupId: id, userId: me)
    }

    @discardableResult
    func removeMember(groupId: UUID, userId: UUID) async -> Bool {
        await mutate {
            _ = try await self.send("DELETE", "/groups/\(groupId.uuidString)/members/\(userId.uuidString)")
            if userId == self.account.userProfile?.id, self.selectedGroupId == groupId {
                self.selectedGroupId = nil
            }
            await self.refresh()
            await self.loadGroupDetail(groupId)
            self.requestDaemonDirectoryRefresh()
            return true
        } ?? false
    }

    @discardableResult
    func setRole(groupId: UUID, userId: UUID, role: GroupRole) async -> Bool {
        await mutate {
            let body = try Self.encoder.encode(["role": role.rawValue])
            _ = try await self.send("PATCH", "/groups/\(groupId.uuidString)/members/\(userId.uuidString)", body: body)
            await self.loadGroupDetail(groupId)
            self.requestDaemonDirectoryRefresh()
            return true
        } ?? false
    }

    // MARK: - Invites

    @discardableResult
    func createInvite(groupId: UUID, email: String, role: GroupRole = .member) async -> InviteInfo? {
        await mutate {
            let body = try Self.encoder.encode(["email": email.trimmingCharacters(in: .whitespaces),
                                                "role": role.rawValue])
            let data = try await self.send("POST", "/groups/\(groupId.uuidString)/invites", body: body)
            let invite = try Self.decoder.decode(InviteInfo.self, from: data)
            await self.loadGroupDetail(groupId)
            return invite
        }
    }

    @discardableResult
    func revokeInvite(groupId: UUID, inviteId: UUID) async -> Bool {
        await mutate {
            _ = try await self.send("DELETE", "/groups/\(groupId.uuidString)/invites/\(inviteId.uuidString)")
            await self.loadGroupDetail(groupId)
            return true
        } ?? false
    }

    @discardableResult
    func acceptInvite(_ invite: IncomingInvite) async -> Bool {
        await mutate {
            let body = try Self.encoder.encode(["token": invite.token])
            _ = try await self.send("POST", "/invites/accept", body: body)
            self.incomingInvites.removeAll { $0.id == invite.id }
            await self.refresh()
            self.requestDaemonDirectoryRefresh()
            return true
        } ?? false
    }

    @discardableResult
    func declineInvite(_ invite: IncomingInvite) async -> Bool {
        await mutate {
            let body = try Self.encoder.encode(["token": invite.token])
            _ = try await self.send("POST", "/invites/decline", body: body)
            self.incomingInvites.removeAll { $0.id == invite.id }
            return true
        } ?? false
    }

    // MARK: - Project registry (decision 13)

    /// Create-or-match by name — the server is idempotent, so this is safe
    /// as the exposure flow's default mapping step.
    @discardableResult
    func createGroupProject(groupId: UUID, name: String) async -> GroupProjectInfo? {
        await mutate {
            let body = try Self.encoder.encode(["name": name])
            let data = try await self.send("POST", "/groups/\(groupId.uuidString)/projects", body: body)
            let project = try Self.decoder.decode(GroupProjectInfo.self, from: data)
            await self.loadGroupDetail(groupId)
            return project
        }
    }

    @discardableResult
    func renameGroupProject(groupId: UUID, projectId: UUID, name: String) async -> Bool {
        await mutate {
            let body = try Self.encoder.encode(["name": name])
            _ = try await self.send("PATCH", "/groups/\(groupId.uuidString)/projects/\(projectId.uuidString)", body: body)
            await self.loadGroupDetail(groupId)
            return true
        } ?? false
    }

    @discardableResult
    func deleteGroupProject(groupId: UUID, projectId: UUID) async -> Bool {
        await mutate {
            _ = try await self.send("DELETE", "/groups/\(groupId.uuidString)/projects/\(projectId.uuidString)")
            await self.loadGroupDetail(groupId)
            return true
        } ?? false
    }

    // MARK: - Daemon coordination

    /// The daemon sole-writes groups.json; this marker asks it to re-fetch
    /// ahead of its 5-minute poll. Watched from the multi-spoke daemon on;
    /// harmless before that (the poll converges anyway).
    func requestDaemonDirectoryRefresh() {
        let dir = URL(fileURLWithPath: NSHomeDirectory() + "/.claude/sync")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try? Data().write(to: dir.appendingPathComponent("groups-refresh-requested"))
    }

    // MARK: - HTTP plumbing

    /// Vapor's default JSON coder is ISO8601 dates (without fractional
    /// seconds) — the client must match or every Date field throws.
    static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }()
    static let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.dateEncodingStrategy = .iso8601
        return e
    }()

    struct RequestError: Error {
        let status: Int
        let reason: String
    }

    private struct VaporError: Decodable {
        let reason: String?
    }

    private func send(_ method: String, _ path: String, body: Data? = nil) async throws -> Data {
        guard let token = account.token else {
            throw RequestError(status: 401, reason: "Not signed in")
        }
        guard let url = URL(string: "\(account.endpoint)\(path)") else {
            throw RequestError(status: 0, reason: "Bad endpoint")
        }
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        if let body {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = body
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw RequestError(status: 0, reason: "No response")
        }
        guard (200..<300).contains(http.statusCode) else {
            let reason = (try? Self.decoder.decode(VaporError.self, from: data))?.reason
                ?? HTTPURLResponse.localizedString(forStatusCode: http.statusCode)
            throw RequestError(status: http.statusCode, reason: reason)
        }
        return data
    }

    /// Wraps a mutation: clears the error, runs, and converts thrown errors
    /// into `errorMessage` for the sidebar/panels to display.
    private func mutate<T>(_ operation: () async throws -> T) async -> T? {
        errorMessage = nil
        isLoading = true
        defer { isLoading = false }
        do {
            return try await operation()
        } catch {
            errorMessage = Self.describe(error)
            return nil
        }
    }

    private static func describe(_ error: any Error) -> String {
        if let requestError = error as? RequestError {
            return requestError.reason
        }
        return error.localizedDescription
    }
}
