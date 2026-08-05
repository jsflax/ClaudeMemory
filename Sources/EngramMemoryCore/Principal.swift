import Foundation

/// The authenticated identity a `MemoryService` operates as.
///
/// Identity is CONSTRUCTOR state, never ambient: the CLI/hook processes build
/// one from the daemon-authored groups.json, the HTTP service builds one from
/// the authenticated token row per request. Service operations themselves
/// carry no identity parameters — a service instance IS a principal's view of
/// the memory graph.
public struct Principal: Sendable, Hashable {
    public enum Kind: String, Sendable, Codable {
        /// A human account (engramdb.io user).
        case user
        /// A service principal (canary agent). Agents authenticate with
        /// scoped tokens and are exempt from seat billing.
        case agent
    }

    /// The stable user/agent id — the value stamped into `authorUserId`.
    /// Nil only in the signed-out CLI case; writers stamp nil and the
    /// daemon-start backfill sweep repairs later (existing behavior).
    public let id: UUID?
    public let kind: Kind
    /// Display name for `[by:]` attribution of the principal's own writes.
    public let displayName: String?
    /// Direct group memberships (read scope = these + their ancestors,
    /// resolved by the storage conformance).
    public let groupIds: [UUID]
    /// Token scopes as issued by the auth layer (e.g. "full", "agent",
    /// "viz-readonly"). Opaque strings here — enforcement is the server's
    /// middleware; conformances may consult them for defense in depth.
    public let scopes: Set<String>

    public init(id: UUID?, kind: Kind, displayName: String? = nil,
                groupIds: [UUID] = [], scopes: Set<String> = []) {
        self.id = id
        self.kind = kind
        self.displayName = displayName
        self.groupIds = groupIds
        self.scopes = scopes
    }

    /// The signed-out / unknown-identity principal (CLI before first sync).
    public static let anonymous = Principal(id: nil, kind: .user)
}

/// Source of the current principal for environments where identity is
/// resolved out-of-band (CLI: groups.json; server: per-request token).
public protocol IdentityProviding: Sendable {
    func currentPrincipal() -> Principal
}

/// Fixed identity — the server case (one per authenticated request) and the
/// test case.
public struct StaticIdentityProvider: IdentityProviding {
    public let principal: Principal
    public init(_ principal: Principal) { self.principal = principal }
    public func currentPrincipal() -> Principal { principal }
}
