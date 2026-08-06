import EngramMemoryCore
import Foundation

/// Which embedder the harness should wire into a service under test.
public enum ContractEmbedding: Sendable {
    /// `DeterministicEmbedder` (or a backend equivalent with the same
    /// determinism guarantees).
    case deterministic
    /// `UnavailableEmbedder`: embed returns nil — the TEI-outage /
    /// missing-model path. Recall must degrade to full-text.
    case unavailable
}

/// What a backend must provide for the contract to run against it.
///
/// The Lattice harness builds `MemoryTools` over a temp sqlite file; the
/// Postgres harness (increment 5) builds `PgMemoryService` over a scratch
/// schema. Every check creates fresh, isolated stores through this factory —
/// checks never share state.
public protocol ContractHarness: Sendable {
    /// A fresh, EMPTY, isolated store operating as `principal`.
    /// `fencing` turns on foreign-content fencing for the read path (the
    /// server-side advise posture).
    func makeService(principal: Principal,
                     embedding: ContractEmbedding,
                     fencing: Bool) async throws -> any MemoryService

    /// A second service over the SAME store as `service`, operating as
    /// `principal` (typically a different author) and/or a different
    /// embedding mode. Return nil when the backend cannot share one store
    /// between two service instances in-process — the dependent checks
    /// (fencing, degraded recall over existing rows) are skipped, not failed.
    func makePeer(of service: any MemoryService,
                  principal: Principal,
                  embedding: ContractEmbedding,
                  fencing: Bool) async throws -> (any MemoryService)?
}

/// One broken invariant. `check` names the invariant; `detail` says what was
/// observed. The wrapping test target records each violation as its own issue.
public struct ContractViolation: Sendable, CustomStringConvertible {
    public let check: String
    public let detail: String

    public init(check: String, detail: String) {
        self.check = check
        self.detail = detail
    }

    public var description: String { "[\(check)] \(detail)" }
}

/// Fixed fixture principals — stable across runs so backend fixtures
/// (schemas, directories) can key off them.
public enum ContractPrincipals {
    public static let userA = Principal(
        id: UUID(uuidString: "AAAAAAAA-0000-0000-0000-00000000000A")!,
        kind: .user, displayName: "contract-user-a",
        groupIds: [], scopes: ["full"])
    public static let userB = Principal(
        id: UUID(uuidString: "BBBBBBBB-0000-0000-0000-00000000000B")!,
        kind: .user, displayName: "contract-user-b",
        groupIds: [], scopes: ["full"])
    public static let agent = Principal(
        id: UUID(uuidString: "CCCCCCCC-0000-0000-0000-00000000000C")!,
        kind: .agent, displayName: "contract-agent",
        groupIds: [], scopes: ["agent"])
}
