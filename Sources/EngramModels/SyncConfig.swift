import Lattice
import Foundation

/// Per-project sync configuration. Stored in synced.db (replicates across devices).
///
/// Each row maps a project name to a sync policy. The special project name `_default`
/// sets the fallback policy for projects without an explicit override.
///
/// When a project is set to `.sync`, non-private memories route to `synced.db`
/// (synced across devices). Private memories always stay in `memory.db` regardless
/// of this setting.
///
/// The `exposedTeams` field (Phase 2) controls which teams see this project's
/// non-private memories via contribution DB dual-write. Empty set = no team exposure.
@Model
public final class SyncConfig {
    /// The project name this config applies to.
    /// Use `"_default"` for the fallback policy.
    @Unique()
    public var project: String

    @LatticeEnum
    public enum Policy: String, Codable, Sendable {
        /// Memories stay in memory.db (device-local).
        case local
        /// Non-private memories go to personal.db (synced across devices).
        case sync
    }

    /// Whether this project's memories should sync.
    public var policy: Policy = .local

    /// Team IDs this project is exposed to (Phase 2).
    /// Non-empty requires `policy == .sync`. Empty set = no team exposure.
    public var exposedTeams: Set<String> = []

    /// When this config was last updated.
    public var updatedAt: Date

    public init(
        project: String,
        policy: Policy,
        exposedTeams: Set<String> = [],
        updatedAt: Date = Date()
    ) {
        self.project = project
        self.policy = policy
        self.exposedTeams = exposedTeams
        self.updatedAt = updatedAt
    }
}
