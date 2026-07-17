import Lattice
import Foundation

/// Maps one group member's local project name to the group's canonical
/// project name (decision 13: group-defined project registry).
///
/// Lives ONLY in group spoke/server DBs (schema `[Memory, Edge,
/// GroupProjectMap]`) — never in the personal hub or personal-sync set.
/// Written by each member's client into the spoke at exposure time; rows
/// travel spoke ↔ server ↔ other members' spokes.
///
/// Rows in a group DB keep their author's local `project` string. Readers
/// resolve `effectiveProject(row) = map[row.authorUserId, row.project] ??
/// row.project` — so unrelated same-named repos map to different group
/// projects, and the same repo under different folder names maps to one.
/// Consumers: recall project boosting, galaxy/nebula grouping, stats and
/// list_topics rollups, and the reconciliation agent's clustering scope.
@Model @Detached
public final class GroupProjectMap {
    /// The member whose local project name this row maps.
    @Indexed()
    public var memberUserId: UUID

    /// The member's local (cwd-derived) project name.
    public var localProject: String

    /// The group's canonical project name (from the group's registry).
    @Indexed()
    public var groupProject: String

    /// When this mapping was last updated.
    public var updatedAt: Date

    public init(
        memberUserId: UUID,
        localProject: String,
        groupProject: String,
        updatedAt: Date = Date()
    ) {
        self.memberUserId = memberUserId
        self.localProject = localProject
        self.groupProject = groupProject
        self.updatedAt = updatedAt
    }
}
