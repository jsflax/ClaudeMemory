import Lattice
import Foundation

/// A directed edge between two memories, forming a knowledge graph.
///
/// Edges reference memories by their globally unique `globalId` (UUID),
/// enabling cross-database edges in multi-DB sync scenarios.
@Model
public final class Edge {
    @LatticeEnum
    public enum Relation: String, Codable, Sendable {
        /// General association between memories.
        case relatesTo = "relates_to"
        /// Memories that conflict with each other.
        case contradicts
        /// Newer knowledge replacing older knowledge.
        case supersedes
        /// Memory was derived/inferred from another.
        case derivedFrom = "derived_from"
        /// Memory is a sub-component of a larger concept.
        case partOf = "part_of"
        /// Memory was consolidated into a summary.
        case summarizedBy = "summarized_by"
    }

    /// The globalId (UUID) of the source memory.
    @Indexed()
    public var sourceGlobalId: UUID

    /// The globalId (UUID) of the target memory.
    @Indexed()
    public var targetGlobalId: UUID

    /// The relationship type.
    public var relation: Relation = .relatesTo

    /// When this edge was created.
    public var createdAt: Date

    public init(
        sourceGlobalId: UUID,
        targetGlobalId: UUID,
        relation: Relation,
        createdAt: Date = Date()
    ) {
        self.sourceGlobalId = sourceGlobalId
        self.targetGlobalId = targetGlobalId
        self.relation = relation
        self.createdAt = createdAt
    }
}
