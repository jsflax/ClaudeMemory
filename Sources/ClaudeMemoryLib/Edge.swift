import Lattice
import Foundation

/// A directed edge between two memories, forming a knowledge graph.
///
/// Edges represent typed relationships between memories:
/// - `relates_to`: General association
/// - `contradicts`: Memories that conflict with each other
/// - `supersedes`: Newer knowledge replacing older knowledge
/// - `derived_from`: Memory was derived/inferred from another
/// - `part_of`: Memory is a sub-component of a larger concept
/// - `summarized_by`: Memory was consolidated into a summary
@Model
public final class Edge {
    /// The primary key of the source memory.
    var sourceId: Int64

    /// The primary key of the target memory.
    var targetId: Int64

    /// The relationship type (e.g., "relates_to", "contradicts", "supersedes").
    var relation: String

    /// When this edge was created.
    var createdAt: Date

    init(
        sourceId: Int64,
        targetId: Int64,
        relation: String,
        createdAt: Date = Date()
    ) {
        self.sourceId = sourceId
        self.targetId = targetId
        self.relation = relation
        self.createdAt = createdAt
    }
}
