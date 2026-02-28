import Lattice
import Foundation

/// Migrate memories and their edges between databases (e.g., local → synced or synced → local).
///
/// Used when a project's SyncConfig changes policy:
/// - `.local` → `.sync`: migrate non-private memories from localLattice to syncedLattice
/// - `.sync` → `.local`: migrate memories from syncedLattice to localLattice
///
/// Private memories (`isPrivate == true`) always stay in localLattice regardless of policy.
public struct SyncMigration {

    /// Result of a migration operation.
    public struct Result {
        public let memoriesMigrated: Int
        public let edgesMigrated: Int
    }

    /// Migrate all non-private memories for the given projects from `source` to `destination`.
    ///
    /// - Snapshots memories for each project from `source`
    /// - Adds them to `destination` (globalId preserved via Lattice's built-in handling)
    /// - Migrates edges where both source AND target globalIds are in the migrated set
    /// - Deletes migrated originals from `source`
    /// - Private memories are skipped (stay in source)
    ///
    /// - Parameters:
    ///   - projects: Project names to migrate.
    ///   - source: The Lattice to move memories FROM.
    ///   - destination: The Lattice to move memories TO.
    /// - Returns: Count of migrated memories and edges.
    public static func migrateProjects(
        _ projects: [String],
        from source: Lattice,
        to destination: Lattice
    ) -> Result {
        var totalMemories = 0
        var totalEdges = 0
        var migratedGlobalIds = Set<UUID>()

        for project in projects {
            // Snapshot non-private memories for this project
            let memories = source.objects(Memory.self)
                .where { $0.project == project && $0.isPrivate == false }
                .snapshot()

            for memory in memories {
                guard let globalId = memory.__globalId else { continue }

                // Create a new Memory with the same content in the destination
                let migrated = Memory(
                    content: memory.content,
                    topic: memory.topic,
                    project: memory.project,
                    source: memory.source,
                    embedding: memory.embedding,
                    expiresAt: memory.expiresAt,
                    importance: memory.importance,
                    isPrivate: memory.isPrivate
                )
                destination.add(migrated, preservingGlobalId: globalId)
                migratedGlobalIds.insert(globalId)
                totalMemories += 1
            }
        }

        // Migrate edges where both endpoints are in the migrated set
        let allEdges = source.objects(Edge.self).snapshot()
        for edge in allEdges {
            guard migratedGlobalIds.contains(edge.sourceGlobalId),
                  migratedGlobalIds.contains(edge.targetGlobalId) else { continue }

            let migratedEdge = Edge(
                sourceGlobalId: edge.sourceGlobalId,
                targetGlobalId: edge.targetGlobalId,
                relation: edge.relation
            )
            destination.add(migratedEdge)
            totalEdges += 1
        }

        // Delete migrated originals from source
        for globalId in migratedGlobalIds {
            // Delete edges referencing this memory
            source.delete(Edge.self, where: { $0.sourceGlobalId == globalId || $0.targetGlobalId == globalId })
            // Delete the memory itself
            source.delete(Memory.self, where: { $0.__globalId == globalId })
        }

        return Result(memoriesMigrated: totalMemories, edgesMigrated: totalEdges)
    }
}
