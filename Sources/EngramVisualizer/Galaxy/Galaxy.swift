import SwiftUI
import Lattice
import Combine
import EngramKit
import os

/// Encapsulates one complete graph data pipeline: Lattice + RenderStore + Simulation + EmbeddingProjection.
/// Each Galaxy renders as a separate cluster in the 3D world.
@MainActor
final class Galaxy: Identifiable {
    let id: String                        // e.g. "personal", "synced", "team-a-bob"
    let displayName: String
    let lattice: Lattice
    let hierarchyLevel: Int               // 0 = individual, 1 = team hive, 2 = org
    let parentGalaxyId: String?           // nil for root-level

    // Per-galaxy pipeline
    let renderStore = GraphRenderStore()
    let simulation3D = ForceSimulation3D()
    let embeddingProjection = EmbeddingProjection()

    // World-space center (set by GalaxyRegistry.computeWorldLayout)
    var worldCenter: SIMD3<Float> = .zero {
        didSet {
            simulation3D.center = worldCenter
            // Wake the sim when center changes so nodes re-converge toward
            // the new center. Without this, a settled galaxy ignores center
            // shifts (e.g. when a second galaxy is added and layout recomputes).
            if oldValue != worldCenter && isLoaded {
                simulation3D.wake()
            }
        }
    }

    // Per-galaxy node filter (for data partitioning — prevents duplication across galaxies)
    // Local galaxy: { !syncedProjects.contains($0.project) || $0.isPrivate }
    // Synced/team galaxies: nil (show everything in that DB)
    // @Sendable to allow use from background threads in loadData.
    var nodeFilter: (@Sendable (Memory) -> Bool)?

    // Per-galaxy mascot fleet (nil until Metal device is available)
    var mascotFleet: MascotFleet?

    var isLoaded = false
    var isInitialLoad = true

    init(id: String, displayName: String, lattice: Lattice,
         hierarchyLevel: Int = 0, parentGalaxyId: String? = nil) {
        self.id = id
        self.displayName = displayName
        self.lattice = lattice
        self.hierarchyLevel = hierarchyLevel
        self.parentGalaxyId = parentGalaxyId
    }
    
    actor GalaxyObserver {
        var nodeObserver: AnyCancellable?
        var edgeObserver: AnyCancellable?
        
        private let latticeRef: LatticeThreadSafeReference
        private let bgRef: LatticeThreadSafeReference
        private let galaxy: Galaxy
        
        init(galaxy: Galaxy, latticeRef: LatticeThreadSafeReference) {
            self.latticeRef = latticeRef
            self.bgRef = latticeRef
            self.galaxy = galaxy
        }
        
        func observe(hiddenProjects: Set<String>,
                     hiddenRelations: Set<String>,
                     timeFilter: Date?,
                     is3D: Bool,
                     soundEnabled: Bool,
                     notificationsEnabled: Bool) {
            guard let lattice = latticeRef.resolve() else { fatalError() }
            
            edgeObserver = lattice.objects(MemoryEdge.self).observe { change in
                guard let bg = self.bgRef.resolve() else { fatalError("This should not happen") }
                switch change {
                case .insert(let pk):
                    guard let edge = bg.object(MemoryEdge.self, primaryKey: pk),
                          let gid = edge.__globalId else { return }
                    let data = EdgeData(id: gid, sourceId: edge.sourceGlobalId,
                                        targetId: edge.targetGlobalId, relation: edge.relation.rawValue)
                    Task { @MainActor in
                        self.galaxy.renderStore.edgePkToGlobalId[pk] = gid
                        GalaxyDataLoader.handleEdgeInsert(data, galaxy: self.galaxy, is3D: is3D, hiddenRelations: hiddenRelations)
                    }
                case .update(let pk):
                    guard let edge = bg.object(MemoryEdge.self, primaryKey: pk),
                          let gid = edge.__globalId else { return }
                    let data = EdgeData(id: gid, sourceId: edge.sourceGlobalId,
                                        targetId: edge.targetGlobalId, relation: edge.relation.rawValue)
                    Task { @MainActor in GalaxyDataLoader.handleEdgeUpdate(data, galaxy: self.galaxy) }
                case .delete(let pk):
                    Task { @MainActor in
                        // Resolve pk → globalId since the object is already deleted
                        guard let gid = self.galaxy.renderStore.edgePkToGlobalId.removeValue(forKey: pk) else { return }
                        GalaxyDataLoader.handleEdgeDelete(gid, galaxy: self.galaxy)
                    }
                }
            }
            
            nodeObserver = lattice.objects(Memory.self).observe { change in
                guard let bg = self.bgRef.resolve() else { fatalError("This should not happen") }
                switch change {
                case .insert(let pk):
                    guard let memory = bg.object(Memory.self, primaryKey: pk),
                          let gid = memory.__globalId else { return }
                    let node = NodeData(
                        id: gid, project: memory.project, topic: memory.topic,
                        label: extractLabel(content: memory.content, topic: memory.topic),
                        content: memory.content,
                        createdAt: memory.createdAt, lastAccessedAt: memory.lastAccessedAt,
                        importance: memory.importance)
                    Task { @MainActor in
                        GalaxyDataLoader.handleNodeInsert(pk: pk, node: node, galaxy: self.galaxy,
                                         is3D: is3D, hiddenProjects: hiddenProjects,
                                         hiddenRelations: hiddenRelations, timeFilter: timeFilter,
                                         soundEnabled: soundEnabled, notificationsEnabled: notificationsEnabled)
                    }
                case .update(let pk):
                    guard let memory = bg.object(Memory.self, primaryKey: pk),
                          let gid = memory.__globalId else { return }
                    let node = NodeData(
                        id: gid, project: memory.project, topic: memory.topic,
                        label: extractLabel(content: memory.content, topic: memory.topic),
                        content: memory.content,
                        createdAt: memory.createdAt, lastAccessedAt: memory.lastAccessedAt,
                        importance: memory.importance)
                    Task { @MainActor in
                        GalaxyDataLoader.handleNodeUpdate(node: node, galaxy: self.galaxy)
                    }
                case .delete(let pk):
                    Task { @MainActor in
                        GalaxyDataLoader.handleNodeDelete(pk, galaxy: self.galaxy, soundEnabled: soundEnabled, notificationsEnabled: notificationsEnabled)
                    }
                }
            }
        }
    }
    
    private var observer: GalaxyObserver?
    
    func setUpObserve(hiddenProjects: Set<String>,
                      hiddenRelations: Set<String>,
                      timeFilter: Date?,
                      is3D: Bool,
                      soundEnabled: Bool,
                      notificationsEnabled: Bool) {
        self.observer = GalaxyObserver(galaxy: self, latticeRef: self.lattice.sendableReference)
        Task {
            await self.observer?.observe(hiddenProjects: hiddenProjects,
                                         hiddenRelations: hiddenRelations,
                                         timeFilter: timeFilter,
                                         is3D: is3D,
                                         soundEnabled: soundEnabled,
                                         notificationsEnabled: notificationsEnabled)
        }
    }
}
