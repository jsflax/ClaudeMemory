import SwiftUI
import Lattice
import Combine
import EngramKit

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
        didSet { simulation3D.center = worldCenter }
    }

    // Per-galaxy node filter (for data partitioning — prevents duplication across galaxies)
    // Local galaxy: { !syncedProjects.contains($0.project) || $0.isPrivate }
    // Synced/team galaxies: nil (show everything in that DB)
    // @Sendable to allow use from background threads in loadData.
    var nodeFilter: (@Sendable (Memory) -> Bool)?

    // Lattice observers
    var nodeObserver: AnyCancellable?
    var edgeObserver: AnyCancellable?

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
}
