import Foundation
import simd

/// Core abstraction for all data the RealityKit scene needs per frame.
///
/// EngramVisualizer implements this via GalaxyRegistryAdapter;
/// EngramPreview implements with mock data.
@MainActor
public protocol SceneDataProvider: AnyObject {
    /// All nodes to potentially render (LOD system filters these).
    var nodes: [RKNodeSnapshot] { get }
    /// All edges to potentially render.
    var edges: [RKEdgeSnapshot] { get }
    /// Hub node IDs.
    var hubs: Set<UUID> { get }
    /// Project → color mapping.
    var projectColorMap: [String: SIMD3<Float>] { get }
    /// Current positions for all nodes.
    var positions: [UUID: SIMD3<Float>] { get }
    /// Nodes with active recall glow — value is elapsed seconds since glow start.
    var glowingNodes: [UUID: Float] { get }
    /// Nodes with arrival glow — value is elapsed seconds since creation.
    var newNodeGlows: [UUID: Float] { get }
    /// Nodes being deleted (dying animation).
    var dyingNodes: Set<UUID> { get }
    /// Currently selected node.
    var selectedNode: UUID? { get set }
    /// Hubs that have been expanded.
    var expandedHubs: Set<UUID> { get }
    /// Node IDs matching current search.
    var searchMatchIds: Set<UUID> { get }
    /// Whether search mode is active (dims non-matching nodes).
    var isSearchActive: Bool { get }
    /// Centroid position per project cluster.
    var projectCentroids: [String: SIMD3<Float>] { get }
    /// Increments on topology change (node/edge add/remove).
    var topologyVersion: UInt64 { get }

    /// Positions as a flat array indexed parallel to `nodes`. Avoids UUID dict lookups.
    var positionArray: [SIMD3<Float>] { get }

    /// Per-frame update — drain pending changes, tick simulation.
    func tick(dt: Float)
}

extension SceneDataProvider {
    public var positionArray: [SIMD3<Float>] {
        nodes.map { positions[$0.id] ?? .zero }
    }
}
