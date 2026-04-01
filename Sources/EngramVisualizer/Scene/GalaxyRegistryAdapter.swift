import Foundation
import simd
import AppKit
import EngramRealityKit

/// Bridges GalaxyRegistry data to SceneDataProvider for EngramRealityKit.
///
/// Reads from GalaxyRegistry's merged data each frame, converting NodeData/EdgeData
/// to RKNodeSnapshot/RKEdgeSnapshot without depending on EngramKit types in the library.
@MainActor
final class GalaxyRegistryAdapter: SceneDataProvider {
    weak var registry: GalaxyRegistry?

    // Cached snapshots — rebuilt on topology change
    private var cachedNodes: [RKNodeSnapshot] = []
    private var cachedEdges: [RKEdgeSnapshot] = []
    private var cachedHubs: Set<UUID> = []
    private var cachedColorMap: [String: SIMD3<Float>] = [:]
    private var lastTopologyVersion: UInt64 = 0
    private var lastColorMapVersion: UInt64 = 0

    // Visual state — elapsed times tracked per-frame
    private var glowStartTimes: [UUID: Date] = [:]
    private var newNodeStartTimes: [UUID: Date] = [:]

    init(registry: GalaxyRegistry) {
        self.registry = registry
    }

    // MARK: - SceneDataProvider

    var nodes: [RKNodeSnapshot] { cachedNodes }
    var edges: [RKEdgeSnapshot] { cachedEdges }
    var hubs: Set<UUID> { cachedHubs }
    var projectColorMap: [String: SIMD3<Float>] { cachedColorMap }

    var positions: [UUID: SIMD3<Float>] {
        registry?.mergedPositions ?? [:]
    }

    var glowingNodes: [UUID: Float] {
        var result: [UUID: Float] = [:]
        let now = Date()
        for (id, start) in glowStartTimes {
            result[id] = Float(now.timeIntervalSince(start))
        }
        return result
    }

    var newNodeGlows: [UUID: Float] {
        var result: [UUID: Float] = [:]
        let now = Date()
        for (id, start) in newNodeStartTimes {
            result[id] = Float(now.timeIntervalSince(start))
        }
        return result
    }

    var dyingNodes: Set<UUID> {
        guard let registry else { return [] }
        return Set(registry.mergedDyingNodes.keys)
    }

    var selectedNode: UUID? {
        get { _selectedNode }
        set { _selectedNode = newValue }
    }
    private var _selectedNode: UUID?

    var expandedHubs: Set<UUID> {
        // Hub expansion is managed by the input handler
        []
    }

    var searchMatchIds: Set<UUID> {
        registry?.mergedSearchMatchIds ?? []
    }

    var isSearchActive: Bool {
        registry?.mergedIsSearchActive ?? false
    }

    var projectCentroids: [String: SIMD3<Float>] {
        guard let registry else { return [:] }
        let positions = registry.mergedPositions
        var sums: [String: (sum: SIMD3<Float>, count: Int)] = [:]
        for node in registry.mergedNodes {
            guard let pos = positions[node.id] else { continue }
            let entry = sums[node.project] ?? (.zero, 0)
            sums[node.project] = (entry.sum + pos, entry.count + 1)
        }
        return sums.mapValues { $0.sum / Float($0.count) }
    }

    var topologyVersion: UInt64 {
        registry?.mergedTopologyVersion ?? 0
    }

    func tick(dt: Float) {
        guard let registry else { return }

        // Drain pending updates from all galaxies
        let drainConfig = DrainConfig(
            hiddenProjects: [],
            hiddenRelations: [],
            timeFilter: nil,
            is3D: true,
            soundEnabled: false,
            notificationsEnabled: false
        )
        for galaxy in registry.galaxies.values {
            galaxy.drainPendingUpdate(config: drainConfig)
        }

        // Tick unified simulation
        registry.unifiedSimulation.tick()

        // Merge render data
        registry.mergeRenderData()

        // Rebuild snapshots on topology change
        let currentTopology = registry.mergedTopologyVersion
        if currentTopology != lastTopologyVersion {
            lastTopologyVersion = currentTopology
            rebuildSnapshots()
            print("[adapter] topology v\(currentTopology): \(cachedNodes.count) nodes, \(cachedEdges.count) edges, mergedEdges=\(registry.mergedEdges.count), positions=\(registry.mergedPositions.count)")
        }

        // Rebuild color map if needed
        let currentColorVersion = registry.mergedColorMapVersion
        if currentColorVersion != lastColorMapVersion {
            lastColorMapVersion = currentColorVersion
            rebuildColorMap()
        }

        // Sync glow state from registry
        syncGlowState()
    }

    // MARK: - Snapshot Rebuilding

    private func rebuildSnapshots() {
        guard let registry else { return }

        let mergedHubs = registry.mergedHubs
        cachedNodes = registry.mergedNodes.map { node in
            RKNodeSnapshot(
                id: node.id,
                project: node.project,
                topic: node.topic,
                label: node.label,
                content: node.content,
                importance: node.importance,
                isHub: mergedHubs.contains(node.id),
                createdAt: node.createdAt,
                lastAccessedAt: node.lastAccessedAt
            )
        }

        cachedEdges = registry.mergedEdges.map { edge in
            RKEdgeSnapshot(
                id: edge.id,
                sourceId: edge.sourceId,
                targetId: edge.targetId,
                relation: edge.relation
            )
        }

        cachedHubs = mergedHubs
    }

    private func rebuildColorMap() {
        guard let registry else { return }
        var colorMap: [String: SIMD3<Float>] = [:]
        for (project, color) in registry.mergedColorMap {
            let nsColor = NSColor(color).usingColorSpace(.sRGB) ?? NSColor(color)
            var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
            nsColor.getRed(&r, green: &g, blue: &b, alpha: &a)
            colorMap[project] = SIMD3<Float>(Float(r), Float(g), Float(b))
        }
        cachedColorMap = colorMap
    }

    private func syncGlowState() {
        guard let registry else { return }

        // Recall glows
        let currentGlows = registry.mergedGlowingNodes
        // Add new glows
        for (id, date) in currentGlows where glowStartTimes[id] == nil {
            glowStartTimes[id] = date
        }
        // Remove expired glows (not in registry anymore)
        for id in glowStartTimes.keys where currentGlows[id] == nil {
            glowStartTimes.removeValue(forKey: id)
        }

        // New node glows
        let currentNew = registry.mergedNewNodeGlows
        for (id, date) in currentNew where newNodeStartTimes[id] == nil {
            newNodeStartTimes[id] = date
        }
        for id in newNodeStartTimes.keys where currentNew[id] == nil {
            newNodeStartTimes.removeValue(forKey: id)
        }
    }
}
