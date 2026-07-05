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

    // Per-frame caches, refreshed once in tick(). The batch systems read
    // these properties several times per frame; computed accessors rebuilt
    // dicts/arrays on every access (positionArray alone was 4×O(n) per
    // frame — the production analog of the preview's V0 hot path).
    private var framePositionArray: [SIMD3<Float>] = []
    private var frameGlowing: [UUID: Float] = [:]
    private var frameNewGlows: [UUID: Float] = [:]
    private var frameCentroids: [String: SIMD3<Float>] = [:]
    private var simIndexByNodeIndex: [Int] = []
    private var simIndexTopologyVersion: UInt64 = .max

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

    /// Flat positions parallel to `nodes`, refreshed once per tick from the
    /// unified simulation's flat arrays (no UUID hashing on access).
    var positionArray: [SIMD3<Float>] { framePositionArray }

    var glowingNodes: [UUID: Float] { frameGlowing }

    var newNodeGlows: [UUID: Float] { frameNewGlows }

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

    var projectCentroids: [String: SIMD3<Float>] { frameCentroids }

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

        refreshFrameCaches()
    }

    /// Once-per-frame snapshot of everything the batch systems poll.
    private func refreshFrameCaches() {
        guard let registry else { return }
        let sim = registry.unifiedSimulation

        // positionArray via flat sim arrays + a topology-cached index map.
        let px = sim.posX, py = sim.posY, pz = sim.posZ
        let n = cachedNodes.count
        if simIndexByNodeIndex.count != n || simIndexTopologyVersion != lastTopologyVersion {
            var idToSim = [UUID: Int](minimumCapacity: sim.nodeIds.count)
            for (si, id) in sim.nodeIds.enumerated() { idToSim[id] = si }
            simIndexByNodeIndex = cachedNodes.map { idToSim[$0.id] ?? -1 }
            simIndexTopologyVersion = lastTopologyVersion
        }
        if framePositionArray.count != n {
            framePositionArray = [SIMD3<Float>](repeating: .zero, count: n)
        }
        simIndexByNodeIndex.withUnsafeBufferPointer { simIdx in
            for i in 0..<n {
                let si = simIdx[i]
                if si >= 0 && si < px.count {
                    framePositionArray[i] = SIMD3<Float>(px[si], py[si], pz[si])
                }
            }
        }

        // Glow elapsed times — one Date() call, one dict build per frame.
        let now = Date()
        frameGlowing.removeAll(keepingCapacity: true)
        for (id, start) in glowStartTimes {
            frameGlowing[id] = Float(now.timeIntervalSince(start))
        }
        frameNewGlows.removeAll(keepingCapacity: true)
        for (id, start) in newNodeStartTimes {
            frameNewGlows[id] = Float(now.timeIntervalSince(start))
        }

        // Project centroids — full scan throttled to every 30 frames (plus
        // topology changes): positions drift during settling, so anchors
        // refresh continuously but not per frame.
        frameCounter &+= 1
        if frameCentroids.isEmpty || lastTopologyVersion != centroidsTopologyVersion
            || frameCounter % 30 == 0 {
            let positions = registry.mergedPositions
            var sums: [String: (sum: SIMD3<Float>, count: Int)] = [:]
            for node in cachedNodes {
                guard let pos = positions[node.id] else { continue }
                let entry = sums[node.project] ?? (.zero, 0)
                sums[node.project] = (entry.sum + pos, entry.count + 1)
            }
            frameCentroids = sums.mapValues { $0.sum / Float($0.count) }
            centroidsTopologyVersion = lastTopologyVersion
        }
    }
    private var centroidsTopologyVersion: UInt64 = .max
    private var frameCounter: UInt64 = 0

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
