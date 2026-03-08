import Combine
import EngramKit
import Lattice
import SwiftUI
import simd

/// Manages N galaxies and produces merged data for the single Metal renderer.
/// All properties are plain (not @Observable) — read directly by renderTick each frame.
@MainActor
final class GalaxyRegistry {
    private(set) var galaxies: [String: Galaxy] = [:]  // id -> Galaxy

    // SyncConfig observation — drives node migration between personal ↔ synced galaxies
    private var syncConfigObserver: AnyCancellable?
    weak var syncManager: SyncManager?

    // Hierarchy spacing
    let levelSpacing: Float = 3000    // Y between hierarchy levels
    let siblingSpacing: Float = 4000  // X between same-level galaxies

    // Merged data for renderer (recomputed when any galaxy's topology changes)
    private(set) var mergedNodes: [NodeData] = []
    private(set) var mergedEdges: [EdgeData] = []
    private(set) var mergedHubs: Set<UUID> = []
    private(set) var mergedColorMap: [String: Color] = [:]
    private(set) var mergedNodeById: [UUID: NodeData] = [:]

    // Node -> galaxy routing (for selection, detail panel, search)
    private(set) var nodeToGalaxy: [UUID: String] = [:]
    var focusedGalaxyId: String?

    // Cross-galaxy edge filtering (pushed from VisualizerConfig)
    var hiddenRelations: Set<String> = []

    // Topology tracking — each galaxy's topologyVersion is summed to detect changes
    private var lastMergedTopologySum: UInt64 = 0

    // MARK: - Galaxy Management

    func register(_ galaxy: Galaxy) {
        galaxies[galaxy.id] = galaxy
        computeWorldLayout()
    }

    func remove(_ galaxyId: String) {
        galaxies.removeValue(forKey: galaxyId)
        computeWorldLayout()
    }

    /// The focused galaxy (for detail panel, search, etc.)
    var focusedGalaxy: Galaxy? {
        if let id = focusedGalaxyId { return galaxies[id] }
        return galaxies.values.first
    }

    /// Route a node ID to its owning galaxy.
    func galaxyForNode(_ nodeId: UUID) -> Galaxy? {
        if let galaxyId = nodeToGalaxy[nodeId] {
            return galaxies[galaxyId]
        }
        return nil
    }

    // MARK: - World Layout

    /// Assigns worldCenter to each galaxy based on hierarchy level.
    /// Level 0 galaxies spread on X axis at Y=0.
    /// Level 1 centered above their children at Y=levelSpacing.
    /// Level 2 at Y=2*levelSpacing.
    func computeWorldLayout() {
        // Group galaxies by hierarchy level
        var byLevel: [Int: [Galaxy]] = [:]
        for galaxy in galaxies.values {
            byLevel[galaxy.hierarchyLevel, default: []].append(galaxy)
        }

        // Layout each level
        for (level, levelGalaxies) in byLevel {
            let sorted = levelGalaxies.sorted(by: { $0.id < $1.id })
            let count = sorted.count
            let totalWidth = Float(count - 1) * siblingSpacing
            let startX = -totalWidth / 2.0

            for (i, galaxy) in sorted.enumerated() {
                galaxy.worldCenter = SIMD3<Float>(
                    startX + Float(i) * siblingSpacing,
                    Float(level) * levelSpacing,
                    0
                )
            }
        }
    }

    // MARK: - Merge

    #if ENGRAM_INSTRUMENTATION
    private var mergeTimingFile: UnsafeMutablePointer<FILE>? = nil
    private var migrationTimingFile: UnsafeMutablePointer<FILE>? = nil

    private func migrationLog(_ line: String) {
        if migrationTimingFile == nil {
            migrationTimingFile = fopen("/tmp/galaxy-migration.csv", "w")
            if let f = migrationTimingFile {
                fputs("timestamp,event,project,direction,from_galaxy,to_galaxy,node_count,edge_count,position_count,elapsed_ms\n", f)
            }
        }
        if let f = migrationTimingFile {
            fputs(line + "\n", f)
            fflush(f)
        }
    }
    #endif

    /// Rebuild merged data from all galaxies. Called each frame by renderTick.
    /// Only does real work when topology has changed.
    func mergeRenderData() {
        // Check if any galaxy's topology changed
        var topologySum: UInt64 = 0
        for galaxy in galaxies.values {
            topologySum &+= galaxy.renderStore.topologyVersion
        }
        guard topologySum != lastMergedTopologySum else { return }
        lastMergedTopologySum = topologySum

        #if ENGRAM_INSTRUMENTATION
        let mergeStart = CFAbsoluteTimeGetCurrent()
        #endif

        // Rebuild merged arrays
        var nodes: [NodeData] = []
        var hubs: Set<UUID> = []
        var colorMap: [String: Color] = [:]
        var nodeById: [UUID: NodeData] = [:]
        var nodeToGalaxy: [UUID: String] = [:]

        for galaxy in galaxies.values {
            let store = galaxy.renderStore
            nodes.append(contentsOf: store.nodes)
            hubs.formUnion(store.hubs)
            colorMap.merge(store.colorMap) { existing, _ in existing }
            nodeById.merge(store.nodeById) { existing, _ in existing }
            for node in store.nodes {
                nodeToGalaxy[node.id] = galaxy.id
            }
        }

        // Cross-galaxy edge merge
        let edges: [EdgeData]
        if galaxies.count <= 1, let only = galaxies.values.first {
            // Single galaxy fast path — use per-galaxy filtered edges directly
            edges = only.renderStore.edges
        } else {
            // Multi-galaxy: iterate ALL galaxies' allEdges, dedup by globalId,
            // filter against merged visible node set + hiddenRelations.
            let mergedVisibleIds = Set(nodes.map(\.id))
            var seen = Set<UUID>()
            var crossEdges: [EdgeData] = []
            for galaxy in galaxies.values {
                for (gid, edge) in galaxy.renderStore.allEdges {
                    guard seen.insert(gid).inserted else { continue }
                    guard mergedVisibleIds.contains(edge.sourceId),
                          mergedVisibleIds.contains(edge.targetId) else { continue }
                    guard !hiddenRelations.contains(edge.relation) else { continue }
                    crossEdges.append(edge)
                }
            }
            edges = crossEdges
        }

        mergedNodes = nodes
        mergedEdges = edges
        mergedHubs = hubs
        mergedColorMap = colorMap
        mergedNodeById = nodeById
        self.nodeToGalaxy = nodeToGalaxy

        #if ENGRAM_INSTRUMENTATION
        let mergeMs = (CFAbsoluteTimeGetCurrent() - mergeStart) * 1000.0
        if mergeMs > 0.5 {
            if mergeTimingFile == nil {
                mergeTimingFile = fopen("/tmp/merge-timing.csv", "w")
                if let f = mergeTimingFile {
                    fputs("timestamp,galaxy_count,node_count,edge_count,merge_ms\n", f)
                }
            }
            if let f = mergeTimingFile {
                let ts = String(format: "%.3f", CFAbsoluteTimeGetCurrent())
                let line = "\(ts),\(galaxies.count),\(nodes.count),\(edges.count),\(String(format: "%.2f", mergeMs))\n"
                fputs(line, f)
                fflush(f)
            }
        }
        #endif
    }

    // MARK: - Merged Accessors (convenience for renderer)

    /// Cached merged positions — rebuilt each frame via updateMergedPositions().
    /// Stored var avoids per-frame dictionary allocation (reuses capacity via removeAll).
    private(set) var cachedMergedPositions: [UUID: SIMD3<Float>] = [:]

    /// Rebuild merged positions from all galaxy simulations. Called each frame by renderTick.
    func updateMergedPositions() {
        #if ENGRAM_INSTRUMENTATION
        let posStart = CFAbsoluteTimeGetCurrent()
        #endif

        if galaxies.count == 1, let only = galaxies.values.first {
            cachedMergedPositions = only.simulation3D.positions
        } else {
            cachedMergedPositions.removeAll(keepingCapacity: true)
            for galaxy in galaxies.values {
                cachedMergedPositions.merge(galaxy.simulation3D.positions) { _, new in new }
            }
        }

        #if ENGRAM_INSTRUMENTATION
        let posMergeMs = (CFAbsoluteTimeGetCurrent() - posStart) * 1000.0
        if posMergeMs > 0.5 {
            if mergeTimingFile == nil {
                mergeTimingFile = fopen("/tmp/merge-timing.csv", "w")
                if let f = mergeTimingFile {
                    fputs("timestamp,galaxy_count,node_count,edge_count,merge_ms\n", f)
                }
            }
            if let f = mergeTimingFile {
                let ts = String(format: "%.3f", CFAbsoluteTimeGetCurrent())
                let line = "\(ts),\(galaxies.count),\(cachedMergedPositions.count),0,\(String(format: "%.2f", posMergeMs))\n"
                fputs(line, f)
                fflush(f)
            }
        }
        #endif
    }

    /// Merged positions from all galaxy simulations — already in world space.
    var mergedPositions: [UUID: SIMD3<Float>] {
        cachedMergedPositions
    }

    /// Merged allEdges from all galaxies.
    var mergedAllEdges: [UUID: EdgeData] {
        var result: [UUID: EdgeData] = [:]
        for galaxy in galaxies.values {
            result.merge(galaxy.renderStore.allEdges) { _, new in new }
        }
        return result
    }

    /// Merged allNodes from all galaxies.
    var mergedAllNodes: [UUID: NodeData] {
        var result: [UUID: NodeData] = [:]
        for galaxy in galaxies.values {
            result.merge(galaxy.renderStore.allNodes) { _, new in new }
        }
        return result
    }

    /// Merged edgesByNode from all galaxies.
    var mergedEdgesByNode: [UUID: [EdgeData]] {
        var result: [UUID: [EdgeData]] = [:]
        for galaxy in galaxies.values {
            for (id, edges) in galaxy.renderStore.edgesByNode {
                result[id, default: []].append(contentsOf: edges)
            }
        }
        return result
    }

    /// Merged visual effects from all galaxies.
    /// Single-galaxy fast path avoids dictionary rebuilds (most common case).
    var mergedGlowingNodes: [UUID: Date] {
        if galaxies.count == 1, let only = galaxies.values.first {
            return only.renderStore.glowingNodes
        }
        var result: [UUID: Date] = [:]
        for galaxy in galaxies.values {
            result.merge(galaxy.renderStore.glowingNodes) { _, new in new }
        }
        return result
    }

    var mergedNewNodeGlows: [UUID: Date] {
        if galaxies.count == 1, let only = galaxies.values.first {
            return only.renderStore.newNodeGlows
        }
        var result: [UUID: Date] = [:]
        for galaxy in galaxies.values {
            result.merge(galaxy.renderStore.newNodeGlows) { _, new in new }
        }
        return result
    }

    var mergedDyingNodes: [UUID: DyingNode] {
        if galaxies.count == 1, let only = galaxies.values.first {
            return only.renderStore.dyingNodes
        }
        var result: [UUID: DyingNode] = [:]
        for galaxy in galaxies.values {
            result.merge(galaxy.renderStore.dyingNodes) { _, new in new }
        }
        return result
    }

    var mergedTopicGroups: [TopicGroupInfo] {
        if galaxies.count == 1, let only = galaxies.values.first {
            return only.renderStore.topicGroups
        }
        var result: [TopicGroupInfo] = []
        for galaxy in galaxies.values {
            result.append(contentsOf: galaxy.renderStore.topicGroups)
        }
        return result
    }

    var mergedClusterGroups: [[UUID]] {
        if galaxies.count == 1, let only = galaxies.values.first {
            return only.renderStore.clusterGroups
        }
        var result: [[UUID]] = []
        for galaxy in galaxies.values {
            result.append(contentsOf: galaxy.renderStore.clusterGroups)
        }
        return result
    }

    var mergedSearchMatchIds: Set<UUID> {
        if galaxies.count == 1, let only = galaxies.values.first {
            return only.renderStore.searchMatchIds
        }
        var result: Set<UUID> = []
        for galaxy in galaxies.values {
            result.formUnion(galaxy.renderStore.searchMatchIds)
        }
        return result
    }

    var mergedIsSearchActive: Bool {
        galaxies.values.contains { $0.renderStore.isSearchActive }
    }

    var mergedVisibleNodeIds: Set<UUID> {
        var result: Set<UUID> = []
        for galaxy in galaxies.values {
            result.formUnion(galaxy.renderStore.visibleNodeIds)
        }
        return result
    }

    var mergedRelationCounts: [(key: String, value: Int)] {
        var counts: [String: Int] = [:]
        for galaxy in galaxies.values {
            for (key, value) in galaxy.renderStore.relationCounts {
                counts[key, default: 0] += value
            }
        }
        return counts.sorted(by: { $0.key < $1.key })
    }

    /// Merged topology version — sum of all galaxy versions.
    var mergedTopologyVersion: UInt64 {
        var sum: UInt64 = 0
        for galaxy in galaxies.values {
            sum &+= galaxy.renderStore.topologyVersion
        }
        return sum
    }

    /// Merged colorMapVersion — sum of all galaxy versions.
    var mergedColorMapVersion: UInt64 {
        var sum: UInt64 = 0
        for galaxy in galaxies.values {
            sum &+= galaxy.renderStore.colorMapVersion
        }
        return sum
    }

    /// Merged edgeCountByNode.
    var mergedEdgeCountByNode: [UUID: Int] {
        var result: [UUID: Int] = [:]
        for galaxy in galaxies.values {
            result.merge(galaxy.renderStore.edgeCountByNode) { a, b in a + b }
        }
        return result
    }

    /// Merged recentNodes (top 50 across all galaxies, deduplicated by UUID).
    var mergedRecentNodes: [NodeData] {
        var seen: Set<UUID> = []
        var all: [NodeData] = []
        for galaxy in galaxies.values {
            for node in galaxy.renderStore.recentNodes {
                if seen.insert(node.id).inserted {
                    all.append(node)
                }
            }
        }
        return Array(all.sorted(by: { $0.createdAt > $1.createdAt }).prefix(50))
    }

    /// Merged filteredEdgeIds.
    var mergedFilteredEdgeIds: Set<UUID> {
        var result: Set<UUID> = []
        for galaxy in galaxies.values {
            result.formUnion(galaxy.renderStore.filteredEdgeIds)
        }
        return result
    }

    /// Merged pkToGlobalId.
    var mergedPkToGlobalId: [Int64: UUID] {
        var result: [Int64: UUID] = [:]
        for galaxy in galaxies.values {
            result.merge(galaxy.renderStore.pkToGlobalId) { _, new in new }
        }
        return result
    }

    // MARK: - Inter-Galaxy Connections

    /// Returns pairs of world-space centers for galaxies connected by parent→child hierarchy.
    var interGalaxyConnections: [(from: SIMD3<Float>, to: SIMD3<Float>, label: String)] {
        var result: [(from: SIMD3<Float>, to: SIMD3<Float>, label: String)] = []
        for galaxy in galaxies.values {
            guard let parentId = galaxy.parentGalaxyId,
                  let parent = galaxies[parentId] else { continue }
            result.append((from: parent.worldCenter, to: galaxy.worldCenter, label: galaxy.displayName))
        }
        return result
    }

    // MARK: - SyncConfig Observation & Node Migration

    /// Observe SyncConfig changes on the personal galaxy's Lattice.
    /// When a project's policy flips, migrate nodes between personal ↔ synced galaxies
    /// and rebuild the personal galaxy's nodeFilter.
    func setupSyncConfigObserver(
        hiddenProjects: Set<String>,
        hiddenRelations: Set<String>,
        timeFilter: Date?,
        is3D: Bool,
        soundEnabled: Bool,
        notificationsEnabled: Bool
    ) {
        guard let personal = galaxies["personal"] else { return }
        let lattice = personal.lattice
        self.hiddenRelations = hiddenRelations

        syncConfigObserver = lattice.objects(SyncConfig.self).observe { [weak self] change in
            switch change {
            case .insert(let pk), .update(let pk):
                // Read config inside @MainActor Task — observer fires on bg thread,
                // and rapid toggles can cause the bg read to see stale values.
                let capturedPk = pk
                Task { @MainActor [weak self] in
                    guard let self,
                          let personal = self.galaxies["personal"],
                          let config = personal.lattice.object(SyncConfig.self, primaryKey: capturedPk) else { return }
                    let project = config.project
                    let policy = config.policy

                    #if ENGRAM_INSTRUMENTATION
                    let migStart = CFAbsoluteTimeGetCurrent()
                    #endif

                    if policy == .sync {
                        // Idempotent: only migrate if nodes are actually in personal
                        guard personal.renderStore.allNodes.values.contains(where: { $0.project == project }) else {
                            #if ENGRAM_INSTRUMENTATION
                            let ts = String(format: "%.3f", CFAbsoluteTimeGetCurrent())
                            self.migrationLog("\(ts),skip_idempotent,\(project),personal→synced,personal,synced,0,0,0,0.00")
                            #endif
                            return
                        }
                        self.ensureSyncedGalaxyExists(
                            hiddenProjects: hiddenProjects,
                            hiddenRelations: hiddenRelations,
                            timeFilter: timeFilter,
                            is3D: is3D,
                            soundEnabled: soundEnabled,
                            notificationsEnabled: notificationsEnabled
                        )
                        let srcPositions = self.captureWorldPositions(project: project, in: "personal")
                        let extracted = self.migrateProjectOut(project, from: "personal")
                        self.loadExtractedNodesIntoGalaxy(
                            "synced", nodes: extracted.nodes,
                            intraProjectEdges: extracted.intraEdges,
                            sourcePositions: srcPositions,
                            hiddenProjects: hiddenProjects,
                            hiddenRelations: hiddenRelations,
                            timeFilter: timeFilter, is3D: is3D
                        )

                        #if ENGRAM_INSTRUMENTATION
                        let migMs = (CFAbsoluteTimeGetCurrent() - migStart) * 1000.0
                        let ts = String(format: "%.3f", CFAbsoluteTimeGetCurrent())
                        self.migrationLog("\(ts),migrate,\(project),personal→synced,personal,synced,\(extracted.nodes.count),\(extracted.intraEdges.count),\(srcPositions.count),\(String(format: "%.2f", migMs))")
                        #endif
                    } else {
                        // Idempotent: only migrate if nodes are actually in synced
                        guard self.galaxies["synced"]?.renderStore.allNodes.values.contains(where: { $0.project == project }) == true else {
                            #if ENGRAM_INSTRUMENTATION
                            let ts = String(format: "%.3f", CFAbsoluteTimeGetCurrent())
                            self.migrationLog("\(ts),skip_idempotent,\(project),synced→personal,synced,personal,0,0,0,0.00")
                            #endif
                            return
                        }
                        let srcPositions = self.captureWorldPositions(project: project, in: "synced")
                        let extracted = self.migrateProjectOut(project, from: "synced")
                        self.loadExtractedNodesIntoGalaxy(
                            "personal", nodes: extracted.nodes,
                            intraProjectEdges: extracted.intraEdges,
                            sourcePositions: srcPositions,
                            hiddenProjects: hiddenProjects,
                            hiddenRelations: hiddenRelations,
                            timeFilter: timeFilter, is3D: is3D
                        )

                        #if ENGRAM_INSTRUMENTATION
                        let migMs = (CFAbsoluteTimeGetCurrent() - migStart) * 1000.0
                        let ts = String(format: "%.3f", CFAbsoluteTimeGetCurrent())
                        self.migrationLog("\(ts),migrate,\(project),synced→personal,synced,personal,\(extracted.nodes.count),\(extracted.intraEdges.count),\(srcPositions.count),\(String(format: "%.2f", migMs))")
                        #endif
                    }
                    self.rebuildPersonalNodeFilter()
                }
            case .delete:
                Task { @MainActor [weak self] in
                    self?.reconcileSyncState()
                    self?.rebuildPersonalNodeFilter()
                }
            }
        }
    }

    /// Create + load the synced galaxy if it doesn't already exist.
    private func ensureSyncedGalaxyExists(
        hiddenProjects: Set<String>,
        hiddenRelations: Set<String>,
        timeFilter: Date?,
        is3D: Bool,
        soundEnabled: Bool,
        notificationsEnabled: Bool
    ) {
        guard galaxies["synced"] == nil,
              let syncedLattice = syncManager?.syncedLattice else { return }
        let synced = Galaxy(id: "synced", displayName: "Synced",
                            lattice: syncedLattice, hierarchyLevel: 0)
        register(synced)
        GalaxyDataLoader.loadData(
            into: synced,
            hiddenProjects: hiddenProjects,
            hiddenRelations: hiddenRelations,
            timeFilter: timeFilter,
            is3D: is3D,
            soundEnabled: soundEnabled,
            notificationsEnabled: notificationsEnabled
        )
    }

    /// Extract a project's nodes from a galaxy's render store. Edges STAY in the
    /// source galaxy's `allEdges`/`edgesByNode` — they're resolved at merge time by
    /// `mergeRenderData()` which filters all galaxies' `allEdges` against the merged
    /// visible node set. This avoids expensive edge removal/insertion during migration.
    ///
    /// Returns extracted nodes + intra-project edges (both endpoints migrating) so the
    /// destination galaxy can wire them into its force simulation via `edgesByNode`.
    @discardableResult
    private func migrateProjectOut(_ project: String, from galaxyId: String) -> (nodes: [NodeData], intraEdges: [EdgeData]) {
        guard let galaxy = galaxies[galaxyId] else { return ([], []) }
        let store = galaxy.renderStore

        #if ENGRAM_INSTRUMENTATION
        let outStart = CFAbsoluteTimeGetCurrent()
        let preNodeCount = store.allNodes.count
        let preSimCount = galaxy.simulation3D.positions.count
        #endif

        // Collect nodes BEFORE removing
        let removedNodes = store.allNodes.values.filter { $0.project == project }
        let removedIds = Set(removedNodes.map(\.id))
        guard !removedIds.isEmpty else { return ([], []) }

        // Collect intra-project edges from store.edgesByNode (O(removed·degree))
        // instead of scanning allEdges (O(allEdges))
        var intraEdges: [EdgeData] = []
        var intraEdgeIds = Set<UUID>()
        for id in removedIds {
            for edge in store.edgesByNode[id] ?? [] {
                guard removedIds.contains(edge.sourceId) && removedIds.contains(edge.targetId) else { continue }
                if intraEdgeIds.insert(edge.id).inserted {
                    intraEdges.append(edge)
                }
            }
        }

        // Remove NODES from all node data structures
        for id in removedIds {
            store.allNodes.removeValue(forKey: id)
            store.nodeById.removeValue(forKey: id)
            store.visibleNodeIds.remove(id)
        }
        store.nodes.removeAll { $0.project == project }
        store.pkToGlobalId = store.pkToGlobalId.filter { !removedIds.contains($0.value) }

        // Edges STAY in allEdges/edgesByNode for cross-galaxy rendering.
        // Filter existing store.edges (O(store.edges)) — NOT allEdges (O(allEdges)).
        store.edges.removeAll { removedIds.contains($0.sourceId) || removedIds.contains($0.targetId) }
        store.filteredEdgeIds = Set(store.edges.map(\.id))

        store.clusterGroups = store.clusterGroups.compactMap { cluster in
            let filtered = cluster.filter { !removedIds.contains($0) }
            return filtered.count >= 2 ? filtered : nil
        }

        // Remove from simulation so ghost nodes don't produce stale positions
        galaxy.simulation3D.removeNodes(removedIds)

        // Lightweight derived data rebuild — skip full recomputeDerivedData.
        // visibleNodeIds and nodeById are already maintained inline above.
        // Only rebuild relation counts, topic groups, and edge counts.
        recomputeStatsOnly(for: galaxy)
        store.bumpTopology()

        #if ENGRAM_INSTRUMENTATION
        let outMs = (CFAbsoluteTimeGetCurrent() - outStart) * 1000.0
        let ts = String(format: "%.3f", CFAbsoluteTimeGetCurrent())
        migrationLog("\(ts),migrate_out,\(project),from_\(galaxyId),\(galaxyId),,\(removedIds.count),\(intraEdges.count),0,\(String(format: "%.2f", outMs))")
        migrationLog("\(ts),migrate_out_counts,\(project),from_\(galaxyId),pre_nodes=\(preNodeCount) post_nodes=\(store.allNodes.count) pre_sim=\(preSimCount) post_sim=\(galaxy.simulation3D.positions.count),,,,")
        #endif

        return (Array(removedNodes), Array(intraEdges))
    }

    /// Capture world-space positions for all nodes of a project in a given galaxy.
    /// Must be called BEFORE migrateProjectOut removes them.
    private func captureWorldPositions(project: String, in galaxyId: String) -> [UUID: SIMD3<Float>] {
        guard let galaxy = galaxies[galaxyId] else { return [:] }
        let positions = galaxy.simulation3D.positions
        var result: [UUID: SIMD3<Float>] = [:]
        for node in galaxy.renderStore.allNodes.values where node.project == project {
            if let pos = positions[node.id] {
                result[node.id] = pos
            }
        }
        return result
    }

    /// Insert previously-extracted nodes into a destination galaxy. Edges are NOT
    /// fully migrated — only intra-project edges (both endpoints migrating) are copied
    /// into the destination's `allEdges`/`edgesByNode` so `insertNodeBatch` can wire
    /// them into the force simulation. Cross-galaxy edges are resolved at merge time
    /// by `mergeRenderData()`.
    private func loadExtractedNodesIntoGalaxy(
        _ galaxyId: String,
        nodes: [NodeData],
        intraProjectEdges: [EdgeData],
        sourcePositions: [UUID: SIMD3<Float>] = [:],
        hiddenProjects: Set<String>,
        hiddenRelations: Set<String>,
        timeFilter: Date?,
        is3D: Bool
    ) {
        guard let galaxy = galaxies[galaxyId] else { return }
        let store = galaxy.renderStore
        guard !nodes.isEmpty else { return }

        #if ENGRAM_INSTRUMENTATION
        let loadStart = CFAbsoluteTimeGetCurrent()
        let preNodeCount = store.allNodes.count
        let preSimCount = galaxy.simulation3D.positions.count
        #endif

        // Copy intra-project edges into destination's allEdges/edgesByNode
        // so insertNodeBatch can wire them into the force simulation
        for edge in intraProjectEdges {
            guard store.allEdges[edge.id] == nil else { continue }
            store.allEdges[edge.id] = edge
            store.edgesByNode[edge.sourceId, default: []].append(edge)
            store.edgesByNode[edge.targetId, default: []].append(edge)
        }

        // insertNodeBatch calls addNode which randomizes positions
        GalaxyDataLoader.insertNodeBatch(
            nodes, into: galaxy,
            hiddenProjects: hiddenProjects,
            hiddenRelations: hiddenRelations,
            timeFilter: timeFilter,
            is3D: is3D
        )

        // Override with source positions so nodes start where they were and
        // the force simulation pulls them toward the destination galaxy center
        for (id, pos) in sourcePositions {
            galaxy.simulation3D.setPosition(id, to: pos)
        }

        // Always wake the sim after adding nodes — without this, the sim stays
        // in its default state with decayed alpha and near-zero smoothedAttenuation,
        // producing negligible force application even though isSettled gets cleared
        // by hasPendingTopologyChanges on the first tick.
        galaxy.simulation3D.wake()

        // Mark initial load complete so renderTick ticks this galaxy's sim.
        // loadData (async) may not have finished yet for newly created galaxies,
        // but migration has populated the sim — it must tick to animate.
        galaxy.isInitialLoad = false

        // insertNodeBatch already maintains visibleNodeIds, nodeById, colorMap, hubs, edges.
        // Only rebuild stats (relation counts, topic groups, edge counts).
        recomputeStatsOnly(for: galaxy)
        store.bumpTopology()

        #if ENGRAM_INSTRUMENTATION
        let loadMs = (CFAbsoluteTimeGetCurrent() - loadStart) * 1000.0
        let ts = String(format: "%.3f", CFAbsoluteTimeGetCurrent())
        migrationLog("\(ts),load_into,\(nodes.first?.project ?? "?"),into_\(galaxyId),,\(galaxyId),\(nodes.count),\(intraProjectEdges.count),\(sourcePositions.count),\(String(format: "%.2f", loadMs))")
        migrationLog("\(ts),load_into_counts,\(nodes.first?.project ?? "?"),into_\(galaxyId),pre_nodes=\(preNodeCount) post_nodes=\(store.allNodes.count) pre_sim=\(preSimCount) post_sim=\(galaxy.simulation3D.positions.count),,,,")
        #endif
    }

    /// Full reconciliation — query Lattice for current synced projects, diff against
    /// what's actually in the personal galaxy's visible set, and fix discrepancies.
    /// Single pass over nodes/edges, single topology bump.
    private func reconcileSyncState() {
        guard let personal = galaxies["personal"] else { return }
        let lattice = personal.lattice

        var syncedProjects = Set<String>()
        for config in lattice.objects(SyncConfig.self).where({ $0.policy == .sync }) {
            syncedProjects.insert(config.project)
        }

        let store = personal.renderStore

        // Collect unique projects that need to move out
        let projectsToRemove = Set(store.nodes.filter { syncedProjects.contains($0.project) }.map(\.project))

        // Collect projects that should be local but aren't visible
        let visibleProjects = Set(store.nodes.map(\.project))
        let allProjects = Set(store.allNodes.values.map(\.project))
        let projectsToAdd = allProjects.subtracting(syncedProjects).subtracting(visibleProjects)

        guard !projectsToRemove.isEmpty || !projectsToAdd.isEmpty else { return }

        // Batch remove — single pass (nodes only; edges stay in allEdges for cross-galaxy rendering)
        if !projectsToRemove.isEmpty {
            let removedIds = Set(store.nodes.filter { projectsToRemove.contains($0.project) }.map(\.id))
            store.nodes.removeAll { projectsToRemove.contains($0.project) }
            for id in removedIds {
                store.nodeById.removeValue(forKey: id)
                store.allNodes.removeValue(forKey: id)
                store.visibleNodeIds.remove(id)
            }
            // Rebuild per-galaxy edges from allEdges against remaining visible nodes
            let remainingVisible = store.visibleNodeIds
            store.edges = store.allEdges.values.filter {
                remainingVisible.contains($0.sourceId) && remainingVisible.contains($0.targetId)
            }
            store.filteredEdgeIds = Set(store.edges.map(\.id))
            store.clusterGroups = store.clusterGroups.compactMap { cluster in
                let filtered = cluster.filter { !removedIds.contains($0) }
                return filtered.count >= 2 ? filtered : nil
            }
        }

        // Batch add — single pass
        for project in projectsToAdd {
            let addedNodes = store.allNodes.values.filter {
                $0.project == project && !store.visibleNodeIds.contains($0.id)
            }
            for node in addedNodes {
                store.nodes.append(node)
                store.nodeById[node.id] = node
                store.visibleNodeIds.insert(node.id)
                personal.simulation3D.addNode(node.id, project: node.project, topic: node.topic)
            }
        }

        if !projectsToAdd.isEmpty {
            // Wire edges for all newly visible nodes in one pass
            let allVisibleIds = store.visibleNodeIds
            let newEdges = store.allEdges.values.filter { edge in
                allVisibleIds.contains(edge.sourceId) && allVisibleIds.contains(edge.targetId) &&
                !store.filteredEdgeIds.contains(edge.id)
            }
            store.edges.append(contentsOf: newEdges)
            store.filteredEdgeIds.formUnion(newEdges.map(\.id))
            for edge in newEdges {
                personal.simulation3D.addEdge(from: edge.sourceId, to: edge.targetId)
            }

            var hubs = Set<UUID>()
            for edge in store.allEdges.values where edge.relation == "part_of" {
                hubs.insert(edge.targetId)
            }
            store.hubs = hubs
        }

        // Single recompute + topology bump
        GalaxyDataLoader.recomputeDerivedData(for: personal)
        store.bumpTopology()
    }

    /// Lightweight stats-only rebuild — used during migration where visibleNodeIds and
    /// nodeById are already maintained inline. Skips the O(nodes) set/dict rebuilds that
    /// `recomputeDerivedData` does, only rebuilding relation counts, topic groups, and
    /// edge counts from the current visible state.
    private func recomputeStatsOnly(for galaxy: Galaxy) {
        let store = galaxy.renderStore
        let nodeIds = store.visibleNodeIds

        // Relation counts
        var counts: [String: Int] = [:]
        for edge in store.edges {
            counts[edge.relation, default: 0] += 1
        }
        store.relationCounts = counts.sorted(by: { $0.key < $1.key })

        // Topic groups
        var groups: [String: (topic: String, project: String, ids: [UUID])] = [:]
        for node in store.nodes {
            guard node.topic != "general", node.topic != "episode" else { continue }
            let key = "\(node.project)|\(node.topic)"
            var entry = groups[key] ?? (topic: node.topic, project: node.project, ids: [])
            entry.ids.append(node.id)
            groups[key] = entry
        }
        store.topicGroups = groups.values
            .filter { $0.ids.count >= 2 }
            .map { TopicGroupInfo(topic: $0.topic, project: $0.project, ids: $0.ids) }

        // Per-node edge counts (only edges with both endpoints visible)
        var edgeCounts: [UUID: Int] = [:]
        for edge in store.allEdges.values {
            guard nodeIds.contains(edge.sourceId), nodeIds.contains(edge.targetId) else { continue }
            edgeCounts[edge.sourceId, default: 0] += 1
            edgeCounts[edge.targetId, default: 0] += 1
        }
        store.edgeCountByNode = edgeCounts
    }

    /// Rebuild the personal galaxy's nodeFilter from current SyncConfig state in Lattice.
    private func rebuildPersonalNodeFilter() {
        guard let personal = galaxies["personal"] else { return }
        let lattice = personal.lattice

        var syncedProjects = Set<String>()
        for config in lattice.objects(SyncConfig.self).where({ $0.policy == .sync }) {
            syncedProjects.insert(config.project)
        }

        if syncedProjects.isEmpty {
            personal.nodeFilter = nil
        } else {
            let captured = syncedProjects
            personal.nodeFilter = { @Sendable memory in
                !captured.contains(memory.project) || memory.isPrivate
            }
        }
    }
}
