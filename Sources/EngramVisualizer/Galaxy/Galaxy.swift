import SwiftUI
import EngramSceneKit
import Lattice
import EngramSceneKit
import Combine
import EngramKit
import os

private let flushLog = Logger(subsystem: "io.engram.app", category: "GalaxyLoader")
private let stallLog = OSLog(subsystem: "io.engram.app", category: "FrameStall")

/// Atomic update produced by the Galaxy actor, consumed by MainActor render loop.
/// Written via OSAllocatedUnfairLock from the Galaxy actor (observer callbacks + loadData).
/// Drained once per frame in renderTick — ensures store + simulation are always consistent.
struct RenderUpdate: Sendable {
    // Incremental changes (from observers)
    var insertedNodes: [(pk: Int64, node: NodeData)] = []
    // pk rides along so a node entering the galaxy's partition via the
    // update path (unknown to the render store) can take the insert route.
    var updatedNodes: [(pk: Int64, node: NodeData)] = []
    var removedNodePks: [Int64] = []
    var insertedEdges: [(pk: Int64, edge: EdgeData)] = []
    var updatedEdges: [EdgeData] = []
    var removedEdgeGids: [UUID] = []

    // Bulk load (initial — replaces entire edge store, batches nodes)
    var bulkEdges: (allEdges: [UUID: EdgeData],
                    pkToGid: [Int64: UUID],
                    byNode: [UUID: [EdgeData]])?
    var bulkNodeBatches: [[(pk: Int64, node: NodeData)]] = []
    var clusterGroups: [[UUID]]?
    var finalize: Bool = false  // recomputeDerivedData + wake sim + mark loaded

    var isEmpty: Bool {
        insertedNodes.isEmpty && updatedNodes.isEmpty && removedNodePks.isEmpty &&
        insertedEdges.isEmpty && updatedEdges.isEmpty && removedEdgeGids.isEmpty &&
        bulkEdges == nil && bulkNodeBatches.isEmpty && clusterGroups == nil && !finalize
    }
}

/// Per-frame config snapshot consumed by the drain. Built once per frame by
/// MetalSceneManager from VisualizerConfig + layout mode, stored on GalaxyRegistry.
struct DrainConfig: Sendable {
    let hiddenProjects: Set<String>
    let hiddenRelations: Set<String>
    let timeFilter: Date?
    let is3D: Bool
    let soundEnabled: Bool
    let notificationsEnabled: Bool
}

/// Encapsulates one complete graph data pipeline: Lattice + RenderStore + Simulation + EmbeddingProjection.
/// Each Galaxy renders as a separate cluster in the 3D world.
/// Serializes the cluster pass across all galaxies. The body is deliberately
/// synchronous — no suspension points, so actor reentrancy cannot interleave
/// two galaxies' passes.
private actor ClusterPassGate {
    static let shared = ClusterPassGate()
    func run<T: Sendable>(_ body: @Sendable () -> T) -> T { body() }
}

actor Galaxy: Identifiable {
    let id: String                        // e.g. "personal", "synced", "team-a-bob"
    let displayName: String
    let latticeRef: LatticeThreadSafeReference
    let hierarchyLevel: Int               // 0 = individual, 1 = team hive, 2 = org
    let parentGalaxyId: String?           // nil for root-level

    // Per-galaxy pipeline
    @MainActor let renderStore = GraphRenderStore()
    @MainActor let embeddingProjection = EmbeddingProjection()

    /// Reference to the unified simulation owned by GalaxyRegistry.
    /// Set during register(). All node/edge ops go through this.
    @MainActor weak var simulation3D: ForceSimulation3D?


    /// Lock-protected update buffer — written by Galaxy actor, drained by MainActor.
    let pendingUpdate = OSAllocatedUnfairLock<RenderUpdate>(initialState: .init())

    /// Lock-protected pk→gid mirror for edges — maintained alongside pendingUpdate so
    /// the edge delete observer can resolve pk→gid without dispatching to MainActor.
    let edgePkToGidLock = OSAllocatedUnfairLock<[Int64: UUID]>(initialState: [:])

    // World-space center (set by GalaxyRegistry.computeWorldLayout)
    @MainActor var worldCenter: SIMD3<Float> = .zero

    // Per-galaxy node filter (for data partitioning — prevents duplication across galaxies)
    // Local galaxy: { !syncedProjects.contains($0.project) || $0.isPrivate }
    // Synced/team galaxies: nil (show everything in that DB)
    // Lock-protected (not actor state): the live observers apply it from
    // Lattice's background callback thread, outside this actor — without
    // observer-path filtering, every post-load memory appeared in BOTH the
    // personal and synced galaxies until restart.
    private let nodeFilterLock = OSAllocatedUnfairLock<(@Sendable (Memory) -> Bool)?>(initialState: nil)
    nonisolated var nodeFilter: (@Sendable (Memory) -> Bool)? {
        nodeFilterLock.withLock { $0 }
    }
    nonisolated func setNodeFilter(_ filter: (@Sendable (Memory) -> Bool)?) {
        nodeFilterLock.withLock { $0 = filter }
    }

    /// effectiveProject resolution (decision 13) for GROUP galaxies: members
    /// derive `project` from their own folder names, so the same shared
    /// project arrives under each author's local string. The resolver maps
    /// (authorUserId, localProject) → the group's canonical name, so
    /// clusters, labels, nebulae, and stats group by ONE project instead of
    /// one per member. Nil (personal/synced) is identity. Same lock idiom as
    /// nodeFilter — the observer path reads it off-actor.
    private let projectResolverLock =
        OSAllocatedUnfairLock<(@Sendable (_ author: UUID?, _ local: String) -> String)?>(initialState: nil)
    nonisolated var projectResolver: (@Sendable (UUID?, String) -> String)? {
        projectResolverLock.withLock { $0 }
    }
    nonisolated func setProjectResolver(_ resolver: (@Sendable (UUID?, String) -> String)?) {
        projectResolverLock.withLock { $0 = resolver }
    }

    /// Resolved display project for a memory in this galaxy.
    nonisolated func displayProject(_ memory: Memory) -> String {
        projectResolver?(memory.authorUserId, memory.project) ?? memory.project
    }
    
    // Per-galaxy mascot fleet (nil until Metal device is available)
    @MainActor var mascotFleet: MascotFleet?

    #if ENGRAM_INSTRUMENTATION
    @MainActor var glowLogFile: UnsafeMutablePointer<FILE>? = nil
    @MainActor var flushTimingFile: UnsafeMutablePointer<FILE>? = nil
    #endif

    @MainActor var isLoaded = false
    @MainActor var isInitialLoad = true

    /// Actor-isolated in-flight flag for loadData. The `isLoaded` guard alone
    /// is racy: it hops to MainActor (a suspension point), so two callers —
    /// onAppear's load and the daemon-connect re-load — can both read `false`
    /// and each run a full-DB scan. The scans serialize on this actor (no
    /// data race), but doubling multi-GB scans is exactly the page-cache
    /// pressure that triggered the sqlite OOM crash. Checked and set
    /// synchronously on the actor, before any await — no window.
    private var loadInFlight = false

    @MainActor init(id: String, displayName: String, lattice: LatticeThreadSafeReference,
         hierarchyLevel: Int = 0, parentGalaxyId: String? = nil) {
        self.id = id
        self.displayName = displayName
        self.latticeRef = lattice
        self.hierarchyLevel = hierarchyLevel
        self.parentGalaxyId = parentGalaxyId
    }
    
    var nodeObserver: AnyCancellable?
    var edgeObserver: AnyCancellable?

    /// Set up live Lattice observers. Callbacks push raw data into pendingUpdate —
    /// all filtering (hidden projects/relations, time) happens during drain via DrainConfig.
    func startObservers() {
        // Sign-out (or any teardown) can invalidate the lattice while the
        // initial load is still in flight — observers just don't start.
        // Crashing here took the app down when signing out mid-load.
        guard let lattice = latticeRef.resolve() else { return }

        edgeObserver = lattice.objects(MemoryEdge.self).observe { [pendingUpdate, edgePkToGidLock] change in
            // Lattice torn down (sign-out) — drop the change, observer dies with it.
            guard let bg = self.latticeRef.resolve() else { return }
            switch change {
            case .insert(let pk):
                guard let edge = bg.object(MemoryEdge.self, primaryKey: pk),
                      let gid = edge.globalId else { return }
                let data = EdgeData(id: gid, sourceId: edge.sourceGlobalId,
                                    targetId: edge.targetGlobalId, relation: edge.relation.rawValue)
                edgePkToGidLock.withLock { $0[pk] = gid }
                pendingUpdate.withLock { $0.insertedEdges.append((pk, data)) }
            case .update(let pk):
                guard let edge = bg.object(MemoryEdge.self, primaryKey: pk),
                      let gid = edge.globalId else { return }
                let data = EdgeData(id: gid, sourceId: edge.sourceGlobalId,
                                    targetId: edge.targetGlobalId, relation: edge.relation.rawValue)
                pendingUpdate.withLock { $0.updatedEdges.append(data) }
            case .delete(let pk):
                // Resolve pk→gid under lock, push to pendingUpdate for drain processing.
                // No MainActor dispatch needed — the lock mirror is maintained alongside
                // pendingUpdate on inserts and bulk loads.
                guard let gid = edgePkToGidLock.withLock({ $0.removeValue(forKey: pk) }) else { return }
                pendingUpdate.withLock { $0.removedEdgeGids.append(gid) }
            }
        }

        nodeObserver = lattice.objects(Memory.self).observe { [pendingUpdate] change in
            // Lattice torn down (sign-out) — drop the change, observer dies with it.
            guard let bg = self.latticeRef.resolve() else { return }
            switch change {
            case .insert(let pk):
                guard let memory = bg.object(Memory.self, primaryKey: pk),
                      let gid = memory.globalId else { return }
                // Same structural partition loadData applies — without it a
                // post-load memory lands in every galaxy whose DB has it.
                if let filter = self.nodeFilter, !filter(memory) { return }
                let node = NodeData(
                    id: gid, project: self.displayProject(memory), topic: memory.topic,
                    label: extractLabel(content: memory.content, topic: memory.topic),
                    content: memory.content,
                    createdAt: memory.createdAt, lastAccessedAt: memory.lastAccessedAt,
                    importance: memory.importance)
                pendingUpdate.withLock { $0.insertedNodes.append((pk, node)) }
            case .update(let pk):
                guard let memory = bg.object(Memory.self, primaryKey: pk),
                      let gid = memory.globalId else { return }
                if let filter = self.nodeFilter, !filter(memory) {
                    // Fell out of this galaxy's partition (e.g. project
                    // toggled to sync) — remove; no-op if never present.
                    pendingUpdate.withLock { $0.removedNodePks.append(pk) }
                    return
                }
                let node = NodeData(
                    id: gid, project: self.displayProject(memory), topic: memory.topic,
                    label: extractLabel(content: memory.content, topic: memory.topic),
                    content: memory.content,
                    createdAt: memory.createdAt, lastAccessedAt: memory.lastAccessedAt,
                    importance: memory.importance)
                pendingUpdate.withLock { $0.updatedNodes.append((pk, node)) }
            case .delete(let pk):
                pendingUpdate.withLock { $0.removedNodePks.append(pk) }
            }
        }
    }
    
    /// Load all data from a galaxy's Lattice into its renderStore + simulation.
    /// Reads ALL data — visual filtering (hiddenProjects, etc.) happens during
    /// drain via DrainConfig. Only `nodeFilter` (structural partition) is applied here.
    func loadData() async {
        // Guard against duplicate loads (onAppear + syncManager.didConnect can
        // both fire). loadInFlight is checked/set synchronously on the actor
        // — airtight; the isLoaded check alone suspends and lets both through.
        guard !loadInFlight else { return }
        loadInFlight = true
        defer { loadInFlight = false }
        guard await !isLoaded else { return }

        let batchSize = 50
        let ref = latticeRef
        let nodeFilter = nodeFilter

        guard let bgLattice = ref.resolve() else { return }

        // 1. Read all edges AND build dictionaries off main actor.
        var allEdges: [UUID: EdgeData] = [:]
        var edgePkToGlobalId: [Int64: UUID] = [:]
        var edgesByNode: [UUID: [EdgeData]] = [:]
        for e in bgLattice.objects(MemoryEdge.self) {
            guard let pk = e.primaryKey, let gid = e.globalId else { continue }
            let ed = EdgeData(id: gid, sourceId: e.sourceGlobalId,
                              targetId: e.targetGlobalId, relation: e.relation.rawValue)
            allEdges[gid] = ed
            edgePkToGlobalId[pk] = gid
            edgesByNode[ed.sourceId, default: []].append(ed)
            edgesByNode[ed.targetId, default: []].append(ed)
        }

        // Push bulk edges into the update buffer + populate pk→gid lock mirror
        let finalEdges = allEdges
        let finalPkToGid = edgePkToGlobalId
        let finalByNode = edgesByNode
        edgePkToGidLock.withLock { $0 = finalPkToGid }
        pendingUpdate.withLock {
            $0.bulkEdges = (finalEdges, finalPkToGid, finalByNode)
        }

        // 2. Read nodes in batches — all off MainActor
        var nodeBatch: [(pk: Int64, node: NodeData)] = []
        for m in bgLattice.objects(Memory.self) {
            guard let gid = m.globalId, let pk = m.primaryKey else { continue }
            if let filter = nodeFilter, !filter(m) { continue }
            nodeBatch.append((pk, NodeData(
                id: gid, project: displayProject(m), topic: m.topic,
                label: extractLabel(content: m.content, topic: m.topic),
                content: m.content,
                createdAt: m.createdAt, lastAccessedAt: m.lastAccessedAt,
                importance: m.importance
            )))
            if nodeBatch.count >= batchSize {
                let batch = nodeBatch
                nodeBatch = []
                pendingUpdate.withLock { $0.bulkNodeBatches.append(batch) }
            }
        }
        if !nodeBatch.isEmpty {
            let batch = nodeBatch
            pendingUpdate.withLock { $0.bulkNodeBatches.append(batch) }
        }

        // 3. Mark finalize — drain loop will recomputeDerivedData + wake sim
        pendingUpdate.withLock { $0.finalize = true }

        // 4. Cluster computation off main actor — push result into update buffer.
        // Serialized ACROSS galaxies: the per-project vector queries are the
        // most allocation-heavy phase of a load, and three galaxies running
        // them concurrently over multi-GB DBs is what pushed the SQLite page
        // cache to ~1GB and into transient OOM (crash 2026-08-05). Nodes and
        // edges above still load fully in parallel; only this tail is gated.
        var allProjects = Set<String>()
        for batch in pendingUpdate.withLock({ $0.bulkNodeBatches }) {
            for (_, node) in batch { allProjects.insert(node.project) }
        }
        let projects = allProjects
        let finalClusters = await ClusterPassGate.shared.run { [ref] in
            guard let lat = ref.resolve() else { return [[UUID]]() }
            var clusters: [[UUID]] = []
            for project in projects {
                clusters.append(contentsOf: findMemoryClusters(
                    in: lat, project: project,
                    minClusterSize: 2, neighborLimit: 20).clusters)
            }
            return clusters
        }
        pendingUpdate.withLock { $0.clusterGroups = finalClusters }
    }
    
    // MARK: insert node batch
    @MainActor func insertNodeBatch(_ batch: [NodeData], config: DrainConfig) {
        let store = renderStore
        let sim3D = simulation3D

        for nd in batch {
            store.allNodes[nd.id] = nd

            let visible = !config.hiddenProjects.contains(nd.project) &&
                (config.timeFilter == nil || nd.createdAt <= config.timeFilter!)
            guard visible else { continue }

            store.nodes.append(nd)
            store.nodeById[nd.id] = nd
            store.visibleNodeIds.insert(nd.id)

            if config.is3D {
                sim3D?.addNode(nd.id, project: nd.project, topic: nd.topic, galaxyId: self.id)
            }

            // Wire edges where BOTH endpoints now exist
            for edge in store.edgesByNode[nd.id] ?? [] {
                let otherId = edge.sourceId == nd.id ? edge.targetId : edge.sourceId
                guard store.visibleNodeIds.contains(otherId) else { continue }
                guard !config.hiddenRelations.contains(edge.relation) else { continue }
                if !store.filteredEdgeIds.contains(edge.id) {
                    store.filteredEdgeIds.insert(edge.id)
                    store.edges.append(edge)
                    if config.is3D {
                        sim3D?.addEdge(from: edge.sourceId, to: edge.targetId)
                    }
                }
            }

            // Hub detection
            if let edges = store.edgesByNode[nd.id] {
                for edge in edges where edge.relation == "part_of" && edge.targetId == nd.id {
                    store.hubs.insert(nd.id)
                    break
                }
            }

            // Assign color for previously unseen project
            if store.colorMap[nd.project] == nil {
                if nd.project == "global" {
                    store.colorMap["global"] = .gray
                } else {
                    let idx = store.colorMap.count - (store.colorMap["global"] != nil ? 1 : 0)
                    store.colorMap[nd.project] = GraphView.goldenAngleColor(at: idx)
                }
            }
        }
    }

    // MARK: - Render Update Drain

    /// Drain all pending updates atomically on MainActor. Called once per frame
    /// from renderTick, before simulation tick. Ensures store + simulation are
    /// always consistent — no partial states visible to the render pipeline.
    ///
    /// Bulk data (initial load) is held until `finalize` arrives — this guarantees
    /// all galaxies are registered and world layout is final before any nodes are
    /// positioned, preventing center-of-origin clustering when galaxies register
    /// asynchronously.
    @MainActor
    func drainPendingUpdate(config: DrainConfig) {
        // Peek: if there's bulk data but no finalize yet, leave it for next frame.
        // This ensures all galaxies are registered (and world layout computed)
        // before any bulk node positions are assigned.
        let hasBulk = pendingUpdate.withLock { !$0.bulkNodeBatches.isEmpty || $0.bulkEdges != nil }
        let hasFinalize = pendingUpdate.withLock { $0.finalize }
        if hasBulk && !hasFinalize {
            // Only drain incremental changes (observer inserts/updates/deletes),
            // NOT bulk data. Bulk waits for finalize.
            let incremental = pendingUpdate.withLock { u -> RenderUpdate in
                var partial = RenderUpdate()
                partial.insertedNodes = u.insertedNodes; u.insertedNodes = []
                partial.updatedNodes = u.updatedNodes; u.updatedNodes = []
                partial.removedNodePks = u.removedNodePks; u.removedNodePks = []
                partial.insertedEdges = u.insertedEdges; u.insertedEdges = []
                partial.updatedEdges = u.updatedEdges; u.updatedEdges = []
                partial.removedEdgeGids = u.removedEdgeGids; u.removedEdgeGids = []
                return partial
            }
            if !incremental.isEmpty {
                applyIncrementalUpdate(incremental,
                                       config: config)
            }
            return
        }

        // Full drain — finalize is present or no bulk data
        let update = pendingUpdate.withLock { u -> RenderUpdate in
            let copy = u; u = .init(); return copy
        }
        guard !update.isEmpty else { return }

        let store = renderStore

        // Bulk edges (initial load)
        if let bulk = update.bulkEdges {
            store.allEdges = bulk.allEdges
            store.edgePkToGlobalId = bulk.pkToGid
            store.edgesByNode = bulk.byNode
        }

        // Bulk node batches (initial load) — only processed when finalize is present,
        // so simulation3D.center is already at the final world position.
        if !update.bulkNodeBatches.isEmpty {
            let totalNodes = update.bulkNodeBatches.reduce(0) { $0 + $1.count }
            print("DRAIN: galaxy=\(id) processing \(totalNodes) bulk nodes, sim.center=\(simulation3D?.center ?? .zero), worldCenter=\(worldCenter)")
        }
        for batch in update.bulkNodeBatches {
            var nodes: [NodeData] = []
            nodes.reserveCapacity(batch.count)
            for (pk, node) in batch {
                store.pkToGlobalId[pk] = node.id
                nodes.append(node)
            }
            insertNodeBatch(nodes, config: config)
        }

        // Incremental changes
        applyIncrementalUpdate(update, config: config)

        // Finalize (end of initial load)
        if update.finalize {
            recomputeDerivedData()
            store.bumpTopology()
            if config.is3D { simulation3D?.wake() }
            isInitialLoad = false
            isLoaded = true
        }

        // Cluster groups (computed off MainActor, applied here)
        if let clusters = update.clusterGroups {
            store.clusterGroups = clusters
        }
    }

    @MainActor
    private func applyIncrementalUpdate(_ update: RenderUpdate, config: DrainConfig) {
        let store = renderStore
        for (pk, node) in update.insertedNodes {
            handleNodeInsert(pk: pk, node: node, config: config)
        }
        for (pk, node) in update.updatedNodes {
            handleNodeUpdate(pk: pk, node: node, config: config)
        }
        for pk in update.removedNodePks {
            handleNodeDelete(pk, config: config)
        }
        for (pk, edge) in update.insertedEdges {
            store.edgePkToGlobalId[pk] = edge.id
            handleEdgeInsert(edge, config: config)
        }
        for edge in update.updatedEdges {
            handleEdgeUpdate(edge)
        }
        for gid in update.removedEdgeGids {
            handleEdgeDelete(gid)
        }
    }

    @MainActor
    func handleNodeInsert(pk: Int64, node: NodeData, config: DrainConfig) {
        let store = renderStore
        store.pendingNodeInserts.append((pk: pk, node: node))
        if store.pendingNodeFlush == nil {
            let capturedConfig = config
            store.pendingNodeFlush = Task { @MainActor in
                await Task.yield()
                flushPendingNodeInserts(config: capturedConfig)
                store.pendingNodeFlush = nil
            }
        }
    }
    
    @MainActor func handleNodeUpdate(pk: Int64, node: NodeData, config: DrainConfig) {
        let t0 = CFAbsoluteTimeGetCurrent()
        defer { ObserverAccumulator.shared.record("nodeUpdate", ms: (CFAbsoluteTimeGetCurrent() - t0) * 1000.0) }
        let store = renderStore
        let gid = node.id
        guard let old = store.allNodes[gid] else {
            // Unknown to this galaxy — either filtered at load or just now
            // entering the partition. A plain dict write here half-inserted
            // (allNodes/nodeById but never nodes[]/simulation); take the
            // real insert path instead.
            handleNodeInsert(pk: pk, node: node, config: config)
            return
        }

        // Notify mascot fleet of the update
        do {
            if old.project != node.project {
                // Project changed: farewell from old mascot, welcome from new
                mascotFleet?.onNodeUpdated(
                    nodeId: gid, project: old.project,
                    changes: NodeChangeInfo(importanceChanged: false, topicChanged: false, isAccessOnly: false)
                )
                mascotFleet?.onNodeUpdated(
                    nodeId: gid, project: node.project,
                    changes: NodeChangeInfo(importanceChanged: false, topicChanged: true, isAccessOnly: false)
                )
            } else {
                let changes = NodeChangeInfo(
                    importanceChanged: old.importance != node.importance,
                    topicChanged: old.topic != node.topic,
                    isAccessOnly: old.topic == node.topic &&
                                  old.importance == node.importance && old.label == node.label &&
                                  old.content == node.content
                )
                mascotFleet?.onNodeUpdated(nodeId: gid, project: node.project, changes: changes)
            }
        }
        if node.lastAccessedAt > old.lastAccessedAt {
            #if ENGRAM_INSTRUMENTATION
            let wasAlreadyGlowing = store.glowingNodes[gid] != nil
            #endif
            store.glowingNodes[gid] = Date()
            #if ENGRAM_INSTRUMENTATION
            if glowLogFile == nil {
                glowLogFile = fopen("/tmp/glow-log.csv", "w")
                if let f = glowLogFile {
                    fputs("timestamp,galaxy,node_label,old_accessed,new_accessed,delta_s,staleness_s,already_glowing,glow_count\n", f)
                }
            }
            if let f = glowLogFile {
                let now = Date()
                let delta = node.lastAccessedAt.timeIntervalSince(old.lastAccessedAt)
                let staleness = now.timeIntervalSince(node.lastAccessedAt)
                let alreadyGlowing = wasAlreadyGlowing
                let ts = String(format: "%.3f", now.timeIntervalSince1970)
                let line = "\(ts),\(id),\(node.label.prefix(30)),\(old.lastAccessedAt),\(node.lastAccessedAt),\(String(format: "%.1f", delta)),\(String(format: "%.1f", staleness)),\(alreadyGlowing),\(store.glowingNodes.count)\n"
                fputs(line, f)
                fflush(f)
            }
            #endif
        }
        let structuralChange = old.project != node.project ||
            old.topic != node.topic ||
            old.importance != node.importance ||
            old.label != node.label
        if !structuralChange {
            // Access-only change: update dicts (O(1)), skip O(n) array scan.
            // nodes[] array gets stale lastAccessedAt but that field doesn't affect
            // rendering — it's only used by activity log which reads from nodeById.
            store.allNodes[gid]?.lastAccessedAt = node.lastAccessedAt
            store.nodeById[gid]?.lastAccessedAt = node.lastAccessedAt
            return
        }
        store.allNodes[gid] = node
        store.nodeById[gid] = node
        if let idx = store.nodes.firstIndex(where: { $0.id == gid }) {
            store.nodes[idx] = node
        }
    }
    
    @MainActor func handleNodeDelete(_ pk: Int64, config: DrainConfig) {
        let soundEnabled = config.soundEnabled
        let t0 = CFAbsoluteTimeGetCurrent()
        defer { ObserverAccumulator.shared.record("nodeDelete", ms: (CFAbsoluteTimeGetCurrent() - t0) * 1000.0) }
        let store = renderStore
        guard let gid = store.pkToGlobalId[pk] else { return }

        // Notify mascot fleet before removing — capture position for absorb animation
        if let nodeData = store.allNodes[gid] {
            let lastPos = simulation3D?.positions[gid]
            mascotFleet?.onNodeDeleted(nodeId: gid, project: nodeData.project, lastPosition: lastPos)
        }

        store.pkToGlobalId.removeValue(forKey: pk)
        store.allNodes.removeValue(forKey: gid)
        store.glowingNodes.removeValue(forKey: gid)
        store.newNodeGlows.removeValue(forKey: gid)
        let removedEdgeIds = Set(store.edges.filter { $0.sourceId == gid || $0.targetId == gid }.map(\.id))
        store.filteredEdgeIds.subtract(removedEdgeIds)
        store.visibleNodeIds.remove(gid)
        // Note: ForceSimulation3D doesn't support surgical remove — it rebuilds via updateGraph().
        // The node disappears from render data immediately; simulation catches up on next rebuild.
        store.nodes.removeAll { $0.id == gid }
        store.nodeById.removeValue(forKey: gid)
        store.edges.removeAll { $0.sourceId == gid || $0.targetId == gid }
        store.hubs.remove(gid)
        store.bumpTopology()
        if !isInitialLoad && soundEnabled {
            DispatchQueue.global(qos: .utility).async { GraphView.removeSound?.play() }
        }
        store.clusterGroups = store.clusterGroups.compactMap { cluster in
            let filtered = cluster.filter { $0 != gid }
            return filtered.count >= 2 ? filtered : nil
        }
    }
    
    @MainActor
    func flushPendingNodeInserts(config: DrainConfig) {
        let t0 = CFAbsoluteTimeGetCurrent()
        let store = renderStore
        let entries = store.pendingNodeInserts
        store.pendingNodeInserts.removeAll(keepingCapacity: true)
        guard !entries.isEmpty else { return }

        #if ENGRAM_INSTRUMENTATION
        let flushStart = CFAbsoluteTimeGetCurrent()
        #endif

        var bumpedTopology = false

        for (pk, node) in entries {
            let gid = node.id
            store.pkToGlobalId[pk] = gid
            store.allNodes[gid] = node
            store.newNodeGlows[gid] = Date()

            let visible = !config.hiddenProjects.contains(node.project) &&
                (config.timeFilter == nil || node.createdAt <= config.timeFilter!)
            guard visible && !store.visibleNodeIds.contains(gid) else { continue }

            store.nodes.append(node)
            store.nodeById[gid] = node
            store.visibleNodeIds.insert(gid)

            // Notify mascot fleet of new node (only after initial load)
            if !isInitialLoad {
                mascotFleet?.onNodeCreated(nodeId: gid, project: node.project)
            }

            if config.is3D {
                simulation3D?.addNode(gid, project: node.project, topic: node.topic, galaxyId: self.id)
            }

            let nodeIds = store.visibleNodeIds
            for edge in store.edgesByNode[gid] ?? [] {
                let otherId = edge.sourceId == gid ? edge.targetId : edge.sourceId
                guard nodeIds.contains(otherId),
                      !config.hiddenRelations.contains(edge.relation) else { continue }
                if config.is3D {
                    simulation3D?.addEdge(from: edge.sourceId, to: edge.targetId)
                }
                if !store.filteredEdgeIds.contains(edge.id) {
                    store.filteredEdgeIds.insert(edge.id)
                    store.edges.append(edge)
                }
                if edge.relation == "part_of" && edge.targetId == gid {
                    store.hubs.insert(gid)
                }
            }

            if store.colorMap[node.project] == nil && node.project != "global" {
                let idx = store.colorMap.count - 1
                store.colorMap[node.project] = GraphView.goldenAngleColor(at: idx)
            }
            bumpedTopology = true
        }

        if bumpedTopology {
            store.bumpTopology()
            if !isInitialLoad && config.soundEnabled {
                DispatchQueue.global(qos: .utility).async { GraphView.addSound?.play() }
            }
        }

        #if ENGRAM_INSTRUMENTATION
        let flushMs = (CFAbsoluteTimeGetCurrent() - flushStart) * 1000.0
        if flushTimingFile == nil {
            flushTimingFile = fopen("/tmp/flush-timing.csv", "w")
            if let f = flushTimingFile {
                fputs("timestamp,galaxy,batch_size,flush_ms,edges_added,topology_bumped\n", f)
            }
        }
        if let f = flushTimingFile {
            let ts = String(format: "%.3f", CFAbsoluteTimeGetCurrent())
            let line = "\(ts),\(id),\(entries.count),\(String(format: "%.2f", flushMs)),\(store.edges.count),\(bumpedTopology)\n"
            fputs(line, f)
            fflush(f)
        }
        #endif
    }
    
    
    @MainActor
    func handleEdgeInsert(_ data: EdgeData, config: DrainConfig) {
        let store = renderStore
        store.pendingEdgeInserts.append((pk: data.id, edge: data))
        if store.pendingEdgeFlush == nil {
            let capturedConfig = config
            store.pendingEdgeFlush = Task { @MainActor in
                await Task.yield()
                flushPendingEdgeInserts(config: capturedConfig)
                store.pendingEdgeFlush = nil
            }
        }
    }

    @MainActor func handleEdgeUpdate(_ data: EdgeData) {
        let store = renderStore
        let old = store.allEdges[data.id]
        store.allEdges[data.id] = data
        // Only scan the edges array when structural fields changed (relation, endpoints).
        // Edge updates that don't change routing (rare) skip the O(edges) linear scan.
        let structuralChange = old == nil ||
            old!.relation != data.relation ||
            old!.sourceId != data.sourceId ||
            old!.targetId != data.targetId
        if structuralChange {
            if let idx = store.edges.firstIndex(where: { $0.id == data.id }) {
                store.edges[idx] = data
            }
        }
    }

    @MainActor func handleEdgeDelete(_ gid: UUID) {
        let store = renderStore
        if let old = store.allEdges[gid] {
            // Note: ForceSimulation3D doesn't support surgical removeEdge — see removeNode note.
            store.filteredEdgeIds.remove(gid)
            store.edges.removeAll { $0.id == gid }
            store.edgesByNode[old.sourceId]?.removeAll { $0.id == gid }
            store.edgesByNode[old.targetId]?.removeAll { $0.id == gid }
            store.edgeCountByNode[old.sourceId, default: 1] -= 1
            store.edgeCountByNode[old.targetId, default: 1] -= 1
            if old.relation == "part_of" {
                let stillHub = store.allEdges.values.contains { $0.id != gid && $0.relation == "part_of" && $0.targetId == old.targetId }
                if !stillHub { store.hubs.remove(old.targetId) }
            }
        }
        store.allEdges.removeValue(forKey: gid)
    }
    
    @MainActor func flushPendingEdgeInserts(config: DrainConfig) {
        let t0 = CFAbsoluteTimeGetCurrent()
        let store = renderStore
        let entries = store.pendingEdgeInserts
        store.pendingEdgeInserts.removeAll(keepingCapacity: true)
        guard !entries.isEmpty else { return }
        defer {
            let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
            if ms > 5 { os_log(.fault, log: stallLog, "HOTPATH: flushEdgeInserts count=%d %.1fms", entries.count, ms) }
        }

        #if ENGRAM_INSTRUMENTATION
        let edgeFlushStart = CFAbsoluteTimeGetCurrent()
        #endif

        var addedVisibleEdge = false

        for (_, data) in entries {
            store.allEdges[data.id] = data
            store.edgesByNode[data.sourceId, default: []].append(data)
            store.edgesByNode[data.targetId, default: []].append(data)
            store.edgeCountByNode[data.sourceId, default: 0] += 1
            store.edgeCountByNode[data.targetId, default: 0] += 1
            let nodeIds = store.visibleNodeIds
            if nodeIds.contains(data.sourceId) && nodeIds.contains(data.targetId) &&
               !config.hiddenRelations.contains(data.relation) {
                store.filteredEdgeIds.insert(data.id)
                store.edges.append(data)
                if config.is3D {
                    simulation3D?.addEdge(from: data.sourceId, to: data.targetId)
                }
                addedVisibleEdge = true
            }
            if data.relation == "part_of" {
                store.hubs.insert(data.targetId)
            }
        }
        // Bump topology whenever visible edges were added — ensures mergeRenderData
        // picks up new edges (not just part_of hub changes).
        if addedVisibleEdge {
            store.bumpTopology()
        }

        #if ENGRAM_INSTRUMENTATION
        let edgeFlushMs = (CFAbsoluteTimeGetCurrent() - edgeFlushStart) * 1000.0
        if flushTimingFile == nil {
            flushTimingFile = fopen("/tmp/flush-timing.csv", "w")
            if let f = flushTimingFile {
                fputs("timestamp,galaxy,batch_size,flush_ms,edges_added,topology_bumped\n", f)
            }
        }
        if let f = flushTimingFile {
            let ts = String(format: "%.3f", CFAbsoluteTimeGetCurrent())
            let line = "\(ts),\(id),\(entries.count),\(String(format: "%.2f", edgeFlushMs)),\(store.edges.count),\(addedVisibleEdge)\n"
            fputs(line, f)
            fflush(f)
        }
        #endif
    }
    
    @MainActor func recomputeDerivedData() {
        let store = renderStore

        // Color map
        let projects = Set(store.allNodes.values.map(\.project)).sorted()
        if store.colorMap["global"] == nil {
            store.colorMap["global"] = .gray
        }
        for project in projects {
            if project == "global" { continue }
            if store.colorMap[project] == nil {
                let idx = store.colorMap.count - 1
                store.colorMap[project] = GraphView.goldenAngleColor(at: idx)
            }
        }

        // Visible node IDs
        store.visibleNodeIds = Set(store.nodes.map(\.id))
        store.nodeById = Dictionary(store.nodes.map { ($0.id, $0) }, uniquingKeysWith: { _, b in b })

        // Relation counts
        let nodeIds = store.visibleNodeIds
        var counts: [String: Int] = [:]
        for edge in store.allEdges.values {
            guard nodeIds.contains(edge.sourceId), nodeIds.contains(edge.targetId) else { continue }
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

        // Per-node edge counts
        var edgeCounts: [UUID: Int] = [:]
        for edge in store.allEdges.values {
            edgeCounts[edge.sourceId, default: 0] += 1
            edgeCounts[edge.targetId, default: 0] += 1
        }
        store.edgeCountByNode = edgeCounts
    }
    
    @MainActor private func recomputeStatsOnly() {
        let store = renderStore
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
    
    @MainActor func setIsInitialLoad(_ isInitialLoad: Bool) {
        self.isInitialLoad = isInitialLoad
    }
}
