import SwiftUI
import Lattice
import Combine
import EngramKit
import os

private let flushLog = Logger(subsystem: "io.engram.app", category: "GalaxyLoader")
private let stallLog = OSLog(subsystem: "io.engram.app", category: "FrameStall")

/// Accumulates observer callback time + counts between frames for stall diagnosis.
@MainActor
final class ObserverAccumulator {
    static let shared = ObserverAccumulator()
    var totalMs: Double = 0
    var nodeUpdateCount: Int = 0
    var nodeInsertCount: Int = 0
    var nodeDeleteCount: Int = 0
    var edgeInsertCount: Int = 0
    var edgeUpdateCount: Int = 0
    var edgeDeleteCount: Int = 0
    var otherCount: Int = 0
    var otherMs: Double = 0  // for non-galaxy-data-loader main-thread work

    func record(_ kind: String, ms: Double) {
        totalMs += ms
        switch kind {
        case "nodeUpdate": nodeUpdateCount += 1
        case "nodeInsert": nodeInsertCount += 1
        case "nodeDelete": nodeDeleteCount += 1
        case "edgeInsert": edgeInsertCount += 1
        case "edgeUpdate": edgeUpdateCount += 1
        case "edgeDelete": edgeDeleteCount += 1
        default: otherCount += 1; otherMs += ms
        }
    }

    func drainAndLog() {
        let total = totalMs
        let nU = nodeUpdateCount, nI = nodeInsertCount, nD = nodeDeleteCount
        let eI = edgeInsertCount, eU = edgeUpdateCount, eD = edgeDeleteCount
        let oC = otherCount, oMs = otherMs
        totalMs = 0; nodeUpdateCount = 0; nodeInsertCount = 0; nodeDeleteCount = 0
        edgeInsertCount = 0; edgeUpdateCount = 0; edgeDeleteCount = 0
        otherCount = 0; otherMs = 0
        let ops = nU + nI + nD + eI + eU + eD + oC
        if ops > 0 || total > 1 {
            os_log(.fault, log: stallLog,
                   "ACCUM: %.1fms total, %d ops (nUpd=%d nIns=%d nDel=%d eIns=%d eUpd=%d eDel=%d other=%d/%.1fms)",
                   total, ops, nU, nI, nD, eI, eU, eD, oC, oMs)
        }
    }
}

/// Data loading, observation, and batched flush logic parameterized on a Galaxy.
/// Extracted from GraphView+DataLoading — same batched loading + observer pattern.
@MainActor
struct GalaxyDataLoader {

    // MARK: - Data Loading

    /// Load all data from a galaxy's Lattice into its renderStore + simulation.
    /// Sets up live observers for incremental updates after initial load.
    static func loadData(
        into galaxy: Galaxy,
        hiddenProjects: Set<String>,
        hiddenRelations: Set<String>,
        timeFilter: Date?,
        is3D: Bool,
        soundEnabled: Bool,
        notificationsEnabled: Bool
    ) {
        let batchSize = 50
        let ref = galaxy.lattice.sendableReference
        let nodeFilter = galaxy.nodeFilter

        // galaxy is @MainActor but only accessed within MainActor.run blocks below.
        // nonisolated(unsafe) suppresses the sending diagnostic — safe because all
        // galaxy mutations happen inside MainActor.run.
        nonisolated(unsafe) let galaxy = galaxy
        Task.detached {
            guard let bgLattice = ref.resolve() else { return }

            // 1. Read all edges off main actor
            var allEdgeBatch: [(EdgeData, Int64)] = []  // (data, pk) — pk for observer delete routing
            for e in bgLattice.objects(MemoryEdge.self) {
                guard let pk = e.primaryKey, let gid = e.__globalId else { continue }
                allEdgeBatch.append((EdgeData(id: gid, sourceId: e.sourceGlobalId,
                                              targetId: e.targetGlobalId, relation: e.relation.rawValue), pk))
            }

            await MainActor.run {
                let store = galaxy.renderStore
                for (ed, pk) in allEdgeBatch {
                    store.allEdges[ed.id] = ed
                    store.edgePkToGlobalId[pk] = ed.id
                    store.edgesByNode[ed.sourceId, default: []].append(ed)
                    store.edgesByNode[ed.targetId, default: []].append(ed)
                }
            }

            // 2. Read nodes in batches
            var nodeBatch: [NodeData] = []
            var pkBatch: [Int64: UUID] = [:]
            for m in bgLattice.objects(Memory.self) {
                guard let gid = m.__globalId, let pk = m.primaryKey else { continue }
                // Apply per-galaxy node filter (for data partitioning)
                if let filter = nodeFilter, !filter(m) { continue }

                pkBatch[pk] = gid
                nodeBatch.append(NodeData(
                    id: gid, project: m.project, topic: m.topic,
                    label: extractLabel(content: m.content, topic: m.topic),
                    content: m.content,
                    createdAt: m.createdAt, lastAccessedAt: m.lastAccessedAt,
                    importance: m.importance
                ))
                if nodeBatch.count >= batchSize {
                    let batch = nodeBatch
                    let pks = pkBatch
                    nodeBatch = []
                    pkBatch = [:]
                    await MainActor.run {
                        galaxy.renderStore.pkToGlobalId.merge(pks) { _, new in new }
                        insertNodeBatch(batch, into: galaxy,
                                        hiddenProjects: hiddenProjects,
                                        hiddenRelations: hiddenRelations,
                                        timeFilter: timeFilter, is3D: is3D)
                    }
                }
            }
            // Flush remainder
            if !nodeBatch.isEmpty {
                let batch = nodeBatch
                let pks = pkBatch
                await MainActor.run {
                    galaxy.renderStore.pkToGlobalId.merge(pks) { _, new in new }
                    insertNodeBatch(batch, into: galaxy,
                                    hiddenProjects: hiddenProjects,
                                    hiddenRelations: hiddenRelations,
                                    timeFilter: timeFilter, is3D: is3D)
                }
            }

            // 3. Final reconciliation
            await MainActor.run {
                recomputeDerivedData(for: galaxy)
                galaxy.renderStore.bumpTopology()
                galaxy.isInitialLoad = false
                galaxy.isLoaded = true

                if is3D { galaxy.simulation3D.wake() }

                // 4. Set up live observers
                setupObservers(for: galaxy, hiddenProjects: hiddenProjects,
                               hiddenRelations: hiddenRelations, timeFilter: timeFilter,
                               is3D: is3D, soundEnabled: soundEnabled,
                               notificationsEnabled: notificationsEnabled)
            }

            // 5. Cluster computation off main actor
            let visibleProjects = await MainActor.run {
                Set(galaxy.renderStore.nodes.map(\.project))
            }
            var clusters: [[UUID]] = []
            for project in visibleProjects {
                let projectClusters = findMemoryClusters(in: bgLattice, project: project, minClusterSize: 2, neighborLimit: 20).clusters
                clusters.append(contentsOf: projectClusters)
            }
            let result = clusters
            await MainActor.run {
                galaxy.renderStore.clusterGroups = result
            }
        }
    }

    // MARK: - Batch Insert

    static func insertNodeBatch(_ batch: [NodeData], into galaxy: Galaxy,
                                hiddenProjects: Set<String>, hiddenRelations: Set<String>,
                                timeFilter: Date?, is3D: Bool) {
        let store = galaxy.renderStore
        let sim3D = galaxy.simulation3D

        for nd in batch {
            store.allNodes[nd.id] = nd

            let visible = !hiddenProjects.contains(nd.project) &&
                (timeFilter == nil || nd.createdAt <= timeFilter!)
            guard visible else { continue }

            store.nodes.append(nd)
            store.nodeById[nd.id] = nd
            store.visibleNodeIds.insert(nd.id)

            if is3D {
                sim3D.addNode(nd.id, project: nd.project, topic: nd.topic)
            }

            // Wire edges where BOTH endpoints now exist
            for edge in store.edgesByNode[nd.id] ?? [] {
                let otherId = edge.sourceId == nd.id ? edge.targetId : edge.sourceId
                guard store.visibleNodeIds.contains(otherId) else { continue }
                guard !hiddenRelations.contains(edge.relation) else { continue }
                if !store.filteredEdgeIds.contains(edge.id) {
                    store.filteredEdgeIds.insert(edge.id)
                    store.edges.append(edge)
                    if is3D {
                        sim3D.addEdge(from: edge.sourceId, to: edge.targetId)
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

    // MARK: - Observers

    @globalActor struct GalaxyLatticeActor {
        actor ActorType {}
        static let shared = ActorType()
    }
    
    static func setupObservers(
        for galaxy: Galaxy,
        hiddenProjects: Set<String>,
        hiddenRelations: Set<String>,
        timeFilter: Date?,
        is3D: Bool,
        soundEnabled: Bool,
        notificationsEnabled: Bool
    ) {
        galaxy.setUpObserve(hiddenProjects: hiddenProjects,
                            hiddenRelations: hiddenRelations,
                            timeFilter: timeFilter,
                            is3D: is3D,
                            soundEnabled: soundEnabled,
                            notificationsEnabled: notificationsEnabled)
    }

    // MARK: - Node Change Handlers

    static func handleNodeInsert(pk: Int64, node: NodeData, galaxy: Galaxy,
                                  is3D: Bool, hiddenProjects: Set<String>,
                                  hiddenRelations: Set<String>, timeFilter: Date?,
                                  soundEnabled: Bool, notificationsEnabled: Bool) {
        let store = galaxy.renderStore
        store.pendingNodeInserts.append((pk: pk, node: node))
        if store.pendingNodeFlush == nil {
            store.pendingNodeFlush = Task { @MainActor in
                await Task.yield()
                flushPendingNodeInserts(galaxy: galaxy, is3D: is3D,
                                        hiddenProjects: hiddenProjects,
                                        hiddenRelations: hiddenRelations,
                                        timeFilter: timeFilter,
                                        soundEnabled: soundEnabled,
                                        notificationsEnabled: notificationsEnabled)
                store.pendingNodeFlush = nil
            }
        }
    }

    #if ENGRAM_INSTRUMENTATION
    private static var glowLogFile: UnsafeMutablePointer<FILE>? = nil
    #endif

    static func handleNodeUpdate(node: NodeData, galaxy: Galaxy) {
        let t0 = CFAbsoluteTimeGetCurrent()
        defer { ObserverAccumulator.shared.record("nodeUpdate", ms: (CFAbsoluteTimeGetCurrent() - t0) * 1000.0) }
        let store = galaxy.renderStore
        let gid = node.id
        let old = store.allNodes[gid]

        // Notify mascot fleet of the update
        if let old {
            if old.project != node.project {
                // Project changed: farewell from old mascot, welcome from new
                galaxy.mascotFleet?.onNodeUpdated(
                    nodeId: gid, project: old.project,
                    changes: NodeChangeInfo(importanceChanged: false, topicChanged: false, isAccessOnly: false)
                )
                galaxy.mascotFleet?.onNodeUpdated(
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
                galaxy.mascotFleet?.onNodeUpdated(nodeId: gid, project: node.project, changes: changes)
            }
        }
        if let old, node.lastAccessedAt > old.lastAccessedAt {
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
                let line = "\(ts),\(galaxy.id),\(node.label.prefix(30)),\(old.lastAccessedAt),\(node.lastAccessedAt),\(String(format: "%.1f", delta)),\(String(format: "%.1f", staleness)),\(alreadyGlowing),\(store.glowingNodes.count)\n"
                fputs(line, f)
                fflush(f)
            }
            #endif
        }
        let structuralChange = old == nil ||
            old!.project != node.project ||
            old!.topic != node.topic ||
            old!.importance != node.importance ||
            old!.label != node.label
        if !structuralChange {
            store.allNodes[gid]?.lastAccessedAt = node.lastAccessedAt
            if let idx = store.nodes.firstIndex(where: { $0.id == gid }) {
                store.nodes[idx].lastAccessedAt = node.lastAccessedAt
            }
            return
        }
        store.allNodes[gid] = node
        store.nodeById[gid] = node
        if let idx = store.nodes.firstIndex(where: { $0.id == gid }) {
            store.nodes[idx] = node
        }
    }

    static func handleNodeDelete(_ pk: Int64, galaxy: Galaxy,
                                  soundEnabled: Bool, notificationsEnabled: Bool) {
        let t0 = CFAbsoluteTimeGetCurrent()
        defer { ObserverAccumulator.shared.record("nodeDelete", ms: (CFAbsoluteTimeGetCurrent() - t0) * 1000.0) }
        let store = galaxy.renderStore
        guard let gid = store.pkToGlobalId[pk] else { return }

        // Notify mascot fleet before removing — capture position for absorb animation
        if let nodeData = store.allNodes[gid] {
            let lastPos = galaxy.simulation3D.positions[gid]
            galaxy.mascotFleet?.onNodeDeleted(nodeId: gid, project: nodeData.project, lastPosition: lastPos)
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
        if !galaxy.isInitialLoad && soundEnabled {
            DispatchQueue.global(qos: .utility).async { GraphView.removeSound?.play() }
        }
        store.clusterGroups = store.clusterGroups.compactMap { cluster in
            let filtered = cluster.filter { $0 != gid }
            return filtered.count >= 2 ? filtered : nil
        }
    }

    // MARK: - Edge Change Handlers

    static func handleEdgeInsert(_ data: EdgeData, galaxy: Galaxy,
                                  is3D: Bool, hiddenRelations: Set<String>) {
        let store = galaxy.renderStore
        store.pendingEdgeInserts.append((pk: data.id, edge: data))
        if store.pendingEdgeFlush == nil {
            store.pendingEdgeFlush = Task { @MainActor in
                await Task.yield()
                flushPendingEdgeInserts(galaxy: galaxy, is3D: is3D, hiddenRelations: hiddenRelations)
                store.pendingEdgeFlush = nil
            }
        }
    }

    static func handleEdgeUpdate(_ data: EdgeData, galaxy: Galaxy) {
        let t0 = CFAbsoluteTimeGetCurrent()
        defer { ObserverAccumulator.shared.record("edgeUpdate", ms: (CFAbsoluteTimeGetCurrent() - t0) * 1000.0) }
        let store = galaxy.renderStore
        store.allEdges[data.id] = data
        if let idx = store.edges.firstIndex(where: { $0.id == data.id }) {
            store.edges[idx] = data
        }
    }

    static func handleEdgeDelete(_ gid: UUID, galaxy: Galaxy) {
        let t0 = CFAbsoluteTimeGetCurrent()
        defer { ObserverAccumulator.shared.record("edgeDelete", ms: (CFAbsoluteTimeGetCurrent() - t0) * 1000.0) }
        let store = galaxy.renderStore
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

    // MARK: - Batched Flush

    #if ENGRAM_INSTRUMENTATION
    private static var flushTimingFile: UnsafeMutablePointer<FILE>? = nil
    private static var observerTimingFile: UnsafeMutablePointer<FILE>? = nil
    #endif

    static func flushPendingNodeInserts(galaxy: Galaxy, is3D: Bool,
                                         hiddenProjects: Set<String>,
                                         hiddenRelations: Set<String>,
                                         timeFilter: Date?,
                                         soundEnabled: Bool,
                                         notificationsEnabled: Bool) {
        let t0 = CFAbsoluteTimeGetCurrent()
        let store = galaxy.renderStore
        let entries = store.pendingNodeInserts
        store.pendingNodeInserts.removeAll(keepingCapacity: true)
        guard !entries.isEmpty else { return }
        defer {
            let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
            if ms > 5 { os_log(.fault, log: stallLog, "HOTPATH: flushNodeInserts count=%d %.1fms", entries.count, ms) }
        }

        #if ENGRAM_INSTRUMENTATION
        let flushStart = CFAbsoluteTimeGetCurrent()
        #endif

        var bumpedTopology = false

        for (pk, node) in entries {
            let gid = node.id
            store.pkToGlobalId[pk] = gid
            store.allNodes[gid] = node
            store.newNodeGlows[gid] = Date()

            let visible = !hiddenProjects.contains(node.project) &&
                (timeFilter == nil || node.createdAt <= timeFilter!)
            guard visible && !store.visibleNodeIds.contains(gid) else { continue }

            store.nodes.append(node)
            store.nodeById[gid] = node
            store.visibleNodeIds.insert(gid)

            // Notify mascot fleet of new node (only after initial load)
            if !galaxy.isInitialLoad {
                galaxy.mascotFleet?.onNodeCreated(nodeId: gid, project: node.project)
            }

            if is3D {
                galaxy.simulation3D.addNode(gid, project: node.project, topic: node.topic)
            }

            let nodeIds = store.visibleNodeIds
            for edge in store.edgesByNode[gid] ?? [] {
                let otherId = edge.sourceId == gid ? edge.targetId : edge.sourceId
                guard nodeIds.contains(otherId),
                      !hiddenRelations.contains(edge.relation) else { continue }
                if is3D {
                    galaxy.simulation3D.addEdge(from: edge.sourceId, to: edge.targetId)
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
            if !galaxy.isInitialLoad && soundEnabled {
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
            let line = "\(ts),\(galaxy.id),\(entries.count),\(String(format: "%.2f", flushMs)),\(store.edges.count),\(bumpedTopology)\n"
            fputs(line, f)
            fflush(f)
        }
        #endif
    }

    static func flushPendingEdgeInserts(galaxy: Galaxy, is3D: Bool, hiddenRelations: Set<String>) {
        let t0 = CFAbsoluteTimeGetCurrent()
        let store = galaxy.renderStore
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

        var bumpedTopology = false

        for (_, data) in entries {
            store.allEdges[data.id] = data
            store.edgesByNode[data.sourceId, default: []].append(data)
            store.edgesByNode[data.targetId, default: []].append(data)
            store.edgeCountByNode[data.sourceId, default: 0] += 1
            store.edgeCountByNode[data.targetId, default: 0] += 1
            let nodeIds = store.visibleNodeIds
            if nodeIds.contains(data.sourceId) && nodeIds.contains(data.targetId) &&
               !hiddenRelations.contains(data.relation) {
                store.filteredEdgeIds.insert(data.id)
                store.edges.append(data)
                if is3D {
                    galaxy.simulation3D.addEdge(from: data.sourceId, to: data.targetId)
                }
            }
            if data.relation == "part_of" {
                store.hubs.insert(data.targetId)
                bumpedTopology = true
            }
        }
        if bumpedTopology {
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
            let line = "\(ts),\(galaxy.id),\(entries.count),\(String(format: "%.2f", edgeFlushMs)),\(store.edges.count),\(bumpedTopology)\n"
            fputs(line, f)
            fflush(f)
        }
        #endif
    }

    // MARK: - Derived Data

    static func recomputeDerivedData(for galaxy: Galaxy) {
        let t0 = CFAbsoluteTimeGetCurrent()
        defer {
            let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
            if ms > 5 { os_log(.fault, log: stallLog, "HOTPATH: recomputeDerivedData nodes=%d edges=%d %.1fms", galaxy.renderStore.nodes.count, galaxy.renderStore.allEdges.count, ms) }
        }
        let store = galaxy.renderStore

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
}
