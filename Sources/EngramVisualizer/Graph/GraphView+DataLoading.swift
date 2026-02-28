import SwiftUI
import Lattice
import Combine
import EngramKit
import UserNotifications

// MARK: - Data Loading, Observation & Batched Flush

extension GraphView {

    // MARK: - Derived Data (cached, recomputed on structural changes)

    /// Recompute all derived data from current filtered nodes/edges. Call after any structural change.
    func recomputeDerivedData() {
        // Color map — preserve existing colors (stable across streaming + filter changes),
        // only assign new colors for projects not yet in the map.
        let projects = uniqueProjects()
        if renderStore.colorMap["global"] == nil {
            renderStore.colorMap["global"] = .gray
        }
        for project in projects {
            if project == "global" { continue }
            if renderStore.colorMap[project] == nil {
                let idx = renderStore.colorMap.count - 1  // -1 for global
                renderStore.colorMap[project] = Self.goldenAngleColor(at: idx)
            }
        }

        // Visible node IDs
        renderStore.visibleNodeIds = Set(renderStore.nodes.map(\.id))
        renderStore.nodeById = Dictionary(renderStore.nodes.map { ($0.id, $0) }, uniquingKeysWith: { _, b in b })

        // Relation counts (uses all edges, not filtered — so hidden relations still show in UI)
        let nodeIds = renderStore.visibleNodeIds
        var counts: [String: Int] = [:]
        for edge in renderStore.allEdges.values {
            guard nodeIds.contains(edge.sourceId), nodeIds.contains(edge.targetId) else { continue }
            counts[edge.relation, default: 0] += 1
        }
        renderStore.relationCounts = counts.sorted(by: { $0.key < $1.key })

        // Topic groups
        var groups: [String: (topic: String, project: String, ids: [UUID])] = [:]
        for node in renderStore.nodes {
            guard node.topic != "general", node.topic != "episode" else { continue }
            let key = "\(node.project)|\(node.topic)"
            var entry = groups[key] ?? (topic: node.topic, project: node.project, ids: [])
            entry.ids.append(node.id)
            groups[key] = entry
        }
        renderStore.topicGroups = groups.values
            .filter { $0.ids.count >= 2 }
            .map { TopicGroupInfo(topic: $0.topic, project: $0.project, ids: $0.ids) }

        // Per-node edge counts (for detail panel)
        var edgeCounts: [UUID: Int] = [:]
        for edge in renderStore.allEdges.values {
            edgeCounts[edge.sourceId, default: 0] += 1
            edgeCounts[edge.targetId, default: 0] += 1
        }
        renderStore.edgeCountByNode = edgeCounts
    }

    func recomputeClusters() {
        // findMemoryClusters returns [[Int64]] (primaryKeys) — convert to UUIDs via cached mapping
        let visibleProjects = Set(renderStore.nodes.map(\.project))
        var all: [[UUID]] = []
        for project in visibleProjects {
            let pkClusters = findMemoryClusters(in: lattice, project: project, minClusterSize: 2, neighborLimit: 20).clusters
            for pkCluster in pkClusters {
                let uuidCluster = pkCluster.compactMap { self.renderStore.pkToGlobalId[$0] }
                if uuidCluster.count >= 2 { all.append(uuidCluster) }
            }
        }
        renderStore.clusterGroups = all
    }

    // MARK: - Data Loading & Observation

    func loadData() {
        let batchSize = 50
        let ref = lattice.sendableReference
        let hiddenProjects = config.hiddenProjects
        let hiddenRelations = config.hiddenRelations
        let timeFilter = debouncedTimeSliderDate
        let is3D = config.dimensionMode == .threeD

        Task.detached {
            guard let bgLattice = ref.resolve() else { return }

            // 1. Read all edges off main actor — pure SQLite reads + struct construction.
            var allEdgeBatch: [EdgeData] = []
            for e in bgLattice.objects(MemoryEdge.self) {
                guard let pk = e.primaryKey else { continue }
                allEdgeBatch.append(EdgeData(id: pk, sourceId: e.sourceGlobalId, targetId: e.targetGlobalId, relation: e.relation.rawValue))
            }

            // Flush edges to main actor in one shot (lightweight — just dict insertions).
            await MainActor.run {
                for ed in allEdgeBatch {
                    renderStore.allEdges[ed.id] = ed
                    renderStore.edgesByNode[ed.sourceId, default: []].append(ed)
                    renderStore.edgesByNode[ed.targetId, default: []].append(ed)
                }
            }

            // 2. Read nodes in batches off main actor, flush each batch to main actor
            //    so nodes pop in incrementally while the run loop stays responsive.
            var nodeBatch: [NodeData] = []
            var pkBatch: [Int64: UUID] = [:]
            for m in bgLattice.objects(Memory.self) {
                guard let gid = m.__globalId, let pk = m.primaryKey else { continue }
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
                        self.renderStore.pkToGlobalId.merge(pks) { _, new in new }
                        insertNodeBatch(batch, hiddenProjects: hiddenProjects,
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
                    self.renderStore.pkToGlobalId.merge(pks) { _, new in new }
                    insertNodeBatch(batch, hiddenProjects: hiddenProjects,
                                    hiddenRelations: hiddenRelations,
                                    timeFilter: timeFilter, is3D: is3D)
                }
            }

            // 3. Final reconciliation on main actor
            await MainActor.run {
                recomputeDerivedData()
                renderStore.bumpTopology()
                // (renderStore is source of truth — no @State sync needed)
                isInitialLoad = false

                // Wake 3D simulation now that topology is stable — during batch loading,
                // force dispatches produced stale results (node count kept changing),
                // consuming alpha budget without doing useful work.
                if is3D { simulation3D.wake() }

                // 4. Set up live observers (uses original lattice from environment)
                edgeObserver = lattice.objects(MemoryEdge.self).observe { change in
                    Task { @MainActor in handleEdgeChange(change) }
                }
                nodeObserver = lattice.objects(Memory.self).observe { change in
                    Task { @MainActor in handleNodeChange(change) }
                }
            }

            // 5. Cluster computation — vector search queries, run off main actor
            // findMemoryClusters returns [[Int64]] (primaryKeys) — convert to UUIDs
            let pkToGlobalId: [Int64: UUID] = Dictionary(
                uniqueKeysWithValues: bgLattice.objects(Memory.self).compactMap { m in
                    guard let pk = m.primaryKey, let gid = m.__globalId else { return nil }
                    return (pk, gid)
                }
            )
            let visibleProjects = await MainActor.run { Set(renderStore.nodes.map(\.project)) }
            var clusters: [[UUID]] = []
            for project in visibleProjects {
                let pkClusters = findMemoryClusters(in: bgLattice, project: project, minClusterSize: 2, neighborLimit: 20).clusters
                for pkCluster in pkClusters {
                    let uuidCluster = pkCluster.compactMap { renderStore.pkToGlobalId[$0] }
                    if uuidCluster.count >= 2 { clusters.append(uuidCluster) }
                }
            }
            let result = clusters
            await MainActor.run {
                renderStore.clusterGroups = result
            }
        }
    }

    /// Insert a batch of nodes into renderStore + simulation. Called from loadData on main actor.
    func insertNodeBatch(_ batch: [NodeData], hiddenProjects: Set<String>,
                                 hiddenRelations: Set<String>, timeFilter: Date?, is3D: Bool) {
        for nd in batch {
            renderStore.allNodes[nd.id] = nd

            let visible = !hiddenProjects.contains(nd.project) &&
                (timeFilter == nil || nd.createdAt <= timeFilter!)
            guard visible else { continue }

            renderStore.nodes.append(nd)
            renderStore.nodeById[nd.id] = nd
            renderStore.visibleNodeIds.insert(nd.id)

            simulation.addNode(nd.id, project: nd.project, topic: nd.topic)
            if is3D {
                simulation3D.addNode(nd.id, project: nd.project, topic: nd.topic)
            }

            // Wire edges where BOTH endpoints now exist (O(degree) via adjacency index)
            for edge in renderStore.edgesByNode[nd.id] ?? [] {
                let otherId = edge.sourceId == nd.id ? edge.targetId : edge.sourceId
                guard renderStore.visibleNodeIds.contains(otherId) else { continue }
                guard !hiddenRelations.contains(edge.relation) else { continue }
                if !renderStore.filteredEdgeIds.contains(edge.id) {
                    renderStore.filteredEdgeIds.insert(edge.id)
                    renderStore.edges.append(edge)
                    simulation.addEdge(from: edge.sourceId, to: edge.targetId)
                    if is3D {
                        simulation3D.addEdge(from: edge.sourceId, to: edge.targetId)
                    }
                }
            }

            // Hub detection
            if let edges = renderStore.edgesByNode[nd.id] {
                for edge in edges where edge.relation == "part_of" && edge.targetId == nd.id {
                    renderStore.hubs.insert(nd.id)
                    break
                }
            }

            // Assign color for previously unseen project
            if renderStore.colorMap[nd.project] == nil {
                if nd.project == "global" {
                    renderStore.colorMap["global"] = .gray
                } else {
                    let idx = renderStore.colorMap.count - (renderStore.colorMap["global"] != nil ? 1 : 0)
                    let color = Self.goldenAngleColor(at: idx)
                    renderStore.colorMap[nd.project] = color
                }
            }
        }
    }

    func handleNodeChange(_ change: CollectionChange) {
        switch change {
        case .insert(let pk):
            // Queue and coalesce — multiple inserts flushed as a single batch
            renderStore.pendingNodeInserts.append(pk)
            if renderStore.pendingNodeFlush == nil {
                renderStore.pendingNodeFlush = Task { @MainActor in
                    await Task.yield()
                    flushPendingNodeInserts()
                    renderStore.pendingNodeFlush = nil
                }
            }

        case .update(let pk):
            guard let memory = lattice.object(Memory.self, primaryKey: pk),
                  let gid = memory.__globalId else { return }
            let old = renderStore.allNodes[gid]
            // Detect recall: lastAccessedAt changed → trigger glow
            if let old, memory.lastAccessedAt > old.lastAccessedAt {
                renderStore.glowingNodes[gid] = Date()
            }
            let newLabel = extractLabel(content: memory.content, topic: memory.topic)
            let structuralChange = old == nil ||
                old!.project != memory.project ||
                old!.topic != memory.topic ||
                old!.importance != memory.importance ||
                old!.label != newLabel
            if !structuralChange {
                renderStore.allNodes[gid]?.lastAccessedAt = memory.lastAccessedAt
                if let idx = renderStore.nodes.firstIndex(where: { $0.id == gid }) {
                    renderStore.nodes[idx].lastAccessedAt = memory.lastAccessedAt
                }
                return
            }
            let node = NodeData(
                id: gid, project: memory.project, topic: memory.topic,
                label: newLabel,
                content: memory.content,
                createdAt: memory.createdAt, lastAccessedAt: memory.lastAccessedAt,
                importance: memory.importance
            )
            renderStore.allNodes[gid] = node
            renderStore.nodeById[gid] = node
            if let idx = renderStore.nodes.firstIndex(where: { $0.id == gid }) {
                renderStore.nodes[idx] = node
            }
            if old?.project != node.project || old?.topic != node.topic {
                recomputeFilteredData()
                rebuildSimulationGraph()
            }

        case .delete(let pk):
            let gid: UUID
            if let cached = renderStore.pkToGlobalId[pk] {
                gid = cached
            } else if let memory = lattice.object(Memory.self, primaryKey: pk),
                      let memGid = memory.__globalId {
                gid = memGid
            } else {
                return
            }
            renderStore.pkToGlobalId.removeValue(forKey: pk)
            if !isInitialLoad && config.notificationsEnabled, let node = renderStore.allNodes[gid] {
                sendMemoryNotification(title: "[\(node.project)] Memory removed", body: node.label, id: gid)
            }
            // Snapshot dying node for fade-out animation before removal
            if let node = renderStore.allNodes[gid], let pos = simulation.positions[gid] {
                renderStore.dyingNodes[gid] = DyingNode(
                    id: gid, position: pos, project: node.project,
                    isHub: renderStore.hubs.contains(gid), importance: node.importance,
                    startTime: Date()
                )
            }
            renderStore.allNodes.removeValue(forKey: gid)
            renderStore.glowingNodes.removeValue(forKey: gid)
            renderStore.newNodeGlows.removeValue(forKey: gid)
            let removedEdgeIds = Set(renderStore.edges.filter { $0.sourceId == gid || $0.targetId == gid }.map(\.id))
            renderStore.filteredEdgeIds.subtract(removedEdgeIds)
            renderStore.visibleNodeIds.remove(gid)
            simulation.removeNode(gid)
            renderStore.nodes.removeAll { $0.id == gid }
            renderStore.nodeById.removeValue(forKey: gid)
            renderStore.edges.removeAll { $0.sourceId == gid || $0.targetId == gid }
            renderStore.hubs.remove(gid)
            renderStore.bumpTopology()
            if !isInitialLoad && config.soundEnabled { Self.removeSound?.play() }
            renderStore.clusterGroups = renderStore.clusterGroups.compactMap { cluster in
                let filtered = cluster.filter { $0 != gid }
                return filtered.count >= 2 ? filtered : nil
            }
        }
    }

    func handleEdgeChange(_ change: CollectionChange) {
        switch change {
        case .insert(let pk):
            renderStore.pendingEdgeInserts.append(pk)
            if renderStore.pendingEdgeFlush == nil {
                renderStore.pendingEdgeFlush = Task { @MainActor in
                    await Task.yield()
                    flushPendingEdgeInserts()
                    renderStore.pendingEdgeFlush = nil
                }
            }

        case .update(let pk):
            guard let edge = lattice.object(MemoryEdge.self, primaryKey: pk) else { return }
            let data = EdgeData(id: pk, sourceId: edge.sourceGlobalId, targetId: edge.targetGlobalId, relation: edge.relation.rawValue)
            renderStore.allEdges[pk] = data
            if let idx = renderStore.edges.firstIndex(where: { $0.id == pk }) {
                renderStore.edges[idx] = data
            }

        case .delete(let pk):
            if let old = renderStore.allEdges[pk] {
                simulation.removeEdge(from: old.sourceId, to: old.targetId)
                renderStore.filteredEdgeIds.remove(pk)
                renderStore.edges.removeAll { $0.id == pk }
                renderStore.edgesByNode[old.sourceId]?.removeAll { $0.id == pk }
                renderStore.edgesByNode[old.targetId]?.removeAll { $0.id == pk }
                renderStore.edgeCountByNode[old.sourceId, default: 1] -= 1
                renderStore.edgeCountByNode[old.targetId, default: 1] -= 1
                if old.relation == "part_of" {
                    let stillHub = renderStore.allEdges.values.contains { $0.id != pk && $0.relation == "part_of" && $0.targetId == old.targetId }
                    if !stillHub { renderStore.hubs.remove(old.targetId) }
                }
            }
            renderStore.allEdges.removeValue(forKey: pk)
        }
    }

    func recomputeFilteredData() {
        renderStore.nodes = renderStore.allNodes.values.filter { node in
            !config.hiddenProjects.contains(node.project) &&
            (debouncedTimeSliderDate == nil || node.createdAt <= debouncedTimeSliderDate!)
        }
        recomputeHubs()
        let nodeIds = Set(renderStore.nodes.map(\.id))
        renderStore.edges = renderStore.allEdges.values.filter { edge in
            nodeIds.contains(edge.sourceId) && nodeIds.contains(edge.targetId) &&
            !config.hiddenRelations.contains(edge.relation)
        }
        renderStore.filteredEdgeIds = Set(renderStore.edges.map(\.id))
        recomputeDerivedData()
        renderStore.bumpTopology()
    }

    func recomputeHubs() {
        var hubs = Set<UUID>()
        for edge in renderStore.allEdges.values where edge.relation == "part_of" {
            hubs.insert(edge.targetId)
        }
        renderStore.hubs = hubs
    }

    func rebuildSimulationGraph() {
        let filtered = renderStore.nodes
        let currentIds = Set(filtered.map(\.id))
        let edgePairs = renderStore.edges.map { ($0.sourceId, $0.targetId) }
        var projectMap: [UUID: String] = [:]
        var topicMap: [UUID: String] = [:]
        for node in filtered {
            projectMap[node.id] = node.project
            topicMap[node.id] = node.topic
        }
        simulation.updateGraph(nodeIds: currentIds, edges: edgePairs, projectForNode: projectMap, topicForNode: topicMap)

        // Also update 3D simulation when in 3D mode
        if config.dimensionMode == .threeD {
            simulation3D.updateGraph(nodeIds: currentIds, edges: edgePairs,
                                     projectForNode: projectMap, topicForNode: topicMap)
        }

        // Mark embedding projection stale if topology changed while in embedding mode
        if config.layoutMode == .embedding {
            projectionTopologyVersion &+= 1
        }
    }

    // MARK: - Batched Node/Edge Insert Flush

    /// Process all queued node inserts as a single batch.
    /// Only @State mutations are the lightweight counters — no body re-evaluation for rendering data.
    func flushPendingNodeInserts() {
        let pks = renderStore.pendingNodeInserts
        renderStore.pendingNodeInserts.removeAll(keepingCapacity: true)
        guard !pks.isEmpty else { return }

        let is3D = config.dimensionMode == .threeD
        let hiddenProjects = config.hiddenProjects
        let hiddenRelations = config.hiddenRelations
        let timeFilter = debouncedTimeSliderDate
        var bumpedTopology = false

        for pk in pks {
            guard let memory = lattice.object(Memory.self, primaryKey: pk),
                  let gid = memory.__globalId else { continue }
            renderStore.pkToGlobalId[pk] = gid
            let node = NodeData(
                id: gid, project: memory.project, topic: memory.topic,
                label: extractLabel(content: memory.content, topic: memory.topic),
                content: memory.content,
                createdAt: memory.createdAt, lastAccessedAt: memory.lastAccessedAt,
                importance: memory.importance
            )
            renderStore.allNodes[gid] = node
            renderStore.newNodeGlows[gid] = Date()
            if !isInitialLoad && config.notificationsEnabled {
                sendMemoryNotification(title: "[\(memory.project)] New memory", body: node.label, id: gid)
            }
            let visible = !hiddenProjects.contains(node.project) &&
                (timeFilter == nil || node.createdAt <= timeFilter!)
            guard visible && !renderStore.visibleNodeIds.contains(gid) else { continue }

            renderStore.nodes.append(node)
            renderStore.nodeById[gid] = node
            renderStore.visibleNodeIds.insert(gid)
            simulation.addNode(gid, project: node.project, topic: node.topic)
            if is3D {
                simulation3D.addNode(gid, project: node.project, topic: node.topic)
            }
            // Add edges for this node from existing edge data (O(degree) via adjacency index)
            let nodeIds = renderStore.visibleNodeIds
            for edge in renderStore.edgesByNode[gid] ?? [] {
                let otherId = edge.sourceId == gid ? edge.targetId : edge.sourceId
                guard nodeIds.contains(otherId),
                      !hiddenRelations.contains(edge.relation) else { continue }
                simulation.addEdge(from: edge.sourceId, to: edge.targetId)
                if is3D {
                    simulation3D.addEdge(from: edge.sourceId, to: edge.targetId)
                }
                if !renderStore.filteredEdgeIds.contains(edge.id) {
                    renderStore.filteredEdgeIds.insert(edge.id)
                    renderStore.edges.append(edge)
                }
                if edge.relation == "part_of" && edge.targetId == gid {
                    renderStore.hubs.insert(gid)
                }
            }
            // Assign color for previously unseen project
            if renderStore.colorMap[node.project] == nil && node.project != "global" {
                let idx = renderStore.colorMap.count - 1
                let color = Self.goldenAngleColor(at: idx)
                renderStore.colorMap[node.project] = color
            }
            bumpedTopology = true
        }
        if bumpedTopology {
            renderStore.bumpTopology()
            if !isInitialLoad && config.soundEnabled { Self.addSound?.play() }
        }
    }

    /// Process all queued edge inserts as a single batch.
    func flushPendingEdgeInserts() {
        let pks = renderStore.pendingEdgeInserts
        renderStore.pendingEdgeInserts.removeAll(keepingCapacity: true)
        guard !pks.isEmpty else { return }

        let is3D = config.dimensionMode == .threeD
        let hiddenRelations = config.hiddenRelations
        var bumpedTopology = false

        for pk in pks {
            guard let edge = lattice.object(MemoryEdge.self, primaryKey: pk) else { continue }
            let data = EdgeData(id: pk, sourceId: edge.sourceGlobalId, targetId: edge.targetGlobalId, relation: edge.relation.rawValue)
            renderStore.allEdges[pk] = data
            renderStore.edgesByNode[data.sourceId, default: []].append(data)
            renderStore.edgesByNode[data.targetId, default: []].append(data)
            renderStore.edgeCountByNode[data.sourceId, default: 0] += 1
            renderStore.edgeCountByNode[data.targetId, default: 0] += 1
            let nodeIds = renderStore.visibleNodeIds
            if nodeIds.contains(data.sourceId) && nodeIds.contains(data.targetId) &&
               !hiddenRelations.contains(data.relation) {
                renderStore.filteredEdgeIds.insert(data.id)
                renderStore.edges.append(data)
                simulation.addEdge(from: data.sourceId, to: data.targetId)
                if is3D {
                    simulation3D.addEdge(from: data.sourceId, to: data.targetId)
                }
            }
            if data.relation == "part_of" {
                renderStore.hubs.insert(data.targetId)
                bumpedTopology = true
            }
        }
        if bumpedTopology {
            renderStore.bumpTopology()
        }
    }

    func sendMemoryNotification(title: String, body: String, id: UUID) {
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        content.userInfo = ["memoryId": id]
        let request = UNNotificationRequest(
            identifier: "memory-\(id)-\(UUID().uuidString)",
            content: content,
            trigger: nil
        )
        Task {
            try? await UNUserNotificationCenter.current().add(request)
        }
    }
}
