import SwiftUI
import Lattice
import Combine
import ClaudeMemoryLib
import UniformTypeIdentifiers
import AppKit

typealias MemoryEdge = ClaudeMemoryLib.Edge

// MARK: - Display data

struct TopicGroupInfo: Equatable {
    let topic: String
    let project: String
    let ids: [Int64]
}

// MARK: - Viewport state (isolated from query data to avoid expensive recomputation on pan/zoom)

@Observable
@MainActor
final class ViewportState {
    var scale: CGFloat = 1.0
    var offset: CGPoint = .zero
    var draggedNode: Int64?
    var hoveredNode: Int64?

    // Animated pan target — lerped each frame by the Canvas timer
    private var targetOffset: CGPoint?
    private let panSpeed: CGFloat = 0.12
    var isAnimatingPan: Bool { targetOffset != nil }

    func animateTo(offset target: CGPoint) {
        targetOffset = target
    }

    /// Call each frame to interpolate toward the target. Returns true while animating.
    @discardableResult
    func tickPan() -> Bool {
        guard let target = targetOffset else { return false }
        let dx = target.x - offset.x
        let dy = target.y - offset.y
        if abs(dx) < 0.5 && abs(dy) < 0.5 {
            offset = target
            targetOffset = nil
            return false
        }
        offset = CGPoint(
            x: offset.x + dx * panSpeed,
            y: offset.y + dy * panSpeed
        )
        return true
    }
}

// MARK: - Root view (owns data store, simulation, and state)

struct GraphView: View {
    @Environment(\.lattice) private var lattice

    // Lightweight data store — replaces @LatticeQuery with manual observation
    @State private var allNodes: [Int64: NodeData] = [:]
    @State private var allEdges: [Int64: EdgeData] = [:]
    @State private var nodeObserver: AnyCancellable?
    @State private var edgeObserver: AnyCancellable?
    @State private var glowingNodes: [Int64: Date] = [:]
    @State private var newNodes: [Int64: Date] = [:]
    @State private var dyingNodes: [Int64: DyingNode] = [:]

    // Cached filtered views — recomputed on structural changes or filter changes
    @State private var cachedFilteredNodes: [NodeData] = []
    @State private var cachedHubs: Set<Int64> = []
    @State private var cachedFilteredEdges: [EdgeData] = []

    // Cached derived data — recomputed only when filtered data changes (not every body eval)
    @State private var cachedProjectColorMap: [String: Color] = [:]
    @State private var cachedTopicGroups: [TopicGroupInfo] = []
    @State private var cachedVisibleNodeIds: Set<Int64> = []
    @State private var cachedRelationCounts: [(key: String, value: Int)] = []
    @State private var cachedEdgeCountByNode: [Int64: Int] = [:]

    @State private var simulation = ForceSimulation()
    @State private var viewport = ViewportState()
    @State private var hiddenProjects: Set<String> = []
    @State private var hiddenRelations: Set<String> = []
    @State private var selectedMemoryId: Int64?
    @State private var scrollMonitor: Any?

    // Search results (owned by SearchBarView, written via bindings)
    @State private var searchMatchIds: Set<Int64> = []
    @State private var isSearchActive: Bool = false

    // Cluster groups (expensive — recomputed only when data or filters change)
    @State private var clusterGroups: [[Int64]] = []

    // Time slider (debounced like search)
    @State private var timeSliderDate: Date?
    @State private var debouncedTimeSliderDate: Date?
    @State private var isTimelinePlaying: Bool = false

    // Minimap PiP
    @State private var minimapDetached: Bool = false
    @State private var minimapPanel = MinimapPanelController()

    // Glow cleanup timer (1s interval — removes entries older than 4.3s)
    private let glowTimer = Timer.publish(every: 1, on: .main, in: .common).autoconnect()

    // MARK: - Derived Data (cached, recomputed on structural changes)

    /// Recompute all derived data from current filtered nodes/edges. Call after any structural change.
    private func recomputeDerivedData() {
        // Color map
        let goldenAngle = 0.381966011250105
        let projects = uniqueProjects()
        var map: [String: Color] = ["global": .gray]
        for (i, project) in projects.enumerated() {
            if project == "global" { continue }
            let hue = Double(i) * goldenAngle
            let h = hue - hue.rounded(.down)
            map[project] = Color(hue: h, saturation: 0.65, brightness: 0.9)
        }
        cachedProjectColorMap = map

        // Visible node IDs
        cachedVisibleNodeIds = Set(cachedFilteredNodes.map(\.id))

        // Relation counts (uses all edges, not filtered — so hidden relations still show in UI)
        let nodeIds = cachedVisibleNodeIds
        var counts: [String: Int] = [:]
        for edge in allEdges.values {
            guard nodeIds.contains(edge.sourceId), nodeIds.contains(edge.targetId) else { continue }
            counts[edge.relation, default: 0] += 1
        }
        cachedRelationCounts = counts.sorted(by: { $0.key < $1.key })

        // Topic groups
        var groups: [String: (topic: String, project: String, ids: [Int64])] = [:]
        for node in cachedFilteredNodes {
            guard node.topic != "general", node.topic != "episode" else { continue }
            let key = "\(node.project)|\(node.topic)"
            var entry = groups[key] ?? (topic: node.topic, project: node.project, ids: [])
            entry.ids.append(node.id)
            groups[key] = entry
        }
        cachedTopicGroups = groups.values
            .filter { $0.ids.count >= 2 }
            .map { TopicGroupInfo(topic: $0.topic, project: $0.project, ids: $0.ids) }

        // Per-node edge counts (for detail panel)
        var edgeCounts: [Int64: Int] = [:]
        for edge in allEdges.values {
            edgeCounts[edge.sourceId, default: 0] += 1
            edgeCounts[edge.targetId, default: 0] += 1
        }
        cachedEdgeCountByNode = edgeCounts
    }

    private func recomputeClusters() {
        let visibleProjects = Set(cachedFilteredNodes.map(\.project))
        var all: [[Int64]] = []
        for project in visibleProjects {
            all.append(contentsOf: findMemoryClusters(in: lattice, project: project, minClusterSize: 2).clusters)
        }
        clusterGroups = all
    }

    // MARK: - Date range for time slider (filtered by visible projects)

    private var dateRange: (earliest: Date, latest: Date) {
        var earliest = Date.distantFuture
        var latest = Date.distantPast
        for node in cachedFilteredNodes {
            if node.createdAt < earliest { earliest = node.createdAt }
            if node.createdAt > latest { latest = node.createdAt }
        }
        if earliest > latest { earliest = Date(); latest = Date() }
        return (earliest, latest)
    }

    // MARK: - Data Loading & Observation

    private func loadData() {
        // One-shot population — extract lightweight primitives from Lattice models
        for m in lattice.objects(Memory.self) {
            guard let pk = m.primaryKey else { continue }
            allNodes[pk] = NodeData(
                id: pk, project: m.project, topic: m.topic,
                label: extractLabel(content: m.content, topic: m.topic),
                createdAt: m.createdAt, lastAccessedAt: m.lastAccessedAt,
                importance: m.importance
            )
        }
        for e in lattice.objects(MemoryEdge.self) {
            guard let pk = e.primaryKey else { continue }
            allEdges[pk] = EdgeData(id: pk, sourceId: e.sourceId, targetId: e.targetId, relation: e.relation)
        }

        recomputeFilteredData()
        rebuildSimulationGraph()
        recomputeClusters()

        // Fine-grained collection observers for cross-process changes
        edgeObserver = lattice.objects(MemoryEdge.self).observe { change in
            Task { @MainActor in handleEdgeChange(change) }
        }
        nodeObserver = lattice.objects(Memory.self).observe { change in
            Task { @MainActor in handleNodeChange(change) }
        }
    }

    private func handleNodeChange(_ change: CollectionChange) {
        switch change {
        case .insert(let pk):
            guard let memory = lattice.object(Memory.self, primaryKey: pk) else { return }
            let node = NodeData(
                id: pk, project: memory.project, topic: memory.topic,
                label: extractLabel(content: memory.content, topic: memory.topic),
                createdAt: memory.createdAt, lastAccessedAt: memory.lastAccessedAt,
                importance: memory.importance
            )
            allNodes[pk] = node
            newNodes[pk] = Date()
            let visible = !hiddenProjects.contains(node.project) &&
                (debouncedTimeSliderDate == nil || node.createdAt <= debouncedTimeSliderDate!)
            if visible {
                cachedFilteredNodes.append(node)
                cachedVisibleNodeIds.insert(pk)
                simulation.addNode(pk, project: node.project, topic: node.topic)
                // Add edges for this node from existing edge data
                let nodeIds = cachedVisibleNodeIds
                for edge in allEdges.values {
                    if (edge.sourceId == pk || edge.targetId == pk) &&
                       nodeIds.contains(edge.sourceId) && nodeIds.contains(edge.targetId) &&
                       !hiddenRelations.contains(edge.relation) {
                        simulation.addEdge(from: edge.sourceId, to: edge.targetId)
                        if !cachedFilteredEdges.contains(where: { $0.id == edge.id }) {
                            cachedFilteredEdges.append(edge)
                        }
                    }
                }
                // Check if this new node is a hub target (any part_of edges point to it)
                for edge in allEdges.values where edge.relation == "part_of" && edge.targetId == pk {
                    cachedHubs.insert(pk)
                    break
                }
            }

        case .update(let pk):
            guard let memory = lattice.object(Memory.self, primaryKey: pk) else { return }
            let old = allNodes[pk]
            // Detect recall: lastAccessedAt changed → trigger glow
            if let old, memory.lastAccessedAt > old.lastAccessedAt {
                glowingNodes[pk] = Date()
            }
            // Only update dict if structural properties changed (avoids unnecessary body re-eval)
            let newLabel = extractLabel(content: memory.content, topic: memory.topic)
            let structuralChange = old == nil ||
                old!.project != memory.project ||
                old!.topic != memory.topic ||
                old!.importance != memory.importance ||
                old!.label != newLabel
            if !structuralChange {
                // Still update lastAccessedAt for recency visualization (no recompute needed)
                allNodes[pk]?.lastAccessedAt = memory.lastAccessedAt
                if let idx = cachedFilteredNodes.firstIndex(where: { $0.id == pk }) {
                    cachedFilteredNodes[idx].lastAccessedAt = memory.lastAccessedAt
                }
                return
            }
            let node = NodeData(
                id: pk, project: memory.project, topic: memory.topic,
                label: newLabel,
                createdAt: memory.createdAt, lastAccessedAt: memory.lastAccessedAt,
                importance: memory.importance
            )
            allNodes[pk] = node
            if let idx = cachedFilteredNodes.firstIndex(where: { $0.id == pk }) {
                cachedFilteredNodes[idx] = node
            }
            if old!.project != node.project || old!.topic != node.topic {
                recomputeFilteredData()
                rebuildSimulationGraph()
            }

        case .delete(let pk):
            // Snapshot dying node for fade-out animation before removal
            if let node = allNodes[pk], let pos = simulation.positions[pk] {
                dyingNodes[pk] = DyingNode(
                    id: pk, position: pos, project: node.project,
                    isHub: cachedHubs.contains(pk), importance: node.importance,
                    startTime: Date()
                )
            }
            allNodes.removeValue(forKey: pk)
            glowingNodes.removeValue(forKey: pk)
            newNodes.removeValue(forKey: pk)
            cachedFilteredNodes.removeAll { $0.id == pk }
            cachedFilteredEdges.removeAll { $0.sourceId == pk || $0.targetId == pk }
            cachedVisibleNodeIds.remove(pk)
            simulation.removeNode(pk)
            cachedHubs.remove(pk)
            // Remove from cluster groups surgically
            clusterGroups = clusterGroups.compactMap { cluster in
                let filtered = cluster.filter { $0 != pk }
                return filtered.count >= 2 ? filtered : nil
            }
        }
    }

    private func handleEdgeChange(_ change: CollectionChange) {
        switch change {
        case .insert(let pk):
            guard let edge = lattice.object(MemoryEdge.self, primaryKey: pk) else { return }
            let data = EdgeData(id: pk, sourceId: edge.sourceId, targetId: edge.targetId, relation: edge.relation)
            allEdges[pk] = data
            cachedEdgeCountByNode[data.sourceId, default: 0] += 1
            cachedEdgeCountByNode[data.targetId, default: 0] += 1
            let nodeIds = cachedVisibleNodeIds
            if nodeIds.contains(data.sourceId) && nodeIds.contains(data.targetId) &&
               !hiddenRelations.contains(data.relation) {
                cachedFilteredEdges.append(data)
                simulation.addEdge(from: data.sourceId, to: data.targetId)
            }
            if data.relation == "part_of" { cachedHubs.insert(data.targetId) }

        case .update(let pk):
            guard let edge = lattice.object(MemoryEdge.self, primaryKey: pk) else { return }
            let data = EdgeData(id: pk, sourceId: edge.sourceId, targetId: edge.targetId, relation: edge.relation)
            allEdges[pk] = data
            if let idx = cachedFilteredEdges.firstIndex(where: { $0.id == pk }) {
                cachedFilteredEdges[idx] = data
            }

        case .delete(let pk):
            if let old = allEdges[pk] {
                simulation.removeEdge(from: old.sourceId, to: old.targetId)
                cachedFilteredEdges.removeAll { $0.id == pk }
                cachedEdgeCountByNode[old.sourceId, default: 1] -= 1
                cachedEdgeCountByNode[old.targetId, default: 1] -= 1
                if old.relation == "part_of" {
                    let stillHub = allEdges.values.contains { $0.id != pk && $0.relation == "part_of" && $0.targetId == old.targetId }
                    if !stillHub { cachedHubs.remove(old.targetId) }
                }
            }
            allEdges.removeValue(forKey: pk)
        }
    }

    private func recomputeFilteredData() {
        cachedFilteredNodes = allNodes.values.filter { node in
            !hiddenProjects.contains(node.project) &&
            (debouncedTimeSliderDate == nil || node.createdAt <= debouncedTimeSliderDate!)
        }
        recomputeHubs()
        let nodeIds = Set(cachedFilteredNodes.map(\.id))
        cachedFilteredEdges = allEdges.values.filter { edge in
            nodeIds.contains(edge.sourceId) && nodeIds.contains(edge.targetId) &&
            !hiddenRelations.contains(edge.relation)
        }
        recomputeDerivedData()
    }

    private func recomputeHubs() {
        var hubs = Set<Int64>()
        for edge in allEdges.values where edge.relation == "part_of" {
            hubs.insert(edge.targetId)
        }
        cachedHubs = hubs
    }

    private func rebuildSimulationGraph() {
        let filtered = cachedFilteredNodes
        let currentIds = Set(filtered.map(\.id))
        let edgePairs = cachedFilteredEdges.map { ($0.sourceId, $0.targetId) }
        var projectMap: [Int64: String] = [:]
        var topicMap: [Int64: String] = [:]
        for node in filtered {
            projectMap[node.id] = node.project
            topicMap[node.id] = node.topic
        }
//        Task {
            simulation.updateGraph(nodeIds: currentIds, edges: edgePairs, projectForNode: projectMap, topicForNode: topicMap)
//        }
    }

    // MARK: - Body

    var body: some View {
        GeometryReader { geo in
            graphContent(size: geo.size)
                .onChange(of: geo.size, initial: true) { _, newSize in
                    simulation.center = CGPoint(x: newSize.width / 2, y: newSize.height / 2)
                }
                .onAppear {
                    loadData()
                    installScrollMonitor()
                }
                .onDisappear {
                    removeScrollMonitor()
                    minimapPanel.dismiss()
                }
                .onPreferenceChange(InlineMinimapFrameKey.self) { frame in
                    if frame.width > 0 { minimapPanel.inlineFrame = frame }
                }
                .onChange(of: selectedMemoryId) { oldId, newId in
                    handleSelectionChange(oldId: oldId, newId: newId, viewSize: geo.size)
                }
                .modifier(keyboardShortcuts)
                .task(id: timeSliderDate) {
                    try? await Task.sleep(for: .milliseconds(80))
                    debouncedTimeSliderDate = timeSliderDate
                }
                .onChange(of: allNodes.count) { oldCount, newCount in
                    guard newCount != oldCount else { return }
                    guard UserDefaults.standard.bool(forKey: "soundEnabled") else { return }
                    let sound = newCount > oldCount
                        ? "/System/Library/Sounds/Funk.aiff"
                        : "/System/Library/Sounds/Bottle.aiff"
                    NSSound(contentsOf: URL(fileURLWithPath: sound), byReference: true)?.play()
                }
                // hiddenProjects changes handled surgically in toggleProject()
                .onChange(of: hiddenRelations) { _, _ in
                    recomputeFilteredData()
                    rebuildSimulationGraph()
                }
                .onChange(of: debouncedTimeSliderDate) { _, _ in
                    recomputeFilteredData()
                    rebuildSimulationGraph()
                }
                .onReceive(glowTimer) { _ in
                    let now = Date()
                    if !glowingNodes.isEmpty {
                        let filtered = glowingNodes.filter { now.timeIntervalSince($0.value) < 4.3 }
                        if filtered.count != glowingNodes.count {
                            glowingNodes = filtered
                        }
                    }
                    if !newNodes.isEmpty {
                        let filtered = newNodes.filter { now.timeIntervalSince($0.value) < 5.5 }
                        if filtered.count != newNodes.count {
                            newNodes = filtered
                        }
                    }
                    if !dyingNodes.isEmpty {
                        let filtered = dyingNodes.filter { now.timeIntervalSince($0.value.startTime) < 3.0 }
                        if filtered.count != dyingNodes.count {
                            dyingNodes = filtered
                        }
                    }
                }
        }
    }

    private var keyboardShortcuts: some ViewModifier {
        GraphKeyboardShortcuts(
            selectedMemoryId: $selectedMemoryId,
            viewport: viewport,
            cycleConnectedNode: cycleConnectedNode,
            exportToPNG: exportToPNG
        )
    }

    // MARK: - Body Helpers

    @ViewBuilder
    private func graphContent(size: CGSize) -> some View {
        let colorMap = cachedProjectColorMap
        ZStack {
            Color(red: 0.051, green: 0.067, blue: 0.09).ignoresSafeArea()

            graphCanvas(colorMap: colorMap)
            graphOverlays(size: size, colorMap: colorMap)
            detailPanel(size: size, colorMap: colorMap)
        }
    }

    private func graphCanvas(colorMap: [String: Color]) -> some View {
        GraphCanvas(
            simulation: simulation,
            nodes: cachedFilteredNodes,
            edges: cachedFilteredEdges,
            hubs: cachedHubs,
            glowingNodes: glowingNodes,
            newNodes: newNodes,
            dyingNodes: dyingNodes,
            searchMatchIds: searchMatchIds,
            isSearchActive: isSearchActive,
            viewport: viewport,
            selectedNode: $selectedMemoryId,
            clusters: clusterGroups,
            topicGroups: cachedTopicGroups,
            colorMap: colorMap
        )
    }

    @ViewBuilder
    private func graphOverlays(size: CGSize, colorMap: [String: Color]) -> some View {
        // Right column: stats + activity log
        VStack(alignment: .trailing, spacing: 8) {
            StatsOverlay(
                visibleMemoryCount: cachedVisibleNodeIds.count,
                visibleEdgeCount: cachedFilteredEdges.count,
                totalMemories: allNodes.count,
                hiddenProjects: hiddenProjects,
                hiddenRelations: hiddenRelations,
                projects: uniqueProjects(),
                allRelationCounts: cachedRelationCounts,
                toggleProject: toggleProject,
                toggleRelation: toggleRelation,
                colorMap: colorMap
            )

            if selectedMemoryId == nil {
                ActivityLogPanel(
                    nodes: cachedFilteredNodes,
                    colorMap: colorMap,
                    onSelect: { id in selectedMemoryId = id }
                )
                .transition(.opacity)
            }

            Spacer()
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .trailing)
        .animation(.easeInOut(duration: 0.25), value: selectedMemoryId == nil)

        VStack {
            HStack {
                SearchBarView(
                    matchIds: $searchMatchIds,
                    isActive: $isSearchActive,
                    lattice: lattice
                )
                SoundToggleButton()
                Spacer()
            }
            .padding(12)
            Spacer()
        }

        // Minimap — inline when docked, floating NSPanel when detached
        if !minimapDetached {
            VStack {
                Spacer()
                HStack {
                    MinimapView(
                        filteredNodes: cachedFilteredNodes,
                        hubs: cachedHubs,
                        glowingNodes: glowingNodes,
                        newNodes: newNodes,
                        dyingNodes: dyingNodes,
                        simulation: simulation,
                        viewport: viewport,
                        viewportSize: size,
                        colorMap: colorMap,
                        pipAction: { minimapDetached = true }
                    )
                    .background(GeometryReader { geo in
                        Color.clear.preference(
                            key: InlineMinimapFrameKey.self,
                            value: geo.frame(in: .global)
                        )
                    })
                    .padding(12)
                    Spacer()
                }
                .padding(.bottom, 44)
            }
        }
        MinimapPanelBridge(
            isDetached: minimapDetached,
            panel: minimapPanel,
            content: floatingMinimap(size: size, colorMap: colorMap),
            onDismiss: { minimapDetached = false }
        )
        .frame(width: 0, height: 0)

        VStack {
            Spacer()
            let range = dateRange
            TimeSliderBar(
                earliestDate: range.earliest,
                latestDate: range.latest,
                sliderDate: $timeSliderDate,
                isPlaying: $isTimelinePlaying
            )
        }
    }

    @ViewBuilder
    private func detailPanel(size: CGSize, colorMap: [String: Color]) -> some View {
        if let memory = selectedMemory {
            let sid = selectedMemoryId!
            let count = cachedEdgeCountByNode[sid] ?? 0
            MemoryDetailPanel(
                memory: memory,
                connectedCount: count,
                onClose: { selectedMemoryId = nil },
                colorMap: colorMap
            )
            .id(memory.primaryKey)
            .frame(maxWidth: min(400, size.width * 0.35), maxHeight: min(500, size.height * 0.7))
            .frame(maxWidth: .infinity, alignment: .trailing)
            .padding(24)
            .compositingGroup()
            .transition(.move(edge: .trailing).combined(with: .opacity))
            .animation(.spring(duration: 0.4, bounce: 0.2), value: selectedMemoryId)
        }
    }

    private func floatingMinimap(size: CGSize, colorMap: [String: Color]) -> some View {
        MinimapView(
            filteredNodes: cachedFilteredNodes,
            hubs: cachedHubs,
            glowingNodes: glowingNodes,
            newNodes: newNodes,
            dyingNodes: dyingNodes,
            simulation: simulation,
            viewport: viewport,
            viewportSize: size,
            colorMap: colorMap,
            pipAction: { minimapPanel.animatedDismiss { minimapDetached = false } },
            isFloating: true
        )
    }

    private func installScrollMonitor() {
        scrollMonitor = NSEvent.addLocalMonitorForEvents(matching: .scrollWheel) { event in
            if let window = event.window {
                let loc = event.locationInWindow
                let rightEdge = window.frame.width
                // Let the activity log or detail panel scroll normally
                if loc.x > rightEdge - 240 { return event }
                if selectedMemoryId != nil && loc.x > rightEdge * 0.6 { return event }
            }
            // Two-finger scroll / trackpad → pan (consume event to prevent ScrollView conflicts)
            viewport.offset = CGPoint(
                x: viewport.offset.x + event.scrollingDeltaX,
                y: viewport.offset.y + event.scrollingDeltaY
            )
            return nil
        }
    }

    private func removeScrollMonitor() {
        if let monitor = scrollMonitor {
            NSEvent.removeMonitor(monitor)
            scrollMonitor = nil
        }
    }

    private func handleSelectionChange(oldId: Int64?, newId: Int64?, viewSize: CGSize) {
        guard let new = newId,
              let worldPos = simulation.positions[new] else { return }

        // Where the node currently sits on screen
        let screenX = worldPos.x * viewport.scale + viewport.offset.x
        let screenY = worldPos.y * viewport.scale + viewport.offset.y

        // Visible region is the left ~60% (right 40% is the detail panel)
        let safeRight = viewSize.width * 0.58
        let margin: CGFloat = 60
        let safeLeft = margin
        let safeTop = margin
        let safeBottom = viewSize.height - margin

        // Only pan if the node is outside the safe region
        var dx: CGFloat = 0, dy: CGFloat = 0
        if screenX > safeRight { dx = safeRight - screenX - margin }
        else if screenX < safeLeft { dx = safeLeft - screenX + margin }
        if screenY > safeBottom { dy = safeBottom - screenY - margin }
        else if screenY < safeTop { dy = safeTop - screenY + margin }

        if dx != 0 || dy != 0 {
            viewport.animateTo(offset: CGPoint(
                x: viewport.offset.x + dx,
                y: viewport.offset.y + dy
            ))
        }
    }

    // MARK: - Selected Memory (single Lattice lookup for detail panel)

    private var selectedMemory: Memory? {
        guard let id = selectedMemoryId else { return nil }
        return lattice.object(Memory.self, primaryKey: id)
    }

    // MARK: - Helpers

    private func toggleProject(_ project: String) {
        if hiddenProjects.contains(project) {
            hiddenProjects.remove(project)
            showProject(project)
        } else {
            hiddenProjects.insert(project)
            hideProject(project)
        }
    }

    /// Surgically remove a project's nodes/edges from filtered data and simulation.
    private func hideProject(_ project: String) {
        let removedIds = Set(cachedFilteredNodes.filter { $0.project == project }.map(\.id))
        guard !removedIds.isEmpty else { return }
        cachedFilteredNodes.removeAll { $0.project == project }
        cachedFilteredEdges.removeAll { removedIds.contains($0.sourceId) || removedIds.contains($0.targetId) }
        cachedVisibleNodeIds.subtract(removedIds)
        for id in removedIds { simulation.removeNode(id) }
        clusterGroups = clusterGroups.compactMap { cluster in
            let filtered = cluster.filter { !removedIds.contains($0) }
            return filtered.count >= 2 ? filtered : nil
        }
        recomputeDerivedData()
    }

    /// Surgically add a project's nodes/edges back into filtered data and simulation.
    private func showProject(_ project: String) {
        let newNodes = allNodes.values.filter {
            $0.project == project &&
            (debouncedTimeSliderDate == nil || $0.createdAt <= debouncedTimeSliderDate!)
        }
        guard !newNodes.isEmpty else { return }
        cachedFilteredNodes.append(contentsOf: newNodes)
        let newIds = Set(newNodes.map(\.id))
        cachedVisibleNodeIds.formUnion(newIds)

        let allVisibleIds = cachedVisibleNodeIds
        let newEdges = allEdges.values.filter { edge in
            allVisibleIds.contains(edge.sourceId) && allVisibleIds.contains(edge.targetId) &&
            !hiddenRelations.contains(edge.relation) &&
            !cachedFilteredEdges.contains(where: { $0.id == edge.id })
        }
        cachedFilteredEdges.append(contentsOf: newEdges)

        for node in newNodes {
            simulation.addNode(node.id, project: node.project, topic: node.topic)
        }
        for edge in newEdges {
            simulation.addEdge(from: edge.sourceId, to: edge.targetId)
        }

        recomputeHubs()
        recomputeDerivedData()
        clusterGroups.append(contentsOf:
            findMemoryClusters(in: lattice, project: project, minClusterSize: 2).clusters
        )
    }

    private func toggleRelation(_ relation: String) {
        if hiddenRelations.contains(relation) {
            hiddenRelations.remove(relation)
        } else {
            hiddenRelations.insert(relation)
        }
    }

    private func uniqueProjects() -> [String] {
        var seen = Set<String>()
        var result: [String] = []
        for node in allNodes.values {
            if seen.insert(node.project).inserted {
                result.append(node.project)
            }
        }
        return result.sorted()
    }

    private func cycleConnectedNode() {
        guard let sel = selectedMemoryId else { return }
        var connected: [Int64] = []
        for edge in allEdges.values {
            if edge.sourceId == sel { connected.append(edge.targetId) }
            if edge.targetId == sel { connected.append(edge.sourceId) }
        }
        guard !connected.isEmpty else { return }
        if let idx = connected.firstIndex(of: sel) {
            selectedMemoryId = connected[(idx + 1) % connected.count]
        } else {
            selectedMemoryId = connected[0]
        }
    }

    private func exportToPNG() {
        guard let window = NSApp.keyWindow,
              let contentView = window.contentView else { return }

        let rep = contentView.bitmapImageRepForCachingDisplay(in: contentView.bounds)
        guard let rep else { return }
        contentView.cacheDisplay(in: contentView.bounds, to: rep)

        guard let pngData = rep.representation(using: .png, properties: [:]) else { return }

        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = "memory-graph.png"
        panel.begin { response in
            if response == .OK, let url = panel.url {
                try? pngData.write(to: url)
            }
        }
    }

    /// Look up a project's color from a pre-built map. Falls back to gray for unknown projects.
    static func projectColor(for project: String, in colorMap: [String: Color]) -> Color {
        colorMap[project] ?? .gray
    }

}

// MARK: - Sound Toggle (isolated to avoid re-rendering GraphView)

struct SoundToggleButton: View {
    @AppStorage("soundEnabled") private var soundEnabled = false

    var body: some View {
        Button {
            soundEnabled.toggle()
        } label: {
            Image(systemName: soundEnabled ? "speaker.wave.2.fill" : "speaker.slash.fill")
                .font(.system(size: 12))
                .foregroundStyle(.white.opacity(soundEnabled ? 0.7 : 0.3))
        }
        .buttonStyle(.plain)
        .help(soundEnabled ? "Mute notifications" : "Unmute notifications")
    }
}

// MARK: - Keyboard Shortcuts Modifier

struct GraphKeyboardShortcuts: ViewModifier {
    @Binding var selectedMemoryId: Int64?
    let viewport: ViewportState
    let cycleConnectedNode: () -> Void
    let exportToPNG: () -> Void

    func body(content: Content) -> some View {
        content
            .onKeyPress(.escape) {
                selectedMemoryId = nil
                return .handled
            }
            .onKeyPress(.tab) {
                cycleConnectedNode()
                return .handled
            }
            .onKeyPress("+") {
                viewport.scale = min(3.0, viewport.scale * 1.1)
                return .handled
            }
            .onKeyPress("=") {
                viewport.scale = min(3.0, viewport.scale * 1.1)
                return .handled
            }
            .onKeyPress("-") {
                viewport.scale = max(0.2, viewport.scale / 1.1)
                return .handled
            }
            .onKeyPress(characters: .alphanumerics, phases: .down) { keyPress in
                if keyPress.characters == "E" && keyPress.modifiers == [.command, .shift] {
                    exportToPNG()
                    return .handled
                }
                return .ignored
            }
    }
}

// MARK: - Search Bar (isolated view — owns its own text state to avoid invalidating GraphView on each keystroke)

struct SearchBarView: View {
    @Binding var matchIds: Set<Int64>
    @Binding var isActive: Bool
    let lattice: Lattice

    @State private var text: String = ""
    @FocusState private var isFocused: Bool

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.white.opacity(0.4))
                .font(.system(size: 12))
            TextField("Search memories…", text: $text)
                .textFieldStyle(.plain)
                .font(.system(size: 12, design: .monospaced))
                .foregroundStyle(.white.opacity(0.9))
                .focused($isFocused)
                .frame(width: 180)
                .onKeyPress(.escape) {
                    text = ""
                    matchIds = []
                    isActive = false
                    isFocused = false
                    return .handled
                }
            if !text.isEmpty {
                Button {
                    text = ""
                    matchIds = []
                    isActive = false
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.white.opacity(0.4))
                        .font(.system(size: 11))
                }
                .buttonStyle(.plain)

                let matchCount = matchIds.count
                Text("\(matchCount) match\(matchCount == 1 ? "" : "es")")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.4))
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color(red: 0.08, green: 0.1, blue: 0.14))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .strokeBorder(.white.opacity(isFocused ? 0.3 : 0.1), lineWidth: 1)
                )
        )
        .task(id: text) {
            let query = text
            guard !query.isEmpty else {
                matchIds = []
                isActive = false
                return
            }
            isActive = true
            try? await Task.sleep(for: .milliseconds(150))
            guard !Task.isCancelled else { return }
            let terms = query.split(separator: " ").map(String.init)
            guard !terms.isEmpty else { matchIds = []; return }
            let ftsQuery: TextQuery = .raw(terms.map { "\($0)*" }.joined(separator: " OR "))
            var ids = Set<Int64>()
            for match in lattice.objects(Memory.self).matching(ftsQuery, on: \.content, limit: 500) {
                if let pk = match.object.primaryKey { ids.insert(pk) }
            }
            matchIds = ids
        }
    }
}
