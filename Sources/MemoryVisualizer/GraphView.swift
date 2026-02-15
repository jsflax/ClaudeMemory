import SwiftUI
import Lattice
import ClaudeMemoryLib
import UniformTypeIdentifiers

typealias MemoryEdge = ClaudeMemoryLib.Edge

// MARK: - Pre-computed display data

struct NodeInfo: Equatable {
    let id: Int64
    let label: String
    let project: String
    let importance: Int
    let isHub: Bool
    let createdAt: Date
}

struct EdgeInfo: Equatable {
    let sourceId: Int64
    let targetId: Int64
    let relation: String
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

// MARK: - Root view (owns queries, computes display data, manages state)

struct GraphView: View {
    @LatticeQuery<Memory>(sort: \Memory.createdAt) var memories: TableResults<Memory>
    @LatticeQuery<MemoryEdge>(sort: \MemoryEdge.createdAt) var edges: TableResults<MemoryEdge>
    @Environment(\.lattice) private var lattice

    @State private var simulation = ForceSimulation()
    @State private var viewport = ViewportState()
    @State private var hiddenProjects: Set<String> = []
    @State private var hiddenRelations: Set<String> = []
    @State private var selectedMemoryId: Int64?
    @State private var scrollMonitor: Any?

    // Search (debounced, FTS5-backed)
    @State private var searchText: String = ""
    @State private var searchMatchIds: Set<Int64> = []
    @FocusState private var isSearchFocused: Bool

    // Time slider (debounced like search)
    @State private var timeSliderDate: Date?
    @State private var debouncedTimeSliderDate: Date?
    @State private var isTimelinePlaying: Bool = false

    /// Builds a color map assigning maximally-separated hues to each project using the golden angle.
    /// Sequential indices guarantee minimum ~137.5° hue separation between adjacent projects.
    private var projectColorMap: [String: Color] {
        let goldenAngle = 0.381966011250105 // (√5 - 1) / 2, conjugate of golden ratio
        let projects = uniqueProjects()
        var map: [String: Color] = ["global": .gray]
        for (i, project) in projects.enumerated() {
            if project == "global" { continue }
            let hue = Double(i) * goldenAngle
            let h = hue - hue.rounded(.down)
            map[project] = Color(hue: h, saturation: 0.65, brightness: 0.9)
        }
        return map
    }

    // MARK: - Filtering Pipeline

    /// Base nodes: filtered by hidden projects and time slider date.
    private var baseNodeInfos: [NodeInfo] {
        let hubIds = computeHubIds()
        return memories.compactMap { memory -> NodeInfo? in
            guard !hiddenProjects.contains(memory.project),
                  let pk = memory.primaryKey else { return nil }
            if let cutoff = debouncedTimeSliderDate, memory.createdAt > cutoff {
                return nil
            }
            return NodeInfo(
                id: pk,
                label: Self.extractLabel(content: memory.content, topic: memory.topic),
                project: memory.project,
                importance: memory.importance,
                isHub: hubIds.contains(pk),
                createdAt: memory.createdAt
            )
        }
    }

    /// Final node list (same as baseNodeInfos — search doesn't hide).
    private var nodeInfos: [NodeInfo] { baseNodeInfos }

    /// All edges between visible nodes (before relation filtering).
    private var allVisibleEdgeInfos: [EdgeInfo] {
        let nodeIds = Set(nodeInfos.map(\.id))
        return edges.compactMap { edge -> EdgeInfo? in
            guard nodeIds.contains(edge.sourceId),
                  nodeIds.contains(edge.targetId) else { return nil }
            return EdgeInfo(sourceId: edge.sourceId, targetId: edge.targetId, relation: edge.relation)
        }
    }

    /// Edges filtered by visible nodes and hidden relations.
    private var edgeInfos: [EdgeInfo] {
        allVisibleEdgeInfos.filter { !hiddenRelations.contains($0.relation) }
    }

    /// Semantic clusters within each project (embedding similarity + Jaccard term overlap).
    private var clusterGroups: [[Int64]] {
        let visibleProjects = Set(memories.compactMap { m -> String? in
            guard !hiddenProjects.contains(m.project) else { return nil }
            return m.project
        })
        var all: [[Int64]] = []
        for project in visibleProjects {
            all.append(contentsOf: findMemoryClusters(in: lattice, project: project, minClusterSize: 2).clusters)
        }
        return all
    }

    // MARK: - Date range for time slider (filtered by visible projects)

    private var dateRange: (earliest: Date, latest: Date) {
        var earliest = Date.distantFuture
        var latest = Date.distantPast
        for memory in memories {
            guard !hiddenProjects.contains(memory.project) else { continue }
            if memory.createdAt < earliest { earliest = memory.createdAt }
            if memory.createdAt > latest { latest = memory.createdAt }
        }
        if earliest > latest { earliest = Date(); latest = Date() }
        return (earliest, latest)
    }

    var body: some View {
        GeometryReader { geo in
            graphContent(size: geo.size)
                .onChange(of: geo.size, initial: true) { _, newSize in
                    simulation.center = CGPoint(x: newSize.width / 2, y: newSize.height / 2)
                }
                .onAppear { installScrollMonitor() }
                .onDisappear { removeScrollMonitor() }
                .onChange(of: selectedMemoryId) { oldId, newId in
                    handleSelectionChange(oldId: oldId, newId: newId, viewSize: geo.size)
                }
                .modifier(keyboardShortcuts)
                .task(id: searchText) {
                    let query = searchText
                    guard !query.isEmpty else {
                        searchMatchIds = []
                        return
                    }
                    try? await Task.sleep(for: .milliseconds(150))
                    guard !Task.isCancelled else { return }
                    let terms = query.split(separator: " ").map(String.init)
                    guard !terms.isEmpty else { searchMatchIds = []; return }
                    // Use prefix matching so "edito" matches "editor"
                    let ftsQuery: TextQuery = .raw(terms.map { "\($0)*" }.joined(separator: " OR "))
                    var ids = Set<Int64>()
                    for match in lattice.objects(Memory.self).matching(ftsQuery, on: \.content, limit: 500) {
                        if let pk = match.object.primaryKey { ids.insert(pk) }
                    }
                    searchMatchIds = ids
                }
                .task(id: timeSliderDate) {
                    try? await Task.sleep(for: .milliseconds(80))
                    debouncedTimeSliderDate = timeSliderDate
                }
        }
    }

    private var keyboardShortcuts: some ViewModifier {
        GraphKeyboardShortcuts(
            selectedMemoryId: $selectedMemoryId,
            searchText: $searchText,
            isSearchFocused: $isSearchFocused,
            viewport: viewport,
            cycleConnectedNode: cycleConnectedNode,
            exportToPNG: exportToPNG
        )
    }

    // MARK: - Body Helpers

    @ViewBuilder
    private func graphContent(size: CGSize) -> some View {
        let nodes = nodeInfos
        let edgeData = edgeInfos
        let matchIds = searchMatchIds
        let isSearching = !searchText.isEmpty
        let colorMap = projectColorMap
        ZStack {
            Color(red: 0.051, green: 0.067, blue: 0.09).ignoresSafeArea()

            graphCanvas(nodes: nodes, edges: edgeData, matchIds: matchIds, isSearching: isSearching, colorMap: colorMap)
            graphOverlays(nodes: nodes, edges: edgeData, size: size, colorMap: colorMap)
            detailPanel(size: size, colorMap: colorMap)
        }
    }

    private func graphCanvas(nodes: [NodeInfo], edges: [EdgeInfo], matchIds: Set<Int64>, isSearching: Bool, colorMap: [String: Color]) -> some View {
        GraphCanvas(
            simulation: simulation,
            nodes: nodes,
            edges: edges,
            searchMatchIds: matchIds,
            isSearchActive: isSearching,
            viewport: viewport,
            selectedNode: $selectedMemoryId,
            clusters: clusterGroups,
            colorMap: colorMap
        )
    }

    @ViewBuilder
    private func graphOverlays(nodes: [NodeInfo], edges: [EdgeInfo], size: CGSize, colorMap: [String: Color]) -> some View {
        StatsOverlay(
            nodes: nodes,
            edgeData: edges,
            totalMemories: memories.count,
            hiddenProjects: hiddenProjects,
            hiddenRelations: hiddenRelations,
            projects: uniqueProjects(),
            allRelationCounts: Dictionary(grouping: allVisibleEdgeInfos, by: \.relation)
                .mapValues(\.count)
                .sorted(by: { $0.key < $1.key }),
            toggleProject: toggleProject,
            toggleRelation: toggleRelation,
            colorMap: colorMap
        )

        VStack {
            HStack {
                searchBar
                Spacer()
            }
            .padding(12)
            Spacer()
        }

        VStack {
            Spacer()
            HStack {
                MinimapView(
                    simulation: simulation,
                    nodes: nodes,
                    viewport: viewport,
                    viewportSize: size,
                    colorMap: colorMap
                )
                .padding(12)
                Spacer()
            }
            .padding(.bottom, 44)
        }

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
            let count = edges.filter { $0.sourceId == sid || $0.targetId == sid }.count
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
            .transition(.move(edge: .trailing).combined(with: .opacity))
            .animation(.spring(duration: 0.4, bounce: 0.2), value: selectedMemoryId)
        }
    }

    private func installScrollMonitor() {
        scrollMonitor = NSEvent.addLocalMonitorForEvents(matching: .scrollWheel) { event in
            // Let the detail panel scroll normally
            if let window = event.window, selectedMemoryId != nil {
                let loc = event.locationInWindow
                if loc.x > window.frame.width * 0.6 { return event }
            }
            // Two-finger scroll / trackpad → pan
            viewport.offset = CGPoint(
                x: viewport.offset.x + event.scrollingDeltaX,
                y: viewport.offset.y + event.scrollingDeltaY
            )
            return event
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

    // MARK: - Search Bar

    private var searchBar: some View {
        HStack(spacing: 8) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.white.opacity(0.4))
                .font(.system(size: 12))
            TextField("Search memories…", text: $searchText)
                .textFieldStyle(.plain)
                .font(.system(size: 12, design: .monospaced))
                .foregroundStyle(.white.opacity(0.9))
                .focused($isSearchFocused)
                .frame(width: 180)
            if !searchText.isEmpty {
                Button {
                    searchText = ""
                    searchMatchIds = []
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.white.opacity(0.4))
                        .font(.system(size: 11))
                }
                .buttonStyle(.plain)

                let matchCount = searchMatchIds.count
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
                        .strokeBorder(.white.opacity(isSearchFocused ? 0.3 : 0.1), lineWidth: 1)
                )
        )
    }

    // MARK: - Selected Memory (live)

    private var selectedMemory: Memory? {
        guard let id = selectedMemoryId else { return nil }
        return memories.first(where: { $0.primaryKey == id })
    }

    // MARK: - Helpers

    private func toggleProject(_ project: String) {
        if hiddenProjects.contains(project) {
            hiddenProjects.remove(project)
        } else {
            hiddenProjects.insert(project)
        }
    }

    private func toggleRelation(_ relation: String) {
        if hiddenRelations.contains(relation) {
            hiddenRelations.remove(relation)
        } else {
            hiddenRelations.insert(relation)
        }
    }

    private func computeHubIds() -> Set<Int64> {
        var hubs = Set<Int64>()
        for edge in edges where edge.relation == "part_of" {
            hubs.insert(edge.targetId)
        }
        return hubs
    }

    private func uniqueProjects() -> [String] {
        var seen = Set<String>()
        var result: [String] = []
        for memory in memories {
            if seen.insert(memory.project).inserted {
                result.append(memory.project)
            }
        }
        return result.sorted()
    }

    private func cycleConnectedNode() {
        guard let sel = selectedMemoryId else { return }
        var connected: [Int64] = []
        for edge in edges {
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

    static func extractLabel(content: String, topic: String) -> String {
        if let headerRange = content.range(of: #"^#{1,3}\s+"#, options: .regularExpression) {
            let title = content[headerRange.upperBound...].prefix(while: { $0 != "\n" })
            return truncate(String(title), to: 30)
        }
        if content.hasPrefix("**"), let end = content.dropFirst(2).range(of: "**") {
            let title = content[content.index(content.startIndex, offsetBy: 2)..<end.lowerBound]
            return truncate(String(title), to: 30)
        }
        if let colon = content.firstIndex(of: ":"),
           content.distance(from: content.startIndex, to: colon) < 30 {
            return String(content[..<colon])
        }
        let firstLine = String(content.prefix(while: { $0 != "\n" }))
        if firstLine.count <= 30 { return firstLine }
        if topic != "general" { return "\(topic): \(truncate(firstLine, to: 22))" }
        return truncate(firstLine, to: 30)
    }

    private static func truncate(_ text: String, to maxLen: Int) -> String {
        if text.count <= maxLen { return text }
        return String(text.prefix(maxLen - 1)) + "…"
    }
}

// MARK: - Keyboard Shortcuts Modifier

struct GraphKeyboardShortcuts: ViewModifier {
    @Binding var selectedMemoryId: Int64?
    @Binding var searchText: String
    var isSearchFocused: FocusState<Bool>.Binding
    let viewport: ViewportState
    let cycleConnectedNode: () -> Void
    let exportToPNG: () -> Void

    func body(content: Content) -> some View {
        content
            .onKeyPress(.escape) {
                selectedMemoryId = nil
                searchText = ""
                isSearchFocused.wrappedValue = false
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
                if keyPress.characters == "f" && keyPress.modifiers == .command {
                    isSearchFocused.wrappedValue = true
                    return .handled
                }
                if keyPress.characters == "E" && keyPress.modifiers == [.command, .shift] {
                    exportToPNG()
                    return .handled
                }
                return .ignored
            }
    }
}
