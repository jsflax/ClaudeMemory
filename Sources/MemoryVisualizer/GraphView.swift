import SwiftUI
import Lattice
import ClaudeMemoryLib

typealias MemoryEdge = ClaudeMemoryLib.Edge

// MARK: - Pre-computed display data

struct NodeInfo: Equatable {
    let id: Int64
    let label: String
    let project: String
    let importance: Int
    let isHub: Bool
}

struct EdgeInfo: Equatable {
    let sourceId: Int64
    let targetId: Int64
    let relation: String
}

// MARK: - Root view (owns queries, computes display data)

struct GraphView: View {
    @LatticeQuery<Memory>(sort: \Memory.createdAt) var memories: TableResults<Memory>
    @LatticeQuery<MemoryEdge>(sort: \MemoryEdge.createdAt) var edges: TableResults<MemoryEdge>

    @State private var simulation = ForceSimulation()
    @State private var hiddenProjects: Set<String> = []
    @State private var scale: CGFloat = 1.0
    @State private var offset: CGPoint = .zero
    @State private var draggedNode: Int64?
    @State private var isPanningBackground: Bool = false
    @State private var panStart: CGPoint = .zero
    @State private var offsetStart: CGPoint = .zero
    @State private var hoveredNode: Int64?
    @State private var selectedMemoryId: Int64?
    @State private var panelSnapshot: MemorySnapshot?
    @State private var scrollMonitor: Any?

    private static let projectColors: [Color] = [
        .blue, .green, .orange, .purple, .cyan, .pink, .mint, .teal, .indigo, .yellow
    ]

    // MARK: - Computed display data (re-evaluated only when queries change)

    private var nodeInfos: [NodeInfo] {
        let hubIds = computeHubIds()
        return memories.compactMap { memory -> NodeInfo? in
            guard !hiddenProjects.contains(memory.project),
                  let pk = memory.primaryKey else { return nil }
            return NodeInfo(
                id: pk,
                label: Self.extractLabel(content: memory.content, topic: memory.topic),
                project: memory.project,
                importance: memory.importance,
                isHub: hubIds.contains(pk)
            )
        }
    }

    private var edgeInfos: [EdgeInfo] {
        let nodeIds = Set(nodeInfos.map(\.id))
        return edges.compactMap { edge -> EdgeInfo? in
            guard nodeIds.contains(edge.sourceId), nodeIds.contains(edge.targetId) else { return nil }
            return EdgeInfo(sourceId: edge.sourceId, targetId: edge.targetId, relation: edge.relation)
        }
    }

    var body: some View {
        GeometryReader { geo in
            let nodes = nodeInfos
            let edgeData = edgeInfos
            ZStack {
                Color(red: 0.051, green: 0.067, blue: 0.09).ignoresSafeArea()

                GraphCanvas(
                    simulation: simulation,
                    nodes: nodes,
                    edges: edgeData,
                    scale: $scale,
                    offset: $offset,
                    draggedNode: $draggedNode,
                    isPanningBackground: $isPanningBackground,
                    panStart: $panStart,
                    offsetStart: $offsetStart,
                    hoveredNode: $hoveredNode,
                    selectedNode: $selectedMemoryId
                )

                statsOverlay(nodes: nodes, edgeData: edgeData)

                if let snapshot = panelSnapshot {
                    let isVisible = selectedMemoryId != nil
                    MemoryDetailPanel(
                        content: snapshot.content,
                        project: snapshot.project,
                        topic: snapshot.topic,
                        importance: snapshot.importance,
                        connectedCount: snapshot.connectedCount,
                        createdAt: snapshot.createdAt,
                        onClose: { selectedMemoryId = nil }
                    )
                    .id(snapshot.id)
                    .frame(maxWidth: min(400, geo.size.width * 0.35), maxHeight: min(500, geo.size.height * 0.7))
                    .frame(maxWidth: .infinity, alignment: .trailing)
                    .padding(24)
                    .offset(x: isVisible ? 0 : 420)
                    .opacity(isVisible ? 1 : 0)
                    .animation(.spring(duration: 0.4, bounce: 0.2), value: isVisible)
                    .allowsHitTesting(isVisible)
                    .onChange(of: isVisible) { _, visible in
                        if !visible {
                            // Clear snapshot after slide-out completes
                            Task {
                                try? await Task.sleep(for: .milliseconds(500))
                                if selectedMemoryId == nil { panelSnapshot = nil }
                            }
                        }
                    }
                }
            }
            .onChange(of: geo.size, initial: true) { _, newSize in
                simulation.center = CGPoint(x: newSize.width / 2, y: newSize.height / 2)
            }
            .onAppear {
                scrollMonitor = NSEvent.addLocalMonitorForEvents(matching: .scrollWheel) { event in
                    // Don't zoom when scrolling over the detail panel
                    if let window = event.window, selectedMemoryId != nil {
                        let loc = event.locationInWindow
                        if loc.x > window.frame.width * 0.6 {
                            return event
                        }
                    }
                    let zoomFactor: CGFloat = event.scrollingDeltaY > 0 ? 1.03 : 0.97
                    scale = max(0.2, min(3.0, scale * zoomFactor))
                    return event
                }
            }
            .onDisappear {
                if let monitor = scrollMonitor {
                    NSEvent.removeMonitor(monitor)
                    scrollMonitor = nil
                }
            }
            .onChange(of: selectedMemoryId) { oldId, newId in
                if let old = oldId { simulation.clearTarget(old) }
                if let new = newId {
                    let targetScreen = CGPoint(x: geo.size.width * 0.25, y: geo.size.height * 0.5)
                    let worldPt = CGPoint(
                        x: (targetScreen.x - offset.x) / scale,
                        y: (targetScreen.y - offset.y) / scale
                    )
                    simulation.setTarget(new, position: worldPt)
                    if let memory = memories.first(where: { $0.primaryKey == new }) {
                        let count = edges.filter { $0.sourceId == new || $0.targetId == new }.count
                        panelSnapshot = MemorySnapshot(
                            id: new, content: memory.content, project: memory.project,
                            topic: memory.topic, importance: memory.importance,
                            connectedCount: count, createdAt: memory.createdAt
                        )
                    }
                }
            }
        }
    }

    // MARK: - Stats Overlay

    private func statsOverlay(nodes: [NodeInfo], edgeData: [EdgeInfo]) -> some View {
        VStack {
            HStack {
                Spacer()
                VStack(alignment: .trailing, spacing: 6) {
                    let visible = nodes.count
                    let total = memories.count
                    if visible < total {
                        Text("\(visible)/\(total) memories")
                            .font(.system(size: 13, weight: .medium, design: .monospaced))
                    } else {
                        Text("\(total) memories")
                            .font(.system(size: 13, weight: .medium, design: .monospaced))
                    }
                    Text("\(edgeData.count) edges")
                        .font(.system(size: 13, weight: .medium, design: .monospaced))

                    Divider().frame(width: 100).overlay(Color.white.opacity(0.2))

                    ForEach(uniqueProjects(), id: \.self) { project in
                        Button {
                            toggleProject(project)
                        } label: {
                            HStack(spacing: 6) {
                                Circle()
                                    .fill(Self.projectColor(for: project))
                                    .frame(width: 8, height: 8)
                                    .opacity(hiddenProjects.contains(project) ? 0.3 : 1.0)
                                Text(project)
                                    .font(.system(size: 11, design: .monospaced))
                                    .strikethrough(hiddenProjects.contains(project))
                            }
                        }
                        .buttonStyle(.plain)
                    }
                }
                .foregroundStyle(.white.opacity(0.7))
                .padding(12)
            }
            Spacer()
        }
    }

    // MARK: - Helpers

    private func toggleProject(_ project: String) {
        if hiddenProjects.contains(project) {
            hiddenProjects.remove(project)
        } else {
            hiddenProjects.insert(project)
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

    static func projectColor(for project: String) -> Color {
        if project == "global" { return .gray }
        let hash = abs(project.hashValue)
        return projectColors[hash % projectColors.count]
    }

    static func extractLabel(content: String, topic: String) -> String {
        // Markdown header → extract heading
        if let headerRange = content.range(of: #"^#{1,3}\s+"#, options: .regularExpression) {
            let title = content[headerRange.upperBound...].prefix(while: { $0 != "\n" })
            return truncate(String(title), to: 30)
        }
        // Bold prefix → extract bold text
        if content.hasPrefix("**"), let end = content.dropFirst(2).range(of: "**") {
            let title = content[content.index(content.startIndex, offsetBy: 2)..<end.lowerBound]
            return truncate(String(title), to: 30)
        }
        // "Name: ..." → extract key
        if let colon = content.firstIndex(of: ":"),
           content.distance(from: content.startIndex, to: colon) < 30 {
            return String(content[..<colon])
        }
        // First line with topic context
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

// MARK: - Canvas view (isolated from queries, driven by timer)

struct GraphCanvas: View {
    let simulation: ForceSimulation
    let nodes: [NodeInfo]
    let edges: [EdgeInfo]
    @Binding var scale: CGFloat
    @Binding var offset: CGPoint
    @Binding var draggedNode: Int64?
    @Binding var isPanningBackground: Bool
    @Binding var panStart: CGPoint
    @Binding var offsetStart: CGPoint
    @Binding var hoveredNode: Int64?
    @Binding var selectedNode: Int64?
    @GestureState private var lastMagnification: CGFloat = 1.0
    @State private var frameCount: UInt64 = 0

    private let baseRadius: CGFloat = 16
    private let hubScale: CGFloat = 1.5
    private let bgColor = Color(red: 0.051, green: 0.067, blue: 0.09)
    private let timer = Timer.publish(every: 1.0 / 60.0, on: .main, in: .common).autoconnect()

    var body: some View {
        Canvas { context, size in
            let _ = frameCount
            let currentIds = Set(nodes.map(\.id))
            if currentIds != Set(simulation.positions.keys) {
                let edgePairs = edges.map { ($0.sourceId, $0.targetId) }
                simulation.updateGraph(nodeIds: currentIds, edges: edgePairs)
            }
            simulation.tick()
            var ctx = context
            ctx.translateBy(x: offset.x, y: offset.y)
            ctx.scaleBy(x: scale, y: scale)
            let positions = simulation.positions
            drawEdges(context: &ctx, positions: positions)
            drawNodes(context: &ctx, positions: positions)
        }
        .onReceive(timer) { _ in
            frameCount &+= 1
        }
        .gesture(dragGesture)
        .gesture(magnificationGesture)
        .simultaneousGesture(
            SpatialTapGesture()
                .onEnded { value in
                    if let nodeId = hitTest(value.location) {
                        selectedNode = selectedNode == nodeId ? nil : nodeId
                    } else {
                        selectedNode = nil
                    }
                }
        )
        .onContinuousHover { phase in
            switch phase {
            case .active(let location): hoveredNode = hitTest(location)
            case .ended: hoveredNode = nil
            }
        }
    }

    // MARK: - Gestures

    private var dragGesture: some Gesture {
        DragGesture(minimumDistance: 2)
            .onChanged { value in
                if draggedNode == nil && !isPanningBackground {
                    if let nodeId = hitTest(value.startLocation) {
                        draggedNode = nodeId
                    } else {
                        isPanningBackground = true
                        panStart = value.startLocation
                        offsetStart = offset
                    }
                }
                if let nodeId = draggedNode {
                    simulation.pinNode(nodeId, at: screenToWorld(value.location))
                } else if isPanningBackground {
                    offset = CGPoint(
                        x: offsetStart.x + (value.location.x - panStart.x),
                        y: offsetStart.y + (value.location.y - panStart.y)
                    )
                }
            }
            .onEnded { _ in
                if let nodeId = draggedNode { simulation.unpinNode(nodeId) }
                draggedNode = nil
                isPanningBackground = false
            }
    }

    private var magnificationGesture: some Gesture {
        MagnifyGesture()
            .updating($lastMagnification) { value, state, _ in
                let delta = value.magnification / state
                state = value.magnification
                scale = max(0.2, min(3.0, scale * delta))
            }
    }

    // MARK: - Hit testing

    private func hitTest(_ screenPoint: CGPoint) -> Int64? {
        let world = screenToWorld(screenPoint)
        let positions = simulation.positions
        var closest: Int64?
        var closestDist: CGFloat = .greatestFiniteMagnitude

        for node in nodes {
            guard let pos = positions[node.id] else { continue }
            let radius = baseRadius * (node.isHub ? hubScale : 1.0)
            let dist = hypot(world.x - pos.x, world.y - pos.y)
            if dist < radius + 4, dist < closestDist {
                closest = node.id
                closestDist = dist
            }
        }
        return closest
    }

    private func screenToWorld(_ screen: CGPoint) -> CGPoint {
        CGPoint(x: (screen.x - offset.x) / scale, y: (screen.y - offset.y) / scale)
    }

    // MARK: - Drawing

    private func drawEdges(context: inout GraphicsContext, positions: [Int64: CGPoint]) {
        let hasSelection = selectedNode != nil
        for edge in edges {
            guard let from = positions[edge.sourceId],
                  let to = positions[edge.targetId] else { continue }

            let isConnected = edge.sourceId == selectedNode || edge.targetId == selectedNode
            let dimmed = hasSelection && !isConnected

            var path = Path()
            path.move(to: from)
            let mid = CGPoint(
                x: (from.x + to.x) / 2 + (from.y - to.y) * 0.08,
                y: (from.y + to.y) / 2 + (to.x - from.x) * 0.08
            )
            path.addQuadCurve(to: to, control: mid)

            if isConnected {
                let color = GraphView.projectColor(for:
                    nodes.first(where: { $0.id == selectedNode })?.project ?? "global")
                context.stroke(path, with: .color(color.opacity(0.6)), lineWidth: 2)
            } else {
                context.stroke(path, with: .color(.white.opacity(dimmed ? 0.05 : 0.15)), lineWidth: 1.2)
            }

            context.draw(
                Text(edge.relation.replacingOccurrences(of: "_", with: " "))
                    .font(.system(size: isConnected ? 9 : 7, weight: isConnected ? .bold : .medium))
                    .foregroundStyle(.white.opacity(isConnected ? 0.7 : (dimmed ? 0.1 : 0.3))),
                at: CGPoint(x: mid.x, y: mid.y - 6)
            )
        }
    }

    private func drawNodes(context: inout GraphicsContext, positions: [Int64: CGPoint]) {
        var placedLabels: [CGRect] = []
        let hasSelection = selectedNode != nil
        let connectedToSelected: Set<Int64> = {
            guard let sel = selectedNode else { return [] }
            var ids = Set<Int64>()
            for edge in edges {
                if edge.sourceId == sel { ids.insert(edge.targetId) }
                if edge.targetId == sel { ids.insert(edge.sourceId) }
            }
            return ids
        }()

        for node in nodes {
            guard let pos = positions[node.id] else { continue }

            let importance = max(1, node.importance)
            let radius = baseRadius * (node.isHub ? hubScale : 1.0) * (1.0 + CGFloat(importance - 1) * 0.08)
            let color = GraphView.projectColor(for: node.project)
            let isSelected = node.id == selectedNode
            let isConnected = connectedToSelected.contains(node.id)
            let dimmed = hasSelection && !isSelected && !isConnected

            // Glow for selected node (large pulsing glow)
            if isSelected {
                let pulse = 1.0 + sin(Double(frameCount) * 0.05) * 0.15
                let glowRadius = (radius + 12) * pulse
                let glowRect = CGRect(
                    x: pos.x - glowRadius, y: pos.y - glowRadius,
                    width: glowRadius * 2, height: glowRadius * 2
                )
                context.fill(Circle().path(in: glowRect), with: .color(color.opacity(0.2)))
                let innerGlow = CGRect(
                    x: pos.x - radius - 6, y: pos.y - radius - 6,
                    width: (radius + 6) * 2, height: (radius + 6) * 2
                )
                context.fill(Circle().path(in: innerGlow), with: .color(color.opacity(0.35)))
            } else if hoveredNode == node.id || draggedNode == node.id {
                let glowRect = CGRect(
                    x: pos.x - radius - 4, y: pos.y - radius - 4,
                    width: (radius + 4) * 2, height: (radius + 4) * 2
                )
                context.fill(Circle().path(in: glowRect), with: .color(color.opacity(0.25)))
            }

            let nodeOpacity: CGFloat = dimmed ? 0.25 : 0.8
            let rect = CGRect(x: pos.x - radius, y: pos.y - radius, width: radius * 2, height: radius * 2)
            context.fill(Circle().path(in: rect), with: .color(color.opacity(nodeOpacity)))
            context.stroke(
                Circle().path(in: rect),
                with: .color(isSelected ? .white : (node.isHub ? .white : color).opacity(dimmed ? 0.2 : 0.6)),
                lineWidth: isSelected ? 3 : (node.isHub ? 2.5 : 1)
            )

            // Label with deconfliction
            let labelSize = CGSize(width: CGFloat(node.label.count) * 5.5 + 8, height: 14)
            let labelRect = bestLabelPlacement(nodePos: pos, nodeRadius: radius,
                                                labelSize: labelSize, placed: &placedLabels)
            let pillPath = RoundedRectangle(cornerRadius: 4).path(in: labelRect.insetBy(dx: -4, dy: -2))
            context.fill(pillPath, with: .color(bgColor.opacity(0.85)))

            context.draw(
                Text(node.label)
                    .font(.system(size: isSelected ? 11 : 9, weight: isSelected ? .bold : .medium))
                    .foregroundStyle(.white.opacity(dimmed ? 0.25 : 0.85)),
                in: labelRect
            )
        }
    }

    private func bestLabelPlacement(nodePos: CGPoint, nodeRadius: CGFloat,
                                     labelSize: CGSize, placed: inout [CGRect]) -> CGRect {
        let gap: CGFloat = 4
        let candidates = [
            CGRect(x: nodePos.x - labelSize.width / 2, y: nodePos.y + nodeRadius + gap,
                   width: labelSize.width, height: labelSize.height),
            CGRect(x: nodePos.x - labelSize.width / 2, y: nodePos.y - nodeRadius - gap - labelSize.height,
                   width: labelSize.width, height: labelSize.height),
            CGRect(x: nodePos.x + nodeRadius + gap, y: nodePos.y - labelSize.height / 2,
                   width: labelSize.width, height: labelSize.height),
            CGRect(x: nodePos.x - nodeRadius - gap - labelSize.width, y: nodePos.y - labelSize.height / 2,
                   width: labelSize.width, height: labelSize.height),
        ]

        var bestRect = candidates[0]
        var bestOverlap: CGFloat = .greatestFiniteMagnitude
        for c in candidates {
            var overlap: CGFloat = 0
            for other in placed {
                let i = c.intersection(other)
                if !i.isNull { overlap += i.width * i.height }
            }
            if overlap == 0 { placed.append(c); return c }
            if overlap < bestOverlap { bestOverlap = overlap; bestRect = c }
        }
        placed.append(bestRect)
        return bestRect
    }
}

// MARK: - Snapshot for panel (persists during slide-out animation)

struct MemorySnapshot {
    let id: Int64
    let content: String
    let project: String
    let topic: String
    let importance: Int
    let connectedCount: Int
    let createdAt: Date
}

// MARK: - Detail Panel (isolated view — typing state doesn't trigger graph re-renders)

struct MemoryDetailPanel: View {
    let content: String
    let project: String
    let topic: String
    let importance: Int
    let connectedCount: Int
    let createdAt: Date
    let onClose: () -> Void

    @State private var typingProgress: Int = 0

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            HStack {
                Circle()
                    .fill(GraphView.projectColor(for: project))
                    .frame(width: 10, height: 10)
                Text(project)
                    .font(.system(size: 11, weight: .semibold, design: .monospaced))
                    .foregroundStyle(GraphView.projectColor(for: project))
                Text("·")
                    .foregroundStyle(.white.opacity(0.3))
                Text(topic)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.5))
                Spacer()
                Button(action: onClose) {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.white.opacity(0.4))
                        .font(.system(size: 16))
                }
                .buttonStyle(.plain)
            }
            .padding(.bottom, 12)

            // Content (typewriter effect)
            ScrollView {
                Text(String(content.prefix(typingProgress)))
                    .font(.system(size: 13, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.9))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .contentShape(Rectangle())
            }

            Spacer(minLength: 12)

            // Footer
            VStack(alignment: .leading, spacing: 4) {
                if connectedCount > 0 {
                    Divider().overlay(Color.white.opacity(0.1))
                    Text("\(connectedCount) connection\(connectedCount == 1 ? "" : "s")")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.4))
                }
                HStack {
                    if importance > 0 {
                        Text("importance: \(importance)")
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.4))
                    }
                    Spacer()
                    Text(createdAt, style: .date)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.3))
                }
            }
        }
        .padding(20)
        .background {
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(red: 0.08, green: 0.1, blue: 0.14))
                .overlay {
                    RoundedRectangle(cornerRadius: 16)
                        .strokeBorder(
                            GraphView.projectColor(for: project).opacity(0.3),
                            lineWidth: 1
                        )
                }
                .shadow(color: .black.opacity(0.5), radius: 30, y: 10)
        }
        .task {
            while !Task.isCancelled && typingProgress < content.count {
                try? await Task.sleep(for: .milliseconds(10))
                typingProgress += 2
            }
            typingProgress = content.count
        }
    }
}
