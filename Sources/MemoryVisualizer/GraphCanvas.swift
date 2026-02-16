import SwiftUI
import ClaudeMemoryLib

// MARK: - File-level helpers

func extractLabel(content: String, topic: String) -> String {
    if let headerRange = content.range(of: #"^#{1,3}\s+"#, options: .regularExpression) {
        let title = content[headerRange.upperBound...].prefix(while: { $0 != "\n" })
        return truncateLabel(String(title), to: 30)
    }
    if content.hasPrefix("**"), let end = content.dropFirst(2).range(of: "**") {
        let title = content[content.index(content.startIndex, offsetBy: 2)..<end.lowerBound]
        return truncateLabel(String(title), to: 30)
    }
    if let colon = content.firstIndex(of: ":"),
       content.distance(from: content.startIndex, to: colon) < 30 {
        return String(content[..<colon])
    }
    let firstLine = String(content.prefix(while: { $0 != "\n" }))
    if firstLine.count <= 30 { return firstLine }
    if topic != "general" { return "\(topic): \(truncateLabel(firstLine, to: 22))" }
    return truncateLabel(firstLine, to: 30)
}

func truncateLabel(_ text: String, to maxLen: Int) -> String {
    if text.count <= maxLen { return text }
    return String(text.prefix(maxLen - 1)) + "…"
}

// MARK: - Lightweight data structs (replaces heavy Lattice model objects)

struct NodeData {
    let id: Int64
    let project: String
    let topic: String
    let label: String
    let createdAt: Date
    var lastAccessedAt: Date
    let importance: Int
}

struct EdgeData {
    let id: Int64
    let sourceId: Int64
    let targetId: Int64
    let relation: String
}

// MARK: - GraphCanvas

/// Pure rendering component for the force-directed graph.
/// Receives all data from GraphView — no Lattice queries or observers.
struct GraphCanvas: View {
    let simulation: ForceSimulation
    let nodes: [NodeData]
    let edges: [EdgeData]
    let hubs: Set<Int64>
    let glowingNodes: [Int64: Date]
    let searchMatchIds: Set<Int64>
    let isSearchActive: Bool
    let viewport: ViewportState
    @Binding var selectedNode: Int64?
    let clusters: [[Int64]]
    let topicGroups: [TopicGroupInfo]
    let colorMap: [String: Color]

    @GestureState private var lastMagnification: CGFloat = 1.0
    @State private var frameCount: UInt64 = 0
    private let baseRadius: CGFloat = 12
    private let hubScale: CGFloat = 1.6
    private let bgColor = Color(red: 0.051, green: 0.067, blue: 0.09)
    private let timer = Timer.publish(every: 1.0 / 60.0, on: .main, in: .common).autoconnect()

    // Edge relation colors
    static let relationColors: [String: Color] = [
        "part_of": .cyan,
        "contradicts": .red,
        "supersedes": .orange,
        "derived_from": .yellow,
        "relates_to": .blue,
        "summarized_by": .purple,
    ]

    var body: some View {
        let mems = nodes
        let hubSet = hubs
        let vEdges = edges
        let glows = glowingNodes
        // Precompute connected set once per body eval (not per Canvas frame)
        let connected: Set<Int64> = {
            guard let sel = selectedNode else { return [] }
            var ids = Set<Int64>()
            for edge in edges {
                if edge.sourceId == sel { ids.insert(edge.targetId) }
                if edge.targetId == sel { ids.insert(edge.sourceId) }
            }
            return ids
        }()
        Canvas(rendersAsynchronously: true) { context, size in
            let _ = frameCount
            viewport.tickPan()
            var ctx = context
            ctx.translateBy(x: viewport.offset.x, y: viewport.offset.y)
            ctx.scaleBy(x: viewport.scale, y: viewport.scale)
            let positions = simulation.positions
            drawClusterHulls(context: &ctx, positions: positions, nodes: mems, hubs: hubSet, edges: vEdges)
            drawEdges(context: &ctx, positions: positions, nodes: mems, hubs: hubSet, edges: vEdges, connectedToSelected: connected)
            drawNodes(context: &ctx, positions: positions, nodes: mems, hubs: hubSet, edges: vEdges, glowingNodes: glows, connectedToSelected: connected)
        }
        .onReceive(timer) { _ in
            Task {
                await simulation.tick()
                let needsRedraw = simulation.isActive
                || !glowingNodes.isEmpty
                || viewport.draggedNode != nil
                || viewport.isAnimatingPan
                || (isSearchActive && !searchMatchIds.isEmpty)
                if needsRedraw { frameCount &+= 1 }
            }
        }
        .gesture(nodeDragGesture)
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
            case .active(let location): viewport.hoveredNode = hitTest(location)
            case .ended: viewport.hoveredNode = nil
            }
        }
    }

    // MARK: - Gestures

    @GestureState private var dragStartOffset: CGPoint?

    private var nodeDragGesture: some Gesture {
        DragGesture(minimumDistance: 2)
            .updating($dragStartOffset) { value, state, _ in
                if state == nil {
                    if let nodeId = hitTest(value.startLocation) {
                        viewport.draggedNode = nodeId
                    }
                    state = viewport.offset
                }
                if let nodeId = viewport.draggedNode {
                    simulation.pinNode(nodeId, at: screenToWorld(value.location))
                } else if let start = state {
                    viewport.offset = CGPoint(
                        x: start.x + value.translation.width,
                        y: start.y + value.translation.height
                    )
                }
            }
            .onEnded { _ in
                if let nodeId = viewport.draggedNode { simulation.unpinNode(nodeId) }
                viewport.draggedNode = nil
            }
    }

    private var magnificationGesture: some Gesture {
        MagnifyGesture()
            .updating($lastMagnification) { value, state, _ in
                let delta = value.magnification / state
                state = value.magnification
                guard let window = NSApp.keyWindow else {
                    viewport.scale = max(0.2, min(3.0, viewport.scale * delta))
                    return
                }
                let mouseLoc = window.mouseLocationOutsideOfEventStream
                let cursor = CGPoint(x: mouseLoc.x, y: window.frame.height - mouseLoc.y)
                let worldPoint = CGPoint(
                    x: (cursor.x - viewport.offset.x) / viewport.scale,
                    y: (cursor.y - viewport.offset.y) / viewport.scale
                )
                let newScale = max(0.2, min(3.0, viewport.scale * delta))
                viewport.offset = CGPoint(
                    x: cursor.x - worldPoint.x * newScale,
                    y: cursor.y - worldPoint.y * newScale
                )
                viewport.scale = newScale
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
            let radius = nodeRadius(for: node)
            let dist = hypot(world.x - pos.x, world.y - pos.y)
            if dist < radius + 4, dist < closestDist {
                closest = node.id
                closestDist = dist
            }
        }
        return closest
    }

    func screenToWorld(_ screen: CGPoint) -> CGPoint {
        CGPoint(
            x: (screen.x - viewport.offset.x) / viewport.scale,
            y: (screen.y - viewport.offset.y) / viewport.scale
        )
    }

    private func nodeRadius(for node: NodeData) -> CGFloat {
        let importance = max(1, node.importance)
        let isHub = hubs.contains(node.id)
        return baseRadius * (isHub ? hubScale : 1.0) * (1.0 + CGFloat(importance - 1) * 0.08)
    }

    // MARK: - Cluster Hulls

    private func drawClusterHulls(context: inout GraphicsContext, positions: [Int64: CGPoint],
                                  nodes: [NodeData], hubs: Set<Int64>, edges: [EdgeData]) {
        // Per-project hulls (faint background)
        var projectPoints: [String: [CGPoint]] = [:]
        for node in nodes {
            guard let pos = positions[node.id] else { continue }
            projectPoints[node.project, default: []].append(pos)
        }
        for (project, points) in projectPoints {
            guard points.count >= 3 else { continue }
            let hull = ConvexHull.compute(points: points)
            guard hull.count >= 3 else { continue }
            let expanded = ConvexHull.expand(hull, by: 30)
            let color = GraphView.projectColor(for: project, in: colorMap)
            var path = Path()
            path.move(to: expanded[0])
            for i in 1..<expanded.count { path.addLine(to: expanded[i]) }
            path.closeSubpath()
            context.fill(path, with: .color(color.opacity(0.04)))
            context.stroke(path, with: .color(color.opacity(0.08)), lineWidth: 1)
        }

        // Per-topic sub-hulls
        let nodeProjectMap = Dictionary(nodes.map { ($0.id, $0.project) }, uniquingKeysWith: { _, last in last })
        for group in topicGroups {
            let visiblePoints = group.ids.compactMap { positions[$0] }
            guard visiblePoints.count >= 2 else { continue }

            let color = GraphView.projectColor(for: group.project, in: colorMap)

            let hullInput: [CGPoint]
            if visiblePoints.count == 2 {
                let p1 = visiblePoints[0], p2 = visiblePoints[1]
                let dx = p2.x - p1.x, dy = p2.y - p1.y
                let dist = max(hypot(dx, dy), 1)
                let px = -dy / dist * 12, py = dx / dist * 12
                hullInput = [
                    CGPoint(x: p1.x + px, y: p1.y + py), CGPoint(x: p1.x - px, y: p1.y - py),
                    CGPoint(x: p2.x + px, y: p2.y + py), CGPoint(x: p2.x - px, y: p2.y - py),
                ]
            } else {
                hullInput = visiblePoints
            }

            let hull = ConvexHull.compute(points: hullInput)
            guard hull.count >= 3 else { continue }
            let expanded = ConvexHull.expand(hull, by: 25)

            var path = Path()
            path.move(to: expanded[0])
            for i in 1..<expanded.count { path.addLine(to: expanded[i]) }
            path.closeSubpath()

            context.fill(path, with: .color(color.opacity(0.06)))
            context.stroke(path, with: .color(color.opacity(0.2)), lineWidth: 1.5)
        }

        // Per-cluster hulls (semantic clusters — dashed)
        for cluster in clusters {
            let visiblePoints = cluster.compactMap { positions[$0] }
            guard visiblePoints.count >= 2 else { continue }

            let project = cluster.first.flatMap { nodeProjectMap[$0] } ?? "global"
            let color = GraphView.projectColor(for: project, in: colorMap)

            let hullInput: [CGPoint]
            if visiblePoints.count == 2 {
                let p1 = visiblePoints[0], p2 = visiblePoints[1]
                let dx = p2.x - p1.x, dy = p2.y - p1.y
                let dist = max(hypot(dx, dy), 1)
                let px = -dy / dist * 12, py = dx / dist * 12
                hullInput = [
                    CGPoint(x: p1.x + px, y: p1.y + py), CGPoint(x: p1.x - px, y: p1.y - py),
                    CGPoint(x: p2.x + px, y: p2.y + py), CGPoint(x: p2.x - px, y: p2.y - py),
                ]
            } else {
                hullInput = visiblePoints
            }

            let hull = ConvexHull.compute(points: hullInput)
            guard hull.count >= 3 else { continue }
            let expanded = ConvexHull.expand(hull, by: 20)

            var path = Path()
            path.move(to: expanded[0])
            for i in 1..<expanded.count { path.addLine(to: expanded[i]) }
            path.closeSubpath()

            context.fill(path, with: .color(color.opacity(0.08)))
            context.stroke(path, with: .color(color.opacity(0.25)), style: StrokeStyle(lineWidth: 1.5, dash: [6, 4]))
        }
    }

    // MARK: - Drawing Edges

    private func drawEdges(context: inout GraphicsContext, positions: [Int64: CGPoint],
                           nodes: [NodeData], hubs: Set<Int64>, edges: [EdgeData],
                           connectedToSelected: Set<Int64> = []) {
        let nodeByPk = Dictionary(nodes.map { ($0.id, $0) }, uniquingKeysWith: { _, last in last })
        let hasSelection = selectedNode != nil
        let selectedProject = selectedNode.flatMap { nodeByPk[$0]?.project } ?? "global"

        for edge in edges {
            guard let from = positions[edge.sourceId],
                  let to = positions[edge.targetId] else { continue }

            let isConnected = edge.sourceId == selectedNode || edge.targetId == selectedNode
            let dimmed = hasSelection && !isConnected
            let searchDimmed = isSearchActive && !searchMatchIds.contains(edge.sourceId) && !searchMatchIds.contains(edge.targetId)

            let targetNode = nodeByPk[edge.targetId]
            let targetRadius = targetNode.map { nodeRadius(for: $0) } ?? baseRadius

            let mid = CGPoint(
                x: (from.x + to.x) / 2 + (from.y - to.y) * 0.08,
                y: (from.y + to.y) / 2 + (to.x - from.x) * 0.08
            )

            let shortenedTo = shortenEndpoint(from: mid, to: to, by: targetRadius + 4)

            var path = Path()
            path.move(to: from)
            path.addQuadCurve(to: shortenedTo, control: mid)

            let relationColor = Self.relationColors[edge.relation] ?? .white

            if isConnected {
                let color = GraphView.projectColor(for: selectedProject, in: colorMap)
                context.stroke(path, with: .color(color.opacity(0.6)), lineWidth: 2)
            } else {
                let baseOpacity: CGFloat = (dimmed || searchDimmed) ? 0.05 : 0.15
                context.stroke(path, with: .color(relationColor.opacity(baseOpacity)), lineWidth: 1.2)
            }

            // Arrow head
            let arrowOpacity: CGFloat = isConnected ? 0.6 : ((dimmed || searchDimmed) ? 0.05 : 0.15)
            let arrowColor = isConnected
                ? GraphView.projectColor(for: selectedProject, in: colorMap)
                : relationColor
            drawArrowHead(context: &context, from: mid, to: shortenedTo, color: arrowColor, opacity: arrowOpacity)

            // Edge label (only at higher zoom — text rendering is expensive)
            if isConnected || viewport.scale >= 0.7 {
                context.draw(
                    Text(edge.relation.replacingOccurrences(of: "_", with: " "))
                        .font(.system(size: isConnected ? 9 : 7, weight: isConnected ? .bold : .medium))
                        .foregroundStyle(relationColor.opacity(isConnected ? 0.7 : ((dimmed || searchDimmed) ? 0.1 : 0.3))),
                    at: CGPoint(x: mid.x, y: mid.y - 6)
                )
            }
        }
    }

    private func shortenEndpoint(from: CGPoint, to: CGPoint, by amount: CGFloat) -> CGPoint {
        let dx = to.x - from.x, dy = to.y - from.y
        let dist = max(hypot(dx, dy), 1)
        let ratio = max(0, (dist - amount) / dist)
        return CGPoint(x: from.x + dx * ratio, y: from.y + dy * ratio)
    }

    private func drawArrowHead(context: inout GraphicsContext, from: CGPoint, to: CGPoint, color: Color, opacity: CGFloat) {
        let dx = to.x - from.x, dy = to.y - from.y
        let dist = max(hypot(dx, dy), 1)
        let ux = dx / dist, uy = dy / dist

        let arrowLength: CGFloat = 8
        let arrowWidth: CGFloat = 5

        let tip = to
        let baseCenter = CGPoint(x: tip.x - ux * arrowLength, y: tip.y - uy * arrowLength)
        let left = CGPoint(x: baseCenter.x + uy * arrowWidth, y: baseCenter.y - ux * arrowWidth)
        let right = CGPoint(x: baseCenter.x - uy * arrowWidth, y: baseCenter.y + ux * arrowWidth)

        var path = Path()
        path.move(to: tip)
        path.addLine(to: left)
        path.addLine(to: right)
        path.closeSubpath()

        var ctx = context
        ctx.opacity = opacity
        ctx.fill(path, with: .color(color))
    }

    // MARK: - Label Tiers (zoom-based visibility)

    private enum LabelTier: Int, Comparable {
        case always    = 0
        case project   = 1
        case hub       = 2
        case topic     = 3
        case prominent = 4
        case normal    = 5
        static func < (lhs: LabelTier, rhs: LabelTier) -> Bool { lhs.rawValue < rhs.rawValue }
    }

    private func labelVisible(tier: LabelTier, scale: CGFloat) -> Bool {
        switch tier {
        case .always:    return true
        case .project:   return scale >= 0.2
        case .hub:       return scale >= 0.3
        case .topic:     return scale >= 0.4
        case .prominent: return scale >= 0.7
        case .normal:    return scale >= 1.8
        }
    }

    private func labelFontSize(tier: LabelTier) -> CGFloat {
        switch tier {
        case .always:    return 10
        case .project:   return 13
        case .hub:       return 10
        case .topic:     return 11
        case .prominent: return 8
        case .normal:    return 7
        }
    }

    // MARK: - Drawing Nodes

    private struct LabelCmd {
        let text: String
        let pos: CGPoint
        let radius: CGFloat
        let tier: LabelTier
        let project: String
        let isHub: Bool
        let dimmed: Bool
    }

    private func drawNodes(context: inout GraphicsContext, positions: [Int64: CGPoint],
                           nodes: [NodeData], hubs: Set<Int64>, edges: [EdgeData],
                           glowingNodes: [Int64: Date],
                           connectedToSelected: Set<Int64> = []) {
        let hasSelection = selectedNode != nil

        var labelCmds: [LabelCmd] = []

        // --- Pass 1: Draw all node circles (regular first, then hubs on top) ---
        let regular = nodes.filter { !hubs.contains($0.id) }
        let hubNodes = nodes.filter { hubs.contains($0.id) }
        let allOrdered = regular + hubNodes
        let now = Date()

        // Precompute recall intensities
        var recallIntensities: [Int64: CGFloat] = [:]
        for node in allOrdered {
            guard let glowStart = glowingNodes[node.id], node.id != selectedNode else { continue }
            let elapsed = now.timeIntervalSince(glowStart)
            let fadeIn: CGFloat = 0.3, hold: CGFloat = 1.5, fadeOut: CGFloat = 2.0
            let total = fadeIn + hold + fadeOut
            let ri: CGFloat
            if elapsed < Double(fadeIn) {
                let t = CGFloat(elapsed) / fadeIn; ri = t * t
            } else if elapsed < Double(fadeIn + hold) {
                ri = 1.0
            } else if elapsed < Double(total) {
                let t = 1.0 - (CGFloat(elapsed) - fadeIn - hold) / fadeOut; ri = t * t
            } else { ri = 0 }
            if ri > 0 { recallIntensities[node.id] = ri }
        }

        // Pass 0: Bloom halos behind all nodes
        for node in allOrdered {
            guard let pos = positions[node.id] else { continue }
            let pk = node.id
            let radius = nodeRadius(for: node)
            let color = GraphView.projectColor(for: node.project, in: colorMap)
            let isSelected = pk == selectedNode
            let searchMatched = isSearchActive && searchMatchIds.contains(pk)

            if let ri = recallIntensities[pk] {
                let pulse = 1.0 + sin(Double(frameCount) * 0.08) * 0.15 * ri
                let bloomRadius = (radius + 20) * pulse
                let bloomRect = CGRect(
                    x: pos.x - bloomRadius, y: pos.y - bloomRadius,
                    width: bloomRadius * 2, height: bloomRadius * 2
                )
                context.fill(Circle().path(in: bloomRect), with: .color(Color(red: 0.6, green: 0.85, blue: 1.0).opacity(0.4 * ri)))
            }
            if searchMatched && !isSelected {
                let pulse = 1.0 + sin(Double(frameCount) * 0.08) * 0.1
                let glowRadius = (radius + 8) * pulse
                let glowRect = CGRect(
                    x: pos.x - glowRadius, y: pos.y - glowRadius,
                    width: glowRadius * 2, height: glowRadius * 2
                )
                context.fill(Circle().path(in: glowRect), with: .color(.white.opacity(0.15)))
            }
            if isSelected {
                let pulse = 1.0 + sin(now.timeIntervalSinceReferenceDate * 3.0) * 0.15
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
            } else if viewport.hoveredNode == pk || viewport.draggedNode == pk {
                let glowRect = CGRect(
                    x: pos.x - radius - 4, y: pos.y - radius - 4,
                    width: (radius + 4) * 2, height: (radius + 4) * 2
                )
                context.fill(Circle().path(in: glowRect), with: .color(color.opacity(0.25)))
            }
        }

        // Pass 1: Node fills + strokes + overlays
        for node in allOrdered {
            guard let pos = positions[node.id] else { continue }
            let pk = node.id

            let isHub = hubs.contains(pk)
            let radius = nodeRadius(for: node)
            let color = GraphView.projectColor(for: node.project, in: colorMap)
            let isSelected = pk == selectedNode
            let isConnected = connectedToSelected.contains(pk)
            let dimmed = hasSelection && !isSelected && !isConnected
            let searchDimmed = isSearchActive && !searchMatchIds.contains(pk)
            let searchMatched = isSearchActive && searchMatchIds.contains(pk)

            // Recency fade: stretched exponential (plateau then drop), 30-day scale
            let daysSinceAccess = now.timeIntervalSince(node.lastAccessedAt) / 86400.0
            let freshness = exp(-pow(daysSinceAccess / 30.0, 1.5))
            let nodeOpacity: CGFloat = (dimmed || searchDimmed) ? 0.15 : (0.12 + freshness * 0.68)
            let rect = CGRect(x: pos.x - radius, y: pos.y - radius, width: radius * 2, height: radius * 2)
            let ri = recallIntensities[pk] ?? 0
            // Draw base node at project color
            let baseStroke = isSelected ? Color.white : (isHub ? Color.white : color)
            let baseStrokeOpacity: CGFloat = (dimmed || searchDimmed) ? 0.1 : (0.08 + freshness * 0.52)
            let baseLineWidth: CGFloat = isSelected ? 3 : (isHub ? 2.5 : 1)
            context.fill(Circle().path(in: rect), with: .color(color.opacity(nodeOpacity)))
            context.stroke(Circle().path(in: rect), with: .color(baseStroke.opacity(baseStrokeOpacity)), lineWidth: baseLineWidth)
            // Overlay hot white-blue blended by recallIntensity (fades smoothly)
            if ri > 0 {
                let hotWhite = Color(red: 0.9, green: 0.95, blue: 1.0)
                context.fill(Circle().path(in: rect), with: .color(hotWhite.opacity(0.85 * ri)))
                context.stroke(Circle().path(in: rect), with: .color(Color(red: 0.8, green: 0.9, blue: 1.0).opacity(0.9 * ri)), lineWidth: 2.5)
            }

            // Determine label tier
            let tier: LabelTier
            if isSelected || pk == viewport.hoveredNode || pk == viewport.draggedNode {
                tier = .always
            } else if isHub {
                tier = .hub
            } else if isConnected || searchMatched {
                tier = .prominent
            } else {
                tier = .normal
            }

            if labelVisible(tier: tier, scale: viewport.scale) {
                labelCmds.append(LabelCmd(
                    text: node.label,
                    pos: pos, radius: radius,
                    tier: tier, project: node.project,
                    isHub: isHub, dimmed: dimmed || searchDimmed
                ))
            }
        }

        // --- Pass 1b: Topic centroid labels ---
        if labelVisible(tier: .topic, scale: viewport.scale) {
            for group in topicGroups {
                let groupPositions = group.ids.compactMap { positions[$0] }
                guard groupPositions.count >= 2 else { continue }
                let cx = groupPositions.map(\.x).reduce(0, +) / CGFloat(groupPositions.count)
                let cy = groupPositions.map(\.y).reduce(0, +) / CGFloat(groupPositions.count)
                labelCmds.append(LabelCmd(
                    text: group.topic, pos: CGPoint(x: cx, y: cy), radius: 0,
                    tier: .topic, project: group.project,
                    isHub: false, dimmed: false
                ))
            }
        }

        // --- Pass 1c: Project cluster labels ---
        if labelVisible(tier: .project, scale: viewport.scale) {
            var projectPts: [String: [CGPoint]] = [:]
            for node in nodes {
                guard let pos = positions[node.id] else { continue }
                projectPts[node.project, default: []].append(pos)
            }
            for (project, points) in projectPts {
                guard points.count >= 2 else { continue }
                let minY = points.map(\.y).min()!
                let cx = points.map(\.x).reduce(0, +) / CGFloat(points.count)
                labelCmds.append(LabelCmd(
                    text: project, pos: CGPoint(x: cx, y: minY - 20), radius: 0,
                    tier: .project, project: project,
                    isHub: false, dimmed: false
                ))
            }
        }

        // --- Pass 2: Draw all labels sorted by priority ---
        labelCmds.sort { $0.tier < $1.tier }
        var placedLabels: [CGRect] = []

        for cmd in labelCmds {
            let fontSize = labelFontSize(tier: cmd.tier)
            let charWidth: CGFloat = fontSize * 0.58
            let labelHeight: CGFloat = fontSize + 4
            let labelWidth = CGFloat(cmd.text.count) * charWidth + 10
            let labelSize = CGSize(width: labelWidth, height: labelHeight)
            let weight: Font.Weight = (cmd.tier <= .hub || cmd.tier == .project || cmd.isHub) ? .bold : .medium

            let labelRect = bestLabelPlacement(
                anchorPos: cmd.pos, clearRadius: cmd.radius,
                labelSize: labelSize, placed: &placedLabels
            )

            // Background pill
            let isHeader = cmd.tier == .project || cmd.tier == .topic || cmd.isHub
            let pillOpacity: CGFloat = isHeader ? 0.92 : 0.85
            let pillRect = labelRect.insetBy(dx: -5, dy: -2)
            let pillPath = RoundedRectangle(cornerRadius: cmd.tier == .project ? 6 : 4).path(in: pillRect)
            context.fill(pillPath, with: .color(bgColor.opacity(pillOpacity)))

            if isHeader {
                let color = GraphView.projectColor(for: cmd.project, in: colorMap)
                let borderOpacity: CGFloat = cmd.tier == .project ? 0.5 : (cmd.tier == .topic ? 0.3 : 0.4)
                context.stroke(pillPath, with: .color(color.opacity(borderOpacity)),
                               lineWidth: cmd.tier == .project ? 1.5 : 1)
            }

            let textOpacity: CGFloat = cmd.dimmed ? 0.15 : (cmd.tier <= .topic ? 0.95 : 0.85)
            context.draw(
                Text(cmd.text)
                    .font(.system(size: fontSize, weight: weight))
                    .foregroundStyle(.white.opacity(textOpacity)),
                in: labelRect
            )
        }
    }

    private func bestLabelPlacement(anchorPos: CGPoint, clearRadius: CGFloat,
                                     labelSize: CGSize, placed: inout [CGRect]) -> CGRect {
        let gap: CGFloat = 5
        let r = clearRadius
        let w = labelSize.width, h = labelSize.height
        let candidates = [
            CGRect(x: anchorPos.x - w / 2, y: anchorPos.y + r + gap, width: w, height: h),
            CGRect(x: anchorPos.x - w / 2, y: anchorPos.y - r - gap - h, width: w, height: h),
            CGRect(x: anchorPos.x + r + gap, y: anchorPos.y - h / 2, width: w, height: h),
            CGRect(x: anchorPos.x - r - gap - w, y: anchorPos.y - h / 2, width: w, height: h),
            CGRect(x: anchorPos.x + r * 0.7 + gap, y: anchorPos.y + r * 0.7 + gap, width: w, height: h),
            CGRect(x: anchorPos.x - r * 0.7 - gap - w, y: anchorPos.y + r * 0.7 + gap, width: w, height: h),
            CGRect(x: anchorPos.x + r * 0.7 + gap, y: anchorPos.y - r * 0.7 - gap - h, width: w, height: h),
            CGRect(x: anchorPos.x - r * 0.7 - gap - w, y: anchorPos.y - r * 0.7 - gap - h, width: w, height: h),
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
