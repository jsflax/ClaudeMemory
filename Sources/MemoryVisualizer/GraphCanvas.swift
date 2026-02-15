import SwiftUI
import ClaudeMemoryLib

/// Canvas rendering + gesture handling for the force-directed graph.
/// Uses ViewportState (Observable) so pan/zoom changes don't trigger GraphView body re-evaluation.
struct GraphCanvas: View {
    let simulation: ForceSimulation
    let nodes: [NodeInfo]
    let edges: [EdgeInfo]
    let searchMatchIds: Set<Int64>
    let isSearchActive: Bool
    let viewport: ViewportState
    @Binding var selectedNode: Int64?
    let clusters: [[Int64]]
    let colorMap: [String: Color]
    @GestureState private var lastMagnification: CGFloat = 1.0
    @State private var frameCount: UInt64 = 0

    private let baseRadius: CGFloat = 16
    private let hubScale: CGFloat = 1.5
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
        Canvas { context, size in
            let _ = frameCount
            let currentIds = Set(nodes.map(\.id))
            if currentIds != Set(simulation.positions.keys) {
                let edgePairs = edges.map { ($0.sourceId, $0.targetId) }
                var projectMap: [Int64: String] = [:]
                for node in nodes { projectMap[node.id] = node.project }
                simulation.updateGraph(nodeIds: currentIds, edges: edgePairs, projectForNode: projectMap)
            }
            simulation.tick()
            viewport.tickPan()
            var ctx = context
            ctx.translateBy(x: viewport.offset.x, y: viewport.offset.y)
            ctx.scaleBy(x: viewport.scale, y: viewport.scale)
            let positions = simulation.positions
            drawClusterHulls(context: &ctx, positions: positions)
            drawEdges(context: &ctx, positions: positions)
            drawNodes(context: &ctx, positions: positions)
        }
        .onReceive(timer) { _ in
            frameCount &+= 1
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

    private var nodeDragGesture: some Gesture {
        DragGesture(minimumDistance: 2)
            .onChanged { value in
                if viewport.draggedNode == nil {
                    if let nodeId = hitTest(value.startLocation) {
                        viewport.draggedNode = nodeId
                    }
                }
                if let nodeId = viewport.draggedNode {
                    simulation.pinNode(nodeId, at: screenToWorld(value.location))
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
                // Zoom anchored to cursor: keep the world point under the cursor fixed
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

    private func nodeRadius(for node: NodeInfo) -> CGFloat {
        let importance = max(1, node.importance)
        return baseRadius * (node.isHub ? hubScale : 1.0) * (1.0 + CGFloat(importance - 1) * 0.08)
    }

    // MARK: - Cluster Hulls

    private func drawClusterHulls(context: inout GraphicsContext, positions: [Int64: CGPoint]) {
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

        // Per-cluster hulls (semantic clusters — more prominent, dashed)
        let nodeProjectMap = Dictionary(uniqueKeysWithValues: nodes.map { ($0.id, $0.project) })
        for cluster in clusters {
            let visiblePoints = cluster.compactMap { positions[$0] }
            guard visiblePoints.count >= 2 else { continue }

            let project = cluster.first.flatMap { nodeProjectMap[$0] } ?? "global"
            let color = GraphView.projectColor(for: project, in: colorMap)

            // For 2-point clusters, add perpendicular offsets to form a capsule shape
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

    private func drawEdges(context: inout GraphicsContext, positions: [Int64: CGPoint]) {
        let hasSelection = selectedNode != nil
        for edge in edges {
            guard let from = positions[edge.sourceId],
                  let to = positions[edge.targetId] else { continue }

            let isConnected = edge.sourceId == selectedNode || edge.targetId == selectedNode
            let dimmed = hasSelection && !isConnected
            let searchDimmed = isSearchActive && !searchMatchIds.contains(edge.sourceId) && !searchMatchIds.contains(edge.targetId)

            let targetNode = nodes.first(where: { $0.id == edge.targetId })
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
                let color = GraphView.projectColor(for:
                    nodes.first(where: { $0.id == selectedNode })?.project ?? "global", in: colorMap)
                context.stroke(path, with: .color(color.opacity(0.6)), lineWidth: 2)
            } else {
                let baseOpacity: CGFloat = (dimmed || searchDimmed) ? 0.05 : 0.15
                context.stroke(path, with: .color(relationColor.opacity(baseOpacity)), lineWidth: 1.2)
            }

            // Arrow head
            let arrowOpacity: CGFloat = isConnected ? 0.6 : ((dimmed || searchDimmed) ? 0.05 : 0.15)
            let arrowColor = isConnected
                ? GraphView.projectColor(for: nodes.first(where: { $0.id == selectedNode })?.project ?? "global", in: colorMap)
                : relationColor
            drawArrowHead(context: &context, from: mid, to: shortenedTo, color: arrowColor, opacity: arrowOpacity)

            // Edge label
            context.draw(
                Text(edge.relation.replacingOccurrences(of: "_", with: " "))
                    .font(.system(size: isConnected ? 9 : 7, weight: isConnected ? .bold : .medium))
                    .foregroundStyle(relationColor.opacity(isConnected ? 0.7 : ((dimmed || searchDimmed) ? 0.1 : 0.3))),
                at: CGPoint(x: mid.x, y: mid.y - 6)
            )
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

    // MARK: - Drawing Nodes

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

            let radius = nodeRadius(for: node)
            let color = GraphView.projectColor(for: node.project, in: colorMap)
            let isSelected = node.id == selectedNode
            let isConnected = connectedToSelected.contains(node.id)
            let dimmed = hasSelection && !isSelected && !isConnected
            let searchDimmed = isSearchActive && !searchMatchIds.contains(node.id)
            let searchMatched = isSearchActive && searchMatchIds.contains(node.id)

            // Glow for search matches
            if searchMatched && !isSelected {
                let pulse = 1.0 + sin(Double(frameCount) * 0.08) * 0.1
                let glowRadius = (radius + 8) * pulse
                let glowRect = CGRect(
                    x: pos.x - glowRadius, y: pos.y - glowRadius,
                    width: glowRadius * 2, height: glowRadius * 2
                )
                context.fill(Circle().path(in: glowRect), with: .color(.white.opacity(0.15)))
            }

            // Glow for selected node
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
            } else if viewport.hoveredNode == node.id || viewport.draggedNode == node.id {
                let glowRect = CGRect(
                    x: pos.x - radius - 4, y: pos.y - radius - 4,
                    width: (radius + 4) * 2, height: (radius + 4) * 2
                )
                context.fill(Circle().path(in: glowRect), with: .color(color.opacity(0.25)))
            }

            let nodeOpacity: CGFloat = (dimmed || searchDimmed) ? 0.15 : 0.8
            let rect = CGRect(x: pos.x - radius, y: pos.y - radius, width: radius * 2, height: radius * 2)
            context.fill(Circle().path(in: rect), with: .color(color.opacity(nodeOpacity)))
            context.stroke(
                Circle().path(in: rect),
                with: .color(isSelected ? .white : (node.isHub ? .white : color).opacity((dimmed || searchDimmed) ? 0.1 : 0.6)),
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
                    .foregroundStyle(.white.opacity((dimmed || searchDimmed) ? 0.15 : 0.85)),
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
