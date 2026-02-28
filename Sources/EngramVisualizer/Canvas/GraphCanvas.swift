import SwiftUI
import EngramKit

// MARK: - GraphCanvas

/// Pure rendering component for the force-directed graph.
/// Receives all data from GraphView — no Lattice queries or observers.
struct GraphCanvas: View {
    @ObservedObject var simulation: ForceSimulation
    let nodes: [NodeData]
    let edges: [EdgeData]
    let hubs: Set<UUID>
    let glowingNodes: [UUID: Date]
    let newNodes: [UUID: Date]
    let dyingNodes: [UUID: DyingNode]
    let searchMatchIds: Set<UUID>
    let isSearchActive: Bool
    let viewport: ViewportState
    @Binding var selectedNode: UUID?
    let clusters: [[UUID]]
    let topicGroups: [TopicGroupInfo]
    let colorMap: [String: Color]
    var layoutMode: LayoutMode = .forceDirected
    var overridePositions: [UUID: CGPoint]? = nil
    var knowledgeVoids: [KnowledgeVoid] = []
    var semanticClusters: [SemanticCluster] = []
    var projectionState: ProjectionState = .idle
    var onAnimationTick: (() -> Void)? = nil

    @GestureState private var lastMagnification: CGFloat = 1.0
    @State var frameCount: UInt64 = 0
    @State var clusterFadeStartFrame: UInt64 = 0
    @State private var hullTransitionFrame: UInt64 = 0  // frame when layout mode last changed
    private let perf = PerfStats()
    let baseRadius: CGFloat = 12
    let hubScale: CGFloat = 1.6
    let bgColor = Color(red: 0.051, green: 0.067, blue: 0.09)
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
        let connected: Set<UUID> = {
            guard let sel = selectedNode else { return [] }
            var ids = Set<UUID>()
            for edge in edges {
                if edge.sourceId == sel { ids.insert(edge.targetId) }
                if edge.targetId == sel { ids.insert(edge.sourceId) }
            }
            return ids
        }()
        Canvas(rendersAsynchronously: true) { context, size in
            let _ = frameCount
            let t0 = CFAbsoluteTimeGetCurrent()
            // FPS from inter-frame interval
            let p = perf
            if p.lastFrameTime > 0 {
                let dt = t0 - p.lastFrameTime
                if dt > 0 { p.fps = p.fps * 0.9 + (1.0 / dt) * 0.1 }
            }
            p.lastFrameTime = t0

            var ctx = context
            ctx.translateBy(x: viewport.offset.x, y: viewport.offset.y)
            ctx.scaleBy(x: viewport.scale, y: viewport.scale)
            let positions = overridePositions ?? simulation.positions
            let t1 = CFAbsoluteTimeGetCurrent()
            // Cross-fade hulls during mode transitions (36 frames / 0.6s)
            let hullTransitionProgress: CGFloat = hullTransitionFrame > 0 && frameCount > hullTransitionFrame
                ? min(1.0, CGFloat(frameCount - hullTransitionFrame) / 36.0)
                : 1.0
            let forceHullFade: CGFloat = layoutMode == .forceDirected ? hullTransitionProgress : (1.0 - hullTransitionProgress)
            let semanticHullFade: CGFloat = layoutMode == .embedding ? hullTransitionProgress : (1.0 - hullTransitionProgress)
            if forceHullFade > 0 {
                drawClusterHulls(context: &ctx, positions: positions, nodes: mems, hubs: hubSet, edges: vEdges, fade: forceHullFade)
            }
            if semanticHullFade > 0 {
                drawSemanticClusters(context: &ctx, positions: positions, nodes: mems, transitionFade: semanticHullFade)
            }
            drawKnowledgeVoids(context: &ctx, positions: positions)
            let t2 = CFAbsoluteTimeGetCurrent()
            drawEdges(context: &ctx, positions: positions, nodes: mems, hubs: hubSet, edges: vEdges, connectedToSelected: connected)
            let t3 = CFAbsoluteTimeGetCurrent()
            drawNodes(context: &ctx, positions: positions, nodes: mems, hubs: hubSet, edges: vEdges, glowingNodes: glows, connectedToSelected: connected)
            let t4 = CFAbsoluteTimeGetCurrent()
            p.drawMs = (t4 - t0) * 1000
            p.hullMs = (t2 - t1) * 1000
            p.edgeMs = (t3 - t2) * 1000
            p.nodeMs = (t4 - t3) * 1000
            context.draw(
                Text("\(Int(p.fps))fps \(String(format: "%.0f", p.drawMs))ms [hull:\(String(format: "%.0f", p.hullMs)) edge:\(String(format: "%.0f", p.edgeMs)) node:\(String(format: "%.0f", p.nodeMs))] \(mems.count)n \(vEdges.count)e")
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(.green.opacity(0.8)),
                at: CGPoint(x: size.width / 2, y: size.height - 16)
            )

            // Semantic mode indicator
            if layoutMode == .embedding && projectionState == .ready {
                context.draw(
                    Text("SEMANTIC VIEW — proximity = embedding similarity")
                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                        .foregroundStyle(.cyan.opacity(0.4)),
                    at: CGPoint(x: size.width / 2, y: 20)
                )
            }

            // Progress is shown via nodes drifting progressively — no overlay needed
            // Log FPS to file every ~1s
            if frameCount % 60 == 0 && frameCount > 0 {
                let line = "[perf] \(Int(p.fps))fps | \(String(format: "%.0f", p.drawMs))ms [hull:\(String(format: "%.0f", p.hullMs)) edge:\(String(format: "%.0f", p.edgeMs)) node:\(String(format: "%.0f", p.nodeMs))] | \(mems.count)n \(vEdges.count)e\n"
                if let data = line.data(using: .utf8) {
                    let fh = FileHandle(forWritingAtPath: "/tmp/visualizer-fps.log")
                        ?? { FileManager.default.createFile(atPath: "/tmp/visualizer-fps.log", contents: nil); return FileHandle(forWritingAtPath: "/tmp/visualizer-fps.log")! }()
                    fh.seekToEndOfFile()
                    fh.write(data)
                    fh.closeFile()
                }
            }
        }
        .onReceive(timer) { _ in
            viewport.tickPan()
            simulation.tick()
            onAnimationTick?()
            // Track cluster fade-in: record the frame when clusters first appear
            if !semanticClusters.isEmpty && clusterFadeStartFrame == 0 {
                clusterFadeStartFrame = frameCount
            } else if semanticClusters.isEmpty && clusterFadeStartFrame != 0 {
                clusterFadeStartFrame = 0
            }
            let isProjecting: Bool = {
                if case .computing = projectionState { return true }
                return false
            }()
            let needsRedraw = simulation.isActive
            || !glowingNodes.isEmpty
            || !newNodes.isEmpty
            || !dyingNodes.isEmpty
            || viewport.draggedNode != nil
            || viewport.isAnimatingPan
            || (isSearchActive && !searchMatchIds.isEmpty)
            || !knowledgeVoids.isEmpty  // void pulse animation needs frameCount
            || isProjecting
            || (clusterFadeStartFrame > 0 && frameCount < clusterFadeStartFrame + 36)
            || (hullTransitionFrame > 0 && frameCount < hullTransitionFrame + 36)
            if needsRedraw { frameCount &+= 1 }
        }
        .onChange(of: layoutMode) { _, _ in
            hullTransitionFrame = max(frameCount, 1)
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

    private func hitTest(_ screenPoint: CGPoint) -> UUID? {
        let world = screenToWorld(screenPoint)
        let positions = overridePositions ?? simulation.positions
        var closest: UUID?
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

    func nodeRadius(for node: NodeData) -> CGFloat {
        let importance = max(1, node.importance)
        let isHub = hubs.contains(node.id)
        return baseRadius * (isHub ? hubScale : 1.0) * (1.0 + CGFloat(importance - 1) * 0.08)
    }
}
