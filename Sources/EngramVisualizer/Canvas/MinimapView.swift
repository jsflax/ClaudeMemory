import SwiftUI
import os

private let minimapLog = Logger(subsystem: "com.claudememory.visualizer", category: "Minimap")

// MARK: - Minimap

struct MinimapView: View {
    let renderStore: GraphRenderStore
    let simulation: ForceSimulation
    let viewport: ViewportState
    let viewportSize: CGSize
    var pipAction: (() -> Void)?
    var isFloating: Bool = false

    // Multi-galaxy support — when set, reads merged nodes/hubs/glows from all galaxies.
    var galaxyRegistry: GalaxyRegistry? = nil

    // 3D mode — when camera3DState is present, renders XZ top-down projection + camera chevron.
    // Camera3DState is @Observable: only MinimapView reads its properties, so only MinimapView
    // re-renders on camera changes — GraphView's body is never re-evaluated.
    var camera3DState: Camera3DState? = nil

    @State private var frameCount: UInt64 = 0
    @State private var isHovered: Bool = false
    @State private var renderedSize: CGSize = .zero
    // DIAGNOSTIC: throttled to 10fps (was 60fps). At 60fps this fires a main-thread
    // @State write + Canvas redraw every 16ms, potentially starving SceneEvents.Update.
    private let timer = Timer.publish(every: 1.0 / 10.0, on: .main, in: .common).autoconnect()

    private let inlineSize = CGSize(width: 160, height: 120)

    var body: some View {
        // Read from galaxyRegistry (merged across all galaxies) when available,
        // otherwise fall back to the single renderStore.
        let nodes = galaxyRegistry?.mergedNodes ?? renderStore.nodes
        let hubIds = galaxyRegistry?.mergedHubs ?? renderStore.hubs
        ZStack(alignment: .topTrailing) {
            minimapCanvas(nodes: nodes, hubIds: hubIds)

            // PiP button (shown on hover)
            if let pipAction {
                Button {
                    pipAction()
                } label: {
                    Image(systemName: isFloating ? "pip.exit" : "pip.enter")
                        .font(.system(size: 10))
                        .foregroundStyle(.white.opacity(0.7))
                        .padding(4)
                        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 4))
                }
                .buttonStyle(.plain)
                .padding(4)
                .opacity(isHovered ? 1 : 0)
                .help(isFloating ? "Dock minimap" : "Pop out minimap")
            }
        }
        .background {
            if isFloating {
                RoundedRectangle(cornerRadius: 12).fill(.ultraThinMaterial)
            } else {
                RoundedRectangle(cornerRadius: 8).fill(Color(red: 0.051, green: 0.067, blue: 0.09).opacity(0.85))
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: isFloating ? 12 : 8))
        .overlay(
            RoundedRectangle(cornerRadius: isFloating ? 12 : 8)
                .strokeBorder(.white.opacity(isFloating ? 0.2 : 0.15), lineWidth: 1)
        )
        .shadow(color: .black.opacity(isFloating ? 0.5 : 0), radius: 12)
        .onHover { isHovered = $0 }
        .onReceive(timer) { _ in
            // Always tick — 10fps is trivial. Ensures Canvas reads fresh renderStore data
            // (glowingNodes, newNodeGlows, dyingNodes, nodes) every frame.
            // The old condition checked captured `let` values that were stale at construction.
            frameCount &+= 1
        }
    }

    @ViewBuilder
    private func minimapCanvas(nodes: [NodeData], hubIds: Set<UUID>) -> some View {
        let is3D = camera3DState != nil
        let canvas = Canvas { context, size in
            let canvasStart = CFAbsoluteTimeGetCurrent()
            let _ = frameCount
            // In 3D mode, project XZ plane; in 2D, use force simulation positions
            let positions: [UUID: CGPoint]
            if let pos3D = camera3DState?.positions {
                var projected: [UUID: CGPoint] = [:]
                for (id, p) in pos3D {
                    projected[id] = CGPoint(x: CGFloat(p.x), y: CGFloat(p.z))
                }
                positions = projected
            } else {
                positions = simulation.positions
            }
            guard !positions.isEmpty else { return }

            let bounds = computeBounds(positions: positions)
            let mapScale = min(size.width / bounds.width, size.height / bounds.height)

            let glows = galaxyRegistry?.mergedGlowingNodes ?? renderStore.glowingNodes
            let arrivals = galaxyRegistry?.mergedNewNodeGlows ?? renderStore.newNodeGlows
            let now = Date()
            struct MiniDot {
                let mx: CGFloat; let my: CGFloat; let dotSize: CGFloat
                let color: Color; let ri: CGFloat; let ai: CGFloat
            }
            var dots: [MiniDot] = []
            let dotScale: CGFloat = min(size.width, size.height) / 120 // scale dots proportionally
            for node in nodes {
                guard let pos = positions[node.id] else { continue }
                let mx = (pos.x - bounds.minX) * mapScale
                let my = (pos.y - bounds.minY) * mapScale
                let colorMap = galaxyRegistry?.mergedColorMap ?? renderStore.colorMap
                let color = GraphView.projectColor(for: node.project, in: colorMap)
                let isHub = hubIds.contains(node.id)
                let dotSize: CGFloat = (isHub ? 4 : 2.5) * dotScale
                let ri: CGFloat
                if let glowStart = glows[node.id] {
                    let elapsed = now.timeIntervalSince(glowStart)
                    let fadeIn: CGFloat = 0.3, hold: CGFloat = 1.5, fadeOut: CGFloat = 2.0
                    let total = fadeIn + hold + fadeOut
                    if elapsed < Double(fadeIn) {
                        let t = CGFloat(elapsed) / fadeIn; ri = t * t
                    } else if elapsed < Double(fadeIn + hold) {
                        ri = 1.0
                    } else if elapsed < Double(total) {
                        let t = 1.0 - (CGFloat(elapsed) - fadeIn - hold) / fadeOut; ri = t * t
                    } else { ri = 0 }
                } else { ri = 0 }
                let ai: CGFloat
                if let arrivalTime = arrivals[node.id] {
                    let elapsed = now.timeIntervalSince(arrivalTime)
                    let fadeIn: CGFloat = 0.5, hold: CGFloat = 2.0, fadeOut: CGFloat = 3.0
                    let total = fadeIn + hold + fadeOut
                    if elapsed < Double(fadeIn) {
                        let t = CGFloat(elapsed) / fadeIn; ai = t * t
                    } else if elapsed < Double(fadeIn + hold) {
                        ai = 1.0
                    } else if elapsed < Double(total) {
                        let t = 1.0 - (CGFloat(elapsed) - fadeIn - hold) / fadeOut; ai = t * t
                    } else { ai = 0 }
                } else { ai = 0 }
                dots.append(MiniDot(mx: mx, my: my, dotSize: dotSize, color: color, ri: ri, ai: ai))
            }

            // Arrival bloom (golden-orange)
            for d in dots where d.ai > 0 {
                let bloomSize = d.dotSize + 8
                let bloomRect = CGRect(x: d.mx - bloomSize / 2, y: d.my - bloomSize / 2, width: bloomSize, height: bloomSize)
                context.fill(Circle().path(in: bloomRect), with: .color(Color(red: 1.0, green: 0.7, blue: 0.2).opacity(0.45 * d.ai)))
            }
            // Recall bloom (white-blue)
            for d in dots where d.ri > 0 {
                let bloomSize = d.dotSize + 6
                let bloomRect = CGRect(x: d.mx - bloomSize / 2, y: d.my - bloomSize / 2, width: bloomSize, height: bloomSize)
                context.fill(Circle().path(in: bloomRect), with: .color(Color(red: 0.6, green: 0.85, blue: 1.0).opacity(0.5 * d.ri)))
            }

            for d in dots {
                let rect = CGRect(x: d.mx - d.dotSize / 2, y: d.my - d.dotSize / 2, width: d.dotSize, height: d.dotSize)
                context.fill(Circle().path(in: rect), with: .color(d.color.opacity(0.8)))
                if d.ai > 0 {
                    let golden = Color(red: 1.0, green: 0.7, blue: 0.2)
                    context.fill(Circle().path(in: rect), with: .color(golden.opacity(0.8 * d.ai)))
                }
                if d.ri > 0 {
                    let hotWhite = Color(red: 0.9, green: 0.95, blue: 1.0)
                    context.fill(Circle().path(in: rect), with: .color(hotWhite.opacity(0.85 * d.ri)))
                }
            }

            // Dying node ghosts (red to black fade-out) — 2D only, ghosts store CGPoint positions
            let dying: [UUID: DyingNode] = is3D ? [:] : (galaxyRegistry?.mergedDyingNodes ?? renderStore.dyingNodes)
            for (_, ghost) in dying {
                let elapsed = now.timeIntervalSince(ghost.startTime)
                let flashIn: CGFloat = 0.3, hold: CGFloat = 1.2, fadeOutD: CGFloat = 1.5
                let total = flashIn + hold + fadeOutD
                guard elapsed < Double(total) else { continue }
                let di: CGFloat
                if elapsed < Double(flashIn) {
                    let t = CGFloat(elapsed) / flashIn; di = t * t
                } else if elapsed < Double(flashIn + hold) {
                    di = 1.0
                } else {
                    let t = 1.0 - (CGFloat(elapsed) - flashIn - hold) / fadeOutD; di = t * t
                }
                let mx = (ghost.position.x - bounds.minX) * mapScale
                let my = (ghost.position.y - bounds.minY) * mapScale
                let dotSize: CGFloat = (ghost.isHub ? 4 : 2.5) * dotScale
                // Red bloom
                let bloomSize = dotSize + 8
                let bloomRect = CGRect(x: mx - bloomSize / 2, y: my - bloomSize / 2, width: bloomSize, height: bloomSize)
                context.fill(Circle().path(in: bloomRect), with: .color(Color(red: 0.9, green: 0.15, blue: 0.1).opacity(0.5 * di)))
                // Red-to-dark dot
                let darkening = elapsed > Double(flashIn) ? min(1.0, (CGFloat(elapsed) - flashIn) / (hold + fadeOutD)) : 0.0
                let r = dotSize * (elapsed > Double(flashIn + hold) ? 1.0 - (1.0 - di) * 0.5 : 1.0)
                let rect = CGRect(x: mx - r / 2, y: my - r / 2, width: r, height: r)
                let red = Color(red: 0.9 * (1.0 - darkening * 0.8), green: 0.15 * (1.0 - darkening), blue: 0.1 * (1.0 - darkening))
                context.fill(Circle().path(in: rect), with: .color(red.opacity(di)))
            }

            if is3D, let camPos = camera3DState?.position, let camTarget = camera3DState?.target, !isFloating {
                // Camera chevron — shows position and look direction on XZ plane
                let camMx = (CGFloat(camPos.x) - bounds.minX) * mapScale
                let camMy = (CGFloat(camPos.z) - bounds.minY) * mapScale

                // Look direction on XZ plane
                let dx = CGFloat(camTarget.x - camPos.x)
                let dz = CGFloat(camTarget.z - camPos.z)
                let angle = atan2(dz, dx)  // angle in minimap space (X right, Y down=Z+)

                // Draw chevron (triangle pointing in look direction)
                let chevronSize: CGFloat = 8 * dotScale
                var chevron = Path()
                // Tip (front)
                chevron.move(to: CGPoint(
                    x: camMx + cos(angle) * chevronSize,
                    y: camMy + sin(angle) * chevronSize
                ))
                // Left wing
                let wingAngle = angle + .pi * 0.75
                chevron.addLine(to: CGPoint(
                    x: camMx + cos(wingAngle) * chevronSize * 0.7,
                    y: camMy + sin(wingAngle) * chevronSize * 0.7
                ))
                // Notch (back center)
                let backAngle = angle + .pi
                chevron.addLine(to: CGPoint(
                    x: camMx + cos(backAngle) * chevronSize * 0.25,
                    y: camMy + sin(backAngle) * chevronSize * 0.25
                ))
                // Right wing
                let rightWingAngle = angle - .pi * 0.75
                chevron.addLine(to: CGPoint(
                    x: camMx + cos(rightWingAngle) * chevronSize * 0.7,
                    y: camMy + sin(rightWingAngle) * chevronSize * 0.7
                ))
                chevron.closeSubpath()

                context.fill(chevron, with: .color(.white.opacity(0.9)))
                context.stroke(chevron, with: .color(.white.opacity(0.4)), lineWidth: 0.5)
            } else if !is3D {
                // 2D viewport rectangle
                let vpWorldX = -viewport.offset.x / viewport.scale
                let vpWorldY = -viewport.offset.y / viewport.scale
                let vpWorldW = viewportSize.width / viewport.scale
                let vpWorldH = viewportSize.height / viewport.scale

                let vpRect = CGRect(
                    x: (vpWorldX - bounds.minX) * mapScale,
                    y: (vpWorldY - bounds.minY) * mapScale,
                    width: vpWorldW * mapScale,
                    height: vpWorldH * mapScale
                )
                context.stroke(
                    Rectangle().path(in: vpRect),
                    with: .color(.white.opacity(0.5)),
                    lineWidth: 1
                )
            }
            let canvasMs = (CFAbsoluteTimeGetCurrent() - canvasStart) * 1000.0
            if frameCount % 60 == 0 || canvasMs > 3 {
                minimapLog.error("[MINIMAP] draw=\(canvasMs, format: .fixed(precision: 2))ms nodes=\(nodes.count) frame=\(frameCount)")
            }
        }

        if isFloating {
            // Fill panel — Canvas adapts to whatever size the panel is
            GeometryReader { geo in
                canvas
                    .onAppear { renderedSize = geo.size }
                    .onChange(of: geo.size) { _, s in renderedSize = s }
                    .onTapGesture { location in
                        navigateTo(location: location, in: renderedSize)
                    }
            }
        } else {
            canvas
                .frame(width: inlineSize.width, height: inlineSize.height)
                .onTapGesture { location in
                    navigateTo(location: location, in: inlineSize)
                }
        }
    }

    private func navigateTo(location: CGPoint, in size: CGSize) {
        // Tap-to-navigate only works in 2D mode
        guard camera3DState == nil else { return }
        let positions = simulation.positions
        guard !positions.isEmpty else { return }
        let bounds = computeBounds(positions: positions)
        let mapScale = min(size.width / bounds.width, size.height / bounds.height)
        let worldX = location.x / mapScale + bounds.minX
        let worldY = location.y / mapScale + bounds.minY
        withAnimation(.easeInOut(duration: 0.3)) {
            viewport.offset = CGPoint(
                x: viewportSize.width / 2 - worldX * viewport.scale,
                y: viewportSize.height / 2 - worldY * viewport.scale
            )
        }
    }

    private func computeBounds(positions: [UUID: CGPoint]) -> (minX: CGFloat, minY: CGFloat, width: CGFloat, height: CGFloat) {
        var minX: CGFloat = .greatestFiniteMagnitude
        var minY: CGFloat = .greatestFiniteMagnitude
        var maxX: CGFloat = -.greatestFiniteMagnitude
        var maxY: CGFloat = -.greatestFiniteMagnitude
        for (_, pos) in positions {
            minX = min(minX, pos.x); minY = min(minY, pos.y)
            maxX = max(maxX, pos.x); maxY = max(maxY, pos.y)
        }
        let padding: CGFloat = 50
        return (minX - padding, minY - padding, max(maxX - minX + padding * 2, 1), max(maxY - minY + padding * 2, 1))
    }
}
