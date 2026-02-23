import SwiftUI
import AppKit
import os

private let minimapLog = Logger(subsystem: "com.claudememory.visualizer", category: "Minimap")

// MARK: - Inline minimap frame preference (for arc animation source/dest)

struct InlineMinimapFrameKey: PreferenceKey {
    static let defaultValue: CGRect = .zero
    static func reduce(value: inout CGRect, nextValue: () -> CGRect) {
        value = nextValue()
    }
}

// MARK: - PiP Mouse Tracker (cursor rects + mouseMoved for non-key panels)

private class PiPMouseTracker: NSView {
    var edgeMargin: CGFloat = 14

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        for area in trackingAreas { removeTrackingArea(area) }
        addTrackingArea(NSTrackingArea(
            rect: bounds,
            options: [.mouseMoved, .mouseEnteredAndExited, .cursorUpdate, .activeAlways],
            owner: self, userInfo: nil
        ))
    }

    override func cursorUpdate(with event: NSEvent) {
        let local = convert(event.locationInWindow, from: nil)
        let m = edgeMargin
        let w = bounds.width, h = bounds.height
        let l = local.x < m, r = local.x > w - m
        let b = local.y < m, t = local.y > h - m
        if (l || r) && (b || t) {
            // NWSE diagonal: topLeft + bottomRight; NESW: topRight + bottomLeft
            if (t && l) || (b && r) {
                MinimapPanelController.nwseCursor.set()
            } else {
                MinimapPanelController.neswCursor.set()
            }
        } else {
            NSCursor.arrow.set()
        }
    }

    override func hitTest(_ point: NSPoint) -> NSView? {
        // Claim hits in corner zones so cursorUpdate fires; pass through elsewhere
        let local = convert(point, from: superview)
        let m = edgeMargin
        let w = bounds.width, h = bounds.height
        let inCorner = (local.x < m || local.x > w - m) && (local.y < m || local.y > h - m)
        return inCorner ? self : nil
    }
}

// MARK: - PiP Panel (simple subclass — all interaction handled by controller's event monitors)

private class PiPPanel: NSPanel {
    func snapToNearestCorner() {
        guard let screen = screen else {
            print("[PiP] snapToNearestCorner: screen is nil!")
            return
        }
        let v = screen.visibleFrame
        let pad: CGFloat = 20
        let f = frame
        let cx = f.midX, cy = f.midY

        let corners: [NSPoint] = [
            NSPoint(x: v.minX + pad, y: v.maxY - f.height - pad),
            NSPoint(x: v.maxX - f.width - pad, y: v.maxY - f.height - pad),
            NSPoint(x: v.minX + pad, y: v.minY + pad),
            NSPoint(x: v.maxX - f.width - pad, y: v.minY + pad),
        ]

        guard let nearest = corners.min(by: {
            hypot($0.x + f.width / 2 - cx, $0.y + f.height / 2 - cy) <
            hypot($1.x + f.width / 2 - cx, $1.y + f.height / 2 - cy)
        }) else {
            print("[PiP] snapToNearestCorner: no nearest corner found!")
            return
        }

        print("[PiP] snapToNearestCorner: from \(f.origin) → \(nearest)")
        setFrame(NSRect(origin: nearest, size: f.size), display: true, animate: true)
    }
}

// MARK: - Minimap PiP Panel Controller

@MainActor
final class MinimapPanelController {
    private var panel: PiPPanel?
    private var hostingView: NSHostingView<AnyView>?
    private var windowCloseObserver: Any?
    private var mouseDownMonitor: Any?
    private var dragMonitor: Any?
    private var animationTimer: Timer?
    private weak var mainWindowRef: NSWindow?

    private let edgeMargin: CGFloat = 14
    private let pipAspect: CGFloat = 4.0 / 3.0
    private let minWidth: CGFloat = 200
    private let maxWidth: CGFloat = 500

    private enum Corner { case topLeft, topRight, bottomLeft, bottomRight }

    // System diagonal resize cursors (no public API — use standard private selectors)
    fileprivate static let nwseCursor: NSCursor = {
        let sel = NSSelectorFromString("_windowResizeNorthWestSouthEastCursor")
        if NSCursor.responds(to: sel),
           let result = NSCursor.perform(sel)?.takeUnretainedValue() as? NSCursor {
            return result
        }
        return .crosshair
    }()

    fileprivate static let neswCursor: NSCursor = {
        let sel = NSSelectorFromString("_windowResizeNorthEastSouthWestCursor")
        if NSCursor.responds(to: sel),
           let result = NSCursor.perform(sel)?.takeUnretainedValue() as? NSCursor {
            return result
        }
        return .crosshair
    }()

    private func cursorForCorner(_ c: Corner) -> NSCursor {
        switch c {
        case .topLeft, .bottomRight: Self.nwseCursor
        case .topRight, .bottomLeft: Self.neswCursor
        }
    }

    private func corner(at screenPoint: NSPoint) -> Corner? {
        guard let f = panel?.frame else { return nil }
        let local = NSPoint(x: screenPoint.x - f.minX, y: screenPoint.y - f.minY)
        let l = local.x < edgeMargin, r = local.x > f.width - edgeMargin
        let b = local.y < edgeMargin, t = local.y > f.height - edgeMargin
        if b && r { return .bottomRight }
        if b && l { return .bottomLeft }
        if t && r { return .topRight }
        if t && l { return .topLeft }
        return nil
    }

    var isDetached: Bool { panel != nil }

    /// SwiftUI global frame of the inline minimap (set via GeometryReader preference)
    var inlineFrame: CGRect = .zero

    /// Convert SwiftUI global frame → AppKit screen coordinates.
    /// Falls back to approximate position if PreferenceKey hasn't fired yet.
    private func inlineMinimapScreenRect() -> NSRect? {
        guard let window = mainWindowRef ?? NSApp.keyWindow ?? NSApp.mainWindow else { return nil }
        let contentRect = window.contentRect(forFrameRect: window.frame)

        if inlineFrame.width > 0 {
            // Exact position from GeometryReader preference
            return NSRect(
                x: contentRect.minX + inlineFrame.origin.x,
                y: contentRect.maxY - inlineFrame.maxY,
                width: inlineFrame.width,
                height: inlineFrame.height
            )
        }

        // Fallback: approximate inline minimap position (bottom-left)
        // Layout: 12px left pad, 56px from bottom (44px time slider + 12px view padding)
        return NSRect(
            x: contentRect.minX + 12,
            y: contentRect.minY + 56,
            width: 160,
            height: 120
        )
    }

    func show<V: View>(content: V, relativeTo mainWindow: NSWindow?) {
        if let hostingView {
            hostingView.rootView = AnyView(content)
            return
        }

        mainWindowRef = mainWindow ?? NSApp.keyWindow ?? NSApp.mainWindow
        let panelSize = NSSize(width: 280, height: 210)
        let panelRect = NSRect(origin: .zero, size: panelSize)

        // Container with rounded corners holds hosting + mouse tracker
        let container = NSView(frame: panelRect)
        container.wantsLayer = true
        container.layer?.backgroundColor = .clear
        container.layer?.cornerRadius = 12
        container.layer?.masksToBounds = true

        let hosting = NSHostingView(rootView: AnyView(content))
        hosting.frame = panelRect
        hosting.autoresizingMask = [.width, .height]
        container.addSubview(hosting)

        // Mouse tracker on top handles cursor rects for corner resize zones
        let tracker = PiPMouseTracker(frame: panelRect)
        tracker.edgeMargin = edgeMargin
        tracker.autoresizingMask = [.width, .height]
        container.addSubview(tracker)

        let p = PiPPanel(
            contentRect: panelRect,
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        p.isFloatingPanel = true
        p.level = .floating
        p.backgroundColor = .clear
        p.isOpaque = false
        p.hasShadow = false
        p.isReleasedWhenClosed = false
        p.isMovableByWindowBackground = false
        p.contentView = container

        // Compute target corner position
        let targetOrigin: NSPoint
        if let screen = mainWindow?.screen ?? NSScreen.main {
            let visible = screen.visibleFrame
            targetOrigin = NSPoint(x: visible.minX + 20, y: visible.minY + 20)
        } else {
            targetOrigin = .zero
        }
        let targetFrame = NSRect(origin: targetOrigin, size: panelSize)

        // Close panel when main window closes
        if let mainWindow {
            windowCloseObserver = NotificationCenter.default.addObserver(
                forName: NSWindow.willCloseNotification, object: mainWindow, queue: .main
            ) { [weak self] _ in
                Task { @MainActor in self?.dismiss() }
            }
        }

        panel = p
        hostingView = hosting
        installEventMonitors()

        // Arc animation from inline minimap to corner
        if let sourceFrame = inlineMinimapScreenRect() {
            p.setFrame(sourceFrame, display: true)
            p.alphaValue = 1
            p.orderFront(nil)
            animateArc(from: sourceFrame, to: targetFrame, appearing: true)
        } else {
            p.setFrame(targetFrame, display: true)
            p.alphaValue = 1
            p.orderFront(nil)
        }
    }

    /// Animated dismiss with arc back to inline minimap position.
    /// Calls completion when done (use to set minimapDetached = false).
    func animatedDismiss(completion: @escaping () -> Void) {
        removeEventMonitors()
        if let obs = windowCloseObserver {
            NotificationCenter.default.removeObserver(obs)
            windowCloseObserver = nil
        }

        guard let p = panel else {
            cleanup()
            completion()
            return
        }

        let sourceFrame = p.frame
        if let targetFrame = inlineMinimapScreenRect() {
            animateArc(from: sourceFrame, to: targetFrame, appearing: false) { [weak self] in
                self?.cleanup()
                completion()
            }
        } else {
            cleanup()
            completion()
        }
    }

    // MARK: - Event monitors (controller-level, bypasses sendEvent entirely)

    private func installEventMonitors() {
        mouseDownMonitor = NSEvent.addLocalMonitorForEvents(
            matching: [.leftMouseDown, .mouseMoved]
        ) { [weak self] event in
            guard let self, let panel = self.panel else { return event }

            switch event.type {
            case .mouseMoved:
                let loc = NSEvent.mouseLocation
                guard NSMouseInRect(loc, panel.frame, false) else {
                    NSCursor.arrow.set() // left the panel — clear any stuck resize cursor
                    return event
                }
                if let c = self.corner(at: loc) {
                    self.cursorForCorner(c).set()
                    return nil // consume so NSHostingView can't override cursor
                }
                NSCursor.arrow.set() // in panel but not a corner — clear resize cursor
                return event

            case .leftMouseDown:
                let loc = NSEvent.mouseLocation
                let inPanel = NSMouseInRect(loc, panel.frame, false)
                print("[PiP] mouseDown at \(loc), panel frame: \(panel.frame), inPanel: \(inPanel)")
                guard inPanel else { return event }
                let isCorner = self.corner(at: loc) != nil
                self.startDrag(from: loc, isResize: isCorner)
                return nil // consume ALL panel clicks to prevent built-in drag competition

            default:
                return event
            }
        }
    }

    private func startDrag(from startLoc: NSPoint, isResize: Bool) {
        guard let panel else { return }
        let startFrame = panel.frame
        let resizeCorner = isResize ? corner(at: startLoc) : nil
        var didDrag = false

        print("[PiP] startDrag: isResize=\(isResize), resizeCorner=\(String(describing: resizeCorner))")
        if let c = resizeCorner { cursorForCorner(c).set() }

        dragMonitor = NSEvent.addLocalMonitorForEvents(matching: [.leftMouseDragged, .leftMouseUp]) {
            [weak self] event in
            guard let self, let panel = self.panel else { return event }

            if event.type == .leftMouseUp {
                print("[PiP] mouseUp detected, didDrag=\(didDrag)")
                self.stopDrag()
                NSCursor.arrow.set()

                if didDrag {
                    panel.snapToNearestCorner()
                } else {
                    // Was a click, not a drag — forward to panel for SwiftUI button handling
                    print("[PiP] forwarding click to panel")
                    if let down = NSEvent.mouseEvent(
                        with: .leftMouseDown, location: event.locationInWindow,
                        modifierFlags: event.modifierFlags, timestamp: event.timestamp - 0.01,
                        windowNumber: panel.windowNumber, context: nil,
                        eventNumber: 0, clickCount: 1, pressure: 1.0
                    ) {
                        panel.sendEvent(down)
                    }
                    panel.sendEvent(event)
                }
                return nil
            }

            // leftMouseDragged
            didDrag = true
            let cur = NSEvent.mouseLocation
            if let c = resizeCorner {
                let dx = cur.x - startLoc.x
                let widthDelta: CGFloat = switch c {
                case .topRight, .bottomRight: dx
                case .topLeft, .bottomLeft: -dx
                }

                let newWidth = max(self.minWidth, min(self.maxWidth, startFrame.width + widthDelta))
                let newHeight = newWidth / self.pipAspect

                var origin = startFrame.origin
                switch c {
                case .bottomRight: origin.y = startFrame.maxY - newHeight
                case .bottomLeft:
                    origin.x = startFrame.maxX - newWidth
                    origin.y = startFrame.maxY - newHeight
                case .topRight: break
                case .topLeft: origin.x = startFrame.maxX - newWidth
                }

                panel.setFrame(NSRect(origin: origin,
                                      size: NSSize(width: newWidth, height: newHeight)),
                               display: true)
            } else {
                // Manual drag
                panel.setFrameOrigin(NSPoint(
                    x: startFrame.origin.x + (cur.x - startLoc.x),
                    y: startFrame.origin.y + (cur.y - startLoc.y)
                ))
            }
            return nil // consume drag events
        }
    }

    private func stopDrag() {
        if let m = dragMonitor {
            NSEvent.removeMonitor(m)
            dragMonitor = nil
        }
    }

    private func removeEventMonitors() {
        if let m = mouseDownMonitor { NSEvent.removeMonitor(m); mouseDownMonitor = nil }
        stopDrag()
    }

    /// Instant dismiss (no animation) — used by bridge cleanup and window close.
    func dismiss() {
        animationTimer?.invalidate()
        animationTimer = nil
        removeEventMonitors()
        if let obs = windowCloseObserver {
            NotificationCenter.default.removeObserver(obs)
            windowCloseObserver = nil
        }
        cleanup()
    }

    private func cleanup() {
        panel?.close()
        panel = nil
        hostingView = nil
    }

    // MARK: - Arc animation (quadratic bezier + scale tween)

    private func animateArc(from source: NSRect, to dest: NSRect, appearing: Bool, completion: (() -> Void)? = nil) {
        guard let panel else { completion?(); return }
        animationTimer?.invalidate()

        let duration: TimeInterval = 0.4
        let startTime = CACurrentMediaTime()

        // Bezier on rect centers for a smooth arc
        let srcCenter = NSPoint(x: source.midX, y: source.midY)
        let dstCenter = NSPoint(x: dest.midX, y: dest.midY)

        // Control point arcs upward proportional to distance
        let midX = (srcCenter.x + dstCenter.x) / 2
        let maxY = max(srcCenter.y, dstCenter.y)
        let dist = hypot(dstCenter.x - srcCenter.x, dstCenter.y - srcCenter.y)
        let arcHeight = max(80, dist * 0.35)
        let controlPt = NSPoint(x: midX, y: maxY + arcHeight)

        animationTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 60.0, repeats: true) {
            [weak self] timer in
            guard let self, let panel = self.panel else {
                timer.invalidate()
                completion?()
                return
            }

            let elapsed = CACurrentMediaTime() - startTime
            let linear = CGFloat(min(elapsed / duration, 1.0))

            // Ease-in-out cubic
            let t: CGFloat = linear < 0.5
                ? 4 * linear * linear * linear
                : 1 - pow(-2 * linear + 2, 3) / 2

            // Quadratic bezier for center position
            let cx = (1 - t) * (1 - t) * srcCenter.x + 2 * (1 - t) * t * controlPt.x + t * t * dstCenter.x
            let cy = (1 - t) * (1 - t) * srcCenter.y + 2 * (1 - t) * t * controlPt.y + t * t * dstCenter.y

            // Linear interpolation for size
            let w = source.width + t * (dest.width - source.width)
            let h = source.height + t * (dest.height - source.height)

            // Alpha: quick fade-in for appearing; stay opaque for dismiss (inline minimap replaces it)
            if appearing {
                panel.alphaValue = min(1, 0.4 + t * 0.6)
            }

            panel.setFrame(
                NSRect(x: cx - w / 2, y: cy - h / 2, width: w, height: h),
                display: true
            )

            if linear >= 1.0 {
                timer.invalidate()
                self.animationTimer = nil
                panel.setFrame(dest, display: true)
                if appearing { panel.alphaValue = 1 }
                completion?()
            }
        }
    }
}

// MARK: - Minimap PiP Bridge (invisible NSViewRepresentable that syncs panel state)

struct MinimapPanelBridge<Content: View>: NSViewRepresentable {
    let isDetached: Bool
    let panel: MinimapPanelController
    let content: Content
    let onDismiss: () -> Void

    func makeNSView(context: Context) -> NSView {
        NSView(frame: .zero)
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        if isDetached {
            panel.show(content: content.environment(\.colorScheme, .dark), relativeTo: nsView.window)
        } else {
            panel.dismiss()
        }
    }
}

// MARK: - Stats Overlay

struct StatsOverlay: View {
    let visibleMemoryCount: Int
    let visibleEdgeCount: Int
    let totalMemories: Int
    let hiddenProjects: Set<String>
    let hiddenRelations: Set<String>
    let projects: [String]
    let allRelationCounts: [(key: String, value: Int)]
    let toggleProject: (String) -> Void
    let toggleRelation: (String) -> Void
    let colorMap: [String: Color]

    private var dbFileSize: String {
        let dbPath = ProcessInfo.processInfo.environment["CLAUDE_MEMORY_DB"]
            ?? NSHomeDirectory() + "/.claude/memory.sqlite"
        let fm = FileManager.default
        var total: Int64 = 0
        for path in [dbPath, dbPath + "-wal", dbPath + "-shm"] {
            if let attrs = try? fm.attributesOfItem(atPath: path),
               let size = attrs[.size] as? Int64 {
                total += size
            }
        }
        guard total > 0 else { return "—" }
        if total < 1024 { return "\(total) B" }
        let kb = Double(total) / 1024
        if kb < 1024 { return String(format: "%.1f KB", kb) }
        let mb = kb / 1024
        return String(format: "%.1f MB", mb)
    }

    var body: some View {
        VStack(alignment: .trailing, spacing: 6) {
            if visibleMemoryCount < totalMemories {
                Text("\(visibleMemoryCount)/\(totalMemories) memories")
                    .font(.system(size: 13, weight: .medium, design: .monospaced))
            } else {
                Text("\(totalMemories) memories")
                    .font(.system(size: 13, weight: .medium, design: .monospaced))
            }
            Text("\(visibleEdgeCount) edges")
                .font(.system(size: 13, weight: .medium, design: .monospaced))
            Text("db: \(dbFileSize)")
                .font(.system(size: 11, design: .monospaced))

            Divider().frame(width: 100).overlay(Color.white.opacity(0.2))

            ScrollView(.vertical, showsIndicators: false) {
                VStack(alignment: .trailing, spacing: 6) {
                    // Project filters
                    ForEach(projects, id: \.self) { project in
                        Button {
                            toggleProject(project)
                        } label: {
                            HStack(spacing: 6) {
                                Circle()
                                    .fill(GraphView.projectColor(for: project, in: colorMap))
                                    .frame(width: 8, height: 8)
                                    .opacity(hiddenProjects.contains(project) ? 0.3 : 1.0)
                                Text(project)
                                    .font(.system(size: 11, design: .monospaced))
                                    .strikethrough(hiddenProjects.contains(project))
                            }
                        }
                        .buttonStyle(.plain)
                    }

                    // Edge type filters (use unfiltered counts so hidden types remain visible)
                    if !allRelationCounts.isEmpty {
                        Divider().frame(width: 100).overlay(Color.white.opacity(0.2))

                        ForEach(allRelationCounts, id: \.key) { relation, count in
                            Button {
                                toggleRelation(relation)
                            } label: {
                                HStack(spacing: 6) {
                                    Circle()
                                        .fill(GraphCanvas.relationColors[relation] ?? .white)
                                        .frame(width: 8, height: 8)
                                        .opacity(hiddenRelations.contains(relation) ? 0.3 : 1.0)
                                    Text("\(relation.replacingOccurrences(of: "_", with: " ")) (\(count))")
                                        .font(.system(size: 11, design: .monospaced))
                                        .strikethrough(hiddenRelations.contains(relation))
                                }
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
            }
            .frame(maxHeight: 300)
        }
        .foregroundStyle(.white.opacity(0.7))
        .padding(12)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 10))
    }
}

// MARK: - Minimap

struct MinimapView: View {
    let filteredNodes: [NodeData]
    let hubs: Set<Int64>
    let glowingNodes: [Int64: Date]
    let newNodes: [Int64: Date]
    let dyingNodes: [Int64: DyingNode]
    let simulation: ForceSimulation
    let viewport: ViewportState
    let viewportSize: CGSize
    let colorMap: [String: Color]
    var pipAction: (() -> Void)?
    var isFloating: Bool = false

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
        let nodes = filteredNodes
        let hubIds = hubs
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
            if camera3DState != nil || simulation.isActive || !glowingNodes.isEmpty || !newNodes.isEmpty || !dyingNodes.isEmpty {
                frameCount &+= 1
            }
        }
    }

    @ViewBuilder
    private func minimapCanvas(nodes: [NodeData], hubIds: Set<Int64>) -> some View {
        let is3D = camera3DState != nil
        let canvas = Canvas { context, size in
            let canvasStart = CFAbsoluteTimeGetCurrent()
            let _ = frameCount
            // In 3D mode, project XZ plane; in 2D, use force simulation positions
            let positions: [Int64: CGPoint]
            if let pos3D = camera3DState?.positions {
                var projected: [Int64: CGPoint] = [:]
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

            let glows = glowingNodes
            let arrivals = newNodes
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
            let dying = is3D ? [:] : dyingNodes
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

    private func computeBounds(positions: [Int64: CGPoint]) -> (minX: CGFloat, minY: CGFloat, width: CGFloat, height: CGFloat) {
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

// MARK: - Time Slider

struct TimeSliderBar: View {
    let earliestDate: Date
    let latestDate: Date
    @Binding var sliderDate: Date?
    @Binding var isPlaying: Bool

    @State private var sliderValue: Double = 1.0

    private var timeRange: TimeInterval {
        max(latestDate.timeIntervalSince(earliestDate), 1)
    }

    var body: some View {
        HStack(spacing: 12) {
            Button {
                if isPlaying {
                    isPlaying = false
                } else {
                    if sliderValue >= 1.0 { sliderValue = 0.0 }
                    isPlaying = true
                }
            } label: {
                Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                    .font(.system(size: 12))
                    .foregroundStyle(.white.opacity(0.7))
            }
            .buttonStyle(.plain)

            Text(dateLabel)
                .font(.system(size: 10, weight: .medium, design: .monospaced))
                .foregroundStyle(.white.opacity(0.5))
                .frame(width: 80, alignment: .leading)

            Slider(value: $sliderValue, in: 0...1)
                .tint(.cyan.opacity(0.6))
                .onChange(of: sliderValue) { _, newVal in
                    if newVal >= 1.0 {
                        sliderDate = nil
                        isPlaying = false
                    } else {
                        let interval = newVal * timeRange
                        sliderDate = earliestDate.addingTimeInterval(interval)
                    }
                }

            Button {
                sliderValue = 1.0
                sliderDate = nil
                isPlaying = false
            } label: {
                Text("All")
                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.5))
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(Color(red: 0.051, green: 0.067, blue: 0.09).opacity(0.9))
        .task(id: isPlaying) {
            guard isPlaying else { return }
            while !Task.isCancelled && isPlaying && sliderValue < 1.0 {
                try? await Task.sleep(for: .milliseconds(50))
                sliderValue = min(1.0, sliderValue + 0.005)
            }
            if sliderValue >= 1.0 {
                isPlaying = false
                sliderDate = nil
            }
        }
    }

    private var dateLabel: String {
        if let date = sliderDate {
            let formatter = DateFormatter()
            formatter.dateStyle = .short
            return formatter.string(from: date)
        }
        return "All time"
    }
}

// MARK: - Layout Mode Picker

struct LayoutModePicker: View {
    let mode: LayoutMode
    let projectionState: ProjectionState
    let onModeChange: (LayoutMode) -> Void

    var body: some View {
        HStack(spacing: 0) {
            ForEach(LayoutMode.allCases, id: \.rawValue) { m in
                Button {
                    onModeChange(m)
                } label: {
                    HStack(spacing: 4) {
                        if m == .embedding, case .computing(let progress) = projectionState {
                            // Circular progress indicator
                            ZStack {
                                Circle()
                                    .stroke(.white.opacity(0.15), lineWidth: 1.5)
                                Circle()
                                    .trim(from: 0, to: progress)
                                    .stroke(.cyan.opacity(0.8), style: StrokeStyle(lineWidth: 1.5, lineCap: .round))
                                    .rotationEffect(.degrees(-90))
                            }
                            .frame(width: 10, height: 10)
                        }
                        Text(m.rawValue)
                            .font(.system(size: 11, weight: mode == m ? .semibold : .regular, design: .monospaced))
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    .background(
                        RoundedRectangle(cornerRadius: 5)
                            .fill(mode == m ? .white.opacity(0.12) : .clear)
                    )
                }
                .buttonStyle(.plain)
                .foregroundStyle(.white.opacity(mode == m ? 0.9 : 0.4))
            }
        }
        .padding(2)
        .background(
            RoundedRectangle(cornerRadius: 7)
                .fill(Color(red: 0.08, green: 0.1, blue: 0.14))
                .overlay(
                    RoundedRectangle(cornerRadius: 7)
                        .strokeBorder(.white.opacity(0.1), lineWidth: 1)
                )
        )
    }
}

// MARK: - Dimension Toggle

struct DimensionToggle: View {
    let mode: DimensionMode
    let onModeChange: (DimensionMode) -> Void

    var body: some View {
        HStack(spacing: 0) {
            ForEach(DimensionMode.allCases, id: \.rawValue) { m in
                Button {
                    onModeChange(m)
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: m == .twoD ? "square" : "cube")
                            .font(.system(size: 10))
                        Text(m.rawValue)
                            .font(.system(size: 11, weight: mode == m ? .semibold : .regular, design: .monospaced))
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 5)
                    .background(
                        RoundedRectangle(cornerRadius: 5)
                            .fill(mode == m ? .white.opacity(0.12) : .clear)
                    )
                }
                .buttonStyle(.plain)
                .foregroundStyle(.white.opacity(mode == m ? 0.9 : 0.4))
                .accessibilityIdentifier("dimension-\(m.rawValue)")
            }
        }
        .accessibilityIdentifier("dimension-toggle")
        .padding(2)
        .background(
            RoundedRectangle(cornerRadius: 7)
                .fill(Color(red: 0.08, green: 0.1, blue: 0.14))
                .overlay(
                    RoundedRectangle(cornerRadius: 7)
                        .strokeBorder(.white.opacity(0.1), lineWidth: 1)
                )
        )
    }
}

// MARK: - Void Toggle Button

struct VoidToggleButton: View {
    @Binding var showVoids: Bool

    var body: some View {
        Button {
            showVoids.toggle()
        } label: {
            HStack(spacing: 4) {
                Image(systemName: "sparkles")
                    .font(.system(size: 10))
                Text("Voids")
                    .font(.system(size: 11, design: .monospaced))
            }
            .foregroundStyle(.white.opacity(showVoids ? 0.8 : 0.35))
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .background(
                RoundedRectangle(cornerRadius: 7)
                    .fill(showVoids
                          ? Color(red: 0.3, green: 0.25, blue: 0.5).opacity(0.4)
                          : Color(red: 0.08, green: 0.1, blue: 0.14))
                    .overlay(
                        RoundedRectangle(cornerRadius: 7)
                            .strokeBorder(.white.opacity(showVoids ? 0.2 : 0.1), lineWidth: 1)
                    )
            )
        }
        .buttonStyle(.plain)
        .help("Toggle knowledge void visualization")
    }
}
