import SwiftUI
import AppKit

// MARK: - Inline minimap frame preference (for arc animation source/dest)

struct InlineMinimapFrameKey: PreferenceKey {
    static let defaultValue: CGRect = .zero
    static func reduce(value: inout CGRect, nextValue: () -> CGRect) {
        value = nextValue()
    }
}

// MARK: - PiP Mouse Tracker (cursor rects + mouseMoved for non-key panels)

class PiPMouseTracker: NSView {
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
        // Let the hosting view handle most clicks — only intercept edge drags via the monitor
        nil
    }
}

// MARK: - PiP Panel (borderless, floating, utility panel for minimap)

class PiPPanel: NSPanel {
    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }

    /// Snap to nearest screen corner with animation.
    func snapToNearestCorner() {
        guard let screen = screen ?? NSScreen.main else { return }
        let v = screen.visibleFrame
        let f = frame
        let pad: CGFloat = 20
        let corners: [NSPoint] = [
            NSPoint(x: v.minX + pad, y: v.maxY - f.height - pad),
            NSPoint(x: v.maxX - f.width - pad, y: v.maxY - f.height - pad),
            NSPoint(x: v.minX + pad, y: v.minY + pad),
            NSPoint(x: v.maxX - f.width - pad, y: v.minY + pad),
        ]
        let center = NSPoint(x: f.midX, y: f.midY)
        let nearest = corners.min(by: { hypot($0.x - center.x, $0.y - center.y) < hypot($1.x - center.x, $1.y - center.y) }) ?? corners[0]
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
