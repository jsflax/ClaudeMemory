import SwiftUI
import MetalKit

/// MTKView subclass that accepts first responder so clicking it defocuses SwiftUI text fields.
final class FocusableMTKView: MTKView {
    override var acceptsFirstResponder: Bool { true }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        super.mouseDown(with: event)
    }
}

/// NSViewRepresentable wrapping MTKView for the raw Metal renderer.
/// Hosts the MetalGraphRenderer and bridges to SwiftUI.
struct MetalViewRepresentable: NSViewRepresentable {

    let renderer: MetalGraphRenderer

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeNSView(context: Context) -> MTKView {
        let mtkView = FocusableMTKView()
        mtkView.device = renderer.device
        mtkView.colorPixelFormat = .bgra8Unorm
        mtkView.depthStencilPixelFormat = .depth32Float
        mtkView.clearColor = MTLClearColor(red: 0.051, green: 0.067, blue: 0.09, alpha: 1.0)
        mtkView.preferredFramesPerSecond = 60
        mtkView.delegate = renderer
        mtkView.isPaused = false
        mtkView.enableSetNeedsDisplay = false
        context.coordinator.observeMinimize(for: mtkView)
        return mtkView
    }

    func updateNSView(_ nsView: MTKView, context: Context) {
        // Nothing to do — renderer drives the loop via MTKViewDelegate
    }

    final class Coordinator {
        private var observers: [NSObjectProtocol] = []

        func observeMinimize(for view: MTKView) {
            let nc = NotificationCenter.default
            observers.append(nc.addObserver(forName: NSWindow.didMiniaturizeNotification, object: nil, queue: .main) { [weak view] n in
                guard let view, n.object as? NSWindow === view.window else { return }
                view.isPaused = true
            })
            observers.append(nc.addObserver(forName: NSWindow.didDeminiaturizeNotification, object: nil, queue: .main) { [weak view] n in
                guard let view, n.object as? NSWindow === view.window else { return }
                view.isPaused = false
            })
        }

        deinit {
            observers.forEach { NotificationCenter.default.removeObserver($0) }
        }
    }
}
