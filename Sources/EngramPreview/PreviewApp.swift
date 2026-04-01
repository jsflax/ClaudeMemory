import SwiftUI
import UserNotifications
import EngramRealityKit

@main
struct EngramPreviewApp: App {
    #if SWIFT_PACKAGE && os(macOS)
    private final class Delegate: NSObject, NSApplicationDelegate, UNUserNotificationCenterDelegate {
        func applicationDidFinishLaunching(_ notification: Notification) {
            NSApp.setActivationPolicy(.regular)
            NSApp.activate(ignoringOtherApps: true)
        }

        nonisolated func userNotificationCenter(
            _ center: UNUserNotificationCenter,
            didReceive response: UNNotificationResponse
        ) async {
            await MainActor.run {
                NSApp.activate(ignoringOtherApps: true)
                (NSApp.keyWindow ?? NSApp.orderedWindows.first)?.makeKeyAndOrderFront(nil)
            }
        }

        nonisolated func userNotificationCenter(
            _ center: UNUserNotificationCenter,
            willPresent notification: UNNotification
        ) async -> UNNotificationPresentationOptions {
            [.banner]
        }
    }

    @NSApplicationDelegateAdaptor(Delegate.self) private var appDelegate
    #endif

    var body: some Scene {
        WindowGroup {
            PreviewContentView()
                .frame(minWidth: 800, minHeight: 600)
        }
        .windowStyle(.automatic)
    }
}

struct PreviewContentView: View {
    @State private var scene = EngramRealityScene()
    @State private var mockProvider = MockGraphProvider.shared
    @State private var camera = CameraController()
    @State private var inputBridge = RKInputBridge()
    @State private var controlsVisible = true

    var body: some View {
        ZStack(alignment: .topTrailing) {
            EngramRealityView(scene: scene)
                .onAppear {
                    // Configure node count from environment (used by instrumentation tests)
                    if let countStr = ProcessInfo.processInfo.environment["PREVIEW_NODE_COUNT"],
                       let count = Int(countStr), count != mockProvider.nodeCount {
                        mockProvider.nodeCount = count
                    }

                    scene.dataProvider = mockProvider
                    scene.cameraProvider = camera
                    camera.centerOnGraph(positions: mockProvider.positions)

                    // Wire input bridge
                    inputBridge.camera = camera
                    inputBridge.scene = scene
                    inputBridge.install()

                    // Per-frame: poll keyboard + update camera smoothing
                    scene.onFrameCallback = { [weak camera, weak inputBridge] dt in
                        inputBridge?.tick(dt: dt)
                        camera?.updateCamera(dt: dt)
                    }
                }
                .onDisappear {
                    inputBridge.uninstall()
                }

            if controlsVisible {
                PreviewControls(provider: mockProvider, camera: camera, scene: scene)
                    .padding()
                    .background(.ultraThinMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .padding()
            }
        }
        .toolbar {
            ToolbarItem {
                Button(controlsVisible ? "Hide Controls" : "Show Controls") {
                    controlsVisible.toggle()
                }
            }
        }
    }
}
