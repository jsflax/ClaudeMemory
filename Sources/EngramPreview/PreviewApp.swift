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

                    // Baseline-harness env hooks:
                    // PREVIEW_AUTO_ORBIT=<deg/s> — steady camera orbit so runs
                    // exercise LOD churn deterministically without input.
                    // PREVIEW_EXIT_AFTER_FRAMES=<n> — flush frame stats + exit(0)
                    // so scripted runs terminate themselves.
                    let orbitRate: Float? = ProcessInfo.processInfo.environment["PREVIEW_AUTO_ORBIT"]
                        .flatMap(Float.init)
                    let exitAfter: UInt64? = ProcessInfo.processInfo.environment["PREVIEW_EXIT_AFTER_FRAMES"]
                        .flatMap(UInt64.init)
                    var framesSeen: UInt64 = 0

                    // Per-frame: poll keyboard + update camera smoothing
                    scene.onFrameCallback = { [weak camera, weak inputBridge, weak scene] dt in
                        inputBridge?.tick(dt: dt)
                        if let orbitRate, let camera {
                            camera.lookRotate(deltaAz: orbitRate * dt * .pi / 180, deltaEl: 0)
                        }
                        camera?.updateCamera(dt: dt)
                        if let exitAfter {
                            framesSeen += 1
                            if framesSeen >= exitAfter {
                                scene?.frameStatsFlush()
                                exit(0)
                            }
                        }
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
