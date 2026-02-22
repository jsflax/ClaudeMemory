import SwiftUI
import Lattice
import EngramKit
import UniformTypeIdentifiers
import ScreenCaptureKit
import Sparkle

@main
struct EngramApp: App {
    let lattice: Lattice
    let config: VisualizerConfig
    private let updaterController: SPUStandardUpdaterController

    #if SWIFT_PACKAGE && os(macOS)
    private final class Delegate: NSObject, NSApplicationDelegate {
        func applicationDidFinishLaunching(_ notification: Notification) {
            NSApp.setActivationPolicy(.regular)
            NSApp.activate(ignoringOtherApps: true)
        }
    }

    @NSApplicationDelegateAdaptor(Delegate.self) private var appDelegate
    #endif

    init() {
        updaterController = SPUStandardUpdaterController(
            startingUpdater: true, updaterDelegate: nil, userDriverDelegate: nil
        )

        let dbPath = ProcessInfo.processInfo.environment["CLAUDE_MEMORY_DB"]
            ?? NSHomeDirectory() + "/.claude/memory.sqlite"
        let lat = try! Lattice(
            Memory.self, Edge.self, Checkpoint.self, VisualizerConfig.self,
            configuration: .init(
                fileURL: URL(fileURLWithPath: dbPath)
            )
        )
        lattice = lat
        config = lat.objects(VisualizerConfig.self).first ?? {
            let c = VisualizerConfig()
            lat.add(c)
            return c
        }()

        Task.detached { CLIInstaller.syncIfNeeded() }
    }

    var body: some Scene {
        WindowGroup("Memory Graph") {
            GraphView()
                .environment(\.lattice, lattice)
                .environment(config)
                .frame(minWidth: 800, minHeight: 600)
        }
        .commands {
            CommandGroup(after: .appInfo) {
                CheckForUpdatesView(updater: updaterController.updater)
            }
            CommandGroup(after: .saveItem) {
                Button("Export as PNG…") {
                    Task {
                        guard let pngData = await captureWindowPNG() else { return }
                        await MainActor.run {
                            let panel = NSSavePanel()
                            panel.allowedContentTypes = [.png]
                            panel.nameFieldStringValue = "memory-graph.png"
                            panel.begin { response in
                                if response == .OK, let url = panel.url {
                                    try? pngData.write(to: url)
                                }
                            }
                        }
                    }
                }
                .keyboardShortcut("e", modifiers: [.command, .shift])
            }
        }
    }
}

/// Capture the key window as PNG using ScreenCaptureKit (works with Metal/RealityKit content).
@MainActor
private func captureWindowPNG() async -> Data? {
    guard let nsWindow = NSApp.keyWindow else { return nil }
    let windowID = CGWindowID(nsWindow.windowNumber)

    do {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        guard let scWindow = content.windows.first(where: { $0.windowID == windowID }) else { return nil }

        let filter = SCContentFilter(desktopIndependentWindow: scWindow)
        let config = SCStreamConfiguration()
        config.width = Int(nsWindow.frame.width * (nsWindow.screen?.backingScaleFactor ?? 2))
        config.height = Int(nsWindow.frame.height * (nsWindow.screen?.backingScaleFactor ?? 2))
        config.captureResolution = .best
        config.showsCursor = false

        let image = try await SCScreenshotManager.captureImage(contentFilter: filter, configuration: config)
        let rep = NSBitmapImageRep(cgImage: image)
        return rep.representation(using: .png, properties: [:])
    } catch {
        // Fallback: NSView bitmap (may miss Metal content)
        guard let contentView = nsWindow.contentView else { return nil }
        let rep = contentView.bitmapImageRepForCachingDisplay(in: contentView.bounds)
        guard let rep else { return nil }
        contentView.cacheDisplay(in: contentView.bounds, to: rep)
        return rep.representation(using: .png, properties: [:])
    }
}
