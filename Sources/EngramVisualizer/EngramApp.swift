import SwiftUI
import Lattice
import EngramKit
import GoogleSignIn
import UniformTypeIdentifiers
import ScreenCaptureKit
import Sparkle

@main
struct EngramApp: App {
    let lattice: Lattice
    let config: VisualizerConfig
    private let updaterController: SPUStandardUpdaterController
    @State private var accountService = AccountService()
    @State private var showingAccount = false

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
        // DIAG: force 3D mode for performance testing
        config.dimensionMode = .threeD

        #if ENABLE_ACCOUNT
        // Configure Google Sign-In if client ID is available
        if let googleClientID = ProcessInfo.processInfo.environment["GOOGLE_CLIENT_ID"]
            ?? Bundle.main.object(forInfoDictionaryKey: "GIDClientID") as? String
        {
            GIDSignIn.sharedInstance.configuration = GIDConfiguration(clientID: googleClientID)
        }
        #endif

        Task.detached { CLIInstaller.syncIfNeeded() }
    }

    var body: some Scene {
        WindowGroup("Engram") {
            GraphView()
                .environment(\.lattice, lattice)
                .environment(config)
                .frame(minWidth: 800, minHeight: 600)
                #if ENABLE_ACCOUNT
                .toolbar {
                    ToolbarItem(placement: .automatic) {
                        HStack(spacing: 12) {
                            SyncStatusView(accountService: accountService)

                            Button {
                                showingAccount.toggle()
                            } label: {
                                Image(systemName: "person.circle")
                            }
                            .popover(isPresented: $showingAccount) {
                                AccountView(accountService: accountService)
                            }
                        }
                        .padding(.leading, 8)
                    }
                }
                #endif
        }
        .commands {
            CommandGroup(after: .appInfo) {
                CheckForUpdatesView(updater: updaterController.updater)
            }
            #if ENABLE_ACCOUNT
            CommandGroup(after: .appInfo) {
                Button("Account...") {
                    showingAccount = true
                }
                .keyboardShortcut(",", modifiers: [.command])
            }
            #endif
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
        MenuBarExtra {
            MenuBarActivityFeed()
                .environment(\.lattice, lattice)
        } label: {
            Image(systemName: "brain.head.profile")
        }
        .menuBarExtraStyle(.window)
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
