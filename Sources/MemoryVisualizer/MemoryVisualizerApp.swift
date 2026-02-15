import SwiftUI
import Lattice
import ClaudeMemoryLib
import UniformTypeIdentifiers

@main
struct MemoryVisualizerApp: App {
    let lattice: Lattice

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
        let dbPath = ProcessInfo.processInfo.environment["CLAUDE_MEMORY_DB"]
            ?? NSHomeDirectory() + "/.claude/memory.sqlite"
        lattice = try! Lattice(
            Memory.self, Edge.self, Checkpoint.self, Episode.self,
            configuration: .init(
                fileURL: URL(fileURLWithPath: dbPath)
            )
        )
    }

    var body: some Scene {
        WindowGroup("Memory Graph") {
            GraphView()
                .environment(\.lattice, lattice)
                .frame(minWidth: 800, minHeight: 600)
        }
        .commands {
            CommandGroup(after: .saveItem) {
                Button("Export as PNG…") {
                    guard let window = NSApp.keyWindow,
                          let contentView = window.contentView else { return }
                    let rep = contentView.bitmapImageRepForCachingDisplay(in: contentView.bounds)
                    guard let rep else { return }
                    contentView.cacheDisplay(in: contentView.bounds, to: rep)
                    guard let pngData = rep.representation(using: .png, properties: [:]) else { return }
                    let panel = NSSavePanel()
                    panel.allowedContentTypes = [.png]
                    panel.nameFieldStringValue = "memory-graph.png"
                    panel.begin { response in
                        if response == .OK, let url = panel.url {
                            try? pngData.write(to: url)
                        }
                    }
                }
                .keyboardShortcut("e", modifiers: [.command, .shift])
            }
        }
    }
}
