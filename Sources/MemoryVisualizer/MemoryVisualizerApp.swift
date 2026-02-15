import SwiftUI
import Lattice
import ClaudeMemoryLib

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
    }
}
