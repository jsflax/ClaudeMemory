import ArgumentParser
import EngramMemoryCore
#if canImport(EngramKit)
import EngramKit
#endif

/// Clears the maintenance-active flag. Called by the shell wrapper after
/// the maintenance subprocess finishes.
struct ClearMaintenance: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "clear-maintenance",
        abstract: "Clear the maintenance-active flag"
    )

    func run() {
        #if canImport(EngramKit)
        setHookState(key: .maintenanceActive, value: "0")
        hookLog("ClearMaintenance: set maintenanceActive=0")
        #else
        hookLog("ClearMaintenance: no-op (remote backend — server owns maintenance)")
        #endif
    }
}
