import ArgumentParser
import EngramKit

/// Clears the maintenance-active flag. Called by the shell wrapper after
/// the maintenance subprocess finishes.
struct ClearMaintenance: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "clear-maintenance",
        abstract: "Clear the maintenance-active flag"
    )

    func run() {
        setHookState(key: .maintenanceActive, value: "0")
        hookLog("ClearMaintenance: set maintenanceActive=0")
    }
}
