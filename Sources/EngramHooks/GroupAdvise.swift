import ArgumentParser
import EngramModels

/// Per-device opt-out for teammates' group-shared memories in hook context
/// injection (advise/on-start/pre-tool). Default is ON (beta posture,
/// decision 15); this knob and the visualizer settings row are the two
/// setter surfaces for the same HookState key.
struct GroupAdvise: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "group-advise",
        abstract: "Show or set whether teammates' group memories are injected as context"
    )

    @Argument(help: "on | off | status (default: status)")
    var mode: String = "status"

    func run() throws {
        switch mode {
        case "on", "off":
            let value = mode == "on" ? "true" : "false"
            setHookState(key: .adviseIncludeGroupMemories, value: value)
            // setHookState silently no-ops when the memory DB doesn't exist
            // yet (openLattice requires the file) — a privacy setting must
            // never claim success it didn't persist. Read back to prove it.
            guard getHookState(key: .adviseIncludeGroupMemories) == value else {
                print("FAILED: no memory database at ~/.claude/memory.sqlite yet — run a Claude session (or the visualizer) once, then retry.")
                throw ExitCode(1)
            }
            print(mode == "on"
                ? "Teammates' group memories in hook injection: ON"
                : "Teammates' group memories in hook injection: OFF (this device only)")
        case "status":
            let include = getHookState(key: .adviseIncludeGroupMemories) != "false"
            print("Teammates' group memories in hook injection: \(include ? "ON (default)" : "OFF")")
        default:
            throw ValidationError("expected 'on', 'off', or 'status'")
        }
    }
}
