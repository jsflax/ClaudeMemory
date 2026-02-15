import ArgumentParser
import ClaudeMemoryLib
import Lattice
import Foundation

/// SessionStart hook: checks if memory maintenance is due and nudges Claude to spawn a maintenance agent.
struct OnStart: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "on-start",
        abstract: "Check maintenance needs on session start (SessionStart hook)"
    )

    func run() async throws {
        let inputData = readStdin()
        guard !inputData.isEmpty else { return }

        let input: SessionStartInput
        do {
            input = try JSONDecoder().decode(SessionStartInput.self, from: inputData)
        } catch {
            hookLog("Failed to parse hook input: \(error)")
            return
        }

        // Check if maintenance is due based on CRUD operation count
        let opCount = Int(getHookState(key: .crudOperationCount) ?? "0") ?? 0
        let lastOpCount = Int(getHookState(key: .maintenanceLastOpCount) ?? "0") ?? 0
        let delta = opCount - lastOpCount

        guard delta >= maintenanceThreshold else {
            hookLog("No maintenance needed (ops delta: \(delta), threshold: \(maintenanceThreshold))")
            return
        }

        let proj = projectName(from: input.cwd) ?? "global"

        let context = """
        ## Memory maintenance due

        \(delta) memory operations since last maintenance (\(opCount) total ops).

        Before doing anything else, use the memory-maintenance agent to run maintenance in the background. \
        Focus on project "\(proj)" and any other projects with significant memory counts.

        Keep it brief — just one Task tool call, then move on to the user's request.
        """

        let output = HookOutput(
            hookSpecificOutput: HookSpecificOutput(
                hookEventName: "SessionStart",
                additionalContext: context
            )
        )
        try writeOutput(output)
        hookLog("Maintenance reminder injected (ops delta: \(delta))")
    }
}
