import ArgumentParser
import EngramKit
import Lattice
import Foundation

/// SessionStart hook: recalls project context and checks maintenance.
struct OnStart: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "on-start",
        abstract: "Recall project context and check maintenance (SessionStart hook)"
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

        let project = projectName(from: input.cwd)
        let proj = project ?? "unknown"

        var sections: [String] = []

        // Recall project context
        if let tools = await initMemoryTools() {
            if let result = try await tools.directRecall(
                query: "project overview \(proj)",
                project: project,
                depth: 1,
                limit: 5
            ) {
                sections.append("## Project context\n\n\(result)")
            }
        }

        // Maintenance nudge (if threshold crossed)
        if let nudge = maintenanceNudge(project: proj) {
            sections.append(nudge)
        }

        guard !sections.isEmpty else { return }

        let output = HookOutput(
            hookSpecificOutput: HookSpecificOutput(
                hookEventName: "SessionStart",
                additionalContext: sections.joined(separator: "\n\n")
            )
        )
        try writeOutput(output)
    }
}
