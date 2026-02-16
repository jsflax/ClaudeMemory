import ArgumentParser
import ClaudeMemoryLib
import Lattice
import Foundation

/// PostToolUseFailure hook: nudges learning and maintenance. Skips expensive recall to stay within timeout.
struct OnFailure: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "on-failure",
        abstract: "Nudge learning and maintenance (PostToolUseFailure hook)"
    )

    func run() async throws {
        let inputData = readStdin()
        guard !inputData.isEmpty else { return }

        let input: PostToolUseFailureInput
        do {
            input = try JSONDecoder().decode(PostToolUseFailureInput.self, from: inputData)
        } catch {
            hookLog("Failed to parse hook input: \(error)")
            return
        }

        guard input.toolName != nil else { return }

        let project = projectName(from: input.cwd)
        let proj = project ?? "unknown"

        var sections: [String] = []

        // Maintenance nudge (if threshold crossed)
        if let nudge = maintenanceNudge(project: proj) {
            sections.append(nudge)
        }

        // Learning nudge
        sections.append(learningNudge(project: proj))

        let output = HookOutput(
            hookSpecificOutput: HookSpecificOutput(
                hookEventName: "PostToolUseFailure",
                additionalContext: sections.joined(separator: "\n\n")
            )
        )
        try writeOutput(output)
    }
}
