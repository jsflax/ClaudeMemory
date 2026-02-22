import ArgumentParser
import EngramKit
import Lattice
import Foundation

/// UserPromptSubmit hook: recalls relevant memories, nudges learning and maintenance.
struct Advise: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Recall relevant memories, nudge learning and maintenance (UserPromptSubmit hook)"
    )

    func run() async throws {
        let inputData = readStdin()
        guard !inputData.isEmpty else { return }

        let input: UserPromptSubmitInput
        do {
            input = try JSONDecoder().decode(UserPromptSubmitInput.self, from: inputData)
        } catch {
            hookLog("Failed to parse hook input: \(error)")
            return
        }

        guard let prompt = input.prompt, !prompt.isEmpty else { return }

        let project = projectName(from: input.cwd)
        let proj = project ?? "unknown"

        var sections: [String] = []

        // Recall relevant memories
        if let tools = await initMemoryTools() {
            if let result = try await tools.directRecall(
                query: prompt,
                project: project,
                depth: 1,
                limit: 5
            ) {
                sections.append("## Relevant memories\n\n\(result)")
            }
        }

        // Maintenance nudge (if threshold crossed)
        if let nudge = maintenanceNudge(project: proj) {
            sections.append(nudge)
        }

        // Learning nudge — skip if stop hook just fired (session-learner already spawned)
        if let state = getSessionState(sessionId: input.sessionId), state.stopNudgeSent {
            state.stopNudgeSent = false
            state.updatedAt = Date()
        } else {
            sections.append(learningNudge(project: proj))
        }

        let output = HookOutput(
            hookSpecificOutput: HookSpecificOutput(
                hookEventName: "UserPromptSubmit",
                additionalContext: sections.joined(separator: "\n\n")
            )
        )
        try writeOutput(output)
    }
}
