import ArgumentParser
import ClaudeMemoryLib
import Lattice
import Foundation

/// UserPromptSubmit hook: recalls relevant memories and injects them as context.
struct Advise: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Recall relevant memories for the user's prompt (UserPromptSubmit hook)"
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

        guard let tools = await initMemoryTools() else { return }

        guard let result = try await tools.directRecall(
            query: prompt,
            project: project,
            depth: 1,
            limit: 5
        ) else {
            return
        }

        let output = HookOutput(
            hookSpecificOutput: HookSpecificOutput(
                hookEventName: "UserPromptSubmit",
                additionalContext: "## Relevant memories\n\n\(result)"
            )
        )
        try writeOutput(output)
    }
}
