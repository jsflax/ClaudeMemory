import ArgumentParser
import ClaudeMemoryLib
import Lattice
import Foundation

/// PostToolUseFailure hook: recalls relevant memories about the failed tool.
struct OnFailure: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "on-failure",
        abstract: "Recall relevant tool usage memories after a failure (PostToolUseFailure hook)"
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

        guard let toolName = input.toolName else { return }

        let project = projectName(from: input.cwd)
        let errorText = input.error ?? "unknown error"

        // Build a query from the tool name + error context
        let query = "\(toolName) usage error: \(String(errorText.prefix(200)))"

        guard let tools = await initMemoryTools() else { return }

        guard let result = try await tools.directRecall(
            query: query,
            project: project,
            depth: 0,
            limit: 3
        ) else {
            return
        }

        let context = """
        ## Tool failure: \(toolName)

        Error: \(String(errorText.prefix(300)))

        Relevant memories about this tool:
        \(result)

        Use these memories to avoid repeating this mistake.
        """

        let output = HookOutput(
            hookSpecificOutput: HookSpecificOutput(
                hookEventName: "PostToolUseFailure",
                additionalContext: context
            )
        )
        try writeOutput(output)
    }
}
