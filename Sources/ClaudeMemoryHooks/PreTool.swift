import ArgumentParser
import ClaudeMemoryLib
import Lattice
import Foundation

/// PreToolUse hook: warns about known mistakes before a tool runs.
struct PreTool: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "pre-tool",
        abstract: "Warn about known mistakes before tool use (PreToolUse hook)"
    )

    func run() async throws {
        let inputData = readStdin()
        guard !inputData.isEmpty else { return }

        let input: PreToolUseInput
        do {
            input = try JSONDecoder().decode(PreToolUseInput.self, from: inputData)
        } catch {
            hookLog("Failed to parse hook input: \(error)")
            return
        }

        guard let toolName = input.toolName else { return }

        // Only check for MCP tools and Bash — these are most likely to have learned patterns
        let isRelevant = toolName.hasPrefix("mcp__") || toolName == "Bash"
        guard isRelevant else { return }

        let project = projectName(from: input.cwd)

        // Build a query from tool name + input context
        var query = "\(toolName) correct usage"
        if let toolInput = input.toolInput {
            // Add key input fields for more targeted recall
            if let command = toolInput.string(forKey: "command") {
                query += " \(String(command.prefix(100)))"
            }
            if let name = toolInput.string(forKey: "name") {
                query += " \(name)"
            }
        }

        guard let tools = await initMemoryTools() else { return }

        // Only surface memories specifically about debugging/mistakes for this tool
        guard let result = try await tools.directRecall(
            query: query,
            project: project,
            depth: 0,
            limit: 2
        ) else {
            return
        }

        // Only inject if the memories are actually about mistakes/patterns with this tool
        // (the recall already filters by relevance, but we check the content mentions the tool)
        let toolBaseName = toolName.replacingOccurrences(of: "mcp__", with: "")
            .replacingOccurrences(of: "__", with: " ")
        let isToolRelevant = result.lowercased().contains(toolBaseName.lowercased())
            || result.lowercased().contains("error")
            || result.lowercased().contains("mistake")
            || result.lowercased().contains("correct")
            || result.lowercased().contains("avoid")
        guard isToolRelevant else { return }

        let output = HookOutput(
            hookSpecificOutput: HookSpecificOutput(
                hookEventName: "PreToolUse",
                additionalContext: "## Known patterns for \(toolName)\n\n\(result)"
            )
        )
        try writeOutput(output)
    }
}
