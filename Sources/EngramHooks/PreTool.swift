import ArgumentParser
import EngramKit
import Lattice
import Foundation

/// PreToolUse hook: recalls memories for Agent tool calls, nudges learning.
struct PreTool: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "pre-tool",
        abstract: "Recall memories for Agent tool calls (PreToolUse hook)"
    )

    /// Internal agent types that don't need memory context.
    private static let skipAgentTypes: Set<String> = [
        "session-learner", "memory-maintenance", "statusline-setup"
    ]

    func run() async throws {
        // Guard against recursion from maintenance/learner subprocesses
        if ProcessInfo.processInfo.environment["CLAUDE_MEMORY_MAINTENANCE"] != nil { return }

        let inputData = readStdin()
        guard !inputData.isEmpty else { return }

        let input: PreToolUseInput
        do {
            input = try JSONDecoder().decode(PreToolUseInput.self, from: inputData)
        } catch {
            hookLog("PreTool: failed to parse hook input: \(error)")
            return
        }

        guard let toolName = input.toolName else { return }

        let project = projectName(from: input.cwd)
        let proj = project ?? "unknown"

        var sections: [String] = []

        // Recall memories for Agent tool calls (Explore, Plan, general-purpose)
        if toolName == "Agent" {
            let subagentType = input.toolInput?.string(forKey: "subagent_type")

            // Skip internal agents that don't need memory context
            if let type = subagentType, Self.skipAgentTypes.contains(type) {
                hookLog("PreTool: skipping recall for internal agent: \(type)")
            } else if let query = input.toolInput?.string(forKey: "description")
                        ?? input.toolInput?.string(forKey: "prompt"),
                      !query.isEmpty {
                let sid = input.sessionId
                sessionLog("PreTool: recalling for \(subagentType ?? "unknown") agent", sessionId: sid)
                hookLog("PreTool: recalling for \(subagentType ?? "unknown") agent, query=\(String(query.prefix(80)))...")
                currentSessionId = sid
                if let tools = await initMemoryTools(sessionId: sid) {
                    sessionLog("PreTool: calling directRecall", sessionId: sid)
                    do {
                        if let result = try await tools.directRecall(
                            query: query,
                            project: project,
                            depth: 1,
                            limit: 5
                        ) {
                            sessionLog("PreTool: directRecall returned \(result.count) chars", sessionId: sid)
                            logRecalledMemories(result, hook: "PreTool", sessionId: sid)
                            sections.append("## Project context\n\n\(result)")
                        } else {
                            sessionLog("PreTool: directRecall returned nil", sessionId: sid)
                            hookLog("PreTool: recall returned nil")
                        }
                    } catch {
                        sessionLog("PreTool: directRecall FAILED: \(error)", sessionId: sid)
                        hookLog("PreTool: recall failed: \(error)")
                    }
                } else {
                    sessionLog("PreTool: initMemoryTools returned nil", sessionId: sid)
                    hookLog("PreTool: failed to initialize memory tools")
                }
            }
        }

        // Learning nudge (throttled, supplements the per-prompt nudge from Advise)
        if let nudge = throttledLearningNudge(project: proj, sessionId: input.sessionId) {
            sections.append(nudge)
        }

        guard !sections.isEmpty else { return }

        let output = HookOutput(
            hookSpecificOutput: HookSpecificOutput(
                hookEventName: "PreToolUse",
                additionalContext: sections.joined(separator: "\n\n")
            )
        )
        try writeOutput(output)
    }
}
