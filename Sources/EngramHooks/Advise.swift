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
        // Guard against recursion: maintenance subprocess sets this env var
        if ProcessInfo.processInfo.environment["CLAUDE_MEMORY_MAINTENANCE"] != nil {
            return
        }

        let inputData = readStdin()
        guard !inputData.isEmpty else { return }

        let input: UserPromptSubmitInput
        do {
            input = try JSONDecoder().decode(UserPromptSubmitInput.self, from: inputData)
        } catch {
            hookLog("Failed to parse hook input: \(error)")
            return
        }

        guard let prompt = input.prompt, !prompt.isEmpty else {
            hookLog("Advise: empty or nil prompt, skipping")
            return
        }

        let project = projectName(from: input.cwd)
        let proj = project ?? "unknown"
        hookLog("Advise: project=\(proj), prompt=\(String(prompt.prefix(80)))...")

        var sections: [String] = []

        // Recall relevant memories
        if let tools = await initMemoryTools() {
            hookLog("Advise: initialized memory tools, running directRecall")
            do {
                if let result = try await tools.directRecall(
                    query: prompt,
                    project: project,
                    depth: 1,
                    limit: 5
                ) {
                    hookLog("Advise: recall returned \(result.count) chars")
                    sections.append("## Relevant memories\n\n\(result)")
                } else {
                    hookLog("Advise: recall returned nil")
                }
            } catch {
                hookLog("Advise: recall failed: \(error)")
            }
        } else {
            hookLog("Advise: failed to initialize memory tools")
        }

        // Spawn maintenance subprocess if threshold crossed (fire-and-forget)
        spawnMaintenanceIfNeeded(project: proj, cwd: input.cwd)

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

    // MARK: - Maintenance Subprocess

    /// The maintenance system prompt, loaded from agents/memory-maintenance.md or inlined.
    private static let maintenanceSystemPrompt: String = loadAgentSystemPrompt(
        name: "memory-maintenance",
        fallback: """
        You are a memory maintenance agent. Your job is to analyze the memory database and improve its organization for better recall quality.

        ## Workflow
        1. Run stats() and list_topics() to understand memory distribution.
        2. Run find_clusters() to discover redundant memories.
        3. Consolidate genuinely redundant memories. Keep distinct ones separate.
        4. Use organize() to create subtopics for large topic groups.
        5. Connect new memories to related existing ones.
        6. Verify improvements with stats() and list_topics().

        ## Guidelines
        - Read full memory content before deciding what to do.
        - Only consolidate memories that are truly saying the same thing.
        - Be efficient — don't over-process clean databases.
        """
    )

    private static let maintenanceAllowedTools = "mcp__memory__*,Read,Grep,Glob,Bash"

    private static let maintenanceLogPath = NSHomeDirectory() + "/.claude/memory-maintenance.log"

    /// Check if maintenance threshold is crossed and spawn the subprocess if so.
    private func spawnMaintenanceIfNeeded(project: String, cwd: String?) {
        let opCount = Int(getHookState(key: .crudOperationCount) ?? "0") ?? 0
        let lastOpCount = Int(getHookState(key: .maintenanceLastOpCount) ?? "0") ?? 0
        let delta = opCount - lastOpCount

        guard delta >= maintenanceThreshold else { return }

        // Update baseline immediately to prevent re-spawning on next advise call
        setHookState(key: .maintenanceLastOpCount, value: String(opCount))

        let prompt = """
        Perform maintenance on the memory database. Focus on project "\(project)".

        Current state: \(opCount) total CRUD operations, \(delta) since last maintenance.

        Follow your system prompt workflow: assess, find redundancy, find communities, take action, connect the graph, verify.
        """

        do {
            try spawnClaudeSubprocess(
                prompt: prompt,
                systemPrompt: Self.maintenanceSystemPrompt,
                allowedTools: Self.maintenanceAllowedTools,
                model: "sonnet",
                envGuard: (key: "CLAUDE_MEMORY_MAINTENANCE", value: "1"),
                logPath: Self.maintenanceLogPath,
                cwd: cwd
            )
            hookLog("Advise: spawned memory-maintenance subprocess (ops delta: \(delta))")
        } catch {
            hookLog("Advise: failed to spawn memory-maintenance: \(error)")
        }
    }
}
