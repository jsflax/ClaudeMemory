import ArgumentParser
import EngramKit
import Lattice
import Foundation

/// Stop hook: spawns a fire-and-forget `claude` CLI subprocess to run session learning
/// outside the conversation, avoiding Conductor UI collision.
struct OnStop: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "on-stop",
        abstract: "Spawn session-learner CLI on stop (Stop hook)"
    )

    func run() async throws {
        // Guard against infinite recursion: session-learner subprocess sets this env var,
        // so if it's present we're inside a learner — don't spawn another.
        if ProcessInfo.processInfo.environment["CLAUDE_MEMORY_LEARNER"] != nil {
            return
        }

        let inputData = readStdin()
        guard !inputData.isEmpty else { return }

        let input: StopInput
        do {
            input = try JSONDecoder().decode(StopInput.self, from: inputData)
        } catch {
            hookLog("Failed to parse hook input: \(error)")
            return
        }

        hookLog("Stop hook: sessionId=\(input.sessionId ?? "nil")")

        guard let transcriptPath = input.transcriptPath else { return }

        let project = projectName(from: input.cwd) ?? "unknown"

        // Mark as sent so the Advise hook skips its learning nudge
        if let state = getSessionState(sessionId: input.sessionId) {
            state.stopNudgeSent = true
            state.updatedAt = Date()
        }

        let prompt = buildPrompt(transcriptPath: transcriptPath, project: project)
        do {
            try spawnSessionLearner(prompt: prompt, cwd: input.cwd)
            hookLog("Stop hook: spawned session-learner CLI for project \(project)")
        } catch {
            hookLog("Stop hook: failed to spawn session-learner: \(error)")
        }
    }

    // MARK: - Prompt Building

    private func buildPrompt(transcriptPath: String, project: String) -> String {
        """
        Review this coding session for project "\(project)" and capture what was learned.

        The full conversation transcript is at: \(transcriptPath)
        It is a JSONL file — one JSON object per line.

        ## How to read the transcript
        - Use Grep and Read to query it selectively. Do NOT try to read the entire file at once — it may be very large.
        - Grep for `"role":"assistant"` lines to find reasoning and decisions.
        - Grep for `"is_error":true` or `"type":"error"` to find errors and debugging.
        - Grep for tool names like `"name":"Edit"`, `"name":"Write"` to find what code was changed.
        - Use Read with offset/limit to page through interesting sections.

        ## What to look for
        - Debugging insights: what went wrong, what the root cause was, how it was fixed
        - Architecture decisions: why something was designed a certain way
        - Patterns and conventions: recurring approaches worth remembering
        - Gotchas: non-obvious pitfalls discovered during the session
        - Corrections: anything that contradicts or updates existing knowledge

        ## Instructions
        1. Recall existing memories for project "\(project)" to check what's already stored.
        2. Query the transcript to understand what happened and why.
        3. Store atomic memories for what you find. Connect them to related existing ones.
        4. Update or correct any stale memories you encounter.
        5. Skip only if the session was genuinely trivial (e.g., a single question with no code changes).
        """
    }

    // MARK: - CLI Spawning

    /// The session-learner system prompt, loaded from agents/session-learner.md or inlined.
    private static let sessionLearnerSystemPrompt: String = loadAgentSystemPrompt(
        name: "session-learner",
        fallback: """
        You are a session learning agent. Your job is to review what happened in a coding session and store the key insights as memories for future recall.

        ## Workflow
        1. Derive the project name from the working directory. Recall existing memories for context.
        2. Check what's already stored to prevent duplicates.
        3. Focus on: debugging insights, architecture decisions, workflow patterns, gotchas.
        4. Store memories using `remember` — atomic, concise, scoped, connected, prioritized.
        5. Connect new memories to related existing ones using edges.
        6. Update stale memories rather than creating duplicates.

        ## Guidelines
        - Be selective. A session with 20 file edits might only produce 2-3 useful memories.
        - Prefer updating existing memories over creating new ones.
        - Keep total turns low: 2-3 recalls, 2-5 remember/update/connect calls, done.
        """
    )

    private static let allowedTools = "mcp__memory__*,Read,Grep,Glob,Bash"

    /// Log file path for session-learner output.
    private static let logPath = NSHomeDirectory() + "/.claude/session-learner.log"

    /// Spawn `claude` CLI as a detached fire-and-forget subprocess for session learning.
    private func spawnSessionLearner(prompt: String, cwd: String?) throws {
        try spawnClaudeSubprocess(
            prompt: prompt,
            systemPrompt: Self.sessionLearnerSystemPrompt,
            allowedTools: Self.allowedTools,
            model: "sonnet",
            envGuard: (key: "CLAUDE_MEMORY_LEARNER", value: "1"),
            logPath: Self.logPath,
            cwd: cwd
        )
    }
}
