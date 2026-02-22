import ArgumentParser
import EngramKit
import Lattice
import Foundation

/// Session signals extracted from the transcript JSONL.
struct SessionSignals {
    let filesEdited: Set<String>
    let bashCommands: [String]
    let errorCount: Int
}

/// Stop hook: spawns a fire-and-forget `claude` CLI subprocess to run session learning
/// outside the conversation, avoiding Conductor UI collision.
struct OnStop: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "on-stop",
        abstract: "Spawn session-learner CLI on stop (Stop hook)"
    )

    /// Tool names that indicate code was changed.
    private static let writeTools: Set<String> = ["Edit", "Write", "NotebookEdit"]

    /// Bash command prefixes that indicate build/test activity.
    private static let buildPatterns = ["swift build", "swift test", "xcodebuild", "npm run build", "npm test", "make", "cargo build", "cargo test", "go build", "go test", "gradle", "mvn"]

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

        let signals = extractSessionSignals(at: transcriptPath)
        hookLog("Stop hook: \(signals.filesEdited.count) files edited, \(signals.errorCount) errors")

        let project = projectName(from: input.cwd)
        let proj = project ?? "unknown"

        // Mark as sent so the Advise hook skips its learning nudge
        if let state = getSessionState(sessionId: input.sessionId) {
            state.stopNudgeSent = true
            state.updatedAt = Date()
        }

        // Build prompt and spawn CLI
        let prompt = buildPrompt(signals: signals, project: proj)
        do {
            try spawnSessionLearner(prompt: prompt, cwd: input.cwd)
            hookLog("Stop hook: spawned session-learner CLI for project \(proj)")
        } catch {
            hookLog("Stop hook: failed to spawn session-learner: \(error)")
        }

        // Return without outputting anything — no "block", no nudge message
    }

    // MARK: - Signal Extraction

    /// Extract session signals from the transcript JSONL.
    private func extractSessionSignals(at path: String) -> SessionSignals {
        guard let data = FileManager.default.contents(atPath: path),
              let content = String(data: data, encoding: .utf8) else {
            return SessionSignals(filesEdited: [], bashCommands: [], errorCount: 0)
        }

        var filesEdited = Set<String>()
        var bashCommands: [String] = []
        var errorCount = 0

        for line in content.components(separatedBy: .newlines) {
            guard !line.isEmpty else { continue }

            // Extract file paths from write tools
            for tool in Self.writeTools {
                if line.contains("\"name\":\"\(tool)\"") {
                    if let filePath = extractStringField(from: line, field: "file_path") {
                        filesEdited.insert(filePath)
                    }
                    break
                }
            }

            // Collect bash commands
            if line.contains("\"name\":\"Bash\"") {
                if let command = extractStringField(from: line, field: "command") {
                    bashCommands.append(command)
                }
            }

            // Count error entries
            if line.contains("\"type\":\"error\"") || line.contains("\"is_error\":true") {
                errorCount += 1
            }
        }

        return SessionSignals(
            filesEdited: filesEdited,
            bashCommands: bashCommands,
            errorCount: errorCount
        )
    }

    /// Extract a string field value from a JSON line using simple string matching.
    /// Handles `"field":"value"` and `"field": "value"` patterns.
    private func extractStringField(from line: String, field: String) -> String? {
        // Try both compact and spaced JSON key separators
        for separator in ["\":\"", "\": \""] {
            let needle = "\"\(field)\(separator)"
            guard let range = line.range(of: needle) else { continue }
            let afterKey = line[range.upperBound...]
            // Find the closing quote — value starts right after needle
            if let endQuote = afterKey.firstIndex(of: "\"") {
                let value = String(afterKey[afterKey.startIndex..<endQuote])
                if !value.isEmpty { return value }
            }
        }
        return nil
    }

    // MARK: - Prompt Building

    /// Build the prompt for the session-learner CLI invocation.
    private func buildPrompt(signals: SessionSignals, project: String) -> String {
        var parts: [String] = []

        parts.append("Review this coding session for project \"\(project)\" and capture what was learned.")
        parts.append("")
        parts.append("## Session signals")
        parts.append("- Files edited: \(signals.filesEdited.count)")
        if !signals.filesEdited.isEmpty {
            let sorted = signals.filesEdited.sorted()
            let listed = sorted.prefix(20).joined(separator: ", ")
            parts.append("  - \(listed)")
            if sorted.count > 20 {
                parts.append("  - ... and \(sorted.count - 20) more")
            }
        }
        parts.append("- Bash commands run: \(signals.bashCommands.count)")
        if !signals.bashCommands.isEmpty {
            // Show unique command prefixes (first 80 chars) to give context
            let unique = Set(signals.bashCommands.map { String($0.prefix(80)) })
            for cmd in unique.sorted().prefix(10) {
                parts.append("  - `\(cmd)`")
            }
        }
        parts.append("- Errors encountered: \(signals.errorCount)")
        parts.append("")
        parts.append("## Instructions")
        parts.append("1. Recall existing memories for this project to check what's already stored.")
        parts.append("2. Read the recently changed files to understand what was done.")
        parts.append("3. Look for: debugging insights, architecture decisions, new patterns, gotchas, workflow discoveries, or corrections to existing knowledge.")
        parts.append("4. Store atomic memories for what you find. Connect them to related existing ones.")
        parts.append("5. Update or correct any stale memories you encounter.")
        parts.append("6. Skip only if the session was genuinely trivial (e.g., a single question with no code changes).")

        return parts.joined(separator: "\n")
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
