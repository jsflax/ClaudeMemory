import ArgumentParser
import ClaudeMemoryLib
import Lattice
import Foundation

/// Session signals extracted from the transcript JSONL.
struct SessionSignals {
    let productiveCount: Int
    let filesEdited: Set<String>
    let bashCommands: [String]
    let errorCount: Int
}

/// Stop hook: spawns a fire-and-forget `claude` CLI subprocess to run session learning
/// outside the conversation, avoiding Conductor UI collision.
struct OnStop: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "on-stop",
        abstract: "Spawn session-learner CLI on stop if significant work detected (Stop hook)"
    )

    /// Tool names that indicate code was changed.
    private static let writeTools: Set<String> = ["Edit", "Write", "NotebookEdit"]

    /// Bash command prefixes that indicate build/test activity.
    private static let buildPatterns = ["swift build", "swift test", "xcodebuild", "npm run build", "npm test", "make", "cargo build", "cargo test", "go build", "go test", "gradle", "mvn"]

    /// Minimum number of productive tool calls to consider the session worth capturing.
    private static let writeThreshold = 5

    func run() async throws {
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
        hookLog("Stop hook: \(signals.productiveCount) productive tool calls, \(signals.filesEdited.count) files edited, \(signals.errorCount) errors")

        guard signals.productiveCount >= Self.writeThreshold else { return }

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

    /// Extract rich session signals from the transcript JSONL.
    private func extractSessionSignals(at path: String) -> SessionSignals {
        guard let data = FileManager.default.contents(atPath: path),
              let content = String(data: data, encoding: .utf8) else {
            return SessionSignals(productiveCount: 0, filesEdited: [], bashCommands: [], errorCount: 0)
        }

        var productiveCount = 0
        var filesEdited = Set<String>()
        var bashCommands: [String] = []
        var errorCount = 0

        for line in content.components(separatedBy: .newlines) {
            guard !line.isEmpty else { continue }

            // Count write tools and extract file paths
            for tool in Self.writeTools {
                if line.contains("\"name\":\"\(tool)\"") {
                    productiveCount += 1
                    // Extract file_path from tool input
                    if let filePath = extractStringField(from: line, field: "file_path") {
                        filesEdited.insert(filePath)
                    }
                    break
                }
            }

            // Count build/test commands and collect them
            if line.contains("\"name\":\"Bash\"") {
                for pattern in Self.buildPatterns {
                    if line.contains(pattern) {
                        productiveCount += 1
                        break
                    }
                }
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
            productiveCount: productiveCount,
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

        parts.append("Analyze the coding session for project \"\(project)\" and store key insights as memories.")
        parts.append("")
        parts.append("## Session signals")
        parts.append("- Productive tool calls: \(signals.productiveCount)")
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
        parts.append("1. Derive project name from the working directory.")
        parts.append("2. Recall existing memories for this project to avoid duplicates.")
        parts.append("3. Read the recently changed files to understand what was done.")
        parts.append("4. Store 2-5 atomic memories for genuinely useful insights (debugging findings, architecture decisions, patterns, gotchas).")
        parts.append("5. Connect new memories to related existing ones using edges.")
        parts.append("6. Update any stale memories you find along the way.")

        return parts.joined(separator: "\n")
    }

    // MARK: - CLI Spawning

    /// The session-learner system prompt, loaded from agents/session-learner.md or inlined.
    private static let sessionLearnerSystemPrompt: String = {
        // Try to load from the agents directory relative to the hooks binary
        let bundlePath = Bundle.main.bundlePath
        let possiblePaths = [
            // Relative to the binary in ~/.claude/bin/
            NSHomeDirectory() + "/.claude/agents/session-learner.md",
            // Development path
            bundlePath + "/../../../agents/session-learner.md",
        ]

        for path in possiblePaths {
            if let content = try? String(contentsOfFile: path, encoding: .utf8) {
                // Strip frontmatter (--- delimited block at the start)
                let stripped = stripFrontmatter(content)
                return stripped
            }
        }

        // Inline fallback
        return """
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
    }()

    /// Strip YAML frontmatter from markdown content.
    private static func stripFrontmatter(_ content: String) -> String {
        guard content.hasPrefix("---") else { return content }
        // Find the closing ---
        let lines = content.components(separatedBy: .newlines)
        var endIndex = 0
        for i in 1..<lines.count {
            if lines[i].hasPrefix("---") {
                endIndex = i + 1
                break
            }
        }
        guard endIndex > 0 else { return content }
        return lines[endIndex...].joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Log file path for session-learner output.
    private static let logPath = NSHomeDirectory() + "/.claude/session-learner.log"

    /// Spawn `claude` CLI as a detached fire-and-forget subprocess, logging output to a file.
    /// Uses /bin/sh with shell redirection to ensure output is captured even with buffered pipes.
    private func spawnSessionLearner(prompt: String, cwd: String?) throws {
        // Shell-escape the prompt and system prompt for embedding in a sh -c command
        let escapedPrompt = prompt.replacingOccurrences(of: "'", with: "'\\''")
        let escapedSystemPrompt = Self.sessionLearnerSystemPrompt.replacingOccurrences(of: "'", with: "'\\''")
        let allowedTools = "mcp__memory__remember,mcp__memory__recall,mcp__memory__update,mcp__memory__forget,mcp__memory__merge,mcp__memory__connect,mcp__memory__graph,mcp__memory__list_topics,mcp__memory__stats,Read,Grep,Glob,Bash"

        let shellCommand = """
        echo '===== session-learner started at '\\''\(ISO8601DateFormatter().string(from: Date()))'\\'' =====' >> '\(Self.logPath)' && \
        claude -p '\(escapedPrompt)' \
          --model sonnet \
          --allowedTools '\(allowedTools)' \
          --no-session-persistence \
          --append-system-prompt '\(escapedSystemPrompt)' \
          >> '\(Self.logPath)' 2>&1
        """

        let sh = Process()
        sh.executableURL = URL(fileURLWithPath: "/bin/sh")
        sh.arguments = ["-c", shellCommand]
        if let cwd {
            sh.currentDirectoryURL = URL(fileURLWithPath: cwd)
        }

        // Detach from our process's stdio
        sh.standardOutput = FileHandle.nullDevice
        sh.standardError = FileHandle.nullDevice

        try sh.run()
        // Don't wait — fire and forget
    }
}
