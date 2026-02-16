import ArgumentParser
import ClaudeMemoryLib
import Lattice
import Foundation

/// Stop hook: blocks if significant code changes were made, nudging Claude to spawn the session-learner.
struct OnStop: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "on-stop",
        abstract: "Block stop if significant work detected, nudge session learning (Stop hook)"
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

        // Don't block twice per session
        hookLog("Stop hook: sessionId=\(input.sessionId ?? "nil")")
        if let state = getSessionState(sessionId: input.sessionId) {
            hookLog("Stop hook: stopNudgeSent=\(state.stopNudgeSent)")
            if state.stopNudgeSent { return }
        }

        guard let transcriptPath = input.transcriptPath else { return }

        let productiveCount = countProductiveTools(at: transcriptPath)
        hookLog("Stop hook: \(productiveCount) productive tool calls")

        guard productiveCount >= Self.writeThreshold else { return }

        let project = projectName(from: input.cwd)
        let proj = project ?? "unknown"

        if let state = getSessionState(sessionId: input.sessionId) {
            state.stopNudgeSent = true
            state.updatedAt = Date()
        }
        hookLog("Stop blocked — \(productiveCount) productive tool calls detected")

        let output = HookOutput(
            decision: "block",
            reason: "Spawn the session-learner agent to capture insights from this session. Project: \(proj)."
        )
        try writeOutput(output)
    }

    /// Count Edit/Write/NotebookEdit and build/test tool calls in the transcript JSONL.
    private func countProductiveTools(at path: String) -> Int {
        guard let data = FileManager.default.contents(atPath: path),
              let content = String(data: data, encoding: .utf8) else {
            return 0
        }

        var count = 0
        for line in content.components(separatedBy: .newlines) {
            // Count write tools
            for tool in Self.writeTools {
                if line.contains("\"name\":\"\(tool)\"") {
                    count += 1
                    break
                }
            }
            // Count build/test commands in Bash calls
            if line.contains("\"name\":\"Bash\"") {
                for pattern in Self.buildPatterns {
                    if line.contains(pattern) {
                        count += 1
                        break
                    }
                }
            }
        }
        return count
    }
}
