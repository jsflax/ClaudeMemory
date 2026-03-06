import ArgumentParser
import EngramKit
import Lattice
import Foundation

/// Gate command: runs TranscriptSampler to check if a transcript contains novel content.
/// If novel excerpts are found, spawns the session-learner `claude` subprocess.
/// If nothing novel, exits without spawning (saving ~$0.05-0.10 per trivial stop event).
///
/// Invoked by OnStop as a fire-and-forget subprocess:
///   memory-hooks sample-gate --transcript-path <path> --project <name> --cwd <dir>
struct SampleGate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "sample-gate",
        abstract: "Gate session-learner spawning on transcript novelty"
    )

    @Option(help: "Path to the JSONL transcript file")
    var transcriptPath: String

    @Option(help: "Project name derived from working directory")
    var project: String

    @Option(help: "Working directory for the session")
    var cwd: String

    // MARK: - Session Learner Config (moved from OnStop)

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
    private static let logPath = NSHomeDirectory() + "/.claude/session-learner.log"
    private static let gateLogPath = NSHomeDirectory() + "/.claude/sample-gate.log"

    func run() async throws {
        gateLog("sample-gate: starting for project \(project), transcript: \(transcriptPath)")

        // Verify transcript exists
        guard FileManager.default.fileExists(atPath: transcriptPath) else {
            gateLog("sample-gate: transcript not found at \(transcriptPath)")
            return
        }

        // Open Lattice and embedding service
        guard let lattice = openLattice() else {
            gateLog("sample-gate: no lattice database, falling through to spawn learner")
            // If we can't check novelty, spawn learner anyway (fail-open)
            try spawnLearner(excerptHint: nil)
            return
        }

        let modelPath = ProcessInfo.processInfo.environment["CLAUDE_MEMORY_MODEL"]
        let embedder = EmbeddingService(modelPath: modelPath)
        let loadStart = Date()
        await embedder.load()
        gateLog("sample-gate: embedder loaded in \(String(format: "%.2f", Date().timeIntervalSince(loadStart)))s")

        // Fail-open if embedding model didn't load (missing CoreML model, etc.)
        guard await embedder.isLoaded else {
            gateLog("sample-gate: embedding model unavailable, falling through to spawn learner")
            try spawnLearner(excerptHint: nil)
            return
        }

        // Run the sampler
        let sampleStart = Date()
        let excerpts = await TranscriptSampler.sample(
            transcriptPath: transcriptPath,
            lattice: lattice,
            embedder: embedder
        )
        gateLog("sample-gate: sampler finished in \(String(format: "%.2f", Date().timeIntervalSince(sampleStart)))s, \(excerpts.count) excerpt(s)")

        if excerpts.isEmpty {
            gateLog("sample-gate: no novel excerpts found, skipping learner for project \(project)")
            return
        }

        gateLog("sample-gate: \(excerpts.count) novel excerpt(s) found (best distance: \(String(format: "%.3f", excerpts.first?.minDistance ?? 0)))")

        // Build hint from sampled excerpts
        let hint = TranscriptSampler.formatForAssessment(excerpts)

        try spawnLearner(excerptHint: hint)
        gateLog("sample-gate: spawned session-learner for project \(project)")
    }

    // MARK: - Learner Spawning

    private func spawnLearner(excerptHint: String?) throws {
        let prompt = buildPrompt(excerptHint: excerptHint)
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

    private func buildPrompt(excerptHint: String?) -> String {
        var prompt = """
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

        if let hint = excerptHint {
            prompt += """


            ## Sampler hints
            The transcript sampler identified these regions as potentially novel. Start your investigation here, but you may find additional insights elsewhere in the transcript.

            \(hint)
            """
        }

        return prompt
    }

    // MARK: - Logging

    private func gateLog(_ message: String) {
        let logPath = Self.gateLogPath
        rotateFileIfNeeded(logPath)
        let timestamp = ISO8601DateFormatter().string(from: Date())
        let line = "\(timestamp) [sample-gate] \(message)\n"
        if let fh = FileHandle(forWritingAtPath: logPath) {
            fh.seekToEndOfFile()
            fh.write(Data(line.utf8))
            fh.closeFile()
        } else {
            FileManager.default.createFile(atPath: logPath, contents: Data(line.utf8))
        }
        // Also log to shared hooks.log
        hookLog(message)
    }
}
