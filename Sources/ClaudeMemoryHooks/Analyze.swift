import ArgumentParser
import ClaudeMemoryLib
import Foundation

/// Stop hook: analyzes the transcript for substantive interactions and nudges learning.
struct Analyze: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Analyze transcript for learning opportunities (Stop hook)"
    )

    // Signal weights
    private static let weightToolFailure = 4
    private static let weightRepeatedToolFailure = 5
    private static let weightFileEdits = 2
    private static let weightLongInteraction = 2
    private static let weightErrorFixCycle = 3
    private static let weightArchitectureKeywords = 1

    // Threshold for "substantive" interaction
    private static let substantiveThreshold = 4

    // Architecture keywords to detect in user messages
    private static let architectureKeywords: Set<String> = [
        "design", "architecture", "decision", "refactor", "pattern",
        "trade-off", "tradeoff", "approach", "strategy",
    ]

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

        // Don't nudge if the stop hook is already active (prevents infinite loop)
        if input.stopHookActive == true { return }

        guard let transcriptPath = input.transcriptPath else {
            hookLog("No transcript_path in hook input")
            return
        }

        // Loop prevention: only nudge once per transcript (survives context continuations)
        let lastNudged = getHookState(key: .lastNudgedSession)
        if lastNudged == transcriptPath {
            hookLog("Already nudged for transcript \(transcriptPath), skipping")
            return
        }

        // Parse transcript
        guard let analysis = TranscriptParser.parse(at: transcriptPath) else {
            hookLog("Could not parse transcript")
            return
        }

        // Calculate signal weights
        var totalWeight = 0
        var signals: [String] = []
        var focusAreas: [String] = []

        // Tool failures
        if !analysis.toolFailures.isEmpty {
            let failedTools = Set(analysis.toolFailures.map(\.tool))
            totalWeight += Self.weightToolFailure
            signals.append("Tool failures: \(analysis.toolFailures.count) (\(failedTools.joined(separator: ", ")))")

            // Check for repeated failures of the same tool
            var failureCounts: [String: Int] = [:]
            for failure in analysis.toolFailures {
                failureCounts[failure.tool, default: 0] += 1
            }
            let repeated = failureCounts.filter { $0.value > 1 }
            if !repeated.isEmpty {
                totalWeight += Self.weightRepeatedToolFailure
                for (tool, count) in repeated {
                    focusAreas.append("Learn the correct usage pattern for \(tool) to avoid this mistake (\(count) failures)")
                }
            } else {
                for failure in analysis.toolFailures.prefix(3) {
                    focusAreas.append("Learn the correct usage of \(failure.tool): \(failure.error.prefix(100))")
                }
            }
        }

        // File edits
        if !analysis.filesEdited.isEmpty {
            totalWeight += Self.weightFileEdits
            let uniqueFiles = Set(analysis.filesEdited)
            signals.append("Files edited: \(uniqueFiles.count) (\(uniqueFiles.prefix(5).joined(separator: ", ")))")
            focusAreas.append("Note any architecture decisions or patterns used in the edited files")
        }

        // Long interaction (>5 tool calls)
        let totalToolCalls = analysis.toolCalls.values.reduce(0, +)
        if totalToolCalls > 5 {
            totalWeight += Self.weightLongInteraction
            signals.append("Long interaction: \(totalToolCalls) tool calls")
            focusAreas.append("Summarize the key outcomes of this session")
        }

        // Error→fix cycle
        if analysis.hasErrorFixCycle {
            totalWeight += Self.weightErrorFixCycle
            signals.append("Error→fix cycle detected")
            focusAreas.append("Remember the debugging insight and root cause")
        }

        // Architecture keywords
        let allUserText = analysis.userMessages.joined(separator: " ").lowercased()
        let matchedKeywords = Self.architectureKeywords.filter { allUserText.contains($0) }
        if !matchedKeywords.isEmpty {
            totalWeight += Self.weightArchitectureKeywords
            signals.append("Architecture keywords: \(matchedKeywords.joined(separator: ", "))")
        }

        // Check threshold
        guard totalWeight >= Self.substantiveThreshold else {
            hookLog("Not substantive (weight: \(totalWeight) < \(Self.substantiveThreshold))")
            return
        }

        // Mark transcript as nudged via Lattice (use transcript path, not session ID,
        // because context continuations/compactions create new session IDs for the same transcript)
        setHookState(key: .lastNudgedSession, value: transcriptPath)

        // Build the learning nudge
        let signalList = signals.map { "- \($0)" }.joined(separator: "\n")
        let focusList = focusAreas.prefix(3).map { "- \($0)" }.joined(separator: "\n")

        let proj = projectName(from: input.cwd) ?? "unknown"

        let nudge = """
        You should learn from this interaction before stopping.

        Use the session-learner agent in the background to capture insights from this session.

        Project: \(proj)

        Signals detected:
        \(signalList)

        Focus on:
        \(focusList)

        Keep it brief — just one Task tool call, then stop.
        """

        let output = HookOutput(decision: "block", reason: nudge)
        try writeOutput(output)
        hookLog("Nudged learning (weight: \(totalWeight))")
    }
}
