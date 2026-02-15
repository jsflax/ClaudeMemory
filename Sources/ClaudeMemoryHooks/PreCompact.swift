import ArgumentParser
import ClaudeMemoryLib
import Lattice
import Foundation

/// PreCompact hook: saves important context before compaction erases it.
struct PreCompact: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "pre-compact",
        abstract: "Save context before compaction (PreCompact hook)"
    )

    func run() async throws {
        let inputData = readStdin()
        guard !inputData.isEmpty else { return }

        let input: PreCompactInput
        do {
            input = try JSONDecoder().decode(PreCompactInput.self, from: inputData)
        } catch {
            hookLog("Failed to parse hook input: \(error)")
            return
        }

        // Only nudge on auto-compaction — manual compaction is intentional
        guard input.trigger == "auto" else { return }

        let context = """
        ## IMPORTANT: Context compaction starting

        Your conversation context is about to be compacted — earlier messages will be \
        summarized and details lost. Before this happens, you MUST save any unsaved \
        knowledge from this session:

        1. Any decisions made, patterns discovered, or bugs debugged → `remember` them now
        2. Any in-progress work → `checkpoint` with current plan, progress, and context
        3. Any corrections or mistakes → `remember` the correct approach

        Do this immediately — after compaction, you will not be able to recall these details.
        """

        let output = HookOutput(
            hookSpecificOutput: HookSpecificOutput(
                hookEventName: "PreCompact",
                additionalContext: context
            )
        )
        try writeOutput(output)
    }
}
