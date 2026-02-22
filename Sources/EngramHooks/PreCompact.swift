import ArgumentParser
import EngramKit
import Lattice
import Foundation

/// PreCompact hook: saves important context before compaction, nudges learning and maintenance.
struct PreCompact: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "pre-compact",
        abstract: "Save context before compaction, nudge learning and maintenance (PreCompact hook)"
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

        let project = projectName(from: input.cwd)
        let proj = project ?? "unknown"

        var sections: [String] = []

        sections.append("""
        ## IMPORTANT: Context compaction starting

        Your conversation context is about to be compacted — earlier messages will be \
        summarized and details lost. Before this happens, you MUST save any unsaved \
        knowledge from this session:

        1. Any decisions made, patterns discovered, or bugs debugged → `remember` them now
        2. Any in-progress work → `checkpoint` with current plan, progress, and context
        3. Any corrections or mistakes → `remember` the correct approach

        Do this immediately — after compaction, you will not be able to recall these details.
        """)

        // Maintenance nudge (if threshold crossed)
        if let nudge = maintenanceNudge(project: proj) {
            sections.append(nudge)
        }

        // Learning nudge
        sections.append(learningNudge(project: proj))

        let output = HookOutput(
            hookSpecificOutput: HookSpecificOutput(
                hookEventName: "PreCompact",
                additionalContext: sections.joined(separator: "\n\n")
            )
        )
        try writeOutput(output)
    }
}
