import ArgumentParser
import ClaudeMemoryLib
import Lattice
import Foundation

/// SessionEnd hook: reserved for future session-end cleanup.
/// Maintenance baseline is reset by the MCP server when organize/consolidate run,
/// NOT here — so the nudge persists until maintenance is actually performed.
struct OnEnd: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "on-end",
        abstract: "Session end cleanup (SessionEnd hook)"
    )

    func run() async throws {
        let inputData = readStdin()
        guard !inputData.isEmpty else { return }
        // Currently a no-op. Maintenance baseline is reset by MCP server
        // when organize/consolidate tools are called.
    }
}
