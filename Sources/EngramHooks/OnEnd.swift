import ArgumentParser
import EngramMemoryCore
import Foundation
#if canImport(EngramKit)
import EngramKit
import Lattice
#endif

/// SessionEnd hook: cleans up per-session state.
struct OnEnd: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "on-end",
        abstract: "Session end cleanup (SessionEnd hook)"
    )

    func run() async throws {
        let inputData = readStdin()
        guard !inputData.isEmpty else { return }

        let input: SessionEndInput
        do {
            input = try JSONDecoder().decode(SessionEndInput.self, from: inputData)
        } catch {
            hookLog("Failed to parse hook input: \(error)")
            return
        }

        guard let sessionId = input.sessionId, !sessionId.isEmpty else { return }

        if RemoteConfig.active != nil {
            removeFileSessionCounters(sessionId: sessionId)
        } else {
            #if canImport(EngramKit)
            guard let lattice = openLattice() else { return }
            if let state = lattice.objects(SessionState.self).where({ $0.sessionId == sessionId }).first {
                lattice.delete(state)
                hookLog("Cleaned up session state for \(sessionId)")
            }
            #endif
        }

        // Clean up session recall and debug logs used by the statusline.
        // Delay cleanup — SessionEnd can fire during mid-session reconnects
        // while the statusline still needs the recall log.
        cleanupSessionLogs(sessionId: sessionId)
    }
}
