import Foundation

// Per-session hook state (nudge throttling, stop/advise handshake) behind
// one accessor with two storages: the local backend keeps the existing
// lattice SessionState rows (byte-identical behavior); the remote backend
// uses a JSON file per session — sandboxes have no lattice, and this state
// is strictly session-local coordination, never memory.

struct SessionCounters: Codable {
    var toolCallCount: Int = 0
    var learningNudgeLastToolCount: Int = 0
    var stopNudgeSent: Bool = false
}

private let stateDir = NSHomeDirectory() + "/.claude/memory-hooks-state"

private func stateFile(for sessionId: String) -> String {
    stateDir + "/\(sessionId).json"
}

/// File-backed read-modify-write. Races between concurrent hooks are
/// tolerable — the counters only throttle nudges.
@discardableResult
func fileSessionCounters<T>(sessionId: String?, _ body: (inout SessionCounters) -> T) -> T? {
    guard let sessionId, !sessionId.isEmpty else { return nil }
    let fm = FileManager.default
    if !fm.fileExists(atPath: stateDir) {
        try? fm.createDirectory(atPath: stateDir, withIntermediateDirectories: true)
    }
    let path = stateFile(for: sessionId)
    var counters = (fm.contents(atPath: path))
        .flatMap { try? JSONDecoder().decode(SessionCounters.self, from: $0) }
        ?? SessionCounters()
    let result = body(&counters)
    if let data = try? JSONEncoder().encode(counters) {
        try? data.write(to: URL(fileURLWithPath: path), options: .atomic)
    }
    return result
}

func removeFileSessionCounters(sessionId: String) {
    try? FileManager.default.removeItem(atPath: stateFile(for: sessionId))
}

/// The unified accessor every hook uses. Remote mode → file store; local
/// mode → the lattice SessionState row (unchanged semantics).
@discardableResult
func updateSessionCounters<T>(sessionId: String?, _ body: (inout SessionCounters) -> T) -> T? {
    if RemoteConfig.active != nil {
        return fileSessionCounters(sessionId: sessionId, body)
    }
    #if canImport(EngramKit)
    return withSessionState(sessionId: sessionId) { state -> T in
        var counters = SessionCounters(
            toolCallCount: state.toolCallCount,
            learningNudgeLastToolCount: state.learningNudgeLastToolCount,
            stopNudgeSent: state.stopNudgeSent)
        let result = body(&counters)
        state.toolCallCount = counters.toolCallCount
        state.learningNudgeLastToolCount = counters.learningNudgeLastToolCount
        state.stopNudgeSent = counters.stopNudgeSent
        state.updatedAt = Date()
        return result
    }
    #else
    // Linux without remote config: nothing to coordinate against.
    return fileSessionCounters(sessionId: sessionId, body)
    #endif
}
