import EngramKit
import Lattice
import Foundation

/// Open Lattice with the full schema at the default database path.
func openLattice() -> Lattice? {
    let dbPath = defaultDbPath
    guard FileManager.default.fileExists(atPath: dbPath) else { return nil }
    return try? Lattice(
        Memory.self, Edge.self, Checkpoint.self, HookState.self, SessionState.self,
        configuration: .init(fileURL: URL(fileURLWithPath: dbPath), migration: engramMigrations)
    )
}

/// Initialize MemoryTools (Lattice + embedding model).
func initMemoryTools() async -> MemoryTools? {
    guard let lattice = openLattice() else {
        hookLog("No memory database at \(defaultDbPath)")
        return nil
    }

    let modelPath = ProcessInfo.processInfo.environment["CLAUDE_MEMORY_MODEL"]
    let embedder = EmbeddingService(modelPath: modelPath)
    await embedder.load()

    return MemoryTools(localRef: lattice.sendableReference, syncedRef: nil, embedder: embedder)
}

/// Get the current total memory count from the database.
func currentMemoryCount() -> Int? {
    guard let lattice = openLattice() else { return nil }
    return lattice.count(Memory.self)
}

/// Read a global HookState value by key.
func getHookState(key: HookState.Key) -> String? {
    guard let lattice = openLattice() else { return nil }
    return lattice.objects(HookState.self)
        .where { $0.key == key }
        .first?.value
}

/// Write a global HookState value by key (upsert).
func setHookState(key: HookState.Key, value: String) {
    guard let lattice = openLattice() else { return }
    if let existing = lattice.objects(HookState.self).where({ $0.key == key }).first {
        existing.value = value
        existing.updatedAt = Date()
    } else {
        lattice.add(HookState(key: key, value: value))
    }
}

/// Get or create the SessionState row for a given session ID.
func getSessionState(sessionId: String?) -> SessionState? {
    guard let sessionId, !sessionId.isEmpty else { return nil }
    guard let lattice = openLattice() else { return nil }
    if let existing = lattice.objects(SessionState.self).where({ $0.sessionId == sessionId }).first {
        return existing
    }
    let state = SessionState(sessionId: sessionId)
    lattice.add(state)
    return state
}

// MARK: - Shared Nudges

/// Returns the learning nudge for the given project (always fires).
func learningNudge(project: String) -> String {
    """
    ## Session learning

    If you haven't already, use the session-learner agent in the background to capture insights from this session.

    Project: \(project)

    IMPORTANT: Answer the user's request FIRST. Put the session-learner Task call at the END of your response. Do NOT relay or summarize its output — launch it silently.
    """
}

/// Threshold for first throttled learning nudge (tool calls into the session).
private let learningNudgeInitialThreshold = 15
/// Interval between subsequent throttled learning nudges (tool calls).
private let learningNudgeInterval = 30

/// Increments the tool call counter and returns a learning nudge only when threshold is crossed.
/// Use this for high-frequency hooks (PostToolUseFailure) to avoid habituation.
func throttledLearningNudge(project: String, sessionId: String?) -> String? {
    guard let state = getSessionState(sessionId: sessionId) else { return nil }

    state.toolCallCount += 1
    state.updatedAt = Date()

    let lastNudgeAt = state.learningNudgeLastToolCount
    let threshold = lastNudgeAt == 0 ? learningNudgeInitialThreshold : learningNudgeInterval
    let delta = state.toolCallCount - lastNudgeAt

    guard delta >= threshold else { return nil }

    state.learningNudgeLastToolCount = state.toolCallCount
    hookLog("Learning nudge injected (tool call \(state.toolCallCount), last nudge at \(lastNudgeAt))")

    return """
    ## Action required: capture session insights

    Spawn the session-learner agent in the background to record what you've learned this session. \
    Project: \(project). Answer the user FIRST, put the Task call at the END, and do NOT relay its output.
    """
}
