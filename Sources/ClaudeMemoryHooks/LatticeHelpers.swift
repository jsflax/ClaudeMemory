import ClaudeMemoryLib
import Lattice
import Foundation

/// The full Lattice schema — must match everywhere Lattice is initialized.
func openLattice(at dbPath: String) throws -> Lattice {
    try Lattice(
        Memory.self, Edge.self, Checkpoint.self, HookState.self,
        configuration: .init(fileURL: URL(fileURLWithPath: dbPath))
    )
}

/// Initialize MemoryTools from the default database path.
/// Returns nil if the database doesn't exist or can't be opened.
func initMemoryTools() async -> MemoryTools? {
    let dbPath = defaultDbPath

    guard FileManager.default.fileExists(atPath: dbPath) else {
        hookLog("No memory database at \(dbPath)")
        return nil
    }

    let lattice: Lattice
    do {
        lattice = try openLattice(at: dbPath)
    } catch {
        hookLog("Failed to open database: \(error)")
        return nil
    }

    let modelPath = ProcessInfo.processInfo.environment["CLAUDE_MEMORY_MODEL"]
    let embedder = EmbeddingService(modelPath: modelPath)
    await embedder.load()

    return MemoryTools(lattice: lattice, embedder: embedder)
}

/// Open Lattice read-only (no embedding model needed) for count/state queries.
func openLatticeReadOnly() -> Lattice? {
    let dbPath = defaultDbPath
    guard FileManager.default.fileExists(atPath: dbPath) else { return nil }
    return try? openLattice(at: dbPath)
}

/// Get the current total memory count from the database.
func currentMemoryCount() -> Int? {
    guard let lattice = openLatticeReadOnly() else { return nil }
    return lattice.count(Memory.self)
}

/// Read a HookState value by key.
func getHookState(key: HookState.Key) -> String? {
    guard let lattice = openLatticeReadOnly() else { return nil }
    return lattice.objects(HookState.self)
        .where { $0.key == key }
        .first?.value
}

/// Write a HookState value by key (upsert).
func setHookState(key: HookState.Key, value: String) {
    guard let lattice = openLatticeReadOnly() else { return }
    if let existing = lattice.objects(HookState.self).where({ $0.key == key }).first {
        existing.value = value
        existing.updatedAt = Date()
    } else {
        lattice.add(HookState(key: key, value: value))
    }
}

// MARK: - Shared Nudges

/// Returns a maintenance nudge if the CRUD operation delta exceeds the threshold.
func maintenanceNudge(project: String) -> String? {
    let opCount = Int(getHookState(key: .crudOperationCount) ?? "0") ?? 0
    let lastOpCount = Int(getHookState(key: .maintenanceLastOpCount) ?? "0") ?? 0
    let delta = opCount - lastOpCount

    guard delta >= maintenanceThreshold else { return nil }

    hookLog("Maintenance nudge injected (ops delta: \(delta))")

    return """
    ## Memory maintenance required

    \(delta) memory operations since last maintenance (\(opCount) total ops). \
    Spawn the memory-maintenance agent in the background now. \
    Focus on project "\(project)".
    """
}

/// Returns the learning nudge for the given project (always fires).
func learningNudge(project: String) -> String {
    """
    ## Session learning

    If you haven't already, use the session-learner agent in the background to capture insights from this session.

    Project: \(project)

    Keep it brief — just one Task tool call, then move on to the user's request.
    """
}

/// Threshold for first throttled learning nudge (tool calls into the session).
private let learningNudgeInitialThreshold = 15
/// Interval between subsequent throttled learning nudges (tool calls).
private let learningNudgeInterval = 30

/// Increments the tool call counter and returns a learning nudge only when threshold is crossed.
/// Use this for high-frequency hooks (PreToolUse, PostToolUseFailure) to avoid habituation.
func throttledLearningNudge(project: String) -> String? {
    let count = (Int(getHookState(key: .toolCallCount) ?? "0") ?? 0) + 1
    setHookState(key: .toolCallCount, value: String(count))

    let lastNudgeAt = Int(getHookState(key: .learningNudgeLastToolCount) ?? "0") ?? 0
    let threshold = lastNudgeAt == 0 ? learningNudgeInitialThreshold : learningNudgeInterval
    let delta = count - lastNudgeAt

    guard delta >= threshold else { return nil }

    setHookState(key: .learningNudgeLastToolCount, value: String(count))
    hookLog("Learning nudge injected (tool call \(count), last nudge at \(lastNudgeAt))")

    return """
    ## Action required: capture session insights

    Spawn the session-learner agent in the background now to record what you've learned this session. \
    Project: \(project). One Task tool call, then continue working.
    """
}
