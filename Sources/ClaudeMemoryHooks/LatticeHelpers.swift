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
    ## Memory maintenance due

    \(delta) memory operations since last maintenance (\(opCount) total ops).

    If you haven't already, use the memory-maintenance agent to run maintenance in the background. \
    Focus on project "\(project)" and any other projects with significant memory counts.

    Keep it brief — just one Task tool call, then move on to the user's request.
    """
}

/// Returns the learning nudge for the given project.
func learningNudge(project: String) -> String {
    """
    ## Session learning

    If you haven't already, use the session-learner agent in the background to capture insights from this session.

    Project: \(project)

    Keep it brief — just one Task tool call, then move on to the user's request.
    """
}
