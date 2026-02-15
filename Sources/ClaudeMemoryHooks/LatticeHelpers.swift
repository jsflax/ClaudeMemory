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
