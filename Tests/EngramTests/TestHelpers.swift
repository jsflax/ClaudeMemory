import Testing
import EngramKit
import Lattice
import MCP
import Foundation

// Lattice isn't Sendable but tests are sequential — safe for test inspection.
extension Lattice: @retroactive @unchecked Sendable {}

/// Helper to extract the text string from a CallTool.Result
func text(from result: CallTool.Result) -> String {
    guard case .text(let text) = result.content.first else {
        return ""
    }
    return text
}

/// Shared embedder — loads the bundled CoreML model once for all tests.
let sharedEmbedder: EmbeddingService = {
    let e = EmbeddingService()
    return e
}()

/// Create a MemoryTools with an isolated temp database and the real embedding model.
/// Uses the legacy single-DB init (syncedLattice: nil) — all existing tests pass unchanged.
func makeTools() async throws -> MemoryTools {
    let path = FileManager.default.temporaryDirectory
        .appending(path: "claude-memory-test-\(UUID().uuidString).sqlite")
    let lattice = try Lattice(Memory.self, Edge.self, Checkpoint.self, HookState.self, configuration: .init(fileURL: path))
    let embedder = sharedEmbedder
    if await !embedder.isLoaded {
        await embedder.load()
    }
    return MemoryTools(ref: lattice.sendableReference, embedder: embedder)
}

/// Wrapper to hold Lattice references for test inspection.
/// Safe for tests — sequential access only.
///
/// After `MemoryTools.init(localLattice:, syncedLattice:)`, `localLattice` has the synced DB
/// ATTACH'd via SQLite. `local.objects()` returns UNION ALL across both DBs.
/// `synced.objects()` sees only the synced DB (its own connection, no ATTACH).
///
/// Lattice caches connections by file path, so "fresh" connections to the same file
/// share the same underlying C++ connection. Use the following verification strategy:
///   - `synced.count == N` → N rows physically in synced DB
///   - `local.count == M` → M rows total across both DBs (via UNION ALL)
///   - If `synced.count == 0` and `local.count == 1` → row is in local only
///   - If `synced.count == 1` and `local.count == 1` → row is in synced only
///   - If `synced.count == 1` and `local.count == 2` → row is in both (shouldn't happen)
final class DualDBTestContext: @unchecked Sendable {
    /// The localLattice — has synced ATTACH'd, UNION ALL views span both DBs.
    let local: Lattice
    /// The syncedLattice — sees only synced DB (independent connection).
    let synced: Lattice
    let tools: MemoryTools

    init(local: Lattice, synced: Lattice, tools: MemoryTools) {
        self.local = local
        self.synced = synced
        self.tools = tools
    }
}

/// Create a MemoryTools with dual databases (local + synced) and the real embedding model.
/// Returns a context object with tools and direct DB references for inspection in tests.
func makeDualDBTools() async throws -> DualDBTestContext {
    let localPath = FileManager.default.temporaryDirectory
        .appending(path: "engram-test-local-\(UUID().uuidString).sqlite")
    let syncedPath = FileManager.default.temporaryDirectory
        .appending(path: "engram-test-synced-\(UUID().uuidString).sqlite")
    let localLattice = try Lattice(Memory.self, Edge.self, Checkpoint.self, HookState.self, configuration: .init(fileURL: localPath))
    let syncedLattice = try Lattice(Memory.self, Edge.self, SyncConfig.self, configuration: .init(fileURL: syncedPath))
    let embedder = sharedEmbedder
    if await !embedder.isLoaded {
        await embedder.load()
    }
    let tools = MemoryTools(localRef: localLattice.sendableReference, syncedRef: syncedLattice.sendableReference, embedder: embedder)
    return DualDBTestContext(local: localLattice, synced: syncedLattice, tools: tools)
}

/// Helper to extract an integer ID from text like "id: 42" or "id:42"
func extractId(from text: String) -> Int? {
    guard let range = text.range(of: "id: ", options: .literal) ?? text.range(of: "id:", options: .literal) else {
        return nil
    }
    let after = text[range.upperBound...]
    let digits = after.prefix(while: { $0.isNumber })
    return Int(digits)
}

/// Helper to extract an edge ID from text like "edge id: 42"
func extractEdgeId(from text: String) -> Int? {
    guard let range = text.range(of: "edge id: ", options: .literal) else {
        return nil
    }
    let after = text[range.upperBound...]
    let digits = after.prefix(while: { $0.isNumber })
    return Int(digits)
}

/// Helper to extract a task ID from text like "task:42"
func extractTaskId(from text: String) -> Int? {
    guard let range = text.range(of: "task:", options: .literal) else {
        return nil
    }
    let after = text[range.upperBound...]
    let digits = after.prefix(while: { $0.isNumber })
    return Int(digits)
}

/// Episodes are now memories — extract ID using the same "id:" format.
func extractEpisodeId(from text: String) -> Int? {
    extractId(from: text)
}

/// Extract memory ID from "Stored memory (id: N, ..."
func extractMemoryId(from text: String) -> Int? {
    extractId(from: text)
}
