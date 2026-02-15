import Testing
import ClaudeMemoryLib
import Lattice
import MCP
import Foundation

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
func makeTools() async throws -> MemoryTools {
    let path = FileManager.default.temporaryDirectory
        .appending(path: "claude-memory-test-\(UUID().uuidString).sqlite")
    let lattice = try Lattice(Memory.self, Edge.self, Checkpoint.self, HookState.self, configuration: .init(fileURL: path))
    let embedder = sharedEmbedder
    if await !embedder.isLoaded {
        await embedder.load()
    }
    return MemoryTools(lattice: lattice, embedder: embedder)
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
