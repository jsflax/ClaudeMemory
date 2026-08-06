import EngramMemoryCore
import Foundation

/// Deterministic bag-of-words embedder for the contract suite: the same text
/// always produces the same 384-dim unit vector, and texts sharing tokens are
/// measurably nearer than disjoint ones. Hashing is FNV-1a (NOT `Hasher`,
/// whose seed varies per process) so fixtures reproduce across runs, machines,
/// and backends.
public struct DeterministicEmbedder: Embedder {
    public init() {}

    public var dimension: Int { 384 }

    public func embed(text: String) async throws -> [Float]? {
        var vec = [Float](repeating: 0, count: 384)
        let tokens = text.lowercased()
            .split(whereSeparator: { !$0.isLetter && !$0.isNumber })
        guard !tokens.isEmpty else { return vec }
        for token in tokens {
            var hash: UInt64 = 0xcbf2_9ce4_8422_2325
            for byte in token.utf8 {
                hash = (hash ^ UInt64(byte)) &* 0x0000_0100_0000_01b3
            }
            let index = Int(hash % 384)
            let sign: Float = (hash >> 32) & 1 == 0 ? 1 : -1
            vec[index] += sign
        }
        let norm = vec.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
        if norm > 0 {
            for i in vec.indices { vec[i] /= norm }
        }
        return vec
    }
}

/// The outage embedder: `embed` returns nil, which the contract requires to
/// surface as FTS-degraded recall and never as a silently defective store.
public struct UnavailableEmbedder: Embedder {
    public init() {}
    public var dimension: Int { 384 }
    public func embed(text: String) async throws -> [Float]? { nil }
}
