import Foundation
import Testing
import EngramKit
import EngramMemoryCore

/// Vector-space parity between the CoreML MiniLM embedder (Apple, what
/// embedded every existing memory) and the TEI HTTP embedder (Linux, what
/// the agents service runs). Both serve paraphrase-MiniLM-L6-v2 — parity is
/// by construction, this suite is the gate that proves it.
///
/// Env-gated: set `TEI_URL` (e.g. http://localhost:8090) with TEI serving
/// `sentence-transformers/paraphrase-MiniLM-L6-v2`; skipped otherwise so
/// the suite doesn't require docker on every dev run.
///
///     docker run -p 8090:80 ghcr.io/huggingface/text-embeddings-inference:cpu-1.6 \
///         --model-id sentence-transformers/paraphrase-MiniLM-L6-v2
@Suite("Embedder parity (CoreML vs TEI)")
struct EmbedderParityTests {

    static let goldenCorpus = [
        "configure the sync daemon for the group relay",
        "sqlite3_column_name returns NULL under memory pressure",
        "the stripe webhook needs the group id in subscription metadata",
        "agents recall their previous debugging sessions across sandboxes",
        "tombstones converge for every member including the author",
        "the projection pipeline replays postgres events into lattice files",
    ]

    static func cosine(_ a: [Float], _ b: [Float]) -> Double {
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0..<min(a.count, b.count) {
            dot += Double(a[i]) * Double(b[i])
            na += Double(a[i]) * Double(a[i])
            nb += Double(b[i]) * Double(b[i])
        }
        guard na > 0, nb > 0 else { return 0 }
        return dot / (na.squareRoot() * nb.squareRoot())
    }

    /// History (Aug 5 2026): this gate caught the original bundled mlmodelc
    /// having been exported with CLS pooling (`co.swiftlm.pooling_strategy:
    /// "cls"`) — a defective, weakly-discriminating space diverging from
    /// TEI/transformers.js at ~0.33–0.51 cosine. The bundle now ships the
    /// mean-pooled re-export (embedding space v2) and this test holds the
    /// BUNDLED model to ≥0.999 against TEI so the defect class cannot
    /// regress. The stored-vector migration is EmbeddingMigration (daemon
    /// startup sweep + `memory-sync migrate-embeddings`).
    @Test func coreMLAndTEIAgreeOnGoldenCorpus() async throws {
        guard let teiURL = ProcessInfo.processInfo.environment["TEI_URL"],
              let url = URL(string: teiURL) else {
            // Needs a TEI sidecar — provisioned in CI; skipped on plain
            // dev runs.
            return
        }
        let coreml = EmbeddingService()
        await coreml.load()
        let tei = HTTPEmbedder(endpoint: url)

        for text in Self.goldenCorpus {
            let a = try await coreml.embed(text: text)
            let b = try await tei.embed(text: text)
            let coremlVec = try #require(a, "CoreML embedder unavailable")
            let teiVec = try #require(b, "TEI unavailable at \(teiURL)")
            #expect(coremlVec.count == 384)
            #expect(teiVec.count == 384)
            let similarity = Self.cosine(coremlVec, teiVec)
            #expect(similarity >= 0.999,
                    "vector-space divergence \(similarity) for: \(text)")
        }
    }
}
