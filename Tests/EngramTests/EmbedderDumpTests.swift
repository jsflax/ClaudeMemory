import Foundation
import Testing
import EngramKit

/// Diagnostic (env-gated): dump CoreML embeddings for the parity corpus as
/// JSON so out-of-process references (TEI, transformers.js) can be
/// triangulated. Not part of the contract — remove or keep gated.
@Suite("Embedder dump (diagnostic)")
struct EmbedderDumpTests {

    @Test func dumpCoreMLVectors() async throws {
        guard let outPath = ProcessInfo.processInfo.environment["EMBED_DUMP_PATH"] else {
            return
        }
        let corpus = [
            "configure the sync daemon for the group relay",
            "sqlite3_column_name returns NULL under memory pressure",
            "the stripe webhook needs the group id in subscription metadata",
            "the cat sat on the mat",
            "a feline rested on the rug",
            "quarterly financial results",
        ]
        let coreml = EmbeddingService()
        await coreml.load()
        var out: [String: [Float]] = [:]
        for text in corpus {
            out[text] = try await coreml.embed(text: text)
        }
        let data = try JSONEncoder().encode(out)
        try data.write(to: URL(fileURLWithPath: outPath))
        #expect(out.values.allSatisfy { $0.count == 384 })
    }
}
