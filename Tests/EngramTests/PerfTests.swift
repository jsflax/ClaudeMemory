import Testing
import EngramKit
import Foundation

@Test func embeddingModelLoadTime() async throws {
    let start = ContinuousClock.now
    let embedder = EmbeddingService()
    await embedder.load()
    let elapsed = ContinuousClock.now - start

    let isLoaded = await embedder.isLoaded
    #expect(isLoaded, "Model should have loaded successfully")

    let ms = elapsed.components.seconds * 1000 + elapsed.components.attoseconds / 1_000_000_000_000_000
    print("Embedding model load time: \(ms)ms")
}
