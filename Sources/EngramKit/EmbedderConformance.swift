import EngramMemoryCore

/// The CoreML MiniLM embedder as an `EngramMemoryCore.Embedder` — the Apple
/// half of the embedding seam. The Linux half is `HTTPEmbedder` against a
/// TEI sidecar serving the same model; `EmbedderParityTests` pins the two
/// to the same vector space (cosine ≥ 0.999 on the golden corpus).
extension EmbeddingService: Embedder {}
