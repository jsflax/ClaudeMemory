import SwiftLM
import Foundation

/// Manages the CoreML embedding model for semantic memory search.
///
/// Loads paraphrase-MiniLM-L6-v2 (bundled as a SwiftPM resource by default)
/// to generate 384-dimensional embeddings for vector similarity search.
/// Falls back to text-based search if the model fails to load.
public actor EmbeddingService {
    private var model: CoreMLEmbeddingModel?
    private let modelPath: String?

    /// Create an embedding service.
    /// - Parameter modelPath: Path to a `.mlpackage` model. If nil, uses the
    ///   bundled paraphrase-MiniLM-L6-v2 model from the app's resource bundle.
    public init(modelPath: String? = nil) {
        self.modelPath = modelPath
    }

    public func load() async {
        let url: URL

        if let modelPath {
            url = URL(fileURLWithPath: modelPath)
        } else if let bundledURL = Bundle.module.url(
            forResource: "paraphrase-MiniLM-L6-v2_Embedding",
            withExtension: "mlpackage"
        ) {
            url = bundledURL
        } else {
            log("No embedding model found (not bundled, no CLAUDE_MEMORY_MODEL). Running in degraded mode (text search only).")
            return
        }

        do {
            // Find the tokenizer — check bundle first, then auto-discover alongside model
            let tokenizerDir: URL?
            if let bundledTokenizer = Bundle.module.url(
                forResource: "paraphrase-MiniLM-L6-v2_tokenizer",
                withExtension: nil
            ) {
                tokenizerDir = bundledTokenizer
            } else {
                tokenizerDir = nil  // Let SwiftLM auto-discover from model path
            }

            model = try await CoreMLEmbeddingModel.load(url: url, tokenizerDirectory: tokenizerDir)
            log("Loaded embedding model (dim=\(model!.embeddingDimension))")
        } catch {
            log("Failed to load embedding model: \(error). Running in degraded mode.")
        }
    }

    public var isLoaded: Bool { model != nil }

    public var dimension: Int { model?.embeddingDimension ?? 0 }

    public func embed(text: String) async throws -> [Float]? {
        guard let model else { return nil }
        return try await model.embed(text: text)
    }

    private func log(_ message: String) {
        FileHandle.standardError.write(Data("[claude-memory] \(message)\n".utf8))
    }
}
