import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// The embedding seam. Conformances: the CoreML MiniLM embedder (EngramKit,
/// Apple-only) and `HTTPEmbedder` below (any platform, against a
/// text-embeddings-inference sidecar serving the SAME model —
/// paraphrase-MiniLM-L6-v2, 384-dim, mean-pooled, L2-normalized — so all
/// producers share one vector space).
public protocol Embedder: Sendable {
    /// nil = embedder unavailable (callers degrade: recall falls back to
    /// FTS, remember enters the pending-embedding state).
    func embed(text: String) async throws -> [Float]?
    var dimension: Int { get }
}

/// Client for a text-embeddings-inference (TEI) `/embed` endpoint.
///
/// TEI reads pooling + normalization from the model's own config, so vector
/// parity with the CoreML embedder is by construction; the increment-3
/// parity test (cosine ≥ 0.999 on a golden corpus) is the gate.
public struct HTTPEmbedder: Embedder {
    public let endpoint: URL
    public let dimension: Int
    private let session: URLSession
    private let timeout: TimeInterval

    public init(endpoint: URL, dimension: Int = 384,
                timeout: TimeInterval = 10) {
        self.endpoint = endpoint
        self.dimension = dimension
        self.timeout = timeout
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = timeout
        self.session = URLSession(configuration: config)
    }

    struct EmbedRequest: Encodable {
        let inputs: [String]
    }

    public func embed(text: String) async throws -> [Float]? {
        var request = URLRequest(url: endpoint.appendingPathComponent("embed"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(EmbedRequest(inputs: [text]))

        let (data, response): (Data, URLResponse)
        do {
            (data, response) = try await session.data(for: request)
        } catch {
            // Transport failure = unavailable, not fatal — callers own the
            // degraded path (FTS fallback / pending_embedding).
            return nil
        }
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            return nil
        }
        let vectors = try JSONDecoder().decode([[Float]].self, from: data)
        guard let vector = vectors.first, vector.count == dimension else {
            return nil
        }
        return vector
    }
}
