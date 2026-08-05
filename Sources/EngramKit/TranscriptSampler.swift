import Foundation
import Lattice

/// Extracts and ranks transcript excerpts by novelty for session-learning triage.
///
/// Reads a Claude Code transcript JSONL, extracts assistant text blocks,
/// embeds them, and ranks by distance to existing memories. Returns the
/// most informative excerpts (a mix of novel and representative content)
/// condensed to fit within a small-model context window.
public struct TranscriptSampler {

    /// A scored excerpt from the transcript.
    public struct ScoredExcerpt: Sendable {
        public let text: String
        public let minDistance: Double  // distance to nearest existing memory
        public let charLength: Int

        public init(text: String, minDistance: Double, charLength: Int) {
            self.text = text
            self.minDistance = minDistance
            self.charLength = charLength
        }
    }

    /// Sample the most informative excerpts from a transcript JSONL file.
    ///
    /// Strategy:
    /// 1. Parse the JSONL and extract assistant text blocks (the reasoning/decisions).
    /// 2. Filter to substantive blocks (>100 chars — skip short acknowledgments).
    /// 3. Embed each block and find its nearest existing memory.
    /// 4. Return the top-K by a composite score: longer + more novel = higher rank.
    ///
    /// - Parameters:
    ///   - transcriptPath: Path to the JSONL transcript file.
    ///   - lattice: Lattice instance to search for existing memories.
    ///   - embedder: Embedding service for vectorizing excerpts.
    ///   - maxExcerpts: Maximum number of excerpts to return (default 5).
    ///   - maxCharsPerExcerpt: Truncate each excerpt to this length (default 500).
    ///   - maxTotalChars: Total character budget for all excerpts combined (default 2000).
    /// - Returns: Array of scored excerpts, sorted by composite score (best first).
    /// Minimum embedding distance for an excerpt to be considered "novel."
    /// Below this threshold, the excerpt is too similar to existing memories
    /// and is filtered out before reaching the assessment stage.
    /// Calibrated from test data:
    ///   - Pure paraphrases: 0.53–0.69 (should be filtered)
    ///   - Subtle new insight in same topic: ~0.8+ (should pass)
    ///   - Different subtopic: ~1.0+ (easily passes)
    ///
    /// v2 space (Aug 2026 mean-pooling fix): anchors re-measured; the old
    /// 0.28 belonged to the compressed CLS-pooled space.
    public static let defaultNoveltyThreshold: Double = 0.72

    public static func sample(
        transcriptPath: String,
        lattice: Lattice,
        embedder: EmbeddingService,
        maxExcerpts: Int = 5,
        maxCharsPerExcerpt: Int = 500,
        maxTotalChars: Int = 2000,
        noveltyThreshold: Double = TranscriptSampler.defaultNoveltyThreshold
    ) async -> [ScoredExcerpt] {
        // 1. Extract assistant text blocks from JSONL
        let blocks = extractAssistantTextBlocks(from: transcriptPath)
        guard !blocks.isEmpty else { return [] }

        // 2. Filter to substantive blocks
        let substantive = blocks.filter { $0.count >= 100 }
        guard !substantive.isEmpty else { return [] }

        // 3. Embed each block and score by novelty
        var scored: [ScoredExcerpt] = []
        let now = Date()

        for block in substantive {
            // Truncate before embedding to keep it focused
            let truncated = String(block.prefix(maxCharsPerExcerpt))

            guard let floats = try? await embedder.embed(text: truncated) else { continue }
            let embedding = Vector<Float>(floats)

            // Find nearest existing memory
            let nearest = lattice.objects(Memory.self)
                .where { $0.expiresAt > now }
                .nearest(to: embedding, on: \.embedding, limit: 1, distance: .l2)

            let minDist = nearest.first.map { $0.distances["embedding"] ?? 1.0 } ?? 1.0

            scored.append(ScoredExcerpt(
                text: truncated,
                minDistance: minDist,
                charLength: truncated.count
            ))
        }

        guard !scored.isEmpty else { return [] }

        // 4. Filter out excerpts below novelty threshold (too similar to existing memories)
        scored = scored.filter { $0.minDistance >= noveltyThreshold }
        guard !scored.isEmpty else { return [] }

        // 5. Rank by composite score: novelty (distance) * length factor
        // Longer blocks are more likely to contain reasoning/decisions.
        // Novel blocks contain content not already in memory.
        let maxLen = Double(scored.map(\.charLength).max() ?? 1)
        scored.sort { a, b in
            let scoreA = a.minDistance * (Double(a.charLength) / maxLen)
            let scoreB = b.minDistance * (Double(b.charLength) / maxLen)
            return scoreA > scoreB
        }

        // 6. Take top-K within total char budget
        var result: [ScoredExcerpt] = []
        var totalChars = 0
        for excerpt in scored.prefix(maxExcerpts) {
            if totalChars + excerpt.charLength > maxTotalChars { break }
            result.append(excerpt)
            totalChars += excerpt.charLength
        }

        return result
    }

    /// Format scored excerpts into a string suitable for FM assessment.
    public static func formatForAssessment(_ excerpts: [ScoredExcerpt]) -> String {
        excerpts.enumerated().map { i, e in
            "[\(i + 1)] \(e.text)"
        }.joined(separator: "\n\n")
    }

    // MARK: - JSONL Parsing

    /// Extract assistant text content blocks from a Claude Code transcript JSONL.
    /// Returns an array of text strings (one per assistant text block).
    public static func extractAssistantTextBlocks(from path: String) -> [String] {
        guard let data = FileManager.default.contents(atPath: path),
              let contents = String(data: data, encoding: .utf8) else {
            return []
        }

        var blocks: [String] = []

        for line in contents.components(separatedBy: .newlines) {
            guard !line.isEmpty else { continue }
            guard let lineData = line.data(using: .utf8) else { continue }

            // Parse the JSONL line
            guard let obj = try? JSONSerialization.jsonObject(with: lineData) as? [String: Any],
                  let type = obj["type"] as? String, type == "assistant",
                  let message = obj["message"] as? [String: Any],
                  let content = message["content"] as? [[String: Any]] else {
                continue
            }

            // Extract text blocks (skip thinking, tool_use, etc.)
            for item in content {
                guard let itemType = item["type"] as? String, itemType == "text",
                      let text = item["text"] as? String,
                      !text.isEmpty else {
                    continue
                }
                blocks.append(text)
            }
        }

        return blocks
    }
}
