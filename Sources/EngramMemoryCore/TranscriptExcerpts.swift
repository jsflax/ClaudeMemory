import Foundation

/// Portable half of transcript sampling: parse a Claude Code transcript
/// JSONL and extract candidate excerpts. The NOVELTY SCORING half is
/// backend-specific — EngramKit's TranscriptSampler embeds+KNNs locally;
/// the remote hooks POST the candidates to the server's /sample-gate,
/// which does the same against the agent's graph.
public enum TranscriptExcerpts {

    /// Substantive-block floor: shorter assistant texts are acknowledgments.
    public static let minBlockChars = 100
    /// Truncation applied before embedding/scoring — keeps vectors focused.
    public static let maxCharsPerExcerpt = 500

    /// Extract assistant text content blocks from a transcript JSONL.
    public static func assistantTextBlocks(from path: String) -> [String] {
        guard let data = FileManager.default.contents(atPath: path),
              let contents = String(data: data, encoding: .utf8) else {
            return []
        }

        var blocks: [String] = []
        for line in contents.components(separatedBy: .newlines) {
            guard !line.isEmpty, let lineData = line.data(using: .utf8) else { continue }
            guard let obj = try? JSONSerialization.jsonObject(with: lineData) as? [String: Any],
                  let type = obj["type"] as? String, type == "assistant",
                  let message = obj["message"] as? [String: Any],
                  let content = message["content"] as? [[String: Any]] else {
                continue
            }
            for item in content {
                guard let itemType = item["type"] as? String, itemType == "text",
                      let text = item["text"] as? String, !text.isEmpty else {
                    continue
                }
                blocks.append(text)
            }
        }
        return blocks
    }

    /// The candidate set both backends score: substantive blocks, truncated.
    public static func candidates(from path: String) -> [String] {
        assistantTextBlocks(from: path)
            .filter { $0.count >= minBlockChars }
            .map { String($0.prefix(maxCharsPerExcerpt)) }
    }
}

/// Sampling thresholds shared by both backends (the local CoreML sampler
/// and the server's /sample-gate) so novelty judgments cannot drift.
public enum SamplerThresholds {
    /// Minimum L2 distance to the nearest existing memory for an excerpt to
    /// count as "novel". v2 space (Aug 2026 mean-pooling recalibration):
    /// paraphrases 0.53–0.69 (filtered), new insight in-topic ~0.8+.
    public static let novelty: Double = 0.72
}
