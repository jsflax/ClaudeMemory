import Lattice
import MCP
import Foundation

public func log(_ message: String) {
    FileHandle.standardError.write(Data("[claude-memory] \(message)\n".utf8))
}

// MARK: - Codable Argument Decoding

extension Optional where Wrapped == [String: MCP.Value] {
    func decode<T: Decodable>(_ type: T.Type) throws -> T {
        let data = try JSONEncoder().encode(self ?? [:])
        return try JSONDecoder().decode(type, from: data)
    }
}

struct FlexibleInt: Decodable {
    let value: Int
    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let int = try? container.decode(Int.self) {
            value = int
        } else if let string = try? container.decode(String.self), let int = Int(string) {
            value = int
        } else {
            throw DecodingError.typeMismatch(
                Int.self,
                DecodingError.Context(codingPath: decoder.codingPath, debugDescription: "Expected an integer or string-encoded integer")
            )
        }
    }
}

/// Decodes an array of integers from either a JSON array or a comma-separated string.
/// Handles MCP clients that pass `[1, 2, 3]` as `"1, 2, 3"`.
struct FlexibleIntArray: Decodable {
    let values: [Int]
    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let array = try? container.decode([FlexibleInt].self) {
            values = array.map(\.value)
        } else if let string = try? container.decode(String.self) {
            let parsed = string.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
            guard !parsed.isEmpty else {
                throw DecodingError.typeMismatch(
                    [Int].self,
                    DecodingError.Context(codingPath: decoder.codingPath, debugDescription: "Expected an array of integers or comma-separated string of integers")
                )
            }
            values = parsed
        } else {
            throw DecodingError.typeMismatch(
                [Int].self,
                DecodingError.Context(codingPath: decoder.codingPath, debugDescription: "Expected an array of integers or comma-separated string of integers")
            )
        }
    }
}

func parseTemporalDate(_ input: String) -> Date? {
    let now = Date()
    let cal = Calendar.current

    // ISO 8601 with time
    let isoFull = ISO8601DateFormatter()
    if let date = isoFull.date(from: input) { return date }

    // ISO 8601 date only: "2024-01-15"
    let isoDate = ISO8601DateFormatter()
    isoDate.formatOptions = [.withFullDate, .withDashSeparatorInDate]
    if let date = isoDate.date(from: input) { return date }

    let lower = input.lowercased().trimmingCharacters(in: .whitespaces)

    // Natural language
    switch lower {
    case "today": return cal.startOfDay(for: now)
    case "yesterday": return cal.date(byAdding: .day, value: -1, to: cal.startOfDay(for: now))
    case "last week": return cal.date(byAdding: .weekOfYear, value: -1, to: now)
    case "last month": return cal.date(byAdding: .month, value: -1, to: now)
    case "last year": return cal.date(byAdding: .year, value: -1, to: now)
    default: break
    }

    // Compact shorthand: "7d", "2w", "3m", "1y"
    let shortPattern = /^(\d+)([dwmy])$/
    if let match = lower.wholeMatch(of: shortPattern) {
        let n = Int(match.1)!
        let component: Calendar.Component = switch String(match.2) {
        case "d": .day
        case "w": .weekOfYear
        case "m": .month
        case "y": .year
        default: .day
        }
        return cal.date(byAdding: component, value: -n, to: now)
    }

    // Relative phrase: "7 days ago", "2 weeks ago"
    let phrasePattern = /^(\d+)\s+(day|week|month|year)s?\s+ago$/
    if let match = lower.wholeMatch(of: phrasePattern) {
        let n = Int(match.1)!
        let component: Calendar.Component = switch String(match.2) {
        case "day": .day
        case "week": .weekOfYear
        case "month": .month
        case "year": .year
        default: .day
        }
        return cal.date(byAdding: component, value: -n, to: now)
    }

    return nil
}

// MARK: - Args Structs

struct RememberArgs: Decodable {
    let content: String
    let topic: String?
    let project: String?
    let source: String?
    let expiresInDays: FlexibleInt?
    let force: Bool?
    let importance: FlexibleInt?

    enum CodingKeys: String, CodingKey {
        case content, topic, project, source, force, importance
        case expiresInDays = "expires_in_days"
    }
}

struct RecallArgs: Decodable {
    let query: String
    let project: String?
    let topic: String?
    let limit: FlexibleInt?
    let depth: FlexibleInt?
    let since: String?
    let before: String?
}

struct ForgetArgs: Decodable {
    let id: FlexibleInt?
    let topic: String?
    let project: String?
}

struct UpdateArgs: Decodable {
    // Targeting (one required)
    let id: FlexibleInt?
    let query: String?
    let project: String?

    // Content editing (all optional, mutually exclusive)
    let content: String?
    let append: String?
    let prepend: String?
    let find: String?
    let replace: String?

    // Field-level metadata updates (all optional)
    let topic: String?
    let source: String?
    let expiresInDays: FlexibleInt?
    let importance: FlexibleInt?

    enum CodingKeys: String, CodingKey {
        case id, query, project, content, append, prepend, find, replace, topic, source, importance
        case expiresInDays = "expires_in_days"
    }
}

struct MergeArgs: Decodable {
    let ids: FlexibleIntArray
    let content: String
    let topic: String?
    let project: String?
}

struct StatsArgs: Decodable {
    let project: String?
}

struct ListTopicsArgs: Decodable {
    let project: String?
}

struct ConnectArgs: Decodable {
    let from: FlexibleInt
    let to: FlexibleInt
    let relation: String
}

struct DisconnectArgs: Decodable {
    let id: FlexibleInt?
    let from: FlexibleInt?
    let to: FlexibleInt?
    let relation: String?
}

struct GraphArgs: Decodable {
    let id: FlexibleInt
    let depth: FlexibleInt?
}

struct TimelineArgs: Decodable {
    let project: String?
    let topic: String?
    let since: String?
    let before: String?
    let groupBy: String?
    let limit: FlexibleInt?

    enum CodingKeys: String, CodingKey {
        case project, topic, since, before, limit
        case groupBy = "group_by"
    }
}

/// Valid relation types for knowledge graph edges.
let validRelations: Set<String> = [
    "relates_to", "contradicts", "supersedes", "derived_from", "part_of", "summarized_by",
]

/// Valid statuses for task checkpoints.
let validStatuses: Set<String> = ["active", "paused", "completed"]

/// Valid statuses for episodes.
let validEpisodeStatuses: Set<String> = ["active", "ended"]

struct CheckpointArgs: Decodable {
    let taskId: FlexibleInt?
    let title: String?
    let status: String?
    let project: String?
    let plan: String?
    let progress: String?
    let context: String?
    enum CodingKeys: String, CodingKey {
        case title, status, project, plan, progress, context
        case taskId = "task_id"
    }
}

struct ResumeArgs: Decodable {
    let taskId: FlexibleInt
    enum CodingKeys: String, CodingKey { case taskId = "task_id" }
}

struct ListTasksArgs: Decodable {
    let project: String?
    let status: String?
    let limit: FlexibleInt?
}

struct BeginEpisodeArgs: Decodable {
    let title: String?
    let project: String?
}

struct EndEpisodeArgs: Decodable {
    let episodeId: FlexibleInt?
    let summary: String?
    enum CodingKeys: String, CodingKey {
        case summary
        case episodeId = "episode_id"
    }
}

struct RecallEpisodeArgs: Decodable {
    let episodeId: FlexibleInt
    let limit: FlexibleInt?
    enum CodingKeys: String, CodingKey {
        case episodeId = "episode_id"
        case limit
    }
}

struct ListEpisodesArgs: Decodable {
    let project: String?
    let status: String?
    let limit: FlexibleInt?
}

struct FindClustersArgs: Decodable {
    let project: String?
    let topic: String?
    let minClusterSize: FlexibleInt?
    let distanceThreshold: FlexibleInt?
    let maxClusters: FlexibleInt?

    enum CodingKeys: String, CodingKey {
        case project, topic
        case minClusterSize = "min_cluster_size"
        case distanceThreshold = "distance_threshold"
        case maxClusters = "max_clusters"
    }
}

struct ConsolidateArgs: Decodable {
    let ids: FlexibleIntArray
    let content: String
    let topic: String?
    let project: String?
    let importance: FlexibleInt?
}
