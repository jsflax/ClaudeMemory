import FoundationModels
import EngramKit

/// Edge relation types for the knowledge graph.
@available(macOS 26.0, iOS 26.0, *)
@Generable
public enum Relation: String, Sendable {
    case relatesTo
    case contradicts
    case supersedes
    case derivedFrom
    case partOf
    case summarizedBy

    /// Convert camelCase enum to snake_case for MCP dispatch.
    var mcpValue: String {
        switch self {
        case .relatesTo: "relates_to"
        case .contradicts: "contradicts"
        case .supersedes: "supersedes"
        case .derivedFrom: "derived_from"
        case .partOf: "part_of"
        case .summarizedBy: "summarized_by"
        }
    }
}

@available(macOS 26.0, iOS 26.0, *)
public struct ConnectTool: Tool {
    public typealias Output = String
    let memoryTools: MemoryTools

    public let name = "connect"
    public let description = """
        Create a directed edge between two memories to form a knowledge graph. \
        Duplicate edges are idempotent.
        """

    @Generable
    public struct Arguments {
        @Guide(description: "The source memory ID.")
        public var from: Int

        @Guide(description: "The target memory ID.")
        public var to: Int

        @Guide(description: "The relationship type: relatesTo, contradicts, supersedes, derivedFrom, partOf, or summarizedBy.")
        public var relation: Relation
    }

    public func call(arguments: Arguments) async throws -> String {
        try await dispatchTool(memoryTools, name: "connect", args: [
            ("from", arguments.from),
            ("to", arguments.to),
            ("relation", arguments.relation.mcpValue),
        ])
    }
}
