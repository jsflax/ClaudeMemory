import FoundationModels
import EngramKit

@available(macOS 26.0, iOS 26.0, *)
public struct DisconnectTool: Tool {
    public typealias Output = String
    let memoryTools: MemoryTools

    public let name = "disconnect"
    public let description = """
        Remove edges from the knowledge graph. Target by edge ID, \
        or by source and target memory IDs with an optional relation filter.
        """

    @Generable
    public struct Arguments {
        @Guide(description: "The edge ID to delete directly.")
        public var id: Int?

        @Guide(description: "The source memory ID (use with 'to').")
        public var from: Int?

        @Guide(description: "The target memory ID (use with 'from').")
        public var to: Int?

        @Guide(description: "Optional relation filter when using from/to: relatesTo, contradicts, supersedes, derivedFrom, partOf, or summarizedBy.")
        public var relation: Relation?
    }

    public func call(arguments: Arguments) async throws -> String {
        try await dispatchTool(memoryTools, name: "disconnect", args: [
            ("id", arguments.id),
            ("from", arguments.from),
            ("to", arguments.to),
            ("relation", arguments.relation?.mcpValue),
        ])
    }
}
