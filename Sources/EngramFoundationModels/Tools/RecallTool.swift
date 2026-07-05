import FoundationModels
import EngramKit

@available(macOS 26.0, iOS 26.0, *)
public struct RecallTool: Tool {
    public typealias Output = String
    let memoryTools: MemoryTools

    public let name = "recall"
    public let description = """
        Search memories by semantic similarity. Returns the most relevant \
        memories ranked by embedding distance, with optional knowledge graph traversal.
        """

    @Generable
    public struct Arguments {
        @Guide(description: "The search query to find relevant memories. Use a broad term like 'recent' if filtering by time.")
        public var query: String?

        @Guide(description: "Filter to a specific project scope. Same-project memories rank higher but cross-project results still surface.")
        public var project: String?

        @Guide(description: "Filter to a specific topic.")
        public var topic: String?

        @Guide(description: "Maximum number of memories to return.")
        public var limit: Int?

        @Guide(description: "Graph traversal depth (0-3). Set to 1+ to follow edges from recalled memories and surface connected knowledge.")
        public var depth: Int?

        @Guide(description: "Only return memories created after this date (ISO 8601 or relative like '7d', 'last week').")
        public var since: String?

        @Guide(description: "Only return memories created before this date.")
        public var before: String?
    }

    public func call(arguments: Arguments) async throws -> String {
        try await dispatchTool(memoryTools, name: "recall", args: [
            ("query", arguments.query ?? "memories"),
            ("project", arguments.project),
            ("topic", arguments.topic),
            ("limit", arguments.limit),
            ("depth", arguments.depth),
            ("since", arguments.since),
            ("before", arguments.before),
        ])
    }
}
