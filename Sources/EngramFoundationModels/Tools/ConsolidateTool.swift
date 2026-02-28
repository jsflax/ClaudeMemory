import FoundationModels
import EngramKit

@available(macOS 26.0, iOS 26.0, *)
public struct ConsolidateTool: Tool {
    public typealias Output = String
    let memoryTools: MemoryTools

    public let name = "consolidate"
    public let description = """
        Consolidate redundant memories into a summary. Original memories are \
        deprioritized (importance set to 0) and linked to the summary via \
        'summarized_by' edges. Only consolidate memories with genuinely \
        overlapping content.
        """

    @Generable
    public struct Arguments {
        @Guide(description: "Memory IDs to consolidate.")
        public var ids: [Int]

        @Guide(description: "Summary content capturing the essential knowledge from the memories.")
        public var content: String

        @Guide(description: "Topic for the summary memory.")
        public var topic: String?

        @Guide(description: "Project for the summary memory.")
        public var project: String?

        @Guide(description: "Importance of the summary memory (1-5, default 3).")
        public var importance: Int?
    }

    public func call(arguments: Arguments) async throws -> String {
        try await dispatchTool(memoryTools, name: "consolidate", args: [
            ("ids", arguments.ids),
            ("content", arguments.content),
            ("topic", arguments.topic),
            ("project", arguments.project),
            ("importance", arguments.importance),
        ])
    }
}
