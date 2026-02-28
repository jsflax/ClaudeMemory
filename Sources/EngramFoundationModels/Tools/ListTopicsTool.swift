import FoundationModels
import EngramKit

@available(macOS 26.0, iOS 26.0, *)
public struct ListTopicsTool: Tool {
    public typealias Output = String
    let memoryTools: MemoryTools

    public let name = "list_topics"
    public let description = """
        List all memory topics with their counts. Optionally filter by project.
        """

    @Generable
    public struct Arguments {
        @Guide(description: "Filter topics to a specific project scope.")
        public var project: String?
    }

    public func call(arguments: Arguments) async throws -> String {
        try await dispatchTool(memoryTools, name: "list_topics", args: [
            ("project", arguments.project),
        ])
    }
}
