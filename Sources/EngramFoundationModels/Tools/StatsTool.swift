import FoundationModels
import EngramKit

@available(macOS 26.0, iOS 26.0, *)
public struct StatsTool: Tool {
    public typealias Output = String
    let memoryTools: MemoryTools

    public let name = "stats"
    public let description = """
        Get an overview of the memory database — memory counts grouped by project and topic.
        """

    @Generable
    public struct Arguments {
        @Guide(description: "Filter stats to a specific project scope.")
        public var project: String?
    }

    public func call(arguments: Arguments) async throws -> String {
        try await dispatchTool(memoryTools, name: "stats", args: [
            ("project", arguments.project),
        ])
    }
}
