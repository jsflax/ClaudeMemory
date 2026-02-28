import FoundationModels
import EngramKit

@available(macOS 26.0, iOS 26.0, *)
public struct ListEpisodesTool: Tool {
    public typealias Output = String
    let memoryTools: MemoryTools

    public let name = "list_episodes"
    public let description = """
        List all episodic memory sessions, optionally filtered by project. \
        Shows episode titles, status, and memory counts.
        """

    @Generable
    public struct Arguments {
        @Guide(description: "Filter episodes to a specific project scope.")
        public var project: String?

        @Guide(description: "Maximum number of episodes to return.")
        public var limit: Int?
    }

    public func call(arguments: Arguments) async throws -> String {
        try await dispatchTool(memoryTools, name: "list_episodes", args: [
            ("project", arguments.project),
            ("limit", arguments.limit),
        ])
    }
}
