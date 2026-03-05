import FoundationModels
import EngramKit

@available(macOS 26.0, iOS 26.0, *)
public struct RecallEpisodeTool: Tool {
    public typealias Output = String
    let memoryTools: MemoryTools

    public let name = "recall_episode"
    public let description = """
        Retrieve the full details of a specific episode, including all memories \
        created during it in chronological order.
        """

    @Generable
    public struct Arguments {
        @Guide(description: "The episode UUID to recall.")
        public var episodeId: String

        @Guide(description: "Maximum number of memories to return from the episode.")
        public var limit: Int?
    }

    public func call(arguments: Arguments) async throws -> String {
        try await dispatchTool(memoryTools, name: "recall_episode", args: [
            ("episode_id", arguments.episodeId),
            ("limit", arguments.limit),
        ])
    }
}
