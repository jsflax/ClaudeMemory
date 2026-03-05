import FoundationModels
import EngramKit

@available(macOS 26.0, iOS 26.0, *)
public struct EndEpisodeTool: Tool {
    public typealias Output = String
    let memoryTools: MemoryTools

    public let name = "end_episode"
    public let description = """
        End an episodic memory session. Provide a summary capturing what was \
        attempted, what worked, and what was decided.
        """

    @Generable
    public struct Arguments {
        @Guide(description: "The episode UUID to end. If omitted, ends the currently active episode.")
        public var episodeId: String?

        @Guide(description: "A summary of the episode — what was attempted, what worked, what was decided.")
        public var summary: String?
    }

    public func call(arguments: Arguments) async throws -> String {
        try await dispatchTool(memoryTools, name: "end_episode", args: [
            ("episode_id", arguments.episodeId),
            ("summary", arguments.summary),
        ])
    }
}
