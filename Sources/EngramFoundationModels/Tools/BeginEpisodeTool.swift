import FoundationModels
import EngramKit

@available(macOS 26.0, iOS 26.0, *)
public struct BeginEpisodeTool: Tool {
    public typealias Output = String
    let memoryTools: MemoryTools

    public let name = "begin_episode"
    public let description = """
        Start an episodic memory session. Memories created during the episode \
        are automatically linked to it. Use for debugging sessions, feature \
        implementations, or any focused work with a narrative arc.
        """

    @Generable
    public struct Arguments {
        @Guide(description: "A descriptive title for the episode (e.g. 'Debugging auth token expiry').")
        public var title: String?

        @Guide(description: "Project scope for the episode.")
        public var project: String?
    }

    public func call(arguments: Arguments) async throws -> String {
        try await dispatchTool(memoryTools, name: "begin_episode", args: [
            ("title", arguments.title),
            ("project", arguments.project),
        ])
    }
}
