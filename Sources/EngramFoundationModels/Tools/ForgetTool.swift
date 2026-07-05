import FoundationModels
import EngramKit

@available(macOS 26.0, iOS 26.0, *)
public struct ForgetTool: Tool {
    public typealias Output = String
    let memoryTools: MemoryTools

    public let name = "forget"
    public let description = """
        Delete memories by ID, topic, or project. Cascades edge cleanup. \
        At least one parameter is required.
        """

    @Generable
    public struct Arguments {
        @Guide(description: "The specific memory UUID to delete.")
        public var id: String?

        @Guide(description: "Delete all memories with this topic.")
        public var topic: String?

        @Guide(description: "Delete all memories in this project scope.")
        public var project: String?
    }

    public func call(arguments: Arguments) async throws -> String {
        try await dispatchTool(memoryTools, name: "forget", args: [
            ("id", arguments.id),
            ("topic", arguments.topic),
            ("project", arguments.project),
        ])
    }
}
