import FoundationModels
import EngramKit

@available(macOS 26.0, iOS 26.0, *)
public struct MergeTool: Tool {
    public typealias Output = String
    let memoryTools: MemoryTools

    public let name = "merge"
    public let description = """
        Consolidate multiple memories into one. The source memories are deleted \
        and their edges are transferred to the new merged memory.
        """

    @Generable
    public struct Arguments {
        @Guide(description: "The memory UUIDs to merge together.")
        public var ids: [String]

        @Guide(description: "The merged content — a well-written combination of the source memories.")
        public var content: String

        @Guide(description: "Topic for the merged memory. Defaults to the most common topic among sources.")
        public var topic: String?

        @Guide(description: "Project for the merged memory. Defaults to the project of the first source.")
        public var project: String?
    }

    public func call(arguments: Arguments) async throws -> String {
        try await dispatchTool(memoryTools, name: "merge", args: [
            ("ids", arguments.ids),
            ("content", arguments.content),
            ("topic", arguments.topic),
            ("project", arguments.project),
        ])
    }
}
