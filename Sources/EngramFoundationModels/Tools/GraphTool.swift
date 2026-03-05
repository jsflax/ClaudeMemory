import FoundationModels
import EngramKit

@available(macOS 26.0, iOS 26.0, *)
public struct GraphTool: Tool {
    public typealias Output = String
    let memoryTools: MemoryTools

    public let name = "graph"
    public let description = """
        View a memory's neighborhood in the knowledge graph. Shows connected \
        memories up to a given depth via BFS traversal.
        """

    @Generable
    public struct Arguments {
        @Guide(description: "The memory UUID to explore.")
        public var id: String

        @Guide(description: "Traversal depth (1-3, default 1).")
        public var depth: Int?
    }

    public func call(arguments: Arguments) async throws -> String {
        try await dispatchTool(memoryTools, name: "graph", args: [
            ("id", arguments.id),
            ("depth", arguments.depth),
        ])
    }
}
