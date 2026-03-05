import FoundationModels
import EngramKit

@available(macOS 26.0, iOS 26.0, *)
public struct OrganizeTool: Tool {
    public typealias Output = String
    let memoryTools: MemoryTools

    public let name = "organize"
    public let description = """
        Batch re-topic memories under a new label. Creates a hub memory and \
        links all specified memories to it via 'part_of' edges. Use instead \
        of calling update in a loop to change topics.
        """

    @Generable
    public struct Arguments {
        @Guide(description: "Memory UUIDs to organize under the label.")
        public var ids: [String]

        @Guide(description: "The topic label to apply (e.g. 'editor-rendering', 'AI-pipeline').")
        public var label: String

        @Guide(description: "Project scope for the hub memory.")
        public var project: String?

        @Guide(description: "Custom summary for the hub memory. Defaults to 'Hub: <label>'.")
        public var summary: String?
    }

    public func call(arguments: Arguments) async throws -> String {
        try await dispatchTool(memoryTools, name: "organize", args: [
            ("ids", arguments.ids),
            ("label", arguments.label),
            ("project", arguments.project),
            ("summary", arguments.summary),
        ])
    }
}
