import FoundationModels
import EngramKit
import Foundation

/// Convenience API for creating LanguageModelSessions with Engram memory tools
/// and an optional fine-tuned adapter.
@available(macOS 26.0, iOS 26.0, *)
public enum EngramSession {

    /// Default instructions for sessions with Engram tools.
    public static let defaultInstructions = """
        You have access to a persistent semantic memory system. \
        Use the provided memory tools to store, recall, and organize \
        information across conversations. Keep memories atomic — one \
        concept per memory. Use project scoping for project-specific \
        knowledge and 'global' for cross-project preferences.
        """

    /// Create a LanguageModelSession with Engram tools.
    ///
    /// - Parameters:
    ///   - memoryTools: The MemoryTools actor providing the memory backend.
    ///   - adapter: Optional trained adapter (`.fmadapter`) for improved tool calling.
    ///   - categories: Which tool categories to include (default: all).
    ///   - instructions: Custom instructions (default: Engram memory instructions).
    /// - Returns: A configured `LanguageModelSession` ready for use.
    public static func create(
        memoryTools: MemoryTools,
        adapter: SystemLanguageModel.Adapter? = nil,
        categories: [ToolCategory] = ToolCategory.allCases,
        instructions: String = defaultInstructions
    ) -> LanguageModelSession {
        let tools = EngramTools.tools(memoryTools: memoryTools, categories: categories)
        let model: SystemLanguageModel
        if let adapter {
            model = SystemLanguageModel(adapter: adapter)
        } else {
            model = .default
        }
        return LanguageModelSession(model: model, tools: tools, instructions: instructions)
    }

    /// Load an Engram adapter from a file URL.
    ///
    /// - Parameter fileURL: Path to the `.fmadapter` bundle.
    /// - Returns: A compiled adapter ready for use.
    public static func loadAdapter(from fileURL: URL) async throws -> SystemLanguageModel.Adapter {
        let adapter = try SystemLanguageModel.Adapter(fileURL: fileURL)
        try await adapter.compile()
        return adapter
    }

    /// Load an Engram adapter by name.
    ///
    /// The adapter must be registered with the system (e.g., via Background Assets).
    ///
    /// - Parameter name: The adapter name.
    /// - Returns: A compiled adapter ready for use.
    public static func loadAdapter(named name: String) async throws -> SystemLanguageModel.Adapter {
        let adapter = try SystemLanguageModel.Adapter(name: name)
        try await adapter.compile()
        return adapter
    }
}
