import FoundationModels
import EngramKit

/// Categories for grouping Engram tools.
@available(macOS 26.0, iOS 26.0, *)
public enum ToolCategory: CaseIterable, Sendable {
    /// Core CRUD: remember, recall, forget, update, list_topics, stats
    case coreCrud
    /// Knowledge graph: connect, disconnect, graph, merge
    case graph
    /// Episodic memory: begin_episode, end_episode, recall_episode, list_episodes
    case episodes
    /// Organization: organize, consolidate
    case organization
}

/// Factory for creating Engram memory tools compatible with Apple's FoundationModels framework.
///
/// Usage:
/// ```swift
/// let session = LanguageModelSession(
///     tools: EngramTools.all(memoryTools: memoryTools)
/// )
/// ```
@available(macOS 26.0, iOS 26.0, *)
public enum EngramTools {

    /// All 16 Engram tools.
    public static func all(memoryTools: MemoryTools) -> [any Tool] {
        tools(memoryTools: memoryTools, categories: ToolCategory.allCases)
    }

    /// Core CRUD tools only (6 tools).
    public static func core(memoryTools: MemoryTools) -> [any Tool] {
        tools(memoryTools: memoryTools, categories: [.coreCrud])
    }

    /// Read-only tools that don't modify state (5 tools):
    /// recall, list_topics, stats, graph, list_episodes.
    public static func readOnly(memoryTools: MemoryTools) -> [any Tool] {
        [
            RecallTool(memoryTools: memoryTools),
            ListTopicsTool(memoryTools: memoryTools),
            StatsTool(memoryTools: memoryTools),
            GraphTool(memoryTools: memoryTools),
            ListEpisodesTool(memoryTools: memoryTools),
            RecallEpisodeTool(memoryTools: memoryTools),
        ]
    }

    /// Tools for the specified categories.
    public static func tools(memoryTools: MemoryTools, categories: [ToolCategory]) -> [any Tool] {
        var result: [any Tool] = []
        for category in categories {
            switch category {
            case .coreCrud:
                result.append(contentsOf: coreCrudTools(memoryTools))
            case .graph:
                result.append(contentsOf: graphTools(memoryTools))
            case .episodes:
                result.append(contentsOf: episodeTools(memoryTools))
            case .organization:
                result.append(contentsOf: organizationTools(memoryTools))
            }
        }
        return result
    }

    private static func coreCrudTools(_ mt: MemoryTools) -> [any Tool] {
        [
            RememberTool(memoryTools: mt),
            RecallTool(memoryTools: mt),
            ForgetTool(memoryTools: mt),
            UpdateTool(memoryTools: mt),
            ListTopicsTool(memoryTools: mt),
            StatsTool(memoryTools: mt),
        ]
    }

    private static func graphTools(_ mt: MemoryTools) -> [any Tool] {
        [
            ConnectTool(memoryTools: mt),
            DisconnectTool(memoryTools: mt),
            GraphTool(memoryTools: mt),
            MergeTool(memoryTools: mt),
        ]
    }

    private static func episodeTools(_ mt: MemoryTools) -> [any Tool] {
        [
            BeginEpisodeTool(memoryTools: mt),
            EndEpisodeTool(memoryTools: mt),
            RecallEpisodeTool(memoryTools: mt),
            ListEpisodesTool(memoryTools: mt),
        ]
    }

    private static func organizationTools(_ mt: MemoryTools) -> [any Tool] {
        [
            OrganizeTool(memoryTools: mt),
            ConsolidateTool(memoryTools: mt),
        ]
    }
}
