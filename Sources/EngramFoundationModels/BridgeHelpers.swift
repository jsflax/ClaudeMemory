import EngramKit
import MCP
import Foundation

/// Dispatches a tool call through `MemoryTools.handle()` and extracts the text result.
///
/// This isolates all MCP type usage to a single file, preventing naming collisions
/// between `MCP.Tool` and `FoundationModels.Tool` in the individual tool files.
@available(macOS 26.0, iOS 26.0, *)
public func dispatchTool(
    _ memoryTools: MemoryTools,
    name: String,
    args: [(String, (any Sendable)?)]
) async throws -> String {
    let arguments = mcpArgs(args)
    let params = CallTool.Parameters(name: name, arguments: arguments)
    let result = try await memoryTools.handle(params)
    return extractText(from: result)
}

/// Converts a list of key-value pairs into MCP argument dictionary,
/// dropping nil values.
func mcpArgs(_ pairs: [(String, (any Sendable)?)]) -> [String: MCP.Value] {
    var dict: [String: MCP.Value] = [:]
    for (key, value) in pairs {
        guard let value else { continue }
        switch value {
        case let s as String:
            dict[key] = .string(s)
        case let i as Int:
            dict[key] = .int(i)
        case let b as Bool:
            dict[key] = .bool(b)
        case let arr as [Int]:
            dict[key] = .array(arr.map { .int($0) })
        case let arr as [String]:
            dict[key] = .array(arr.map { .string($0) })
        default:
            // Fallback: convert to string
            dict[key] = .string(String(describing: value))
        }
    }
    return dict
}

/// Extracts the text string from a `CallTool.Result`.
func extractText(from result: CallTool.Result) -> String {
    guard let first = result.content.first else { return "" }
    switch first {
    case .text(let text):
        return text
    default:
        return ""
    }
}
