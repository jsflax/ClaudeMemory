import Foundation

// MARK: - Transcript Parsing

/// Analysis of the latest turn in a Claude Code transcript.
struct TurnAnalysis {
    /// Tool calls by name → count.
    var toolCalls: [String: Int] = [:]
    /// Tool failures: (tool name, error message).
    var toolFailures: [(tool: String, error: String)] = []
    /// Files edited (paths from Edit/Write tool inputs).
    var filesEdited: [String] = []
    /// User message text from the current turn.
    var userMessages: [String] = []
    /// Whether a Bash error was followed by edits and success.
    var hasErrorFixCycle: Bool = false
}

/// Parses a Claude Code transcript JSONL file and extracts signals.
///
/// Transcript format (each line is a JSON object):
/// - `{"type": "assistant", "message": {"role": "assistant", "content": [{"type": "tool_use", "id": "...", "name": "Edit", "input": {...}}, ...]}}`
/// - `{"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "...", "is_error": true, "content": "..."}, {"type": "text", "text": "..."}]}}`
struct TranscriptParser {

    /// Parse the transcript at the given path and return analysis.
    static func parse(at path: String) -> TurnAnalysis? {
        guard let data = FileManager.default.contents(atPath: path),
              let content = String(data: data, encoding: .utf8) else {
            hookLog("Could not read transcript at \(path)")
            return nil
        }

        let lines = content.components(separatedBy: "\n").filter { !$0.isEmpty }
        guard !lines.isEmpty else { return nil }

        var analysis = TurnAnalysis()
        var sawBashError = false
        var sawEditAfterError = false

        // Map tool_use_id → tool name for correlating results to tools
        var toolUseNames: [String: String] = [:]

        for line in lines {
            guard let lineData = line.data(using: .utf8),
                  let json = try? JSONSerialization.jsonObject(with: lineData) as? [String: Any] else {
                continue
            }

            let topType = json["type"] as? String ?? ""
            guard let message = json["message"] as? [String: Any],
                  let contentArray = message["content"] as? [[String: Any]] else {
                continue
            }

            switch topType {
            case "assistant":
                for block in contentArray {
                    guard block["type"] as? String == "tool_use" else { continue }
                    let toolName = block["name"] as? String ?? "unknown"
                    let toolId = block["id"] as? String ?? ""

                    // Track tool_use_id → name for error correlation
                    if !toolId.isEmpty {
                        toolUseNames[toolId] = toolName
                    }

                    analysis.toolCalls[toolName, default: 0] += 1

                    // Check for file edits
                    if toolName == "Edit" || toolName == "Write" || toolName == "NotebookEdit" {
                        if let input = block["input"] as? [String: Any],
                           let filePath = input["file_path"] as? String {
                            analysis.filesEdited.append(filePath)
                        }
                        // Track error→fix cycles
                        if sawBashError {
                            sawEditAfterError = true
                        }
                    }
                }

            case "user":
                for block in contentArray {
                    let blockType = block["type"] as? String ?? ""

                    switch blockType {
                    case "text":
                        if let text = block["text"] as? String {
                            analysis.userMessages.append(text)
                        }

                    case "tool_result":
                        let isError = block["is_error"] as? Bool ?? false
                        if isError {
                            let toolId = block["tool_use_id"] as? String ?? ""
                            let toolName = toolUseNames[toolId] ?? toolId
                            let errorText = block["content"] as? String ?? "unknown error"
                            analysis.toolFailures.append((tool: toolName, error: String(errorText.prefix(200))))

                            // Track Bash errors for error→fix cycle detection
                            if toolName == "Bash" {
                                sawBashError = true
                            }
                        } else {
                            // Success after error+edit = error→fix cycle
                            if sawBashError && sawEditAfterError {
                                analysis.hasErrorFixCycle = true
                                sawBashError = false
                                sawEditAfterError = false
                            }
                        }

                    default:
                        break
                    }
                }

            default:
                break
            }
        }

        return analysis
    }
}
