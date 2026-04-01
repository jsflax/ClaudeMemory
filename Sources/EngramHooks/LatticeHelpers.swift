import EngramKit
import Lattice
import Foundation

/// Open Lattice with the full schema at the default database path.
func openLattice() -> Lattice? {
    let dbPath = defaultDbPath
    guard FileManager.default.fileExists(atPath: dbPath) else { return nil }
    return try? Lattice(
        Memory.self, Edge.self, Checkpoint.self, HookState.self, SessionState.self,
        configuration: .init(fileURL: URL(fileURLWithPath: dbPath), migration: engramMigrations)
    )
}

/// Initialize MemoryTools (Lattice + embedding model).
func initMemoryTools() async -> MemoryTools? {
    guard let lattice = openLattice() else {
        hookLog("No memory database at \(defaultDbPath)")
        return nil
    }

    let modelPath = ProcessInfo.processInfo.environment["CLAUDE_MEMORY_MODEL"]
    let embedder = EmbeddingService(modelPath: modelPath)
    await embedder.load()

    return MemoryTools(localRef: lattice.sendableReference, syncedRef: nil, embedder: embedder)
}

/// Get the current total memory count from the database.
func currentMemoryCount() -> Int? {
    guard let lattice = openLattice() else { return nil }
    return lattice.count(Memory.self)
}

/// Read a global HookState value by key.
func getHookState(key: HookState.Key) -> String? {
    guard let lattice = openLattice() else { return nil }
    return lattice.objects(HookState.self)
        .where { $0.key == key }
        .first?.value
}

/// Write a global HookState value by key (upsert).
func setHookState(key: HookState.Key, value: String) {
    guard let lattice = openLattice() else { return }
    if let existing = lattice.objects(HookState.self).where({ $0.key == key }).first {
        existing.value = value
        existing.updatedAt = Date()
    } else {
        lattice.add(HookState(key: key, value: value))
    }
}

/// Get or create the SessionState row for a given session ID.
func getSessionState(sessionId: String?) -> SessionState? {
    guard let sessionId, !sessionId.isEmpty else { return nil }
    guard let lattice = openLattice() else { return nil }
    if let existing = lattice.objects(SessionState.self).where({ $0.sessionId == sessionId }).first {
        return existing
    }
    let state = SessionState(sessionId: sessionId)
    lattice.add(state)
    return state
}

// MARK: - ANSI Colors

private enum ANSIColor {
    static let reset   = "\u{1B}[0m"
    static let bold    = "\u{1B}[1m"
    static let dim     = "\u{1B}[2m"
    static let cyan    = "\u{1B}[36m"
    static let magenta = "\u{1B}[35m"
    static let green   = "\u{1B}[32m"
    static let yellow  = "\u{1B}[33m"
    static let red     = "\u{1B}[31m"
    static let blue    = "\u{1B}[34m"
}

/// Color-code a distance value: green (<0.20), yellow (0.20-0.35), red (>0.35).
private func colorDist(_ dist: String) -> String {
    guard let d = Double(dist) else { return dist }
    let color = d < 0.20 ? ANSIColor.green : d < 0.35 ? ANSIColor.yellow : ANSIColor.red
    return "\(color)\(dist)\(ANSIColor.reset)"
}

// MARK: - Recall Logging

/// Parse a directRecall result string and log a compact summary of each recalled memory.
/// Format: one line per memory with id (short), project/topic, distance, and content preview.
func logRecalledMemories(_ result: String, hook: String) {
    // Each memory block starts with [id:UUID] [project/topic]
    // Direct results: [id:UUID] [project/topic] (distance: 0.123...) content
    // Connected results come after "--- Connected (graph traversal, depth: N) ---"

    var directCount = 0
    var connectedCount = 0
    var inConnected = false
    var lines: [String] = []

    for block in result.components(separatedBy: "\n\n") {
        let trimmed = block.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { continue }

        if trimmed.hasPrefix("--- Connected") {
            inConnected = true
            continue
        }
        if trimmed.hasPrefix("⚠️ Weak recall") {
            lines.append("  \(ANSIColor.yellow)\(ANSIColor.bold)⚠️  weak recall signal\(ANSIColor.reset)")
            continue
        }

        // Parse [id:UUID] [project/topic] ...
        guard trimmed.hasPrefix("[id:") else { continue }

        // Extract short ID (first 8 chars of UUID)
        let idEnd = trimmed.firstIndex(of: "]") ?? trimmed.startIndex
        let idStr = String(trimmed[trimmed.index(trimmed.startIndex, offsetBy: 4)..<idEnd])
        let shortId = String(idStr.prefix(8))

        // Extract [project/topic]
        let afterId = trimmed[trimmed.index(after: idEnd)...]
        var projTopic = ""
        if let ptStart = afterId.firstIndex(of: "["),
           let ptEnd = afterId.firstIndex(of: "]") {
            projTopic = String(afterId[afterId.index(after: ptStart)..<ptEnd])
        }

        // Extract distance if present
        var dist = ""
        if let distRange = trimmed.range(of: "distance: ") {
            let distStart = distRange.upperBound
            if let distEnd = trimmed[distStart...].firstIndex(where: { $0 == "," || $0 == ")" }) {
                dist = String(trimmed[distStart..<distEnd])
            }
        }

        // Extract edge info for connected memories
        var edge = ""
        if inConnected {
            if let edgeRange = trimmed.range(of: "--[") {
                let edgeStart = edgeRange.lowerBound
                if let edgeEnd = trimmed[edgeStart...].range(of: "]--") {
                    edge = " \(ANSIColor.blue)" + String(trimmed[edgeStart...edgeEnd.upperBound]) + ANSIColor.reset
                }
            }
        }

        // Content preview: everything after the metadata parenthetical
        var preview = ""
        // Find content after last ) that follows the metadata
        if let lastParen = trimmed.range(of: ") ", options: .backwards) {
            preview = String(trimmed[lastParen.upperBound...].prefix(60))
                .replacingOccurrences(of: "\n", with: " ")
        } else {
            // Fallback: grab last portion
            let parts = trimmed.split(separator: " ", maxSplits: 3)
            if parts.count > 2 {
                preview = String(parts.last?.prefix(60) ?? "")
                    .replacingOccurrences(of: "\n", with: " ")
            }
        }

        if inConnected {
            connectedCount += 1
        } else {
            directCount += 1
        }

        let distLabel = dist.isEmpty ? "" : " \(colorDist(dist))"
        let prefix = inConnected
            ? "  \(ANSIColor.dim)├\(ANSIColor.reset) "
            : "  \(ANSIColor.dim)•\(ANSIColor.reset) "
        lines.append("\(prefix)\(ANSIColor.cyan)\(shortId)\(ANSIColor.reset) \(ANSIColor.magenta)[\(projTopic)]\(ANSIColor.reset)\(distLabel)\(edge) \(ANSIColor.dim)\(preview)\(ANSIColor.reset)")
    }

    if lines.isEmpty { return }

    hookLog("\(ANSIColor.bold)\(hook): recalled \(directCount) direct + \(connectedCount) connected\(ANSIColor.reset)")
    for line in lines {
        hookLog(line)
    }
}

// MARK: - Shared Nudges

/// Returns the learning nudge for the given project (always fires).
func learningNudge(project: String) -> String {
    """
    ## Session learning

    If you haven't already, use the session-learner agent in the background to capture insights from this session.

    Project: \(project)

    IMPORTANT: Answer the user's request FIRST. Put the session-learner Task call at the END of your response. Do NOT relay or summarize its output — launch it silently.
    """
}

/// Threshold for first throttled learning nudge (tool calls into the session).
private let learningNudgeInitialThreshold = 15
/// Interval between subsequent throttled learning nudges (tool calls).
private let learningNudgeInterval = 30

/// Increments the tool call counter and returns a learning nudge only when threshold is crossed.
/// Use this for high-frequency hooks (PostToolUseFailure) to avoid habituation.
func throttledLearningNudge(project: String, sessionId: String?) -> String? {
    guard let state = getSessionState(sessionId: sessionId) else { return nil }

    state.toolCallCount += 1
    state.updatedAt = Date()

    let lastNudgeAt = state.learningNudgeLastToolCount
    let threshold = lastNudgeAt == 0 ? learningNudgeInitialThreshold : learningNudgeInterval
    let delta = state.toolCallCount - lastNudgeAt

    guard delta >= threshold else { return nil }

    state.learningNudgeLastToolCount = state.toolCallCount
    hookLog("Learning nudge injected (tool call \(state.toolCallCount), last nudge at \(lastNudgeAt))")

    return """
    ## Action required: capture session insights

    Spawn the session-learner agent in the background to record what you've learned this session. \
    Project: \(project). Answer the user FIRST, put the Task call at the END, and do NOT relay its output.
    """
}
