import EngramMemoryCore
import Foundation
#if canImport(EngramKit)
import EngramKit
import Lattice
#endif

#if canImport(EngramKit)

/// Open Lattice with the full schema at the default database path.
func openLattice(sessionId: String? = nil) -> Lattice? {
    let dbPath = defaultDbPath
    guard FileManager.default.fileExists(atPath: dbPath) else { return nil }

    // Always set up lattice debug logging so errors are captured even
    // for short-lived hooks that crash before initMemoryTools runs.
    // Always enable debug logging — stderr is captured in hooks.log
    Lattice.setLogLevel(.debug)

    if let sid = sessionId ?? ProcessInfo.processInfo.environment["CLAUDE_SESSION_ID"], !sid.isEmpty {
        let logDir = NSHomeDirectory() + "/.claude/memory-logs"
        let logPath = logDir + "/lattice-\(sid).log"
        let fm = FileManager.default
        if !fm.fileExists(atPath: logDir) {
            try? fm.createDirectory(atPath: logDir, withIntermediateDirectories: true)
        }
        Lattice.setLogLevel(.debug)
        Lattice.setLogFile(URL(fileURLWithPath: logPath))
    }

    return try? Lattice(
        Memory.self, Edge.self, Checkpoint.self, HookState.self, SessionState.self,
        configuration: .init(fileURL: URL(fileURLWithPath: dbPath), migration: engramMigrations)
    )
}

/// Initialize MemoryTools (Lattice + embedding model).
func initMemoryTools(sessionId: String? = nil) async -> MemoryTools? {
    sessionLog("initMemoryTools: opening lattice at \(defaultDbPath)", sessionId: sessionId)
    guard let lattice = openLattice() else {
        sessionLog("initMemoryTools: FAILED — no database at \(defaultDbPath)", sessionId: sessionId)
        hookLog("No memory database at \(defaultDbPath)")
        return nil
    }
    sessionLog("initMemoryTools: lattice opened", sessionId: sessionId)

    // Synced DB — same open the MCP server does (main.swift). Without it,
    // hook recall (advise/on-start/pre-tool) was blind to cross-device
    // memories: readLattice saw syncedLattice == nil and every synced
    // project silently fell back to local-only. Plain open, no WSS/IPC —
    // the daemon owns sync; hooks only read.
    var syncedRef: LatticeThreadSafeReference?
    let claudeDir = (defaultDbPath as NSString).deletingLastPathComponent
    let syncedDbPath = SyncService.syncedDbPath(claudeDir: claudeDir)
    if FileManager.default.fileExists(atPath: syncedDbPath) {
        let synced = try? Lattice(
            Memory.self, Edge.self, SyncConfig.self,
            configuration: .init(fileURL: URL(fileURLWithPath: syncedDbPath),
                                 migration: engramMigrations)
        )
        syncedRef = synced?.sendableReference
        sessionLog("initMemoryTools: synced lattice \(synced != nil ? "opened" : "FAILED to open") at \(syncedDbPath)", sessionId: sessionId)
    }

    // Per-session Lattice debug logging
    if let sid = sessionId, !sid.isEmpty {
        let logPath = memoryLogsDir + "/lattice-\(sid).log"
        let logURL = URL(fileURLWithPath: logPath)
        let fm = FileManager.default
        if !fm.fileExists(atPath: memoryLogsDir) {
            try? fm.createDirectory(atPath: memoryLogsDir, withIntermediateDirectories: true)
        }
        Lattice.setLogLevel(.debug)
        Lattice.setLogFile(logURL)
        sessionLog("initMemoryTools: lattice debug logging → \(logPath)", sessionId: sessionId)
    }

    // Group spokes — the same membership-scoped union the MCP server reads.
    // Without these the advise hook injects only personal memories, which is
    // the whole point of group sync missing from the surface where teammates'
    // knowledge is supposed to show up unprompted.
    var groupRefs: [MemoryTools.GroupSpokeRef] = []
    for spoke in SyncService.discoverGroupSpokes(claudeDir: claudeDir) {
        guard let lattice = try? Lattice(
            Memory.self, Edge.self, GroupProjectMap.self,
            configuration: .init(fileURL: URL(fileURLWithPath: spoke.path),
                                 migration: engramMigrations)
        ) else {
            sessionLog("initMemoryTools: FAILED to open group spoke at \(spoke.path)", sessionId: sessionId)
            continue
        }
        groupRefs.append(.init(groupId: spoke.groupId, path: spoke.path,
                               ref: lattice.sendableReference))
    }
    sessionLog("initMemoryTools: \(groupRefs.count) group spoke(s) attached", sessionId: sessionId)

    let modelPath = ProcessInfo.processInfo.environment["CLAUDE_MEMORY_MODEL"]
    sessionLog("initMemoryTools: loading embedder (modelPath=\(modelPath ?? "bundled"))", sessionId: sessionId)
    let embedder = EmbeddingService(modelPath: modelPath)
    await embedder.load()
    sessionLog("initMemoryTools: embedder loaded", sessionId: sessionId)

    let tools = MemoryTools(localRef: lattice.sendableReference, syncedRef: syncedRef,
                            groupRefs: groupRefs, embedder: embedder)
    // Every hook recall INJECTS its output into a tool-capable session, so
    // the hook path always fences foreign-authored content; the per-device
    // opt-out ("memory-hooks group-advise off" or the visualizer settings
    // row) excludes it entirely. One site — Advise, OnStart, and PreTool
    // all init through here.
    let includeGroup = lattice.objects(HookState.self)
        .where { $0.key == .adviseIncludeGroupMemories }
        .first?.value != "false"
    await tools.setForeignContentPolicy(fence: true, exclude: !includeGroup)
    return tools
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
        // Non-throwing helper by design (hook call sites treat state writes
        // as best-effort); a failed upsert logs rather than propagating.
        do { try lattice.add(HookState(key: key, value: value)) }
        catch { hookLog("setHookState(\(key)) failed: \(error)") }
    }
}

/// Get or create the SessionState row for a given session ID.
/// Run `body` with the session state while HOLDING the backing lattice.
/// The old shape returned the object after the function-scope lattice was
/// released — with the weak instance cache, a solo hook process then
/// deallocated the lattice under the live object, and the next property
/// WRITE trapped in the C++ backend (production memory-hooks OnStop
/// crashes, Jul 3-4: SessionState.stopNudgeSent.setter → setBool SIGTRAP).
@discardableResult
func withSessionState<T>(sessionId: String?, _ body: (SessionState) -> T) -> T? {
    guard let sessionId, !sessionId.isEmpty else { return nil }
    guard let lattice = openLattice(sessionId: sessionId) else { return nil }
    let state: SessionState
    if let existing = lattice.objects(SessionState.self).where({ $0.sessionId == sessionId }).first {
        state = existing
    } else {
        state = SessionState(sessionId: sessionId)
        do { try lattice.add(state) }
        catch {
            hookLog("withSessionState: could not create SessionState: \(error)")
            return nil
        }
    }
    return withExtendedLifetime(lattice) { body(state) }
}

#endif

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

/// Path to the per-session recall log for the statusline.
private let memoryLogsDir = NSHomeDirectory() + "/.claude/memory-logs"

/// Parse a directRecall result string and log a compact summary of each recalled memory.
/// Format: one line per memory with id (short), project/topic, distance, and content preview.
/// When sessionId is provided, also writes a statusline-optimized version to the session recall log.
func logRecalledMemories(_ result: String, hook: String, sessionId: String? = nil) {
    // Each memory block starts with [id:UUID] [project/topic]
    // Direct results: [id:UUID] [project/topic] (distance: 0.123...) content
    // Connected results come after "--- Connected (graph traversal, depth: N) ---"

    var directCount = 0
    var connectedCount = 0
    var inConnected = false
    var lines: [String] = []
    var statusLines: [String] = []

    for block in result.components(separatedBy: "\n\n") {
        let trimmed = block.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { continue }

        if trimmed.hasPrefix("--- Connected") {
            inConnected = true
            continue
        }
        if trimmed.hasPrefix("⚠️ Weak recall") {
            lines.append("  \(ANSIColor.yellow)\(ANSIColor.bold)⚠️  weak recall signal\(ANSIColor.reset)")
            statusLines.append("  \(ANSIColor.yellow)\(ANSIColor.bold)⚠️  weak recall\(ANSIColor.reset)")
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
        var statusEdge = ""
        if inConnected {
            if let edgeRange = trimmed.range(of: "--[") {
                let edgeStart = edgeRange.lowerBound
                if let edgeEnd = trimmed[edgeStart...].range(of: "]--") {
                    let fullEdge = String(trimmed[edgeStart...edgeEnd.upperBound])
                    edge = " \(ANSIColor.blue)" + fullEdge + ANSIColor.reset

                    // Statusline: extract edge type and direction, plus target short ID
                    let edgeType = String(fullEdge.drop(while: { $0 != "[" }).dropFirst()
                        .prefix(while: { $0 != "]" }))
                    let isOutgoing = fullEdge.hasSuffix(">")

                    // Extract target short ID from content after edge
                    var targetShortId = ""
                    if let targetIdRange = trimmed.range(of: "[id:", range: edgeEnd.upperBound..<trimmed.endIndex) {
                        let tidStart = targetIdRange.upperBound
                        targetShortId = String(trimmed[tidStart...].prefix(8))
                    }

                    let arrow = isOutgoing ? "→" : "←"
                    statusEdge = " \(ANSIColor.blue)\(edgeType) \(arrow)\(targetShortId.isEmpty ? "" : " \(targetShortId)")\(ANSIColor.reset)"
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

        // Hook log line (verbose)
        let distLabel = dist.isEmpty ? "" : " \(colorDist(dist))"
        let prefix = inConnected
            ? "  \(ANSIColor.dim)├\(ANSIColor.reset) "
            : "  \(ANSIColor.dim)•\(ANSIColor.reset) "
        lines.append("\(prefix)\(ANSIColor.cyan)\(shortId)\(ANSIColor.reset) \(ANSIColor.magenta)[\(projTopic)]\(ANSIColor.reset)\(distLabel)\(edge) \(ANSIColor.dim)\(preview)\(ANSIColor.reset)")

        // Statusline log line (compact)
        let shortDist = dist.isEmpty ? "" : {
            // ".22" instead of "0.216"
            if let d = Double(dist) {
                return " \(colorDist(String(format: ".%02d", Int(d * 100))))"
            }
            return " \(colorDist(dist))"
        }()
        let statusPrefix = inConnected
            ? "  \(ANSIColor.dim)├\(ANSIColor.reset) "
            : "  \(ANSIColor.dim)•\(ANSIColor.reset) "
        let statusContent = inConnected
            ? statusEdge
            : "\(shortDist) \(ANSIColor.dim)\(String(preview.prefix(50)))\(ANSIColor.reset)"
        statusLines.append("\(statusPrefix)\(ANSIColor.cyan)\(shortId)\(ANSIColor.reset) \(ANSIColor.magenta)[\(projTopic)]\(ANSIColor.reset)\(statusContent)")
    }

    if lines.isEmpty { return }

    // Write to hooks.log (verbose format)
    hookLog("\(ANSIColor.bold)\(hook): recalled \(directCount) direct + \(connectedCount) connected\(ANSIColor.reset)")
    for line in lines {
        hookLog(line)
    }

    // Write to session recall log (statusline format)
    if let sessionId, !sessionId.isEmpty {
        writeSessionRecallLog(
            sessionId: sessionId,
            directCount: directCount,
            connectedCount: connectedCount,
            lines: statusLines
        )
    }
}

/// Write the statusline-optimized recall log for this session (overwrites on each recall).
private func writeSessionRecallLog(sessionId: String, directCount: Int, connectedCount: Int, lines: [String]) {
    sessionLog("writeSessionRecallLog: dir=\(memoryLogsDir), sessionId=\(sessionId), \(directCount)+\(connectedCount) entries", sessionId: sessionId)
    let fm = FileManager.default
    if !fm.fileExists(atPath: memoryLogsDir) {
        sessionLog("writeSessionRecallLog: creating directory", sessionId: sessionId)
        try? fm.createDirectory(atPath: memoryLogsDir, withIntermediateDirectories: true)
    }
    let path = memoryLogsDir + "/recall-\(sessionId).log"
    let df = DateFormatter()
    df.dateFormat = "h:mm a"
    let timeStr = df.string(from: Date())
    var content = "\(directCount)+\(connectedCount) recalled @ \(timeStr)\n"
    for line in lines {
        content += line + "\n"
    }
    do {
        try content.write(toFile: path, atomically: true, encoding: .utf8)
        sessionLog("writeSessionRecallLog: wrote \(content.count) bytes to \(path)", sessionId: sessionId)
    } catch {
        sessionLog("writeSessionRecallLog: FAILED: \(error)", sessionId: sessionId)
        hookLog("Failed to write session recall log: \(error)")
    }
}

/// Write a skip indicator to the recall log (keeps previous recall visible, adds skip note).
func writeSessionRecallSkip(sessionId: String) {
    let path = memoryLogsDir + "/recall-\(sessionId).log"
    // If a recall log already exists, don't overwrite it — the previous recall is still useful.
    // Only write the skip file if there's nothing to show yet.
    if FileManager.default.fileExists(atPath: path) { return }
    let fm = FileManager.default
    if !fm.fileExists(atPath: memoryLogsDir) {
        try? fm.createDirectory(atPath: memoryLogsDir, withIntermediateDirectories: true)
    }
    let dim = "\u{1B}[2m"
    let rst = "\u{1B}[0m"
    let content = "\(dim)recall skipped (conversational filler)\(rst)\n"
    try? content.write(toFile: path, atomically: true, encoding: .utf8)
}

/// Delete session logs only if they're older than 5 minutes (avoids race with mid-session reconnects).
func cleanupSessionLogs(sessionId: String) {
    let fm = FileManager.default
    let paths = [
        memoryLogsDir + "/recall-\(sessionId).log",
        memoryLogsDir + "/debug-\(sessionId).log",
        memoryLogsDir + "/lattice-\(sessionId).log",
        memoryLogsDir + "/lattice-mcp-\(sessionId).log",
    ]
    let staleThreshold: TimeInterval = 300 // 5 minutes
    for path in paths {
        guard let attrs = try? fm.attributesOfItem(atPath: path),
              let modified = attrs[.modificationDate] as? Date,
              Date().timeIntervalSince(modified) > staleThreshold else { continue }
        try? fm.removeItem(atPath: path)
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
    let shouldNudge = updateSessionCounters(sessionId: sessionId) { state -> Bool in
        state.toolCallCount += 1

        let lastNudgeAt = state.learningNudgeLastToolCount
        let threshold = lastNudgeAt == 0 ? learningNudgeInitialThreshold : learningNudgeInterval
        let delta = state.toolCallCount - lastNudgeAt

        guard delta >= threshold else { return false }
        state.learningNudgeLastToolCount = state.toolCallCount
        return true
    }
    guard shouldNudge == true else { return nil }
    hookLog("Learning nudge injected (throttled)")

    return """
    ## Action required: capture session insights

    Spawn the session-learner agent in the background to record what you've learned this session. \
    Project: \(project). Answer the user FIRST, put the Task call at the END, and do NOT relay its output.
    """
}
