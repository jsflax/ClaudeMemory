import Foundation

// Hook/process logging shared by every surface (EngramKit tools, hooks —
// local AND remote/Linux): per-session debug logs under
// ~/.claude/memory-logs.

private let memoryLogsDir = NSHomeDirectory() + "/.claude/memory-logs"

/// Global session ID for per-session debug logging in short-lived hook processes.
nonisolated(unsafe) public var currentSessionId: String?

/// Write a timestamped line to ~/.claude/memory-logs/debug-<sessionId>.log.
public func sessionLog(_ message: String, sessionId: String? = nil) {
    let sid = sessionId ?? currentSessionId
    guard let sid, !sid.isEmpty else { return }
    let dir = memoryLogsDir
    let path = dir + "/debug-\(sid).log"
    let fm = FileManager.default
    if !fm.fileExists(atPath: dir) {
        try? fm.createDirectory(atPath: dir, withIntermediateDirectories: true)
    }
    let timestamp = ISO8601DateFormatter().string(from: Date())
    let line = "\(timestamp) \(message)\n"
    if let fh = FileHandle(forWritingAtPath: path) {
        fh.seekToEndOfFile()
        fh.write(Data(line.utf8))
        fh.closeFile()
    } else {
        fm.createFile(atPath: path, contents: Data(line.utf8))
    }
}
