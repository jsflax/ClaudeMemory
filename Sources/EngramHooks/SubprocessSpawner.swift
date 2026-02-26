import Foundation

/// Rotate a log file if it exceeds `maxBytes`. Keeps one `.1` backup.
func rotateFileIfNeeded(_ path: String, maxBytes: Int = 256 * 1024) {
    let fm = FileManager.default
    guard let attrs = try? fm.attributesOfItem(atPath: path),
          let size = attrs[.size] as? Int,
          size > maxBytes else { return }
    let backup = path + ".1"
    try? fm.removeItem(atPath: backup)
    try? fm.moveItem(atPath: path, toPath: backup)
}

/// Spawn a `claude` CLI subprocess as a fire-and-forget background process.
///
/// Uses `/bin/sh -c` with shell redirection to capture output to a log file.
/// The subprocess is fully detached from the calling process's stdio.
///
/// - Parameters:
///   - prompt: The prompt to pass via `claude -p`.
///   - systemPrompt: Appended as `--append-system-prompt`.
///   - allowedTools: Comma-separated tool names for `--allowedTools`.
///   - model: Model name (e.g. "sonnet").
///   - envGuard: Environment variable set on the subprocess to prevent recursion.
///   - logPath: Absolute path to the log file for stdout/stderr capture.
///   - cwd: Optional working directory for the subprocess.
func spawnClaudeSubprocess(
    prompt: String,
    systemPrompt: String,
    allowedTools: String,
    model: String,
    envGuard: (key: String, value: String),
    logPath: String,
    cwd: String?
) throws {
    let escapedPrompt = prompt.replacingOccurrences(of: "'", with: "'\\''")
    let escapedSystemPrompt = systemPrompt.replacingOccurrences(of: "'", with: "'\\''")

    rotateFileIfNeeded(logPath)

    let shellCommand = """
    echo '===== started at '\\''\(ISO8601DateFormatter().string(from: Date()))'\\'' =====' >> '\(logPath)' && \
    claude -p '\(escapedPrompt)' \
      --model \(model) \
      --allowedTools '\(allowedTools)' \
      --no-session-persistence \
      --output-format text \
      --append-system-prompt '\(escapedSystemPrompt)' \
      >> '\(logPath)' 2>&1
    """

    let sh = Process()
    sh.executableURL = URL(fileURLWithPath: "/bin/sh")
    sh.arguments = ["-c", shellCommand]
    sh.environment = ProcessInfo.processInfo.environment.merging(
        [envGuard.key: envGuard.value],
        uniquingKeysWith: { _, new in new }
    )
    if let cwd {
        sh.currentDirectoryURL = URL(fileURLWithPath: cwd)
    }

    // Detach from our process's stdio
    sh.standardOutput = FileHandle.nullDevice
    sh.standardError = FileHandle.nullDevice

    try sh.run()
    // Don't wait — fire and forget
}

/// Strip YAML frontmatter (--- delimited block) from markdown content.
func stripFrontmatter(_ content: String) -> String {
    guard content.hasPrefix("---") else { return content }
    let lines = content.components(separatedBy: .newlines)
    var endIndex = 0
    for i in 1..<lines.count {
        if lines[i].hasPrefix("---") {
            endIndex = i + 1
            break
        }
    }
    guard endIndex > 0 else { return content }
    return lines[endIndex...].joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
}

/// Load an agent system prompt from `~/.claude/agents/<name>.md`, stripping frontmatter.
/// Falls back to the provided inline default if the file doesn't exist.
func loadAgentSystemPrompt(name: String, fallback: String) -> String {
    let possiblePaths = [
        NSHomeDirectory() + "/.claude/agents/\(name).md",
        Bundle.main.bundlePath + "/../../../agents/\(name).md",
    ]

    for path in possiblePaths {
        if let content = try? String(contentsOfFile: path, encoding: .utf8) {
            return stripFrontmatter(content)
        }
    }

    return fallback
}
