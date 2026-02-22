import Foundation

enum CLIInstaller {
    private static let installDir = NSHomeDirectory() + "/.claude/bin"
    private static let agentsDir = NSHomeDirectory() + "/.claude/agents"
    private static let versionFile = NSHomeDirectory() + "/.claude/bin/.memory-version"

    static func syncIfNeeded() {
        guard let bundledVersion = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String,
              let cliDir = Bundle.main.resourceURL?.appendingPathComponent("cli")
        else { return }

        let fm = FileManager.default
        guard fm.fileExists(atPath: cliDir.path) else { return }

        let installedVersion = (try? String(contentsOfFile: versionFile, encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines)

        guard installedVersion == nil || compareVersions(bundledVersion, isNewerThan: installedVersion!) else {
            return
        }

        do {
            try fm.createDirectory(atPath: installDir, withIntermediateDirectories: true)

            // Copy CLI binaries
            let binaries = ["memory", "memory-hooks"]
            for binary in binaries {
                let src = cliDir.appendingPathComponent(binary)
                let dst = URL(fileURLWithPath: installDir).appendingPathComponent(binary)
                if fm.fileExists(atPath: src.path) {
                    try? fm.removeItem(at: dst)
                    try fm.copyItem(at: src, to: dst)
                    // Make executable
                    try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: dst.path)
                }
            }

            // Copy .bundle resources
            if let contents = try? fm.contentsOfDirectory(atPath: cliDir.path) {
                for item in contents where item.hasSuffix(".bundle") {
                    let src = cliDir.appendingPathComponent(item)
                    let dst = URL(fileURLWithPath: installDir).appendingPathComponent(item)
                    try? fm.removeItem(at: dst)
                    try fm.copyItem(at: src, to: dst)
                }
            }

            // Copy agent definitions
            let agentsSrc = cliDir.appendingPathComponent("agents")
            if fm.fileExists(atPath: agentsSrc.path) {
                try fm.createDirectory(atPath: agentsDir, withIntermediateDirectories: true)
                if let agents = try? fm.contentsOfDirectory(atPath: agentsSrc.path) {
                    for agent in agents where agent.hasSuffix(".md") {
                        let src = agentsSrc.appendingPathComponent(agent)
                        let dst = URL(fileURLWithPath: agentsDir).appendingPathComponent(agent)
                        try? fm.removeItem(at: dst)
                        try fm.copyItem(at: src, to: dst)
                    }
                }
            }

            // Write version marker
            try bundledVersion.write(toFile: versionFile, atomically: true, encoding: .utf8)

            // Register MCP server if not already registered
            if installedVersion == nil {
                registerMCPServer()
            }
        } catch {
            // Best-effort — don't crash the app if CLI sync fails
        }
    }

    private static func registerMCPServer() {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [
            "claude", "mcp", "add", "--scope", "user", "--transport", "stdio",
            "memory", "--", "\(installDir)/memory"
        ]
        // Unset CLAUDECODE to avoid nested session issues
        var env = ProcessInfo.processInfo.environment
        env.removeValue(forKey: "CLAUDECODE")
        process.environment = env
        try? process.run()
    }

    private static func compareVersions(_ a: String, isNewerThan b: String) -> Bool {
        let aParts = a.split(separator: ".").compactMap { Int($0) }
        let bParts = b.split(separator: ".").compactMap { Int($0) }
        let count = max(aParts.count, bParts.count)
        for i in 0..<count {
            let aVal = i < aParts.count ? aParts[i] : 0
            let bVal = i < bParts.count ? bParts[i] : 0
            if aVal > bVal { return true }
            if aVal < bVal { return false }
        }
        return false
    }
}
