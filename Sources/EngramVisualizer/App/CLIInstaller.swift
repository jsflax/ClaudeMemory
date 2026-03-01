import Foundation
import EngramKit

enum CLIInstaller {
    private static let claudeDir = NSHomeDirectory() + "/.claude"
    private static let installDir = claudeDir + "/bin"
    private static let agentsDir = claudeDir + "/agents"
    private static let versionFile = installDir + "/.memory-version"
    private static let claudeMDPath = claudeDir + "/CLAUDE.md"
    private static let settingsPath = claudeDir + "/settings.json"

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

        let isFirstInstall = installedVersion == nil

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

            // Register MCP server (re-register on upgrade to update path if needed)
            registerMCPServer(isFirstInstall: isFirstInstall)

            // Install CLAUDE.md instructions and hooks config
            installClaudeMD()
            installSettings()
        } catch {
            // Best-effort — don't crash the app if CLI sync fails
        }
    }

    // MARK: - MCP Server Registration

    private static func registerMCPServer(isFirstInstall: Bool) {
        let env = cleanEnv()

        // Remove first to ensure clean state (matches install.sh behavior)
        if !isFirstInstall {
            let remove = Process()
            remove.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            remove.arguments = ["claude", "mcp", "remove", "memory"]
            remove.environment = env
            try? remove.run()
            remove.waitUntilExit()
        }

        let add = Process()
        add.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        add.arguments = [
            "claude", "mcp", "add", "--scope", "user", "--transport", "stdio",
            "memory", "--", "\(installDir)/memory"
        ]
        add.environment = env
        try? add.run()
    }

    // MARK: - CLAUDE.md

    private static func installClaudeMD() {
        let fm = FileManager.default
        let memoryBlock = InstallConfig.memoryBlockMarkdown

        if fm.fileExists(atPath: claudeMDPath) {
            guard let existing = try? String(contentsOfFile: claudeMDPath, encoding: .utf8) else { return }

            if existing.contains("\n# Memory\n") || existing.hasPrefix("# Memory\n") {
                // Replace existing Memory section
                let lines = existing.components(separatedBy: "\n")
                var result: [String] = []
                var skipping = false
                for line in lines {
                    if line == "# Memory" {
                        skipping = true
                        continue
                    }
                    if skipping && line.hasPrefix("# ") {
                        skipping = false
                    }
                    if !skipping {
                        result.append(line)
                    }
                }
                // Trim trailing blank lines
                while let last = result.last, last.trimmingCharacters(in: .whitespaces).isEmpty {
                    result.removeLast()
                }
                let updated = result.joined(separator: "\n") + "\n\n" + memoryBlock + "\n"
                try? updated.write(toFile: claudeMDPath, atomically: true, encoding: .utf8)
            } else {
                // Append Memory section
                let updated = existing.trimmingCharacters(in: .newlines) + "\n\n" + memoryBlock + "\n"
                try? updated.write(toFile: claudeMDPath, atomically: true, encoding: .utf8)
            }
        } else {
            // Create new file
            try? (memoryBlock + "\n").write(toFile: claudeMDPath, atomically: true, encoding: .utf8)
        }
    }

    // MARK: - settings.json (hooks)

    private static func installSettings() {
        let fm = FileManager.default
        let hooksConfig = InstallConfig.hooksSettings(installDir: installDir)

        if fm.fileExists(atPath: settingsPath) {
            // Merge into existing settings
            guard let data = fm.contents(atPath: settingsPath),
                  var existing = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { return }

            // Merge permissions.allow
            if var existingPerms = existing["permissions"] as? [String: Any],
               var existingAllow = existingPerms["allow"] as? [String] {
                if !existingAllow.contains("mcp__memory__*") {
                    existingAllow.append("mcp__memory__*")
                    existingPerms["allow"] = existingAllow
                    existing["permissions"] = existingPerms
                }
            } else {
                existing["permissions"] = hooksConfig["permissions"]
            }

            // Replace hooks, keep our hooks up to date
            var hooks = hooksConfig["hooks"] as! [String: Any]
            if var existingHooks = existing["hooks"] as? [String: Any] {
                // Overwrite our hook keys, preserve any user-added hooks
                for (key, value) in hooks {
                    existingHooks[key] = value
                }
                hooks = existingHooks
            }
            existing["hooks"] = hooks

            if let json = try? JSONSerialization.data(withJSONObject: existing, options: [.prettyPrinted, .sortedKeys]) {
                try? json.write(to: URL(fileURLWithPath: settingsPath))
            }
        } else {
            // Create new settings file
            if let json = try? JSONSerialization.data(withJSONObject: hooksConfig, options: [.prettyPrinted, .sortedKeys]) {
                try? json.write(to: URL(fileURLWithPath: settingsPath))
            }
        }
    }

    // MARK: - Helpers

    private static func cleanEnv() -> [String: String] {
        var env = ProcessInfo.processInfo.environment
        env.removeValue(forKey: "CLAUDECODE")
        return env
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
