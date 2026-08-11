import Foundation
import Testing

@testable import EngramKit

// ============================================================================
// The installer used to merge hooks per EVENT KEY — every key we ship was
// overwritten wholesale on every app launch — which silently reset any
// timeout the user had raised. Observed live: UserPromptSubmit 60 → 5, the
// harness then killing the advise hook mid-recall, memory silently missing
// from every prompt. These tests pin the per-HOOK merge that replaced it.
// ============================================================================

private let installDir = "/Users/tester/.claude/bin"

/// The real shipped config, so a change to InstallConfig is exercised here too.
private var shipped: [String: Any] { InstallConfig.hooksSettings(installDir: installDir) }
private var shippedHooks: [String: Any] { shipped["hooks"] as! [String: Any] }

// MARK: - Reading helpers

private func groups(_ hooks: [String: Any], _ event: String) -> [[String: Any]] {
    hooks[event] as? [[String: Any]] ?? []
}

private func entries(_ hooks: [String: Any], _ event: String) -> [[String: Any]] {
    groups(hooks, event).flatMap { $0["hooks"] as? [[String: Any]] ?? [] }
}

/// Every hook entry under `event` whose command is ours for `subcommand`.
private func ours(_ hooks: [String: Any], _ event: String,
                  _ subcommand: String) -> [[String: Any]] {
    entries(hooks, event).filter {
        guard let command = $0["command"] as? String else { return false }
        return HookSettingsMerge.command(command, registers: "memory-hooks \(subcommand)")
    }
}

private func timeout(_ entry: [String: Any]) -> Int? {
    (entry["timeout"] as? NSNumber)?.intValue
}

private func hook(_ command: String, timeout: Int) -> [String: Any] {
    ["type": "command", "command": command, "timeout": timeout]
}

/// Round-trip through JSONSerialization so the tests see the same NSNumber /
/// NSArray bridging the installer sees when it reads settings.json off disk.
private func asLoadedJSON(_ dictionary: [String: Any]) -> [String: Any] {
    let data = try! JSONSerialization.data(withJSONObject: dictionary)
    return try! JSONSerialization.jsonObject(with: data) as! [String: Any]
}

// MARK: - Tests

@Test("Fresh install registers every shipped hook exactly once")
func hookMergeFreshInstall() {
    let merged = HookSettingsMerge.mergedHooks(existing: [:], shipped: shippedHooks)

    for (event, value) in shippedHooks {
        let shippedEntries = (value as! [[String: Any]])
            .flatMap { $0["hooks"] as! [[String: Any]] }
        #expect(entries(merged, event).count == shippedEntries.count,
                "\(event) should carry exactly the shipped hooks")
    }
    #expect(ours(merged, "UserPromptSubmit", "advise").count == 1)
    #expect(timeout(ours(merged, "UserPromptSubmit", "advise")[0]) == 60)
    // Matchers ride along on a fresh install.
    #expect(groups(merged, "PreToolUse").first?["matcher"] as? String == "Agent")
    #expect(groups(merged, "PreCompact").first?["matcher"] as? String == "auto")
}

@Test("A timeout the user raised survives re-running the installer")
func hookMergePreservesRaisedTimeout() {
    // The user pushed advise from 60 to 180 because their database is large.
    let existing = asLoadedJSON([
        "UserPromptSubmit": [["hooks": [
            hook("\(installDir)/memory-hooks advise 2>/dev/null", timeout: 180)
        ]]]
    ])

    let merged = HookSettingsMerge.mergedHooks(existing: existing, shipped: shippedHooks)
    let advise = ours(merged, "UserPromptSubmit", "advise")

    #expect(advise.count == 1, "must not duplicate the registration")
    #expect(timeout(advise[0]) == 180, "the user's raised timeout is theirs to keep")
}

@Test("A timeout the user LOWERED is raised back to the shipped floor")
func hookMergeRaisesLoweredTimeout() {
    // max(), not "existing wins": the 5 → 60 fix has to reach the machines
    // still carrying a 5 written by an older installer.
    let existing = asLoadedJSON([
        "UserPromptSubmit": [["hooks": [
            hook("\(installDir)/memory-hooks advise 2>/dev/null", timeout: 5)
        ]]]
    ])

    let merged = HookSettingsMerge.mergedHooks(existing: existing, shipped: shippedHooks)
    #expect(timeout(ours(merged, "UserPromptSubmit", "advise")[0]) == 60)
}

@Test("Other hooks registered under the same event key are left alone")
func hookMergePreservesForeignHooksUnderOurEvents() {
    let existing = asLoadedJSON([
        "UserPromptSubmit": [
            ["hooks": [hook("/opt/peon-ping/ping.sh prompt", timeout: 2)]],
            ["hooks": [hook("\(installDir)/memory-hooks advise 2>/dev/null", timeout: 90)]],
        ],
        // An event key we don't ship at all.
        "Notification": [["hooks": [hook("/opt/peon-ping/ping.sh notify", timeout: 2)]]],
    ])

    let merged = HookSettingsMerge.mergedHooks(existing: existing, shipped: shippedHooks)

    let prompt = entries(merged, "UserPromptSubmit")
    #expect(prompt.contains { ($0["command"] as? String) == "/opt/peon-ping/ping.sh prompt" },
            "the user's own UserPromptSubmit hook must survive")
    #expect(prompt.count == 2, "one theirs, one ours — no duplicates")
    #expect(timeout(ours(merged, "UserPromptSubmit", "advise")[0]) == 90)

    let notification = entries(merged, "Notification")
    #expect(notification.count == 1)
    #expect(notification[0]["command"] as? String == "/opt/peon-ping/ping.sh notify")
}

@Test("A hook of theirs sharing OUR group keeps its matcher and its entry")
func hookMergeSharedGroupIsNotRetargeted() {
    // Our pre-tool hook and one of theirs in the same matcher group. We may
    // refresh our command, but must not retarget a matcher their hook rides on.
    let existing = asLoadedJSON([
        "PreToolUse": [[
            "matcher": "Bash|Agent",
            "hooks": [
                hook("/old/bin/memory-hooks pre-tool 2>/dev/null", timeout: 30),
                hook("/opt/audit/log-tool.sh", timeout: 4),
            ],
        ]]
    ])

    let merged = HookSettingsMerge.mergedHooks(existing: existing, shipped: shippedHooks)
    let preToolGroups = groups(merged, "PreToolUse")

    #expect(preToolGroups.count == 1, "ours was found in their group — nothing appended")
    #expect(preToolGroups[0]["matcher"] as? String == "Bash|Agent",
            "a shared group's matcher stays as the user left it")
    #expect(entries(merged, "PreToolUse").count == 2)
    #expect(ours(merged, "PreToolUse", "pre-tool")[0]["command"] as? String
                == "\(installDir)/memory-hooks pre-tool 2>/dev/null",
            "our command is still refreshed to the current install path")
    #expect(timeout(ours(merged, "PreToolUse", "pre-tool")[0]) == 30)
}

@Test("A moved install directory updates the command and keeps the timeout")
func hookMergeUpdatesChangedInstallPath() {
    // Registered by an older install under a different path, with a raised
    // timeout and an obsolete flag.
    let existing = asLoadedJSON([
        "UserPromptSubmit": [["hooks": [
            hook("/usr/local/engram/bin/memory-hooks advise --legacy 2>/dev/null", timeout: 120)
        ]]],
        "SessionStart": [["hooks": [
            hook("/usr/local/engram/bin/memory-hooks on-start 2>/dev/null", timeout: 5)
        ]]],
    ])

    let merged = HookSettingsMerge.mergedHooks(existing: existing, shipped: shippedHooks)

    let advise = ours(merged, "UserPromptSubmit", "advise")
    #expect(advise.count == 1, "matched by binary+subcommand, not by path — no duplicate")
    #expect(advise[0]["command"] as? String
                == "\(installDir)/memory-hooks advise 2>/dev/null",
            "the stale path and the dropped flag are both replaced")
    #expect(timeout(advise[0]) == 120, "the raised timeout still survives the path change")

    #expect(ours(merged, "SessionStart", "on-start")[0]["command"] as? String
                == "\(installDir)/memory-hooks on-start 2>/dev/null")
}

@Test("Re-running the merge changes nothing (idempotent)")
func hookMergeIsIdempotent() {
    let existing = asLoadedJSON([
        "UserPromptSubmit": [
            ["hooks": [hook("/opt/peon-ping/ping.sh prompt", timeout: 2)]],
            ["hooks": [hook("\(installDir)/memory-hooks advise 2>/dev/null", timeout: 90)]],
        ]
    ])

    let once = HookSettingsMerge.mergedHooks(existing: existing, shipped: shippedHooks)
    let twice = HookSettingsMerge.mergedHooks(existing: once, shipped: shippedHooks)

    for event in shippedHooks.keys {
        #expect(entries(once, event).count == entries(twice, event).count,
                "\(event) grew on the second pass")
    }
    #expect(timeout(ours(twice, "UserPromptSubmit", "advise")[0]) == 90)
}

@Test("Subcommand identity is bounded, not a loose substring")
func hookIdentityMatchingIsBounded() {
    #expect(HookSettingsMerge.hookIdentity(
        command: "\(installDir)/memory-hooks advise 2>/dev/null") == "memory-hooks advise")
    #expect(HookSettingsMerge.hookIdentity(command: "/bin/echo hello") == nil)
    #expect(HookSettingsMerge.hookIdentity(command: "\(installDir)/memory-hooks") == nil)

    #expect(HookSettingsMerge.command("/x/memory-hooks on-start 2>/dev/null",
                                      registers: "memory-hooks on-start"))
    #expect(!HookSettingsMerge.command("/x/memory-hooks on-start-extra",
                                       registers: "memory-hooks on-start"),
            "a longer subcommand is a different hook")
    #expect(!HookSettingsMerge.command("/x/my-memory-hooks on-start",
                                       registers: "memory-hooks on-start"),
            "a different binary is a different hook")
    #expect(HookSettingsMerge.command("sh -c \"/x/memory-hooks advise\"",
                                      registers: "memory-hooks advise"),
            "a wrapped invocation is still ours")
}

// MARK: - Whole-settings merge

@Test("Merging whole settings unions permissions and keeps unrelated keys")
func settingsMergePreservesUnrelatedConfiguration() {
    let existing = asLoadedJSON([
        "model": "opus",
        "statusLine": ["type": "command", "command": "/opt/statusline.sh"],
        "permissions": [
            "allow": ["Bash(git:*)"],
            "deny": ["Read(./.env)"],
        ],
        "hooks": [
            "UserPromptSubmit": [["hooks": [
                hook("\(installDir)/memory-hooks advise 2>/dev/null", timeout: 180)
            ]]]
        ],
    ])

    let merged = HookSettingsMerge.merged(existingSettings: existing, shipped: shipped)

    #expect(merged["model"] as? String == "opus")
    #expect((merged["statusLine"] as? [String: Any])?["command"] as? String
                == "/opt/statusline.sh")

    let permissions = merged["permissions"] as! [String: Any]
    let allow = permissions["allow"] as! [String]
    #expect(allow.contains("Bash(git:*)"), "their allow entries stay")
    #expect(allow.contains("mcp__memory__*"), "ours is added")
    #expect(permissions["deny"] as? [String] == ["Read(./.env)"],
            "deny was dropped by the old installer whenever allow looked unfamiliar")

    let hooks = merged["hooks"] as! [String: Any]
    #expect(timeout(ours(hooks, "UserPromptSubmit", "advise")[0]) == 180)
    #expect(ours(hooks, "SessionEnd", "on-end").count == 1, "missing hooks still get added")
}

@Test("Settings with no permissions object get the shipped one")
func settingsMergeAddsMissingPermissions() {
    let merged = HookSettingsMerge.merged(existingSettings: asLoadedJSON(["model": "opus"]),
                                          shipped: shipped)
    let allow = (merged["permissions"] as? [String: Any])?["allow"] as? [String]
    #expect(allow == ["mcp__memory__*"])
}

@Test("Merged settings stay JSON-serializable")
func settingsMergeRoundTripsThroughJSON() {
    let existing = asLoadedJSON([
        "hooks": ["Notification": [["hooks": [hook("/opt/peon-ping/ping.sh notify", timeout: 2)]]]]
    ])
    let merged = HookSettingsMerge.merged(existingSettings: existing, shipped: shipped)
    #expect(JSONSerialization.isValidJSONObject(merged))
    #expect((try? JSONSerialization.data(withJSONObject: merged,
                                         options: [.prettyPrinted, .sortedKeys])) != nil)
}
