import Foundation

/// Merging our hook registrations into a user's `~/.claude/settings.json`
/// WITHOUT trampling what they wrote there.
///
/// The installer used to merge per event key — `existingHooks[key] = value`
/// for every key we ship — which silently reset any timeout the user had
/// raised, on every app launch and every upgrade. Observed live: a
/// UserPromptSubmit timeout of 60 knocked back to 5, so the harness killed
/// the advise hook mid-recall on a large database and the turn ran with no
/// memory at all, with no error anywhere.
///
/// So merge per HOOK instead. A registration is ours if its command names our
/// binary and subcommand (`memory-hooks advise`); ours gets its command
/// refreshed — install paths and flags must still be able to change — and
/// keeps `max(existing, shipped)` as its timeout. Everything else under the
/// same event key is left exactly as found. Today the user's own hooks
/// survive only because we happen not to ship the keys they use; that is
/// luck, and this makes it design.
public enum HookSettingsMerge {

    /// The hook binary we own. Matched on the basename so a moved install
    /// directory still resolves to the same registration.
    public static let hookBinaryName = "memory-hooks"

    /// The one permission the installer needs in `permissions.allow`.
    public static let requiredPermission = "mcp__memory__*"

    // MARK: - Whole-file merge

    /// Merge the shipped install config into a parsed `settings.json`.
    ///
    /// Only `permissions.allow` and `hooks` are touched; every other key the
    /// user has (statusLine, model, env, their own top-level settings) is
    /// carried through untouched.
    public static func merged(existingSettings: [String: Any],
                              shipped: [String: Any]) -> [String: Any] {
        var result = existingSettings

        // permissions.allow: a union, never a replacement. The old code
        // replaced the whole `permissions` object whenever `allow` was
        // missing or oddly typed, which took any `deny` list with it.
        if var permissions = result["permissions"] as? [String: Any] {
            var allow = permissions["allow"] as? [String] ?? []
            if !allow.contains(requiredPermission) {
                allow.append(requiredPermission)
            }
            permissions["allow"] = allow
            result["permissions"] = permissions
        } else if let shippedPermissions = shipped["permissions"] {
            result["permissions"] = shippedPermissions
        }

        result["hooks"] = mergedHooks(
            existing: result["hooks"] as? [String: Any] ?? [:],
            shipped: shipped["hooks"] as? [String: Any] ?? [:])
        return result
    }

    // MARK: - Hook merge

    /// Merge shipped hook registrations into existing ones, per hook.
    ///
    /// - Parameters:
    ///   - existing: the `hooks` object from the user's settings.json.
    ///   - shipped: the `hooks` object from `InstallConfig.hooksSettings`.
    /// - Returns: the merged `hooks` object. Event keys we don't ship are
    ///   returned untouched.
    public static func mergedHooks(existing: [String: Any],
                                   shipped: [String: Any]) -> [String: Any] {
        var result = existing

        for (event, shippedValue) in shipped {
            guard let shippedGroups = asGroups(shippedValue) else {
                // A shape we don't recognise. Install it only where the user
                // has nothing, rather than overwriting something we can't read.
                if result[event] == nil { result[event] = shippedValue }
                continue
            }

            var groups = asGroups(result[event]) ?? []

            for shippedGroup in shippedGroups {
                let shippedEntries = asEntries(shippedGroup["hooks"]) ?? []
                var unmatched: [[String: Any]] = []

                for entry in shippedEntries {
                    guard let command = entry["command"] as? String,
                          let identity = hookIdentity(command: command),
                          let hit = locate(identity: identity, in: groups) else {
                        unmatched.append(entry)
                        continue
                    }
                    groups[hit.group] = applying(shipped: entry,
                                                 to: groups[hit.group],
                                                 at: hit.entry,
                                                 shippedMatcher: shippedGroup["matcher"])
                }

                // Nothing of ours registered yet (fresh install, or a
                // subcommand added in this release): append it, carrying the
                // shipped matcher.
                if !unmatched.isEmpty {
                    var group = shippedGroup
                    group["hooks"] = unmatched
                    groups.append(group)
                }
            }

            result[event] = groups
        }

        return result
    }

    /// `"~/.claude/bin/memory-hooks advise 2>/dev/null"` → `"memory-hooks advise"`.
    ///
    /// Path-independent by design: an upgrade that moves the install
    /// directory must still recognise the registration it wrote last time,
    /// otherwise it appends a duplicate and the user runs both.
    public static func hookIdentity(command: String) -> String? {
        let tokens = command.split(separator: " ").map(String.init)
        for (index, token) in tokens.enumerated() where index + 1 < tokens.count {
            guard (token as NSString).lastPathComponent == hookBinaryName else { continue }
            let subcommand = tokens[index + 1]
            // A flag, not a subcommand — we can't identify the registration.
            guard !subcommand.hasPrefix("-") else { return nil }
            return "\(hookBinaryName) \(subcommand)"
        }
        return nil
    }

    /// Does `command` register `identity`?
    ///
    /// Substring match, so wrappers (`sh -c "… memory-hooks advise …"`) still
    /// resolve — but bounded on both sides, so `memory-hooks on-start` never
    /// claims `memory-hooks on-start-extra`, nor `my-memory-hooks on-start`.
    public static func command(_ command: String, registers identity: String) -> Bool {
        // A path separator or a shell quote is as good a boundary as a space.
        let quotes: Set<Character> = ["\"", "'", "`"]
        let terminators: Set<Character> = [";", "&", "|", ")"]

        var searchStart = command.startIndex
        while let range = command.range(of: identity, range: searchStart..<command.endIndex) {
            let leadingOK: Bool
            if range.lowerBound == command.startIndex {
                leadingOK = true
            } else {
                let previous = command[command.index(before: range.lowerBound)]
                leadingOK = previous == "/" || previous.isWhitespace || quotes.contains(previous)
            }
            let trailingOK: Bool
            if range.upperBound == command.endIndex {
                trailingOK = true
            } else {
                let next = command[range.upperBound]
                trailingOK = next.isWhitespace || quotes.contains(next) || terminators.contains(next)
            }
            if leadingOK && trailingOK { return true }
            searchStart = range.upperBound
        }
        return false
    }

    // MARK: - Internals

    private struct Hit { let group: Int; let entry: Int }

    private static func locate(identity: String, in groups: [[String: Any]]) -> Hit? {
        for (groupIndex, group) in groups.enumerated() {
            guard let entries = asEntries(group["hooks"]) else { continue }
            for (entryIndex, entry) in entries.enumerated() {
                guard let existing = entry["command"] as? String,
                      command(existing, registers: identity) else { continue }
                return Hit(group: groupIndex, entry: entryIndex)
            }
        }
        return nil
    }

    /// Update one existing hook entry in place from the shipped one.
    private static func applying(shipped: [String: Any],
                                 to group: [[String: Any]].Element,
                                 at entryIndex: Int,
                                 shippedMatcher: Any?) -> [String: Any] {
        var group = group
        var entries = asEntries(group["hooks"]) ?? []
        guard entries.indices.contains(entryIndex) else { return group }
        var entry = entries[entryIndex]

        // The command always takes the shipped value — a new install path or
        // a changed flag has to land, that is the whole point of re-running
        // the installer.
        if let command = shipped["command"] { entry["command"] = command }
        if entry["type"] == nil, let type = shipped["type"] { entry["type"] = type }

        // The timeout is the user's to raise. Keep the larger of the two:
        // their deliberate increase survives, and a shipped increase (like
        // UserPromptSubmit going 5 → 60) still reaches everyone who never
        // touched it.
        switch (intValue(entry["timeout"]), intValue(shipped["timeout"])) {
        case let (existing?, shippedTimeout?): entry["timeout"] = max(existing, shippedTimeout)
        case let (existing?, nil): entry["timeout"] = existing
        case let (nil, shippedTimeout?): entry["timeout"] = shippedTimeout
        case (nil, nil): break
        }

        entries[entryIndex] = entry
        group["hooks"] = entries

        // The matcher belongs to the GROUP, and the group may also hold the
        // user's own hooks — retarget it only when ours is the sole occupant.
        if entries.count == 1, let shippedMatcher {
            group["matcher"] = shippedMatcher
        }
        return group
    }

    private static func asGroups(_ value: Any?) -> [[String: Any]]? {
        value as? [[String: Any]]
    }

    private static func asEntries(_ value: Any?) -> [[String: Any]]? {
        value as? [[String: Any]]
    }

    /// JSON numbers arrive as `NSNumber`; a hand-edited settings.json can
    /// even carry `"timeout": "60"`.
    private static func intValue(_ value: Any?) -> Int? {
        switch value {
        case let int as Int: return int
        case let double as Double: return Int(double)
        case let number as NSNumber: return number.intValue
        case let string as String: return Int(string)
        default: return nil
        }
    }
}
