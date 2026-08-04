import Foundation

/// The two placement rules for group galaxies (increment 6).
///
/// They live here, away from the renderer, because both must be
/// DETERMINISTIC: they decide where a node is drawn, and a rule that
/// depends on dictionary iteration order moves memories between galaxies on
/// every relaunch. Pure functions over plain values so they can be tested
/// without a Metal device.
public enum GroupHierarchy {

    /// Hierarchy level per attached group: height above the deepest attached
    /// descendant, so a leaf team is 1 and its org is 2.
    ///
    /// Height rather than depth because the layout stacks levels on Y with
    /// personal/synced at 0 — a team belongs directly above the user's own
    /// graph, with its parent org above that, regardless of how tall the
    /// server-side tree happens to be.
    ///
    /// Only ATTACHED groups count. An ancestor the user isn't a member of
    /// (or whose spoke hasn't synced) must not push its child into an empty
    /// band with nothing beneath it.
    ///
    /// - Parameters:
    ///   - present: groups with a live spoke on this machine.
    ///   - parentOf: parent link per group, from the daemon's directory.
    ///     Entries for absent groups are ignored.
    public static func levels(present: Set<UUID>,
                              parentOf: [UUID: UUID?]) -> [UUID: Int] {
        // Children restricted to the attached set, built once: the recursive
        // form rescans `present` per node, which is O(n²) on a wide org.
        var children: [UUID: [UUID]] = [:]
        for id in present {
            guard let parent = parentOf[id] ?? nil, present.contains(parent) else { continue }
            children[parent, default: []].append(id)
        }

        var memo: [UUID: Int] = [:]
        func height(_ id: UUID, _ onPath: Set<UUID>) -> Int {
            if let cached = memo[id] { return cached }
            // Cycle guard: a reparent race can produce a loop in the
            // directory, and this must not recurse forever.
            guard !onPath.contains(id) else { return 1 }
            let kids = children[id] ?? []
            let value = kids.isEmpty
                ? 1
                : 1 + (kids.map { height($0, onPath.union([id])) }.max() ?? 0)
            memo[id] = value
            return value
        }
        var result: [UUID: Int] = [:]
        for id in present { result[id] = height(id, []) }
        return result
    }

    /// Which attached group galaxy renders a project exposed to several of
    /// them.
    ///
    /// The most specific group wins — the leaf team rather than the org that
    /// contains it — because that is the context the memory was shared into.
    /// Ties break on sorted group id so the choice is stable across launches
    /// and across machines.
    ///
    /// Returns nil when the project is exposed to no ATTACHED group, meaning
    /// the caller should fall back to synced/personal.
    public static func owningGroup(exposedGroupIds: Set<String>,
                                   attachedLevels: [UUID: Int]) -> UUID? {
        let candidates = exposedGroupIds
            .compactMap(UUID.init(uuidString:))
            .filter { attachedLevels[$0] != nil }
        guard !candidates.isEmpty else { return nil }
        return candidates.min { lhs, rhs in
            let l = attachedLevels[lhs] ?? .max
            let r = attachedLevels[rhs] ?? .max
            return l == r ? lhs.uuidString < rhs.uuidString : l < r
        }
    }
}
