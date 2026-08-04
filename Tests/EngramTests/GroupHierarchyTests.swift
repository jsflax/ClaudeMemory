import Testing
import EngramKit
import Foundation

// ============================================================================
// Increment 6 — where a group galaxy is DRAWN, and which one owns a project
// shared with several groups. Both rules must be deterministic: they decide
// node placement, and a rule that depends on iteration order would shuffle
// memories between galaxies on every relaunch.
// ============================================================================

// MARK: - Hierarchy level (height above the deepest attached descendant)

@Test func soloGroupSitsOneLevelAbovePersonal() {
    let team = UUID()
    let levels = GroupHierarchy.levels(present: [team], parentOf: [team: nil])
    // Personal and synced are level 0; a team belongs directly above them.
    #expect(levels[team] == 1)
}

@Test func orgSitsAboveItsTeam() {
    let org = UUID(), team = UUID()
    let levels = GroupHierarchy.levels(
        present: [org, team],
        parentOf: [org: nil, team: org])
    #expect(levels[team] == 1)
    #expect(levels[org] == 2)
}

@Test func levelsCountOnlyAttachedDescendants() {
    // The user belongs to the ORG but not to the team beneath it, so no
    // spoke exists for the team. The org must render at level 1 — pushing it
    // to 2 would leave an empty band with nothing underneath.
    let org = UUID(), unattachedTeam = UUID()
    let levels = GroupHierarchy.levels(
        present: [org],
        parentOf: [org: nil, unattachedTeam: org])
    #expect(levels[org] == 1)
    #expect(levels[unattachedTeam] == nil)
}

@Test func deepChainStacksMonotonically() {
    let root = UUID(), mid = UUID(), leaf = UUID()
    let levels = GroupHierarchy.levels(
        present: [root, mid, leaf],
        parentOf: [root: nil, mid: root, leaf: mid])
    #expect(levels[leaf] == 1)
    #expect(levels[mid] == 2)
    #expect(levels[root] == 3)
}

@Test func widestBranchDeterminesParentLevel() {
    // Two siblings of different depth: the parent must clear the taller one.
    let org = UUID(), shallow = UUID(), deep = UUID(), deeper = UUID()
    let levels = GroupHierarchy.levels(
        present: [org, shallow, deep, deeper],
        parentOf: [org: nil, shallow: org, deep: org, deeper: deep])
    #expect(levels[shallow] == 1)
    #expect(levels[deeper] == 1)
    #expect(levels[deep] == 2)
    #expect(levels[org] == 3)
}

@Test func parentCycleTerminatesInsteadOfHanging() {
    // A reparent race can leave a loop in the directory. Rendering must
    // degrade, not spin forever.
    let a = UUID(), b = UUID()
    let levels = GroupHierarchy.levels(present: [a, b], parentOf: [a: b, b: a])
    #expect(levels[a] != nil)
    #expect(levels[b] != nil)
}

// MARK: - Which group owns a multi-exposed project

@Test func mostSpecificGroupOwnsASharedProject() {
    let org = UUID(), team = UUID()
    let levels = [org: 2, team: 1]
    // Exposed to BOTH the org and the team inside it: the team wins, because
    // that is the context the memory was shared into.
    let owner = GroupHierarchy.owningGroup(
        exposedGroupIds: [org.uuidString, team.uuidString],
        attachedLevels: levels)
    #expect(owner == team)
}

@Test func tiesBreakOnSortedIdSoPlacementIsStable() {
    // Two unrelated teams at the same level — the choice must be the same
    // every launch, not "whichever the dictionary yielded first".
    let a = UUID(uuidString: "00000000-0000-0000-0000-0000000000AA")!
    let b = UUID(uuidString: "00000000-0000-0000-0000-0000000000BB")!
    let levels = [a: 1, b: 1]
    let exposed: Set<String> = [a.uuidString, b.uuidString]
    let first = GroupHierarchy.owningGroup(exposedGroupIds: exposed, attachedLevels: levels)
    #expect(first == a)
    // Repeat with the input built in the other order — same answer.
    for _ in 0..<20 {
        #expect(GroupHierarchy.owningGroup(
            exposedGroupIds: Set([b.uuidString, a.uuidString]),
            attachedLevels: levels) == a)
    }
}

@Test func unattachedGroupNeverOwnsAProject() {
    // Exposed to a group whose spoke isn't on this machine (not yet synced,
    // or revoked): nil, so the caller falls back to synced/personal rather
    // than routing nodes into a galaxy that doesn't exist.
    let absent = UUID()
    let owner = GroupHierarchy.owningGroup(
        exposedGroupIds: [absent.uuidString], attachedLevels: [:])
    #expect(owner == nil)
}

@Test func unexposedProjectHasNoGroupOwner() {
    #expect(GroupHierarchy.owningGroup(exposedGroupIds: [], attachedLevels: [UUID(): 1]) == nil)
}

@Test func malformedExposureEntryIsIgnoredRatherThanCrashing() {
    // exposedGroups is a Set<String> written by several code paths; a
    // non-UUID entry must not take down the renderer.
    let team = UUID()
    let owner = GroupHierarchy.owningGroup(
        exposedGroupIds: ["not-a-uuid", team.uuidString],
        attachedLevels: [team: 1])
    #expect(owner == team)
}
