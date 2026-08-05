import Testing
import EngramKit
import EngramModels
import Lattice
import MCP
import Foundation

// ============================================================================
// Graph-of-graphs recall — the battle-test for "groups are a graph of
// graphs". A member of several graphs (org tree + concept graph) recalls
// ONCE and sees every graph's contribution, attributed ([by:]) and
// provenanced ([via:]), with no duplicates and no cross-graph bleed.
//
// Unlike GroupReadLatticeTests (deliberately directory-less), this suite
// writes a groups.json fixture: names for [via:], members for [by:], and a
// selfUserId so own rows render clean. GroupDirectory.fileURL is
// process-global, hence `.serialized` + save/restore around every test.
// ============================================================================

private struct TopologySpoke {
    let groupId: UUID
    let name: String
    let parentId: UUID?
    let path: String
    let lattice: Lattice
}

private struct TopologyContext {
    let tools: MemoryTools
    let local: Lattice
    let selfUserId: UUID
    let spokes: [String: TopologySpoke]  // by group name
}

/// Builds N spokes (one per (name, parentId) entry), a groups.json fixture
/// naming them (plus any spokeless groups like the org root), and
/// MemoryTools over the lot. Caller restores `GroupDirectory.fileURL`.
private func makeTopologyTools(
    spoked: [(name: String, parentId: UUID?)],
    spokeless: [(id: UUID, name: String, parentId: UUID?)] = [],
    members: [UUID: String] = [:],
    selfUserId: UUID
) async throws -> TopologyContext {
    let dir = FileManager.default.temporaryDirectory
        .appending(path: "engram-topology-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

    let local = try Lattice(Memory.self, Edge.self, Checkpoint.self, HookState.self,
                            SyncConfig.self,
                            configuration: .init(fileURL: dir.appending(path: "memory.sqlite")))

    var spokes: [String: TopologySpoke] = [:]
    var refs: [MemoryTools.GroupSpokeRef] = []
    var groupInfos: [GroupDirectory.GroupInfo] = []
    for entry in spoked {
        let gid = UUID()
        let path = dir.appending(path: "group-\(gid.uuidString).sqlite").path
        let lattice = try Lattice(Memory.self, Edge.self, GroupProjectMap.self,
                                  configuration: .init(fileURL: URL(fileURLWithPath: path)))
        spokes[entry.name] = TopologySpoke(groupId: gid, name: entry.name,
                                           parentId: entry.parentId,
                                           path: path, lattice: lattice)
        refs.append(.init(groupId: gid, path: path, ref: lattice.sendableReference))
        groupInfos.append(.init(id: gid, name: entry.name, parentId: entry.parentId,
                                myRole: "member", root: entry.parentId == nil))
    }
    for g in spokeless {
        groupInfos.append(.init(id: g.id, name: g.name, parentId: g.parentId,
                                myRole: "member", root: g.parentId == nil))
    }

    let file = GroupDirectory.GroupsFile(
        selfUserId: selfUserId,
        updatedAt: Date(),
        groups: groupInfos,
        members: members.reduce(into: [:]) { $0[$1.key.uuidString] = $1.value })
    let fileURL = dir.appending(path: "groups.json")
    let encoder = JSONEncoder()
    encoder.dateEncodingStrategy = .iso8601  // GroupDirectory decodes iso8601
    try encoder.encode(file).write(to: fileURL)
    GroupDirectory.fileURL = fileURL

    let embedder = sharedEmbedder
    if await !embedder.isLoaded { await embedder.load() }
    let tools = MemoryTools(localRef: local.sendableReference, syncedRef: nil,
                            groupRefs: refs, embedder: embedder)
    return TopologyContext(tools: tools, local: local,
                           selfUserId: selfUserId, spokes: spokes)
}

@discardableResult
private func addMemory(_ content: String, project: String, author: UUID?,
                       to lattice: Lattice) async throws -> UUID {
    let mem = Memory(
        content: content, topic: "general", project: project,
        embedding: Vector<Float>(try await sharedEmbedder.embed(text: content)!),
        authorUserId: author)
    try lattice.add(mem)
    return mem.globalId!
}

@Suite(.serialized)
struct GroupTopologyRecallTests {

    /// One recall unions the org's teams AND the concept graph, with every
    /// graph attributed and provenanced — the founder's canary/mobile shape,
    /// expressed as user-level multi-membership.
    @Test func recall_unionsCanaryTeamsAndConceptGraphs() async throws {
        let savedURL = GroupDirectory.fileURL
        defer { GroupDirectory.fileURL = savedURL }

        let carol = UUID(), alice = UUID(), bob = UUID(), dave = UUID()
        let canaryId = UUID()
        let ctx = try await makeTopologyTools(
            spoked: [("team-ios", canaryId), ("team-web", canaryId), ("mobile", nil)],
            spokeless: [(canaryId, "canary", nil)],
            members: [alice: "Alice", bob: "Bob", dave: "Dave", carol: "Carol"],
            selfUserId: carol)

        try await addMemory("release checklist: carol's local step is signing the build",
                            project: "demo", author: carol, to: ctx.local)
        try await addMemory("release checklist: alice says ios needs a TestFlight pass first",
                            project: "demo", author: alice,
                            to: ctx.spokes["team-ios"]!.lattice)
        try await addMemory("release checklist: bob says web deploys behind the flag",
                            project: "demo", author: bob,
                            to: ctx.spokes["team-web"]!.lattice)
        try await addMemory("release checklist: dave says mobile ships both stores together",
                            project: "demo", author: dave,
                            to: ctx.spokes["mobile"]!.lattice)

        let out = text(from: try await ctx.tools.handle(CallTool.Parameters(
            name: "recall",
            arguments: ["query": .string("release checklist"), "project": .string("demo")])))

        // Every graph contributes to ONE response.
        #expect(out.contains("signing the build"))
        #expect(out.contains("TestFlight pass"), "team-ios graph missing")
        #expect(out.contains("behind the flag"), "team-web graph missing")
        #expect(out.contains("both stores together"), "concept graph missing")

        // Attribution resolves through the member directory…
        #expect(out.contains("[by:Alice]"))
        #expect(out.contains("[by:Bob]"))
        #expect(out.contains("[by:Dave]"))
        // …and provenance names the graph each row came from.
        #expect(out.contains("[via:team-ios]"))
        #expect(out.contains("[via:team-web]"))
        #expect(out.contains("[via:mobile]"))

        // Own local row: no badge, no via.
        let ownLine = out.split(separator: "\n").first { $0.contains("signing the build") }
        #expect(ownLine != nil)
        #expect(ownLine?.contains("[by:") == false, "own row must not carry a by-badge")
        #expect(ownLine?.contains("[via:") == false, "hub row must not carry a via-marker")

        // No memory rendered twice.
        for needle in ["signing the build", "TestFlight pass", "behind the flag",
                       "both stores together"] {
            #expect(out.components(separatedBy: needle).count == 2,
                    "'\(needle)' rendered more than once")
        }
    }

    /// decision 13 across TWO graphs: my mapping rows in both spokes plus a
    /// teammate's differently-named mapping in one of them chain into one
    /// resolved set, so recall/stats treat every alias as the same project.
    @Test func resolveQueryProjects_chainsAcrossTwoSpokes() async throws {
        let savedURL = GroupDirectory.fileURL
        defer { GroupDirectory.fileURL = savedURL }

        let carol = UUID(), teammate = UUID()
        let ctx = try await makeTopologyTools(
            spoked: [("team-ios", nil), ("mobile", nil)],
            members: [teammate: "Terry"],
            selfUserId: carol)

        // My rows: ios-app → Mobile in BOTH graphs I exposed it to.
        try ctx.spokes["team-ios"]!.lattice.add(GroupProjectMap(
            memberUserId: carol, localProject: "ios-app", groupProject: "Mobile"))
        try ctx.spokes["mobile"]!.lattice.add(GroupProjectMap(
            memberUserId: carol, localProject: "ios-app", groupProject: "Mobile"))
        // Teammate's local name for the SAME canonical project, in one graph.
        try ctx.spokes["mobile"]!.lattice.add(GroupProjectMap(
            memberUserId: teammate, localProject: "web-app", groupProject: "Mobile"))

        let resolved = await ctx.tools.resolveQueryProjects("ios-app")
        #expect(resolved.contains("ios-app"))
        #expect(resolved.contains("web-app"),
                "teammate's alias for the same canonical project must resolve")
    }

    /// KNN candidate identity across spokes (latticecore 1.2.6): colliding
    /// rowids in different graphs must NOT let one graph's candidate drag an
    /// unrelated row out of another graph with a borrowed distance. Red
    /// against latticecore ≤1.2.5 (the C++ red-first proof lives in
    /// AttachKnnCrossArmTests); this is the client-level guard that outlives
    /// core internals.
    @Test func recall_neverBorrowsDistancesAcrossSpokesWithCollidingRowids() async throws {
        let savedURL = GroupDirectory.fileURL
        defer { GroupDirectory.fileURL = savedURL }

        let me = UUID(), a = UUID(), b = UUID()
        let ctx = try await makeTopologyTools(
            spoked: [("graphics", nil), ("baking", nil)],
            members: [a: "Ada", b: "Blaise"],
            selfUserId: me)

        // Same rowids (1..3) in both spokes, disjoint topics.
        for text in ["quaternion camera interpolation eases the orbit",
                     "quaternion slerp keeps the camera path smooth",
                     "camera interpolation needs renormalized quaternions"] {
            try await addMemory(text, project: "demo", author: a,
                                to: ctx.spokes["graphics"]!.lattice)
        }
        for text in ["sourdough hydration ratios around seventy percent",
                     "sourdough starter needs consistent feeding",
                     "higher hydration makes an open sourdough crumb"] {
            try await addMemory(text, project: "demo", author: b,
                                to: ctx.spokes["baking"]!.lattice)
        }

        let out = text(from: try await ctx.tools.handle(CallTool.Parameters(
            name: "recall",
            arguments: ["query": .string("quaternion camera interpolation"),
                        "limit": .int(3)])))

        #expect(out.contains("quaternion"))
        #expect(!out.contains("sourdough"),
                "unrelated graph's rows surfaced — cross-arm rowid borrow")
    }
}
