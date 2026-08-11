import Testing
import EngramKit
import EngramModels
import Lattice
import Foundation

// MARK: - findMemoryClusters statement budget + cluster-shape pin
//
// `findMemoryClusters` used to run ONE vec0 KNN per memory — and because the
// bridge inlines the query vector as a hex blob literal, each of those was a
// multi-KB statement that MATCHed the WHOLE vector table before post-filtering
// to the project. The visualizer calls this once per project on every graph
// load, so a 28K-memory database paid ~28K such queries (plus one live
// `SELECT col WHERE id=?` per field per candidate row) — the "projects took
// five minutes to load" report.
//
// These tests pin BOTH halves of the fix:
//   1. the grouping the algorithm produces (must not change), and
//   2. the statement count it costs (must stay O(1) in the memory count).
//
// The budget test measures with the THREAD-LOCAL statement counter —
// `findMemoryClusters` is synchronous, so every statement it issues is on the
// measuring thread and no parallel suite can land in the window. (Recall's
// equivalent test can't do that: it's async and thread-hops, which is why CI
// gives it a quiet serial invocation.) The suite is `.serialized` so its own
// three fixtures don't build concurrently.

@Suite("Clustering perf regression", .serialized)
struct ClusteringPerfRegressionTests {

    // MARK: - Fixture

    /// Deterministic LCG — the fixture pins the CLUSTERING algorithm, so it
    /// builds its own vectors rather than paying for (and depending on) the
    /// embedding model.
    private struct Rand {
        var state: UInt64
        mutating func next() -> Double {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Double((state >> 11) & 0x1F_FFFF_FFFF_FFFF) / Double(1 << 53)
        }
        /// Box-Muller normal deviate.
        mutating func normal() -> Float {
            let u1 = Swift.max(next(), 1e-12)
            let u2 = next()
            return Float((-2 * Foundation.log(u1)).squareRoot() * Foundation.cos(2 * .pi * u2))
        }
    }

    private static func unitVector(dims: Int, rng: inout Rand) -> [Float] {
        var v = (0..<dims).map { _ in rng.normal() }
        let norm = v.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
        if norm > 0 { for i in v.indices { v[i] /= norm } }
        return v
    }

    /// `groups` well-separated tight clusters of `perGroup` memories each.
    /// Group members share a five-token vocabulary (Jaccard ≈ 0.7 inside a
    /// group, 0 across groups) and sit within L2 ≈ 0.05 of their group
    /// centroid; centroids are near-orthogonal (L2 ≈ 1.41), well outside the
    /// 0.775 default distance threshold.
    private static func makeFixture(
        groups: Int, perGroup: Int, project: String = "ClusterPerf",
        dims: Int = 384, seed: UInt64 = 0xC1A5732,
        into lattice: Lattice
    ) throws -> [[UUID]] {
        var rng = Rand(state: seed)
        var expected: [[UUID]] = []
        for g in 0..<groups {
            let centroid = unitVector(dims: dims, rng: &rng)
            var ids: [UUID] = []
            for m in 0..<perGroup {
                var v = centroid
                for i in v.indices { v[i] += 0.002 * rng.normal() }
                let norm = v.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
                for i in v.indices { v[i] /= norm }
                let mem = Memory(
                    content: "grp\(g)alpha grp\(g)beta grp\(g)gamma grp\(g)delta grp\(g)epsilon item\(g)x\(m)",
                    topic: "topic\(g)",
                    project: project,
                    embedding: Vector<Float>(v))
                try lattice.add(mem)
                if let gid = mem.globalId { ids.append(gid) }
            }
            expected.append(ids)
        }
        return expected
    }

    private static func makeLattice() throws -> Lattice {
        let path = FileManager.default.temporaryDirectory
            .appending(path: "cluster-perf-\(UUID().uuidString).sqlite")
        return try Lattice(
            Memory.self, Edge.self, Checkpoint.self, HookState.self, SyncConfig.self,
            configuration: .init(fileURL: path))
    }

    /// One 200-memory fixture, shared by the grouping pin and the budget
    /// measurement. Building it twice doubled this suite's write traffic,
    /// and those writes are noise inside `recall_statementBudget`'s
    /// process-global measurement window when the whole suite runs.
    nonisolated(unsafe) private static let sharedFixture: (lattice: Lattice, expected: [[UUID]]) = {
        // A fixture that cannot be built is a hard failure either way.
        let lattice = try! makeLattice()
        let expected = try! makeFixture(groups: 10, perGroup: 20, into: lattice)
        return (lattice, expected)
    }()

    /// Normalize a cluster list to a comparable canonical form: each cluster
    /// as a sorted id set, the list sorted by its smallest id. Cluster ORDER
    /// and within-cluster order depend on `Set<UUID>` iteration, which is
    /// seeded per process — only the GROUPING is a stable contract.
    private static func canonical(_ clusters: [[UUID]]) -> [[String]] {
        clusters
            .map { $0.map(\.uuidString).sorted() }
            .sorted { ($0.first ?? "") < ($1.first ?? "") }
    }

    // MARK: - 1. Grouping pin

    /// The grouping must survive the rewrite: 10 tight, well-separated groups
    /// of 20 come back as exactly those 10 groups.
    @Test func clusters_groupingIsStable() throws {
        let (lattice, expected) = Self.sharedFixture

        let result = findMemoryClusters(
            in: lattice, project: "ClusterPerf",
            minClusterSize: 2, maxClusters: 10, neighborLimit: 20)

        #expect(Self.canonical(result.clusters) == Self.canonical(expected),
                "clustering grouping changed for the pinned fixture")

        // Distances must still be cached for every clustered pair the
        // handler formats an average similarity from.
        for cluster in result.clusters {
            let seedDistances = cluster.compactMap { result.distances[$0] }
            #expect(!seedDistances.isEmpty, "distance cache empty for a cluster")
        }
    }

    /// Group membership must be decided among the QUERIED PROJECT's memories.
    /// A second project full of near-identical vectors must not evict a
    /// memory's true same-project neighbours from its candidate list.
    @Test func clusters_neighborsAreProjectScoped() throws {
        let lattice = try Self.makeLattice()
        let expected = try Self.makeFixture(groups: 2, perGroup: 20, into: lattice)
        // Decoys: 60 memories in ANOTHER project sitting on top of group 0's
        // centroid. Under a global-then-post-filter KNN these crowd out the
        // real neighbours; under a project-scoped one they are invisible.
        var rng = Rand(state: 0xDEC0)
        // Sit the decoys on the group CENTROID (the mean of its members), so
        // every member is closer to a decoy than to its own siblings — the
        // exact shape a global top-k with post-filtering gets wrong.
        var decoyCentroid = [Float](repeating: 0, count: 384)
        for gid in expected[0] {
            let e = try #require(
                lattice.objects(Memory.self).where { $0.globalId == gid }.first
            ).embedding.elements
            for i in decoyCentroid.indices { decoyCentroid[i] += e[i] }
        }
        for i in decoyCentroid.indices { decoyCentroid[i] /= Float(expected[0].count) }
        for m in 0..<60 {
            var v = decoyCentroid
            for i in v.indices { v[i] += 0.0002 * rng.normal() }
            let norm = v.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
            for i in v.indices { v[i] /= norm }
            try lattice.add(Memory(
                content: "decoy\(m) noise filler text",
                topic: "decoy", project: "OtherProject",
                embedding: Vector<Float>(v)))
        }

        let result = findMemoryClusters(
            in: lattice, project: "ClusterPerf",
            minClusterSize: 2, maxClusters: 10, neighborLimit: 20)

        #expect(Self.canonical(result.clusters) == Self.canonical(expected),
                "a foreign project's memories changed this project's clusters")
    }

    // MARK: - 2. Statement budget

    /// Clustering must cost ONE snapshot of the candidate rows and nothing
    /// per memory beyond it. Measured on this fixture: 20,802 statements
    /// before the rewrite (104 per memory — a vec0 KNN plus live per-field
    /// reads for each), 601 after (3 per memory, which is Lattice's
    /// `materializedSnapshot()` row-fill floor: fill + pk prime + row-cache
    /// refetch). The neighbour pass itself now issues zero SQL.
    @Test func clusters_statementBudget() throws {
        // Touch the shared fixture BEFORE the measurement window — the lazy
        // build would otherwise land inside it.
        let lattice = Self.sharedFixture.lattice

        // THREAD-LOCAL counter, not the process-global one. `recall_
        // statementBudget` has to use the global counter because recall is
        // async and thread-hops, which is why CI gives it a quiet serial
        // invocation; `findMemoryClusters` is fully synchronous, so its
        // statements are all on this thread and no other suite's traffic can
        // land in the window. Min-of-3 anyway, for anything the test harness
        // itself might do on this thread.
        var used = UInt64.max
        var clusterCount = 0
        for _ in 0..<3 {
            let base = Lattice.threadSQLStatementCount
            let result = findMemoryClusters(
                in: lattice, project: "ClusterPerf",
                minClusterSize: 2, maxClusters: 10, neighborLimit: 20)
            used = min(used, Lattice.threadSQLStatementCount - base)
            clusterCount = result.clusters.count
        }
        print("[cluster-perf] 200 memories → \(used) SQL statements, \(clusterCount) clusters")

        #expect(clusterCount == 10)
        // 4 per memory + slack: the snapshot floor with headroom for a
        // Lattice-side row-fill change, and still 25x under the
        // one-KNN-per-memory failure mode this test guards.
        #expect(used <= 4 * 200 + 50,
                "findMemoryClusters issued \(used) SQL statements for 200 memories — statement-budget regression")
    }
}
