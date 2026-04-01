import XCTest
import EngramKit
import Lattice

/// Focused insert stress test.
/// Seeds a realistic DB, teleports in, inserts memories in bursts, and dumps timing CSVs.
/// Requires the Engram-UITesting scheme (ENGRAM_INSTRUMENTATION flag).
@MainActor
final class InsertStressTests: XCTestCase {
    let app = XCUIApplication()

    private var localDbPath: String!
    private var syncedDbPath: String!
    private var localIds: [UUID] = []
    private let testUUID = UUID().uuidString

    private let csvPaths = [
        "/tmp/metal-frame-timing.csv",
        "/tmp/draw-timing.csv",
        "/tmp/flush-timing.csv",
        "/tmp/pack-timing.csv",
        "/tmp/merge-timing.csv",
        "/tmp/atlas-timing.log",
        "/tmp/label-diag.csv",
        "/tmp/audio-timing.csv",
    ]

    override func setUpWithError() throws {
        continueAfterFailure = false

        let tmpDir = NSTemporaryDirectory() + "insert-stress-\(testUUID)/"
        try FileManager.default.createDirectory(atPath: tmpDir, withIntermediateDirectories: true)
        localDbPath = tmpDir + "memory.sqlite"
        let syncDir = tmpDir + "sync/"
        try FileManager.default.createDirectory(atPath: syncDir, withIntermediateDirectories: true)
        syncedDbPath = syncDir + "memory-synced.sqlite"

        for path in csvPaths { try? FileManager.default.removeItem(atPath: path) }

        // Seed ~2185 local nodes
        localIds = seedMultiProjectDatabase(
            at: localDbPath,
            projects: [
                ("Engram", 1000), ("Lattice", 500), ("ClaudeMemory", 200),
                ("engram-server", 120), ("sidescroller", 100), ("global", 200),
                ("LatticeCore", 40), ("SwiftLM", 25),
            ],
            staleAge: 7200,
            withSyncConfig: ["Engram", "Lattice", "ClaudeMemory", "engram-server", "LatticeCore", "SwiftLM"]
        )

        // Seed ~1885 synced nodes
        _ = seedMultiProjectDatabase(
            at: syncedDbPath,
            projects: [
                ("Engram", 1000), ("Lattice", 500), ("ClaudeMemory", 200),
                ("engram-server", 120), ("LatticeCore", 40), ("SwiftLM", 25),
            ],
            staleAge: 10800
        )

        app.launchEnvironment["CLAUDE_MEMORY_DB"] = localDbPath
        app.launchEnvironment["ENGRAM_FORCE_SOUND"] = "1"
        app.launchEnvironment["ENGRAM_TEST_NO_NOTIFY"] = "1"
        app.launchEnvironment["ENGRAM_TEST_INSERT_DELAY"] = "9999"  // disable timed insert
        app.launchArguments += ["-subscription_status", "active"]
        app.launchArguments += ["-subscription_tier", "premium"]
    }

    override func tearDownWithError() throws {
        let testDir = (localDbPath! as NSString).deletingLastPathComponent
        try? FileManager.default.removeItem(atPath: testDir)
    }

    // MARK: - Test

    func testInsertStress() throws {
        app.launch()

        let window = app.windows.firstMatch
        XCTAssertTrue(window.waitForExistence(timeout: 15), "App window not found")
        sleep(12) // let simulation settle

        // Defocus search bar
        app.typeKey(.escape, modifierFlags: [])
        usleep(200_000)

        // Record baseline frame count
        let baselineFrames = frameCount()

        // ── Teleport into a cluster ──
        window.typeKey("t", modifierFlags: [])
        sleep(2)

        // ── Idle baseline (3s) ──
        sleep(3)
        let preInsertFrames = frameCount()
        takeScreenshot(name: "insert-stress-00-pre")

        // Use non-synced projects so nodeFilter accepts them into the personal galaxy.
        // "sidescroller" and "global" are NOT in the sync config.

        // ── Burst 1: Insert 10 memories ──
        let burst1 = insertMemories(at: localDbPath, project: "sidescroller", count: 10)
        sleep(3)
        let postBurst1Frames = frameCount()
        takeScreenshot(name: "insert-stress-01-burst10")

        // ── Burst 2: Insert 25 memories ──
        let burst2 = insertMemories(at: localDbPath, project: "global", count: 25)
        sleep(4)
        let postBurst2Frames = frameCount()
        takeScreenshot(name: "insert-stress-02-burst25")

        // ── Burst 3: Insert 50 memories while moving camera ──
        let burst3 = insertMemories(at: localDbPath, project: "sidescroller", count: 50)
        holdKey(VK.w, duration: 1.0)
        holdKey(VK.j, duration: 1.0)
        sleep(3)
        let postBurst3Frames = frameCount()
        takeScreenshot(name: "insert-stress-03-burst50-moving")

        // ── Burst 4: Rapid small inserts (simulates real usage) ──
        var rapidIds: [UUID] = []
        for _ in 0..<10 {
            rapidIds.append(contentsOf: insertMemories(at: localDbPath, project: "global", count: 3))
            usleep(300_000) // 300ms between bursts
        }
        sleep(3)
        let postBurst4Frames = frameCount()
        takeScreenshot(name: "insert-stress-04-rapid-small")

        // ── Cleanup: delete all inserted memories ──
        let allInserted = burst1 + burst2 + burst3 + rapidIds
        deleteMemories(at: localDbPath, globalIds: allInserted)
        sleep(2)
        takeScreenshot(name: "insert-stress-05-cleanup")

        // ═══════════════════════════════════════
        //   ANALYSIS
        // ═══════════════════════════════════════

        print("\n" + String(repeating: "═", count: 80))
        print("  INSERT STRESS TEST RESULTS")
        print(String(repeating: "═", count: 80))

        print("\nFrame counts:")
        print("  Baseline→PreInsert: \(preInsertFrames - baselineFrames) frames in ~5s")
        print("  Burst 1 (10 nodes):  \(postBurst1Frames - preInsertFrames) frames in ~3s")
        print("  Burst 2 (25 nodes):  \(postBurst2Frames - postBurst1Frames) frames in ~4s")
        print("  Burst 3 (50 nodes):  \(postBurst3Frames - postBurst2Frames) frames in ~5s")
        print("  Burst 4 (30 rapid):  \(postBurst4Frames - postBurst3Frames) frames in ~6s")

        // Analyze metal frame timing
        analyzeFrames()

        // Analyze flush timing
        analyzeFlushes()

        print("\n" + String(repeating: "═", count: 80))
    }

    // MARK: - Analysis

    private func frameCount() -> Int {
        let path = "/tmp/metal-frame-timing.csv"
        guard let data = FileManager.default.contents(atPath: path),
              let csv = String(data: data, encoding: .utf8) else { return 0 }
        return csv.components(separatedBy: "\n").count - 2
    }

    private func analyzeFrames() {
        let path = "/tmp/metal-frame-timing.csv"
        guard let data = FileManager.default.contents(atPath: path),
              let csv = String(data: data, encoding: .utf8) else {
            print("\n[No metal-frame-timing.csv]")
            return
        }

        // CSV: frame,dt_ms,wall_dt_ms,total_ms,sim_ms,mascot_ms,nodes_ms,edges_ms,neb_ms,labels_ms,flow_ms,node_count,edge_count,reason,audio_ms
        let lines = csv.components(separatedBy: "\n").dropFirst().filter { !$0.isEmpty }
        guard !lines.isEmpty else { print("[Empty frame timing]"); return }

        struct Frame {
            let idx: Int, wallDt: Double, total: Double
            let drain: Double, sim: Double, mascot: Double, nodes: Double, edges: Double
            let neb: Double, labels: Double, flow: Double
            let nodeCount: Int, edgeCount: Int, reason: String, audio: Double
        }

        var frames: [Frame] = []
        for (i, line) in lines.enumerated() {
            let c = line.components(separatedBy: ",")
            guard c.count >= 15 else { continue }
            frames.append(Frame(
                idx: i, wallDt: Double(c[2]) ?? 0, total: Double(c[3]) ?? 0,
                drain: Double(c[4]) ?? 0,
                sim: Double(c[5]) ?? 0, mascot: Double(c[6]) ?? 0,
                nodes: Double(c[7]) ?? 0, edges: Double(c[8]) ?? 0, neb: Double(c[9]) ?? 0,
                labels: Double(c[10]) ?? 0, flow: Double(c[11]) ?? 0,
                nodeCount: Int(c[12]) ?? 0, edgeCount: Int(c[13]) ?? 0,
                reason: c[14].trimmingCharacters(in: .whitespacesAndNewlines),
                audio: c.count > 15 ? (Double(c[15].trimmingCharacters(in: .whitespacesAndNewlines)) ?? 0) : 0))
        }

        let totals = frames.map(\.total).sorted()
        let wallDts = frames.filter { $0.wallDt > 0 }.map(\.wallDt).sorted()

        print("\n── METAL FRAME TIMING ──")
        print("Frames: \(frames.count)  MaxNodes: \(frames.map(\.nodeCount).max() ?? 0)  MaxEdges: \(frames.map(\.edgeCount).max() ?? 0)")
        print("total_ms:  p50=\(fmt(percentile(totals, 0.5)))  p95=\(fmt(percentile(totals, 0.95)))  p99=\(fmt(percentile(totals, 0.99)))  worst=\(fmt(totals.last ?? 0))")
        print("wall_dt:   p50=\(fmt(percentile(wallDts, 0.5)))  p95=\(fmt(percentile(wallDts, 0.95)))  p99=\(fmt(percentile(wallDts, 0.99)))  worst=\(fmt(wallDts.last ?? 0))")

        let stalls33 = frames.filter { $0.wallDt > 33 }.count
        let stalls50 = frames.filter { $0.wallDt > 50 }.count
        let stalls100 = frames.filter { $0.wallDt > 100 }.count
        print("Stalls: >33ms=\(stalls33)  >50ms=\(stalls50)  >100ms=\(stalls100)")

        // Sub-phase breakdown for sim frames
        let simFrames = frames.filter { $0.reason == "sim" }
        if !simFrames.isEmpty {
            print("\nSub-phase p95 (sim frames only, \(simFrames.count) frames):")
            let phases: [(String, KeyPath<Frame, Double>)] = [
                ("sim", \.sim), ("mascot", \.mascot), ("nodes", \.nodes),
                ("edges", \.edges), ("nebulae", \.neb), ("labels", \.labels),
                ("flow", \.flow), ("audio", \.audio)
            ]
            var bottlenecks: [(String, Double)] = []
            for (name, kp) in phases {
                let vals = simFrames.map { $0[keyPath: kp] }.sorted()
                let p95 = percentile(vals, 0.95)
                let worst = vals.last ?? 0
                bottlenecks.append((name, p95))
                print("  \(name.padding(toLength: 10, withPad: " ", startingAt: 0)) p95=\(fmt(p95))  worst=\(fmt(worst))")
            }
            let ranked = bottlenecks.sorted { $0.1 > $1.1 }
            print("\nBOTTLENECK RANKING (by p95):")
            for (i, (name, val)) in ranked.enumerated() {
                let bar = String(repeating: "#", count: min(Int(val / 0.1), 40))
                print("  #\(i+1) \(name.padding(toLength: 10, withPad: " ", startingAt: 0)) \(fmt(val))ms \(bar)")
            }
        }

        // Worst 15 frames
        let worst = frames.sorted { $0.wallDt > $1.wallDt }.prefix(15)
        print("\nWorst 15 frames by wall_dt:")
        print("  frame  wall_dt  total   sim   nodes edges labels  neb  mascot audio  nodeN edgeN reason")
        for fr in worst {
            print(String(format: "  %5d %7.1f %6.1f %5.1f %6.1f %5.1f %6.1f %5.1f %6.1f %5.1f %5d %5d  %@",
                fr.idx, fr.wallDt, fr.total, fr.sim, fr.nodes, fr.edges, fr.labels, fr.neb, fr.mascot, fr.audio, fr.nodeCount, fr.edgeCount, fr.reason))
        }
    }

    private func analyzeFlushes() {
        let path = "/tmp/flush-timing.csv"
        guard let data = FileManager.default.contents(atPath: path),
              let csv = String(data: data, encoding: .utf8) else {
            print("\n[No flush-timing.csv — no observer flushes during test]")
            return
        }

        let lines = csv.components(separatedBy: "\n").dropFirst().filter { !$0.isEmpty }
        guard !lines.isEmpty else { print("[Empty flush timing]"); return }

        // CSV: timestamp,galaxy,batch_size,flush_ms,...
        var flushMs: [Double] = []
        var batchSizes: [Int] = []
        for line in lines {
            let c = line.components(separatedBy: ",")
            guard c.count >= 4 else { continue }
            flushMs.append(Double(c[3]) ?? 0)
            batchSizes.append(Int(c[2]) ?? 0)
        }

        let sorted = flushMs.sorted()
        print("\n── FLUSH TIMING ──")
        print("Flush events: \(lines.count)")
        print("flush_ms:   p50=\(fmt(percentile(sorted, 0.5)))  p95=\(fmt(percentile(sorted, 0.95)))  worst=\(fmt(sorted.last ?? 0))")
        print("batch_size: avg=\(String(format: "%.0f", batchSizes.isEmpty ? 0 : Double(batchSizes.reduce(0,+)) / Double(batchSizes.count)))  max=\(batchSizes.max() ?? 0)")

        // Print each flush event
        print("\nAll flushes:")
        print("  timestamp           galaxy          batch  flush_ms")
        for line in lines {
            let c = line.components(separatedBy: ",")
            guard c.count >= 4 else { continue }
            print("  \(c[0].padding(toLength: 19, withPad: " ", startingAt: 0)) \(c[1].padding(toLength: 15, withPad: " ", startingAt: 0)) \(c[2].padding(toLength: 6, withPad: " ", startingAt: 0)) \(c[3])")
        }
    }
}
