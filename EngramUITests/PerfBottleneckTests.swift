import XCTest
import EngramKit
import Lattice

/// Comprehensive performance bottleneck test.
/// Exercises every interaction type in labeled phases, captures ALL instrumentation
/// CSVs, and produces a ranked bottleneck analysis.
///
/// Requires the Engram-UITesting scheme (ENGRAM_INSTRUMENTATION flag).
@MainActor
final class PerfBottleneckTests: XCTestCase {
    let app = XCUIApplication()

    private enum VK {
        static let w: CGKeyCode = 13
        static let a: CGKeyCode = 0
        static let s: CGKeyCode = 1
        static let d: CGKeyCode = 2
        static let q: CGKeyCode = 12
        static let e: CGKeyCode = 14
        static let i: CGKeyCode = 34
        static let j: CGKeyCode = 38
        static let k: CGKeyCode = 40
        static let l: CGKeyCode = 37
        static let t: CGKeyCode = 17
        static let shift: CGKeyCode = 56
    }

    private var localDbPath: String!
    private var syncedDbPath: String!
    private var localIds: [UUID] = []
    private let testUUID = UUID().uuidString

    // All CSV paths we collect
    private let csvPaths = [
        "/tmp/metal-frame-timing.csv",
        "/tmp/draw-timing.csv",
        "/tmp/flush-timing.csv",
        "/tmp/pack-timing.csv",
        "/tmp/merge-timing.csv",
        "/tmp/atlas-timing.log",
        "/tmp/center-log.csv",
        "/tmp/glow-log.csv",
        "/tmp/galaxy-migration.csv",
    ]

    override func setUpWithError() throws {
        continueAfterFailure = false

        let tmpDir = NSTemporaryDirectory()
        localDbPath = tmpDir + "perf-test-\(testUUID).sqlite"
        syncedDbPath = (localDbPath as NSString).deletingPathExtension + "-synced.sqlite"

        // Clean ALL previous timing data
        for path in csvPaths { try? FileManager.default.removeItem(atPath: path) }

        // Seed a realistically large multi-project database
        let projects: [(name: String, count: Int)] = [
            ("Engram", 80), ("Lattice", 50), ("canary-sdk-ios", 40),
            ("canary-sdk-android", 30), ("global", 60), ("engram-server", 25),
        ]
        localIds = seedMultiProjectDatabase(
            at: localDbPath,
            projects: projects,
            staleAge: 7200,
            withSyncConfig: ["Engram", "global"]
        )

        // Seed synced DB for multi-galaxy perf
        seedMultiProjectDatabase(
            at: syncedDbPath,
            projects: [("Engram", 80), ("global", 60)],
            staleAge: 10800
        )

        app.launchEnvironment["CLAUDE_MEMORY_DB"] = localDbPath
        app.launchEnvironment["ENGRAM_TEST_NO_NOTIFY"] = "1"
        app.launchEnvironment["ENGRAM_TEST_INSERT_DELAY"] = "45"
        app.launchEnvironment["TEST_AUTH_TOKEN"] = "mock-\(testUUID)"
        app.launchEnvironment["TEST_SYNC_CHANNEL"] = "perf-\(testUUID)"
        app.launchArguments += ["-subscription_status", "active"]
        app.launchArguments += ["-subscription_tier", "premium"]
    }

    override func tearDownWithError() throws {
        for path in [localDbPath!, syncedDbPath!] {
            try? FileManager.default.removeItem(atPath: path)
            for suffix in ["-wal", "-shm"] {
                try? FileManager.default.removeItem(atPath: path + suffix)
            }
        }
    }

    // MARK: - Main Test

    func testPerfBottlenecks() throws {
        app.launch()

        let window = app.windows.firstMatch
        XCTAssertTrue(window.waitForExistence(timeout: 15), "App window not found")
        sleep(5) // simulation settle for ~285 nodes

        // Defocus search bar
        app.typeKey(.escape, modifierFlags: [])
        usleep(200_000)

        // ── Phase 0: Idle baseline ──
        takeScreenshot(name: "perf-00-idle")
        sleep(3)
        let idleFrameCount = currentFrameCount()

        // ── Phase 1: WASD camera movement ──
        holdKey(VK.w, duration: 1.0)
        holdKey(VK.s, duration: 1.0)
        holdKey(VK.a, duration: 1.0)
        holdKey(VK.d, duration: 1.0)
        sleep(1)
        takeScreenshot(name: "perf-01-wasd")

        // ── Phase 2: Vertical + look ──
        holdKey(VK.e, duration: 0.8)
        holdKey(VK.q, duration: 0.8)
        holdKey(VK.i, duration: 0.6)
        holdKey(VK.k, duration: 0.6)
        holdKey(VK.j, duration: 0.6)
        holdKey(VK.l, duration: 0.6)
        sleep(1)
        takeScreenshot(name: "perf-02-look")

        // ── Phase 3: Sprint ──
        holdKeys([VK.shift, VK.w], duration: 1.5)
        holdKeys([VK.shift, VK.d], duration: 1.0)
        sleep(1)
        takeScreenshot(name: "perf-03-sprint")

        // ── Phase 4: Teleport through all projects ──
        for p in 0..<6 {
            window.typeKey("t", modifierFlags: [])
            sleep(2)
            if p == 0 || p == 3 { takeScreenshot(name: "perf-04-teleport-\(p)") }
        }

        // ── Phase 5: Node selection ──
        let center = window.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
        center.tap()
        sleep(1)
        holdKey(VK.w, duration: 0.8) // move with selection
        holdKey(VK.j, duration: 0.5)
        sleep(1)
        takeScreenshot(name: "perf-05-selection")

        // Deselect
        let corner = window.coordinate(withNormalizedOffset: CGVector(dx: 0.05, dy: 0.05))
        corner.tap()
        sleep(1)

        // ── Phase 6: Recall glow ──
        let recallIds = Array(localIds.prefix(5))
        triggerRecall(at: localDbPath, globalIds: recallIds)
        sleep(3)
        takeScreenshot(name: "perf-06-glow")

        // ── Phase 7: Movement during glow (stress: glow + camera + render) ──
        holdKey(VK.w, duration: 1.0)
        holdKey(VK.j, duration: 0.5)
        sleep(2)

        // ── Phase 8: Wait for insert at ~45s from launch ──
        // (ENGRAM_TEST_INSERT_DELAY=45, we're at ~40s)
        sleep(8)
        takeScreenshot(name: "perf-08-post-insert")

        // ── Phase 9: Sync toggle (project migration) ──
        // Toggle "Lattice" project to sync → triggers migration personal→synced
        toggleSyncConfig(at: localDbPath, project: "Lattice", policy: .sync)
        sleep(3)
        takeScreenshot(name: "perf-09a-sync-on")

        // Toggle it back to local → triggers migration synced→personal
        toggleSyncConfig(at: localDbPath, project: "Lattice", policy: .local)
        sleep(3)
        takeScreenshot(name: "perf-09b-sync-off")

        // Rapid toggle stress: toggle canary-sdk-ios on and off quickly
        toggleSyncConfig(at: localDbPath, project: "canary-sdk-ios", policy: .sync)
        usleep(500_000)
        toggleSyncConfig(at: localDbPath, project: "canary-sdk-ios", policy: .local)
        sleep(3)
        takeScreenshot(name: "perf-09c-rapid-toggle")

        // ── Phase 10: Rapid teleport stress ──
        for _ in 0..<5 {
            window.typeKey("t", modifierFlags: [])
            usleep(400_000)
        }
        sleep(2)
        takeScreenshot(name: "perf-10-rapid-teleport")

        // ── Phase 11: Final idle ──
        sleep(5)
        takeScreenshot(name: "perf-11-final")

        // ═══════════════════════════════════════
        //   ANALYSIS
        // ═══════════════════════════════════════

        let report = BottleneckReport()

        report.addSection(analyzeMetalFrames())
        report.addSection(analyzeDrawPipeline())
        report.addSection(analyzeFlushTiming())
        report.addSection(analyzePackTiming())
        report.addSection(analyzeMergeTiming())
        report.addSection(analyzeAtlasTiming())
        report.addSection(analyzeGlowLog())
        report.addSection(analyzeCenterLog())
        report.addSection(analyzeMigrationTiming())

        report.printSummary()

        // Hard assertion: p95 frame time under budget
        if let p95 = report.metalP95 {
            XCTAssertLessThan(p95, 33.0,
                "p95 frame time \(String(format: "%.1f", p95))ms exceeds 30fps budget (33ms)")
        }
    }

    // MARK: - Frame Count

    private func currentFrameCount() -> Int {
        let path = "/tmp/metal-frame-timing.csv"
        guard let data = FileManager.default.contents(atPath: path),
              let csv = String(data: data, encoding: .utf8) else { return 0 }
        return csv.components(separatedBy: "\n").count - 2 // header + trailing newline
    }

    // MARK: - Metal Frame Timing

    private func analyzeMetalFrames() -> BottleneckSection {
        var section = BottleneckSection(title: "METAL FRAME TIMING")
        let path = "/tmp/metal-frame-timing.csv"
        guard let data = FileManager.default.contents(atPath: path),
              let csv = String(data: data, encoding: .utf8) else {
            section.addLine("[No data]")
            return section
        }

        // CSV: frame,dt_ms,wall_dt_ms,total_ms,sim_ms,mascot_ms,nodes_ms,edges_ms,neb_ms,labels_ms,flow_ms,node_count,edge_count,reason
        let lines = csv.components(separatedBy: "\n").dropFirst().filter { !$0.isEmpty }
        guard !lines.isEmpty else { section.addLine("[Empty]"); return section }

        struct Frame {
            let idx: Int, dt: Double, wallDt: Double, total: Double
            let sim: Double, mascot: Double, nodes: Double, edges: Double
            let neb: Double, labels: Double, flow: Double
            let nodeCount: Int, edgeCount: Int, reason: String
        }
        var frames: [Frame] = []
        for (i, line) in lines.enumerated() {
            let c = line.components(separatedBy: ",")
            guard c.count >= 14 else { continue }
            frames.append(Frame(
                idx: i, dt: Double(c[1]) ?? 0, wallDt: Double(c[2]) ?? 0,
                total: Double(c[3]) ?? 0, sim: Double(c[4]) ?? 0, mascot: Double(c[5]) ?? 0,
                nodes: Double(c[6]) ?? 0, edges: Double(c[7]) ?? 0, neb: Double(c[8]) ?? 0,
                labels: Double(c[9]) ?? 0, flow: Double(c[10]) ?? 0,
                nodeCount: Int(c[11]) ?? 0, edgeCount: Int(c[12]) ?? 0,
                reason: c[13].trimmingCharacters(in: .whitespacesAndNewlines)))
        }

        let totals = frames.map(\.total).sorted()
        let wallDts = frames.filter { $0.wallDt > 0 }.map(\.wallDt).sorted()
        let p50 = pct(totals, 0.50), p95 = pct(totals, 0.95), p99 = pct(totals, 0.99)
        let wp50 = pct(wallDts, 0.50), wp95 = pct(wallDts, 0.95), wp99 = pct(wallDts, 0.99)
        section.metalP95 = wp95

        let maxNodeCount = frames.map(\.nodeCount).max() ?? 0
        let maxEdgeCount = frames.map(\.edgeCount).max() ?? 0

        section.addLine("Frames: \(frames.count)  Nodes: \(maxNodeCount)  Edges: \(maxEdgeCount)")
        section.addLine("total_ms:  p50=\(f(p50))  p95=\(f(p95))  p99=\(f(p99))  worst=\(f(totals.last ?? 0))")
        section.addLine("wall_dt:   p50=\(f(wp50))  p95=\(f(wp95))  p99=\(f(wp99))  worst=\(f(wallDts.last ?? 0))")

        // Stalls
        let stalls25 = frames.filter { $0.wallDt > 25 }.count
        let stalls50 = frames.filter { $0.wallDt > 50 }.count
        section.addLine("Stalls: >25ms=\(stalls25)  >50ms=\(stalls50)")

        // Reason breakdown
        var reasons: [String: [Frame]] = [:]
        for fr in frames { reasons[fr.reason, default: []].append(fr) }
        section.addLine("")
        section.addLine("By reason:")
        for (reason, rFrames) in reasons.sorted(by: { $0.value.count > $1.value.count }) {
            let rTotals = rFrames.map(\.total).sorted()
            let rp50 = pct(rTotals, 0.50)
            let rp95 = pct(rTotals, 0.95)
            section.addLine("  \(reason.padding(toLength: 8, withPad: " ", startingAt: 0)) \(String(format: "%5d", rFrames.count)) frames  p50=\(f(rp50))  p95=\(f(rp95))")
        }

        // Per-phase averages (identify which phase is expensive)
        let simFrames = reasons["sim"] ?? []
        if !simFrames.isEmpty {
            section.addLine("")
            section.addLine("Sub-phase averages (sim frames only):")
            let phases: [(String, KeyPath<Frame, Double>)] = [
                ("sim", \.sim), ("mascot", \.mascot), ("nodes", \.nodes),
                ("edges", \.edges), ("nebulae", \.neb), ("labels", \.labels), ("flow", \.flow)
            ]
            var bottlenecks: [(String, Double)] = []
            for (name, kp) in phases {
                let vals = simFrames.map { $0[keyPath: kp] }
                let avg = vals.reduce(0, +) / Double(vals.count)
                let max = vals.max() ?? 0
                let sorted = vals.sorted()
                let sp95 = pct(sorted, 0.95)
                section.addLine("  \(name.padding(toLength: 10, withPad: " ", startingAt: 0)) avg=\(f(avg))  p95=\(f(sp95))  worst=\(f(max))")
                bottlenecks.append((name, sp95))
            }
            // Rank bottlenecks
            let ranked = bottlenecks.sorted { $0.1 > $1.1 }
            section.addLine("")
            section.addLine("BOTTLENECK RANKING (by p95):")
            for (i, (name, val)) in ranked.enumerated() {
                let bar = String(repeating: "█", count: min(Int(val / 0.1), 40))
                section.addLine("  #\(i+1) \(name.padding(toLength: 10, withPad: " ", startingAt: 0)) \(f(val))ms \(bar)")
            }
        }

        // Worst 10 frames
        let worst = frames.sorted { $0.total > $1.total }.prefix(10)
        section.addLine("")
        section.addLine("Worst 10 frames:")
        section.addLine("  frame  wall_dt  total   sim   nodes edges labels  neb  reason")
        for fr in worst {
            section.addLine(String(format: "  %5d %7.1f %6.1f %5.1f %6.1f %5.1f %6.1f %5.1f  %@",
                fr.idx, fr.wallDt, fr.total, fr.sim, fr.nodes, fr.edges, fr.labels, fr.neb, fr.reason))
        }

        return section
    }

    // MARK: - Draw Pipeline

    private func analyzeDrawPipeline() -> BottleneckSection {
        var section = BottleneckSection(title: "DRAW PIPELINE")
        let path = "/tmp/draw-timing.csv"
        guard let data = FileManager.default.contents(atPath: path),
              let csv = String(data: data, encoding: .utf8) else {
            section.addLine("[No data]")
            return section
        }

        let lines = csv.components(separatedBy: "\n").dropFirst().filter { !$0.isEmpty }
        guard !lines.isEmpty else { section.addLine("[Empty]"); return section }

        // CSV: frame,total_draw_ms,callback_ms,wait_ms,encode_ms
        var totals: [Double] = [], callbacks: [Double] = []
        var waits: [Double] = [], encodes: [Double] = []
        for line in lines {
            let c = line.components(separatedBy: ",")
            guard c.count >= 5 else { continue }
            totals.append(Double(c[1]) ?? 0)
            callbacks.append(Double(c[2]) ?? 0)
            waits.append(Double(c[3]) ?? 0)
            encodes.append(Double(c[4]) ?? 0)
        }

        let st = totals.sorted(), sc = callbacks.sorted(), sw = waits.sorted(), se = encodes.sorted()
        section.addLine("Draws: \(lines.count)")
        section.addLine("total:    p50=\(f(pct(st, 0.5)))  p95=\(f(pct(st, 0.95)))  worst=\(f(st.last ?? 0))")
        section.addLine("callback: p50=\(f(pct(sc, 0.5)))  p95=\(f(pct(sc, 0.95)))  worst=\(f(sc.last ?? 0))")
        section.addLine("wait:     p50=\(f(pct(sw, 0.5)))  p95=\(f(pct(sw, 0.95)))  worst=\(f(sw.last ?? 0))")
        section.addLine("encode:   p50=\(f(pct(se, 0.5)))  p95=\(f(pct(se, 0.95)))  worst=\(f(se.last ?? 0))")

        let badWaits = waits.filter { $0 > 5 }.count
        if badWaits > 0 { section.addLine("⚠️ GPU wait >5ms: \(badWaits) frames") }
        return section
    }

    // MARK: - Flush Timing

    private func analyzeFlushTiming() -> BottleneckSection {
        var section = BottleneckSection(title: "FLUSH TIMING (main-thread observer batches)")
        let path = "/tmp/flush-timing.csv"
        guard let data = FileManager.default.contents(atPath: path),
              let csv = String(data: data, encoding: .utf8) else {
            section.addLine("[No data — no observer flushes during test]")
            return section
        }

        let lines = csv.components(separatedBy: "\n").dropFirst().filter { !$0.isEmpty }
        guard !lines.isEmpty else { section.addLine("[Empty]"); return section }

        var flushMs: [Double] = []
        var batchSizes: [Int] = []
        for line in lines {
            let c = line.components(separatedBy: ",")
            guard c.count >= 4 else { continue }
            flushMs.append(Double(c[3]) ?? 0)
            batchSizes.append(Int(c[2]) ?? 0)
        }

        let sorted = flushMs.sorted()
        section.addLine("Flush events: \(lines.count)")
        section.addLine("flush_ms:   p50=\(f(pct(sorted, 0.5)))  p95=\(f(pct(sorted, 0.95)))  worst=\(f(sorted.last ?? 0))")
        section.addLine("batch_size: avg=\(String(format: "%.0f", batchSizes.isEmpty ? 0 : Double(batchSizes.reduce(0,+)) / Double(batchSizes.count)))  max=\(batchSizes.max() ?? 0)")

        let heavyFlushes = flushMs.filter { $0 > 5 }
        if !heavyFlushes.isEmpty {
            section.addLine("⚠️ Heavy flushes (>5ms): \(heavyFlushes.count) — worst=\(f(heavyFlushes.max()!))")
        }
        return section
    }

    // MARK: - Pack Timing

    private func analyzePackTiming() -> BottleneckSection {
        var section = BottleneckSection(title: "PACK NODE INSTANCES")
        let path = "/tmp/pack-timing.csv"
        guard let data = FileManager.default.contents(atPath: path),
              let csv = String(data: data, encoding: .utf8) else {
            section.addLine("[No data]")
            return section
        }

        let lines = csv.components(separatedBy: "\n").dropFirst().filter { !$0.isEmpty }
        guard !lines.isEmpty else { section.addLine("[Empty]"); return section }

        // CSV: timestamp,precompute_ms,loop_ms,total_ms,node_count
        var precomputes: [Double] = [], loops: [Double] = [], packTotals: [Double] = []
        for line in lines {
            let c = line.components(separatedBy: ",")
            guard c.count >= 5 else { continue }
            precomputes.append(Double(c[1]) ?? 0)
            loops.append(Double(c[2]) ?? 0)
            packTotals.append(Double(c[3]) ?? 0)
        }

        let sp = precomputes.sorted(), sl = loops.sorted(), st = packTotals.sorted()
        section.addLine("Pack calls: \(lines.count)")
        section.addLine("precompute: p50=\(f(pct(sp, 0.5)))  p95=\(f(pct(sp, 0.95)))  worst=\(f(sp.last ?? 0))")
        section.addLine("loop:       p50=\(f(pct(sl, 0.5)))  p95=\(f(pct(sl, 0.95)))  worst=\(f(sl.last ?? 0))")
        section.addLine("total:      p50=\(f(pct(st, 0.5)))  p95=\(f(pct(st, 0.95)))  worst=\(f(st.last ?? 0))")
        return section
    }

    // MARK: - Merge Timing

    private func analyzeMergeTiming() -> BottleneckSection {
        var section = BottleneckSection(title: "GALAXY REGISTRY MERGE")
        let path = "/tmp/merge-timing.csv"
        guard let data = FileManager.default.contents(atPath: path),
              let csv = String(data: data, encoding: .utf8) else {
            section.addLine("[No data — merge below 0.5ms threshold or single galaxy]")
            return section
        }

        let lines = csv.components(separatedBy: "\n").dropFirst().filter { !$0.isEmpty }
        guard !lines.isEmpty else { section.addLine("[Empty]"); return section }

        section.addLine("Merge events (>0.5ms): \(lines.count)")
        for line in lines.prefix(10) {
            section.addLine("  \(line)")
        }
        if lines.count > 10 { section.addLine("  ... (\(lines.count - 10) more)") }
        return section
    }

    // MARK: - Atlas Timing

    private func analyzeAtlasTiming() -> BottleneckSection {
        var section = BottleneckSection(title: "LABEL ATLAS REGEN")
        let path = "/tmp/atlas-timing.log"
        guard let data = FileManager.default.contents(atPath: path),
              let content = String(data: data, encoding: .utf8) else {
            section.addLine("[No data — no atlas regens during test]")
            return section
        }

        let lines = content.components(separatedBy: "\n").filter { !$0.isEmpty }
        section.addLine("Atlas regens: \(lines.count)")
        for line in lines { section.addLine("  \(line)") }
        return section
    }

    // MARK: - Glow Log

    private func analyzeGlowLog() -> BottleneckSection {
        var section = BottleneckSection(title: "RECALL GLOW EVENTS")
        let events = parseGlowLog()
        if events.isEmpty {
            section.addLine("[No glow events]")
            return section
        }

        section.addLine("Total: \(events.count)")

        var nodeGlowCounts: [String: Int] = [:]
        var galaxyCounts: [String: Int] = [:]
        for event in events {
            nodeGlowCounts[event.nodeLabel, default: 0] += 1
            galaxyCounts[event.galaxy, default: 0] += 1
        }

        for (galaxy, count) in galaxyCounts.sorted(by: { $0.key < $1.key }) {
            section.addLine("  \(galaxy): \(count)")
        }

        let repeaters = nodeGlowCounts.filter { $0.value > 1 }
        if !repeaters.isEmpty {
            section.addLine("⚠️ Repeated glows: \(repeaters.count) nodes")
            for (label, count) in repeaters.sorted(by: { $0.value > $1.value }).prefix(5) {
                section.addLine("  \(label.prefix(30)): \(count)x")
            }
        }

        let refires = events.filter(\.alreadyGlowing).count
        if refires > 0 { section.addLine("Re-fires: \(refires)") }
        return section
    }

    // MARK: - Center Log

    private func analyzeCenterLog() -> BottleneckSection {
        var section = BottleneckSection(title: "CENTER-ON-GRAPH")
        let path = "/tmp/center-log.csv"
        guard let data = FileManager.default.contents(atPath: path),
              let csv = String(data: data, encoding: .utf8) else {
            section.addLine("[No center events]")
            return section
        }

        let lines = csv.components(separatedBy: "\n").dropFirst().filter { !$0.isEmpty }
        section.addLine("Center events: \(lines.count)")

        let withKeys = lines.filter { line in
            let cols = line.components(separatedBy: ",")
            return cols.count >= 3 && (Int(cols[2]) ?? 0) > 0
        }
        if !withKeys.isEmpty {
            section.addLine("⚠️ Center fired with keys held: \(withKeys.count)")
        }
        return section
    }

    // MARK: - Migration Timing

    private func analyzeMigrationTiming() -> BottleneckSection {
        var section = BottleneckSection(title: "GALAXY MIGRATION (sync toggles)")
        let path = "/tmp/galaxy-migration.csv"
        guard let data = FileManager.default.contents(atPath: path),
              let csv = String(data: data, encoding: .utf8) else {
            section.addLine("[No data — no sync toggles occurred]")
            return section
        }

        let lines = csv.components(separatedBy: "\n").dropFirst().filter { !$0.isEmpty }
        guard !lines.isEmpty else { section.addLine("[Empty]"); return section }

        // CSV: timestamp,action,project,direction,src,dst,node_count,edge_count,position_count,ms
        section.addLine("Migration events: \(lines.count)")
        section.addLine("")
        section.addLine("action              project             direction           nodes edges  pos    ms")

        var totalMigrateMs: Double = 0
        var migrateCount = 0

        for line in lines {
            let c = line.components(separatedBy: ",")
            guard c.count >= 10 else { continue }
            let action = c[1].padding(toLength: 19, withPad: " ", startingAt: 0)
            let project = c[2].padding(toLength: 19, withPad: " ", startingAt: 0)
            let direction = c[3].padding(toLength: 19, withPad: " ", startingAt: 0)
            let nodes = c[6].padding(toLength: 5, withPad: " ", startingAt: 0)
            let edges = c[7].padding(toLength: 5, withPad: " ", startingAt: 0)
            let positions = c[8].padding(toLength: 5, withPad: " ", startingAt: 0)
            let ms = c[9].trimmingCharacters(in: .whitespacesAndNewlines)
            section.addLine("\(action) \(project) \(direction) \(nodes) \(edges) \(positions) \(ms.padding(toLength: 8, withPad: " ", startingAt: 0))")

            if c[1] == "migrate" || c[1] == "migrate_out" || c[1] == "load_into" {
                if let msVal = Double(ms) { totalMigrateMs += msVal; migrateCount += 1 }
            }
        }

        if migrateCount > 0 {
            section.addLine("")
            section.addLine("Total migration work: \(String(format: "%.1f", totalMigrateMs))ms across \(migrateCount) operations")
            let heavyOps = lines.filter { line in
                let c = line.components(separatedBy: ",")
                return c.count >= 10 && (Double(c[9].trimmingCharacters(in: .whitespacesAndNewlines)) ?? 0) > 10
            }
            if !heavyOps.isEmpty {
                section.addLine("⚠️ Heavy migrations (>10ms): \(heavyOps.count)")
            }
        }
        return section
    }

    // MARK: - Helpers

    private nonisolated func holdKey(_ keyCode: CGKeyCode, duration: TimeInterval) {
        let src = CGEventSource(stateID: .combinedSessionState)
        CGEvent(keyboardEventSource: src, virtualKey: keyCode, keyDown: true)?.post(tap: .cgSessionEventTap)
        Thread.sleep(forTimeInterval: duration)
        CGEvent(keyboardEventSource: src, virtualKey: keyCode, keyDown: false)?.post(tap: .cgSessionEventTap)
        Thread.sleep(forTimeInterval: 0.05)
    }

    private nonisolated func holdKeys(_ keyCodes: [CGKeyCode], duration: TimeInterval) {
        let src = CGEventSource(stateID: .combinedSessionState)
        for kc in keyCodes {
            CGEvent(keyboardEventSource: src, virtualKey: kc, keyDown: true)?.post(tap: .cgSessionEventTap)
        }
        Thread.sleep(forTimeInterval: duration)
        for kc in keyCodes.reversed() {
            CGEvent(keyboardEventSource: src, virtualKey: kc, keyDown: false)?.post(tap: .cgSessionEventTap)
        }
        Thread.sleep(forTimeInterval: 0.05)
    }

    private func takeScreenshot(name: String) {
        let screenshot = XCUIScreen.main.screenshot()
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
        try? screenshot.pngRepresentation.write(to: URL(fileURLWithPath: "/tmp/\(name).png"))
    }

    private func f(_ v: Double) -> String { String(format: "%7.2f", v) }

    private func pct(_ sorted: [Double], _ p: Double) -> Double {
        guard !sorted.isEmpty else { return 0 }
        return sorted[min(Int(Double(sorted.count) * p), sorted.count - 1)]
    }
}

// MARK: - Sync Config Toggle Helper

/// Opens a separate Lattice connection and sets a project's sync policy.
/// The xproc notification triggers the app's GalaxyRegistry SyncConfig observer.
func toggleSyncConfig(at path: String, project: String, policy: SyncConfig.Policy) {
    let lattice = try! Lattice(
        Memory.self, Edge.self, SyncConfig.self,
        configuration: .init(
            fileURL: URL(fileURLWithPath: path),
            migration: engramMigrations
        )
    )

    // Check if config exists
    let existing = lattice.objects(SyncConfig.self).first(where: { $0.project == project })
    if let existing {
        existing.policy = policy
        existing.updatedAt = Date()
    } else {
        lattice.add(SyncConfig(project: project, policy: policy))
    }
}

// MARK: - Report Infrastructure

private struct BottleneckSection {
    let title: String
    var lines: [String] = []
    var metalP95: Double?

    init(title: String) { self.title = title }

    mutating func addLine(_ line: String) { lines.append(line) }
}

private class BottleneckReport {
    private var sections: [BottleneckSection] = []
    var metalP95: Double?

    func addSection(_ section: BottleneckSection) {
        sections.append(section)
        if let p = section.metalP95 { metalP95 = p }
    }

    func printSummary() {
        print("")
        print("╔═══════════════════════════════════════════════════════════════════════════╗")
        print("║                    PERFORMANCE BOTTLENECK REPORT                         ║")
        print("╠═══════════════════════════════════════════════════════════════════════════╣")

        for section in sections {
            print("║                                                                           ║")
            print("║  ── \(section.title) \(String(repeating: "─", count: max(0, 55 - section.title.count)))  ║")
            for line in section.lines {
                print("  \(line)")
            }
        }

        // Summary: collect all warnings
        let warnings = sections.flatMap { s in s.lines.filter { $0.contains("⚠️") } }
        if !warnings.isEmpty {
            print("")
            print("║  ── WARNINGS ────────────────────────────────────────────────  ║")
            for w in warnings { print("  \(w)") }
        }

        print("║                                                                           ║")
        print("╚═══════════════════════════════════════════════════════════════════════════╝")
        print("")
    }
}
