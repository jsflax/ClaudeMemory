import XCTest
import Network
import EngramKit
import Lattice

/// Tests 3D insert jitter by launching the app with a test-insert flag,
/// waiting for settle, letting the app insert a memory on our behalf,
/// then analyzing the Metal frame timing CSV for spikes.
@MainActor
final class InsertJitterTests: XCTestCase {
    let app = XCUIApplication()

    override func setUpWithError() throws {
        continueAfterFailure = false
        // Clean previous timing data
        try? FileManager.default.removeItem(atPath: "/tmp/metal-frame-timing.csv")
        try? FileManager.default.removeItem(atPath: "/tmp/flush-timing.csv")
        try? FileManager.default.removeItem(atPath: "/tmp/draw-timing.csv")
        try? FileManager.default.removeItem(atPath: "/tmp/atlas-timing.log")
        try? FileManager.default.removeItem(atPath: "/tmp/center-log.csv")
        try? FileManager.default.removeItem(atPath: "/tmp/glow-log.csv")
        // Tell the app to insert a test memory after a delay (in seconds)
        app.launchEnvironment["ENGRAM_TEST_INSERT_DELAY"] = "12"
        // Disable notifications to isolate their main-thread impact
        app.launchEnvironment["ENGRAM_TEST_NO_NOTIFY"] = "1"
        app.launch()
        // Wait for app to launch and load data
        sleep(5)
    }

    // MARK: - Insert Jitter Test

    func testInsertMemoryJitter() throws {
        let window = app.windows.firstMatch
        XCTAssertTrue(window.exists, "App window not found")

        // Defocus search bar so keys go to the 3D scene
        app.typeKey(.escape, modifierFlags: [])
        usleep(200_000)

        // Teleport to center of graph
        app.typeKey("t", modifierFlags: [])
        sleep(1)

        takeScreenshot(name: "jitter-01-after-teleport")

        // Camera movement while waiting for insert at ~12s from launch
        // 5s setup + 1s teleport = 6s elapsed, ~6s until insert
        for _ in 0..<3 {
            holdKey(VK.e, duration: 0.8)   // rise
            holdKey(VK.q, duration: 0.8)   // descend
        }

        takeScreenshot(name: "jitter-03-around-insert")

        // Keep moving after insert to see recovery
        for _ in 0..<3 {
            holdKey(VK.i, duration: 0.5)   // look up
            holdKey(VK.k, duration: 0.5)   // look down
            holdKey(VK.w, duration: 0.5)   // forward
            holdKey(VK.s, duration: 0.5)   // back
        }

        takeScreenshot(name: "jitter-04-post-insert")
        sleep(2)

        // Analyze timing data
        analyzeMetalFrameTiming()
        analyzeFlushTiming()
        analyzeDrawTiming()
        analyzeAtlasTiming()
        analyzeCenterLog()
        analyzeGlowLog()
    }

    // MARK: - Metal Frame Timing Analysis

    private func analyzeMetalFrameTiming() {
        let csvPath = "/tmp/metal-frame-timing.csv"
        guard let csvData = FileManager.default.contents(atPath: csvPath),
              let csv = String(data: csvData, encoding: .utf8) else {
            XCTFail("No Metal frame timing data at \(csvPath). Is the app running in DEBUG mode?")
            return
        }

        let lines = csv.components(separatedBy: "\n").dropFirst().filter { !$0.isEmpty }
        guard !lines.isEmpty else { XCTFail("No frame data collected"); return }

        // CSV columns:
        // frame,dt_ms,wall_dt_ms,total_ms,drain_ms,sim_ms,mascot_ms,nodes_ms,edges_ms,neb_ms,labels_ms,flow_ms,node_count,edge_count,reason

        var frames: [(frame: Int, dt: Double, wallDt: Double, total: Double, drain: Double, sim: Double, mascot: Double,
                       nodes: Double, edges: Double, neb: Double, labels: Double, flow: Double,
                       nodeCount: Int, edgeCount: Int, reason: String)] = []

        for line in lines {
            let cols = line.components(separatedBy: ",")
            guard cols.count >= 15 else { continue }
            frames.append((
                frame: Int(cols[0]) ?? 0,
                dt: Double(cols[1]) ?? 0,
                wallDt: Double(cols[2]) ?? 0,
                total: Double(cols[3]) ?? 0,
                drain: Double(cols[4]) ?? 0,
                sim: Double(cols[5]) ?? 0,
                mascot: Double(cols[6]) ?? 0,
                nodes: Double(cols[7]) ?? 0,
                edges: Double(cols[8]) ?? 0,
                neb: Double(cols[9]) ?? 0,
                labels: Double(cols[10]) ?? 0,
                flow: Double(cols[11]) ?? 0,
                nodeCount: Int(cols[12]) ?? 0,
                edgeCount: Int(cols[13]) ?? 0,
                reason: cols[14].trimmingCharacters(in: .whitespacesAndNewlines)
            ))
        }

        // Detect insert by node_count change (last time it changes by exactly 1)
        var insertFrameIdx: Int?
        for i in stride(from: frames.count - 1, through: 1, by: -1) {
            let delta = frames[i].nodeCount - frames[i-1].nodeCount
            if delta == 1 {
                insertFrameIdx = i
                break
            }
        }

        let sortedTotals = frames.map(\.total).sorted()
        let sortedWallDts = frames.filter { $0.wallDt > 0 }.map(\.wallDt).sorted()

        let p50 = percentile(sortedTotals, 0.50)
        let p95 = percentile(sortedTotals, 0.95)
        let p99 = percentile(sortedTotals, 0.99)
        let worst = sortedTotals.last ?? 0

        let wallP50 = percentile(sortedWallDts, 0.50)
        let wallP95 = percentile(sortedWallDts, 0.95)
        let wallP99 = percentile(sortedWallDts, 0.99)
        let wallWorst = sortedWallDts.last ?? 0

        // Frames where wall_dt significantly exceeds expected 16.67ms = main thread stall
        let stalledFrames = frames.filter { $0.wallDt > 25 }
        let badStalls = frames.filter { $0.wallDt > 50 }

        let simFrames = frames.filter { $0.reason == "sim" }
        let avgNodes = simFrames.isEmpty ? 0 : simFrames.map(\.nodes).reduce(0, +) / Double(simFrames.count)
        let avgEdges = simFrames.isEmpty ? 0 : simFrames.map(\.edges).reduce(0, +) / Double(simFrames.count)
        let avgLabels = simFrames.isEmpty ? 0 : simFrames.map(\.labels).reduce(0, +) / Double(simFrames.count)
        let avgSim = simFrames.isEmpty ? 0 : simFrames.map(\.sim).reduce(0, +) / Double(simFrames.count)
        let avgMascot = simFrames.isEmpty ? 0 : simFrames.map(\.mascot).reduce(0, +) / Double(simFrames.count)
        let avgNeb = simFrames.isEmpty ? 0 : simFrames.map(\.neb).reduce(0, +) / Double(simFrames.count)
        let avgFlow = simFrames.isEmpty ? 0 : simFrames.map(\.flow).reduce(0, +) / Double(simFrames.count)

        print("""

        ╔═══════════════════════════════════════════════════════════════════════╗
        ║               METAL 3D INSERT JITTER PROFILE                         ║
        ╠═══════════════════════════════════════════════════════════════════════╣
        ║ Total frames: \(String(format: "%6d", frames.count))   Stalled (>25ms wall_dt): \(String(format: "%4d", stalledFrames.count))   Bad (>50ms): \(String(format: "%4d", badStalls.count))  ║
        ╠═══════════════════════════════════════════════════════════════════════╣
        ║ total_ms (CPU work inside renderTick):                               ║
        ║   p50=\(String(format: "%7.2f", p50))  p95=\(String(format: "%7.2f", p95))  p99=\(String(format: "%7.2f", p99))  worst=\(String(format: "%7.2f", worst))                ║
        ║ wall_dt_ms (actual time between frames — includes main-thread work): ║
        ║   p50=\(String(format: "%7.2f", wallP50))  p95=\(String(format: "%7.2f", wallP95))  p99=\(String(format: "%7.2f", wallP99))  worst=\(String(format: "%7.2f", wallWorst))                ║
        ╠═══════════════════════════════════════════════════════════════════════╣
        ║ Sub-phase averages (ms):                                             ║
        ║   sim=\(String(format: "%6.2f", avgSim))  mascot=\(String(format: "%5.2f", avgMascot))  nodes=\(String(format: "%6.2f", avgNodes))  edges=\(String(format: "%6.2f", avgEdges))  neb=\(String(format: "%5.2f", avgNeb))  labels=\(String(format: "%6.2f", avgLabels))  ║
        ╚═══════════════════════════════════════════════════════════════════════╝

        """)

        // Dump frames around the insert (detected by node_count +1)
        if let idx = insertFrameIdx {
            let start = max(0, idx - 5)
            let end = min(frames.count, idx + 10)
            print("        Frames around insert (frame \(frames[idx].frame), node_count \(frames[idx-1].nodeCount) -> \(frames[idx].nodeCount)):")
            print("        frame  wall_dt  total_ms   sim_ms  nodes  edges  labels    neb  nCount eCount reason")
            for i in start..<end {
                let f = frames[i]
                let marker = i == idx ? " <-- INSERT" : ""
                print(String(format: "        %5d %7.1f %8.2f %7.2f %6.2f %6.2f %6.2f %6.2f %5d  %5d  %@%@",
                    f.frame, f.wallDt, f.total, f.sim, f.nodes, f.edges, f.labels, f.neb,
                    f.nodeCount, f.edgeCount, f.reason, marker))
            }
            print("")
        } else {
            print("        [No insert detected by node_count change]")
            print("")
        }

        // Worst 10 frames by wall_dt (reveals main-thread stalls)
        let worstByWall = frames.filter { $0.wallDt > 0 }.sorted { $0.wallDt > $1.wallDt }.prefix(10)
        print("        Worst 10 frames by wall_dt (main-thread stalls):")
        print("        frame  wall_dt  total_ms   sim_ms  nodes  edges  labels    neb  nCount eCount reason")
        for f in worstByWall {
            let gap = f.wallDt - f.total
            print(String(format: "        %5d %7.1f %8.2f %7.2f %6.2f %6.2f %6.2f %6.2f %5d  %5d  %@  gap=%.1fms",
                f.frame, f.wallDt, f.total, f.sim, f.nodes, f.edges, f.labels, f.neb,
                f.nodeCount, f.edgeCount, f.reason, gap))
        }
        print("")

        // Worst 10 frames by total_ms (CPU work inside renderTick)
        let worstByTotal = frames.sorted { $0.total > $1.total }.prefix(10)
        print("        Worst 10 frames by total_ms (renderTick CPU work):")
        print("        frame  wall_dt  total_ms   sim_ms  nodes  edges  labels    neb  nCount eCount reason")
        for f in worstByTotal {
            print(String(format: "        %5d %7.1f %8.2f %7.2f %6.2f %6.2f %6.2f %6.2f %5d  %5d  %@",
                f.frame, f.wallDt, f.total, f.sim, f.nodes, f.edges, f.labels, f.neb,
                f.nodeCount, f.edgeCount, f.reason))
        }
        print("")
    }

    // MARK: - Flush Timing Analysis

    private func analyzeFlushTiming() {
        let csvPath = "/tmp/flush-timing.csv"
        guard let csvData = FileManager.default.contents(atPath: csvPath),
              let csv = String(data: csvData, encoding: .utf8) else {
            print("        [No flush timing data at \(csvPath)]")
            return
        }

        let lines = csv.components(separatedBy: "\n").dropFirst().filter { !$0.isEmpty }
        guard !lines.isEmpty else { print("        [No flush events recorded]"); return }

        // CSV: timestamp,kind,count,total_ms,loop_ms,sim2d_add_ms,sim3d_add_ms,edge_wire_ms,notify_ms,sim_batch_ms,bump_ms,node_count,edge_count
        print("        ╔══════════════════════════════════════════════════════════════════════════════════╗")
        print("        ║                    FLUSH EVENTS (main-thread work)                              ║")
        print("        ╠══════════════════════════════════════════════════════════════════════════════════╣")
        print("        kind   cnt  total   loop  sim2D  sim3D  edges notify  batch   bump  nodes  edges")
        for line in lines {
            let cols = line.components(separatedBy: ",")
            guard cols.count >= 13 else { continue }
            let kind = cols[1]
            let count = cols[2]
            let totalMs = Double(cols[3]) ?? 0
            let loopMs = Double(cols[4]) ?? 0
            let sim2D = Double(cols[5]) ?? 0
            let sim3D = Double(cols[6]) ?? 0
            let edgeWire = Double(cols[7]) ?? 0
            let notify = Double(cols[8]) ?? 0
            let simBatch = Double(cols[9]) ?? 0
            let bump = Double(cols[10]) ?? 0
            let nc = cols[11]
            let ec = cols[12].trimmingCharacters(in: .whitespacesAndNewlines)
            print("        \(kind.padding(toLength: 5, withPad: " ", startingAt: 0)) \(count.padding(toLength: 3, withPad: " ", startingAt: 0)) \(String(format: "%6.1f %6.1f %6.1f %6.1f %6.1f %6.1f %6.1f %6.1f", totalMs, loopMs, sim2D, sim3D, edgeWire, notify, simBatch, bump)) \(nc.padding(toLength: 6, withPad: " ", startingAt: 0)) \(ec)")
        }
        print("        ╚═══════════════════════════════════════════════════╝")
        print("")
    }

    // MARK: - Draw Timing Analysis

    private func analyzeDrawTiming() {
        let csvPath = "/tmp/draw-timing.csv"
        guard let csvData = FileManager.default.contents(atPath: csvPath),
              let csv = String(data: csvData, encoding: .utf8) else {
            print("        [No draw timing data at \(csvPath)]")
            return
        }

        let lines = csv.components(separatedBy: "\n").dropFirst().filter { !$0.isEmpty }
        guard !lines.isEmpty else { print("        [No draw timing events recorded]"); return }

        // CSV: frame,total_draw_ms,callback_ms,wait_ms,encode_ms
        // Only show frames where total_draw > 20ms or wait > 5ms
        print("        ╔════════════════════════════════════════════════════════════════╗")
        print("        ║           DRAW PIPELINE TIMING (spikes only)                  ║")
        print("        ╠════════════════════════════════════════════════════════════════╣")
        print("        frame  total_draw  callback    wait  encode")
        var shown = 0
        for line in lines {
            let cols = line.components(separatedBy: ",")
            guard cols.count >= 5 else { continue }
            let totalDraw = Double(cols[1]) ?? 0
            let wait = Double(cols[3]) ?? 0
            guard totalDraw > 20 || wait > 5 else { continue }
            let frame = cols[0]
            let callback = Double(cols[2]) ?? 0
            let encode = Double(cols[4]) ?? 0
            print("        \(frame.padding(toLength: 6, withPad: " ", startingAt: 0)) \(String(format: "%10.2f %8.2f %7.2f %7.2f", totalDraw, callback, wait, encode))")
            shown += 1
            if shown >= 20 { break }
        }
        print("        ╚════════════════════════════════════════════════════════════════╝")
        print("")
    }

    // MARK: - Atlas Timing Analysis

    private func analyzeAtlasTiming() {
        let path = "/tmp/atlas-timing.log"
        guard let data = FileManager.default.contents(atPath: path),
              let content = String(data: data, encoding: .utf8) else {
            print("        [No atlas timing data at \(path)]")
            return
        }
        print("        ╔════════════════════════════════════════════════════════════════╗")
        print("        ║                   LABEL ATLAS REGEN TIMING                    ║")
        print("        ╠════════════════════════════════════════════════════════════════╣")
        for line in content.components(separatedBy: "\n") where !line.isEmpty {
            print("        \(line)")
        }
        print("        ╚════════════════════════════════════════════════════════════════╝")
        print("")
    }

    // MARK: - Center Log Analysis

    private func analyzeCenterLog() {
        let path = "/tmp/center-log.csv"
        guard let data = FileManager.default.contents(atPath: path),
              let csv = String(data: data, encoding: .utf8) else {
            print("        [No center log at \(path) — centerOnGraph may not have fired]")
            return
        }

        let lines = csv.components(separatedBy: "\n").dropFirst().filter { !$0.isEmpty }
        print("        ╔════════════════════════════════════════════════════════════════════════╗")
        print("        ║                   CENTER-ON-GRAPH EVENTS                              ║")
        print("        ╠════════════════════════════════════════════════════════════════════════╣")
        print("        frame    elapsed  keys  positions  camTarget(xyz)           targetPos(xyz)")
        for line in lines {
            let cols = line.components(separatedBy: ",")
            guard cols.count >= 10 else { continue }
            print("        \(cols[0].padding(toLength: 8, withPad: " ", startingAt: 0)) \(cols[1].padding(toLength: 8, withPad: " ", startingAt: 0)) \(cols[2].padding(toLength: 5, withPad: " ", startingAt: 0)) \(cols[3].padding(toLength: 10, withPad: " ", startingAt: 0)) (\(cols[4]),\(cols[5]),\(cols[6]))  (\(cols[7]),\(cols[8]),\(cols[9].trimmingCharacters(in: .whitespacesAndNewlines)))")
        }
        print("        Total center events: \(lines.count)")
        // Check for events where keys were held during centering
        let withKeys = lines.filter { line in
            let cols = line.components(separatedBy: ",")
            return cols.count >= 3 && (Int(cols[2]) ?? 0) > 0
        }
        if !withKeys.isEmpty {
            print("        ⚠️ CENTER fired with keys held: \(withKeys.count) events (potential jerk source)")
        }
        print("        ╚════════════════════════════════════════════════════════════════════════╝")
        print("")
    }

    // MARK: - Glow Log Analysis

    private func analyzeGlowLog() {
        let path = "/tmp/glow-log.csv"
        guard let data = FileManager.default.contents(atPath: path),
              let csv = String(data: data, encoding: .utf8) else {
            print("        [No glow log at \(path) — no recall glows triggered]")
            return
        }

        let lines = csv.components(separatedBy: "\n").dropFirst().filter { !$0.isEmpty }
        print("        ╔════════════════════════════════════════════════════════════════════════╗")
        print("        ║                   RECALL GLOW EVENTS                                  ║")
        print("        ╠════════════════════════════════════════════════════════════════════════╣")
        print("        timestamp       galaxy    node_label                      delta_s  stale_s  refire  glowCt")
        for line in lines {
            let cols = line.components(separatedBy: ",")
            guard cols.count >= 9 else { continue }
            let ts = cols[0]
            let galaxy = cols[1].padding(toLength: 9, withPad: " ", startingAt: 0)
            let label = cols[2].padding(toLength: 30, withPad: " ", startingAt: 0)
            let delta = cols[5].padding(toLength: 8, withPad: " ", startingAt: 0)
            let staleness = cols[6].padding(toLength: 8, withPad: " ", startingAt: 0)
            let alreadyGlowing = cols[7].padding(toLength: 6, withPad: " ", startingAt: 0)
            let count = cols[8].trimmingCharacters(in: .whitespacesAndNewlines)
            print("        \(ts)  \(galaxy) \(label) \(delta) \(staleness) \(alreadyGlowing) \(count)")
        }
        // Count per-galaxy
        var galaxyCounts: [String: Int] = [:]
        for line in lines {
            let cols = line.components(separatedBy: ",")
            guard cols.count >= 2 else { continue }
            galaxyCounts[cols[1], default: 0] += 1
        }
        print("        Total glow events: \(lines.count)")
        for (galaxy, count) in galaxyCounts.sorted(by: { $0.key < $1.key }) {
            print("          \(galaxy): \(count)")
        }
        // Check for repeated glows on the same node
        var nodeGlowCounts: [String: Int] = [:]
        for line in lines {
            let cols = line.components(separatedBy: ",")
            guard cols.count >= 3 else { continue }
            nodeGlowCounts[cols[2], default: 0] += 1
        }
        let repeated = nodeGlowCounts.filter { $0.value > 1 }.sorted(by: { $0.value > $1.value })
        if !repeated.isEmpty {
            print("        ⚠️ Nodes with repeated glows (potential re-trigger):")
            for (node, count) in repeated.prefix(10) {
                print("          \(node): \(count) times")
            }
        }
        // Staleness analysis
        var stalenesses: [Double] = []
        var refireCount = 0
        for line in lines {
            let cols = line.components(separatedBy: ",")
            guard cols.count >= 9 else { continue }
            if let s = Double(cols[6]) { stalenesses.append(s) }
            if cols[7].trimmingCharacters(in: .whitespacesAndNewlines) == "true" { refireCount += 1 }
        }
        if !stalenesses.isEmpty {
            let minS = stalenesses.min()!
            let maxS = stalenesses.max()!
            let avgS = stalenesses.reduce(0, +) / Double(stalenesses.count)
            let liveCount = stalenesses.filter { $0 < 10 }.count
            let staleCount = stalenesses.filter { $0 >= 10 }.count
            print("        Staleness: min=\(String(format: "%.1f", minS))s  max=\(String(format: "%.1f", maxS))s  avg=\(String(format: "%.1f", avgS))s")
            print("        Live (< 10s): \(liveCount)  Stale (>= 10s): \(staleCount)")
            print("        Re-fires (already glowing): \(refireCount)")
        }
        print("        ╚════════════════════════════════════════════════════════════════════════╝")
        print("")
    }

}

// MARK: - Mock Sync Server

/// Minimal WebSocket server for E2E sync testing.
/// Accepts connections and upgrades to WebSocket, but ignores all messages.
/// Lattice handles WSS failure gracefully — IPC works independently.
///
/// Note: XCUITest runner may be sandboxed (Operation not permitted on bind).
/// If the listener fails to start, `port` returns a fallback port.
/// The app will fail to connect to WSS — that's fine, IPC sync still works.
final class MockSyncServer: @unchecked Sendable {
    private let listener: NWListener?
    private let ready = DispatchSemaphore(value: 0)
    nonisolated(unsafe) private var _port: UInt16 = 0

    var port: UInt16 { _port }

    init() throws {
        let params = NWParameters.tcp
        let ws = NWProtocolWebSocket.Options()
        params.defaultProtocolStack.applicationProtocols.insert(ws, at: 0)
        let l = try NWListener(using: params, on: .any)
        self.listener = l
        l.newConnectionHandler = { conn in
            conn.start(queue: .global())
        }
        l.stateUpdateHandler = { [weak self] state in
            if case .ready = state {
                self?._port = l.port?.rawValue ?? 9999
                self?.ready.signal()
            } else if case .failed = state {
                self?._port = 9999  // fallback — app WSS will fail, IPC still works
                self?.ready.signal()
            }
        }
        l.start(queue: .global())
        // Wait up to 2s for listener to bind
        _ = ready.wait(timeout: .now() + 2)
        if _port == 0 { _port = 9999 }
    }

    func stop() { listener?.cancel() }
}

// Test helpers moved to TestHelpers.swift

// MARK: - Glow Sync E2E Tests

/// Tests IPC catch-up stale glow behavior and live recall glow correctness.
/// Requires the Engram-UITesting scheme (TEST_AUTH_TOKEN + TEST_SYNC_CHANNEL flags).
@MainActor
final class GlowSyncTests: XCTestCase {
    let app = XCUIApplication()
    private var server: MockSyncServer!
    private var localDbPath: String!
    private var syncedDbPath: String!
    private let testUUID = UUID().uuidString

    override func setUpWithError() throws {
        continueAfterFailure = false

        server = try MockSyncServer()

        let tmpDir = NSTemporaryDirectory()
        localDbPath = tmpDir + "glow-test-local-\(testUUID).sqlite"
        syncedDbPath = (localDbPath as NSString).deletingPathExtension + "-synced.sqlite"

        // Clean previous logs
        try? FileManager.default.removeItem(atPath: "/tmp/glow-log.csv")
        try? FileManager.default.removeItem(atPath: "/tmp/metal-frame-timing.csv")
    }

    override func tearDownWithError() throws {
        server?.stop()
        try? FileManager.default.removeItem(atPath: localDbPath)
        try? FileManager.default.removeItem(atPath: syncedDbPath)
        // Clean WAL/SHM files
        for suffix in ["-wal", "-shm"] {
            try? FileManager.default.removeItem(atPath: localDbPath + suffix)
            try? FileManager.default.removeItem(atPath: syncedDbPath + suffix)
        }
    }

    private func launchApp() {
        app.launchEnvironment["CLAUDE_MEMORY_DB"] = localDbPath
        app.launchEnvironment["ENGRAM_TEST_NO_NOTIFY"] = "1"
        app.launchEnvironment["TEST_AUTH_TOKEN"] = "mock-token-\(testUUID)"
        app.launchEnvironment["TEST_SYNC_CHANNEL"] = "test-glow-\(testUUID)"

        // The app reads sync_endpoint from UserDefaults (existing code path).
        // Point it at our mock server.
        app.launchArguments += ["-sync_endpoint", "http://localhost:\(server.port)"]
        app.launchArguments += ["-subscription_status", "active"]
        app.launchArguments += ["-subscription_tier", "premium"]

        app.launch()
    }

    // MARK: - Test: Catch-Up Stale Glows (Bug Reproduction)

    /// Reproduces the stale glow bug: IPC catch-up replays old lastAccessedAt changes,
    /// causing mass glow events with high staleness values.
    /// Seeds a realistic multi-project graph to stress the renderer.
    func testCatchUpStaleGlows() throws {
        // Counts mirror the live synced DB distribution (~3,400 total memories)
        let projects: [(name: String, count: Int)] = [
            ("Engram", 900), ("Lattice", 450), ("ClaudeMemory", 200),
            ("engram-server", 120), ("sidescroller", 100), ("global", 200),
            ("LatticeCore", 40), ("SwiftLM", 25)
        ]

        // Seed local DB: multi-project, lastAccessedAt = 1hr ago
        let localIds = seedMultiProjectDatabase(
            at: localDbPath,
            projects: projects,
            staleAge: 3600,
            withSyncConfig: ["Engram", "Lattice", "global"]
        )

        // Seed synced DB: same projects (synced ones), lastAccessedAt = 2hrs ago (staler)
        seedMultiProjectDatabase(
            at: syncedDbPath,
            projects: [("Engram", 900), ("Lattice", 450), ("global", 200)],
            staleAge: 7200
        )

        // Launch app — IPC catch-up pushes newer timestamps from local → synced
        launchApp()
        sleep(20) // wait for load + IPC catch-up + glow processing (~2,000 node graph)

        // Take screenshot for visual inspection
        let screenshot = XCUIScreen.main.screenshot()
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = "stale-glow-catchup"
        attachment.lifetime = .keepAlways
        add(attachment)

        // Analyze glow log
        let events = parseGlowLog()
        print("\n╔════════════════════════════════════════════════════════════════╗")
        print("║          CATCH-UP STALE GLOW ANALYSIS                        ║")
        print("╠════════════════════════════════════════════════════════════════╣")
        print("  Total glow events: \(events.count)")
        print("  Node count: \(localIds.count) local, \(projects.filter { ["Engram","Lattice","global"].contains($0.name) }.map(\.count).reduce(0,+)) synced")

        let staleEvents = events.filter { $0.stalenessSeconds > 10 }
        let liveEvents = events.filter { $0.stalenessSeconds <= 10 }
        print("  Stale (> 10s): \(staleEvents.count)")
        print("  Live (<= 10s): \(liveEvents.count)")

        if !events.isEmpty {
            let avgStaleness = events.map(\.stalenessSeconds).reduce(0, +) / Double(events.count)
            let maxStaleness = events.map(\.stalenessSeconds).max() ?? 0
            print("  Avg staleness: \(String(format: "%.1f", avgStaleness))s")
            print("  Max staleness: \(String(format: "%.1f", maxStaleness))s")
        }

        // Per-galaxy breakdown
        var galaxyCounts: [String: Int] = [:]
        for event in events { galaxyCounts[event.galaxy, default: 0] += 1 }
        for (galaxy, count) in galaxyCounts.sorted(by: { $0.key < $1.key }) {
            print("  \(galaxy): \(count) glow events")
        }

        for event in events.prefix(20) {
            print("  \(event.galaxy)  \(event.nodeLabel.prefix(25))  stale=\(String(format: "%.1f", event.stalenessSeconds))s  refire=\(event.alreadyGlowing)")
        }
        print("╚════════════════════════════════════════════════════════════════╝\n")

        // Analyze frame timing for performance regression
        let timingPath = "/tmp/metal-frame-timing.csv"
        if let data = FileManager.default.contents(atPath: timingPath),
           let csv = String(data: data, encoding: .utf8) {
            let lines = csv.components(separatedBy: "\n").dropFirst().filter { !$0.isEmpty }
            let totals = lines.compactMap { line -> Double? in
                let cols = line.components(separatedBy: ",")
                guard cols.count >= 4 else { return nil }
                return Double(cols[3])
            }
            if !totals.isEmpty {
                let sorted = totals.sorted()
                let p50 = sorted[Int(Double(sorted.count) * 0.5)]
                let p95 = sorted[Int(Double(sorted.count) * 0.95)]
                let worst = sorted.last!
                print("  Frame timing: p50=\(String(format: "%.1f", p50))ms  p95=\(String(format: "%.1f", p95))ms  worst=\(String(format: "%.1f", worst))ms  (\(lines.count) frames)")
                XCTAssertLessThan(p95, 33.0, "p95 frame time should be under 33ms (30fps) with \(localIds.count) nodes")
            }
        }

        XCTAssertFalse(localIds.isEmpty, "Seed should have produced globalIds")
    }

    // MARK: - Test: Live Recall Glow (Correct Behavior)

    /// Verifies that a live recall triggers the expected number of glows
    /// with low staleness and no re-fires.
    /// Seeds a realistic multi-project graph.
    func testLiveRecallGlow() throws {
        // Counts mirror the live synced DB distribution (~3,400 total memories)
        let projects: [(name: String, count: Int)] = [
            ("Engram", 900), ("Lattice", 450), ("ClaudeMemory", 200),
            ("engram-server", 120), ("global", 200)
        ]

        // Seed local DB: multi-project, lastAccessedAt = 1hr ago (old enough to not glow)
        let localIds = seedMultiProjectDatabase(
            at: localDbPath,
            projects: projects,
            staleAge: 3600  // old enough that initial load won't trigger glows
        )

        launchApp()
        sleep(20) // wait for full load + simulation settle (~1,870 nodes)

        // Clean glow log AFTER initial load to isolate live recalls
        try? FileManager.default.removeItem(atPath: "/tmp/glow-log.csv")
        sleep(1) // ensure glow timer has cleaned up any stale entries

        // Trigger recall on 2 specific Engram memories from the test process
        let engramIds = Array(localIds.prefix(2))
        triggerRecall(at: localDbPath, globalIds: engramIds)

        sleep(5) // wait for xproc notification → observer → glow

        let screenshot = XCUIScreen.main.screenshot()
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = "live-recall-glow"
        attachment.lifetime = .keepAlways
        add(attachment)

        let events = parseGlowLog()
        print("\n╔════════════════════════════════════════════════════════════════╗")
        print("║          LIVE RECALL GLOW ANALYSIS                            ║")
        print("╠════════════════════════════════════════════════════════════════╣")
        print("  Total glow events: \(events.count)")
        print("  Total nodes in graph: \(localIds.count)")

        for event in events {
            print("  \(event.galaxy)  \(event.nodeLabel.prefix(25))  stale=\(String(format: "%.1f", event.stalenessSeconds))s  refire=\(event.alreadyGlowing)")
        }

        let refires = events.filter(\.alreadyGlowing)
        let stale = events.filter { $0.stalenessSeconds > 5 }
        print("  Re-fires: \(refires.count)")
        print("  Stale (> 5s): \(stale.count)")
        print("╚════════════════════════════════════════════════════════════════╝\n")

        // Assertions
        XCTAssertEqual(events.count, 2, "Expected exactly 2 glow events for 2 recalled memories")
        XCTAssertEqual(stale.count, 0, "Live recalls should have staleness < 5s")
    }

    // MARK: - Test: lastAccessedAt Flicker Loop

    /// Detects the infinite glow flicker bug: after a single recall trigger,
    /// glow events should fire once per node and stop. If the observer or IPC
    /// relay keeps re-triggering, glow events accumulate indefinitely.
    func testLastAccessedFlickerLoop() throws {
        let projects: [(name: String, count: Int)] = [
            ("Engram", 40), ("Lattice", 30), ("global", 20)
        ]

        // Seed local DB: lastAccessedAt = 2hrs ago (well past glow window)
        let localIds = seedMultiProjectDatabase(
            at: localDbPath,
            projects: projects,
            staleAge: 7200
        )

        launchApp()
        sleep(12) // wait for full load + simulation settle

        // Clean glow log AFTER initial load to isolate our trigger
        try? FileManager.default.removeItem(atPath: "/tmp/glow-log.csv")
        sleep(1)

        // Trigger recall on exactly 3 Engram nodes
        let targetIds = Array(localIds.prefix(3))
        triggerRecall(at: localDbPath, globalIds: targetIds)

        // Sample 1: shortly after trigger — should see exactly 3 glow events
        sleep(5)
        let events1 = parseGlowLog()
        let count1 = events1.count

        // Sample 2: wait another 10s — if looping, count keeps growing
        sleep(10)
        let events2 = parseGlowLog()
        let count2 = events2.count

        // Sample 3: wait another 15s — definitive proof of infinite loop
        sleep(15)
        let events3 = parseGlowLog()
        let count3 = events3.count

        let screenshot = XCUIScreen.main.screenshot()
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = "flicker-loop-final"
        attachment.lifetime = .keepAlways
        add(attachment)

        // Detailed analysis
        print("\n╔════════════════════════════════════════════════════════════════╗")
        print("║          LAST-ACCESSED FLICKER LOOP ANALYSIS                  ║")
        print("╠════════════════════════════════════════════════════════════════╣")
        print("  Triggered recall on \(targetIds.count) nodes")
        print("  Sample 1 (t+5s):  \(count1) glow events")
        print("  Sample 2 (t+15s): \(count2) glow events")
        print("  Sample 3 (t+30s): \(count3) glow events")
        print("  Growth rate: \(count2 > count1 ? "+\(count2 - count1) between s1→s2" : "stable s1→s2"), "
            + "\(count3 > count2 ? "+\(count3 - count2) between s2→s3" : "stable s2→s3")")

        // Per-node breakdown — which nodes are re-firing?
        var nodeEventCounts: [String: Int] = [:]
        var nodeTimestamps: [String: [Double]] = [:]
        for event in events3 {
            let key = event.nodeLabel
            nodeEventCounts[key, default: 0] += 1
            nodeTimestamps[key, default: []].append(event.timestamp)
        }

        let repeaters = nodeEventCounts.filter { $0.value > 1 }.sorted { $0.value > $1.value }
        if !repeaters.isEmpty {
            print("\n  REPEATING NODES (evidence of flicker loop):")
            for (label, count) in repeaters.prefix(10) {
                let timestamps = nodeTimestamps[label] ?? []
                // Compute intervals between consecutive firings
                var intervals: [Double] = []
                for i in 1..<timestamps.count {
                    intervals.append(timestamps[i] - timestamps[i-1])
                }
                let avgInterval = intervals.isEmpty ? 0 : intervals.reduce(0, +) / Double(intervals.count)
                let minInterval = intervals.min() ?? 0
                print("    \(label.prefix(30)): \(count) events, avg_interval=\(String(format: "%.2f", avgInterval))s, min_interval=\(String(format: "%.2f", minInterval))s")
            }
        }

        // Re-fire analysis
        let refires = events3.filter(\.alreadyGlowing)
        print("\n  Re-fires (glow reset while already glowing): \(refires.count) / \(events3.count)")
        if !refires.isEmpty {
            print("  Re-fire timestamps:")
            for event in refires.prefix(20) {
                print("    t=\(String(format: "%.3f", event.timestamp)) \(event.galaxy) \(event.nodeLabel.prefix(25)) stale=\(String(format: "%.1f", event.stalenessSeconds))s")
            }
        }

        // Galaxy breakdown
        var galaxyCounts: [String: Int] = [:]
        for event in events3 { galaxyCounts[event.galaxy, default: 0] += 1 }
        print("\n  Per-galaxy:")
        for (galaxy, count) in galaxyCounts.sorted(by: { $0.key < $1.key }) {
            print("    \(galaxy): \(count)")
        }
        print("╚════════════════════════════════════════════════════════════════╝\n")

        // Frame timing around the glow period
        analyzeMetalFrameTimingForFlicker()

        // Assertions
        // After a single trigger of 3 nodes, we expect at most 3 glow events
        // (one per node). Allow small margin for race conditions.
        XCTAssertLessThanOrEqual(count1, 6,
            "Sample 1: expected ~3 glow events for 3 recalled nodes, got \(count1)")

        // The count should NOT keep growing — that's the infinite loop
        XCTAssertEqual(count3, count2,
            "Glow events still accumulating between s2 and s3 — infinite flicker loop detected! "
            + "s2=\(count2) s3=\(count3) delta=\(count3 - count2)")

        XCTAssertEqual(count2, count1,
            "Glow events accumulated between s1 and s2 — flicker re-trigger detected! "
            + "s1=\(count1) s2=\(count2) delta=\(count2 - count1)")
    }

    // MARK: - Test: lastAccessedAt Flicker Loop WITH Sync

    /// Same as testLastAccessedFlickerLoop but with sync enabled.
    /// The IPC relay may propagate lastAccessedAt between personal and synced
    /// galaxies, creating a feedback loop.
    func testLastAccessedFlickerLoopWithSync() throws {
        let projects: [(name: String, count: Int)] = [
            ("Engram", 40), ("Lattice", 30), ("global", 20)
        ]

        // Seed local DB: lastAccessedAt = 2hrs ago, WITH sync config
        let localIds = seedMultiProjectDatabase(
            at: localDbPath,
            projects: projects,
            staleAge: 7200,
            withSyncConfig: ["Engram", "global"]
        )

        // Seed synced DB: synced projects only, lastAccessedAt = 3hrs ago (staler)
        seedMultiProjectDatabase(
            at: syncedDbPath,
            projects: [("Engram", 40), ("global", 20)],
            staleAge: 10800
        )

        launchApp()
        sleep(15) // longer settle for sync + IPC catch-up

        // Clean glow log AFTER initial load + catch-up to isolate our trigger
        try? FileManager.default.removeItem(atPath: "/tmp/glow-log.csv")
        sleep(2)

        // Trigger recall on exactly 3 Engram nodes
        let targetIds = Array(localIds.prefix(3))
        triggerRecall(at: localDbPath, globalIds: targetIds)

        // Sample at multiple intervals to detect infinite loop
        sleep(5)
        let events1 = parseGlowLog()
        let count1 = events1.count

        sleep(10)
        let events2 = parseGlowLog()
        let count2 = events2.count

        sleep(15)
        let events3 = parseGlowLog()
        let count3 = events3.count

        let screenshot = XCUIScreen.main.screenshot()
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = "flicker-loop-sync"
        attachment.lifetime = .keepAlways
        add(attachment)

        print("\n╔════════════════════════════════════════════════════════════════╗")
        print("║      LAST-ACCESSED FLICKER LOOP (WITH SYNC) ANALYSIS         ║")
        print("╠════════════════════════════════════════════════════════════════╣")
        print("  Triggered recall on \(targetIds.count) nodes (sync enabled)")
        print("  Sample 1 (t+5s):  \(count1) glow events")
        print("  Sample 2 (t+15s): \(count2) glow events")
        print("  Sample 3 (t+30s): \(count3) glow events")
        print("  Growth rate: \(count2 > count1 ? "+\(count2 - count1) between s1→s2" : "stable s1→s2"), "
            + "\(count3 > count2 ? "+\(count3 - count2) between s2→s3" : "stable s2→s3")")

        // Per-node breakdown
        var nodeEventCounts: [String: Int] = [:]
        var nodeTimestamps: [String: [Double]] = [:]
        for event in events3 {
            nodeEventCounts[event.nodeLabel, default: 0] += 1
            nodeTimestamps[event.nodeLabel, default: []].append(event.timestamp)
        }

        let repeaters = nodeEventCounts.filter { $0.value > 1 }.sorted { $0.value > $1.value }
        if !repeaters.isEmpty {
            print("\n  REPEATING NODES (evidence of flicker loop):")
            for (label, count) in repeaters.prefix(10) {
                let timestamps = nodeTimestamps[label] ?? []
                var intervals: [Double] = []
                for i in 1..<timestamps.count {
                    intervals.append(timestamps[i] - timestamps[i-1])
                }
                let avgInterval = intervals.isEmpty ? 0 : intervals.reduce(0, +) / Double(intervals.count)
                let minInterval = intervals.min() ?? 0
                print("    \(label.prefix(30)): \(count) events, avg_interval=\(String(format: "%.2f", avgInterval))s, min_interval=\(String(format: "%.2f", minInterval))s")
            }
        }

        // Galaxy breakdown — key indicator: if both personal and synced galaxy
        // show glows for the same nodes, the IPC relay is bouncing lastAccessedAt
        var galaxyCounts: [String: Int] = [:]
        var galaxyNodes: [String: Set<String>] = [:]
        for event in events3 {
            galaxyCounts[event.galaxy, default: 0] += 1
            galaxyNodes[event.galaxy, default: []].insert(event.nodeLabel)
        }
        print("\n  Per-galaxy:")
        for (galaxy, count) in galaxyCounts.sorted(by: { $0.key < $1.key }) {
            let nodes = galaxyNodes[galaxy] ?? []
            print("    \(galaxy): \(count) events, \(nodes.count) unique nodes")
        }

        // Cross-galaxy overlap — same node glowing in multiple galaxies?
        let allGalaxyNodeSets = Array(galaxyNodes.values)
        if allGalaxyNodeSets.count >= 2 {
            let overlap = allGalaxyNodeSets[0].intersection(allGalaxyNodeSets[1])
            if !overlap.isEmpty {
                print("  ⚠️ CROSS-GALAXY OVERLAP: \(overlap.count) nodes glow in BOTH galaxies")
                print("     This means the IPC relay is propagating lastAccessedAt bidirectionally!")
                for label in overlap.prefix(5) {
                    print("     - \(label)")
                }
            }
        }

        let refires = events3.filter(\.alreadyGlowing)
        print("\n  Re-fires: \(refires.count) / \(events3.count)")
        print("╚════════════════════════════════════════════════════════════════╝\n")

        analyzeMetalFrameTimingForFlicker()

        // Assertions — with sync, we expect at most 6 events (3 per galaxy × 2 galaxies)
        // but NOT a growing count
        XCTAssertEqual(count3, count2,
            "WITH SYNC: Glow events still accumulating between s2 and s3 — infinite flicker loop! "
            + "s2=\(count2) s3=\(count3) delta=\(count3 - count2)")
    }

    // MARK: - Flicker Frame Timing

    private func analyzeMetalFrameTimingForFlicker() {
        let csvPath = "/tmp/metal-frame-timing.csv"
        guard let csvData = FileManager.default.contents(atPath: csvPath),
              let csv = String(data: csvData, encoding: .utf8) else {
            print("        [No Metal frame timing data]")
            return
        }

        let lines = csv.components(separatedBy: "\n").dropFirst().filter { !$0.isEmpty }
        guard !lines.isEmpty else { return }

        // Count frames by reason to see how many are "glow" vs "sim"
        var reasonCounts: [String: Int] = [:]
        var glowFrameTotals: [Double] = []
        for line in lines {
            let cols = line.components(separatedBy: ",")
            guard cols.count >= 14 else { continue }
            let reason = cols[13].trimmingCharacters(in: .whitespacesAndNewlines)
            reasonCounts[reason, default: 0] += 1
            if reason == "glow", let total = Double(cols[3]) {
                glowFrameTotals.append(total)
            }
        }

        print("        ╔════════════════════════════════════════════════════════════════╗")
        print("        ║           FLICKER FRAME ANALYSIS                              ║")
        print("        ╠════════════════════════════════════════════════════════════════╣")
        print("        Total frames: \(lines.count)")
        for (reason, count) in reasonCounts.sorted(by: { $0.value > $1.value }) {
            let pct = Double(count) / Double(lines.count) * 100
            print("          \(reason): \(count) (\(String(format: "%.0f", pct))%)")
        }
        if !glowFrameTotals.isEmpty {
            let sorted = glowFrameTotals.sorted()
            let p50 = sorted[Int(Double(sorted.count) * 0.5)]
            let p95 = sorted[min(Int(Double(sorted.count) * 0.95), sorted.count - 1)]
            print("        Glow frame timing: p50=\(String(format: "%.2f", p50))ms p95=\(String(format: "%.2f", p95))ms count=\(glowFrameTotals.count)")
        }
        // If glow frames dominate after glows should have expired (~4.5s after trigger),
        // that's evidence the glowingNodes dict is never emptying.
        let glowPct = Double(reasonCounts["glow", default: 0]) / Double(max(1, lines.count))
        if glowPct > 0.5 {
            print("        ⚠️ GLOW frames are \(String(format: "%.0f", glowPct * 100))% of all frames — glowingNodes never emptied!")
        }
        print("        ╚════════════════════════════════════════════════════════════════╝\n")
    }
}
