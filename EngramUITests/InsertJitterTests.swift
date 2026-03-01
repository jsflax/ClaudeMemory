import XCTest

/// Tests 3D insert jitter by launching the app with a test-insert flag,
/// waiting for settle, letting the app insert a memory on our behalf,
/// then analyzing the Metal frame timing CSV for spikes.
@MainActor
final class InsertJitterTests: XCTestCase {
    let app = XCUIApplication()

    // macOS virtual key codes
    private enum VK {
        static let w: CGKeyCode = 13
        static let s: CGKeyCode = 1
        static let q: CGKeyCode = 12
        static let e: CGKeyCode = 14
        static let i: CGKeyCode = 34
        static let k: CGKeyCode = 40
        static let t: CGKeyCode = 17
    }

    override func setUpWithError() throws {
        continueAfterFailure = false
        // Clean previous timing data
        try? FileManager.default.removeItem(atPath: "/tmp/metal-frame-timing.csv")
        try? FileManager.default.removeItem(atPath: "/tmp/flush-timing.csv")
        try? FileManager.default.removeItem(atPath: "/tmp/draw-timing.csv")
        try? FileManager.default.removeItem(atPath: "/tmp/atlas-timing.log")
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
    }

    // MARK: - Screenshot Helper

    private func takeScreenshot(name: String) {
        let screenshot = XCUIScreen.main.screenshot()
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
        try? screenshot.pngRepresentation.write(to: URL(fileURLWithPath: "/tmp/\(name).png"))
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
        // frame,dt_ms,wall_dt_ms,total_ms,sim_ms,mascot_ms,nodes_ms,edges_ms,neb_ms,labels_ms,flow_ms,node_count,edge_count,reason

        var frames: [(frame: Int, dt: Double, wallDt: Double, total: Double, sim: Double, mascot: Double,
                       nodes: Double, edges: Double, neb: Double, labels: Double, flow: Double,
                       nodeCount: Int, edgeCount: Int, reason: String)] = []

        for line in lines {
            let cols = line.components(separatedBy: ",")
            guard cols.count >= 14 else { continue }
            frames.append((
                frame: Int(cols[0]) ?? 0,
                dt: Double(cols[1]) ?? 0,
                wallDt: Double(cols[2]) ?? 0,
                total: Double(cols[3]) ?? 0,
                sim: Double(cols[4]) ?? 0,
                mascot: Double(cols[5]) ?? 0,
                nodes: Double(cols[6]) ?? 0,
                edges: Double(cols[7]) ?? 0,
                neb: Double(cols[8]) ?? 0,
                labels: Double(cols[9]) ?? 0,
                flow: Double(cols[10]) ?? 0,
                nodeCount: Int(cols[11]) ?? 0,
                edgeCount: Int(cols[12]) ?? 0,
                reason: cols[13].trimmingCharacters(in: .whitespacesAndNewlines)
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

    // MARK: - Key Holding Helper

    private nonisolated func holdKey(_ keyCode: CGKeyCode, duration: TimeInterval) {
        let src = CGEventSource(stateID: .combinedSessionState)
        let down = CGEvent(keyboardEventSource: src, virtualKey: keyCode, keyDown: true)
        let up = CGEvent(keyboardEventSource: src, virtualKey: keyCode, keyDown: false)
        down?.post(tap: .cgSessionEventTap)
        Thread.sleep(forTimeInterval: duration)
        up?.post(tap: .cgSessionEventTap)
        usleep(50_000) // brief gap between inputs
    }

    private func percentile(_ sorted: [Double], _ p: Double) -> Double {
        guard !sorted.isEmpty else { return 0 }
        let idx = min(Int(Double(sorted.count) * p), sorted.count - 1)
        return sorted[idx]
    }
}
