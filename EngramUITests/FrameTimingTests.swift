import XCTest
import CoreGraphics

@MainActor
final class FrameTimingTests: XCTestCase {
    let app = XCUIApplication()

    // macOS virtual key codes for WASD/IJKL/QE/TR controls
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
        static let r: CGKeyCode = 15
        static let shift: CGKeyCode = 56
    }

    override func setUpWithError() throws {
        continueAfterFailure = false
        app.launch()
        // Wait for app to launch, load data, and force simulation to settle
        sleep(5)
    }

    // MARK: - Frame Timing Test

    func testCapture3DFrameTiming() throws {
        // 3D mode is the default (config.dimensionMode = .threeD) — no button click needed
        let window = app.windows.firstMatch
        XCTAssertTrue(window.exists, "App window not found")

        // Phase 1: Idle baseline — graph should be visible and centered
        sleep(3)
        takeScreenshot(name: "01-idle-baseline")

        // Phase 2: Dolly forward using W key
        holdKey(VK.w, duration: 1.5)
        sleep(1)
        takeScreenshot(name: "02-dolly-forward")

        // Phase 3: Look around using IJKL
        holdKey(VK.j, duration: 0.8)  // look left
        usleep(300_000)
        holdKey(VK.l, duration: 0.8)  // look right
        usleep(300_000)
        holdKey(VK.i, duration: 0.5)  // look up
        usleep(300_000)
        takeScreenshot(name: "03-looked-around")

        // Phase 4: Strafe left/right using A/D
        holdKey(VK.d, duration: 1.0)  // strafe right
        usleep(300_000)
        holdKey(VK.a, duration: 1.0)  // strafe left
        usleep(300_000)
        takeScreenshot(name: "04-strafed")

        // Phase 5: Rise and descend using Q/E
        holdKey(VK.e, duration: 0.8)  // rise
        usleep(300_000)
        holdKey(VK.q, duration: 0.8)  // descend
        usleep(300_000)
        takeScreenshot(name: "05-vertical")

        // Phase 6: Sprint movement (Shift + W)
        holdKeys([VK.shift, VK.w], duration: 1.0)
        sleep(1)
        takeScreenshot(name: "06-sprint-forward")

        // Phase 7: Teleport to next project
        window.typeKey("t", modifierFlags: [])
        sleep(2)
        takeScreenshot(name: "07-teleport")

        // Phase 8: Select a node (click center of window)
        let center = window.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.5))
        center.tap()
        sleep(2)
        takeScreenshot(name: "08-node-selected")

        // Phase 9: Move while node is selected (stress: selection + edges + movement)
        holdKey(VK.w, duration: 1.0)
        usleep(500_000)
        holdKey(VK.j, duration: 0.5)  // look left with selection active
        sleep(1)
        takeScreenshot(name: "09-move-with-selection")

        // Phase 10: Deselect and pull back
        let corner = window.coordinate(withNormalizedOffset: CGVector(dx: 0.05, dy: 0.05))
        corner.tap()
        sleep(1)
        holdKey(VK.s, duration: 2.0)  // pull back
        sleep(2)
        takeScreenshot(name: "10-pulled-back")

        // Read and analyze profiling CSV
        analyzeProfilingData()
    }

    // MARK: - Teleport Proximity Test

    func testTeleportProximity() throws {
        // 3D mode is the default — no button click needed
        let window = app.windows.firstMatch
        XCTAssertTrue(window.exists)
        sleep(5) // let force sim settle

        // Clean up any previous teleport log
        try? FileManager.default.removeItem(atPath: "/tmp/teleport-log.csv")

        takeScreenshot(name: "teleport-01-initial")

        // Teleport through projects using T key
        let teleportCount = 5
        for i in 0..<teleportCount {
            window.typeKey("t", modifierFlags: [])
            sleep(2) // let camera settle
            takeScreenshot(name: "teleport-\(String(format: "%02d", i + 2))-after-t\(i + 1)")
        }

        // Rapid teleport test (label stomping)
        for _ in 0..<3 {
            window.typeKey("t", modifierFlags: [])
            usleep(300_000) // 300ms between — should NOT stomp previous label
        }
        sleep(2)
        takeScreenshot(name: "teleport-rapid-final")

        // Verify teleport log
        analyzeTeleportLog(expectedMinTeleports: teleportCount)
    }

    // MARK: - Key Holding Helpers

    /// Hold a single key for a specified duration using CGEvent.
    /// CGEvent posts directly to the session event tap, so the app receives
    /// a real keyDown followed by keyUp with the key held for `duration` seconds.
    private nonisolated func holdKey(_ keyCode: CGKeyCode, duration: TimeInterval) {
        let src = CGEventSource(stateID: .combinedSessionState)
        let down = CGEvent(keyboardEventSource: src, virtualKey: keyCode, keyDown: true)
        let up = CGEvent(keyboardEventSource: src, virtualKey: keyCode, keyDown: false)
        down?.post(tap: .cgSessionEventTap)
        Thread.sleep(forTimeInterval: duration)
        up?.post(tap: .cgSessionEventTap)
        Thread.sleep(forTimeInterval: 0.05) // brief settle after release
    }

    /// Hold multiple keys simultaneously for a specified duration.
    /// Useful for Shift+W (sprint), etc.
    private nonisolated func holdKeys(_ keyCodes: [CGKeyCode], duration: TimeInterval) {
        let src = CGEventSource(stateID: .combinedSessionState)
        // Press all keys
        for kc in keyCodes {
            CGEvent(keyboardEventSource: src, virtualKey: kc, keyDown: true)?
                .post(tap: .cgSessionEventTap)
        }
        Thread.sleep(forTimeInterval: duration)
        // Release all keys (reverse order)
        for kc in keyCodes.reversed() {
            CGEvent(keyboardEventSource: src, virtualKey: kc, keyDown: false)?
                .post(tap: .cgSessionEventTap)
        }
        Thread.sleep(forTimeInterval: 0.05)
    }

    // MARK: - Teleport Log Analysis

    private func analyzeTeleportLog(expectedMinTeleports: Int) {
        let logPath = "/tmp/teleport-log.csv"
        guard let data = FileManager.default.contents(atPath: logPath),
              let csv = String(data: data, encoding: .utf8) else {
            XCTFail("No teleport log at \(logPath)")
            return
        }

        let lines = csv.components(separatedBy: "\n").dropFirst().filter { !$0.isEmpty }
        XCTAssertGreaterThanOrEqual(lines.count, expectedMinTeleports,
            "Expected at least \(expectedMinTeleports) teleports, got \(lines.count)")

        print("\n╔══════════════════════════════════════════════════════╗")
        print("║           TELEPORT PROXIMITY RESULTS                ║")
        print("╠══════════════════════════════════════════════════════╣")

        for line in lines {
            let cols = line.components(separatedBy: ",")
            guard cols.count >= 7 else { continue }
            let project = cols[0]
            let orbitRadius = Double(cols[4]) ?? 0
            let rkDistance = Double(cols[5]) ?? 0
            let nodeCount = Int(cols[6]) ?? 0

            let maxAllowedDistance: Double = 2.0
            XCTAssertLessThan(rkDistance, maxAllowedDistance,
                "Teleport to '\(project)' landed \(String(format: "%.3f", rkDistance)) RK units away (max \(maxAllowedDistance))")

            print("║ \(project.padding(toLength: 20, withPad: " ", startingAt: 0)) nodes=\(String(format: "%3d", nodeCount)) orbit=\(String(format: "%6.0f", orbitRadius)) dist=\(String(format: "%.3f", rkDistance)) ║")
        }

        print("╚══════════════════════════════════════════════════════╝\n")
    }

    // MARK: - Screenshot Helper

    private func takeScreenshot(name: String) {
        let screenshot = XCUIScreen.main.screenshot()
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
        // Also save to /tmp for easy CLI viewing
        let data = screenshot.pngRepresentation
        try? data.write(to: URL(fileURLWithPath: "/tmp/\(name).png"))
    }

    // MARK: - Profiling Data Analysis

    private func analyzeProfilingData() {
        let csvPath = "/tmp/frame-timing.csv"
        guard let csvData = FileManager.default.contents(atPath: csvPath),
              let csv = String(data: csvData, encoding: .utf8) else {
            XCTFail("No profiling data at \(csvPath)")
            return
        }

        let lines = csv.components(separatedBy: "\n").dropFirst().filter { !$0.isEmpty }
        guard !lines.isEmpty else { XCTFail("No frame data collected"); return }

        // Parse CSV columns:
        // frame,dt_ms,work_ms,expand_ms,nodes_ms,edges_ms,neb_ms,node_count,edge_count,labels_ms,flow_ms,idle,entities
        var dts: [Double] = [], works: [Double] = [], expands: [Double] = []
        var nodes: [Double] = [], edges: [Double] = [], nebs: [Double] = [], labels: [Double] = []
        var flows: [Double] = []
        var slowFrames = 0, verySlowFrames = 0
        var worstDt = 0.0, worstWork = 0.0
        var nodeCount = 0, edgeCount = 0, entityCount = 0
        var idleCount = 0

        for line in lines {
            let cols = line.components(separatedBy: ",")
            guard cols.count >= 10 else { continue }
            let dt = Double(cols[1]) ?? 0
            let work = Double(cols[2]) ?? 0
            let expand = Double(cols[3]) ?? 0
            let node = Double(cols[4]) ?? 0
            let edge = Double(cols[5]) ?? 0
            let neb = Double(cols[6]) ?? 0
            let nc = Int(cols[7]) ?? 0
            let ec = Int(cols[8]) ?? 0
            let label = Double(cols[9]) ?? 0
            let flow = cols.count > 10 ? (Double(cols[10]) ?? 0) : 0
            let idle = cols.count > 11 ? (Int(cols[11]) ?? 0) : 0
            let ent = cols.count > 12 ? (Int(cols[12]) ?? 0) : 0

            dts.append(dt)
            works.append(work)
            expands.append(expand)
            nodes.append(node)
            edges.append(edge)
            nebs.append(neb)
            labels.append(label)
            flows.append(flow)
            if dt > 20 { slowFrames += 1 }
            if dt > 50 { verySlowFrames += 1 }
            if idle == 1 { idleCount += 1 }
            worstDt = max(worstDt, dt)
            worstWork = max(worstWork, work)
            nodeCount = max(nodeCount, nc)
            edgeCount = max(edgeCount, ec)
            entityCount = max(entityCount, ent)
        }

        let count = Double(dts.count)
        let avgDt = dts.reduce(0, +) / count
        let avgWork = works.reduce(0, +) / count
        let avgExpand = expands.reduce(0, +) / count
        let avgNodes = nodes.reduce(0, +) / count
        let avgEdges = edges.reduce(0, +) / count
        let avgNebs = nebs.reduce(0, +) / count
        let avgLabels = labels.reduce(0, +) / count
        let avgFlows = flows.reduce(0, +) / count

        let sortedDt = dts.sorted()
        let sortedWork = works.sorted()
        let p50Dt = sortedDt[Int(count * 0.50)]
        let p95Dt = sortedDt[Int(count * 0.95)]
        let p99Dt = sortedDt[min(Int(count * 0.99), sortedDt.count - 1)]
        let p95Work = sortedWork[Int(count * 0.95)]
        let p99Work = sortedWork[min(Int(count * 0.99), sortedWork.count - 1)]

        // Identify active frames only (dt < 100ms, likely not idle-skipped)
        let activeFrames = dts.filter { $0 < 100 && $0 > 0.1 }
        let activeAvgDt = activeFrames.isEmpty ? 0 : activeFrames.reduce(0, +) / Double(activeFrames.count)
        let activeFps = activeAvgDt > 0 ? 1000.0 / activeAvgDt : 0

        print("""

        ╔═══════════════════════════════════════════════════════════════╗
        ║              3D FRAME TIMING PROFILE RESULTS                 ║
        ╠═══════════════════════════════════════════════════════════════╣
        ║ Scene: \(String(format: "%4d", nodeCount)) nodes, \(String(format: "%5d", edgeCount)) edges, \(String(format: "%4d", entityCount)) entities               ║
        ║ Total frames: \(String(format: "%6d", dts.count))   Active: \(String(format: "%6d", dts.count - idleCount))   Idle: \(String(format: "%6d", idleCount))            ║
        ║ Slow (>20ms): \(String(format: "%6d", slowFrames)) (\(String(format: "%5.1f", Double(slowFrames)/count*100))%)                            ║
        ║ Very slow (>50ms): \(String(format: "%4d", verySlowFrames)) (\(String(format: "%5.1f", Double(verySlowFrames)/count*100))%)                            ║
        ╠═══════════════════════════════════════════════════════════════╣
        ║ Frame dt (ms):                                               ║
        ║   avg=\(String(format: "%7.1f", avgDt))  p50=\(String(format: "%7.1f", p50Dt))  p95=\(String(format: "%7.1f", p95Dt))  p99=\(String(format: "%7.1f", p99Dt))  worst=\(String(format: "%7.1f", worstDt)) ║
        ║ Active FPS: \(String(format: "%5.1f", activeFps))                                            ║
        ║ Work time (ms):                                              ║
        ║   avg=\(String(format: "%7.2f", avgWork))  p95=\(String(format: "%7.2f", p95Work))  p99=\(String(format: "%7.2f", p99Work))  worst=\(String(format: "%7.2f", worstWork))          ║
        ╠═══════════════════════════════════════════════════════════════╣
        ║ Sub-phase averages (ms):                                     ║
        ║   expansion:            \(String(format: "%7.3f", avgExpand))                                  ║
        ║   nodes (batch stamp):  \(String(format: "%7.3f", avgNodes))                                  ║
        ║   edges (batch build):  \(String(format: "%7.3f", avgEdges))                                  ║
        ║   flow particles:       \(String(format: "%7.3f", avgFlows))                                  ║
        ║   nebulae:              \(String(format: "%7.3f", avgNebs))                                  ║
        ║   labels (native):      \(String(format: "%7.3f", avgLabels))                                  ║
        ╚═══════════════════════════════════════════════════════════════╝

        """)

        // Reason breakdown: count frames by reason flags
        var reasonCounts: [String: Int] = [:]
        for line in lines {
            let cols = line.components(separatedBy: ",")
            let reason = cols.count > 13 ? cols[13] : "?"
            reasonCounts[reason, default: 0] += 1
        }
        print("        Update reasons:")
        for (reason, count) in reasonCounts.sorted(by: { $0.value > $1.value }) {
            let total = Double(dts.count)
            print("          \(reason): \(count) frames (\(String(format: "%.1f", Double(count)/total*100))%)")
        }
        print("")

        // Dump the worst 20 frames for detailed analysis
        let indexedDts = dts.enumerated().map { ($0.offset, $0.element) }
        let worstFrames = indexedDts.sorted { $0.1 > $1.1 }.prefix(20)
        print("        Worst 20 frames:")
        for (idx, dt) in worstFrames {
            let cols = lines[idx].components(separatedBy: ",")
            let work = cols.count > 2 ? cols[2] : "?"
            let n = cols.count > 4 ? cols[4] : "?"
            let e = cols.count > 5 ? cols[5] : "?"
            let reason = cols.count > 13 ? cols[13] : "?"
            print("          frame=\(cols[0]) dt=\(String(format: "%7.1f", dt))ms work=\(work)ms nodes=\(n)ms edges=\(e)ms reason=\(reason)")
        }
        print("")
    }
}
