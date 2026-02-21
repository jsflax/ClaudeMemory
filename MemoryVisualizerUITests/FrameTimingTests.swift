import XCTest

@MainActor
final class FrameTimingTests: XCTestCase {
    let app = XCUIApplication()

    override func setUpWithError() throws {
        continueAfterFailure = false
        app.launch()
        sleep(5)
    }

    func testCapture3DFrameTiming() throws {
        // Switch to 3D mode
        let threeDButton = app.buttons.matching(NSPredicate(format: "label == '3D'")).firstMatch
        XCTAssertTrue(threeDButton.waitForExistence(timeout: 10), "3D button not found")
        threeDButton.tap()
        sleep(5) // let 3D view initialize, force sim settle, and nebulae start

        let window = app.windows.firstMatch
        XCTAssertTrue(window.exists)

        // Use window center as the main interaction point
        let center = window.coordinate(withNormalizedOffset: CGVector(dx: 0.4, dy: 0.5))

        // Phase 1: Idle baseline (graph should be visible and centered)
        sleep(3)
        takeScreenshot(name: "01-idle-baseline")

        // Phase 2: Dolly IN toward the graph (scroll up = zoom in on macOS natural scroll)
        // XCUITest scroll deltaY: positive = scroll content down = zoom out with natural scroll
        // So negative deltaY = zoom in
        for _ in 0..<20 {
            center.scroll(byDeltaX: 0, deltaY: -8)
            usleep(80_000)
        }
        sleep(2)
        takeScreenshot(name: "02-zoomed-in")

        // Phase 3: Gentle orbit while zoomed in (small drags to stay on-graph)
        let slightRight = window.coordinate(withNormalizedOffset: CGVector(dx: 0.55, dy: 0.5))
        let slightLeft = window.coordinate(withNormalizedOffset: CGVector(dx: 0.25, dy: 0.5))
        let slightUp = window.coordinate(withNormalizedOffset: CGVector(dx: 0.4, dy: 0.4))
        let slightDown = window.coordinate(withNormalizedOffset: CGVector(dx: 0.4, dy: 0.6))

        // Orbit right while close
        center.press(forDuration: 0.05, thenDragTo: slightRight, withVelocity: .slow, thenHoldForDuration: 0.05)
        sleep(1)
        // Orbit left
        center.press(forDuration: 0.05, thenDragTo: slightLeft, withVelocity: .slow, thenHoldForDuration: 0.05)
        sleep(1)
        // Orbit up
        center.press(forDuration: 0.05, thenDragTo: slightUp, withVelocity: .slow, thenHoldForDuration: 0.05)
        sleep(1)
        takeScreenshot(name: "03-orbited-close")

        // Phase 4: Click near center to select a node
        center.tap()
        sleep(3)
        takeScreenshot(name: "04-node-selected")

        // Phase 5: Orbit with selection active
        center.press(forDuration: 0.05, thenDragTo: slightRight, withVelocity: .slow, thenHoldForDuration: 0.05)
        sleep(2)

        // Phase 6: Deselect
        let corner = window.coordinate(withNormalizedOffset: CGVector(dx: 0.05, dy: 0.05))
        corner.tap()
        sleep(2)

        // Phase 7: Fast orbit (stress test)
        let farRight = window.coordinate(withNormalizedOffset: CGVector(dx: 0.7, dy: 0.5))
        center.press(forDuration: 0.05, thenDragTo: farRight, withVelocity: .fast, thenHoldForDuration: 0.05)
        sleep(2)

        // Phase 8: Orbit down and zoom back out
        center.press(forDuration: 0.05, thenDragTo: slightDown, withVelocity: .slow, thenHoldForDuration: 0.05)
        sleep(1)
        for _ in 0..<20 {
            center.scroll(byDeltaX: 0, deltaY: 8)  // zoom out
            usleep(80_000)
        }
        sleep(2)
        takeScreenshot(name: "05-zoomed-out")

        // Read and analyze profiling CSV
        analyzeProfilingData()
    }

    // MARK: - Helpers

    private func takeScreenshot(name: String) {
        let screenshot = XCUIScreen.main.screenshot()
        let attachment = XCTAttachment(screenshot: screenshot)
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
        // Also save to /tmp for easy viewing
        let data = screenshot.pngRepresentation
        try? data.write(to: URL(fileURLWithPath: "/tmp/\(name).png"))
    }

    private func analyzeProfilingData() {
        let csvPath = "/tmp/frame-timing.csv"
        guard let csvData = FileManager.default.contents(atPath: csvPath),
              let csv = String(data: csvData, encoding: .utf8) else {
            XCTFail("No profiling data at \(csvPath)")
            return
        }

        let lines = csv.components(separatedBy: "\n").dropFirst().filter { !$0.isEmpty }
        guard !lines.isEmpty else { XCTFail("No frame data collected"); return }

        var dts: [Double] = [], works: [Double] = [], repos: [Double] = []
        var nodes: [Double] = [], edges: [Double] = [], nebs: [Double] = [], labels: [Double] = []
        var slowFrames = 0
        var worstDt = 0.0
        var worstWork = 0.0

        for line in lines {
            let cols = line.components(separatedBy: ",")
            guard cols.count >= 10 else { continue }
            let dt = Double(cols[1]) ?? 0
            let work = Double(cols[2]) ?? 0
            let repo = Double(cols[3]) ?? 0
            let node = Double(cols[4]) ?? 0
            let edge = Double(cols[5]) ?? 0
            let neb = Double(cols[6]) ?? 0
            let label = Double(cols[9]) ?? 0

            dts.append(dt)
            works.append(work)
            repos.append(repo)
            nodes.append(node)
            edges.append(edge)
            nebs.append(neb)
            labels.append(label)
            if dt > 20 { slowFrames += 1 }
            worstDt = max(worstDt, dt)
            worstWork = max(worstWork, work)
        }

        let count = Double(dts.count)
        let avgDt = dts.reduce(0, +) / count
        let avgWork = works.reduce(0, +) / count
        let avgRepos = repos.reduce(0, +) / count
        let avgNodes = nodes.reduce(0, +) / count
        let avgEdges = edges.reduce(0, +) / count
        let avgNebs = nebs.reduce(0, +) / count
        let avgLabels = labels.reduce(0, +) / count

        let p95Dt = dts.sorted()[Int(count * 0.95)]
        let p95Work = works.sorted()[Int(count * 0.95)]

        print("""

        ╔══════════════════════════════════════════════════════╗
        ║           3D FRAME TIMING PROFILE RESULTS           ║
        ╠══════════════════════════════════════════════════════╣
        ║ Total frames: \(String(format: "%6d", dts.count))                                ║
        ║ Slow frames (>20ms): \(String(format: "%6d", slowFrames)) (\(String(format: "%.1f", Double(slowFrames)/count*100))%)               ║
        ╠══════════════════════════════════════════════════════╣
        ║ Frame dt (ms):  avg=\(String(format: "%6.1f", avgDt))  p95=\(String(format: "%6.1f", p95Dt))  worst=\(String(format: "%6.1f", worstDt)) ║
        ║ Work time(ms):  avg=\(String(format: "%6.2f", avgWork))  p95=\(String(format: "%6.2f", p95Work))  worst=\(String(format: "%6.2f", worstWork)) ║
        ╠══════════════════════════════════════════════════════╣
        ║ Sub-phase averages (ms):                             ║
        ║   repos (position+fog): \(String(format: "%7.3f", avgRepos))                       ║
        ║   nodes (material+fx):  \(String(format: "%7.3f", avgNodes))                       ║
        ║   edges (geom+opacity): \(String(format: "%7.3f", avgEdges))                       ║
        ║   nebulae:              \(String(format: "%7.3f", avgNebs))                       ║
        ║   canvas labels:        \(String(format: "%7.3f", avgLabels))                       ║
        ╚══════════════════════════════════════════════════════╝

        """)
    }
}
