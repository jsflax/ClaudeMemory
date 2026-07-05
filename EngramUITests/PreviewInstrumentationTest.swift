import XCTest

/// Launches EngramPreview at multiple node counts, waits for force simulation
/// settlement, then parses instrumentation output (per-frame CSV + cluster report JSON)
/// and prints a structured report for performance and cluster quality analysis.
final class PreviewInstrumentationTest: XCTestCase {
    let nodeCounts = [200, 1000, 5000, 10000, 20000]
    let basePath = "/tmp/preview-instrumentation"

    @MainActor func testPreviewInstrumentation() throws {
        continueAfterFailure = true
        var forceP50s: [(count: Int, ms: Double)] = []

        print("\n" + String(repeating: "=", count: 75))
        print("  PREVIEW INSTRUMENTATION REPORT")
        print(String(repeating: "=", count: 75))

        for count in nodeCounts {
            let logPath = "\(basePath)-\(count)"

            // Clean previous output files
            for suffix in ["-frames.csv", "-final.json", "-settled"] {
                try? FileManager.default.removeItem(atPath: "\(logPath)\(suffix)")
            }

            // Launch preview with instrumentation env vars
            let app = XCUIApplication(bundleIdentifier: "io.engram.preview")
            app.launchEnvironment = [
                "PREVIEW_INSTRUMENTATION": "1",
                "PREVIEW_NODE_COUNT": "\(count)",
                "PREVIEW_LOG_PATH": logPath
            ]
            app.launch()

            // Timeout scales with node count — simulation needs ~920 frames to settle
            // (alpha_decay=0.995, floor=0.01), and frame rate drops at higher counts.
            let timeout: TimeInterval
            switch count {
            case ...200: timeout = 60
            case ...1000: timeout = 120
            default: timeout = 300
            }

            let settledPath = "\(logPath)-settled"
            let deadline = Date().addingTimeInterval(timeout)
            var appCrashed = false
            while !FileManager.default.fileExists(atPath: settledPath) && Date() < deadline {
                Thread.sleep(forTimeInterval: 0.5)
                // Check if app is still running
                if app.state == .notRunning {
                    appCrashed = true
                    break
                }
            }

            let settled = FileManager.default.fileExists(atPath: settledPath)

            // Parse output files
            let frames = parsePreviewFrames(path: "\(logPath)-frames.csv")
            let report = parseClusterReport(path: "\(logPath)-final.json")

            // Print report section
            print("\n=== NODE COUNT: \(count) ===")

            if appCrashed {
                print("  WARNING: App crashed (\(frames.count) frames recorded before crash)")
            } else if !settled {
                print("  WARNING: Did not settle within \(Int(timeout))s (\(frames.count) frames recorded)")
            }

            // -- Performance --
            print("\n-- PERFORMANCE --")
            if let r = report {
                let framesToSettle = r["framesToSettle"] as? Int ?? 0
                print("  Frames to settle: \(framesToSettle)")
            }

            if let r = report, let perf = r["performance"] as? [String: Any] {
                if let force = perf["forceMs"] as? [String: Any] {
                    let p50 = force["p50"] as? Double ?? 0
                    let p95 = force["p95"] as? Double ?? 0
                    let mx = force["max"] as? Double ?? 0
                    print(String(format: "  Force compute:  p50=%.1fms  p95=%.1fms  max=%.1fms", p50, p95, mx))
                    forceP50s.append((count: count, ms: p50))
                }
                if let wall = perf["wallMs"] as? [String: Any] {
                    let p50 = wall["p50"] as? Double ?? 0
                    let p95 = wall["p95"] as? Double ?? 0
                    let mx = wall["max"] as? Double ?? 0
                    print(String(format: "  Wall time:      p50=%.1fms  p95=%.1fms  max=%.1fms", p50, p95, mx))
                }
            } else if !frames.isEmpty {
                // Fallback: compute stats from CSV frames
                let sortedForce = frames.map(\.forceMs).sorted()
                let sortedWall = frames.map(\.wallMs).sorted()
                let fp50 = percentile(sortedForce, 0.5)
                let fp95 = percentile(sortedForce, 0.95)
                let fmax = sortedForce.last ?? 0
                print(String(format: "  Force compute:  p50=%.1fms  p95=%.1fms  max=%.1fms", fp50, fp95, fmax))
                forceP50s.append((count: count, ms: fp50))
                let wp50 = percentile(sortedWall, 0.5)
                let wp95 = percentile(sortedWall, 0.95)
                let wmax = sortedWall.last ?? 0
                print(String(format: "  Wall time:      p50=%.1fms  p95=%.1fms  max=%.1fms", wp50, wp95, wmax))
            }

            // -- Cluster Quality --
            if let r = report {
                print("\n-- CLUSTER QUALITY --")

                if let projects = r["projects"] as? [[String: Any]] {
                    print("  Project          Centroid (x,y,z)             R75    Nodes")
                    for proj in projects {
                        let name = (proj["name"] as? String ?? "?").padding(toLength: 16, withPad: " ", startingAt: 0)
                        let centroid = proj["centroid"] as? [Double] ?? [0, 0, 0]
                        let r75 = proj["radiusP75"] as? Double ?? 0
                        let nc = proj["nodeCount"] as? Int ?? 0
                        let cx = centroid.count > 0 ? centroid[0] : 0
                        let cy = centroid.count > 1 ? centroid[1] : 0
                        let cz = centroid.count > 2 ? centroid[2] : 0
                        print(String(format: "  %@ (%7.1f,%7.1f,%7.1f)   %6.1f %5d",
                                     name as NSString, cx, cy, cz, r75, nc))
                    }
                }

                if let sep = r["projectSeparation"] as? [String: Any] {
                    let minS = sep["min"] as? Double ?? 0
                    let avgS = sep["avg"] as? Double ?? 0
                    let ratio = sep["sepRadiusRatio"] as? Double ?? 0
                    print(String(format: "\n  Project separation: min=%.1f avg=%.1f  sep/R75=%.2f %@",
                                 minS, avgS, ratio,
                                 ratio > 2 ? "(good)" : ratio > 1 ? "(marginal)" : "(overlapping)" as NSString))
                }

                if let topicSep = r["topicSeparation"] as? [[String: Any]], !topicSep.isEmpty {
                    print("\n  Per-project quality:")
                    print("    Project          compact  sep/R75  topicOverlap  topicMinSep")
                    for ts in topicSep {
                        let proj = (ts["project"] as? String ?? "?").padding(toLength: 16, withPad: " ", startingAt: 0)
                        let compact = ts["compactness"] as? Double ?? 0
                        let sepR = ts["sepRatio"] as? Double ?? 0
                        let overlap = ts["topicOverlapRate"] as? Double ?? 0
                        let minS = ts["minSep"] as? Double ?? 0
                        print(String(format: "    %@   %5.2f    %5.2f      %4.1f%%        %6.1f",
                                     proj as NSString, compact, sepR, overlap * 100, minS))
                    }
                }

                if let strag = r["stragglers"] as? [String: Any] {
                    let sc = strag["count"] as? Int ?? 0
                    let sr = strag["rate"] as? Double ?? 0
                    print(String(format: "\n  Stragglers: %d/%d (%.1f%%)", sc, count, sr * 100))
                }
            }

            app.terminate()
        }

        // -- Scaling Analysis --
        print("\n=== SCALING ANALYSIS ===")
        if let baseline = forceP50s.first, baseline.ms > 0 {
            for entry in forceP50s {
                let multiplier = entry.ms / baseline.ms
                let nRatio = Double(entry.count) / Double(baseline.count)
                let expectedMultiplier = nRatio * nRatio  // O(n^2) scaling
                if entry.count == baseline.count {
                    print(String(format: "  %5d:  force p50=%.1fms  (baseline)", entry.count, entry.ms))
                } else {
                    print(String(format: "  %5d:  force p50=%.1fms  (%.1fx, expected %.0fx at O(n\u{00B2}))",
                                 entry.count, entry.ms, multiplier, expectedMultiplier))
                }
            }
        }

        print("\n" + String(repeating: "=", count: 75) + "\n")
    }
}
