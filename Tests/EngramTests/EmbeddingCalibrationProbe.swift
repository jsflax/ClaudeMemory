import Foundation
import Testing
import EngramKit
import EngramModels
import Lattice

/// Diagnostic probe (env-gated): measures distance distributions on a
/// MIGRATED database so the v2-space thresholds are set from real data, not
/// guesses. Point MIGRATED_DB_PATH at a space-v2 copy of a real hub.
///
/// Emits, as JSON on stdout:
///  - nnDistances: nearest-OTHER-row L2 distance for a sample of rows
///    (the near-duplicate landscape → conflict threshold)
///  - queryDistances: top-10 L2 distances for characteristic recall queries
///    (the relevance landscape → recall/web max-distance)
@Suite("Embedding calibration probe (diagnostic)")
struct EmbeddingCalibrationProbe {

    @Test func measureDistances() async throws {
        guard let dbPath = ProcessInfo.processInfo.environment["MIGRATED_DB_PATH"] else {
            return
        }
        let lattice = try Lattice(
            Memory.self, Edge.self, Checkpoint.self,
            HookState.self, SessionState.self, SyncConfig.self,
            configuration: .init(fileURL: URL(fileURLWithPath: dbPath),
                                 migration: engramMigrations)
        )

        // 1. Nearest-neighbor distances over a sample of stored rows.
        let rows = lattice.objects(Memory.self).snapshot()
        var nnDistances: [Double] = []
        var sameProjectNN: [Double] = []
        let stridedSample = stride(from: 0, to: rows.count, by: max(1, rows.count / 300))
        for index in stridedSample {
            let row = rows[index]
            let hits = lattice.objects(Memory.self)
                .nearest(to: row.embedding, on: \.embedding, limit: 3, distance: .l2)
            for hit in hits {
                guard hit.object.globalId != row.globalId else { continue }
                nnDistances.append(Double(hit.distance))
                if hit.object.project == row.project {
                    sameProjectNN.append(Double(hit.distance))
                }
                break
            }
        }

        // 2. Characteristic recall queries → top-10 distance spread.
        let embedder = EmbeddingService()
        await embedder.load()
        let queries = [
            "how does the sync daemon relay group memories",
            "stripe billing seat count",
            "lattice attach chain recall",
            "embedding model vector search quality",
            "xcode build lockfile dependency pin",
            "web visualizer nebula rendering",
        ]
        var queryDistances: [String: [Double]] = [:]
        for query in queries {
            guard let vec = try await embedder.embed(text: query) else { continue }
            let hits = lattice.objects(Memory.self)
                .nearest(to: Vector<Float>(vec), on: \.embedding, limit: 10, distance: .l2)
            queryDistances[query] = hits.map { Double($0.distance) }
        }

        func percentiles(_ values: [Double]) -> [String: Double] {
            guard !values.isEmpty else { return [:] }
            let sorted = values.sorted()
            func p(_ q: Double) -> Double { sorted[min(sorted.count - 1, Int(q * Double(sorted.count)))] }
            return ["p05": p(0.05), "p25": p(0.25), "p50": p(0.50),
                    "p75": p(0.75), "p95": p(0.95)]
        }

        let out: [String: Any] = [
            "rowCount": rows.count,
            "nnPercentiles": percentiles(nnDistances),
            "sameProjectNNPercentiles": percentiles(sameProjectNN),
            "queryDistances": queryDistances.mapValues { $0.map { round($0 * 1000) / 1000 } },
        ]
        let data = try JSONSerialization.data(withJSONObject: out, options: [.sortedKeys])
        print("CALIBRATION_JSON_BEGIN")
        print(String(data: data, encoding: .utf8)!)
        print("CALIBRATION_JSON_END")
        #expect(!nnDistances.isEmpty)
        lattice.close()
    }
}
