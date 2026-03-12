@testable import Engram
@preconcurrency import AVFoundation
import XCTest
import simd

/// Performance tests for the real ProximityTonePool class.
/// Tests run hosted in the app (TEST_HOST) so @testable import works.
@MainActor
final class ProximityTonePerfTests: XCTestCase {

    nonisolated(unsafe) static var sharedEngine: AVAudioEngine!
    nonisolated(unsafe) static var sharedEnvironment: AVAudioEnvironmentNode!
    private var pool: ProximityTonePool!
    private var positions: [UUID: SIMD3<Float>] = [:]
    private let cameraPosition = SIMD3<Float>(0, 0, 0)

    override class func setUp() {
        super.setUp()
        let eng = AVAudioEngine()
        let env = AVAudioEnvironmentNode()
        eng.attach(env)
        eng.connect(env, to: eng.mainMixerNode, format: AudioSynthesis.monoFormat)
        try? eng.start()
        sharedEngine = eng
        sharedEnvironment = env
    }

    override func setUp() {
        super.setUp()

        pool = ProximityTonePool(environment: Self.sharedEnvironment, poolSize: 8)
        pool.setup()

        // 3500 deterministic random positions in [-1000, 1000]³
        srand48(42)
        positions.removeAll()
        for _ in 0..<3500 {
            let x = Float(drand48() * 2000 - 1000)
            let y = Float(drand48() * 2000 - 1000)
            let z = Float(drand48() * 2000 - 1000)
            positions[UUID()] = SIMD3<Float>(x, y, z)
        }
    }

    override func tearDown() {
        pool.stopAll()
        pool = nil
        positions.removeAll()
        super.tearDown()
    }

    override class func tearDown() {
        sharedEngine?.stop()
        sharedEngine = nil
        sharedEnvironment = nil
        super.tearDown()
    }

    // MARK: - Full update() — 60 frames (1 second at 60fps)

    func testUpdate_60Frames() {
        let glowing: [UUID: Date] = [:]
        let newNodes: [UUID: Date] = [:]

        measure {
            for _ in 0..<60 {
                self.pool.update(
                    positions: self.positions,
                    cameraPosition: self.cameraPosition,
                    selectedNode: nil,
                    glowingNodes: glowing,
                    newNodes: newNodes,
                    scaleFactor: 1.0 / 200.0
                )
            }
            self.pool.stopAll()
        }
    }

    // MARK: - Rescan only (6Hz path)

    func testRescan_100Iterations() {
        measure {
            for _ in 0..<100 {
                self.pool.rescan(
                    positions: self.positions,
                    cameraPosition: self.cameraPosition,
                    selectedNode: nil
                )
            }
        }
    }

    // MARK: - Voice update only (60Hz path)

    func testVoiceUpdate_6000Frames() {
        // Prime with one rescan so voices are assigned
        pool.rescan(positions: positions, cameraPosition: cameraPosition, selectedNode: nil)

        let glowing: [UUID: Date] = [:]
        let newNodes: [UUID: Date] = [:]

        measure {
            for _ in 0..<6000 {
                self.pool.updateVoicePositions(
                    positions: self.positions,
                    cameraPosition: self.cameraPosition,
                    glowingNodes: glowing,
                    newNodes: newNodes,
                    scaleFactor: 1.0 / 200.0
                )
            }
        }
    }

    // MARK: - Dense cluster (worst case for bounded insertion)

    func testRescan_DenseCluster() {
        // All 3500 nodes within 200 units of camera
        var dense: [UUID: SIMD3<Float>] = [:]
        srand48(99)
        for _ in 0..<3500 {
            dense[UUID()] = SIMD3<Float>(
                Float(drand48() * 400 - 200),
                Float(drand48() * 400 - 200),
                Float(drand48() * 400 - 200)
            )
        }

        measure {
            for _ in 0..<100 {
                self.pool.rescan(
                    positions: dense,
                    cameraPosition: self.cameraPosition,
                    selectedNode: nil
                )
            }
        }
    }

    // MARK: - Correctness

    func testSelectedNodeExcluded() {
        let closeId = UUID()
        var testPositions = positions
        testPositions[closeId] = SIMD3<Float>(1, 1, 1) // right next to camera

        let glowing: [UUID: Date] = [:]
        let newNodes: [UUID: Date] = [:]

        // Run enough frames to trigger rescan
        for _ in 0..<11 {
            pool.update(
                positions: testPositions,
                cameraPosition: cameraPosition,
                selectedNode: closeId,
                glowingNodes: glowing,
                newNodes: newNodes,
                scaleFactor: 1.0 / 200.0
            )
        }
        pool.stopAll()
    }

    func testEmptyPositions() {
        let glowing: [UUID: Date] = [:]
        let newNodes: [UUID: Date] = [:]

        pool.update(
            positions: [:],
            cameraPosition: cameraPosition,
            selectedNode: nil,
            glowingNodes: glowing,
            newNodes: newNodes,
            scaleFactor: 1.0 / 200.0
        )
    }

    func testAllNodesFarAway() {
        var farPositions: [UUID: SIMD3<Float>] = [:]
        for _ in 0..<100 {
            farPositions[UUID()] = SIMD3<Float>(
                Float.random(in: 500...1000),
                Float.random(in: 500...1000),
                Float.random(in: 500...1000)
            )
        }

        pool.rescan(positions: farPositions, cameraPosition: cameraPosition, selectedNode: nil)
        pool.stopAll()
    }
}
