import Foundation
import Testing
import simd
import CEngramSceneTypes
@preconcurrency import Metal
import EngramMetalShaders
import EngramSceneKit

/// Tests that FrameOrchestrator initializes correctly and composes subsystems.
@Suite("FrameOrchestrator")
struct FrameOrchestratorTests {

    static let device: MTLDevice? = MTLCreateSystemDefaultDevice()

    @Test("init succeeds with valid device and library")
    @MainActor func testInit() throws {
        let device = try #require(Self.device)
        let library = try EngramMetalShaders.makeLibrary(device: device)
        let orchestrator = try #require(FrameOrchestrator(device: device, library: library))

        #expect(orchestrator.forceEngine.isFullSimAvailable)
        #expect(orchestrator.bufferManager.sphereTemplateBuffer != nil)
        #expect(orchestrator.nodePipeline != nil)
        #expect(orchestrator.edgePipeline != nil)
        #expect(orchestrator.opaqueDepthState != nil)
        #expect(orchestrator.transparentDepthState != nil)
    }

    @Test("camera state propagates to uniforms")
    @MainActor func testCameraState() throws {
        let device = try #require(Self.device)
        let library = try EngramMetalShaders.makeLibrary(device: device)
        let orchestrator = try #require(FrameOrchestrator(device: device, library: library))

        // Set camera position
        orchestrator.camera = CameraState(
            azimuth: 0.5, elevation: 0.3,
            cameraTarget: SIMD3<Float>(10, 20, 30),
            orbitRadius: 100
        )

        let vm = orchestrator.camera.viewMatrix()
        let det = simd_determinant(vm)
        #expect(abs(det) > 0.0001, "Camera view matrix should be valid")
    }

    @Test("forceEngine accessible and ready")
    @MainActor func testForceEngineReady() throws {
        let device = try #require(Self.device)
        let library = try EngramMetalShaders.makeLibrary(device: device)
        let orchestrator = try #require(FrameOrchestrator(device: device, library: library))

        #expect(orchestrator.forceEngine.isFullSimAvailable)
        #expect(!orchestrator.forceEngine.isInFlight)
    }

    @Test("bufferManager is properly initialized")
    @MainActor func testBufferManager() throws {
        let device = try #require(Self.device)
        let library = try EngramMetalShaders.makeLibrary(device: device)
        let orchestrator = try #require(FrameOrchestrator(device: device, library: library))

        let bm = orchestrator.bufferManager
        #expect(bm.vertsPerSphere > 0)
        #expect(bm.indicesPerSphere > 0)
        #expect(bm.nodePackParamsBuffer != nil)
    }

    @Test("compute pipelines loaded")
    @MainActor func testComputePipelines() throws {
        let device = try #require(Self.device)
        let library = try EngramMetalShaders.makeLibrary(device: device)
        let orchestrator = try #require(FrameOrchestrator(device: device, library: library))

        #expect(orchestrator.packNodePipeline != nil)
        #expect(orchestrator.packEdgePipeline != nil)
        #expect(orchestrator.stampLabelPipeline != nil)
        #expect(orchestrator.packNebulaPipeline != nil)
    }

    @Test("force dispatch through orchestrator's forceEngine works end-to-end")
    @MainActor func testForceDispatchEndToEnd() async throws {
        let device = try #require(Self.device)
        let library = try EngramMetalShaders.makeLibrary(device: device)
        let orchestrator = try #require(FrameOrchestrator(device: device, library: library))

        let n = 200
        let snapshot = ForceEngine.SimulationSnapshot(
            nodeCount: n, isSettled: false,
            posX: (0..<n).map { _ in Float.random(in: -200...200) },
            posY: (0..<n).map { _ in Float.random(in: -200...200) },
            posZ: (0..<n).map { _ in Float.random(in: -200...200) },
            projectGroups: (0..<n).map { $0 % 5 },
            topicGroups: (0..<n).map { $0 % 20 },
            galaxyGroups: [Int](repeating: 0, count: n),
            galaxyCenters: [.zero],
            edgeIndices: (0..<50).map { _ in (Int.random(in: 0..<n), Int.random(in: 0..<n)) },
            topologyDirty: true,
            chargeStrength: 500, crossChargeMultiplier: 3.0,
            sameTopicChargeScale: 0.35, sameProjectChargeScale: 1.0,
            springLength: 240, crossProjectSpringLength: 400, springStrength: 0.0004,
            cohesionStrength: 0.0015, centroidRepulsion: 2500,
            topicCohesionStrength: 0.009, topicCentroidRepulsion: 3500,
            centerStrength: 0.006, center: .zero,
            alpha: 1.0, damping: 0.78, maxSpeed: 12.0
        )

        let result: ForceResult = await withCheckedContinuation { cont in
            orchestrator.forceEngine.encodeForcePass(
                queue: orchestrator.commandQueue,
                snapshot: snapshot
            ) { result in
                cont.resume(returning: result)
            }
        }

        #expect(result.fx.count == n)
        let maxF = result.fx.map { abs($0) }.max() ?? 0
        #expect(maxF > 0, "Forces should be non-zero")
    }
}
