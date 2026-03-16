import Foundation
import Testing
import simd
import CEngramSceneTypes
@preconcurrency import Metal
import EngramMetalShaders
import EngramSceneKit

@Suite("GPUSimulationState")
struct GPUSimulationStateTests {

    static let device: MTLDevice? = MTLCreateSystemDefaultDevice()

    @Test("init creates integrate pipeline")
    @MainActor func testInit() throws {
        let device = try #require(Self.device)
        let library = try EngramMetalShaders.makeLibrary(device: device)
        let sim = GPUSimulationState(device: device, library: library)
        #expect(sim.integratePipeline != nil, "integrate_positions pipeline should load")
    }

    @Test("ensureCapacity allocates buffers")
    @MainActor func testEnsureCapacity() throws {
        let device = try #require(Self.device)
        let library = try EngramMetalShaders.makeLibrary(device: device)
        let sim = GPUSimulationState(device: device, library: library)

        sim.ensureCapacity(1000)
        #expect(sim.positionBuffer != nil)
        #expect(sim.velocityBuffer != nil)
        #expect(sim.forceBuffer != nil)
        #expect(sim.positionBuffer!.length >= 1000 * MemoryLayout<SIMD3<Float>>.stride)
    }

    @Test("uploadPositions roundtrips correctly")
    @MainActor func testUploadRoundtrip() throws {
        let device = try #require(Self.device)
        let library = try EngramMetalShaders.makeLibrary(device: device)
        let sim = GPUSimulationState(device: device, library: library)

        let n = 500
        let x = (0..<n).map { Float($0) * 1.5 }
        let y = (0..<n).map { Float($0) * 2.0 }
        let z = (0..<n).map { Float($0) * 0.5 }

        sim.uploadPositions(x: x, y: y, z: z)
        #expect(sim.nodeCount == n)

        sim.readBackPositions()
        #expect(sim.cpuPositions.count == n)
        #expect(abs(sim.cpuPositions[0].x - 0) < 0.001)
        #expect(abs(sim.cpuPositions[1].x - 1.5) < 0.001)
        #expect(abs(sim.cpuPositions[99].y - 198) < 0.001)
    }

    @Test("integration kernel moves positions")
    @MainActor func testIntegrationKernel() async throws {
        let device = try #require(Self.device)
        let library = try EngramMetalShaders.makeLibrary(device: device)
        let queue = try #require(device.makeCommandQueue())
        let sim = GPUSimulationState(device: device, library: library)

        let n = 100
        // All positions at origin, zero velocity
        sim.uploadPositions(
            x: [Float](repeating: 0, count: n),
            y: [Float](repeating: 0, count: n),
            z: [Float](repeating: 0, count: n))

        // Write non-zero forces
        let forceBuf = try #require(sim.forceBuffer)
        let fPtr = forceBuf.contents().bindMemory(to: SIMD3<Float>.self, capacity: n)
        for i in 0..<n {
            fPtr[i] = SIMD3<Float>(1.0, 0.0, 0.0)  // push +X
        }

        // Encode integration
        let cmdBuf = try #require(queue.makeCommandBuffer())
        let enc = try #require(cmdBuf.makeComputeCommandEncoder())
        sim.encodeIntegration(encoder: enc, damping: 1.0, maxSpeed: 100.0)
        enc.endEncoding()
        cmdBuf.commit()
        await cmdBuf.completed()

        // Read back — positions should have moved in +X
        sim.readBackPositions()
        #expect(sim.cpuPositions[0].x > 0, "Position should have moved +X from force")
        #expect(abs(sim.cpuPositions[0].y) < 0.001, "Y should be unchanged")
    }

    @Test("alpha decay reaches floor and settles")
    @MainActor func testAlphaDecayAndSettlement() throws {
        let device = try #require(Self.device)
        let library = try EngramMetalShaders.makeLibrary(device: device)
        let sim = GPUSimulationState(device: device, library: library)

        #expect(sim.alpha == 1.0)
        #expect(!sim.isSettled)

        // Decay until floor
        for _ in 0..<2000 {
            sim.decayAlpha()
        }
        #expect(sim.alpha <= sim.alphaFloor + 0.001, "Alpha should reach floor")

        // Keep decaying until settled (30 frames at floor)
        for _ in 0..<50 {
            sim.decayAlpha()
        }
        #expect(sim.isSettled, "Should settle after 30 frames at alpha floor")
    }

    @Test("wake resets settlement")
    @MainActor func testWake() throws {
        let device = try #require(Self.device)
        let library = try EngramMetalShaders.makeLibrary(device: device)
        let sim = GPUSimulationState(device: device, library: library)

        // Settle first
        for _ in 0..<2050 {
            sim.decayAlpha()
        }
        #expect(sim.isSettled)

        sim.wake()
        #expect(!sim.isSettled)
        #expect(sim.alpha == 1.0)
    }

    @Test("readBackIfNeeded respects interval")
    @MainActor func testReadbackInterval() throws {
        let device = try #require(Self.device)
        let library = try EngramMetalShaders.makeLibrary(device: device)
        let sim = GPUSimulationState(device: device, library: library)

        sim.uploadPositions(x: [1.0], y: [2.0], z: [3.0])
        sim.readbackInterval = 3

        // First call shouldn't readback (interval not reached)
        #expect(!sim.readBackIfNeeded())
        #expect(!sim.readBackIfNeeded())
        // Third call should readback
        #expect(sim.readBackIfNeeded())
        #expect(sim.cpuPositions.count == 1)
    }
}
