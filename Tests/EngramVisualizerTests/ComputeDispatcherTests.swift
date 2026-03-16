import Foundation
import Testing
import simd
import CEngramSceneTypes
@preconcurrency import Metal
import EngramMetalShaders
import EngramSceneKit

/// Tests that ComputeDispatcher correctly dispatches GPU compute kernels.
@Suite("ComputeDispatcher")
struct ComputeDispatcherTests {

    static let device: MTLDevice? = MTLCreateSystemDefaultDevice()

    @Test("packNodes produces non-zero output for 8K nodes")
    @MainActor func testPackNodes8K() throws {
        let device = try #require(Self.device)
        let library = try EngramMetalShaders.makeLibrary(device: device)
        let queue = try #require(device.makeCommandQueue())
        let n = 8000

        let pipeline = try device.makeComputePipelineState(
            function: try #require(library.makeFunction(name: "pack_node_instances")))

        let inputBuf = try #require(device.makeBuffer(
            length: n * MemoryLayout<NodePackInput>.stride, options: .storageModeShared))
        let ni = inputBuf.contents().bindMemory(to: NodePackInput.self, capacity: n)
        for i in 0..<n {
            ni[i] = NodePackInput(
                position: SIMD3<Float>(Float.random(in: -500...500),
                                       Float.random(in: -500...500),
                                       Float.random(in: -500...500)),
                baseRadius: 1.0,
                baseColor: SIMD3<Float>(0.5, 0.5, 0.8),
                packedState: 0)
        }

        let instanceBuf = try #require(device.makeBuffer(
            length: n * MemoryLayout<NodeInstance>.stride, options: .storageModeShared))
        let lightBuf = try #require(device.makeBuffer(
            length: 16 * MemoryLayout<PointLightEntry>.stride, options: .storageModeShared))
        let lightCountBuf = try #require(device.makeBuffer(
            length: MemoryLayout<UInt32>.stride, options: .storageModeShared))
        let paramsBuf = try #require(device.makeBuffer(
            length: MemoryLayout<NodePackParams>.stride, options: .storageModeShared))
        paramsBuf.contents().bindMemory(to: NodePackParams.self, capacity: 1).pointee =
            NodePackParams(nodeCount: UInt32(n), scaleFactor: 1.0/200.0, nodeRadius: 0.008,
                          animationTime: 0, projectCount: 15, _pad0: 0, _pad1: 0, _pad2: 0)
        let projIdxBuf = try #require(device.makeBuffer(
            length: n * MemoryLayout<UInt32>.stride, options: .storageModeShared))
        // Position buffer — GPU-resident positions read by pack_node_instances
        let posBuf = try #require(device.makeBuffer(
            length: n * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared))
        let positions = posBuf.contents().bindMemory(to: SIMD3<Float>.self, capacity: n)
        for i in 0..<n {
            positions[i] = SIMD3<Float>(Float.random(in: -500...500),
                                        Float.random(in: -500...500),
                                        Float.random(in: -500...500))
        }

        let cmdBuf = try #require(queue.makeCommandBuffer())
        let enc = try #require(cmdBuf.makeComputeCommandEncoder())

        ComputeDispatcher.packNodes(
            encoder: enc, pipeline: pipeline,
            inputBuffer: inputBuf, instanceBuffer: instanceBuf,
            lightOutputBuffer: lightBuf, lightCountBuffer: lightCountBuf,
            paramsBuffer: paramsBuf, projIdxBuffer: projIdxBuf,
            positionBuffer: posBuf,
            nodeCount: n)

        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Verify output has non-zero positions
        let output = instanceBuf.contents().bindMemory(to: NodeInstance.self, capacity: n)
        var nonZero = 0
        for i in 0..<n {
            if output[i].position.x != 0 || output[i].position.y != 0 || output[i].position.z != 0 {
                nonZero += 1
            }
        }
        #expect(nonZero > n / 2, "Most node instances should have non-zero positions")
    }

    @Test("packEdges produces non-zero output")
    @MainActor func testPackEdges() throws {
        let device = try #require(Self.device)
        let library = try EngramMetalShaders.makeLibrary(device: device)
        let queue = try #require(device.makeCommandQueue())

        let n = 500
        let edgeCount = 1000

        let pipeline = try device.makeComputePipelineState(
            function: try #require(library.makeFunction(name: "pack_edge_instances")))

        // Position buffer
        let posBuf = try #require(device.makeBuffer(
            length: n * MemoryLayout<SIMD3<Float>>.stride, options: .storageModeShared))
        let positions = posBuf.contents().bindMemory(to: SIMD3<Float>.self, capacity: n)
        for i in 0..<n {
            positions[i] = SIMD3<Float>(Float.random(in: -500...500),
                                        Float.random(in: -500...500),
                                        Float.random(in: -500...500))
        }

        // Edge descriptors
        let descBuf = try #require(device.makeBuffer(
            length: edgeCount * MemoryLayout<EdgeDescriptor>.stride, options: .storageModeShared))
        let ed = descBuf.contents().bindMemory(to: EdgeDescriptor.self, capacity: edgeCount)
        for i in 0..<edgeCount {
            ed[i] = EdgeDescriptor(
                sourceIdx: UInt32(Int.random(in: 0..<n)),
                targetIdx: UInt32(Int.random(in: 0..<n)),
                sourceRadius: 0.008, targetRadius: 0.008,
                color: SIMD4<Float>(0.3, 0.5, 0.8, 1.0),
                baseRadius: 0.002, state: 0, _pad0: 0, _pad1: 0)
        }

        let paramsBuf = try #require(device.makeBuffer(
            length: MemoryLayout<PackEdgeParams>.stride, options: .storageModeShared))
        paramsBuf.contents().bindMemory(to: PackEdgeParams.self, capacity: 1).pointee =
            PackEdgeParams(edgeCount: UInt32(edgeCount), scaleFactor: 1.0/200.0, _pad0: 0, _pad1: 0)

        let outputBuf = try #require(device.makeBuffer(
            length: edgeCount * MemoryLayout<EdgeInstance>.stride, options: .storageModeShared))

        let cmdBuf = try #require(queue.makeCommandBuffer())
        let enc = try #require(cmdBuf.makeComputeCommandEncoder())

        ComputeDispatcher.packEdges(
            encoder: enc, pipeline: pipeline,
            descriptorBuffer: descBuf, positionBuffer: posBuf,
            paramsBuffer: paramsBuf, outputBuffer: outputBuf,
            edgeCount: edgeCount)

        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Verify output has non-zero positions
        let output = outputBuf.contents().bindMemory(to: EdgeInstance.self, capacity: edgeCount)
        var nonZero = 0
        for i in 0..<edgeCount {
            if output[i].sourcePos.x != 0 || output[i].sourcePos.y != 0 {
                nonZero += 1
            }
        }
        #expect(nonZero > edgeCount / 2, "Most edge instances should have non-zero positions")
    }

    @Test("packNodes with zero count is no-op")
    @MainActor func testPackNodesZero() throws {
        let device = try #require(Self.device)
        let library = try EngramMetalShaders.makeLibrary(device: device)
        let queue = try #require(device.makeCommandQueue())

        let pipeline = try device.makeComputePipelineState(
            function: try #require(library.makeFunction(name: "pack_node_instances")))

        let dummyBuf = try #require(device.makeBuffer(length: 64, options: .storageModeShared))

        let cmdBuf = try #require(queue.makeCommandBuffer())
        let enc = try #require(cmdBuf.makeComputeCommandEncoder())

        // Should not crash with zero count
        ComputeDispatcher.packNodes(
            encoder: enc, pipeline: pipeline,
            inputBuffer: dummyBuf, instanceBuffer: dummyBuf,
            lightOutputBuffer: dummyBuf, lightCountBuffer: dummyBuf,
            paramsBuffer: dummyBuf, projIdxBuffer: dummyBuf,
            positionBuffer: dummyBuf,
            nodeCount: 0)

        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        // No crash = pass
    }
}
