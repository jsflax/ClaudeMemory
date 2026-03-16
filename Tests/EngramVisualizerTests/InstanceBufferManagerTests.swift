import Foundation
import Testing
import simd
import CEngramSceneTypes
@preconcurrency import Metal
import EngramSceneKit

@Suite("InstanceBufferManager")
struct InstanceBufferManagerTests {

    static let device: MTLDevice? = MTLCreateSystemDefaultDevice()

    @Test("init creates sphere and cylinder templates")
    @MainActor func testTemplatesCreated() throws {
        let device = try #require(Self.device)
        let bm = InstanceBufferManager(device: device)
        #expect(bm.sphereTemplateBuffer != nil)
        #expect(bm.sphereTemplateIndexBuffer != nil)
        #expect(bm.cylinderIndexBuffer != nil)
        #expect(bm.vertsPerSphere > 0)
        #expect(bm.indicesPerSphere > 0)
    }

    @Test("ensureNodeBuffers resizes at capacity boundary")
    @MainActor func testNodeBufferResize() throws {
        let device = try #require(Self.device)
        let bm = InstanceBufferManager(device: device)

        #expect(bm.nodeInstanceCapacity == 0)
        #expect(bm.nodeInstanceBuffer == nil)

        bm.ensureNodeBuffers(nodeCount: 100)
        #expect(bm.nodeInstanceCapacity >= 100)
        #expect(bm.nodeInstanceBuffer != nil)
        let firstCapacity = bm.nodeInstanceCapacity

        // Same count shouldn't re-allocate
        let firstBuffer = bm.nodeInstanceBuffer
        bm.ensureNodeBuffers(nodeCount: 100)
        #expect(bm.nodeInstanceBuffer === firstBuffer, "Should not re-allocate for same count")

        // Exceeding capacity should grow
        bm.ensureNodeBuffers(nodeCount: firstCapacity + 1)
        #expect(bm.nodeInstanceCapacity > firstCapacity)
    }

    @Test("ensureEdgeBuffers resizes correctly")
    @MainActor func testEdgeBufferResize() throws {
        let device = try #require(Self.device)
        let bm = InstanceBufferManager(device: device)

        bm.ensureEdgeBuffers(edgeCount: 500)
        #expect(bm.edgeInstanceCapacity >= 500)
        let firstCap = bm.edgeInstanceCapacity

        bm.ensureEdgeBuffers(edgeCount: 500)
        #expect(bm.edgeInstanceCapacity == firstCap, "No resize needed")

        bm.ensureEdgeBuffers(edgeCount: firstCap + 1)
        #expect(bm.edgeInstanceCapacity > firstCap)
    }

    @Test("ensureLabelBuffers creates vertex and index buffers")
    @MainActor func testLabelBufferSetup() throws {
        let device = try #require(Self.device)
        let bm = InstanceBufferManager(device: device)

        bm.ensureLabelBuffers(labelCount: 50)
        #expect(bm.labelVertexBuffer != nil)
        #expect(bm.labelIndexBuffer != nil)
        #expect(bm.labelInstanceBuffer != nil)
        #expect(bm.labelVertexCapacity >= 50)
    }

    @Test("handles zero count gracefully")
    @MainActor func testZeroCounts() throws {
        let device = try #require(Self.device)
        let bm = InstanceBufferManager(device: device)

        bm.ensureNodeBuffers(nodeCount: 0)
        #expect(bm.nodeInstanceBuffer == nil, "Zero nodes shouldn't allocate")

        bm.ensureEdgeBuffers(edgeCount: 0)
        #expect(bm.edgeInstanceBuffer == nil, "Zero edges shouldn't allocate")
    }

    @Test("sphere template generates correct topology")
    func testSphereTemplate() {
        let template = InstanceBufferManager.generateSphereTemplate(segments: 8, rings: 6)
        #expect(template.vertices.count > 0)
        #expect(template.indices.count > 0)
        // All indices should be valid
        let maxIdx = UInt32(template.vertices.count - 1)
        for idx in template.indices {
            #expect(idx <= maxIdx, "Index \(idx) out of bounds (max \(maxIdx))")
        }
        // Index count should be multiple of 3 (triangles)
        #expect(template.indices.count % 3 == 0)
    }

    @Test("param buffers pre-allocated on init")
    @MainActor func testParamBuffers() throws {
        let device = try #require(Self.device)
        let bm = InstanceBufferManager(device: device)
        #expect(bm.labelStampParamsBuffer != nil)
        #expect(bm.packEdgeParamsBuffer != nil)
        #expect(bm.nodePackParamsBuffer != nil)
        #expect(bm.nebulaPackParamsBuffer != nil)
        #expect(bm.labelPackParamsBuffer != nil)
        #expect(bm.pointLightOutputBuffer != nil)
        #expect(bm.pointLightCountBuffer != nil)
    }
}
