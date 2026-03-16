@preconcurrency import Metal
import CEngramSceneTypes
import simd

/// Manages all GPU instance buffers for the render pipeline.
/// Extracted from MetalGraphRenderer — owns buffer allocation, capacity tracking,
/// and template mesh generation. No rendering or compute dispatch logic.
@MainActor
public final class InstanceBufferManager {
    private let device: MTLDevice

    // Sphere template (instanced node rendering)
    public private(set) var sphereTemplateBuffer: MTLBuffer?
    public private(set) var sphereTemplateIndexBuffer: MTLBuffer?
    public private(set) var vertsPerSphere: Int = 0
    public private(set) var indicesPerSphere: Int = 0

    // Cylinder template (instanced edge rendering)
    public private(set) var cylinderIndexBuffer: MTLBuffer?

    // Node instance buffers
    public var nodeInstanceBuffer: MTLBuffer?
    public var nodeInstanceCapacity: Int = 0
    public var actualNodeCount: Int = 0

    // Edge instance buffers
    public var edgeInstanceBuffer: MTLBuffer?
    public var edgeInstanceCapacity: Int = 0
    public var actualEdgeCount: Int = 0
    public var edgeDataDirty: Bool = false

    // GPU edge packing buffers
    public var edgeDescriptorBuffer: MTLBuffer?
    public var edgeDescriptorCapacity: Int = 0
    public var nodePositionBuffer: MTLBuffer?
    public var nodePositionCapacity: Int = 0
    public var packEdgeParamsBuffer: MTLBuffer?

    // Label batch buffers
    public var labelVertexBuffer: MTLBuffer?
    public var labelIndexBuffer: MTLBuffer?
    public var labelInstanceBuffer: MTLBuffer?
    public var labelStampParamsBuffer: MTLBuffer?
    public var labelInstanceCapacity: Int = 0
    public var labelVertexCapacity: Int = 0
    public var actualLabelCount: Int = 0

    // Label atlas
    public var labelAtlasTexture: MTLTexture?
    public var labelSampler: MTLSamplerState?

    // Nebula buffers
    public var nebulaVertexBuffer: MTLBuffer?
    public var nebulaIndexBuffer: MTLBuffer?
    public var nebulaVertexCapacity: Int = 0
    public var actualNebulaVertexCount: Int = 0

    // Flow particle buffers
    public var flowVertexBuffer: MTLBuffer?
    public var flowIndexBuffer: MTLBuffer?
    public var flowVertexCapacity: Int = 0
    public var actualFlowParticleCount: Int = 0

    // GPU node packing buffers
    public var nodePackInputBuffer: MTLBuffer?
    public var nodePackInputCapacity: Int = 0
    public var pointLightOutputBuffer: MTLBuffer?
    public var pointLightCountBuffer: MTLBuffer?
    public var nodePackParamsBuffer: MTLBuffer?
    public var nodeProjectIdxBuffer: MTLBuffer?
    public var nodeProjectIdxCapacity: Int = 0

    // GPU nebula packing buffers
    public var nebulaGroupInputBuffer: MTLBuffer?
    public var nebulaGroupInputCapacity: Int = 0
    public var nebulaPackParamsBuffer: MTLBuffer?

    // GPU label packing buffers
    public var labelMetadataBuffer: MTLBuffer?
    public var labelMetadataCapacity: Int = 0
    public var labelPackParamsBuffer: MTLBuffer?

    // Nebula group count (set by scene manager, read by compute dispatcher)
    public var nebulaGroupCount: Int = 0

    public init(device: MTLDevice) {
        self.device = device

        setupSphereTemplate()
        setupCylinderTemplate()
        setupLabelSampler()

        // Pre-allocate small param buffers
        labelStampParamsBuffer = device.makeBuffer(length: MemoryLayout<LabelStampParams>.stride, options: .storageModeShared)
        packEdgeParamsBuffer = device.makeBuffer(length: MemoryLayout<PackEdgeParams>.stride, options: .storageModeShared)
        nodePackParamsBuffer = device.makeBuffer(length: MemoryLayout<NodePackParams>.stride, options: .storageModeShared)
        nebulaPackParamsBuffer = device.makeBuffer(length: MemoryLayout<NebulaPackParams>.stride, options: .storageModeShared)
        labelPackParamsBuffer = device.makeBuffer(length: MemoryLayout<LabelPackParams>.stride, options: .storageModeShared)
        pointLightOutputBuffer = device.makeBuffer(length: 16 * MemoryLayout<PointLightEntry>.stride, options: .storageModeShared)
        pointLightCountBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
    }

    // MARK: - Buffer Capacity Management

    public func ensureNodeBuffers(nodeCount: Int) {
        guard nodeCount > 0, nodeCount > nodeInstanceCapacity else { return }
        let needed = max(nodeCount * 2, 512)
        nodeInstanceBuffer = device.makeBuffer(
            length: needed * MemoryLayout<NodeInstance>.stride,
            options: .storageModeShared
        )
        nodeInstanceCapacity = needed
    }

    public func ensureEdgeBuffers(edgeCount: Int) {
        guard edgeCount > edgeInstanceCapacity else { return }
        let needed = max(edgeCount * 2, 2048)
        edgeInstanceBuffer = device.makeBuffer(
            length: needed * MemoryLayout<EdgeInstance>.stride,
            options: .storageModeShared
        )
        edgeInstanceCapacity = needed
    }

    public func ensureLabelBuffers(labelCount: Int) {
        let needed = max(labelCount * 2, 512)
        guard needed > labelVertexCapacity else { return }

        let totalVerts = needed * 4
        let totalIndices = needed * 6

        labelVertexBuffer = device.makeBuffer(
            length: totalVerts * MemoryLayout<BatchVertex>.stride,
            options: .storageModeShared
        )

        let indexBytes = totalIndices * MemoryLayout<UInt32>.stride
        labelIndexBuffer = device.makeBuffer(length: indexBytes, options: .storageModeShared)
        if let buf = labelIndexBuffer {
            let indices = buf.contents().bindMemory(to: UInt32.self, capacity: totalIndices)
            for i in 0..<needed {
                let vBase = UInt32(i * 4)
                let iBase = i * 6
                indices[iBase]     = vBase
                indices[iBase + 1] = vBase + 2
                indices[iBase + 2] = vBase + 1
                indices[iBase + 3] = vBase + 1
                indices[iBase + 4] = vBase + 2
                indices[iBase + 5] = vBase + 3
            }
        }

        labelVertexCapacity = needed

        if labelInstanceCapacity < needed {
            labelInstanceBuffer = device.makeBuffer(
                length: needed * MemoryLayout<LabelInstance>.stride,
                options: .storageModeShared
            )
            labelInstanceCapacity = needed
        }
    }

    public func ensureEdgeDescriptorBuffer(count: Int) {
        guard count > edgeDescriptorCapacity else { return }
        let needed = max(count * 2, 2048)
        edgeDescriptorBuffer = device.makeBuffer(
            length: needed * MemoryLayout<EdgeDescriptor>.stride,
            options: .storageModeShared
        )
        edgeDescriptorCapacity = needed
    }

    public func ensureNodePositionBuffer(count: Int) {
        guard count > nodePositionCapacity else { return }
        let needed = max(count * 2, 512)
        nodePositionBuffer = device.makeBuffer(
            length: needed * MemoryLayout<SIMD3<Float>>.stride,
            options: .storageModeShared
        )
        nodePositionCapacity = needed
    }

    public func ensureNodePackBuffers(count: Int) {
        guard count > nodePackInputCapacity else { return }
        let needed = max(count * 2, 512)
        nodePackInputBuffer = device.makeBuffer(
            length: needed * MemoryLayout<NodePackInput>.stride,
            options: .storageModeShared
        )
        nodeProjectIdxBuffer = device.makeBuffer(
            length: needed * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )
        nodePackInputCapacity = needed
        nodeProjectIdxCapacity = needed
    }

    public func ensureNebulaGroupInputBuffer(count: Int) {
        guard count > nebulaGroupInputCapacity else { return }
        let needed = max(count * 2, 32)
        nebulaGroupInputBuffer = device.makeBuffer(
            length: needed * MemoryLayout<NebulaGroupInput>.stride,
            options: .storageModeShared
        )
        nebulaGroupInputCapacity = needed
    }

    // MARK: - Template Setup

    private func setupSphereTemplate() {
        let template = Self.generateSphereTemplate(segments: 16, rings: 10)
        vertsPerSphere = template.vertices.count
        indicesPerSphere = template.indices.count

        var gpuVerts = template.vertices.map { v in
            SphereTemplateVertex(position: v.0, normal: v.1)
        }
        let vertCount = gpuVerts.count
        sphereTemplateBuffer = gpuVerts.withUnsafeMutableBufferPointer { ptr in
            device.makeBuffer(bytes: ptr.baseAddress!,
                              length: MemoryLayout<SphereTemplateVertex>.stride * vertCount,
                              options: .storageModeShared)
        }

        var indices = template.indices
        let indexCount = indices.count
        sphereTemplateIndexBuffer = indices.withUnsafeMutableBufferPointer { ptr in
            device.makeBuffer(bytes: ptr.baseAddress!,
                              length: MemoryLayout<UInt32>.stride * indexCount,
                              options: .storageModeShared)
        }
    }

    private func setupCylinderTemplate() {
        var indices: [UInt32] = []
        indices.reserveCapacity(36)
        for seg in 0..<6 {
            let next = (seg + 1) % 6
            let b0 = UInt32(seg)
            let b1 = UInt32(next)
            let t0 = UInt32(seg + 6)
            let t1 = UInt32(next + 6)
            indices.append(contentsOf: [b0, t0, t1, b0, t1, b1])
        }
        let cylIndexCount = indices.count
        cylinderIndexBuffer = indices.withUnsafeMutableBufferPointer { ptr in
            device.makeBuffer(bytes: ptr.baseAddress!,
                              length: MemoryLayout<UInt32>.stride * cylIndexCount,
                              options: .storageModeShared)
        }
    }

    private func setupLabelSampler() {
        let desc = MTLSamplerDescriptor()
        desc.minFilter = .linear
        desc.magFilter = .linear
        desc.sAddressMode = .clampToEdge
        desc.tAddressMode = .clampToEdge
        labelSampler = device.makeSamplerState(descriptor: desc)
    }

    // MARK: - Sphere Template Generation

    nonisolated public static func generateSphereTemplate(segments: Int = 10, rings: Int = 6)
        -> (vertices: [(SIMD3<Float>, SIMD3<Float>)], indices: [UInt32])
    {
        var vertices: [(SIMD3<Float>, SIMD3<Float>)] = []
        var indices: [UInt32] = []

        vertices.append((.init(0, 1, 0), .init(0, 1, 0)))

        for ring in 1..<rings {
            let phi = Float.pi * Float(ring) / Float(rings)
            let sinPhi = sin(phi)
            let cosPhi = cos(phi)
            for seg in 0..<segments {
                let theta = 2.0 * Float.pi * Float(seg) / Float(segments)
                let pos = SIMD3<Float>(sinPhi * cos(theta), cosPhi, sinPhi * sin(theta))
                vertices.append((pos, pos))
            }
        }

        vertices.append((.init(0, -1, 0), .init(0, -1, 0)))

        for seg in 0..<segments {
            let next = (seg + 1) % segments
            indices.append(0)
            indices.append(UInt32(1 + seg))
            indices.append(UInt32(1 + next))
        }

        for ring in 0..<(rings - 2) {
            let ringStart = 1 + ring * segments
            let nextStart = 1 + (ring + 1) * segments
            for seg in 0..<segments {
                let next = (seg + 1) % segments
                let a = UInt32(ringStart + seg)
                let b = UInt32(ringStart + next)
                let c = UInt32(nextStart + seg)
                let d = UInt32(nextStart + next)
                indices.append(contentsOf: [a, c, b, b, c, d])
            }
        }

        let bottomPole = UInt32(vertices.count - 1)
        let lastStart = 1 + (rings - 2) * segments
        for seg in 0..<segments {
            let next = (seg + 1) % segments
            indices.append(bottomPole)
            indices.append(UInt32(lastStart + next))
            indices.append(UInt32(lastStart + seg))
        }

        return (vertices, indices)
    }
}
