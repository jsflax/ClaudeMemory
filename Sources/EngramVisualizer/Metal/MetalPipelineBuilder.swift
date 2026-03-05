import Metal

/// Factory for MTLRenderPipelineState objects.
/// Creates pipelines for nodes, edges, labels, nebulae, and flow particles.
@MainActor
struct MetalPipelineBuilder {

    /// Vertex attribute layout matching BatchVertex (48 bytes).
    private static func batchVertexDescriptor() -> MTLVertexDescriptor {
        let vd = MTLVertexDescriptor()
        // position: float3 at offset 0
        vd.attributes[0].format = .float3
        vd.attributes[0].offset = 0
        vd.attributes[0].bufferIndex = 0
        // normal: float3 at offset 12
        vd.attributes[1].format = .float3
        vd.attributes[1].offset = 12
        vd.attributes[1].bufferIndex = 0
        // uv: float2 at offset 24
        vd.attributes[2].format = .float2
        vd.attributes[2].offset = 24
        vd.attributes[2].bufferIndex = 0
        // color: float4 at offset 32
        vd.attributes[3].format = .float4
        vd.attributes[3].offset = 32
        vd.attributes[3].bufferIndex = 0
        // stride
        vd.layouts[0].stride = 48
        vd.layouts[0].stepRate = 1
        vd.layouts[0].stepFunction = .perVertex
        return vd
    }

    // MARK: - Node Pipeline (opaque, lit, depth-write ON)

    static func buildNodePipeline(device: MTLDevice, library: MTLLibrary) throws -> MTLRenderPipelineState {
        guard let vertexFn = library.makeFunction(name: "node_vertex"),
              let fragmentFn = library.makeFunction(name: "node_fragment") else {
            throw MetalPipelineError.functionNotFound("node_vertex or node_fragment")
        }

        let desc = MTLRenderPipelineDescriptor()
        desc.label = "Node Pipeline"
        desc.vertexFunction = vertexFn
        desc.fragmentFunction = fragmentFn
        desc.vertexDescriptor = batchVertexDescriptor()
        desc.colorAttachments[0].pixelFormat = .bgra8Unorm
        // Opaque — no blending
        desc.colorAttachments[0].isBlendingEnabled = false
        desc.depthAttachmentPixelFormat = .depth32Float

        return try device.makeRenderPipelineState(descriptor: desc)
    }

    // MARK: - Edge Pipeline (transparent, unlit, depth-write OFF)

    static func buildEdgePipeline(device: MTLDevice, library: MTLLibrary) throws -> MTLRenderPipelineState {
        guard let vertexFn = library.makeFunction(name: "edge_vertex"),
              let fragmentFn = library.makeFunction(name: "edge_fragment") else {
            throw MetalPipelineError.functionNotFound("edge_vertex or edge_fragment")
        }

        let desc = MTLRenderPipelineDescriptor()
        desc.label = "Edge Pipeline"
        desc.vertexFunction = vertexFn
        desc.fragmentFunction = fragmentFn
        desc.vertexDescriptor = batchVertexDescriptor()
        desc.colorAttachments[0].pixelFormat = .bgra8Unorm
        // Premultiplied alpha blending
        desc.colorAttachments[0].isBlendingEnabled = true
        desc.colorAttachments[0].rgbBlendOperation = .add
        desc.colorAttachments[0].alphaBlendOperation = .add
        desc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        desc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        desc.colorAttachments[0].sourceAlphaBlendFactor = .one
        desc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        desc.depthAttachmentPixelFormat = .depth32Float

        return try device.makeRenderPipelineState(descriptor: desc)
    }

    // MARK: - Label Pipeline (transparent, textured, depth-write OFF)

    static func buildLabelPipeline(device: MTLDevice, library: MTLLibrary) throws -> MTLRenderPipelineState {
        guard let vertexFn = library.makeFunction(name: "label_vertex"),
              let fragmentFn = library.makeFunction(name: "label_fragment") else {
            throw MetalPipelineError.functionNotFound("label_vertex or label_fragment")
        }

        let desc = MTLRenderPipelineDescriptor()
        desc.label = "Label Pipeline"
        desc.vertexFunction = vertexFn
        desc.fragmentFunction = fragmentFn
        desc.vertexDescriptor = batchVertexDescriptor()
        desc.colorAttachments[0].pixelFormat = .bgra8Unorm
        desc.colorAttachments[0].isBlendingEnabled = true
        desc.colorAttachments[0].rgbBlendOperation = .add
        desc.colorAttachments[0].alphaBlendOperation = .add
        desc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        desc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        desc.colorAttachments[0].sourceAlphaBlendFactor = .one
        desc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        desc.depthAttachmentPixelFormat = .depth32Float

        return try device.makeRenderPipelineState(descriptor: desc)
    }

    // MARK: - Nebula Pipeline (transparent, fBm noise, depth-write OFF)

    static func buildNebulaPipeline(device: MTLDevice, library: MTLLibrary) throws -> MTLRenderPipelineState {
        guard let vertexFn = library.makeFunction(name: "nebula_vertex"),
              let fragmentFn = library.makeFunction(name: "nebula_fragment") else {
            throw MetalPipelineError.functionNotFound("nebula_vertex or nebula_fragment")
        }

        let desc = MTLRenderPipelineDescriptor()
        desc.label = "Nebula Pipeline"
        desc.vertexFunction = vertexFn
        desc.fragmentFunction = fragmentFn
        // Nebula uses its own vertex layout — no shared vertex descriptor needed
        // We'll use manual buffer reads in the vertex shader
        desc.colorAttachments[0].pixelFormat = .bgra8Unorm
        desc.colorAttachments[0].isBlendingEnabled = true
        desc.colorAttachments[0].rgbBlendOperation = .add
        desc.colorAttachments[0].alphaBlendOperation = .add
        desc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        desc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        desc.colorAttachments[0].sourceAlphaBlendFactor = .one
        desc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        desc.depthAttachmentPixelFormat = .depth32Float

        return try device.makeRenderPipelineState(descriptor: desc)
    }

    // MARK: - Flow Particle Pipeline (transparent, soft-circle, depth-write OFF)

    static func buildFlowPipeline(device: MTLDevice, library: MTLLibrary) throws -> MTLRenderPipelineState {
        guard let vertexFn = library.makeFunction(name: "flow_vertex"),
              let fragmentFn = library.makeFunction(name: "flow_fragment") else {
            throw MetalPipelineError.functionNotFound("flow_vertex or flow_fragment")
        }

        let desc = MTLRenderPipelineDescriptor()
        desc.label = "Flow Particle Pipeline"
        desc.vertexFunction = vertexFn
        desc.fragmentFunction = fragmentFn
        desc.colorAttachments[0].pixelFormat = .bgra8Unorm
        desc.colorAttachments[0].isBlendingEnabled = true
        desc.colorAttachments[0].rgbBlendOperation = .add
        desc.colorAttachments[0].alphaBlendOperation = .add
        desc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        desc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        desc.colorAttachments[0].sourceAlphaBlendFactor = .one
        desc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        desc.depthAttachmentPixelFormat = .depth32Float

        return try device.makeRenderPipelineState(descriptor: desc)
    }

    // MARK: - Mascot Pipeline (opaque, PBR textured, depth-write ON)

    static func buildMascotPipeline(device: MTLDevice, library: MTLLibrary) throws -> MTLRenderPipelineState {
        guard let vertexFn = library.makeFunction(name: "mascot_vertex_instanced"),
              let fragmentFn = library.makeFunction(name: "mascot_fragment_instanced") else {
            throw MetalPipelineError.functionNotFound("mascot_vertex_instanced or mascot_fragment_instanced")
        }

        let desc = MTLRenderPipelineDescriptor()
        desc.label = "Mascot Pipeline"
        desc.vertexFunction = vertexFn
        desc.fragmentFunction = fragmentFn
        // No vertex descriptor — manual buffer reads in shader
        desc.colorAttachments[0].pixelFormat = .bgra8Unorm
        desc.colorAttachments[0].isBlendingEnabled = false
        desc.depthAttachmentPixelFormat = .depth32Float

        return try device.makeRenderPipelineState(descriptor: desc)
    }

    // MARK: - Arcane Circle Pipeline (additive transparent, depth-write OFF)

    static func buildArcaneCirclePipeline(device: MTLDevice, library: MTLLibrary) throws -> MTLRenderPipelineState {
        guard let vertexFn = library.makeFunction(name: "arcane_vertex"),
              let fragmentFn = library.makeFunction(name: "arcane_circle_fragment") else {
            throw MetalPipelineError.functionNotFound("arcane_vertex or arcane_circle_fragment")
        }

        let desc = MTLRenderPipelineDescriptor()
        desc.label = "Arcane Circle Pipeline"
        desc.vertexFunction = vertexFn
        desc.fragmentFunction = fragmentFn
        // Additive blending — glowing magic circle adds light
        desc.colorAttachments[0].pixelFormat = .bgra8Unorm
        desc.colorAttachments[0].isBlendingEnabled = true
        desc.colorAttachments[0].rgbBlendOperation = .add
        desc.colorAttachments[0].alphaBlendOperation = .add
        desc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        desc.colorAttachments[0].destinationRGBBlendFactor = .one
        desc.colorAttachments[0].sourceAlphaBlendFactor = .one
        desc.colorAttachments[0].destinationAlphaBlendFactor = .one
        desc.depthAttachmentPixelFormat = .depth32Float

        return try device.makeRenderPipelineState(descriptor: desc)
    }

    // MARK: - Node Ring Pipeline (additive transparent, depth-write OFF)

    static func buildNodeRingPipeline(device: MTLDevice, library: MTLLibrary) throws -> MTLRenderPipelineState {
        guard let vertexFn = library.makeFunction(name: "node_ring_vertex"),
              let fragmentFn = library.makeFunction(name: "node_ring_fragment") else {
            throw MetalPipelineError.functionNotFound("node_ring_vertex or node_ring_fragment")
        }

        let desc = MTLRenderPipelineDescriptor()
        desc.label = "Node Ring Pipeline"
        desc.vertexFunction = vertexFn
        desc.fragmentFunction = fragmentFn
        // Additive blending — rings add light (same as arcane circle)
        desc.colorAttachments[0].pixelFormat = .bgra8Unorm
        desc.colorAttachments[0].isBlendingEnabled = true
        desc.colorAttachments[0].rgbBlendOperation = .add
        desc.colorAttachments[0].alphaBlendOperation = .add
        desc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        desc.colorAttachments[0].destinationRGBBlendFactor = .one
        desc.colorAttachments[0].sourceAlphaBlendFactor = .one
        desc.colorAttachments[0].destinationAlphaBlendFactor = .one
        desc.depthAttachmentPixelFormat = .depth32Float

        return try device.makeRenderPipelineState(descriptor: desc)
    }

    // MARK: - Holo Screen Pipeline (transparent, alpha blended, depth-write OFF)

    static func buildHoloScreenPipeline(device: MTLDevice, library: MTLLibrary) throws -> MTLRenderPipelineState {
        guard let vertexFn = library.makeFunction(name: "holo_vertex"),
              let fragmentFn = library.makeFunction(name: "holo_fragment") else {
            throw MetalPipelineError.functionNotFound("holo_vertex or holo_fragment")
        }

        let desc = MTLRenderPipelineDescriptor()
        desc.label = "Holo Screen Pipeline"
        desc.vertexFunction = vertexFn
        desc.fragmentFunction = fragmentFn
        // Premultiplied alpha blending — shader outputs color * alpha
        desc.colorAttachments[0].pixelFormat = .bgra8Unorm
        desc.colorAttachments[0].isBlendingEnabled = true
        desc.colorAttachments[0].rgbBlendOperation = .add
        desc.colorAttachments[0].alphaBlendOperation = .add
        desc.colorAttachments[0].sourceRGBBlendFactor = .one
        desc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        desc.colorAttachments[0].sourceAlphaBlendFactor = .one
        desc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        desc.depthAttachmentPixelFormat = .depth32Float

        return try device.makeRenderPipelineState(descriptor: desc)
    }

    // MARK: - Depth Stencil States

    /// Opaque pass: depth test ON, depth write ON.
    static func buildOpaqueDepthState(device: MTLDevice) -> MTLDepthStencilState? {
        let desc = MTLDepthStencilDescriptor()
        desc.label = "Opaque Depth"
        desc.depthCompareFunction = .less
        desc.isDepthWriteEnabled = true
        return device.makeDepthStencilState(descriptor: desc)
    }

    /// Transparent pass: depth test ON, depth write OFF.
    static func buildTransparentDepthState(device: MTLDevice) -> MTLDepthStencilState? {
        let desc = MTLDepthStencilDescriptor()
        desc.label = "Transparent Depth"
        desc.depthCompareFunction = .less
        desc.isDepthWriteEnabled = false
        return device.makeDepthStencilState(descriptor: desc)
    }
}

enum MetalPipelineError: Error {
    case functionNotFound(String)
}
