@preconcurrency import Metal
import CEngramSceneTypes

/// Stateless render encoding — all pipelines, depth states, and buffers passed explicitly.
/// Extracted from MetalGraphRenderer's render pass section.
/// Mascot rendering is NOT here — MascotFleet has its own draw methods.
@MainActor
public struct RenderEncoder {

    /// Encode instanced node spheres (opaque sub-pass).
    public static func encodeOpaqueNodes(
        encoder: MTLRenderCommandEncoder,
        pipeline: MTLRenderPipelineState,
        depthState: MTLDepthStencilState,
        templateBuffer: MTLBuffer,
        templateIndexBuffer: MTLBuffer,
        instanceBuffer: MTLBuffer,
        frameUniformBuffer: MTLBuffer,
        lightUniformBuffer: MTLBuffer,
        indicesPerSphere: Int,
        instanceCount: Int
    ) {
        guard instanceCount > 0 else { return }
        encoder.setRenderPipelineState(pipeline)
        encoder.setDepthStencilState(depthState)
        encoder.setCullMode(.back)
        encoder.setVertexBuffer(templateBuffer, offset: 0, index: 0)
        encoder.setVertexBuffer(frameUniformBuffer, offset: 0, index: 1)
        encoder.setVertexBuffer(instanceBuffer, offset: 0, index: 2)
        encoder.setFragmentBuffer(frameUniformBuffer, offset: 0, index: 0)
        encoder.setFragmentBuffer(lightUniformBuffer, offset: 0, index: 1)
        encoder.drawIndexedPrimitives(
            type: .triangle,
            indexCount: indicesPerSphere,
            indexType: .uint32,
            indexBuffer: templateIndexBuffer,
            indexBufferOffset: 0,
            instanceCount: instanceCount
        )
    }

    /// Encode nebula billboards (transparent, drawn first — largest, most distant).
    public static func encodeNebulae(
        encoder: MTLRenderCommandEncoder,
        pipeline: MTLRenderPipelineState,
        vertexBuffer: MTLBuffer,
        indexBuffer: MTLBuffer,
        frameUniformBuffer: MTLBuffer,
        vertexCount: Int
    ) {
        guard vertexCount > 0 else { return }
        encoder.setRenderPipelineState(pipeline)
        encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        encoder.setVertexBuffer(frameUniformBuffer, offset: 0, index: 1)
        encoder.setFragmentBuffer(frameUniformBuffer, offset: 0, index: 0)
        let indexCount = (vertexCount / 4) * 6
        encoder.drawIndexedPrimitives(
            type: .triangle,
            indexCount: indexCount,
            indexType: .uint32,
            indexBuffer: indexBuffer,
            indexBufferOffset: 0
        )
    }

    /// Encode instanced edges (transparent, instanced cylinder).
    public static func encodeTransparentEdges(
        encoder: MTLRenderCommandEncoder,
        pipeline: MTLRenderPipelineState,
        cylinderIndexBuffer: MTLBuffer,
        instanceBuffer: MTLBuffer,
        frameUniformBuffer: MTLBuffer,
        instanceCount: Int
    ) {
        guard instanceCount > 0 else { return }
        encoder.setRenderPipelineState(pipeline)
        encoder.setVertexBuffer(frameUniformBuffer, offset: 0, index: 0)
        encoder.setVertexBuffer(instanceBuffer, offset: 0, index: 1)
        encoder.setFragmentBuffer(frameUniformBuffer, offset: 0, index: 0)
        encoder.drawIndexedPrimitives(
            type: .triangle,
            indexCount: 36,
            indexType: .uint32,
            indexBuffer: cylinderIndexBuffer,
            indexBufferOffset: 0,
            instanceCount: instanceCount
        )
    }

    /// Encode text labels (transparent, textured quads).
    public static func encodeLabels(
        encoder: MTLRenderCommandEncoder,
        pipeline: MTLRenderPipelineState,
        vertexBuffer: MTLBuffer,
        indexBuffer: MTLBuffer,
        frameUniformBuffer: MTLBuffer,
        atlasTexture: MTLTexture?,
        sampler: MTLSamplerState?,
        labelCount: Int
    ) {
        guard labelCount > 0 else { return }
        encoder.setRenderPipelineState(pipeline)
        encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        encoder.setVertexBuffer(frameUniformBuffer, offset: 0, index: 1)
        encoder.setFragmentBuffer(frameUniformBuffer, offset: 0, index: 0)
        if let atlas = atlasTexture {
            encoder.setFragmentTexture(atlas, index: 0)
        }
        if let sampler {
            encoder.setFragmentSamplerState(sampler, index: 0)
        }
        let indexCount = labelCount * 6
        encoder.drawIndexedPrimitives(
            type: .triangle,
            indexCount: indexCount,
            indexType: .uint32,
            indexBuffer: indexBuffer,
            indexBufferOffset: 0
        )
    }

    /// Encode flow particles (transparent, billboard quads).
    public static func encodeFlowParticles(
        encoder: MTLRenderCommandEncoder,
        pipeline: MTLRenderPipelineState,
        vertexBuffer: MTLBuffer,
        indexBuffer: MTLBuffer,
        frameUniformBuffer: MTLBuffer,
        particleCount: Int
    ) {
        guard particleCount > 0 else { return }
        encoder.setRenderPipelineState(pipeline)
        encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        encoder.setVertexBuffer(frameUniformBuffer, offset: 0, index: 1)
        let indexCount = particleCount * 6
        encoder.drawIndexedPrimitives(
            type: .triangle,
            indexCount: indexCount,
            indexType: .uint32,
            indexBuffer: indexBuffer,
            indexBufferOffset: 0
        )
    }
}
