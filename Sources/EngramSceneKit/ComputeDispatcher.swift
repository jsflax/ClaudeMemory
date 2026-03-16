@preconcurrency import Metal
import CEngramSceneTypes

/// Stateless GPU compute dispatch — all inputs passed explicitly.
/// Extracted from MetalGraphRenderer's packNodesGPU, packEdgesGPU, packNebulaGPU, stampLabels.
@MainActor
public struct ComputeDispatcher {

    /// GPU dispatch: NodePackInput[] + positions[] → NodeInstance[] + point lights.
    public static func packNodes(
        encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        inputBuffer: MTLBuffer,
        instanceBuffer: MTLBuffer,
        lightOutputBuffer: MTLBuffer,
        lightCountBuffer: MTLBuffer,
        paramsBuffer: MTLBuffer,
        projIdxBuffer: MTLBuffer,
        positionBuffer: MTLBuffer,
        nodeCount: Int
    ) {
        guard nodeCount > 0 else { return }

        // Reset point light counter
        lightCountBuffer.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = 0

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(instanceBuffer, offset: 0, index: 1)
        encoder.setBuffer(lightOutputBuffer, offset: 0, index: 2)
        encoder.setBuffer(lightOutputBuffer, offset: 0, index: 3)
        encoder.setBuffer(lightCountBuffer, offset: 0, index: 4)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 5)
        encoder.setBuffer(projIdxBuffer, offset: 0, index: 6)
        encoder.setBuffer(positionBuffer, offset: 0, index: 7)

        let threadgroupSize = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        let threadgroups = (nodeCount + threadgroupSize - 1) / threadgroupSize
        encoder.dispatchThreadgroups(
            MTLSize(width: threadgroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1)
        )
    }

    /// GPU dispatch: EdgeDescriptor[] + positions[] → EdgeInstance[].
    public static func packEdges(
        encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        descriptorBuffer: MTLBuffer,
        positionBuffer: MTLBuffer,
        paramsBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        edgeCount: Int
    ) {
        guard edgeCount > 0 else { return }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(descriptorBuffer, offset: 0, index: 0)
        encoder.setBuffer(positionBuffer, offset: 0, index: 1)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 2)
        encoder.setBuffer(outputBuffer, offset: 0, index: 3)

        let threadgroupSize = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        let threadgroups = (edgeCount + threadgroupSize - 1) / threadgroupSize
        encoder.dispatchThreadgroups(
            MTLSize(width: threadgroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1)
        )
    }

    /// GPU dispatch: NebulaGroupInput[] → NebulaQuadVertex[].
    public static func packNebulae(
        encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        inputBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        paramsBuffer: MTLBuffer,
        groupCount: Int
    ) {
        guard groupCount > 0 else { return }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 2)

        let threadgroupSize = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        let threadgroups = (groupCount + threadgroupSize - 1) / threadgroupSize
        encoder.dispatchThreadgroups(
            MTLSize(width: threadgroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1)
        )
    }

    /// GPU dispatch: LabelInstance[] → BatchVertex[] (label quad stamping).
    public static func stampLabels(
        encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        instanceBuffer: MTLBuffer,
        paramsBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        labelCount: Int
    ) {
        guard labelCount > 0 else { return }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(instanceBuffer, offset: 0, index: 0)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)

        let threadgroupSize = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        let threadgroups = (labelCount + threadgroupSize - 1) / threadgroupSize
        encoder.dispatchThreadgroups(
            MTLSize(width: threadgroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1)
        )
    }
}
