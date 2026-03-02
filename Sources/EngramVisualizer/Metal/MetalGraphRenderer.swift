import MetalKit
import simd
import os
import SwiftUI

private let renderLog = Logger(subsystem: "io.engram.app", category: "MetalRenderer")

/// MTKViewDelegate that drives the raw Metal render loop.
/// Replaces RealityKit's `re::DrawingManager::commitFrameInternal()` bottleneck.
@MainActor
final class MetalGraphRenderer: NSObject {

    // Metal core
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let library: MTLLibrary
    let bufferPool: MetalBufferPool

    // Pipelines
    var nodePipeline: MTLRenderPipelineState!
    var edgePipeline: MTLRenderPipelineState!
    var labelPipeline: MTLRenderPipelineState!
    var nebulaPipeline: MTLRenderPipelineState!
    var flowPipeline: MTLRenderPipelineState!
    var mascotPipeline: MTLRenderPipelineState!
    var arcaneCirclePipeline: MTLRenderPipelineState!
    var nodeRingPipeline: MTLRenderPipelineState!
    var holoScreenPipeline: MTLRenderPipelineState!

    // Depth stencil states
    var opaqueDepthState: MTLDepthStencilState!
    var transparentDepthState: MTLDepthStencilState!

    // Compute pipelines
    var stampNodePipeline: MTLComputePipelineState!
    var stampLabelPipeline: MTLComputePipelineState!
    var stampEdgePipeline: MTLComputePipelineState!

    // Depth texture
    var depthTexture: MTLTexture?
    var drawableSize: CGSize = .zero

    // Sphere template
    var sphereTemplateBuffer: MTLBuffer?
    var sphereTemplateIndices: [UInt32] = []
    var vertsPerSphere: Int = 0
    var indicesPerSphere: Int = 0

    // Node batch buffers
    var nodeVertexBuffer: MTLBuffer?
    var nodeIndexBuffer: MTLBuffer?
    var nodeInstanceBuffer: MTLBuffer?
    var nodeStampParamsBuffer: MTLBuffer?
    var nodeInstanceCapacity: Int = 0
    var nodeVertexCapacity: Int = 0
    var actualNodeCount: Int = 0

    // Edge batch buffers
    var edgeVertexBuffer: MTLBuffer?
    var edgeIndexBuffer: MTLBuffer?
    var edgeInstanceBuffer: MTLBuffer?
    var edgeStampParamsBuffer: MTLBuffer?
    var edgeVertexCapacity: Int = 0
    var edgeInstanceCapacity: Int = 0
    var actualEdgeCount: Int = 0
    var actualEdgeVertexCount: Int = 0

    // Label batch buffers
    var labelVertexBuffer: MTLBuffer?
    var labelIndexBuffer: MTLBuffer?
    var labelInstanceBuffer: MTLBuffer?
    var labelStampParamsBuffer: MTLBuffer?
    var labelInstanceCapacity: Int = 0
    var labelVertexCapacity: Int = 0
    var actualLabelCount: Int = 0

    // Label atlas
    var labelAtlasTexture: MTLTexture?
    var labelSampler: MTLSamplerState?

    // Nebula buffers
    var nebulaVertexBuffer: MTLBuffer?
    var nebulaIndexBuffer: MTLBuffer?
    var nebulaVertexCapacity: Int = 0
    var actualNebulaVertexCount: Int = 0

    // Flow particle buffers
    var flowVertexBuffer: MTLBuffer?
    var flowIndexBuffer: MTLBuffer?
    var flowVertexCapacity: Int = 0
    var actualFlowParticleCount: Int = 0

    // Galaxy registry — used to draw fleet mascots
    weak var galaxyRegistryRef: GalaxyRegistry?

    // Shared state
    var camera: CameraController!
    var animationTime: Float = 0
    var maintenancePulse: Float = 0
    var frameCount: UInt64 = 0

    // Lighting setup (matches Graph3DScene's 3 directional lights)
    var lightingUniforms = LightingUniforms()

    // Callback for bridging render events to the scene manager
    var onFrameCallback: ((Float) -> Void)?

    init?(create: Bool = true) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            renderLog.error("[MetalGraphRenderer] No Metal device available")
            return nil
        }
        guard let queue = device.makeCommandQueue() else {
            renderLog.error("[MetalGraphRenderer] Failed to create command queue")
            return nil
        }
        guard let library = device.makeDefaultLibrary() else {
            renderLog.error("[MetalGraphRenderer] No default Metal library")
            return nil
        }

        self.device = device
        self.commandQueue = queue
        self.library = library
        self.bufferPool = MetalBufferPool(device: device)

        super.init()

        setupPipelines()
        setupSphereTemplate()
        setupLighting()
        setupLabelSampler()

        nodeStampParamsBuffer = device.makeBuffer(length: MemoryLayout<StampParams>.stride, options: .storageModeShared)
        labelStampParamsBuffer = device.makeBuffer(length: MemoryLayout<LabelStampParams>.stride, options: .storageModeShared)
        edgeStampParamsBuffer = device.makeBuffer(length: MemoryLayout<EdgeStampParams>.stride, options: .storageModeShared)

        renderLog.info("[MetalGraphRenderer] Initialized with device: \(device.name)")
    }

    // MARK: - Setup

    private func setupPipelines() {
        do {
            nodePipeline = try MetalPipelineBuilder.buildNodePipeline(device: device, library: library)
            edgePipeline = try MetalPipelineBuilder.buildEdgePipeline(device: device, library: library)
            labelPipeline = try MetalPipelineBuilder.buildLabelPipeline(device: device, library: library)
            nebulaPipeline = try MetalPipelineBuilder.buildNebulaPipeline(device: device, library: library)
            flowPipeline = try MetalPipelineBuilder.buildFlowPipeline(device: device, library: library)
            mascotPipeline = try MetalPipelineBuilder.buildMascotPipeline(device: device, library: library)
            arcaneCirclePipeline = try MetalPipelineBuilder.buildArcaneCirclePipeline(device: device, library: library)
            nodeRingPipeline = try MetalPipelineBuilder.buildNodeRingPipeline(device: device, library: library)
            holoScreenPipeline = try MetalPipelineBuilder.buildHoloScreenPipeline(device: device, library: library)
        } catch {
            renderLog.error("[MetalGraphRenderer] Pipeline build failed: \(error)")
        }

        opaqueDepthState = MetalPipelineBuilder.buildOpaqueDepthState(device: device)
        transparentDepthState = MetalPipelineBuilder.buildTransparentDepthState(device: device)

        // Compute pipelines for existing stamp kernels
        if let fn = library.makeFunction(name: "stamp_node_spheres") {
            stampNodePipeline = try? device.makeComputePipelineState(function: fn)
        }
        if let fn = library.makeFunction(name: "stamp_label_quads") {
            stampLabelPipeline = try? device.makeComputePipelineState(function: fn)
        }
        if let fn = library.makeFunction(name: "stamp_edge_cylinders") {
            stampEdgePipeline = try? device.makeComputePipelineState(function: fn)
        }
    }

    private func setupSphereTemplate() {
        let template = generateSphereTemplate(segments: 16, rings: 10)
        vertsPerSphere = template.vertices.count
        indicesPerSphere = template.indices.count
        sphereTemplateIndices = template.indices

        // Upload template vertices
        var gpuVerts = template.vertices.map { v in
            SphereTemplateVertex(position: v.0, normal: v.1)
        }
        let vertCount = gpuVerts.count
        sphereTemplateBuffer = gpuVerts.withUnsafeMutableBufferPointer { ptr in
            device.makeBuffer(bytes: ptr.baseAddress!,
                              length: MemoryLayout<SphereTemplateVertex>.stride * vertCount,
                              options: .storageModeShared)
        }
    }

    private func setupLighting() {
        // Key light: bright, upper-right
        let keyDir = normalize(SIMD3<Float>(3, 5, 2))
        lightingUniforms.directionalLights.0 = DirectionalLight(
            direction: keyDir, intensity: 1.2,
            color: SIMD3<Float>(1, 1, 1), _pad: 0
        )

        // Fill light: opposite side — compensates for missing IBL
        let fillDir = normalize(SIMD3<Float>(-4, 0, -1))
        lightingUniforms.directionalLights.1 = DirectionalLight(
            direction: fillDir, intensity: 0.35,
            color: SIMD3<Float>(0.5, 0.6, 0.8), _pad: 0
        )

        // Rim light: from below-behind
        let rimDir = normalize(SIMD3<Float>(0, -3, -4))
        lightingUniforms.directionalLights.2 = DirectionalLight(
            direction: rimDir, intensity: 0.45,
            color: SIMD3<Float>(0.6, 0.6, 0.8), _pad: 0
        )

        lightingUniforms.directionalLightCount = 3
        lightingUniforms.pointLightCount = 0
        lightingUniforms.ambientIntensity = 0.18
    }

    private func setupLabelSampler() {
        let desc = MTLSamplerDescriptor()
        desc.minFilter = .linear
        desc.magFilter = .linear
        desc.sAddressMode = .clampToEdge
        desc.tAddressMode = .clampToEdge
        labelSampler = device.makeSamplerState(descriptor: desc)
    }

    // MARK: - Depth Texture

    private func rebuildDepthTexture(width: Int, height: Int) {
        // Apple Silicon max 2D texture dimension is 16384
        guard width > 0, height > 0, width <= 16384, height <= 16384 else {
            renderLog.warning("[MetalGraphRenderer] rebuildDepthTexture rejected: \(width)x\(height)")
            return
        }
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float,
            width: width, height: height,
            mipmapped: false
        )
        desc.usage = .renderTarget
        desc.storageMode = .private
        depthTexture = device.makeTexture(descriptor: desc)
    }

    // MARK: - Buffer Management

    func ensureNodeBuffers(nodeCount: Int) {
        guard nodeCount > 0, vertsPerSphere > 0 else { return }
        guard nodeCount > nodeVertexCapacity else { return }
        let needed = max(nodeCount * 2, 512)

        let totalVerts = needed * vertsPerSphere
        let totalIndices = needed * indicesPerSphere

        // Vertex buffer (output of compute stamp)
        nodeVertexBuffer = device.makeBuffer(
            length: totalVerts * MemoryLayout<BatchVertex>.stride,
            options: .storageModeShared
        )

        // Index buffer (pre-filled)
        let indexBytes = totalIndices * MemoryLayout<UInt32>.stride
        nodeIndexBuffer = device.makeBuffer(length: indexBytes, options: .storageModeShared)
        if let buf = nodeIndexBuffer {
            let indices = buf.contents().bindMemory(to: UInt32.self, capacity: totalIndices)
            let vpn = UInt32(vertsPerSphere)
            for node in 0..<needed {
                let baseVert = UInt32(node) * vpn
                let baseIdx = node * indicesPerSphere
                for i in 0..<indicesPerSphere {
                    indices[baseIdx + i] = baseVert + sphereTemplateIndices[i]
                }
            }
        }

        nodeVertexCapacity = needed

        // Instance buffer
        if nodeInstanceCapacity < needed {
            nodeInstanceBuffer = device.makeBuffer(
                length: needed * MemoryLayout<NodeInstance>.stride,
                options: .storageModeShared
            )
            nodeInstanceCapacity = needed
        }
    }

    func ensureEdgeBuffers(edgeCount: Int) {
        let vertsPerEdge = 12
        let indicesPerEdge = 36
        guard edgeCount > edgeVertexCapacity else { return }
        let needed = max(edgeCount * 2, 2048)

        if true {
            let totalVerts = needed * vertsPerEdge
            let totalIndices = needed * indicesPerEdge

            edgeVertexBuffer = device.makeBuffer(
                length: totalVerts * MemoryLayout<BatchVertex>.stride,
                options: .storageModeShared
            )

            let indexBytes = totalIndices * MemoryLayout<UInt32>.stride
            edgeIndexBuffer = device.makeBuffer(length: indexBytes, options: .storageModeShared)
            if let buf = edgeIndexBuffer {
                let indices = buf.contents().bindMemory(to: UInt32.self, capacity: totalIndices)
                for i in 0..<needed {
                    let vBase = UInt32(i * 12)
                    let iBase = i * 36
                    for seg in 0..<6 {
                        let next = (seg + 1) % 6
                        let b0 = vBase + UInt32(seg)
                        let b1 = vBase + UInt32(next)
                        let t0 = vBase + UInt32(seg + 6)
                        let t1 = vBase + UInt32(next + 6)
                        let idx = iBase + seg * 6
                        indices[idx]     = b0; indices[idx + 1] = t0; indices[idx + 2] = t1
                        indices[idx + 3] = b0; indices[idx + 4] = t1; indices[idx + 5] = b1
                    }
                }
            }

            edgeVertexCapacity = needed
        }

        // Instance buffer for GPU compute stamp
        if needed > edgeInstanceCapacity {
            edgeInstanceBuffer = device.makeBuffer(
                length: needed * MemoryLayout<EdgeInstance>.stride,
                options: .storageModeShared
            )
            edgeInstanceCapacity = needed
        }
    }

    func ensureLabelBuffers(labelCount: Int) {
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

    // MARK: - Stamp Nodes (GPU Compute)

    func stampNodes(commandBuffer: MTLCommandBuffer) {
        guard actualNodeCount > 0,
              let pipeline = stampNodePipeline,
              let templateBuf = sphereTemplateBuffer,
              let instanceBuf = nodeInstanceBuffer,
              let paramsBuf = nodeStampParamsBuffer,
              let outputBuf = nodeVertexBuffer else { return }

        // Set params
        let paramsPtr = paramsBuf.contents().bindMemory(to: StampParams.self, capacity: 1)
        paramsPtr.pointee = StampParams(
            vertsPerNode: UInt32(vertsPerSphere),
            nodeCount: UInt32(actualNodeCount)
        )

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(templateBuf, offset: 0, index: 0)
        encoder.setBuffer(instanceBuf, offset: 0, index: 1)
        encoder.setBuffer(paramsBuf, offset: 0, index: 2)
        encoder.setBuffer(outputBuf, offset: 0, index: 3)

        let totalVerts = actualNodeCount * vertsPerSphere
        let threadgroupSize = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        let threadgroups = (totalVerts + threadgroupSize - 1) / threadgroupSize
        encoder.dispatchThreadgroups(
            MTLSize(width: threadgroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1)
        )
        encoder.endEncoding()

        // Zero stale vertices
        let bvStride = MemoryLayout<BatchVertex>.stride
        let writtenBytes = totalVerts * bvStride
        let totalBytes = nodeVertexCapacity * vertsPerSphere * bvStride
        if writtenBytes < totalBytes, let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.fill(buffer: outputBuf, range: writtenBytes..<totalBytes, value: 0)
            blit.endEncoding()
        }
    }

    // MARK: - Stamp Edges (GPU Compute)

    func stampEdges(commandBuffer: MTLCommandBuffer) {
        guard actualEdgeCount > 0,
              let pipeline = stampEdgePipeline,
              let instanceBuf = edgeInstanceBuffer,
              let paramsBuf = edgeStampParamsBuffer,
              let outputBuf = edgeVertexBuffer else { return }

        let vertsPerEdge: UInt32 = 12

        // Set params
        let paramsPtr = paramsBuf.contents().bindMemory(to: EdgeStampParams.self, capacity: 1)
        paramsPtr.pointee = EdgeStampParams(
            vertsPerEdge: vertsPerEdge,
            edgeCount: UInt32(actualEdgeCount)
        )

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(instanceBuf, offset: 0, index: 0)
        encoder.setBuffer(paramsBuf, offset: 0, index: 1)
        encoder.setBuffer(outputBuf, offset: 0, index: 2)

        let totalVerts = actualEdgeCount * Int(vertsPerEdge)
        let threadgroupSize = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        let threadgroups = (totalVerts + threadgroupSize - 1) / threadgroupSize
        encoder.dispatchThreadgroups(
            MTLSize(width: threadgroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1)
        )
        encoder.endEncoding()

        // Zero stale vertices
        let bvStride = MemoryLayout<BatchVertex>.stride
        let writtenBytes = totalVerts * bvStride
        let totalBytes = edgeVertexCapacity * Int(vertsPerEdge) * bvStride
        if writtenBytes < totalBytes, let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.fill(buffer: outputBuf, range: writtenBytes..<totalBytes, value: 0)
            blit.endEncoding()
        }

        // Update vertex count for draw call
        actualEdgeVertexCount = totalVerts
    }

    // MARK: - Stamp Labels (GPU Compute)

    func stampLabels(commandBuffer: MTLCommandBuffer) {
        guard actualLabelCount > 0,
              let pipeline = stampLabelPipeline,
              let instanceBuf = labelInstanceBuffer,
              let paramsBuf = labelStampParamsBuffer,
              let outputBuf = labelVertexBuffer else { return }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(instanceBuf, offset: 0, index: 0)
        encoder.setBuffer(paramsBuf, offset: 0, index: 1)
        encoder.setBuffer(outputBuf, offset: 0, index: 2)

        let threadgroupSize = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        let threadgroups = (actualLabelCount + threadgroupSize - 1) / threadgroupSize
        encoder.dispatchThreadgroups(
            MTLSize(width: threadgroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1)
        )
        encoder.endEncoding()

        // Zero stale
        let bvStride2 = MemoryLayout<BatchVertex>.stride
        let writtenBytes = actualLabelCount * 4 * bvStride2
        let totalBytes = labelVertexCapacity * 4 * bvStride2
        if writtenBytes < totalBytes, let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.fill(buffer: outputBuf, range: writtenBytes..<totalBytes, value: 0)
            blit.endEncoding()
        }
    }

    // MARK: - Sphere Template Generation

    private func generateSphereTemplate(segments: Int = 10, rings: Int = 6)
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

    #if ENGRAM_INSTRUMENTATION
    var drawTimingFile: UnsafeMutablePointer<FILE>? = nil
    var interFrameFile: UnsafeMutablePointer<FILE>? = nil
    var lastDrawEnd: Double = 0
    var canaryScheduleTime: Double = 0
    var runLoopFile: UnsafeMutablePointer<FILE>? = nil
    private var runLoopObserver: CFRunLoopObserver?
    private var runLoopSourceStart: Double = 0

    /// Install a CFRunLoopObserver that times each source/timer/observer pass.
    /// Logs passes >5ms to /tmp/runloop-timing.csv so we can see what's blocking.
    func installRunLoopObserver() {
        let fp = fopen("/tmp/runloop-timing.csv", "w")
        fputs("timestamp,activity,duration_ms\n", fp)
        runLoopFile = fp

        // BeforeSources observer — records start time
        let beforeObs = CFRunLoopObserverCreateWithHandler(nil, CFRunLoopActivity.beforeSources.rawValue, true, -1) { [weak self] _, _ in
            self?.runLoopSourceStart = CFAbsoluteTimeGetCurrent()
        }
        // AfterWaiting observer — records when run loop resumes from sleep
        let afterWaitObs = CFRunLoopObserverCreateWithHandler(nil, CFRunLoopActivity.afterWaiting.rawValue, true, -1) { [weak self] _, _ in
            self?.runLoopSourceStart = CFAbsoluteTimeGetCurrent()
        }
        // BeforeWaiting observer — logs time since last source start
        let beforeWaitObs = CFRunLoopObserverCreateWithHandler(nil, CFRunLoopActivity.beforeWaiting.rawValue, true, 1000000) { [weak self] _, _ in
            guard let self, self.runLoopSourceStart > 0, let f = self.runLoopFile else { return }
            let ms = (CFAbsoluteTimeGetCurrent() - self.runLoopSourceStart) * 1000.0
            if ms > 5 {
                let line = "\(String(format: "%.3f", CFAbsoluteTimeGetCurrent())),sources_to_wait,\(String(format: "%.1f", ms))\n"
                fputs(line, f)
                fflush(f)
            }
        }
        if let beforeObs {
            CFRunLoopAddObserver(CFRunLoopGetMain(), beforeObs, .commonModes)
        }
        if let afterWaitObs {
            CFRunLoopAddObserver(CFRunLoopGetMain(), afterWaitObs, .commonModes)
        }
        if let beforeWaitObs {
            CFRunLoopAddObserver(CFRunLoopGetMain(), beforeWaitObs, .commonModes)
        }
        runLoopObserver = beforeWaitObs
    }
    #endif
}

// MARK: - MTKViewDelegate

extension MetalGraphRenderer: MTKViewDelegate {

    nonisolated func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        MainActor.assumeIsolated {
            drawableSize = size
            rebuildDepthTexture(width: Int(size.width), height: Int(size.height))
        }
    }

    nonisolated func draw(in view: MTKView) {
        MainActor.assumeIsolated {
            drawFrame(in: view)
        }
    }

    private func drawFrame(in view: MTKView) {
        let dtSec = Float(1.0 / Double(max(view.preferredFramesPerSecond, 1)))
        animationTime += dtSec
        frameCount &+= 1

        #if ENGRAM_INSTRUMENTATION
        let drawStart = CFAbsoluteTimeGetCurrent()
        #endif

        // Let the scene manager do its per-frame work (input, simulation, data packing)
        onFrameCallback?(dtSec)

        #if ENGRAM_INSTRUMENTATION
        let afterCallback = CFAbsoluteTimeGetCurrent()
        #endif

        // Wait for a free frame slot
        bufferPool.waitForNextFrame()

        #if ENGRAM_INSTRUMENTATION
        let afterWait = CFAbsoluteTimeGetCurrent()
        #endif

        guard let drawable = view.currentDrawable,
              drawable.texture.width > 0,
              drawable.texture.height > 0,
              let renderPassDesc = view.currentRenderPassDescriptor,
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            bufferPool.signalFrameComplete()
            return
        }

        // Ensure depth texture exists
        if depthTexture == nil || drawableSize != CGSize(width: drawable.texture.width, height: drawable.texture.height) {
            rebuildDepthTexture(width: drawable.texture.width, height: drawable.texture.height)
        }
        renderPassDesc.depthAttachment.texture = depthTexture
        renderPassDesc.depthAttachment.loadAction = .clear
        renderPassDesc.depthAttachment.storeAction = .dontCare
        renderPassDesc.depthAttachment.clearDepth = 1.0

        // Build frame uniforms
        let aspect = Float(drawableSize.width / drawableSize.height)
        let viewMat = camera.viewMatrix()
        let projMat = camera.projectionMatrix(aspect: aspect)
        let vpMat = projMat * viewMat

        let frameUniforms = FrameUniforms(
            viewMatrix: viewMat,
            projectionMatrix: projMat,
            viewProjectionMatrix: vpMat,
            cameraPosition: camera.cameraPosition * camera.scaleFactor,
            time: animationTime,
            bgColor: SIMD3<Float>(0.051, 0.067, 0.09),
            fogNear: 0.6,
            fogFar: 7.0,
            maintenancePulse: maintenancePulse, _pad1: 0, _pad2: 0
        )
        bufferPool.updateFrameUniforms(frameUniforms)
        bufferPool.updateLightingUniforms(lightingUniforms)

        // GPU compute: stamp node spheres, edge cylinders, label quads
        stampNodes(commandBuffer: commandBuffer)
        stampEdges(commandBuffer: commandBuffer)
        stampLabels(commandBuffer: commandBuffer)

        // Render pass
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDesc) else {
            commandBuffer.commit()
            bufferPool.signalFrameComplete()
            return
        }

        let frameUniformBuf = bufferPool.currentFrameUniformBuffer
        let lightUniformBuf = bufferPool.currentLightingUniformBuffer

        // ── Opaque sub-pass: Nodes ──
        if actualNodeCount > 0,
           let vtxBuf = nodeVertexBuffer,
           let idxBuf = nodeIndexBuffer {
            encoder.setRenderPipelineState(nodePipeline)
            encoder.setDepthStencilState(opaqueDepthState)
            encoder.setCullMode(.back)
            encoder.setVertexBuffer(vtxBuf, offset: 0, index: 0)
            encoder.setVertexBuffer(frameUniformBuf, offset: 0, index: 1)
            encoder.setFragmentBuffer(frameUniformBuf, offset: 0, index: 0)
            encoder.setFragmentBuffer(lightUniformBuf, offset: 0, index: 1)
            let indexCount = actualNodeCount * indicesPerSphere
            encoder.drawIndexedPrimitives(
                type: .triangle,
                indexCount: indexCount,
                indexType: .uint32,
                indexBuffer: idxBuf,
                indexBufferOffset: 0
            )
        }

        // ── Opaque sub-pass: Mascot fleet (per-galaxy) ──
        if let mascotPSO = mascotPipeline, let registry = galaxyRegistryRef {
            for galaxy in registry.galaxies.values {
                galaxy.mascotFleet?.drawAll(
                    encoder: encoder,
                    frameUniformBuf: frameUniformBuf,
                    lightUniformBuf: lightUniformBuf,
                    pipeline: mascotPSO,
                    depthState: opaqueDepthState
                )
            }
        }

        // ── Transparent sub-pass ──
        encoder.setDepthStencilState(transparentDepthState)
        encoder.setCullMode(.none)

        // Nebulae (drawn first — largest, most distant, lowest alpha)
        if actualNebulaVertexCount > 0,
           let vtxBuf = nebulaVertexBuffer,
           let idxBuf = nebulaIndexBuffer {
            encoder.setRenderPipelineState(nebulaPipeline)
            encoder.setVertexBuffer(vtxBuf, offset: 0, index: 0)
            encoder.setVertexBuffer(frameUniformBuf, offset: 0, index: 1)
            encoder.setFragmentBuffer(frameUniformBuf, offset: 0, index: 0)
            let indexCount = (actualNebulaVertexCount / 4) * 6
            encoder.drawIndexedPrimitives(
                type: .triangle,
                indexCount: indexCount,
                indexType: .uint32,
                indexBuffer: idxBuf,
                indexBufferOffset: 0
            )
        }

        // Edges
        if actualEdgeVertexCount > 0,
           let vtxBuf = edgeVertexBuffer,
           let idxBuf = edgeIndexBuffer {
            encoder.setRenderPipelineState(edgePipeline)
            encoder.setVertexBuffer(vtxBuf, offset: 0, index: 0)
            encoder.setVertexBuffer(frameUniformBuf, offset: 0, index: 1)
            encoder.setFragmentBuffer(frameUniformBuf, offset: 0, index: 0)
            let edgeCount = actualEdgeVertexCount / 12
            let indexCount = edgeCount * 36
            encoder.drawIndexedPrimitives(
                type: .triangle,
                indexCount: indexCount,
                indexType: .uint32,
                indexBuffer: idxBuf,
                indexBufferOffset: 0
            )
        }

        // Labels
        if actualLabelCount > 0,
           let vtxBuf = labelVertexBuffer,
           let idxBuf = labelIndexBuffer {
            encoder.setRenderPipelineState(labelPipeline)
            encoder.setVertexBuffer(vtxBuf, offset: 0, index: 0)
            encoder.setVertexBuffer(frameUniformBuf, offset: 0, index: 1)
            encoder.setFragmentBuffer(frameUniformBuf, offset: 0, index: 0)
            if let atlas = labelAtlasTexture {
                encoder.setFragmentTexture(atlas, index: 0)
            }
            if let sampler = labelSampler {
                encoder.setFragmentSamplerState(sampler, index: 0)
            }
            let indexCount = actualLabelCount * 6
            encoder.drawIndexedPrimitives(
                type: .triangle,
                indexCount: indexCount,
                indexType: .uint32,
                indexBuffer: idxBuf,
                indexBufferOffset: 0
            )
        }

        // Mascot thruster particles (grouped by effect type for minimal pipeline switches)
        if let registry = galaxyRegistryRef {
            for galaxy in registry.galaxies.values {
                galaxy.mascotFleet?.drawAllThrusterParticles(
                    encoder: encoder, frameUniformBuf: frameUniformBuf, pipeline: flowPipeline
                )
            }
        }

        // Mascot arcane circles (additive glow)
        if let arcanePSO = arcaneCirclePipeline, let registry = galaxyRegistryRef {
            for galaxy in registry.galaxies.values {
                galaxy.mascotFleet?.drawAllArcaneCircles(
                    encoder: encoder, frameUniformBuf: frameUniformBuf, pipeline: arcanePSO
                )
            }
        }

        // Node inspection rings (additive glow)
        if let ringPSO = nodeRingPipeline, let registry = galaxyRegistryRef {
            for galaxy in registry.galaxies.values {
                galaxy.mascotFleet?.drawAllNodeRings(
                    encoder: encoder, frameUniformBuf: frameUniformBuf, pipeline: ringPSO
                )
            }
        }

        // Mascot holo info screens (alpha blended)
        if let holoPSO = holoScreenPipeline, let registry = galaxyRegistryRef {
            for galaxy in registry.galaxies.values {
                galaxy.mascotFleet?.drawAllHoloScreens(
                    encoder: encoder, frameUniformBuf: frameUniformBuf, pipeline: holoPSO
                )
            }
        }

        // Flow particles
        if actualFlowParticleCount > 0,
           let vtxBuf = flowVertexBuffer,
           let idxBuf = flowIndexBuffer {
            encoder.setRenderPipelineState(flowPipeline)
            encoder.setVertexBuffer(vtxBuf, offset: 0, index: 0)
            encoder.setVertexBuffer(frameUniformBuf, offset: 0, index: 1)
            let indexCount = actualFlowParticleCount * 6
            encoder.drawIndexedPrimitives(
                type: .triangle,
                indexCount: indexCount,
                indexType: .uint32,
                indexBuffer: idxBuf,
                indexBufferOffset: 0
            )
        }

        encoder.endEncoding()

        commandBuffer.present(drawable)
        let pool = bufferPool
        commandBuffer.addCompletedHandler { _ in
            pool.signalFrameComplete()
        }
        commandBuffer.commit()

        #if ENGRAM_INSTRUMENTATION
        let drawEnd = CFAbsoluteTimeGetCurrent()
        let callbackMs = (afterCallback - drawStart) * 1000.0
        let waitMs = (afterWait - afterCallback) * 1000.0
        let encodeMs = (drawEnd - afterWait) * 1000.0
        let totalDrawMs = (drawEnd - drawStart) * 1000.0

        // Inter-frame analysis: how long between drawFrame end and next drawFrame start
        let interFrameMs: Double
        if lastDrawEnd > 0 {
            interFrameMs = (drawStart - lastDrawEnd) * 1000.0
        } else {
            interFrameMs = 0
        }

        // Process canary: how long did the main queue block take to execute?
        // If canaryDelayMs >> 0, something is occupying the main thread after drawFrame
        let canaryDelayMs: Double
        if canaryScheduleTime > 0 {
            // canaryScheduleTime was set by the previous frame's canary when it executed
            // We compare it to when drawFrame ended to see the dispatch delay
            canaryDelayMs = (canaryScheduleTime - lastDrawEnd) * 1000.0
        } else {
            canaryDelayMs = 0
        }

        lastDrawEnd = drawEnd

        // Schedule a canary block on the main queue to detect main-thread contention
        // If the main thread is free, this executes almost immediately after drawFrame returns.
        // If something else runs between frames, this is delayed.
        DispatchQueue.main.async { [weak self] in
            self?.canaryScheduleTime = CFAbsoluteTimeGetCurrent()
        }

        if drawTimingFile == nil {
            drawTimingFile = fopen("/tmp/draw-timing.csv", "w")
            if let f = drawTimingFile {
                fputs("frame,total_draw_ms,callback_ms,wait_ms,encode_ms,inter_frame_ms,canary_delay_ms\n", f)
            }
        }
        if let f = drawTimingFile {
            let line = "\(frameCount),\(String(format: "%.2f", totalDrawMs)),\(String(format: "%.2f", callbackMs)),\(String(format: "%.2f", waitMs)),\(String(format: "%.2f", encodeMs)),\(String(format: "%.2f", interFrameMs)),\(String(format: "%.2f", canaryDelayMs))\n"
            fputs(line, f)
            fflush(f)
        }
        #endif
    }
}
