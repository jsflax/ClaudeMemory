import MetalKit
import CEngramSceneTypes
import Synchronization
import simd
import os

private let renderLog = Logger(subsystem: "io.engram.app", category: "FrameOrchestrator")

/// Extension point for app-target draw passes (e.g. MascotFleet).
/// FrameOrchestrator calls these at the appropriate points in the render loop.
public protocol ExtraDrawPass {
    @MainActor func encodeCompute(encoder: MTLComputeCommandEncoder, dt: Float, camera: CameraState)
    @MainActor func encodeOpaque(encoder: MTLRenderCommandEncoder, frameUniform: MTLBuffer, lightUniform: MTLBuffer)
    @MainActor func encodeTransparent(encoder: MTLRenderCommandEncoder, frameUniform: MTLBuffer)
}

/// Thin MTKViewDelegate coordinator that orchestrates the render loop.
/// All subsystems are injected — zero app-target dependencies.
/// Replaces MetalGraphRenderer as the single MTKViewDelegate.
@MainActor
public final class FrameOrchestrator: NSObject, MTKViewDelegate {

    // Metal core
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let library: MTLLibrary

    // Subsystems
    public let forceEngine: ForceEngine
    public let bufferManager: InstanceBufferManager
    public let bufferPool: MetalBufferPool
    public let gpuSim: GPUSimulationState

    // Camera (value type — written by scene manager each frame)
    public var camera: CameraState = CameraState()

    // Pipelines (render)
    public var nodePipeline: MTLRenderPipelineState!
    public var edgePipeline: MTLRenderPipelineState!
    public var labelPipeline: MTLRenderPipelineState!
    public var nebulaPipeline: MTLRenderPipelineState!
    public var flowPipeline: MTLRenderPipelineState!

    // Depth stencil states
    public var opaqueDepthState: MTLDepthStencilState!
    public var transparentDepthState: MTLDepthStencilState!

    // Compute pipelines
    public var stampLabelPipeline: MTLComputePipelineState?
    public var packEdgePipeline: MTLComputePipelineState?
    public var packNodePipeline: MTLComputePipelineState?
    public var packNebulaPipeline: MTLComputePipelineState?

    // Depth texture
    private var depthTexture: MTLTexture?
    private var drawableSize: CGSize = .zero

    // Frame state
    public var animationTime: Float = 0
    public var maintenancePulse: Float = 0
    public var frameCount: UInt64 = 0
    public var lightingUniforms = LightingUniforms()

    // Extension points
    public var extraDrawPasses: [ExtraDrawPass] = []
    public var onFrameCallback: ((Float) -> Void)?

    /// Callback to get the current force simulation snapshot each frame.
    /// Set by the scene manager. Returns nil when forces shouldn't be encoded.
    public var forceSnapshotProvider: (() -> ForceEngine.SimulationSnapshot?)?

    /// Called after force results are delivered.
    public var onForceResult: ((ForceResult) -> Void)?

    // Force in-flight gate — skip force encoding if previous CB hasn't completed.
    // Prevents GPU saturation when force compute exceeds frame budget.
    private var forceInFlight = false
    // Atomic flag set by force CB completion handler (background thread).
    // Read + cleared on main thread at start of next drawFrame. No Task needed.
    private let forceReadbackReady = Atomic<Bool>(false)

    // Render ramp — avoid 0 → 8K nodes + 91K edges in one frame.
    // Ramps up by 500 per frame so the GPU has time to absorb the load.
    private var nodeRenderCap: Int = 0
    private var edgeRenderCap: Int = 0
    private let renderRampPerFrame: Int = 500

    // Stall detection
    private var lastFrameTime: Double = 0
    private var stallCanaryScheduleTime: Double = 0
    private static let stallLog = OSLog(subsystem: "io.engram.app", category: "FrameStall")
    private static var gpuLogConfigured = false

    public init?(device: MTLDevice, library: MTLLibrary) {
        guard let queue = device.makeCommandQueue() else {
            renderLog.error("[FrameOrchestrator] Failed to create command queue")
            return nil
        }

        if !Self.gpuLogConfigured {
            GPULog.configure(path: "/Users/jason/localdev/Engram/gpu-forces.log")
            Self.gpuLogConfigured = true
        }

        self.device = device
        self.commandQueue = queue
        self.library = library
        self.bufferPool = MetalBufferPool(device: device)
        self.bufferManager = InstanceBufferManager(device: device)

        // Create MetalForceCompute for ForceEngine
        guard let fc = MetalForceCompute(device: device, library: library) else {
            renderLog.error("[FrameOrchestrator] Failed to create MetalForceCompute")
            return nil
        }
        self.forceEngine = ForceEngine(forceCompute: fc)
        self.gpuSim = GPUSimulationState(device: device, library: library)

        super.init()

        setupPipelines()
        setupLighting()

        GPULog.log("INSTANCED RENDERING: nodes=\(bufferManager.sphereTemplateBuffer != nil) (\(bufferManager.vertsPerSphere)v/\(bufferManager.indicesPerSphere)i template), edges=\(bufferManager.cylinderIndexBuffer != nil) (12v/36i template)")
    }

    // MARK: - Setup

    private func setupPipelines() {
        do {
            nodePipeline = try MetalPipelineBuilder.buildNodePipeline(device: device, library: library)
            edgePipeline = try MetalPipelineBuilder.buildEdgePipeline(device: device, library: library)
            labelPipeline = try MetalPipelineBuilder.buildLabelPipeline(device: device, library: library)
            nebulaPipeline = try MetalPipelineBuilder.buildNebulaPipeline(device: device, library: library)
            flowPipeline = try MetalPipelineBuilder.buildFlowPipeline(device: device, library: library)
        } catch {
            renderLog.error("[FrameOrchestrator] Pipeline build failed: \(error)")
        }

        opaqueDepthState = MetalPipelineBuilder.buildOpaqueDepthState(device: device)
        transparentDepthState = MetalPipelineBuilder.buildTransparentDepthState(device: device)

        // Compute pipelines
        if let fn = library.makeFunction(name: "stamp_label_quads") {
            stampLabelPipeline = try? device.makeComputePipelineState(function: fn)
        }
        if let fn = library.makeFunction(name: "pack_edge_instances") {
            packEdgePipeline = try? device.makeComputePipelineState(function: fn)
        }
        if let fn = library.makeFunction(name: "pack_node_instances") {
            packNodePipeline = try? device.makeComputePipelineState(function: fn)
        }
        if let fn = library.makeFunction(name: "pack_nebula_vertices") {
            packNebulaPipeline = try? device.makeComputePipelineState(function: fn)
        }
    }

    private func setupLighting() {
        let keyDir = normalize(SIMD3<Float>(3, 5, 2))
        lightingUniforms.directionalLights.0 = DirectionalLight(
            direction: keyDir, intensity: 1.2,
            color: SIMD3<Float>(1, 1, 1), _pad: 0
        )
        let fillDir = normalize(SIMD3<Float>(-4, 0, -1))
        lightingUniforms.directionalLights.1 = DirectionalLight(
            direction: fillDir, intensity: 0.35,
            color: SIMD3<Float>(0.5, 0.6, 0.8), _pad: 0
        )
        let rimDir = normalize(SIMD3<Float>(0, -3, -4))
        lightingUniforms.directionalLights.2 = DirectionalLight(
            direction: rimDir, intensity: 0.45,
            color: SIMD3<Float>(0.6, 0.6, 0.8), _pad: 0
        )
        lightingUniforms.directionalLightCount = 3
        lightingUniforms.pointLightCount = 0
        lightingUniforms.ambientIntensity = 0.18
    }

    // MARK: - Depth Texture

    private func rebuildDepthTexture(width: Int, height: Int) {
        guard width > 0, height > 0, width <= 16384, height <= 16384 else { return }
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: width, height: height, mipmapped: false
        )
        desc.usage = .renderTarget
        desc.storageMode = .private
        depthTexture = device.makeTexture(descriptor: desc)
    }

    // MARK: - MTKViewDelegate

    nonisolated public func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        MainActor.assumeIsolated {
            drawableSize = size
            rebuildDepthTexture(width: Int(size.width), height: Int(size.height))
        }
    }

    nonisolated public func draw(in view: MTKView) {
        MainActor.assumeIsolated {
            drawFrame(in: view)
        }
    }

    // MARK: - Draw Frame

    private func drawFrame(in view: MTKView) {
        let frameStart = CFAbsoluteTimeGetCurrent()
        let dtSec = Float(1.0 / Double(max(view.preferredFramesPerSecond, 1)))
        animationTime += dtSec
        frameCount &+= 1

        // Check if previous force CB completed — read back results on main thread.
        // No Task { @MainActor } needed — avoids run loop stall from queued tasks.
        if forceReadbackReady.exchange(false, ordering: .acquiring) {
            forceInFlight = false
            if let forces = forceEngine.forceCompute.readBackForces() {
                let result = ForceResult(fx: forces.fx, fy: forces.fy, fz: forces.fz)
                onForceResult?(result)
            }
        }

        // --- CB 1: Force compute (BEFORE drawable) ---
        // Separate command buffer gives GPU scheduler a preemption point.
        // GPU works on forces while CPU does renderTick + waits for drawable.
        // Skip if previous force CB still in flight — prevents GPU saturation.
        if !forceInFlight, let snapshot = forceSnapshotProvider?() {
            let fc = forceEngine.forceCompute
            if !snapshot.isSettled, snapshot.nodeCount > 1, fc.isFullSimAvailable {
                if snapshot.topologyDirty {
                    fc.setTopologyDirty(edges: snapshot.edgeIndices)
                }
                if fc.topologyDirtyForDispatch {
                    fc.rebuildTopology(
                        nodeCount: snapshot.nodeCount,
                        projectGroups: snapshot.projectGroups,
                        topicGroups: snapshot.topicGroups)
                }
                if let forceCB = commandQueue.makeCommandBuffer(),
                   let forceEncoder = forceCB.makeComputeCommandEncoder() {
                    fc.encodeForces(
                        encoder: forceEncoder,
                        x: snapshot.posX, y: snapshot.posY, z: snapshot.posZ,
                        projectGroups: snapshot.projectGroups, topicGroups: snapshot.topicGroups,
                        galaxyGroups: snapshot.galaxyGroups, galaxyCenters: snapshot.galaxyCenters,
                        chargeStrength: snapshot.chargeStrength,
                        crossChargeMultiplier: snapshot.crossChargeMultiplier,
                        sameTopicChargeScale: snapshot.sameTopicChargeScale,
                        sameProjectChargeScale: snapshot.sameProjectChargeScale,
                        springLength: snapshot.springLength,
                        crossProjectSpringLength: snapshot.crossProjectSpringLength,
                        springStrength: snapshot.springStrength,
                        cohesionStrength: snapshot.cohesionStrength,
                        centroidRepulsion: snapshot.centroidRepulsion,
                        topicCohesionStrength: snapshot.topicCohesionStrength,
                        topicCentroidRepulsion: snapshot.topicCentroidRepulsion,
                        centerStrength: snapshot.centerStrength, center: snapshot.center,
                        alpha: snapshot.alpha, damping: snapshot.damping, maxSpeed: snapshot.maxSpeed)
                    forceEncoder.endEncoding()
                    let onForce = onForceResult
                    forceCB.addCompletedHandler { @Sendable _ in
                        // Set flag — main thread reads back on next frame.
                        // No Task { @MainActor } — those pile up and stall the run loop.
                        self.forceReadbackReady.store(true, ordering: .releasing)
                    }
                    // Commit BEFORE drawable — GPU starts forces while CPU does renderTick
                    forceInFlight = true
                    forceCB.commit()
                }
            }
        }

        // --- CPU work: renderTick (runs while GPU does force compute) ---
        let cbStart = CFAbsoluteTimeGetCurrent()
        onFrameCallback?(dtSec)
        let cbMs = (CFAbsoluteTimeGetCurrent() - cbStart) * 1000.0

        // --- Acquire drawable (by now force CB may have completed) ---
        let drawableStart = CFAbsoluteTimeGetCurrent()
        guard let drawable = view.currentDrawable else {
            let waitMs = (CFAbsoluteTimeGetCurrent() - drawableStart) * 1000.0
            if waitMs > 5 {
                GPULog.log("DRAWABLE nil \(String(format: "%.0f", waitMs))ms frame=\(frameCount)")
            }
            return
        }
        let drawableWaitMs = (CFAbsoluteTimeGetCurrent() - drawableStart) * 1000.0
        if drawableWaitMs > 5 {
            GPULog.log("DRAWABLE slow \(String(format: "%.0f", drawableWaitMs))ms frame=\(frameCount)")
        }

        let encodeStart = CFAbsoluteTimeGetCurrent()
        guard drawable.texture.width > 0,
              drawable.texture.height > 0,
              let renderPassDesc = view.currentRenderPassDescriptor,
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            return
        }
        bufferPool.advanceFrame()

        // Ensure depth texture
        if depthTexture == nil || drawableSize != CGSize(width: drawable.texture.width, height: drawable.texture.height) {
            rebuildDepthTexture(width: drawable.texture.width, height: drawable.texture.height)
        }
        renderPassDesc.depthAttachment.texture = depthTexture
        renderPassDesc.depthAttachment.loadAction = .clear
        renderPassDesc.depthAttachment.storeAction = .dontCare
        renderPassDesc.depthAttachment.clearDepth = 1.0

        // Build frame uniforms from camera
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

        // --- CB 2: Packing + Render (AFTER drawable) ---
        // Same queue as force CB — GPU finishes forces first, then runs this.
        let bm = bufferManager
        if let computeEncoder = commandBuffer.makeComputeCommandEncoder() {
            // Pack nodes
            if let pl = packNodePipeline,
               let inputBuf = bm.nodePackInputBuffer,
               let instanceBuf = bm.nodeInstanceBuffer,
               let lightBuf = bm.pointLightOutputBuffer,
               let lightCountBuf = bm.pointLightCountBuffer,
               let paramsBuf = bm.nodePackParamsBuffer,
               let projIdxBuf = bm.nodeProjectIdxBuffer,
               let posBuf = bm.nodePositionBuffer {
                ComputeDispatcher.packNodes(
                    encoder: computeEncoder, pipeline: pl,
                    inputBuffer: inputBuf, instanceBuffer: instanceBuf,
                    lightOutputBuffer: lightBuf, lightCountBuffer: lightCountBuf,
                    paramsBuffer: paramsBuf, projIdxBuffer: projIdxBuf,
                    positionBuffer: posBuf,
                    nodeCount: bm.actualNodeCount)
            }

            // Pack edges
            if bm.edgeDataDirty,
               let pl = packEdgePipeline,
               let descBuf = bm.edgeDescriptorBuffer,
               let posBuf = bm.nodePositionBuffer,
               let paramsBuf = bm.packEdgeParamsBuffer,
               let outputBuf = bm.edgeInstanceBuffer {
                ComputeDispatcher.packEdges(
                    encoder: computeEncoder, pipeline: pl,
                    descriptorBuffer: descBuf, positionBuffer: posBuf,
                    paramsBuffer: paramsBuf, outputBuffer: outputBuf,
                    edgeCount: bm.actualEdgeCount)
                bm.edgeDataDirty = false
            }

            // Pack nebulae
            if let pl = packNebulaPipeline,
               let inputBuf = bm.nebulaGroupInputBuffer,
               let outputBuf = bm.nebulaVertexBuffer,
               let paramsBuf = bm.nebulaPackParamsBuffer {
                ComputeDispatcher.packNebulae(
                    encoder: computeEncoder, pipeline: pl,
                    inputBuffer: inputBuf, outputBuffer: outputBuf,
                    paramsBuffer: paramsBuf, groupCount: bm.nebulaGroupCount)
                // Memory barrier for nebula vertex buffer
                if bm.nebulaGroupCount > 0, let buf = bm.nebulaVertexBuffer {
                    computeEncoder.memoryBarrier(resources: [buf])
                }
            }

            // Stamp labels
            if let pl = stampLabelPipeline,
               let instanceBuf = bm.labelInstanceBuffer,
               let paramsBuf = bm.labelStampParamsBuffer,
               let outputBuf = bm.labelVertexBuffer {
                ComputeDispatcher.stampLabels(
                    encoder: computeEncoder, pipeline: pl,
                    instanceBuffer: instanceBuf, paramsBuffer: paramsBuf,
                    outputBuffer: outputBuf, labelCount: bm.actualLabelCount)
            }

            // Extra draw passes (mascot compute, etc.)
            for pass in extraDrawPasses {
                pass.encodeCompute(encoder: computeEncoder, dt: dtSec, camera: camera)
            }

            computeEncoder.endEncoding()
        }

        // 6. Render pass
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDesc) else {
            commandBuffer.commit()
            return
        }

        let frameUniformBuf = bufferPool.currentFrameUniformBuffer
        let lightUniformBuf = bufferPool.currentLightingUniformBuffer

        // Opaque: Nodes (ramped)
        if bm.actualNodeCount > 0 {
            nodeRenderCap = min(nodeRenderCap + renderRampPerFrame, bm.actualNodeCount)
        } else {
            nodeRenderCap = 0
        }
        if let templateBuf = bm.sphereTemplateBuffer,
           let templateIdxBuf = bm.sphereTemplateIndexBuffer,
           let instanceBuf = bm.nodeInstanceBuffer {
            RenderEncoder.encodeOpaqueNodes(
                encoder: renderEncoder, pipeline: nodePipeline,
                depthState: opaqueDepthState,
                templateBuffer: templateBuf, templateIndexBuffer: templateIdxBuf,
                instanceBuffer: instanceBuf,
                frameUniformBuffer: frameUniformBuf, lightUniformBuffer: lightUniformBuf,
                indicesPerSphere: bm.indicesPerSphere, instanceCount: min(bm.actualNodeCount, nodeRenderCap))
        }

        // Opaque: Extra draw passes
        for pass in extraDrawPasses {
            pass.encodeOpaque(encoder: renderEncoder, frameUniform: frameUniformBuf, lightUniform: lightUniformBuf)
        }

        // Switch to transparent sub-pass
        renderEncoder.setDepthStencilState(transparentDepthState)
        renderEncoder.setCullMode(.none)

        // Transparent: Nebulae
        if let vtxBuf = bm.nebulaVertexBuffer,
           let idxBuf = bm.nebulaIndexBuffer {
            RenderEncoder.encodeNebulae(
                encoder: renderEncoder, pipeline: nebulaPipeline,
                vertexBuffer: vtxBuf, indexBuffer: idxBuf,
                frameUniformBuffer: frameUniformBuf,
                vertexCount: bm.actualNebulaVertexCount)
        }

        // Transparent: Edges (ramped)
        if bm.actualEdgeCount > 0 {
            edgeRenderCap = min(edgeRenderCap + renderRampPerFrame, bm.actualEdgeCount)
        } else {
            edgeRenderCap = 0
        }
        if let cylIdxBuf = bm.cylinderIndexBuffer,
           let instanceBuf = bm.edgeInstanceBuffer {
            RenderEncoder.encodeTransparentEdges(
                encoder: renderEncoder, pipeline: edgePipeline,
                cylinderIndexBuffer: cylIdxBuf, instanceBuffer: instanceBuf,
                frameUniformBuffer: frameUniformBuf,
                instanceCount: min(bm.actualEdgeCount, edgeRenderCap))
        }

        // Transparent: Labels
        if let vtxBuf = bm.labelVertexBuffer,
           let idxBuf = bm.labelIndexBuffer {
            RenderEncoder.encodeLabels(
                encoder: renderEncoder, pipeline: labelPipeline,
                vertexBuffer: vtxBuf, indexBuffer: idxBuf,
                frameUniformBuffer: frameUniformBuf,
                atlasTexture: bm.labelAtlasTexture, sampler: bm.labelSampler,
                labelCount: bm.actualLabelCount)
        }

        // Transparent: Extra draw passes
        for pass in extraDrawPasses {
            pass.encodeTransparent(encoder: renderEncoder, frameUniform: frameUniformBuf)
        }

        // Transparent: Flow particles
        if let vtxBuf = bm.flowVertexBuffer,
           let idxBuf = bm.flowIndexBuffer {
            RenderEncoder.encodeFlowParticles(
                encoder: renderEncoder, pipeline: flowPipeline,
                vertexBuffer: vtxBuf, indexBuffer: idxBuf,
                frameUniformBuffer: frameUniformBuf,
                particleCount: bm.actualFlowParticleCount)
        }

        renderEncoder.endEncoding()

        // Present + commit render CB
        commandBuffer.present(drawable)
        commandBuffer.addCompletedHandler { @Sendable cb in
            if cb.status == .error {
                GPULog.log("RENDER ERROR: \(cb.error?.localizedDescription ?? "unknown")")
            }
        }
        commandBuffer.commit()

        let totalMs = (CFAbsoluteTimeGetCurrent() - frameStart) * 1000.0
        let encodeMs = (CFAbsoluteTimeGetCurrent() - encodeStart) * 1000.0
        GPULog.log("COMMITTED frame=\(frameCount) n=\(min(bm.actualNodeCount, nodeRenderCap))/\(bm.actualNodeCount) e=\(min(bm.actualEdgeCount, edgeRenderCap))/\(bm.actualEdgeCount) cb=\(String(format:"%.0f",cbMs))ms enc=\(String(format:"%.0f",encodeMs))ms total=\(String(format:"%.0f",totalMs))ms")
        if encodeMs > 10 {
            os_log(.fault, log: Self.stallLog, "HOTPATH: encode+present %.1fms", encodeMs)
        }
    }
}
