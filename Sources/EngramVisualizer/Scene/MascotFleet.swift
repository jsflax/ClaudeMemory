import Metal
import simd
import SwiftUI

/// Notification protocol for memory lifecycle events → mascot animations.
@MainActor
protocol MascotNotifier: AnyObject {
    func onNodeCreated(nodeId: UUID, project: String)
    func onNodeDeleted(nodeId: UUID, project: String, lastPosition: SIMD3<Float>?)
    func onNodeUpdated(nodeId: UUID, project: String, changes: NodeChangeInfo)
}

/// Describes what changed on a node update.
struct NodeChangeInfo {
    let importanceChanged: Bool
    let topicChanged: Bool
    let isAccessOnly: Bool
}

/// Manages a fleet of per-project mascots within a single galaxy.
/// Each project gets its own MascotSystem that patrols only that project's nodes.
@MainActor
final class MascotFleet: MascotNotifier {
    let galaxyId: String
    private(set) var mascots: [String: MascotSystem] = [:]  // project -> mascot
    private let device: MTLDevice
    private let sharedResources: MascotSharedResources
    private let maxMascots = 10

    /// Cached per-project positions — persists across frames when simulation is settled.
    var cachedNodesByProject: [String: [UUID: SIMD3<Float>]] = [:]

    /// Instance buffer for batched instanced drawing.
    private var instanceBuffer: MTLBuffer?
    private var instanceBufferCapacity: Int = 0

    // MARK: - GPU Thruster Particle System

    private static let maxParticles: Int = 2048
    private static let maxSpawnsPerFrame: Int = 256  // 10 mascots × ~20 spawns/frame

    private var particleBuffer: MTLBuffer?         // persistent ThrusterParticleGPU[maxParticles]
    private var spawnRequestBuffer: MTLBuffer?      // ThrusterSpawnRequest[maxSpawnsPerFrame]
    private var particleOutputBuffer: MTLBuffer?    // FlowParticleVertex[maxParticles] (packed output)
    private var particleCountBuffer: MTLBuffer?     // atomic_uint (draw count)
    private var particleIndexBuffer: MTLBuffer?     // pre-filled quad indices
    private var particleParamsBuffer: MTLBuffer?    // ThrusterComputeParams
    private var spawnRingOffset: UInt32 = 0         // ring buffer write cursor
    private(set) var gpuParticleCount: Int = 0      // read back from atomic counter (1-frame latency)

    // Compute pipelines
    private var advectPipeline: MTLComputePipelineState?
    private var spawnPipeline: MTLComputePipelineState?
    private var packPipeline: MTLComputePipelineState?

    // MARK: - GPU Matrix Compute
    private var animStateBuffer: MTLBuffer?
    private var animStateCapacity: Int = 0
    private var matrixComputePipeline: MTLComputePipelineState?

    /// Which project's mascot is currently in chat mode (only one at a time).
    private(set) var chattingMascotProject: String?

    /// Ordered array of mascots matching instance buffer layout (set each frame).
    private var orderedMascots: [MascotSystem] = []

    init(galaxyId: String, device: MTLDevice, sharedResources: MascotSharedResources, library: MTLLibrary? = nil) {
        self.galaxyId = galaxyId
        self.device = device
        self.sharedResources = sharedResources

        setupGPUParticles(library: library)
    }

    private func setupGPUParticles(library: MTLLibrary?) {
        let maxP = Self.maxParticles
        let maxS = Self.maxSpawnsPerFrame

        // Persistent particle state buffer (zeroed = all dead)
        particleBuffer = device.makeBuffer(
            length: maxP * MemoryLayout<ThrusterParticleGPU>.stride,
            options: .storageModeShared
        )
        // Zero it to mark all particles as dead
        if let buf = particleBuffer {
            memset(buf.contents(), 0, buf.length)
        }

        spawnRequestBuffer = device.makeBuffer(
            length: maxS * MemoryLayout<ThrusterSpawnRequest>.stride,
            options: .storageModeShared
        )

        particleOutputBuffer = device.makeBuffer(
            length: maxP * MemoryLayout<FlowParticleVertex>.stride,
            options: .storageModeShared
        )

        // Atomic counter buffer (single uint32)
        particleCountBuffer = device.makeBuffer(
            length: MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )

        particleParamsBuffer = device.makeBuffer(
            length: MemoryLayout<ThrusterComputeParams>.stride,
            options: .storageModeShared
        )

        // Pre-fill quad index buffer for maxParticles quads
        let idxCount = maxP * 6
        particleIndexBuffer = device.makeBuffer(
            length: idxCount * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )
        if let buf = particleIndexBuffer {
            let indices = buf.contents().bindMemory(to: UInt32.self, capacity: idxCount)
            for i in 0..<maxP {
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

        // Build compute pipelines
        guard let lib = library ?? device.makeDefaultLibrary() else { return }
        if let fn = lib.makeFunction(name: "thruster_advect") {
            advectPipeline = try? device.makeComputePipelineState(function: fn)
        }
        if let fn = lib.makeFunction(name: "thruster_spawn") {
            spawnPipeline = try? device.makeComputePipelineState(function: fn)
        }
        if let fn = lib.makeFunction(name: "thruster_pack_vertices") {
            packPipeline = try? device.makeComputePipelineState(function: fn)
        }
        if let fn = lib.makeFunction(name: "mascot_compute_matrices") {
            matrixComputePipeline = try? device.makeComputePipelineState(function: fn)
        }
    }

    // MARK: - Project Sync

    /// Add/remove mascots to match the active project set.
    /// `colorMap` provides tint colors per project.
    func syncProjects(active: Set<String>, colorMap: [String: NSColor]) {
        // Remove mascots for projects that are no longer active
        for project in mascots.keys where !active.contains(project) {
            mascots.removeValue(forKey: project)
            if chattingMascotProject == project { chattingMascotProject = nil }
        }

        // Add mascots for new projects (up to max)
        for project in active {
            guard mascots[project] == nil else { continue }
            guard mascots.count < maxMascots else { break }

            let tint = colorMap[project].map { nsColorToTint($0) } ?? SIMD3<Float>(0, 0.8, 1.0)
            let mascot = MascotSystem(
                device: device, shared: sharedResources,
                projectId: project, tint: tint
            )
            mascots[project] = mascot
        }
    }

    // MARK: - Per-Frame Update

    func update(
        dt: Float,
        camera: CameraController,
        positions: [UUID: SIMD3<Float>],
        nodesByProject: [String: [UUID: SIMD3<Float>]],
        nodeInfo: [UUID: MascotNodeInfo],
        maintenanceActive: Bool
    ) {
        for (project, mascot) in mascots {
            let projectPositions = nodesByProject[project] ?? [:]

            // Build node info for current target only
            var info: [UUID: MascotNodeInfo] = [:]
            if let targetId = mascot.currentTargetId, let ni = nodeInfo[targetId] {
                info[targetId] = ni
            }

            // Handle maintenance state — visual overlay only, doesn't block movement
            if maintenanceActive != mascot.isBusyMode {
                if maintenanceActive {
                    mascot.enterBusyMode()
                } else {
                    mascot.exitBusyMode()
                }
            }

            mascot.update(dt: dt, camera: camera, nodePositions: projectPositions, nodeInfo: info)
        }

        // Pack instance buffer from CPU-computed matrices
        packInstanceBuffer()
    }

    // MARK: - Instance Buffer Packing

    /// Pack instance data from per-mascot uniformBuffers (CPU-computed matrices).
    /// GPU matrix compute kernel (mascot_compute_matrices) is available but not yet wired in
    /// because arcane circle, node rings, and holo screen effects still depend on CPU-side bodyMatrix.
    private func packInstanceBuffer() {
        let count = mascots.count
        guard count > 0 else {
            orderedMascots = []
            return
        }

        // Ensure buffer capacity
        if instanceBufferCapacity < count {
            let cap = max(count, 10)
            instanceBuffer = device.makeBuffer(
                length: cap * MemoryLayout<MascotInstanceData>.stride,
                options: .storageModeShared
            )
            instanceBufferCapacity = cap
        }

        guard let buf = instanceBuffer else { return }
        let ptr = buf.contents().bindMemory(to: MascotInstanceData.self, capacity: count)

        orderedMascots = Array(mascots.values)
        for (i, mascot) in orderedMascots.enumerated() {
            guard let uniformBuf = mascot.uniformBuffer else { continue }
            let uniforms = uniformBuf.contents().bindMemory(to: MascotUniforms.self, capacity: 1).pointee
            ptr[i].parts = uniforms.parts
            ptr[i].projectTint = mascot.projectTint
            ptr[i]._instancePad = 0
        }
    }

    /// Dispatch GPU matrix compute kernel to fill instance buffer from anim states.
    /// Reserved for future use when arcane/holo/ring effects are also GPU-ified.
    func dispatchMatrixCompute(encoder: MTLComputeCommandEncoder) {
        // Currently a no-op — instance buffer is packed from CPU-computed matrices.
        // The GPU kernel (mascot_compute_matrices) and anim state packing infrastructure
        // are ready to be activated when effect subsystems no longer depend on CPU bodyMatrix.
    }

    // MARK: - Drawing (opaque pass — instanced)

    func drawAll(
        encoder: MTLRenderCommandEncoder,
        frameUniformBuf: MTLBuffer,
        lightUniformBuf: MTLBuffer,
        pipeline: MTLRenderPipelineState,
        depthState: MTLDepthStencilState
    ) {
        let count = orderedMascots.count
        guard count > 0,
              let instBuf = instanceBuffer,
              let firstMascot = orderedMascots.first,
              firstMascot.parts.count == 5,
              let baseTex = firstMascot.baseColorTexture,
              let mrTex = firstMascot.metalRoughTexture,
              let samp = firstMascot.sampler else { return }

        encoder.setRenderPipelineState(pipeline)
        encoder.setDepthStencilState(depthState)
        encoder.setCullMode(.front)

        // Shared textures and sampler
        encoder.setFragmentTexture(baseTex, index: 0)
        encoder.setFragmentTexture(mrTex, index: 1)
        encoder.setFragmentSamplerState(samp, index: 0)

        // Instance buffer (shared across all parts)
        encoder.setVertexBuffer(instBuf, offset: 0, index: 2)
        encoder.setFragmentBuffer(instBuf, offset: 0, index: 2)

        // Frame + lighting uniforms
        encoder.setVertexBuffer(frameUniformBuf, offset: 0, index: 1)
        encoder.setFragmentBuffer(frameUniformBuf, offset: 0, index: 0)
        encoder.setFragmentBuffer(lightUniformBuf, offset: 0, index: 1)

        // Draw each part type with instancing (5 instanced draws instead of N*5)
        let parts = firstMascot.parts
        for partIdx in 0..<5 {
            let part = parts[partIdx]
            encoder.setVertexBuffer(part.vertexBuffer, offset: 0, index: 0)

            var pidx = UInt32(partIdx)
            encoder.setVertexBytes(&pidx, length: 4, index: 3)

            encoder.drawIndexedPrimitives(
                type: .triangle,
                indexCount: part.indexCount,
                indexType: .uint32,
                indexBuffer: part.indexBuffer,
                indexBufferOffset: 0,
                instanceCount: count
            )
        }
    }

    // MARK: - GPU Particle Compute Dispatch

    /// Collect spawn requests from all mascots and dispatch GPU particle compute kernels.
    /// Call this once per frame, before the render pass.
    func dispatchParticleCompute(encoder: MTLComputeCommandEncoder, dt: Float, camera: CameraController) {
        guard let pBuf = particleBuffer,
              let sBuf = spawnRequestBuffer,
              let oBuf = particleOutputBuffer,
              let cBuf = particleCountBuffer,
              let paramBuf = particleParamsBuffer,
              let advect = advectPipeline,
              let spawn = spawnPipeline,
              let pack = packPipeline else { return }

        let maxP = UInt32(Self.maxParticles)
        let sf = camera.scaleFactor
        let camPos = camera.cameraPosition * sf
        let camFwd = camera.forward

        // Collect spawn requests from all mascots
        var allSpawns: [ThrusterSpawnRequest] = []
        for mascot in mascots.values {
            let toMascot = mascot.currentPosition - camPos
            let behindCamera = simd_dot(toMascot, camFwd) < -mascot.lastDynamicScale * 2.0
            guard !behindCamera else { continue }
            let spawns = mascot.generateThrusterSpawns(
                dt: dt, bodyMatrix: mascot.lastBodyMatrix, dynamicScale: mascot.lastDynamicScale
            )
            allSpawns.append(contentsOf: spawns)
        }

        let spawnCount = min(allSpawns.count, Self.maxSpawnsPerFrame)

        // Read back previous frame's draw count (1-frame latency)
        gpuParticleCount = Int(cBuf.contents().bindMemory(to: UInt32.self, capacity: 1).pointee)

        // Reset atomic counter to 0 for this frame
        cBuf.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = 0

        // Write spawn requests
        if spawnCount > 0 {
            let spawnPtr = sBuf.contents().bindMemory(to: ThrusterSpawnRequest.self, capacity: spawnCount)
            for i in 0..<spawnCount {
                spawnPtr[i] = allSpawns[i]
            }
        }

        // Write params
        let params = paramBuf.contents().bindMemory(to: ThrusterComputeParams.self, capacity: 1)
        params.pointee = ThrusterComputeParams(
            dt: dt, damping: 1.5,
            totalParticles: maxP,
            spawnCount: UInt32(spawnCount),
            spawnOffset: spawnRingOffset,
            _pad0: 0, _pad1: 0, _pad2: 0
        )
        spawnRingOffset = (spawnRingOffset + UInt32(spawnCount)) % maxP

        // Dispatch advect kernel
        let tgSize = min(advect.maxTotalThreadsPerThreadgroup, 256)
        let tgCount = (Int(maxP) + tgSize - 1) / tgSize

        encoder.setComputePipelineState(advect)
        encoder.setBuffer(pBuf, offset: 0, index: 0)
        encoder.setBuffer(paramBuf, offset: 0, index: 1)
        encoder.dispatchThreadgroups(
            MTLSize(width: tgCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1)
        )

        // Dispatch spawn kernel
        if spawnCount > 0 {
            let spawnTgSize = min(spawn.maxTotalThreadsPerThreadgroup, 256)
            let spawnTgCount = (spawnCount + spawnTgSize - 1) / spawnTgSize

            encoder.setComputePipelineState(spawn)
            encoder.setBuffer(pBuf, offset: 0, index: 0)
            encoder.setBuffer(sBuf, offset: 0, index: 1)
            encoder.setBuffer(paramBuf, offset: 0, index: 2)
            encoder.dispatchThreadgroups(
                MTLSize(width: spawnTgCount, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: spawnTgSize, height: 1, depth: 1)
            )
        }

        // Dispatch pack kernel
        encoder.setComputePipelineState(pack)
        encoder.setBuffer(pBuf, offset: 0, index: 0)
        encoder.setBuffer(oBuf, offset: 0, index: 1)
        encoder.setBuffer(cBuf, offset: 0, index: 2)
        encoder.setBuffer(paramBuf, offset: 0, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: tgCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1)
        )
    }

    // MARK: - Drawing (transparent pass — grouped by effect type)

    func drawAllThrusterParticles(
        encoder: MTLRenderCommandEncoder,
        frameUniformBuf: MTLBuffer,
        pipeline: MTLRenderPipelineState
    ) {
        guard gpuParticleCount > 0,
              let vtxBuf = particleOutputBuffer,
              let idxBuf = particleIndexBuffer else { return }

        encoder.setRenderPipelineState(pipeline)
        encoder.setVertexBuffer(vtxBuf, offset: 0, index: 0)
        encoder.setVertexBuffer(frameUniformBuf, offset: 0, index: 1)
        let indexCount = min(gpuParticleCount, Self.maxParticles) * 6
        encoder.drawIndexedPrimitives(
            type: .triangle,
            indexCount: indexCount,
            indexType: .uint32,
            indexBuffer: idxBuf,
            indexBufferOffset: 0
        )
    }

    func drawAllArcaneCircles(
        encoder: MTLRenderCommandEncoder,
        frameUniformBuf: MTLBuffer,
        pipeline: MTLRenderPipelineState
    ) {
        for mascot in mascots.values where mascot.arcaneVisible {
            mascot.drawArcaneCircle(
                encoder: encoder,
                frameUniformBuf: frameUniformBuf,
                pipeline: pipeline
            )
        }
    }

    func drawAllScanRings(
        encoder: MTLRenderCommandEncoder,
        frameUniformBuf: MTLBuffer,
        pipeline: MTLRenderPipelineState
    ) {
        for mascot in mascots.values where mascot.scanRingsVisible {
            mascot.drawScanRings(
                encoder: encoder,
                frameUniformBuf: frameUniformBuf,
                pipeline: pipeline
            )
        }
    }

    func drawAllConjureOrbs(
        encoder: MTLRenderCommandEncoder,
        frameUniformBuf: MTLBuffer,
        pipeline: MTLRenderPipelineState
    ) {
        for mascot in mascots.values where mascot.conjureOrbVisible {
            mascot.drawConjureOrb(
                encoder: encoder,
                frameUniformBuf: frameUniformBuf,
                pipeline: pipeline
            )
        }
    }

    func drawAllNodeRings(
        encoder: MTLRenderCommandEncoder,
        frameUniformBuf: MTLBuffer,
        pipeline: MTLRenderPipelineState
    ) {
        for mascot in mascots.values where mascot.ringsVisible {
            mascot.drawNodeRings(
                encoder: encoder,
                frameUniformBuf: frameUniformBuf,
                pipeline: pipeline
            )
        }
    }

    func drawAllHoloScreens(
        encoder: MTLRenderCommandEncoder,
        frameUniformBuf: MTLBuffer,
        pipeline: MTLRenderPipelineState
    ) {
        for mascot in mascots.values where mascot.holoVisible {
            mascot.drawHoloScreen(
                encoder: encoder,
                frameUniformBuf: frameUniformBuf,
                pipeline: pipeline
            )
        }
    }

    // MARK: - Chat Mode

    /// Enter chat mode for the mascot nearest to the camera.
    func enterChat(cameraPosition: SIMD3<Float>) {
        // Find nearest mascot
        var bestProject: String?
        var bestDist: Float = .greatestFiniteMagnitude
        for (project, mascot) in mascots {
            let dist = simd_length(mascot.currentPosition - cameraPosition)
            if dist < bestDist {
                bestDist = dist
                bestProject = project
            }
        }
        if let project = bestProject {
            mascots[project]?.setChatting(true)
            chattingMascotProject = project
        }
    }

    /// Exit chat mode.
    func exitChat() {
        if let project = chattingMascotProject {
            mascots[project]?.setChatting(false)
            chattingMascotProject = nil
        }
    }

    // MARK: - MascotNotifier

    func onNodeCreated(nodeId: UUID, project: String) {
        guard let mascot = mascots[project] else { return }
        // Position will be set when the node appears in the simulation
        mascot.enqueueTask(.create(nodeId: nodeId, position: .zero))
    }

    func onNodeDeleted(nodeId: UUID, project: String, lastPosition: SIMD3<Float>?) {
        guard let mascot = mascots[project] else { return }
        mascot.enqueueTask(.delete(nodeId: nodeId, position: lastPosition ?? mascot.currentPosition))
    }

    func onNodeUpdated(nodeId: UUID, project: String, changes: NodeChangeInfo) {
        guard let mascot = mascots[project] else { return }
        guard !changes.isAccessOnly else { return }  // too frequent, no visual meaning

        // Limit pending reactions to 3 — drop oldest if overflow
        let pendingReactions = mascot.taskQueue.filter {
            if case .react = $0 { return true } else { return false }
        }.count
        guard pendingReactions < 3 else { return }

        let type: MascotSystem.ReactionType
        if changes.importanceChanged {
            type = .importanceBoost
        } else if changes.topicChanged {
            type = .inspect
        } else {
            type = .headTurn
        }
        mascot.enqueueTask(.react(nodeId: nodeId, type: type))
    }

    // MARK: - Query

    /// Whether any mascot is currently inspecting a node (for node instance packing).
    var anyInspecting: Bool {
        mascots.values.contains { $0.arcaneIntensity > 0.01 }
    }

    /// Find the mascot inspecting a given node, if any.
    func inspectingMascot(for nodeId: UUID) -> MascotSystem? {
        mascots.values.first { $0.currentTargetId == nodeId && $0.arcaneIntensity > 0.01 }
    }

    // MARK: - Helpers

    private func nsColorToTint(_ color: NSColor) -> SIMD3<Float> {
        let c = color.usingColorSpace(.sRGB) ?? color
        return SIMD3<Float>(Float(c.redComponent), Float(c.greenComponent), Float(c.blueComponent))
    }
}
