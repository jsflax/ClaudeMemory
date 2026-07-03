import RealityKit
import Metal
import simd
import EngramSceneKit

/// 48-byte vertex struct matching LowLevelMesh single-buffer layout (edges/labels).
struct BatchVertex {
    var px: Float, py: Float, pz: Float     // position (12)
    var nx: Float, ny: Float, nz: Float     // normal (12)
    var u: Float, v: Float                   // uv0 (8)
    var cr: Float, cg: Float, cb: Float, ca: Float  // color (16)
}

/// Instanced node rendering system.
///
/// Two code paths controlled by `#available(macOS 26, *)`:
/// - **macOS 26+**: `MeshInstanceCollection` — true GPU instancing with per-instance
///   transforms and visual data in a `LowLevelTexture`.
/// - **< macOS 26**: Two-buffer `LowLevelMesh` — buffer 0 holds static template geometry
///   (written once), buffer 1 holds per-instance data (32 bytes/vert, written per frame).
///   A geometry modifier transforms template verts to world positions on GPU.
///
/// Compared to the old single-buffer approach (48 bytes × 60 verts × N = 2,880N bytes/frame),
/// the two-buffer path writes 32 bytes × 60 verts × N = 1,920N bytes/frame (33% reduction)
/// and eliminates CPU-side position math (offset added on GPU by geometry modifier).
@MainActor
public final class NodeBatchSystem {
    /// Reusable staging array for buffer 1 — avoids per-frame allocation.
    private var instanceStaging: [LowLevelMeshFactory.NodeInstanceAttribs] = []
    /// Pre-allocated staging buffer for texture blit (macOS 26+). Avoids per-frame MTLBuffer creation.
    private var texStagingBuffer: MTLBuffer?
    private var texStagingCapacity: Int = 0
    /// Last instance count — update MeshInstancesComponent part only when this changes.
    private var lastInstanceCount: Int = -1

    public init() {}

    public func update(
        scene: EngramRealityScene,
        dataProvider: SceneDataProvider,
        visibleSet: VisibleSet,
        topologyChanged: Bool,
        animationTime: Float,
        scaleFactor: Float,
        commandBuffer: MTLCommandBuffer? = nil
    ) {
        if #available(macOS 26, *) {
            updateWithMeshInstances(
                scene: scene, dataProvider: dataProvider,
                visibleSet: visibleSet, animationTime: animationTime,
                scaleFactor: scaleFactor, commandBuffer: commandBuffer
            )
        } else {
            updateWithTwoBufferMesh(
                scene: scene, dataProvider: dataProvider,
                visibleSet: visibleSet, animationTime: animationTime,
                scaleFactor: scaleFactor, commandBuffer: commandBuffer
            )
        }
    }

    // MARK: - < macOS 26: Two-buffer LowLevelMesh + Geometry Modifier

    /// Two-buffer path: buffer 0 = static template (written once by LowLevelMeshFactory),
    /// buffer 1 = per-instance transform + visual data (written per frame here).
    /// The geometry modifier reads buffer 1 attributes to transform template verts on GPU.
    private func updateWithTwoBufferMesh(
        scene: EngramRealityScene,
        dataProvider: SceneDataProvider,
        visibleSet: VisibleSet,
        animationTime: Float,
        scaleFactor: Float,
        commandBuffer: MTLCommandBuffer? = nil
    ) {
        let visibleCount = visibleSet.totalNodeCount
        guard visibleCount > 0 else { return }

        LowLevelMeshFactory.ensureNodeBatchMesh(scene: scene, capacity: visibleCount)
        guard let mesh = scene.nodeBatchMesh else { return }

        let nodes = dataProvider.nodes
        let positions = dataProvider.positions
        let positionArray = dataProvider.positionArray
        let dyingNodes = dataProvider.dyingNodes
        let colorMap = dataProvider.projectColorMap
        let vps = scene.vertsPerSphere

        // Ensure staging buffer is large enough (one attrib struct per vertex)
        let totalVerts = scene.nodeBatchCapacity * vps
        if instanceStaging.count < totalVerts {
            instanceStaging = [LowLevelMeshFactory.NodeInstanceAttribs](
                repeating: LowLevelMeshFactory.NodeInstanceAttribs(
                    ox: 0, oy: 0, oz: 0, scale: 0,
                    cr: 0, cg: 0, cb: 0, packedAlpha: 0
                ),
                count: totalVerts
            )
        }

        var instanceIdx = 0

        let glowingNodes = dataProvider.glowingNodes
        let newNodeGlows = dataProvider.newNodeGlows
        let selectedNode = dataProvider.selectedNode
        let searchMatchIds = dataProvider.searchMatchIds
        let isSearchActive = dataProvider.isSearchActive

        func writeNode(nodeIndex: Int) {
            let node = nodes[nodeIndex]
            let pos = nodeIndex < positionArray.count ? positionArray[nodeIndex] : (positions[node.id] ?? .zero)

            let scaledPos = pos * scaleFactor
            var color = colorMap[node.project] ?? SIMD3<Float>(0.5, 0.5, 0.5)
            let baseRadius: Float = node.isHub ? 12.0 : 8.0
            let radius = baseRadius * scaleFactor

            // Dying nodes: dim the color (shader has no separate dying opacity)
            if dyingNodes.contains(node.id) {
                color *= 0.3
            }

            // Compute packed visual state for the surface shader:
            //   stateType + intensity * 0.01  (+ 10.0 if search-dimmed)
            //   stateType: 0=normal, 0.25=selected, 0.5=recall, 0.75=arrival, 1.0=search match
            //   Intensity occupies 0.00–0.01 range within each 0.25-wide state band.
            var packedState: Float = 0.0
            if searchMatchIds.contains(node.id) {
                packedState = 1.0
            } else if let elapsed = newNodeGlows[node.id] {
                let intensity = arrivalGlowIntensity(elapsed: elapsed)
                packedState = 0.75 + intensity * 0.01
            } else if let elapsed = glowingNodes[node.id] {
                let intensity = recallGlowIntensity(elapsed: elapsed)
                packedState = 0.5 + intensity * 0.01
            } else if selectedNode == node.id {
                packedState = 0.25 + 0.5 * 0.01
            }

            // Search dimming: non-matched nodes during active search
            if isSearchActive && !searchMatchIds.contains(node.id) {
                packedState += 10.0
            }

            // Build per-instance attribs (same value for all 60 verts of this sphere)
            let attrib = LowLevelMeshFactory.NodeInstanceAttribs(
                ox: scaledPos.x, oy: scaledPos.y, oz: scaledPos.z, scale: radius,
                cr: color.x, cg: color.y, cb: color.z, packedAlpha: packedState
            )

            // Fill all verts for this instance with the same attribs
            let baseVert = instanceIdx * vps
            for vi in 0..<vps {
                instanceStaging[baseVert + vi] = attrib
            }
            instanceIdx += 1
        }

        for idx in visibleSet.nearNodes { writeNode(nodeIndex: idx) }
        for idx in visibleSet.midNodes { writeNode(nodeIndex: idx) }
        for idx in visibleSet.farNodes { writeNode(nodeIndex: idx) }

        // Zero remaining instances
        let usedVerts = instanceIdx * vps
        if usedVerts < totalVerts {
            let zeroAttrib = LowLevelMeshFactory.NodeInstanceAttribs(
                ox: 0, oy: 0, oz: 0, scale: 0,
                cr: 0, cg: 0, cb: 0, packedAlpha: 0
            )
            for vi in usedVerts..<totalVerts {
                instanceStaging[vi] = zeroAttrib
            }
        }

        // GPU-synchronized write to buffer 1 only (buffer 0 is static template)
        guard let cmdBuf = commandBuffer ?? scene.commandQueue.makeCommandBuffer() else { return }
        let destBuffer = mesh.replace(bufferIndex: 1, using: cmdBuf)
        let dest = destBuffer.contents().bindMemory(
            to: LowLevelMeshFactory.NodeInstanceAttribs.self, capacity: totalVerts
        )
        instanceStaging.withUnsafeBufferPointer { src in
            dest.update(from: src.baseAddress!, count: totalVerts)
        }
        if commandBuffer == nil { cmdBuf.commit() }
    }

    // MARK: - macOS 26+: MeshInstanceCollection

    /// MeshInstanceCollection path: one sphere template mesh, N instance transforms.
    /// Per-instance visual data (color, alpha) stored in a LowLevelTexture sampled
    /// by the geometry modifier using instance_id().
    @available(macOS 26, *)
    private func updateWithMeshInstances(
        scene: EngramRealityScene,
        dataProvider: SceneDataProvider,
        visibleSet: VisibleSet,
        animationTime: Float,
        scaleFactor: Float,
        commandBuffer: MTLCommandBuffer? = nil
    ) {
        let visibleCount = visibleSet.totalNodeCount
        guard visibleCount > 0 else { return }

        // Ensure sphere template entity and instance texture exist
        scene.ensureNodeInstanceResources(capacity: visibleCount)
        guard let entity = scene.nodeTemplateEntity,
              let instanceData = scene.nodeInstanceData,
              let instanceTexture = scene.nodeInstanceTexture else {
            return
        }

        let nodes = dataProvider.nodes
        let positions = dataProvider.positions
        let positionArray = dataProvider.positionArray
        let dyingNodes = dataProvider.dyingNodes
        let colorMap = dataProvider.projectColorMap
        let glowingNodes = dataProvider.glowingNodes
        let newNodeGlows = dataProvider.newNodeGlows
        let selectedNode = dataProvider.selectedNode
        let searchMatchIds = dataProvider.searchMatchIds
        let isSearchActive = dataProvider.isSearchActive

        // Build per-instance visual data for texture
        let texWidth = scene.nodeInstanceTextureWidth
        var texData = [SIMD4<Float16>](repeating: .zero, count: texWidth)

        var instanceIdx = 0

        // Write transforms synchronously (GPU-safe, no command buffer)
        instanceData.replaceMutableTransforms { transforms in
            func writeNode(nodeIndex: Int) {
                guard instanceIdx < visibleCount, instanceIdx < transforms.count else { return }
                let node = nodes[nodeIndex]
                let pos = nodeIndex < positionArray.count ? positionArray[nodeIndex] : (positions[node.id] ?? .zero)

                let scaledPos = pos * scaleFactor
                let baseRadius: Float = node.isHub ? 12.0 : 8.0
                let radius = baseRadius * scaleFactor
                var color = colorMap[node.project] ?? SIMD3<Float>(0.5, 0.5, 0.5)

                if dyingNodes.contains(node.id) { color *= 0.3 }

                // Compute packed visual state (same encoding as two-buffer path)
                var packedState: Float = 0.0
                if searchMatchIds.contains(node.id) {
                    packedState = 1.0
                } else if let elapsed = newNodeGlows[node.id] {
                    let intensity = arrivalGlowIntensity(elapsed: elapsed)
                    packedState = 0.75 + intensity * 0.01
                } else if let elapsed = glowingNodes[node.id] {
                    let intensity = recallGlowIntensity(elapsed: elapsed)
                    packedState = 0.5 + intensity * 0.01
                } else if selectedNode == node.id {
                    packedState = 0.25 + 0.5 * 0.01
                }
                if isSearchActive && !searchMatchIds.contains(node.id) {
                    packedState += 10.0
                }

                let s = simd_float4x4(diagonal: SIMD4<Float>(radius, radius, radius, 1.0))
                let t = simd_float4x4(columns: (
                    SIMD4<Float>(1, 0, 0, 0),
                    SIMD4<Float>(0, 1, 0, 0),
                    SIMD4<Float>(0, 0, 1, 0),
                    SIMD4<Float>(scaledPos.x, scaledPos.y, scaledPos.z, 1)
                ))
                transforms[instanceIdx] = t * s

                texData[instanceIdx] = SIMD4<Float16>(
                    Float16(color.x), Float16(color.y), Float16(color.z), Float16(packedState)
                )
                instanceIdx += 1
            }

            for idx in visibleSet.nearNodes { writeNode(nodeIndex: idx) }
            for idx in visibleSet.midNodes { writeNode(nodeIndex: idx) }
            for idx in visibleSet.farNodes { writeNode(nodeIndex: idx) }
        }

        instanceData.instanceCount = instanceIdx

        // Update per-instance color texture via staging buffer + blit
        let bytesPerRow = texWidth * MemoryLayout<SIMD4<Float16>>.stride
        if texStagingCapacity < bytesPerRow {
            texStagingBuffer = scene.device.makeBuffer(length: bytesPerRow, options: .storageModeShared)
            texStagingCapacity = bytesPerRow
        }
        if let stagingBuf = texStagingBuffer,
           let cmdBuf = commandBuffer ?? scene.commandQueue.makeCommandBuffer() {
            let texMTL = instanceTexture.replace(using: cmdBuf)
            texData.withUnsafeBytes { ptr in
                stagingBuf.contents().copyMemory(from: ptr.baseAddress!, byteCount: bytesPerRow)
            }
            let blit = cmdBuf.makeBlitCommandEncoder()!
            blit.copy(
                from: stagingBuf, sourceOffset: 0,
                sourceBytesPerRow: bytesPerRow, sourceBytesPerImage: bytesPerRow,
                sourceSize: MTLSize(width: texWidth, height: 1, depth: 1),
                to: texMTL, destinationSlice: 0, destinationLevel: 0,
                destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
            )
            blit.endEncoding()
            if commandBuffer == nil { cmdBuf.commit() }
        }

        // Create MeshInstancesComponent exactly ONCE (see EdgeBatchSystem:
        // instanceData.instanceCount already drives the drawn count, and
        // recreating on count change races the render thread's draw-call
        // encode — use-after-free SIGSEGV under per-frame count churn).
        if entity.components[MeshInstancesComponent.self] == nil {
            guard let mesh = entity.model?.mesh else { return }
            do {
                let comp = try MeshInstancesComponent(
                    mesh: mesh,
                    instances: instanceData,
                    bounds: LowLevelMeshFactory.batchMeshBounds
                )
                entity.components.set(comp)
            } catch {
                return
            }
        }
    }
}
