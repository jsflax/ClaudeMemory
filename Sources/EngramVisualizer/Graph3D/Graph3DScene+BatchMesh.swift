import SwiftUI
import RealityKit
import Metal
import simd
import os

private let frameLog = Logger(subsystem: "com.claudememory.visualizer", category: "3DFrameTiming")

extension Graph3DScene {

    // MARK: - Batch Mesh Helpers

    /// Standard vertex layout shared across all batch meshes (nodes, edges, labels).
    /// 48 bytes per vertex: position(12) + normal(12) + uv(8) + color(16).
    static func makeBatchMeshDescriptor(vertexCapacity: Int, indexCapacity: Int) -> LowLevelMesh.Descriptor {
        var desc = LowLevelMesh.Descriptor()
        desc.vertexCapacity = vertexCapacity
        desc.indexCapacity = indexCapacity
        desc.vertexAttributes = [
            .init(semantic: .position, format: .float3, offset: 0),
            .init(semantic: .normal, format: .float3, offset: 12),
            .init(semantic: .uv0, format: .float2, offset: 24),
            .init(semantic: .color, format: .float4, offset: 32),
        ]
        desc.vertexLayouts = [.init(bufferIndex: 0, bufferStride: 48)]
        desc.indexType = .uint32
        return desc
    }

    /// Generous bounds used for all batch meshes — avoids per-frame recalculation.
    static let batchMeshBounds = BoundingBox(min: SIMD3(-10, -10, -10), max: SIMD3(10, 10, 10))

    /// Assign a LowLevelMesh to a ModelEntity, creating the entity if needed.
    @discardableResult
    static func assignBatchEntity(
        mesh: LowLevelMesh,
        material: CustomMaterial?,
        existingEntity: inout ModelEntity?,
        name: String,
        container: Entity
    ) -> ModelEntity? {
        guard let resource = try? MeshResource(from: mesh),
              let material else { return nil }
        if let entity = existingEntity {
            entity.model = ModelComponent(mesh: resource, materials: [material])
            return entity
        } else {
            let entity = ModelEntity(mesh: resource, materials: [material])
            entity.name = name
            container.addChild(entity)
            existingEntity = entity
            return entity
        }
    }

    // MARK: - Color Helpers

    func nodeColorFloat3(for project: String, colorMap: [String: Color]) -> SIMD3<Float> {
        if let cached = nodeColorCache[project] { return cached }
        let color = colorMap[project] ?? .gray
        let nsColor = NSColor(color).usingColorSpace(.sRGB) ?? NSColor(color)
        var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
        nsColor.getRed(&r, green: &g, blue: &b, alpha: &a)
        let c = SIMD3<Float>(Float(r), Float(g), Float(b))
        nodeColorCache[project] = c
        return c
    }

    func edgeColorFloat3(for project: String?, colorMap: [String: Color]) -> SIMD3<Float> {
        let key = project ?? "__default"
        if let cached = edgeColorCache[key] { return cached }
        if let project, let swiftColor = colorMap[project] {
            let nsColor = NSColor(swiftColor).usingColorSpace(.sRGB) ?? NSColor(swiftColor)
            var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
            nsColor.getRed(&r, green: &g, blue: &b, alpha: &a)
            let c = SIMD3<Float>(
                Float(min(1.0, r * 1.4 + 0.15)),
                Float(min(1.0, g * 1.4 + 0.15)),
                Float(min(1.0, b * 1.4 + 0.15))
            )
            edgeColorCache[key] = c
            return c
        } else {
            let c = SIMD3<Float>(0.6, 0.6, 0.6)
            edgeColorCache[key] = c
            return c
        }
    }

    // MARK: - Node Batch Mesh

    // MARK: - Node Batch Mesh

    /// Create or resize the LowLevelMesh for batched node rendering (real sphere geometry).
    func ensureNodeBatchMesh(capacity: Int) {
        guard capacity > nodeBatchCapacity else { return }
        let newCapacity = max(capacity * 2, 512)

        let totalVerts = newCapacity * vertsPerSphere
        let totalIndices = newCapacity * indicesPerSphere
        let desc = Self.makeBatchMeshDescriptor(vertexCapacity: totalVerts, indexCapacity: totalIndices)

        guard let mesh = try? LowLevelMesh(descriptor: desc) else {
            frameLog.error("[nodeBatch] ❌ LowLevelMesh creation failed")
            return
        }

        // Pre-fill index buffer: stamp template indices for each node instance
        let templateIndices = sphereTemplateIndices
        let vpn = UInt32(vertsPerSphere)
        mesh.withUnsafeMutableIndices { raw in
            let indices = raw.bindMemory(to: UInt32.self)
            for node in 0..<newCapacity {
                let baseVert = UInt32(node) * vpn
                let baseIdx = node * indicesPerSphere
                for i in 0..<indicesPerSphere {
                    indices[baseIdx + i] = baseVert + templateIndices[i]
                }
            }
        }

        mesh.parts.replaceAll([
            LowLevelMesh.Part(indexCount: totalIndices, topology: .triangle, materialIndex: 0, bounds: Self.batchMeshBounds)
        ])

        nodeBatchMesh = mesh
        nodeBatchCapacity = newCapacity
        lastNodePartIndexCount = totalIndices
        Self.assignBatchEntity(mesh: mesh, material: nodeBatchMaterial, existingEntity: &nodeBatchEntity, name: "node_batch", container: rootEntity)
        frameLog.error("[nodeBatch] ✅ mesh created: \(newCapacity) nodes, \(totalVerts) verts, \(totalIndices) indices")
    }

    // MARK: - Edge Batch Mesh

    /// Create or resize the LowLevelMesh for batched edge rendering.
    func ensureEdgeBatchMesh(capacity: Int) {
        guard capacity > edgeBatchCapacity else { return }
        let newCapacity = max(capacity * 2, 2048)

        let totalVerts = newCapacity * 12   // 12 vertices per cylinder (6-segment tube)
        let totalIndices = newCapacity * 36 // 36 indices per cylinder (6 quad faces × 2 triangles × 3)
        let desc = Self.makeBatchMeshDescriptor(vertexCapacity: totalVerts, indexCapacity: totalIndices)

        guard let mesh = try? LowLevelMesh(descriptor: desc) else { return }

        // Pre-fill index buffer (6-sided cylinder topology: 6 quad faces per edge)
        mesh.withUnsafeMutableIndices { raw in
            let indices = raw.bindMemory(to: UInt32.self)
            for i in 0..<newCapacity {
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

        mesh.parts.replaceAll([
            LowLevelMesh.Part(indexCount: totalIndices, topology: .triangle, materialIndex: 0, bounds: Self.batchMeshBounds)
        ])

        edgeBatchMesh = mesh
        edgeBatchCapacity = newCapacity
        lastEdgePartIndexCount = totalIndices
        Self.assignBatchEntity(mesh: mesh, material: edgeBatchMaterial, existingEntity: &edgeBatchEntity, name: "edge_batch", container: edgeContainer)
    }

    // MARK: - Update Node Batch

    func updateNodeBatch(positions: [UUID: SIMD3<Float>],
                         nodes: [NodeData], hubs: Set<UUID>,
                         colorMap: [String: Color],
                         selectedNode: UUID?,
                         glowingNodes: [UUID: Date],
                         newNodes: [UUID: Date]) {
        // Invalidate SIMD color caches when the color map changes
        if let version = renderStore?.colorMapVersion, version != lastColorMapVersion {
            nodeColorCache.removeAll(keepingCapacity: true)
            edgeColorCache.removeAll(keepingCapacity: true)
            lastColorMapVersion = version
        }

        let nodeCount = positions.count
        guard nodeCount > 0, vertsPerSphere > 0 else {
            if lastNodePartIndexCount != 0 {
                frameLog.error("[nodeBatch] ⚠️ ZERO-GUARD: clearing parts (nodeCount=\(nodeCount), vps=\(self.vertsPerSphere))")
                nodeBatchMesh?.parts.replaceAll([])
                lastNodePartIndexCount = 0
            }
            return
        }

        let prevCapacity = nodeBatchCapacity
        ensureNodeBatchMesh(capacity: nodeCount)
        if nodeBatchCapacity != prevCapacity {
            frameLog.error("[nodeBatch] ⚠️ CAPACITY GREW: \(prevCapacity) → \(self.nodeBatchCapacity) (nodeCount=\(nodeCount))")
        }
        guard let mesh = nodeBatchMesh else { return }

        let nodeById = renderStore?.nodeById ?? [:]
        let now = Date()
        let camPos = cameraPosition
        let fogNear: Float = 100
        let fogFar: Float = 1200

        // Ensure instance array is large enough
        if instanceArray.count < nodeCount {
            instanceArray = [NodeInstance](repeating: NodeInstance(px: 0, py: 0, pz: 0, scale: 0, cr: 0, cg: 0, cb: 0, packed: 0), count: max(nodeCount * 2, 512))
        }

        // Fill instance array + track point lights
        var activeLightIds = Set<UUID>()
        var idx = 0

        for (id, pos) in positions {
            guard let nodeData = nodeById[id] else { continue }

            let worldPos = pos * scaleFactor
            let isHub = hubs.contains(id)
            let importance = max(1, nodeData.importance)
            let baseRadius: Float = isHub ? nodeRadius * 1.6 : nodeRadius
            let r = baseRadius * (1.0 + Float(importance - 1) * 0.08)

            // Recall glow intensity
            let ri: Float = {
                guard let glowStart = glowingNodes[id], id != selectedNode else { return 0 }
                let elapsed = Float(now.timeIntervalSince(glowStart))
                let fadeIn: Float = 1.0, hold: Float = 1.5, fadeOut: Float = 2.0
                if elapsed < fadeIn { let t = elapsed / fadeIn; return t * t * t }
                else if elapsed < fadeIn + hold { return 1.0 }
                else if elapsed < fadeIn + hold + fadeOut {
                    let t = 1.0 - (elapsed - fadeIn - hold) / fadeOut; return t * t
                }
                return 0
            }()

            // Arrival intensity
            let ai: Float = {
                guard let arrivalTime = newNodes[id] else { return 0 }
                let elapsed = Float(now.timeIntervalSince(arrivalTime))
                let fadeIn: Float = 0.8, hold: Float = 2.0, fadeOut: Float = 3.0
                if elapsed < fadeIn { let t = elapsed / fadeIn; return t * t * t }
                else if elapsed < fadeIn + hold { return 1.0 }
                else if elapsed < fadeIn + hold + fadeOut {
                    let t = 1.0 - (elapsed - fadeIn - hold) / fadeOut; return t * t
                }
                return 0
            }()

            let searchDimmed = renderIsSearchActive && !renderSearchMatchIds.contains(id)
            let searchMatched = renderIsSearchActive && renderSearchMatchIds.contains(id) && id != selectedNode

            let curState: Float
            let curIntensity: Float
            if id == selectedNode {
                curState = 1; curIntensity = 0
            } else if ri > 0 {
                curState = 2; curIntensity = ri
            } else if ai > 0 {
                curState = 3; curIntensity = ai
            } else if searchMatched {
                curState = 4; curIntensity = 0
            } else {
                curState = 0; curIntensity = 0
            }

            let packedState: Float = curState + (searchDimmed ? 10.0 : 0.0) + curIntensity * 0.01

            let color: SIMD3<Float> = id == selectedNode
                ? SIMD3<Float>(1, 1, 1)
                : nodeColorFloat3(for: nodeData.project, colorMap: colorMap)

            instanceArray[idx] = NodeInstance(
                px: worldPos.x, py: worldPos.y, pz: worldPos.z,
                scale: r,
                cr: color.x, cg: color.y, cb: color.z,
                packed: packedState
            )

            idx += 1

            // Point lights for glowing/arriving/search-matched nodes.
            // Reuses deactivated lights from the pool to avoid addChild scene graph churn.
            if !Self.DIAG_SKIP_POINT_LIGHTS, ri > 0 || ai > 0 || searchMatched {
                activeLightIds.insert(id)
                let lightEntity: Entity
                if let existing = pointLightEntities[id] {
                    lightEntity = existing
                } else if let recycled = pointLightEntities.first(where: { !activeLightIds.contains($0.key) && $0.key != id }) {
                    // Recycle a deactivated light entity — avoids addChild()
                    pointLightEntities.removeValue(forKey: recycled.key)
                    lightEntity = recycled.value
                    lightEntity.name = "ptlight_\(id)"
                    pointLightEntities[id] = lightEntity
                } else {
                    lightEntity = Entity()
                    lightEntity.name = "ptlight_\(id)"
                    rootEntity.addChild(lightEntity)
                    pointLightEntities[id] = lightEntity
                }
                lightEntity.position = worldPos

                let dist = simd_length(pos - camPos)
                let fogT = max(Float(0), min(Float(1), (dist - fogNear) / (fogFar - fogNear)))
                let depthFade = max(Float(0.0), 1.0 - fogT * 0.95)

                if ri > 0 {
                    let lightPulse = 1.0 + sin(animationTime * 3.0) * 0.4
                    lightEntity.components.set(PointLightComponent(
                        color: NSColor(red: 0.7, green: 0.9, blue: 1.0, alpha: 1),
                        intensity: 20000 * ri * depthFade * lightPulse,
                        attenuationRadius: 2.0 * depthFade + 0.1
                    ))
                } else if ai > 0 {
                    let lightPulse = 1.0 + sin(animationTime * 2.5) * 0.4
                    lightEntity.components.set(PointLightComponent(
                        color: NSColor(red: 1.0, green: 0.8, blue: 0.3, alpha: 1),
                        intensity: 20000 * ai * depthFade * lightPulse,
                        attenuationRadius: 2.0 * depthFade + 0.1
                    ))
                } else {
                    // Search-matched: distance-independent glow
                    let lightPulse = 1.0 + sin(animationTime * 4.0) * 0.3
                    lightEntity.components.set(PointLightComponent(
                        color: NSColor(red: 0.0, green: 0.9, blue: 1.0, alpha: 1),
                        intensity: 20000 * lightPulse,
                        attenuationRadius: 2.5
                    ))
                }
            }
        }
        let actualNodeCount = idx
        if actualNodeCount != nodeCount && renderFrameCount % 60 == 0 {
            frameLog.error("[nodeBatch] ⚠️ MISMATCH: actualNodeCount=\(actualNodeCount) vs positions.count=\(nodeCount) nodes.count=\(nodes.count)")
        }

        // --- GPU stamp: Metal compute kernel writes sphere vertices into mesh buffer ---
        // ~500 nodes × 146 verts = 73K vertices stamped in parallel on GPU.
        // CPU only uploads ~16KB of instance data (vs 3.4MB of vertex data previously).
        // Uses LowLevelMesh.replace(bufferIndex:using:) for synchronized triple-buffer writes.
        let vps = vertsPerSphere
        guard let queue = metalCommandQueue,
              let pipeline = stampComputePipeline,
              let templateBuf = sphereTemplateBuffer,
              let paramsBuf = stampParamsBuffer,
              let cmdBuf = queue.makeCommandBuffer() else {
            frameLog.error("[nodeBatch] ❌ GPU stamp unavailable — skipping frame")
            return
        }

        // Ensure instance GPU buffer is large enough
        if stampInstanceCapacity < actualNodeCount {
            let newCap = max(actualNodeCount * 2, 512)
            stampInstanceBuffer = metalDevice?.makeBuffer(
                length: newCap * MemoryLayout<NodeInstance>.stride,
                options: .storageModeShared
            )
            stampInstanceCapacity = newCap
        }
        guard let instanceBuf = stampInstanceBuffer else { return }

        // Upload instance data to shared GPU buffer
        instanceArray.withUnsafeBufferPointer { ptr in
            guard let base = ptr.baseAddress else { return }
            instanceBuf.contents().copyMemory(
                from: base,
                byteCount: actualNodeCount * MemoryLayout<NodeInstance>.stride
            )
        }

        // Set dispatch parameters
        let paramsPtr = paramsBuf.contents().bindMemory(to: StampParams.self, capacity: 1)
        paramsPtr.pointee = StampParams(
            vertsPerNode: UInt32(vps),
            nodeCount: UInt32(actualNodeCount)
        )

        // Get writable output buffer — synchronized with RealityKit's triple-buffering
        let outputBuf = mesh.replace(bufferIndex: 0, using: cmdBuf)

        // Encode compute dispatch
        guard let encoder = cmdBuf.makeComputeCommandEncoder() else {
            cmdBuf.commit()
            return
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(templateBuf, offset: 0, index: 0)
        encoder.setBuffer(instanceBuf, offset: 0, index: 1)
        encoder.setBuffer(paramsBuf, offset: 0, index: 2)
        encoder.setBuffer(outputBuf, offset: 0, index: 3)

        let totalVerts = actualNodeCount * vps
        let threadgroupSize = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        let threadgroups = (totalVerts + threadgroupSize - 1) / threadgroupSize
        encoder.dispatchThreadgroups(
            MTLSize(width: threadgroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1)
        )
        encoder.endEncoding()

        // Zero stale vertices beyond actualNodeCount to prevent ghost rendering.
        // Part indexCount covers full capacity, so unwritten slots must be degenerate.
        let writtenBytes = actualNodeCount * vps * 48  // 48 bytes per BatchVertex
        let totalBytes = nodeBatchCapacity * vps * 48
        if writtenBytes < totalBytes, let blit = cmdBuf.makeBlitCommandEncoder() {
            blit.fill(buffer: outputBuf, range: writtenBytes..<totalBytes, value: 0)
            blit.endEncoding()
        }

        cmdBuf.commit()

        // Use capacity-based index count — parts.replaceAll only fires when recovering
        // from the zero-guard (which clears parts and sets lastNodePartIndexCount=0).
        // During normal operation, capacity doesn't change so this never fires.
        let capacityIndexCount = nodeBatchCapacity * indicesPerSphere
        if capacityIndexCount != lastNodePartIndexCount {
            frameLog.error("[nodeBatch] ⚠️ PARTS RECOVERY: \(self.lastNodePartIndexCount) → \(capacityIndexCount)")
            let generousBounds = BoundingBox(min: SIMD3(-10, -10, -10), max: SIMD3(10, 10, 10))
            mesh.parts.replaceAll([
                LowLevelMesh.Part(
                    indexCount: capacityIndexCount,
                    topology: .triangle,
                    materialIndex: 0,
                    bounds: generousBounds
                )
            ])
            lastNodePartIndexCount = capacityIndexCount
        }

        // Deactivate stale point lights instead of removing them from the scene graph.
        // removeFromParent() triggers scene graph rebuilds → flicker. Setting intensity to 0
        // keeps the entity in the tree (no rebuild) but makes it invisible/free.
        // Deactivated lights accumulate in pointLightEntities and are reused when needed.
        if Self.DIAG_SKIP_POINT_LIGHTS {
            for (_, e) in pointLightEntities {
                e.components.set(PointLightComponent(color: .black, intensity: 0, attenuationRadius: 0))
            }
        } else {
            for id in pointLightEntities.keys where !activeLightIds.contains(id) {
                pointLightEntities[id]?.components.set(
                    PointLightComponent(color: .black, intensity: 0, attenuationRadius: 0)
                )
            }
        }
    }

    /// Update dying nodes — individual PBR entities with red flash animation (max ~5 at a time).

    // MARK: - Update Dying Nodes

    func updateDyingNodes(positions: [UUID: SIMD3<Float>],
                          dyingNodes: [UUID: DyingNode]) {
        let now = Date()

        // Capture 3D positions for dying nodes while they're still in positions
        for id in dyingNodes.keys {
            if dyingNodePos3D[id] == nil, let pos = positions[id] {
                dyingNodePos3D[id] = pos
            }
        }

        for (id, dying) in dyingNodes {
            if positions[id] != nil { continue }  // still live, handled in batch

            let elapsed = Float(now.timeIntervalSince(dying.startTime))
            let flashIn: Float = 0.3, hold: Float = 1.2, fadeOut: Float = 1.5
            let total = flashIn + hold + fadeOut

            guard elapsed < total else {
                if let entity = dyingNodeEntities[id] {
                    entity.removeFromParent()
                    dyingNodeEntities.removeValue(forKey: id)
                }
                dyingNodePos3D.removeValue(forKey: id)
                continue
            }

            // Create entity on first frame if needed
            if dyingNodeEntities[id] == nil {
                let pos3D = dyingNodePos3D[id] ?? .zero
                let entity = ModelEntity(mesh: nodeMesh, materials: [PhysicallyBasedMaterial()])
                entity.position = pos3D * scaleFactor
                entity.name = "dying_\(id)"
                rootEntity.addChild(entity)
                dyingNodeEntities[id] = entity
            }
            guard let entity = dyingNodeEntities[id] else { continue }

            let di: Float
            if elapsed < flashIn {
                let t = elapsed / flashIn; di = t * t
            } else if elapsed < flashIn + hold {
                di = 1.0
            } else {
                let t = 1.0 - (elapsed - flashIn - hold) / fadeOut; di = t * t
            }

            let darkening = elapsed > flashIn ? min(1.0, (elapsed - flashIn) / (hold + fadeOut)) : 0.0
            var mat = PhysicallyBasedMaterial()
            mat.baseColor = .init(tint: NSColor(
                red: CGFloat(0.9 * (1 - darkening * 0.8)),
                green: CGFloat(0.15 * (1 - darkening)),
                blue: CGFloat(0.1 * (1 - darkening)),
                alpha: 1
            ))
            mat.emissiveColor = .init(color: NSColor(red: 1, green: 0.2, blue: 0.1, alpha: 1))
            mat.emissiveIntensity = 2.5 * di
            mat.roughness = .init(floatLiteral: 0.3)
            entity.model?.materials = [mat]

            entity.components.set(PointLightComponent(
                color: NSColor(red: 1.0, green: 0.15, blue: 0.05, alpha: 1),
                intensity: 15000 * di,
                attenuationRadius: 1.5
            ))

            let shrink: Float = elapsed > flashIn + hold ? 1.0 - (1.0 - di) * 0.5 : 1.0
            let baseR: Float = dying.isHub ? nodeRadius * 1.6 : nodeRadius
            let impR = baseR * (1.0 + Float(max(1, dying.importance) - 1) * 0.08)
            entity.scale = SIMD3<Float>(repeating: impR * shrink)
            entity.components.set(OpacityComponent(opacity: di))
        }

        // Clean up finished dying entities
        for id in dyingNodeEntities.keys where dyingNodes[id] == nil {
            dyingNodeEntities[id]?.removeFromParent()
            dyingNodeEntities.removeValue(forKey: id)
        }
    }

    /// Rebuild the batched edge mesh. All edges are rendered as a single LowLevelMesh
    /// with per-edge state encoded in vertex UV and color in vertex color attribute.
    /// This produces 1 draw call instead of 1238 separate entities.

    // MARK: - Update Edge Batch

    func updateEdgeBatch(positions: [UUID: SIMD3<Float>],
                         edges: [EdgeData],
                         nodes: [NodeData],
                         hubs: Set<UUID>,
                         layoutMode: LayoutMode,
                         colorMap: [String: Color],
                         selectedNode: UUID?) {
        let edgeCount = edges.count
        guard edgeCount > 0 else {
            if lastEdgePartIndexCount != 0 {
                frameLog.error("[edgeBatch] ⚠️ ZERO-GUARD: clearing parts (edgeCount=0)")
                edgeBatchMesh?.parts.replaceAll([])
                lastEdgePartIndexCount = 0
            }
            return
        }

        ensureEdgeBatchMesh(capacity: edgeCount)
        guard let mesh = edgeBatchMesh else { return }

        let isSemanticMode = layoutMode == .embedding
        let nodeProject: [UUID: String] = Dictionary(
            nodes.map { ($0.id, $0.project) }, uniquingKeysWith: { _, last in last }
        )

        // Build per-node radius lookup for endpoint inset (same formula as updateNodeBatch)
        var nodeRadii: [UUID: Float] = [:]
        for node in nodes {
            let isHub = hubs.contains(node.id)
            let importance = max(1, node.importance)
            let baseR: Float = isHub ? nodeRadius * 1.6 : nodeRadius
            nodeRadii[node.id] = baseR * (1.0 + Float(importance - 1) * 0.08)
        }

        // --- GPU-synchronized edge buffer write using replace+blit pattern ---
        // Uses staging MTLBuffer for CPU vertex generation, then blit-copies to the
        // RealityKit-managed output buffer via LowLevelMesh.replace(bufferIndex:using:).
        // This eliminates flicker from replaceUnsafeMutableBytes synchronization gaps.
        guard let queue = metalCommandQueue,
              let cmdBuf = queue.makeCommandBuffer() else {
            frameLog.error("[edgeBatch] ❌ Metal queue unavailable — skipping frame")
            return
        }

        let vertsPerEdge = 12
        let bytesPerVertex = 48  // sizeof(BatchVertex)
        let totalStagingVerts = edgeBatchCapacity * vertsPerEdge
        let totalStagingBytes = totalStagingVerts * bytesPerVertex

        // Ensure staging buffer is large enough
        if edgeStagingCapacity < edgeBatchCapacity {
            edgeStagingBuffer = metalDevice?.makeBuffer(
                length: totalStagingBytes,
                options: .storageModeShared
            )
            edgeStagingCapacity = edgeBatchCapacity
        }
        guard let stagingBuf = edgeStagingBuffer else { return }

        // CPU-fill staging buffer with edge cylinder geometry
        var actualEdges = 0
        let vertices = stagingBuf.contents().bindMemory(to: BatchVertex.self, capacity: totalStagingVerts)
        var vi = 0

        for edge in edges {
            guard let from = positions[edge.sourceId],
                  let to = positions[edge.targetId] else {
                let zero = BatchVertex(px: 0, py: 0, pz: 0, nx: 0, ny: 1, nz: 0,
                                           u: 0, v: 0, cr: 0, cg: 0, cb: 0, ca: 0)
                for _ in 0..<12 { vertices[vi] = zero; vi += 1 }
                continue
            }

            let p1 = from * scaleFactor
            let p2 = to * scaleFactor
            let delta = p2 - p1
            let length = simd_length(delta)

            let r1 = nodeRadii[edge.sourceId] ?? nodeRadius
            let r2 = nodeRadii[edge.targetId] ?? nodeRadius
            guard length > r1 + r2 else {
                let zero = BatchVertex(px: 0, py: 0, pz: 0, nx: 0, ny: 1, nz: 0,
                                           u: 0, v: 0, cr: 0, cg: 0, cb: 0, ca: 0)
                for _ in 0..<12 { vertices[vi] = zero; vi += 1 }
                continue
            }

            let dir = delta / length
            let p1inset = p1 + dir * r1
            let p2inset = p2 - dir * r2

            let basisUp: SIMD3<Float> = abs(dir.y) < 0.99 ? SIMD3(0, 1, 0) : SIMD3(1, 0, 0)
            let basisRight = simd_normalize(cross(dir, basisUp))
            let basisForward = simd_normalize(cross(basisRight, dir))

            let connected = edge.sourceId == selectedNode || edge.targetId == selectedNode
            let radius = connected ? edgeRadius * 2.5 : edgeRadius * 1.3

            let searchDimmed = renderIsSearchActive
                && !renderSearchMatchIds.contains(edge.sourceId)
                && !renderSearchMatchIds.contains(edge.targetId)
            let state: Float
            if connected { state = 1 }
            else if searchDimmed { state = 2 }
            else if isSemanticMode { state = 3 }
            else { state = 0 }

            let color: SIMD3<Float>
            if connected {
                color = SIMD3<Float>(1, 1, 1)
            } else {
                color = edgeColorFloat3(for: nodeProject[edge.sourceId], colorMap: colorMap)
            }

            // Cylinder: 12 vertices (bottom ring 0..5, top ring 6..11)
            for seg in 0..<6 {
                let angle = Float(seg) * (.pi * 2 / 6)
                let c = cos(angle), s = sin(angle)
                let offset = (basisRight * c + basisForward * s) * radius
                let normal = simd_normalize(basisRight * c + basisForward * s)
                let bp = p1inset + offset
                vertices[vi] = BatchVertex(px: bp.x, py: bp.y, pz: bp.z,
                                               nx: normal.x, ny: normal.y, nz: normal.z,
                                               u: state, v: 0,
                                               cr: color.x, cg: color.y, cb: color.z, ca: 1)
                vi += 1
            }
            for seg in 0..<6 {
                let angle = Float(seg) * (.pi * 2 / 6)
                let c = cos(angle), s = sin(angle)
                let offset = (basisRight * c + basisForward * s) * radius
                let normal = simd_normalize(basisRight * c + basisForward * s)
                let tp = p2inset + offset
                vertices[vi] = BatchVertex(px: tp.x, py: tp.y, pz: tp.z,
                                               nx: normal.x, ny: normal.y, nz: normal.z,
                                               u: state, v: 1,
                                               cr: color.x, cg: color.y, cb: color.z, ca: 1)
                vi += 1
            }
            actualEdges += 1
        }

        // Get writable output buffer — synchronized with RealityKit's triple-buffering
        let outputBuf = mesh.replace(bufferIndex: 0, using: cmdBuf)

        // Blit copy from staging to output
        let writtenBytes = vi * bytesPerVertex
        if let blit = cmdBuf.makeBlitCommandEncoder() {
            if writtenBytes > 0 {
                blit.copy(from: stagingBuf, sourceOffset: 0,
                          to: outputBuf, destinationOffset: 0,
                          size: writtenBytes)
            }
            // Zero stale vertices beyond actual edges to prevent ghost rendering
            if writtenBytes < totalStagingBytes {
                blit.fill(buffer: outputBuf, range: writtenBytes..<totalStagingBytes, value: 0)
            }
            blit.endEncoding()
        }

        cmdBuf.commit()

        // Use capacity-based index count — parts.replaceAll only fires when recovering
        // from the zero-guard (which clears parts and sets lastEdgePartIndexCount=0).
        let capacityIndexCount = edgeBatchCapacity * 36
        if capacityIndexCount != lastEdgePartIndexCount {
            frameLog.error("[edgeBatch] ⚠️ PARTS RECOVERY: \(self.lastEdgePartIndexCount) → \(capacityIndexCount)")
            let generousBounds = BoundingBox(min: SIMD3(-10, -10, -10), max: SIMD3(10, 10, 10))
            mesh.parts.replaceAll([
                LowLevelMesh.Part(
                    indexCount: capacityIndexCount,
                    topology: .triangle,
                    materialIndex: 0,
                    bounds: generousBounds
                )
            ])
            lastEdgePartIndexCount = capacityIndexCount
        }
    }

    // MARK: - Diagnostics

    // MARK: - Diagnostics: disable expensive features to isolate bottleneck
    // Toggle these to measure the impact of each subsystem.
    // RESULTS: record dt after each toggle, then remove this block.
    static let DIAG_SKIP_NEBULAE = false      // skip ALL particle emitters
    static let DIAG_SKIP_POINT_LIGHTS = false  // skip per-node point lights
    static let DIAG_SKIP_GEOM_MOD = false     // skip geometry modifier on node batch
    static let DIAG_SKIP_NODE_BATCH = false   // skip node LowLevelMesh update
    static let DIAG_SKIP_EDGE_BATCH = false   // skip edge LowLevelMesh update
    static let DIAG_SKIP_LABEL_BATCH = false  // skip label LowLevelMesh update
}
