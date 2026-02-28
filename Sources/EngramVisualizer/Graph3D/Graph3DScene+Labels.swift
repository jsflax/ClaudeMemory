import SwiftUI
import RealityKit
import Metal
import simd
import os

private let frameLog = Logger(subsystem: "com.claudememory.visualizer", category: "3DFrameTiming")

extension Graph3DScene {

    // MARK: - Label Atlas Generation

    func generateLabelAtlas(nodes: [NodeData], hubs: Set<UUID>, projects: Set<String>) {
        let atlasW = 4096
        let padding: CGFloat = 4
        let fontSize: CGFloat = 28
        let projFontSize: CGFloat = 40

        // Pre-compute all label sizes (measurement pass)
        struct LabelSize { let width: CGFloat; let height: CGFloat }
        var nodeSizes: [LabelSize] = []
        nodeSizes.reserveCapacity(nodes.count)
        for node in nodes {
            let isHub = hubs.contains(node.id)
            let weight: NSFont.Weight = isHub ? .bold : .medium
            let font = NSFont.monospacedSystemFont(ofSize: fontSize, weight: weight)
            let attrs: [NSAttributedString.Key: Any] = [.font: font, .foregroundColor: NSColor.white]
            let size = (node.label as NSString).size(withAttributes: attrs)
            nodeSizes.append(LabelSize(width: ceil(size.width) + padding * 2, height: ceil(size.height) + padding))
        }

        let projFont = NSFont.systemFont(ofSize: projFontSize, weight: .bold)
        let projAttrs: [NSAttributedString.Key: Any] = [.font: projFont, .foregroundColor: NSColor.white]
        let sortedProjects = projects.sorted()
        var projSizes: [LabelSize] = []
        for project in sortedProjects {
            let size = (project as NSString).size(withAttributes: projAttrs)
            projSizes.append(LabelSize(width: ceil(size.width) + padding * 2, height: ceil(size.height) + padding))
        }

        // Simulate packing to compute needed height
        var cursorX: CGFloat = 2
        var cursorY: CGFloat = 0
        var rowHeight: CGFloat = 0
        for s in nodeSizes {
            if cursorX + s.width > CGFloat(atlasW) { cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0 }
            cursorX += s.width + padding; rowHeight = max(rowHeight, s.height)
        }
        cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0
        for s in projSizes {
            if cursorX + s.width > CGFloat(atlasW) { cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0 }
            cursorX += s.width + padding; rowHeight = max(rowHeight, s.height)
        }
        let neededH = Int(ceil(cursorY + rowHeight + padding))

        let scale = 2  // retina
        let neededPixelW = atlasW * scale
        let neededPixelH = max(512, neededH + 32) * scale

        // Determine if existing LowLevelTexture can be reused or needs (re)creation.
        // Over-allocate height (2x or min 2048pt) to minimize rare material-swap resizes.
        let needsNewLLT = labelAtlasLLT == nil
            || neededPixelW > labelAtlasAllocW
            || neededPixelH > labelAtlasAllocH

        if needsNewLLT {
            let allocH = max(neededPixelH, 2048 * scale)
            labelAtlasAllocW = neededPixelW
            labelAtlasAllocH = allocH
        }

        // Use allocated dimensions for UV computation so UVs map correctly
        // into the (potentially larger) LowLevelTexture.
        let uvAtlasW = labelAtlasAllocW / scale
        let uvAtlasH = labelAtlasAllocH / scale
        labelAtlasAspectCorrection = Float(uvAtlasW) / (2.0 * Float(uvAtlasH))

        // CGContext at full allocated pixel size (matches LLT dimensions)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil, width: labelAtlasAllocW, height: labelAtlasAllocH,
            bitsPerComponent: 8, bytesPerRow: labelAtlasAllocW * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            frameLog.error("[labelAtlas] ❌ CGContext creation failed")
            return
        }

        // Flip for correct text rendering (Core Graphics is bottom-up)
        ctx.translateBy(x: 0, y: CGFloat(labelAtlasAllocH))
        ctx.scaleBy(x: CGFloat(scale), y: CGFloat(-scale))

        // Drawing pass — node labels (UVs relative to allocated atlas size)
        var rects: [UUID: (u0: Float, v0: Float, u1: Float, v1: Float)] = [:]
        cursorX = 2; cursorY = 0; rowHeight = 0

        for (i, node) in nodes.enumerated() {
            let s = nodeSizes[i]
            if cursorX + s.width > CGFloat(atlasW) { cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0 }
            if cursorY + s.height > CGFloat(uvAtlasH) { break }

            let isHub = hubs.contains(node.id)
            let weight: NSFont.Weight = isHub ? .bold : .medium
            let font = NSFont.monospacedSystemFont(ofSize: fontSize, weight: weight)
            let attrs: [NSAttributedString.Key: Any] = [.font: font, .foregroundColor: NSColor.white]

            let drawPoint = CGPoint(x: cursorX + padding, y: cursorY + padding * 0.5)
            NSGraphicsContext.saveGraphicsState()
            NSGraphicsContext.current = NSGraphicsContext(cgContext: ctx, flipped: true)
            (node.label as NSString).draw(at: drawPoint, withAttributes: attrs)
            NSGraphicsContext.restoreGraphicsState()

            rects[node.id] = (
                Float(cursorX) / Float(uvAtlasW), Float(cursorY) / Float(uvAtlasH),
                Float(cursorX + s.width) / Float(uvAtlasW), Float(cursorY + s.height) / Float(uvAtlasH)
            )
            cursorX += s.width + padding; rowHeight = max(rowHeight, s.height)
        }

        // Drawing pass — project labels
        var projRects: [String: (u0: Float, v0: Float, u1: Float, v1: Float)] = [:]
        cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0

        for (i, project) in sortedProjects.enumerated() {
            let s = projSizes[i]
            if cursorX + s.width > CGFloat(atlasW) { cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0 }
            if cursorY + s.height > CGFloat(uvAtlasH) { break }

            let drawPoint = CGPoint(x: cursorX + padding, y: cursorY + padding * 0.5)
            NSGraphicsContext.saveGraphicsState()
            NSGraphicsContext.current = NSGraphicsContext(cgContext: ctx, flipped: true)
            (project as NSString).draw(at: drawPoint, withAttributes: projAttrs)
            NSGraphicsContext.restoreGraphicsState()

            projRects[project] = (
                Float(cursorX) / Float(uvAtlasW), Float(cursorY) / Float(uvAtlasH),
                Float(cursorX + s.width) / Float(uvAtlasW), Float(cursorY + s.height) / Float(uvAtlasH)
            )
            cursorX += s.width + padding; rowHeight = max(rowHeight, s.height)
        }

        // Get raw pixel data from CGContext
        guard let pixelData = ctx.data else {
            frameLog.error("[labelAtlas] ❌ CGContext data is nil")
            return
        }
        let bytesPerRow = labelAtlasAllocW * 4

        // Helper: blit CPU pixel data into a private-storage MTLTexture via staging buffer.
        func blitPixelData(to llt: LowLevelTexture, device: MTLDevice, queue: MTLCommandQueue,
                           data: UnsafeRawPointer, width: Int, height: Int, bytesPerRow: Int) -> Bool {
            let totalBytes = bytesPerRow * height
            guard let staging = device.makeBuffer(bytes: data, length: totalBytes, options: .storageModeShared),
                  let cmdBuf = queue.makeCommandBuffer() else { return false }
            let dstTexture = llt.replace(using: cmdBuf)
            guard let blit = cmdBuf.makeBlitCommandEncoder() else { cmdBuf.commit(); return false }
            blit.copy(
                from: staging, sourceOffset: 0, sourceBytesPerRow: bytesPerRow, sourceBytesPerImage: totalBytes,
                sourceSize: MTLSize(width: width, height: height, depth: 1),
                to: dstTexture, destinationSlice: 0, destinationLevel: 0, destinationOrigin: MTLOrigin()
            )
            blit.endEncoding()
            cmdBuf.commit()
            return true
        }

        guard let device = metalDevice, let queue = metalCommandQueue else {
            frameLog.error("[labelAtlas] ❌ Metal device or command queue unavailable")
            return
        }

        if needsNewLLT {
            // Create new LowLevelTexture + TextureResource + material swap (rare)
            let allocW = self.labelAtlasAllocW
            let allocH = self.labelAtlasAllocH
            let desc = LowLevelTexture.Descriptor(
                textureType: .type2D,
                pixelFormat: .rgba8Unorm,
                width: allocW,
                height: allocH,
                mipmapLevelCount: 1,
                textureUsage: [.shaderRead]
            )

            guard let llt = try? LowLevelTexture(descriptor: desc) else {
                frameLog.error("[labelAtlas] ❌ LowLevelTexture creation failed")
                return
            }

            guard blitPixelData(to: llt, device: device, queue: queue,
                                data: pixelData, width: allocW, height: allocH, bytesPerRow: bytesPerRow) else {
                frameLog.error("[labelAtlas] ❌ blit to new LLT failed")
                return
            }

            guard let textureResource = try? TextureResource(from: llt) else {
                frameLog.error("[labelAtlas] ❌ TextureResource from LowLevelTexture failed")
                return
            }

            labelAtlasLLT = llt
            labelAtlasTexture = textureResource

            // Material swap — only on first creation or rare resize
            if var mat = labelBatchMaterial {
                mat.custom.texture = .init(CustomMaterial.Texture(textureResource))
                labelBatchMaterial = mat
                labelBatchEntity?.model?.materials = [mat]
            }

            frameLog.error("[labelAtlas] ✅ NEW LLT: \(rects.count) node + \(projRects.count) project labels, alloc=\(allocW/scale)×\(allocH/scale), needed=\(atlasW)×\(neededPixelH/scale)")
        } else {
            // In-place update — NO material swap, no entity rebuild.
            guard let llt = labelAtlasLLT else { return }
            let allocW = self.labelAtlasAllocW
            let allocH = self.labelAtlasAllocH

            guard blitPixelData(to: llt, device: device, queue: queue,
                                data: pixelData, width: allocW, height: allocH, bytesPerRow: bytesPerRow) else {
                frameLog.error("[labelAtlas] ❌ blit to existing LLT failed")
                return
            }

            frameLog.error("[labelAtlas] ✅ IN-PLACE update: \(rects.count) node + \(projRects.count) project labels")
        }

        labelAtlasRects = rects
        projectLabelAtlasRects = projRects
        labelAtlasNodeIds = Set(nodes.map(\.id))
        labelAtlasHubIds = hubs
        labelAtlasProjects = projects
    }

    // MARK: - Label Batch Mesh

    /// Create or resize the LowLevelMesh for batched label rendering.
    /// 4 vertices per label (billboard quad), 6 indices per quad.
    func ensureLabelBatchMesh(capacity: Int) {
        guard capacity > labelBatchCapacity else { return }
        let newCapacity = max(capacity * 2, 512)

        let totalVerts = newCapacity * 4   // 4 verts per quad
        let totalIndices = newCapacity * 6 // 6 indices per quad (2 triangles)
        let desc = Self.makeBatchMeshDescriptor(vertexCapacity: totalVerts, indexCapacity: totalIndices)

        guard let mesh = try? LowLevelMesh(descriptor: desc) else {
            frameLog.error("[labelBatch] ❌ LowLevelMesh creation failed")
            return
        }

        // Pre-fill index buffer: 2 triangles per quad (CCW winding for billboard facing camera)
        mesh.withUnsafeMutableIndices { raw in
            let indices = raw.bindMemory(to: UInt32.self)
            for i in 0..<newCapacity {
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

        mesh.parts.replaceAll([
            LowLevelMesh.Part(indexCount: totalIndices, topology: .triangle, materialIndex: 0, bounds: Self.batchMeshBounds)
        ])

        labelBatchMesh = mesh
        labelBatchCapacity = newCapacity
        lastLabelPartIndexCount = totalIndices
        Self.assignBatchEntity(mesh: mesh, material: labelBatchMaterial, existingEntity: &labelBatchEntity, name: "label_batch", container: rootEntity)
        frameLog.error("[labelBatch] ✅ mesh created: \(newCapacity) labels capacity")
    }

    // MARK: - Update Label Batch

    /// Per-frame vertex stamping for label billboard quads.
    /// Encodes position (anchor), normal (corner offset), UV (atlas), color (opacity).
    func updateLabelBatch(positions: [UUID: SIMD3<Float>],
                          nodes: [NodeData], hubs: Set<UUID>,
                          selectedNode: UUID?) {
        let nodeCount = positions.count
        guard nodeCount > 0 else {
            if lastLabelPartIndexCount != 0 {
                frameLog.error("[labelBatch] ⚠️ ZERO-GUARD: clearing parts")
                labelBatchMesh?.parts.replaceAll([])
                lastLabelPartIndexCount = 0
            }
            return
        }

        // Regenerate atlas if node set, hub set, or project set changed.
        // Debounce: wait 60 frames (~1s) between regens to prevent flicker from
        // texture/UV mismatch in RealityKit's triple-buffered vertex data.
        // Nodes without atlas rects just skip rendering until the next regen.
        let currentNodeIds = Set(positions.keys)
        let currentProjects = Set(nodes.map(\.project))
        let atlasNeedsRegen = labelAtlasTexture == nil || currentNodeIds != labelAtlasNodeIds || hubs != labelAtlasHubIds || currentProjects != labelAtlasProjects
        if atlasNeedsRegen {
            let isFirstAtlas = labelAtlasTexture == nil
            let framesSinceRegen = renderFrameCount &- labelAtlasRegenFrame
            if isFirstAtlas || framesSinceRegen >= 60 {
                generateLabelAtlas(nodes: nodes, hubs: hubs, projects: currentProjects)
                labelAtlasRegenFrame = renderFrameCount
                // Sync labelAtlasNodeIds to match the comparison source (positions.keys)
                // to prevent perpetual regen when nodes and positions sets differ.
                labelAtlasNodeIds = currentNodeIds
            }
        }
        guard labelAtlasTexture != nil else { return }

        // Compute per-project centroids directly from node positions
        var projectNodePositions: [String: [SIMD3<Float>]] = [:]
        for node in nodes {
            guard let pos = positions[node.id] else { continue }
            projectNodePositions[node.project, default: []].append(pos)
        }
        projectCentroids.removeAll(keepingCapacity: true)
        for (project, pts) in projectNodePositions where pts.count >= 2 {
            var sum = SIMD3<Float>.zero
            var maxY: Float = -.greatestFiniteMagnitude
            for p in pts { sum += p; maxY = max(maxY, p.y) }
            let centroid = sum / Float(pts.count)
            var maxDist: Float = 0
            for p in pts { maxDist = max(maxDist, simd_length(p - centroid)) }
            projectCentroids[project] = (centroid: centroid, radius: maxDist, maxY: maxY)
        }

        // Pre-compute project colors outside the unsafe buffer closure
        var projectColors: [String: SIMD3<Float>] = [:]
        for project in projectCentroids.keys {
            projectColors[project] = nodeColorFloat3(for: project, colorMap: renderColorMap)
        }

        let projectLabelCount = projectCentroids.count
        ensureLabelBatchMesh(capacity: nodeCount + projectLabelCount)
        guard let mesh = labelBatchMesh else { return }

        let nodeById = renderStore?.nodeById ?? [:]
        let camPos = cameraPosition
        let sf = scaleFactor
        let aspectCorr = labelAtlasAspectCorrection

        // Merge expanded child positions
        var allPositions = positions
        for (id, pos) in expandedChildPositions {
            allPositions[id] = pos
        }
        var expandedChildren = Set<UUID>()
        for hubId in expandedHubs {
            for childId in childrenOfHub(hubId) {
                expandedChildren.insert(childId)
            }
        }

        // Depth range for normalization
        var minDepth: Float = .greatestFiniteMagnitude
        var maxDepth: Float = 0
        for (_, pos) in allPositions {
            let d = simd_length(pos - camPos)
            minDepth = min(minDepth, d)
            maxDepth = max(maxDepth, d)
        }
        let depthRange = max(1.0, maxDepth - minDepth)

        // --- GPU stamp: Metal compute kernel writes label quad vertices into mesh buffer ---
        // ~500 labels × 64 bytes = ~32KB instance upload (vs 96KB of vertex data previously).
        // Uses LowLevelMesh.replace(bufferIndex:using:) for synchronized triple-buffer writes.
        guard let queue = metalCommandQueue,
              let pipeline = labelStampPipeline,
              let paramsBuf = labelParamsBuffer,
              let cmdBuf = queue.makeCommandBuffer() else {
            frameLog.error("[labelBatch] ❌ GPU stamp unavailable — skipping frame")
            return
        }

        // Build instance array on CPU
        let totalLabelCapacity = labelBatchCapacity
        let maxNodeLabels = totalLabelCapacity - projectLabelCount
        var instances = [LabelInstance]()
        instances.reserveCapacity(min(allPositions.count + projectLabelCount, totalLabelCapacity))

        for (id, pos3D) in allPositions {
            guard instances.count < maxNodeLabels,
                  let nodeData = nodeById[id],
                  let rect = labelAtlasRects[id] else {
                continue
            }

            let isSelected = id == selectedNode
            let isHub = hubs.contains(id)
            let importance = Float(max(1, nodeData.importance))
            let isSearchMatch = renderIsSearchActive && renderSearchMatchIds.contains(id)
            let searchDimmed = renderIsSearchActive && !renderSearchMatchIds.contains(id)

            let maxVisible: Float = (isSelected || expandedChildren.contains(id) || isSearchMatch)
                ? .greatestFiniteMagnitude
                : (isHub ? 1200 : (200 + importance * 80))

            let baseOpacity: Float = isSelected ? 0.95 : (isHub ? 0.8 : 0.6)
            let halfH: Float = isSelected ? 0.025 : (isHub ? 0.022 : 0.018)
            let textAspect = (rect.u1 - rect.u0) / max(0.001, rect.v1 - rect.v0) * aspectCorr

            // Anchor position: node world pos scaled + Y offset above sphere
            let anchor = pos3D * sf + SIMD3<Float>(0, nodeRadius * 1.8, 0)

            var flags: UInt32 = 0
            if isSelected { flags |= 1 }
            if isSearchMatch { flags |= 2 }
            if searchDimmed { flags |= 4 }

            instances.append(LabelInstance(
                ax: anchor.x, ay: anchor.y, az: anchor.z,
                halfH: halfH,
                u0: rect.u0, v0: rect.v0, u1: rect.u1, v1: rect.v1,
                cr: 1, cg: 1, cb: 1, baseOpacity: baseOpacity,
                textAspect: textAspect,
                maxVisible: maxVisible,
                forwardBias: 0,
                flags: flags
            ))
        }

        // Project labels — larger quads floating above each project cluster
        for (project, centroidData) in projectCentroids {
            guard let rect = projectLabelAtlasRects[project] else { continue }

            let labelY = centroidData.maxY * sf + 0.15
            let anchor = SIMD3<Float>(centroidData.centroid.x * sf, labelY, centroidData.centroid.z * sf)
            let color = projectColors[project] ?? SIMD3<Float>(1, 1, 1)
            let textAspect = (rect.u1 - rect.u0) / max(0.001, rect.v1 - rect.v0) * aspectCorr

            instances.append(LabelInstance(
                ax: anchor.x, ay: anchor.y, az: anchor.z,
                halfH: 0.15,
                u0: rect.u0, v0: rect.v0, u1: rect.u1, v1: rect.v1,
                cr: color.x, cg: color.y, cb: color.z, baseOpacity: 0.9,
                textAspect: textAspect,
                maxVisible: 12000,
                forwardBias: 0.25,
                flags: 0
            ))
        }

        let actualLabelCount = instances.count

        // Ensure instance GPU buffer is large enough
        if labelInstanceCapacity < actualLabelCount {
            let newCap = max(actualLabelCount * 2, 512)
            labelInstanceBuffer = metalDevice?.makeBuffer(
                length: newCap * MemoryLayout<LabelInstance>.stride,
                options: .storageModeShared
            )
            labelInstanceCapacity = newCap
        }
        guard let instanceBuf = labelInstanceBuffer else { return }

        // Upload instance data to shared GPU buffer
        instances.withUnsafeBufferPointer { ptr in
            guard let base = ptr.baseAddress else { return }
            instanceBuf.contents().copyMemory(
                from: base,
                byteCount: actualLabelCount * MemoryLayout<LabelInstance>.stride
            )
        }

        // Set dispatch parameters — camera pos in scaled world coords for depth calc
        let scaledCamPos = camPos * sf
        let paramsPtr = paramsBuf.contents().bindMemory(to: LabelStampParams.self, capacity: 1)
        paramsPtr.pointee = LabelStampParams(
            cx: scaledCamPos.x, cy: scaledCamPos.y, cz: scaledCamPos.z,
            minDepth: minDepth * sf, depthRange: depthRange * sf,
            labelCount: UInt32(actualLabelCount)
        )

        // Get writable output buffer — synchronized with RealityKit's triple-buffering
        let outputBuf = mesh.replace(bufferIndex: 0, using: cmdBuf)

        // Encode compute dispatch
        guard let encoder = cmdBuf.makeComputeCommandEncoder() else {
            cmdBuf.commit()
            return
        }
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

        // Zero stale vertices beyond actualLabelCount to prevent ghost rendering
        let writtenBytes = actualLabelCount * 4 * 48  // 4 verts × 48 bytes per BatchVertex
        let totalBytes = labelBatchCapacity * 4 * 48
        if writtenBytes < totalBytes, let blit = cmdBuf.makeBlitCommandEncoder() {
            blit.fill(buffer: outputBuf, range: writtenBytes..<totalBytes, value: 0)
            blit.endEncoding()
        }

        cmdBuf.commit()

        // Capacity-based recovery (same pattern as node/edge)
        let capacityIndexCount = labelBatchCapacity * 6
        if capacityIndexCount != lastLabelPartIndexCount {
            let generousBounds = BoundingBox(min: SIMD3(-10, -10, -10), max: SIMD3(10, 10, 10))
            mesh.parts.replaceAll([
                LowLevelMesh.Part(
                    indexCount: capacityIndexCount,
                    topology: .triangle,
                    materialIndex: 0,
                    bounds: generousBounds
                )
            ])
            lastLabelPartIndexCount = capacityIndexCount
        }
    }
}
