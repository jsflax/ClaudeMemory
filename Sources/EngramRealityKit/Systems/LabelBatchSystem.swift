import RealityKit
import Metal
import simd
import EngramSceneKit

/// Instanced billboard label rendering system.
///
/// Labels are camera-facing quads with texture atlas UV coordinates.
/// Atlas regeneration is debounced (60-frame minimum between rebuilds).
@MainActor
public final class LabelBatchSystem {
    private static let vertsPerLabel = 4
    private static let indicesPerLabel = 6

    private var lastAtlasVersion: UInt64 = 0
    private var lastAtlasFrame: UInt64 = 0
    private var labelStaging: [BatchVertex] = []

    // Cached cluster data — recomputed every N frames instead of every frame
    private var cachedProjectMaxY: [String: Float] = [:]
    private var cachedTopicSums: [String: (sum: SIMD3<Float>, count: Int, project: String, maxY: Float)] = [:]
    private var lastClusterFrame: UInt64 = 0
    private let clusterRescanInterval: UInt64 = 10

    public init() {}

    public func update(
        scene: EngramRealityScene,
        dataProvider: SceneDataProvider,
        visibleSet: VisibleSet,
        topologyChanged: Bool,
        cameraPosition: SIMD3<Float>,
        scaleFactor: Float,
        frameCount: UInt64,
        commandBuffer: MTLCommandBuffer? = nil
    ) {
        let visibleCount = visibleSet.visibleLabelIndices.count
        let projCount = dataProvider.projectCentroids.count
        let topicCount = Set(dataProvider.nodes.map(\.topic)).count
        let totalLabelCount = visibleCount + projCount + topicCount
        if frameCount % 120 == 0 {
            print("[labels] frame=\(frameCount) visibleNodes=\(visibleCount) projects=\(projCount) topics=\(topicCount) totalNodes=\(dataProvider.nodes.count) projRects=\(scene.labelAtlasGenerator.projectRects.count) topicRects=\(scene.labelAtlasGenerator.topicRects.count) atlasFrame=\(lastAtlasFrame)")
        }
        guard totalLabelCount > 0 else { return }

        // Regenerate atlas if topology changed (debounced), or on first frame with data
        let hasNodes = !dataProvider.nodes.isEmpty
        let needsInitialAtlas = lastAtlasFrame == 0 && hasNodes
        if (topologyChanged || needsInitialAtlas) && (frameCount - lastAtlasFrame) > 60 || needsInitialAtlas {
            let projects = Set(dataProvider.nodes.map(\.project))
            let topics = Set(dataProvider.nodes.map(\.topic))
            scene.labelAtlasGenerator.regenerateAtlas(
                nodes: dataProvider.nodes,
                hubs: dataProvider.hubs,
                projects: projects,
                topics: topics
            )
            lastAtlasFrame = frameCount

            // Update material with new atlas texture
            if let texture = scene.labelAtlasGenerator.atlasTexture,
               let labelEntity = scene.labelBatchEntity {
                let mat = MaterialFactory.makeLabelMaterial(device: scene.device, atlasTexture: texture)
                labelEntity.model?.materials = [mat]
            }
        }

        LowLevelMeshFactory.ensureLabelBatchMesh(scene: scene, capacity: totalLabelCount)
        guard let mesh = scene.labelBatchMesh else { return }

        let nodes = dataProvider.nodes
        let positions = dataProvider.positions
        let positionArray = dataProvider.positionArray
        let atlasRects = scene.labelAtlasGenerator.nodeRects
        let aspectCorrection = scene.labelAtlasGenerator.aspectCorrection
        let selectedNode = dataProvider.selectedNode
        let isSearchActive = dataProvider.isSearchActive
        let searchMatchIds = dataProvider.searchMatchIds
        let colorMap = dataProvider.projectColorMap

        // Camera orientation for billboard
        let camPos = cameraPosition * scaleFactor
        let camRight: SIMD3<Float>
        let camUp: SIMD3<Float>

        if let provider = scene.cameraProvider {
            let state = provider.cameraState
            camRight = state.right
            camUp = state.up
        } else {
            camRight = SIMD3<Float>(1, 0, 0)
            camUp = SIMD3<Float>(0, 1, 0)
        }

        let totalVerts = scene.labelBatchCapacity * Self.vertsPerLabel
        if labelStaging.count < totalVerts {
            labelStaging = [BatchVertex](repeating: BatchVertex(
                px: 0, py: 0, pz: 0, nx: 0, ny: 0, nz: 0,
                u: 0, v: 0, cr: 0, cg: 0, cb: 0, ca: 0
            ), count: totalVerts)
        }

        var instanceIdx = 0
        for nodeIdx in visibleSet.visibleLabelIndices {
            let node = nodes[nodeIdx]
            guard let rect = atlasRects[node.id] else { continue }
            let pos = nodeIdx < positionArray.count ? positionArray[nodeIdx] : (positions[node.id] ?? .zero)

            let anchor = pos * scaleFactor + SIMD3<Float>(0, 12.0, 0)

            let baseHalfH: Float = node.isHub ? 2.5 : 2.0
            let halfH = baseHalfH
            let halfW = halfH * (rect.u1 - rect.u0) / max(rect.v1 - rect.v0, 0.001) * aspectCorrection

            let color = colorMap[node.project] ?? SIMD3<Float>(0.8, 0.8, 0.8)

            let dist = simd_length(anchor - camPos)
            var opacity: Float = 1.0 - min(1.0, dist / 1200.0)
            opacity = max(0.1, opacity)
            if node.id == selectedNode { opacity = 1.0 }
            if isSearchActive && !searchMatchIds.contains(node.id) { opacity *= 0.2 }

            let r = camRight * halfW
            let u = camUp * halfH

            let baseVert = instanceIdx * Self.vertsPerLabel
            labelStaging[baseVert + 0] = BatchVertex(
                px: anchor.x - r.x - u.x, py: anchor.y - r.y - u.y, pz: anchor.z - r.z - u.z,
                nx: 0, ny: 0, nz: 1, u: rect.u0, v: rect.v0,
                cr: color.x, cg: color.y, cb: color.z, ca: opacity
            )
            labelStaging[baseVert + 1] = BatchVertex(
                px: anchor.x + r.x - u.x, py: anchor.y + r.y - u.y, pz: anchor.z + r.z - u.z,
                nx: 0, ny: 0, nz: 1, u: rect.u1, v: rect.v0,
                cr: color.x, cg: color.y, cb: color.z, ca: opacity
            )
            labelStaging[baseVert + 2] = BatchVertex(
                px: anchor.x - r.x + u.x, py: anchor.y - r.y + u.y, pz: anchor.z - r.z + u.z,
                nx: 0, ny: 0, nz: 1, u: rect.u0, v: rect.v1,
                cr: color.x, cg: color.y, cb: color.z, ca: opacity
            )
            labelStaging[baseVert + 3] = BatchVertex(
                px: anchor.x + r.x + u.x, py: anchor.y + r.y + u.y, pz: anchor.z + r.z + u.z,
                nx: 0, ny: 0, nz: 1, u: rect.u1, v: rect.v1,
                cr: color.x, cg: color.y, cb: color.z, ca: opacity
            )
            instanceIdx += 1
        }

        // Compute per-project max Y + topic sums (throttled — every 10 frames or on topology change)
        if frameCount - lastClusterFrame >= clusterRescanInterval
            || cachedProjectMaxY.isEmpty || topologyChanged {
            lastClusterFrame = frameCount
            cachedProjectMaxY.removeAll(keepingCapacity: true)
            cachedTopicSums.removeAll(keepingCapacity: true)
            for node in nodes {
                guard let pos = positions[node.id] else { continue }
                cachedProjectMaxY[node.project] = max(
                    cachedProjectMaxY[node.project] ?? -Float.greatestFiniteMagnitude, pos.y)
                let entry = cachedTopicSums[node.topic] ?? (.zero, 0, node.project, -Float.greatestFiniteMagnitude)
                cachedTopicSums[node.topic] = (entry.sum + pos, entry.count + 1, node.project, max(entry.maxY, pos.y))
            }
        }
        let projectMaxY = cachedProjectMaxY

        // Project cluster labels — large, floating above top of cluster
        // Much larger than node labels and visible from across the scene
        let projRects = scene.labelAtlasGenerator.projectRects
        if frameCount % 120 == 0 && !dataProvider.projectCentroids.isEmpty {
            print("[labels] projCentroids=\(Array(dataProvider.projectCentroids.keys)) projRects=\(Array(projRects.keys)) staging=\(labelStaging.count) instanceIdx=\(instanceIdx)")
        }
        for (project, centroid) in dataProvider.projectCentroids {
            guard let rect = projRects[project] else { continue }
            let color = colorMap[project] ?? SIMD3<Float>(0.9, 0.9, 0.9)
            let topY = projectMaxY[project] ?? centroid.y
            let anchor = SIMD3<Float>(centroid.x, topY + 40.0, centroid.z) * scaleFactor
            let halfH: Float = 14.0
            let halfW = halfH * (rect.u1 - rect.u0) / max(rect.v1 - rect.v0, 0.001) * aspectCorrection
            let dist = simd_length(anchor - camPos)
            let opacity: Float = max(0.5, 1.0 - min(1.0, dist / 8000.0))
            let r = camRight * halfW
            let u = camUp * halfH
            let baseVert = instanceIdx * Self.vertsPerLabel
            guard baseVert + 3 < labelStaging.count else { break }
            labelStaging[baseVert + 0] = BatchVertex(
                px: anchor.x - r.x - u.x, py: anchor.y - r.y - u.y, pz: anchor.z - r.z - u.z,
                nx: 0, ny: 0, nz: 1, u: rect.u0, v: rect.v0,
                cr: color.x, cg: color.y, cb: color.z, ca: opacity)
            labelStaging[baseVert + 1] = BatchVertex(
                px: anchor.x + r.x - u.x, py: anchor.y + r.y - u.y, pz: anchor.z + r.z - u.z,
                nx: 0, ny: 0, nz: 1, u: rect.u1, v: rect.v0,
                cr: color.x, cg: color.y, cb: color.z, ca: opacity)
            labelStaging[baseVert + 2] = BatchVertex(
                px: anchor.x - r.x + u.x, py: anchor.y - r.y + u.y, pz: anchor.z - r.z + u.z,
                nx: 0, ny: 0, nz: 1, u: rect.u0, v: rect.v1,
                cr: color.x, cg: color.y, cb: color.z, ca: opacity)
            labelStaging[baseVert + 3] = BatchVertex(
                px: anchor.x + r.x + u.x, py: anchor.y + r.y + u.y, pz: anchor.z + r.z + u.z,
                nx: 0, ny: 0, nz: 1, u: rect.u1, v: rect.v1,
                cr: color.x, cg: color.y, cb: color.z, ca: opacity)
            instanceIdx += 1
        }

        // Topic cluster labels — mid-size, at topic centroid
        // Larger than node labels but smaller than project labels
        let topicRects = scene.labelAtlasGenerator.topicRects
        if !topicRects.isEmpty {
            // Use cached topic centroids (recomputed above on throttle interval)
            let topicSums = cachedTopicSums
            for (topic, data) in topicSums where data.count >= 2 {
                guard let rect = topicRects[topic] else { continue }
                let centroid = data.sum / Float(data.count)
                let color = colorMap[data.project] ?? SIMD3<Float>(0.7, 0.7, 0.7)
                let anchor = SIMD3<Float>(centroid.x, data.maxY + 25.0, centroid.z) * scaleFactor
                let halfH: Float = 8.0
                let halfW = halfH * (rect.u1 - rect.u0) / max(rect.v1 - rect.v0, 0.001) * aspectCorrection
                let dist = simd_length(anchor - camPos)
                let opacity: Float = max(0.3, 1.0 - min(1.0, dist / 5000.0))
                let r = camRight * halfW
                let u = camUp * halfH
                let baseVert = instanceIdx * Self.vertsPerLabel
                guard baseVert + 3 < labelStaging.count else { break }
                labelStaging[baseVert + 0] = BatchVertex(
                    px: anchor.x - r.x - u.x, py: anchor.y - r.y - u.y, pz: anchor.z - r.z - u.z,
                    nx: 0, ny: 0, nz: 1, u: rect.u0, v: rect.v0,
                    cr: color.x, cg: color.y, cb: color.z, ca: opacity)
                labelStaging[baseVert + 1] = BatchVertex(
                    px: anchor.x + r.x - u.x, py: anchor.y + r.y - u.y, pz: anchor.z + r.z - u.z,
                    nx: 0, ny: 0, nz: 1, u: rect.u1, v: rect.v0,
                    cr: color.x, cg: color.y, cb: color.z, ca: opacity)
                labelStaging[baseVert + 2] = BatchVertex(
                    px: anchor.x - r.x + u.x, py: anchor.y - r.y + u.y, pz: anchor.z - r.z + u.z,
                    nx: 0, ny: 0, nz: 1, u: rect.u0, v: rect.v1,
                    cr: color.x, cg: color.y, cb: color.z, ca: opacity)
                labelStaging[baseVert + 3] = BatchVertex(
                    px: anchor.x + r.x + u.x, py: anchor.y + r.y + u.y, pz: anchor.z + r.z + u.z,
                    nx: 0, ny: 0, nz: 1, u: rect.u1, v: rect.v1,
                    cr: color.x, cg: color.y, cb: color.z, ca: opacity)
                instanceIdx += 1
            }
        }

        let usedVerts = instanceIdx * Self.vertsPerLabel
        if usedVerts < totalVerts {
            memset(&labelStaging[usedVerts], 0, (totalVerts - usedVerts) * MemoryLayout<BatchVertex>.stride)
        }

        // GPU-synchronized write
        guard let cmdBuf = commandBuffer ?? scene.commandQueue.makeCommandBuffer() else { return }
        let destBuffer = mesh.replace(bufferIndex: 0, using: cmdBuf)
        let dest = destBuffer.contents().bindMemory(to: BatchVertex.self, capacity: totalVerts)
        labelStaging.withUnsafeBufferPointer { src in
            dest.update(from: src.baseAddress!, count: totalVerts)
        }
        if commandBuffer == nil { cmdBuf.commit() }
    }
}
