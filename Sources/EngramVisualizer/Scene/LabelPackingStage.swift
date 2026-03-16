import Metal
import CEngramSceneTypes
import simd
import SwiftUI
import EngramSceneKit

/// Packs label instance data each frame.
/// Extracted from MetalSceneManager to keep that file focused on orchestration.
@MainActor
final class LabelPackingStage {

    #if ENGRAM_INSTRUMENTATION
    var labelDiagFile: UnsafeMutablePointer<FILE>? = nil
    #endif

    func pack(
        positions: [UUID: SIMD3<Float>],
        nodes: [NodeData],
        hubs: Set<UUID>,
        edges: [EdgeData],
        selectedNode: UUID?,
        camera: CameraController,
        hubExpansion: HubExpansionController,
        labelAtlas: LabelAtlasGenerator,
        projectCentroids: [String: (centroid: SIMD3<Float>, radius: Float, maxY: Float, count: Int)],
        cachedMinDepth: Float,
        cachedMaxDepth: Float,
        renderColorMap: [String: Color],
        topicGroups: [TopicGroupInfo],
        isSearchActive: Bool,
        searchMatchIds: Set<UUID>,
        bufferManager: InstanceBufferManager,
        galaxyRegistry: GalaxyRegistry?,
        scaleFactor: Float,
        nodeRadius: Float,
        renderFrameCount: UInt64,
        nodeColorFloat3: (String, [String: Color]) -> SIMD3<Float>
    ) {
        #if ENGRAM_INSTRUMENTATION
        if labelDiagFile == nil {
            labelDiagFile = fopen("/tmp/label-diag.csv", "w")
            if let f = labelDiagFile {
                fputs("frame,positions,atlasRects,missingRects,instances,atlasRegen,depthRange,minDepth,maxDepth,camX,camY,camZ,projLabels,topicLabels,galaxyLabels,pendingRegen,isGenerating\n", f)
            }
        }
        var diagAtlasRegen = false
        var diagMissingRects = 0
        #endif

        let nodeCount = positions.count
        guard nodeCount > 0 else {
            bufferManager.actualLabelCount = 0
            return
        }

        let storeVersion = galaxyRegistry?.mergedTopologyVersion ?? 0

        // Regen atlas if needed — O(1) version checks.
        // On topology change, set pendingAtlasRegen and dispatch on the NEXT frame
        // to avoid stacking atlas dispatch overhead on the same frame as the insert.
        let currentTopicGroupCount = topicGroups.count
        let currentTopicLabels = Array(Set(topicGroups.map(\.topic))).sorted()
        let atlasNeedsRegen = bufferManager.labelAtlasTexture == nil
            || storeVersion != labelAtlas.lastAtlasTopologyVersion
            || currentTopicGroupCount != labelAtlas.lastAtlasTopicGroupCount
        if atlasNeedsRegen {
            let isFirstAtlas = bufferManager.labelAtlasTexture == nil
            if isFirstAtlas {
                // First atlas must be synchronous — nothing to show until it's ready
                let currentProjects = Set(nodes.map(\.project))
                let currentGalaxyNames = galaxyRegistry?.galaxies.values.map(\.displayName) ?? []
                labelAtlas.generateLabelAtlas(nodes: nodes, hubs: hubs, projects: currentProjects,
                                              galaxyNames: currentGalaxyNames, topicLabels: currentTopicLabels,
                                              bufferManager: bufferManager)
                labelAtlas.labelAtlasRegenFrame = renderFrameCount
                #if ENGRAM_INSTRUMENTATION
                diagAtlasRegen = true
                #endif
            } else {
                // Defer dispatch to next frame — just record that regen is needed
                labelAtlas.pendingAtlasRegen = true
            }
            labelAtlas.lastAtlasTopologyVersion = storeVersion
            labelAtlas.lastAtlasTopicGroupCount = currentTopicGroupCount
        } else if labelAtlas.pendingAtlasRegen && !labelAtlas.isAtlasGenerating {
            // Dispatch the deferred atlas regen (runs on the frame after topology change)
            let framesSinceRegen = renderFrameCount &- labelAtlas.labelAtlasRegenFrame
            if framesSinceRegen >= 60 {
                let currentProjects = Set(nodes.map(\.project))
                let currentGalaxyNames = galaxyRegistry?.galaxies.values.map(\.displayName) ?? []
                labelAtlas.dispatchAtlasRegen(nodes: nodes, hubs: hubs, projects: currentProjects,
                                              galaxyNames: currentGalaxyNames, topicLabels: currentTopicLabels,
                                              bufferManager: bufferManager,
                                              renderFrameCount: renderFrameCount)
                labelAtlas.labelAtlasRegenFrame = renderFrameCount
                labelAtlas.pendingAtlasRegen = false
                #if ENGRAM_INSTRUMENTATION
                diagAtlasRegen = true
                #endif
            }
            // If throttle not met, keep pendingAtlasRegen = true for the next frame
        }
        guard bufferManager.labelAtlasTexture != nil else { return }

        // Build label frame using pure builder
        let sf = scaleFactor

        // projectCentroids already computed by packNodeInstances — just build color lookup
        var projectColors: [String: SIMD3<Float>] = [:]
        for project in projectCentroids.keys {
            projectColors[project] = nodeColorFloat3(project, renderColorMap)
        }

        let nodeById = galaxyRegistry?.mergedNodeById ?? [:]

        let hasExpansions = !hubExpansion.expandedChildPositions.isEmpty
        let allPositions: [UUID: SIMD3<Float>]
        if hasExpansions {
            var merged = positions
            for (id, pos) in hubExpansion.expandedChildPositions { merged[id] = pos }
            allPositions = merged
        } else {
            allPositions = positions
        }
        var expandedChildren = Set<UUID>()
        if hasExpansions {
            let edgeTuples = edges.map { (sourceId: $0.sourceId, targetId: $0.targetId, relation: $0.relation) }
            for hubId in hubExpansion.expandedHubs {
                for childId in hubExpansion.childrenOfHub(hubId, edges: edgeTuples) {
                    expandedChildren.insert(childId)
                }
            }
        }

        // Build node label entries from allPositions
        var nodeLabels: [(id: UUID, position: SIMD3<Float>, isHub: Bool,
                          importance: Int, isSelected: Bool,
                          isSearchMatch: Bool, searchDimmed: Bool,
                          isExpandedChild: Bool)] = []
        nodeLabels.reserveCapacity(allPositions.count)
        #if ENGRAM_INSTRUMENTATION
        var diagMissingRects = 0
        #endif
        for (id, pos3D) in allPositions {
            guard let nodeData = nodeById[id] else { continue }
            #if ENGRAM_INSTRUMENTATION
            if labelAtlas.labelAtlasRects[id] == nil { diagMissingRects += 1 }
            #endif
            nodeLabels.append((
                id: id, position: pos3D,
                isHub: hubs.contains(id),
                importance: nodeData.importance,
                isSelected: id == selectedNode,
                isSearchMatch: isSearchActive && searchMatchIds.contains(id),
                searchDimmed: isSearchActive && !searchMatchIds.contains(id),
                isExpandedChild: expandedChildren.contains(id)
            ))
        }

        // Convert atlas rects
        var atlasRects: [UUID: AtlasRect] = [:]
        for (id, rect) in labelAtlas.labelAtlasRects {
            atlasRects[id] = AtlasRect(u0: rect.u0, v0: rect.v0, u1: rect.u1, v1: rect.v1)
        }
        var projLabelRects: [String: AtlasRect] = [:]
        for (k, rect) in labelAtlas.projectLabelAtlasRects {
            projLabelRects[k] = AtlasRect(u0: rect.u0, v0: rect.v0, u1: rect.u1, v1: rect.v1)
        }
        var galLabelRects: [String: AtlasRect] = [:]
        for (k, rect) in labelAtlas.galaxyLabelAtlasRects {
            galLabelRects[k] = AtlasRect(u0: rect.u0, v0: rect.v0, u1: rect.u1, v1: rect.v1)
        }
        var topLabelRects: [String: AtlasRect] = [:]
        for (k, rect) in labelAtlas.topicLabelAtlasRects {
            topLabelRects[k] = AtlasRect(u0: rect.u0, v0: rect.v0, u1: rect.u1, v1: rect.v1)
        }

        // Build galaxy labels
        var galaxyLabels: [(name: String, center: SIMD3<Float>)] = []
        if let registry = galaxyRegistry, registry.galaxies.count > 1 {
            for galaxy in registry.galaxies.values {
                galaxyLabels.append((name: galaxy.displayName, center: galaxy.worldCenter))
            }
        }

        // Build topic groups with member positions
        var topicGroupsInput: [(topic: String, project: String, memberPositions: [SIMD3<Float>])] = []
        for group in topicGroups {
            guard group.ids.count >= 3 else { continue }
            let memberPositions = group.ids.compactMap { allPositions[$0] }
            guard memberPositions.count >= 3 else { continue }
            topicGroupsInput.append((topic: group.topic, project: group.project, memberPositions: memberPositions))
        }

        // Build centroids in the format the label builder expects
        var labelCentroids: [String: (centroid: SIMD3<Float>, maxY: Float, count: Int)] = [:]
        for (project, data) in projectCentroids {
            labelCentroids[project] = (centroid: data.centroid, maxY: data.maxY, count: data.count)
        }

        let labelInput = SceneLabelFrameInput(
            nodeLabels: nodeLabels,
            atlasRects: atlasRects,
            projectCentroids: labelCentroids,
            projectLabelRects: projLabelRects,
            projectColors: projectColors,
            galaxyLabels: galaxyLabels,
            galaxyLabelRects: galLabelRects,
            topicGroups: topicGroupsInput,
            topicLabelRects: topLabelRects,
            scaleFactor: sf,
            nodeRadius: nodeRadius,
            aspectCorrection: labelAtlas.labelAtlasAspectCorrection
        )
        let labelFrame = buildSceneLabelFrame(labelInput)
        let instances = labelFrame.instances
        let actualLabelCount = instances.count

        bufferManager.ensureLabelBuffers(labelCount: actualLabelCount)
        if let instanceBuf = bufferManager.labelInstanceBuffer {
            instances.withUnsafeBufferPointer { ptr in
                guard let base = ptr.baseAddress else { return }
                instanceBuf.contents().copyMemory(
                    from: base,
                    byteCount: actualLabelCount * MemoryLayout<LabelInstance>.stride
                )
            }
        }

        // Set stamp params
        let depthRange = max(50.0, cachedMaxDepth - cachedMinDepth)
        let scaledCamPos = camera.cameraPosition * sf
        if let paramsBuf = bufferManager.labelStampParamsBuffer {
            let paramsPtr = paramsBuf.contents().bindMemory(to: LabelStampParams.self, capacity: 1)
            paramsPtr.pointee = LabelStampParams(
                cameraPos: scaledCamPos,
                minDepth: cachedMinDepth * sf,
                depthRange: depthRange * sf,
                labelCount: UInt32(actualLabelCount),
                _pad0: 0, _pad1: 0
            )
        }

        bufferManager.actualLabelCount = actualLabelCount

        #if ENGRAM_INSTRUMENTATION
        if let f = labelDiagFile {
            let line = String(format: "%llu,%d,%d,%d,%d,%d,%.2f,%.2f,%.2f,%.3f,%.3f,%.3f,%d,%d,%d,%d,%d\n",
                renderFrameCount,
                allPositions.count,
                labelAtlas.labelAtlasRects.count,
                diagMissingRects,
                actualLabelCount,
                diagAtlasRegen ? 1 : 0,
                depthRange, cachedMinDepth, cachedMaxDepth,
                camera.cameraPosition.x, camera.cameraPosition.y, camera.cameraPosition.z,
                projectCentroids.count,
                topicGroups.count,
                (galaxyRegistry?.galaxies.count ?? 0) > 1 ? galaxyRegistry!.galaxies.count : 0,
                labelAtlas.pendingAtlasRegen ? 1 : 0,
                labelAtlas.isAtlasGenerating ? 1 : 0)
            fputs(line, f)
            if renderFrameCount % 10 == 0 { fflush(f) }
        }
        #endif
    }
}
