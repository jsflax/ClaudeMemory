import Metal
import CEngramSceneTypes
import simd
import SwiftUI
import EngramSceneKit

/// Result of node packing — consumed by MetalSceneManager to feed edge packing, labels, and nebulae.
struct NodePackResult {
    var nodeIndexMap: [UUID: UInt32]
    var projectCentroids: [String: (centroid: SIMD3<Float>, radius: Float, maxY: Float, count: Int)]
    var minDepth: Float
    var maxDepth: Float
    /// Cached per-node radii — needed by EdgePackingStage for edge resolution.
    var cachedNodeRadii: [UUID: Float]
    /// Cached per-node project — needed by EdgePackingStage for edge resolution.
    var cachedNodeProject: [UUID: String]
}

/// Packs node instance data each frame.
/// Extracted from MetalSceneManager to keep that file focused on orchestration.
@MainActor
final class NodePackingStage {

    // Cached per-node data rebuilt only on topology change
    private var nodeIndexMap: [UUID: UInt32] = [:]
    private var cachedNodeRadii: [UUID: Float] = [:]
    private var cachedNodeProject: [UUID: String] = [:]
    private var lastTopologyNodeCount: Int = 0

    // Node instance staging
    private var instanceArray: [NodeInstance] = []

    // Cached depth range from last pack
    private var cachedMinDepth: Float = .greatestFiniteMagnitude
    private var cachedMaxDepth: Float = 0

    func pack(
        nodes: [NodeData],
        positions: [UUID: SIMD3<Float>],
        selectedNode: UUID?,
        expandedHubs: Set<UUID>,
        expandedChildPositions: [UUID: SIMD3<Float>],
        glowingNodes: [UUID: Date],
        newNodes: [UUID: Date],
        dyingNodes: [UUID: DyingNode],
        searchMatchIds: Set<UUID>,
        isSearchActive: Bool,
        renderColorMap: [String: Color],
        topicGroups: [TopicGroupInfo],
        nodeColorCache: inout [String: SIMD3<Float>],
        edgeColorCache: inout [String: SIMD3<Float>],
        bufferManager: InstanceBufferManager,
        lightingUniforms: inout LightingUniforms,
        animationTime: Float,
        scaleFactor: Float,
        nodeRadius: Float,
        cameraPosition: SIMD3<Float>,
        hubs: Set<UUID>,
        galaxyRegistry: GalaxyRegistry?,
        renderFrameCount: UInt64
    ) -> NodePackResult {
        guard !nodes.isEmpty else {
            bufferManager.actualNodeCount = 0
            GPULog.log("PACK NODES: empty")
            return NodePackResult(
                nodeIndexMap: nodeIndexMap,
                projectCentroids: [:],
                minDepth: cachedMinDepth,
                maxDepth: cachedMaxDepth,
                cachedNodeRadii: cachedNodeRadii,
                cachedNodeProject: cachedNodeProject
            )
        }

        #if ENGRAM_INSTRUMENTATION
        let packStart = CFAbsoluteTimeGetCurrent()
        #endif

        let colorMap = renderColorMap
        let now = Date()

        // Pre-compute inspecting nodes and birthing elapsed times across all galaxies
        var inspectingIntensity: [UUID: Float] = [:]
        var birthingElapsed: [UUID: Float] = [:]
        var birthingGalaxyMap: [UUID: String] = [:]
        if let registry = galaxyRegistry {
            for galaxy in registry.galaxies.values {
                if let fleet = galaxy.mascotFleet {
                    for mascot in fleet.mascots.values where mascot.arcaneIntensity > 0.01 {
                        if let targetId = mascot.currentTargetId {
                            inspectingIntensity[targetId] = mascot.arcaneIntensity
                        }
                    }
                }
                for (nodeId, start) in galaxy.renderStore.birthingNodes {
                    birthingElapsed[nodeId] = Float(now.timeIntervalSince(start))
                    birthingGalaxyMap[nodeId] = galaxy.id
                }
            }
        }

        // Convert glow/arrival Dates → elapsed Floats for the pure builder
        var glowElapsed: [UUID: Float] = [:]
        for (id, date) in glowingNodes { glowElapsed[id] = Float(now.timeIntervalSince(date)) }
        var arrivalElapsed: [UUID: Float] = [:]
        for (id, date) in newNodes { arrivalElapsed[id] = Float(now.timeIntervalSince(date)) }

        // Pre-convert project colors to SIMD3<Float>
        var projectColors: [String: SIMD3<Float>] = [:]
        for (project, _) in colorMap {
            projectColors[project] = ColorHelpers.nodeColorFloat3(for: project, colorMap: colorMap, cache: &nodeColorCache)
        }

        #if ENGRAM_INSTRUMENTATION
        let packPrecomputeMs = (CFAbsoluteTimeGetCurrent() - packStart) * 1000.0
        let packLoopStart = CFAbsoluteTimeGetCurrent()
        #endif

        // Build the NodeFrame using the pure builder (testable, no GPU dependency)
        let t_input = CFAbsoluteTimeGetCurrent()
        let nodeCount = nodes.count

        // Rebuild cached data only on topology change
        if nodeCount != lastTopologyNodeCount {
            nodeIndexMap.removeAll(keepingCapacity: true)
            cachedNodeProject.removeAll(keepingCapacity: true)
            cachedNodeRadii.removeAll(keepingCapacity: true)
            for (i, node) in nodes.enumerated() {
                nodeIndexMap[node.id] = UInt32(i)
                cachedNodeProject[node.id] = node.project
                let isHub = hubs.contains(node.id)
                let importance = max(1, node.importance)
                let baseR: Float = isHub ? nodeRadius * 1.6 : nodeRadius
                cachedNodeRadii[node.id] = baseR * (1.0 + Float(importance - 1) * 0.08)
            }
            lastTopologyNodeCount = nodeCount
        }

        // Build parallel arrays for the builder (no UUID dict lookups in hot loop)
        var nodeIds = [UUID](repeating: UUID(), count: nodeCount)
        var nodeProj = [String](repeating: "", count: nodeCount)
        var nodeImp = [Int](repeating: 1, count: nodeCount)
        var nodePos = [SIMD3<Float>](repeating: .zero, count: nodeCount)
        var hasPos = [Bool](repeating: false, count: nodeCount)
        var nodeRad = [Float](repeating: nodeRadius, count: nodeCount)
        var nodeHub = [Bool](repeating: false, count: nodeCount)
        var selectedIdx: Int? = nil
        var searchMatchIdx = Set<Int>()

        for (i, node) in nodes.enumerated() {
            nodeIds[i] = node.id
            nodeProj[i] = node.project
            nodeImp[i] = node.importance
            if let pos = positions[node.id] {
                nodePos[i] = pos
                hasPos[i] = true
            }
            nodeRad[i] = cachedNodeRadii[node.id] ?? nodeRadius
            nodeHub[i] = hubs.contains(node.id)
            if node.id == selectedNode { selectedIdx = i }
            if isSearchActive && searchMatchIds.contains(node.id) { searchMatchIdx.insert(i) }
        }

        let frameInput = SceneNodeFrameInput(
            nodeIds: nodeIds,
            nodeProjects: nodeProj,
            nodeImportances: nodeImp,
            nodePositions: nodePos,
            hasPosition: hasPos,
            nodeRadii: nodeRad,
            nodeIsHub: nodeHub,
            projectColors: projectColors,
            selectedNodeIndex: selectedIdx,
            glowingNodes: glowElapsed,
            newNodes: arrivalElapsed,
            isSearchActive: isSearchActive,
            searchMatchIndices: searchMatchIdx,
            inspectingIntensity: inspectingIntensity,
            birthingElapsed: birthingElapsed,
            nodeRadius: nodeRadius,
            cameraPosition: cameraPosition
        )
        let t_build = CFAbsoluteTimeGetCurrent()
        let nodeFrame = buildSceneNodeFrame(frameInput)
        let t_built = CFAbsoluteTimeGetCurrent()
        let actualCount = nodeFrame.packInputs.count

        // Apply side effect: remove completed births from render stores
        for nodeId in nodeFrame.completedBirths {
            if let galaxyId = birthingGalaxyMap[nodeId] {
                galaxyRegistry?.galaxies[galaxyId]?.renderStore.birthingNodes.removeValue(forKey: nodeId)
            }
        }

        // Upload NodeFrame to GPU buffers
        let t_upload = CFAbsoluteTimeGetCurrent()
        bufferManager.ensureNodeBuffers(nodeCount: nodeCount)
        bufferManager.ensureNodePackBuffers(count: nodeCount)
        bufferManager.ensureNodePositionBuffer(count: nodeCount)

        guard let packBuf = bufferManager.nodePackInputBuffer,
              let posBuf = bufferManager.nodePositionBuffer else {
            bufferManager.actualNodeCount = 0
            return NodePackResult(
                nodeIndexMap: nodeIndexMap,
                projectCentroids: [:],
                minDepth: cachedMinDepth,
                maxDepth: cachedMaxDepth,
                cachedNodeRadii: cachedNodeRadii,
                cachedNodeProject: cachedNodeProject
            )
        }

        let packPtr = packBuf.contents().bindMemory(to: NodePackInput.self, capacity: nodeCount)
        let posPtr = posBuf.contents().bindMemory(to: SIMD3<Float>.self, capacity: nodeCount)

        nodeFrame.packInputs.withUnsafeBufferPointer { ptr in
            guard let base = ptr.baseAddress else { return }
            packPtr.update(from: base, count: actualCount)
        }
        nodeFrame.nodePositions.withUnsafeBufferPointer { ptr in
            guard let base = ptr.baseAddress else { return }
            posPtr.update(from: base, count: actualCount)
        }

        // Finalize centroid + depth caches
        var projectCentroids: [String: (centroid: SIMD3<Float>, radius: Float, maxY: Float, count: Int)] = [:]
        for (project, data) in nodeFrame.projectCentroids {
            projectCentroids[project] = (centroid: data.centroid, radius: 0, maxY: data.maxY, count: data.count)
        }
        cachedMinDepth = nodeFrame.minDepth
        cachedMaxDepth = nodeFrame.maxDepth

        // Set GPU pack params — the pack_node_instances kernel runs in the compute pass
        if let paramsBuf = bufferManager.nodePackParamsBuffer {
            let paramsPtr = paramsBuf.contents().bindMemory(to: NodePackParams.self, capacity: 1)
            paramsPtr.pointee = NodePackParams(
                nodeCount: UInt32(actualCount),
                scaleFactor: scaleFactor,
                nodeRadius: nodeRadius,
                animationTime: animationTime,
                projectCount: UInt32(projectCentroids.count),
                _pad0: 0, _pad1: 0, _pad2: 0
            )
        }

        bufferManager.actualNodeCount = actualCount

        // Read back point lights from GPU after the compute pass completes (1-frame latency).
        if let lightCountBuf = bufferManager.pointLightCountBuffer,
           let lightBuf = bufferManager.pointLightOutputBuffer {
            let count = min(Int(lightCountBuf.contents().bindMemory(to: UInt32.self, capacity: 1).pointee), 16)
            let lights = lightBuf.contents().bindMemory(to: PointLightEntry.self, capacity: count)
            for i in 0..<count {
                setPointLight(index: i, position: lights[i].position, color: lights[i].color,
                              intensity: lights[i].intensity, attenuation: lights[i].attenuation,
                              lightingUniforms: &lightingUniforms)
            }
            lightingUniforms.pointLightCount = UInt32(count)
        }

        if !nodeIndexMap.isEmpty {
            bufferManager.edgeDataDirty = true
        }

        let t_done = CFAbsoluteTimeGetCurrent()
        let packTotalCheck = (t_done - t_input) * 1000.0
        if packTotalCheck > 10 {
            print("[engram:pack] n=\(actualCount) input=\(String(format: "%.1f", (t_build - t_input) * 1000))ms build=\(String(format: "%.1f", (t_built - t_build) * 1000))ms upload=\(String(format: "%.1f", (t_done - t_upload) * 1000))ms total=\(String(format: "%.1f", packTotalCheck))ms")
        }

        #if ENGRAM_INSTRUMENTATION
        let packLoopMs = (CFAbsoluteTimeGetCurrent() - packLoopStart) * 1000.0
        let packTotalMs = (CFAbsoluteTimeGetCurrent() - packStart) * 1000.0
        do {
            struct PackTiming { nonisolated(unsafe) static var file: UnsafeMutablePointer<FILE>? = nil }
            if PackTiming.file == nil {
                PackTiming.file = fopen("/tmp/pack-timing.csv", "w")
                if let f = PackTiming.file {
                    fputs("frame,pack_precompute_ms,pack_loop_ms,pack_total_ms,node_count\n", f)
                }
            }
            if let f = PackTiming.file {
                let line = "\(renderFrameCount),\(String(format: "%.2f", packPrecomputeMs)),\(String(format: "%.2f", packLoopMs)),\(String(format: "%.2f", packTotalMs)),\(actualCount)\n"
                fputs(line, f)
                fflush(f)
            }
        }
        #endif

        return NodePackResult(
            nodeIndexMap: nodeIndexMap,
            projectCentroids: projectCentroids,
            minDepth: cachedMinDepth,
            maxDepth: cachedMaxDepth,
            cachedNodeRadii: cachedNodeRadii,
            cachedNodeProject: cachedNodeProject
        )
    }

    private func setPointLight(index: Int, position: SIMD3<Float>, color: SIMD3<Float>, intensity: Float, attenuation: Float, lightingUniforms: inout LightingUniforms) {
        let light = PointLightData(
            position: position,
            intensity: intensity,
            color: color,
            attenuationRadius: attenuation
        )
        withUnsafeMutablePointer(to: &lightingUniforms.pointLights) { tuple in
            let ptr = UnsafeMutableRawPointer(tuple).bindMemory(to: PointLightData.self, capacity: 16)
            ptr[index] = light
        }
    }
}
