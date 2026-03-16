import Metal
import CEngramSceneTypes
import simd
import SwiftUI
import EngramSceneKit

/// Packs edge vertex data each frame.
/// Extracted from MetalSceneManager to keep that file focused on orchestration.
@MainActor
final class EdgePackingStage {

    // GPU edge packing state
    private var cachedResolvedEdges: [ResolvedEdge] = []
    private var lastEdgeDescriptorTopologyCount: Int = 0
    private var lastEdgeDescriptorSelection: UUID? = nil
    private var lastEdgeDescriptorSearchActive: Bool = false
    private var lastEdgeDescriptorSearchMatchIds: Set<UUID> = []
    private var edgeDescriptorsDirty: Bool = true
    var nodePositionsDirty: Bool = true

    /// Mark positions as dirty — called at the start of each frame by MetalSceneManager.
    func markPositionsDirty() {
        nodePositionsDirty = true
    }

    /// Uploads node positions to the GPU position buffer.
    /// Falls back to dict iteration if called standalone (e.g., edge-only update).
    func uploadNodePositions(
        nodeIndexMap: [UUID: UInt32],
        positions: [UUID: SIMD3<Float>],
        bufferManager: InstanceBufferManager
    ) {
        // If nodePositionsDirty is false, packNodeInstances already uploaded this frame
        guard nodePositionsDirty else { return }
        let nodeCount = nodeIndexMap.count
        guard nodeCount > 0 else { return }
        bufferManager.ensureNodePositionBuffer(count: nodeCount)
        guard let posBuf = bufferManager.nodePositionBuffer else { return }

        let posPtr = posBuf.contents().bindMemory(to: SIMD3<Float>.self, capacity: nodeCount)
        for (id, idx) in nodeIndexMap {
            posPtr[Int(idx)] = positions[id] ?? .zero
        }
        nodePositionsDirty = false
        bufferManager.edgeDataDirty = true
    }

    /// Rebuilds edge descriptors when topology, selection, or search state changes.
    private func rebuildEdgeDescriptors(
        nodeIndexMap: [UUID: UInt32],
        selectedNode: UUID?,
        isSearchActive: Bool,
        searchMatchIds: Set<UUID>,
        layoutMode: LayoutMode,
        edgeRadius: Float,
        scaleFactor: Float,
        bufferManager: InstanceBufferManager,
        galaxyRegistry: GalaxyRegistry?,
        renderNodeCount: Int
    ) {
        let interGalaxyConns = galaxyRegistry?.interGalaxyConnections ?? []
        let totalEdgeCapacity = cachedResolvedEdges.count + interGalaxyConns.count
        guard totalEdgeCapacity > 0 else {
            bufferManager.actualEdgeCount = 0
            return
        }

        // Build per-frame input (lightweight — no UUID lookups)
        let selectedIdx = selectedNode.flatMap { nodeIndexMap[$0] }
        var dimmedIndices = Set<UInt32>()
        if isSearchActive {
            for (id, idx) in nodeIndexMap where !searchMatchIds.contains(id) {
                dimmedIndices.insert(idx)
            }
        }

        let edgeInput = SceneEdgeFrameInput(
            resolvedEdges: cachedResolvedEdges,
            selectedIdx: selectedIdx,
            isSearchActive: isSearchActive,
            searchDimmedSourceIdx: dimmedIndices,
            isSemanticMode: layoutMode == .embedding,
            edgeRadius: edgeRadius,
            nodeCount: nodeIndexMap.count,
            interGalaxyConnections: interGalaxyConns.map { (from: $0.from, to: $0.to) }
        )
        let edgeFrame = buildSceneEdgeFrame(edgeInput)

        bufferManager.ensureEdgeBuffers(edgeCount: totalEdgeCapacity)
        bufferManager.ensureEdgeDescriptorBuffer(count: totalEdgeCapacity)
        guard let descBuf = bufferManager.edgeDescriptorBuffer else { return }

        let descriptors = descBuf.contents().bindMemory(to: EdgeDescriptor.self, capacity: totalEdgeCapacity)
        edgeFrame.descriptors.withUnsafeBufferPointer { ptr in
            guard let base = ptr.baseAddress else { return }
            descriptors.update(from: base, count: edgeFrame.count)
        }

        // Upload inter-galaxy virtual positions
        if !edgeFrame.interGalaxyPositions.isEmpty {
            let interGalaxyBase = Int(nodeIndexMap.count)
            let totalPositions = interGalaxyBase + edgeFrame.interGalaxyPositions.count
            bufferManager.ensureNodePositionBuffer(count: totalPositions)
            if let posBuf = bufferManager.nodePositionBuffer {
                let posPtr = posBuf.contents().bindMemory(to: SIMD3<Float>.self, capacity: totalPositions)
                for (i, pos) in edgeFrame.interGalaxyPositions.enumerated() {
                    posPtr[interGalaxyBase + i] = pos
                }
            }
        }

        bufferManager.actualEdgeCount = edgeFrame.count

        if let paramsBuf = bufferManager.packEdgeParamsBuffer {
            let paramsPtr = paramsBuf.contents().bindMemory(to: PackEdgeParams.self, capacity: 1)
            paramsPtr.pointee = PackEdgeParams(
                edgeCount: UInt32(edgeFrame.count),
                scaleFactor: scaleFactor,
                _pad0: 0, _pad1: 0
            )
        }

        edgeDescriptorsDirty = false
        bufferManager.edgeDataDirty = true
        lastEdgeDescriptorSelection = selectedNode
        lastEdgeDescriptorSearchActive = isSearchActive
        lastEdgeDescriptorSearchMatchIds = searchMatchIds
        lastEdgeDescriptorTopologyCount = renderNodeCount
    }

    /// Updates edge GPU data: uploads positions every frame, rebuilds descriptors only when needed.
    func pack(
        edges: [EdgeData],
        nodes: [NodeData],
        positions: [UUID: SIMD3<Float>],
        nodeIndexMap: [UUID: UInt32],
        selectedNode: UUID?,
        expandedHubs: Set<UUID>,
        expandedChildPositions: [UUID: SIMD3<Float>],
        hubs: Set<UUID>,
        searchMatchIds: Set<UUID>,
        isSearchActive: Bool,
        renderColorMap: [String: Color],
        edgeColorCache: inout [String: SIMD3<Float>],
        nodeColorCache: inout [String: SIMD3<Float>],
        bufferManager: InstanceBufferManager,
        scaleFactor: Float,
        edgeRadius: Float,
        nodeRadius: Float,
        layoutMode: LayoutMode,
        galaxyRegistry: GalaxyRegistry?,
        cachedNodeRadii: [UUID: Float],
        cachedNodeProject: [UUID: String]
    ) {
        let interGalaxyConns = galaxyRegistry?.interGalaxyConnections ?? []
        guard edges.count + interGalaxyConns.count > 0 else {
            bufferManager.actualEdgeCount = 0
            return
        }

        // Re-resolve edges on topology change (one-time UUID work)
        let currentNodeCount = nodes.count
        if currentNodeCount != lastEdgeDescriptorTopologyCount || edgeDescriptorsDirty {
            let colorMap = renderColorMap
            var projectColors: [String: SIMD3<Float>] = [:]
            for project in cachedNodeProject.values {
                if projectColors[project] == nil {
                    projectColors[project] = ColorHelpers.nodeColorFloat3(for: project, colorMap: colorMap, cache: &nodeColorCache)
                }
            }
            cachedResolvedEdges = resolveEdges(
                edges: edges.map { (sourceId: $0.sourceId, targetId: $0.targetId, relation: $0.relation) },
                nodeIndexMap: nodeIndexMap,
                nodeRadii: cachedNodeRadii,
                nodeProjects: cachedNodeProject,
                projectColors: projectColors,
                defaultEdgeRadius: edgeRadius
            )
        }

        // Check if descriptors need rebuilding (per-frame state changes)
        let descriptorsNeedRebuild = edgeDescriptorsDirty
            || currentNodeCount != lastEdgeDescriptorTopologyCount
            || selectedNode != lastEdgeDescriptorSelection
            || isSearchActive != lastEdgeDescriptorSearchActive
            || searchMatchIds != lastEdgeDescriptorSearchMatchIds

        if descriptorsNeedRebuild {
            rebuildEdgeDescriptors(
                nodeIndexMap: nodeIndexMap,
                selectedNode: selectedNode,
                isSearchActive: isSearchActive,
                searchMatchIds: searchMatchIds,
                layoutMode: layoutMode,
                edgeRadius: edgeRadius,
                scaleFactor: scaleFactor,
                bufferManager: bufferManager,
                galaxyRegistry: galaxyRegistry,
                renderNodeCount: currentNodeCount
            )
        }

        // Upload positions every frame (O(nodes) contiguous write — fast)
        uploadNodePositions(
            nodeIndexMap: nodeIndexMap,
            positions: positions,
            bufferManager: bufferManager
        )
    }
}
