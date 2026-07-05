// EdgeFrameBuilder — pure function converting edge data → GPU-ready EdgeDescriptor[].

import Foundation
import simd
import CEngramSceneTypes

// MARK: - Input / Output Types

/// Pre-resolved edge using integer indices. Built once on topology change,
/// not per-frame. Eliminates UUID dictionary lookups from the hot loop.
public struct ResolvedEdge: Sendable {
    public let sourceIdx: UInt32
    public let targetIdx: UInt32
    public let sourceRadius: Float
    public let targetRadius: Float
    public let color: SIMD3<Float>      // pre-computed edge color (0.6x project color)
    public let isConnectedToSelected: Bool  // set false; rechecked per-frame via selectedIdx

    @inlinable
    public init(sourceIdx: UInt32, targetIdx: UInt32, sourceRadius: Float, targetRadius: Float,
                color: SIMD3<Float>, isConnectedToSelected: Bool = false) {
        self.sourceIdx = sourceIdx
        self.targetIdx = targetIdx
        self.sourceRadius = sourceRadius
        self.targetRadius = targetRadius
        self.color = color
        self.isConnectedToSelected = isConnectedToSelected
    }
}

/// Resolve edges from UUID-based to index-based. Call once on topology change.
public func resolveEdges(
    edges: [(sourceId: UUID, targetId: UUID, relation: String)],
    nodeIndexMap: [UUID: UInt32],
    nodeRadii: [UUID: Float],
    nodeProjects: [UUID: String],
    projectColors: [String: SIMD3<Float>],
    defaultEdgeRadius: Float
) -> [ResolvedEdge] {
    let defaultColor = SIMD3<Float>(0.35, 0.35, 0.4)
    var resolved = [ResolvedEdge]()
    resolved.reserveCapacity(edges.count)

    for edge in edges {
        guard let srcIdx = nodeIndexMap[edge.sourceId],
              let tgtIdx = nodeIndexMap[edge.targetId] else { continue }

        let r1 = nodeRadii[edge.sourceId] ?? defaultEdgeRadius
        let r2 = nodeRadii[edge.targetId] ?? defaultEdgeRadius

        let color: SIMD3<Float>
        if let project = nodeProjects[edge.sourceId],
           let projColor = projectColors[project] {
            color = projColor * 0.6  // inline edge dimming
        } else {
            color = defaultColor
        }

        resolved.append(ResolvedEdge(
            sourceIdx: srcIdx, targetIdx: tgtIdx,
            sourceRadius: r1, targetRadius: r2,
            color: color
        ))
    }
    return resolved
}

/// Lightweight per-frame input — only things that change frame-to-frame.
public struct SceneEdgeFrameInput: Sendable {
    public let resolvedEdges: [ResolvedEdge]
    public let selectedIdx: UInt32?          // index of selected node (not UUID)
    public let isSearchActive: Bool
    public let searchDimmedSourceIdx: Set<UInt32>  // indices of non-matching nodes
    public let isSemanticMode: Bool
    public let edgeRadius: Float
    public let nodeCount: Int                // for inter-galaxy base offset
    public let interGalaxyConnections: [(from: SIMD3<Float>, to: SIMD3<Float>)]

    public init(
        resolvedEdges: [ResolvedEdge],
        selectedIdx: UInt32? = nil,
        isSearchActive: Bool = false,
        searchDimmedSourceIdx: Set<UInt32> = [],
        isSemanticMode: Bool = false,
        edgeRadius: Float = 0.004,
        nodeCount: Int = 0,
        interGalaxyConnections: [(from: SIMD3<Float>, to: SIMD3<Float>)] = []
    ) {
        self.resolvedEdges = resolvedEdges
        self.selectedIdx = selectedIdx
        self.isSearchActive = isSearchActive
        self.searchDimmedSourceIdx = searchDimmedSourceIdx
        self.isSemanticMode = isSemanticMode
        self.edgeRadius = edgeRadius
        self.nodeCount = nodeCount
        self.interGalaxyConnections = interGalaxyConnections
    }
}

public struct SceneEdgeFrame: Sendable {
    public var descriptors: [EdgeDescriptor]
    public var interGalaxyPositions: [SIMD3<Float>]
    public var count: Int

    public init() {
        self.descriptors = []
        self.interGalaxyPositions = []
        self.count = 0
    }
}

// MARK: - Builder

/// Per-frame edge packing. The hot loop uses only integer-indexed data —
/// zero UUID dictionary lookups.
@inlinable
public func buildSceneEdgeFrame(_ input: SceneEdgeFrameInput) -> SceneEdgeFrame {
    var frame = SceneEdgeFrame()
    let edgeCount = input.resolvedEdges.count
    let totalCapacity = edgeCount + input.interGalaxyConnections.count
    guard totalCapacity > 0 else { return frame }

    // Pre-allocate, write by index
    frame.descriptors = [EdgeDescriptor](
        repeating: EdgeDescriptor(sourceIdx: 0, targetIdx: 0, sourceRadius: 0, targetRadius: 0,
                                   color: .zero, baseRadius: 0, state: 0, _pad0: 0, _pad1: 0),
        count: totalCapacity)

    let normalRadius = input.edgeRadius * 1.3
    let connectedRadius = input.edgeRadius * 2.5
    let selectedIdx = input.selectedIdx
    let isSearch = input.isSearchActive
    let isSemantic = input.isSemanticMode
    var ei = 0

    for edge in input.resolvedEdges {
        let connected = selectedIdx != nil && (edge.sourceIdx == selectedIdx! || edge.targetIdx == selectedIdx!)
        let radius = connected ? connectedRadius : normalRadius

        let state: Float
        if connected {
            state = 1
        } else if isSearch
            && input.searchDimmedSourceIdx.contains(edge.sourceIdx)
            && input.searchDimmedSourceIdx.contains(edge.targetIdx) {
            state = 2
        } else if isSemantic {
            state = 3
        } else {
            state = 0
        }

        let color = connected ? SIMD3<Float>(1, 1, 1) : edge.color

        frame.descriptors[ei] = EdgeDescriptor(
            sourceIdx: edge.sourceIdx,
            targetIdx: edge.targetIdx,
            sourceRadius: edge.sourceRadius,
            targetRadius: edge.targetRadius,
            color: SIMD4<Float>(color.x, color.y, color.z, 1),
            baseRadius: radius,
            state: state,
            _pad0: 0, _pad1: 0
        )
        ei += 1
    }

    // Inter-galaxy connections
    let interGalaxyBase = UInt32(input.nodeCount)
    for (i, conn) in input.interGalaxyConnections.enumerated() {
        frame.interGalaxyPositions.append(conn.from)
        frame.interGalaxyPositions.append(conn.to)

        frame.descriptors[ei] = EdgeDescriptor(
            sourceIdx: interGalaxyBase + UInt32(i * 2),
            targetIdx: interGalaxyBase + UInt32(i * 2 + 1),
            sourceRadius: 0, targetRadius: 0,
            color: SIMD4<Float>(0.5, 0.5, 0.6, 0.4),
            baseRadius: input.edgeRadius * 4.0,
            state: 2,
            _pad0: 0, _pad1: 0
        )
        ei += 1
    }

    if ei < totalCapacity {
        frame.descriptors.removeSubrange(ei..<totalCapacity)
    }
    frame.count = ei
    return frame
}
