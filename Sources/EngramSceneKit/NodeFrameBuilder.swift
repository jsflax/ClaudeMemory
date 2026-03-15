// SceneNodeFrameBuilder — pure function converting domain node data → GPU-ready NodePackInput[].
// No @MainActor, no Metal imports. Fully testable.

import Foundation
import simd
@_exported import CEngramSceneTypes

// MARK: - Input / Output Types

/// Flattened node data for frame building. Uses Float elapsed times (not Dates)
/// so the builder is deterministic and testable.
public struct SceneNodeFrameInput: Sendable {
    public let nodes: [(id: UUID, project: String, importance: Int)]
    public let positions: [UUID: SIMD3<Float>]
    public let hubs: Set<UUID>
    public let projectColors: [String: SIMD3<Float>]
    public let selectedNode: UUID?
    public let glowingNodes: [UUID: Float]       // elapsed seconds since glow start
    public let newNodes: [UUID: Float]            // elapsed seconds since arrival
    public let isSearchActive: Bool
    public let searchMatchIds: Set<UUID>
    public let inspectingIntensity: [UUID: Float]
    public let birthingElapsed: [UUID: Float]     // elapsed seconds since birth start
    public let nodeRadius: Float
    public let cameraPosition: SIMD3<Float>
    /// Pre-built node index map (cached, only rebuilt on topology change)
    public let nodeIndexMap: [UUID: UInt32]
    /// Pre-built node radii (cached, only rebuilt on topology change)
    public let nodeRadii: [UUID: Float]

    public init(
        nodes: [(id: UUID, project: String, importance: Int)],
        positions: [UUID: SIMD3<Float>],
        hubs: Set<UUID>,
        projectColors: [String: SIMD3<Float>],
        selectedNode: UUID?,
        glowingNodes: [UUID: Float],
        newNodes: [UUID: Float],
        isSearchActive: Bool,
        searchMatchIds: Set<UUID>,
        inspectingIntensity: [UUID: Float],
        birthingElapsed: [UUID: Float],
        nodeRadius: Float,
        cameraPosition: SIMD3<Float>,
        nodeIndexMap: [UUID: UInt32] = [:],
        nodeRadii: [UUID: Float] = [:]
    ) {
        self.nodes = nodes
        self.positions = positions
        self.hubs = hubs
        self.projectColors = projectColors
        self.selectedNode = selectedNode
        self.glowingNodes = glowingNodes
        self.newNodes = newNodes
        self.isSearchActive = isSearchActive
        self.searchMatchIds = searchMatchIds
        self.inspectingIntensity = inspectingIntensity
        self.birthingElapsed = birthingElapsed
        self.nodeRadius = nodeRadius
        self.cameraPosition = cameraPosition
        self.nodeIndexMap = nodeIndexMap
        self.nodeRadii = nodeRadii
    }
}

/// GPU-ready node frame output.
/// nodeIndexMap/nodeRadii/nodeProjects are NOT rebuilt here — they're cached
/// in MetalSceneManager and only rebuilt on topology change (not every frame).
public struct SceneNodeFrame: Sendable {
    public var packInputs: [NodePackInput]
    public var nodePositions: [SIMD3<Float>]
    public var projectCentroids: [String: (centroid: SIMD3<Float>, maxY: Float, count: Int)]
    public var minDepth: Float
    public var maxDepth: Float
    public var completedBirths: [UUID]

    public init() {
        self.packInputs = []
        self.nodePositions = []
        self.projectCentroids = [:]
        self.minDepth = .greatestFiniteMagnitude
        self.maxDepth = 0
        self.completedBirths = []
    }
}

// MARK: - Builder

/// Build a GPU-ready SceneNodeFrame from domain inputs. Pure function — no side effects.
public func buildSceneNodeFrame(_ input: SceneNodeFrameInput) -> SceneNodeFrame {
    var frame = SceneNodeFrame()
    let nodeCount = input.nodes.count
    guard nodeCount > 0 else { return frame }

    // Pre-allocate at full capacity — write by index, not append (avoids realloc)
    frame.packInputs = [NodePackInput](repeating: NodePackInput(position: .zero, baseRadius: 0, baseColor: .zero, packedState: 0), count: nodeCount)
    frame.nodePositions = [SIMD3<Float>](repeating: .zero, count: nodeCount)

    let hasPrebuiltRadii = !input.nodeRadii.isEmpty
    var centroidSums: [String: (sum: SIMD3<Float>, count: Int, maxY: Float)] = [:]
    var depthMin: Float = .greatestFiniteMagnitude
    var depthMax: Float = 0
    var actualCount = 0

    for node in input.nodes {
        guard let pos = input.positions[node.id] else { continue }

        // Depth range
        let d = simd_length(pos - input.cameraPosition)
        depthMin = min(depthMin, d)
        depthMax = max(depthMax, d)

        // Centroid accumulation
        var entry = centroidSums[node.project, default: (.zero, 0, -.greatestFiniteMagnitude)]
        entry.sum += pos
        entry.count += 1
        entry.maxY = max(entry.maxY, pos.y)
        centroidSums[node.project] = entry

        // Radius: use pre-built cache if available, otherwise compute
        let r: Float
        if hasPrebuiltRadii, let cached = input.nodeRadii[node.id] {
            r = cached
        } else {
            let isHub = input.hubs.contains(node.id)
            let importance = max(1, node.importance)
            let baseRadius: Float = isHub ? input.nodeRadius * 1.6 : input.nodeRadius
            r = baseRadius * (1.0 + Float(importance - 1) * 0.08)
        }

        // Visual state computation
        let ri: Float = {
            guard let elapsed = input.glowingNodes[node.id], node.id != input.selectedNode else { return 0 }
            return recallGlowIntensity(elapsed: elapsed)
        }()

        let ai: Float = {
            guard let elapsed = input.newNodes[node.id] else { return 0 }
            return arrivalGlowIntensity(elapsed: elapsed)
        }()

        let searchDimmed = input.isSearchActive && !input.searchMatchIds.contains(node.id)
        let searchMatched = input.isSearchActive && input.searchMatchIds.contains(node.id) && node.id != input.selectedNode
        let isInspecting = input.inspectingIntensity[node.id] != nil

        let birthElapsed = input.birthingElapsed[node.id]
        let isBirthing = birthElapsed != nil
        let birthInt: Float = birthElapsed.map { birthIntensity(elapsed: $0) } ?? 0

        if birthInt >= 1.0 && isBirthing {
            frame.completedBirths.append(node.id)
        }

        // State priority: birthing > selected > searchMatch > dimmed > inspect > recall > arrival > normal
        let curState: Float
        let curIntensity: Float
        if isBirthing {
            curState = 6; curIntensity = birthInt
        } else if node.id == input.selectedNode {
            curState = 1; curIntensity = 0
        } else if searchMatched {
            curState = 4; curIntensity = 0
        } else if searchDimmed {
            curState = 0; curIntensity = 0
        } else if isInspecting {
            curState = 5; curIntensity = input.inspectingIntensity[node.id] ?? 0.5
        } else if ri > 0 {
            curState = 2; curIntensity = ri
        } else if ai > 0 {
            curState = 3; curIntensity = ai
        } else {
            curState = 0; curIntensity = 0
        }

        let packedState = packNodeState(stateType: curState, searchDimmed: searchDimmed, intensity: curIntensity)

        let color: SIMD3<Float> = node.id == input.selectedNode
            ? SIMD3<Float>(1, 1, 1)
            : (input.projectColors[node.project] ?? SIMD3<Float>(0.5, 0.5, 0.5))

        frame.packInputs[actualCount] = NodePackInput(
            position: pos,
            baseRadius: r,
            baseColor: color,
            packedState: packedState
        )
        frame.nodePositions[actualCount] = pos
        actualCount += 1
    }

    // Trim to actual count (nodes without positions were skipped)
    if actualCount < nodeCount {
        frame.packInputs.removeSubrange(actualCount..<nodeCount)
        frame.nodePositions.removeSubrange(actualCount..<nodeCount)
    }

    // Finalize centroids
    for (project, data) in centroidSums where data.count >= 2 {
        let centroid = data.sum / Float(data.count)
        frame.projectCentroids[project] = (centroid: centroid, maxY: data.maxY, count: data.count)
    }
    frame.minDepth = depthMin
    frame.maxDepth = depthMax

    return frame
}
