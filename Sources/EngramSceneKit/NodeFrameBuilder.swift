// SceneNodeFrameBuilder — pure function converting domain node data → GPU-ready NodePackInput[].
// No @MainActor, no Metal imports. Fully testable.

import Foundation
import simd
@_exported import CEngramSceneTypes

// MARK: - Input / Output Types

/// Flattened node data for frame building. Uses Float elapsed times (not Dates)
/// so the builder is deterministic and testable.
public struct SceneNodeFrameInput: Sendable {
    /// Parallel arrays — all indexed by node order (same as renderNodes array).
    /// Nodes without positions have .zero in nodePositions and false in hasPosition.
    public let nodeIds: [UUID]
    public let nodeProjects: [String]         // project name per node (for color + centroid)
    public let nodeImportances: [Int]         // importance per node
    public let nodePositions: [SIMD3<Float>]  // position per node (parallel, not dict)
    public let hasPosition: [Bool]            // true if node has a position
    public let nodeRadii: [Float]             // pre-computed radius per node (parallel)
    public let nodeIsHub: [Bool]              // hub flag per node (parallel)
    public let projectColors: [String: SIMD3<Float>]
    public let selectedNodeIndex: Int?        // index into arrays, not UUID
    public let glowingNodes: [UUID: Float]    // sparse — typically 0-5 entries
    public let newNodes: [UUID: Float]        // sparse — typically 0-5 entries
    public let isSearchActive: Bool
    public let searchMatchIndices: Set<Int>   // indices, not UUIDs
    public let inspectingIntensity: [UUID: Float] // sparse
    public let birthingElapsed: [UUID: Float]     // sparse
    public let nodeRadius: Float
    public let cameraPosition: SIMD3<Float>

    public init(
        nodeIds: [UUID],
        nodeProjects: [String],
        nodeImportances: [Int],
        nodePositions: [SIMD3<Float>],
        hasPosition: [Bool],
        nodeRadii: [Float],
        nodeIsHub: [Bool],
        projectColors: [String: SIMD3<Float>],
        selectedNodeIndex: Int?,
        glowingNodes: [UUID: Float],
        newNodes: [UUID: Float],
        isSearchActive: Bool,
        searchMatchIndices: Set<Int>,
        inspectingIntensity: [UUID: Float],
        birthingElapsed: [UUID: Float],
        nodeRadius: Float,
        cameraPosition: SIMD3<Float>
    ) {
        self.nodeIds = nodeIds
        self.nodeProjects = nodeProjects
        self.nodeImportances = nodeImportances
        self.nodePositions = nodePositions
        self.hasPosition = hasPosition
        self.nodeRadii = nodeRadii
        self.nodeIsHub = nodeIsHub
        self.projectColors = projectColors
        self.selectedNodeIndex = selectedNodeIndex
        self.glowingNodes = glowingNodes
        self.newNodes = newNodes
        self.isSearchActive = isSearchActive
        self.searchMatchIndices = searchMatchIndices
        self.inspectingIntensity = inspectingIntensity
        self.birthingElapsed = birthingElapsed
        self.nodeRadius = nodeRadius
        self.cameraPosition = cameraPosition
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
    let nodeCount = input.nodeIds.count
    guard nodeCount > 0 else { return frame }

    // Pre-allocate at full capacity — write by index
    frame.packInputs = [NodePackInput](repeating: NodePackInput(position: .zero, baseRadius: 0, baseColor: .zero, packedState: 0), count: nodeCount)
    frame.nodePositions = [SIMD3<Float>](repeating: .zero, count: nodeCount)

    var centroidSums: [String: (sum: SIMD3<Float>, count: Int, maxY: Float)] = [:]
    var depthMin: Float = .greatestFiniteMagnitude
    var depthMax: Float = 0
    var actualCount = 0

    let isSearch = input.isSearchActive
    let selectedIdx = input.selectedNodeIndex
    let camPos = input.cameraPosition

    for i in 0..<nodeCount {
        guard input.hasPosition[i] else { continue }
        let pos = input.nodePositions[i]
        let nodeId = input.nodeIds[i]
        let project = input.nodeProjects[i]
        let r = input.nodeRadii[i]

        // Depth range
        let d = simd_length(pos - camPos)
        depthMin = min(depthMin, d)
        depthMax = max(depthMax, d)

        // Centroid accumulation
        var entry = centroidSums[project, default: (.zero, 0, -.greatestFiniteMagnitude)]
        entry.sum += pos
        entry.count += 1
        entry.maxY = max(entry.maxY, pos.y)
        centroidSums[project] = entry

        // Visual state — sparse dict lookups only for effects (0-5 entries each, cheap misses)
        let ri: Float = {
            guard let elapsed = input.glowingNodes[nodeId], i != selectedIdx else { return 0 }
            return recallGlowIntensity(elapsed: elapsed)
        }()
        let ai: Float = {
            guard let elapsed = input.newNodes[nodeId] else { return 0 }
            return arrivalGlowIntensity(elapsed: elapsed)
        }()

        let searchDimmed = isSearch && !input.searchMatchIndices.contains(i)
        let searchMatched = isSearch && input.searchMatchIndices.contains(i) && i != selectedIdx
        let isInspecting = input.inspectingIntensity[nodeId] != nil

        let birthElapsed = input.birthingElapsed[nodeId]
        let isBirthing = birthElapsed != nil
        let birthInt: Float = birthElapsed.map { birthIntensity(elapsed: $0) } ?? 0

        if birthInt >= 1.0 && isBirthing {
            frame.completedBirths.append(nodeId)
        }

        let curState: Float
        let curIntensity: Float
        if isBirthing {
            curState = 6; curIntensity = birthInt
        } else if i == selectedIdx {
            curState = 1; curIntensity = 0
        } else if searchMatched {
            curState = 4; curIntensity = 0
        } else if searchDimmed {
            curState = 0; curIntensity = 0
        } else if isInspecting {
            curState = 5; curIntensity = input.inspectingIntensity[nodeId] ?? 0.5
        } else if ri > 0 {
            curState = 2; curIntensity = ri
        } else if ai > 0 {
            curState = 3; curIntensity = ai
        } else {
            curState = 0; curIntensity = 0
        }

        let packedState = packNodeState(stateType: curState, searchDimmed: searchDimmed, intensity: curIntensity)
        let color: SIMD3<Float> = i == selectedIdx
            ? SIMD3<Float>(1, 1, 1)
            : (input.projectColors[project] ?? SIMD3<Float>(0.5, 0.5, 0.5))

        frame.packInputs[actualCount] = NodePackInput(
            position: pos, baseRadius: r, baseColor: color, packedState: packedState
        )
        frame.nodePositions[actualCount] = pos
        actualCount += 1
    }

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
