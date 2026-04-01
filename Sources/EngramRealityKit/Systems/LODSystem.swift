import simd
import Foundation

/// Distance-based LOD + culling system. Runs before batch systems.
/// Determines which nodes/edges/labels are visible each frame.
///
/// LOD tiers:
/// - Near (< 500): Full sphere, glow effects, labels
/// - Mid (500–2000): Reduced detail, hub/important labels only
/// - Far (2000–5000): Point sprite, no labels
/// - Culled (> 5000): Hidden
///
/// Render budget caps:
/// - Max ~8,000 node instances (near + mid + far combined)
/// - Max ~30,000 edge instances
/// - Max ~2,000 label instances
@MainActor
public final class LODSystem {

    /// Render budget caps.
    public var maxNodeInstances: Int = 8_000
    public var maxEdgeInstances: Int = 30_000
    public var maxLabelInstances: Int = 2_000

    public init() {}

    /// Compute which nodes/edges/labels are visible this frame.
    public func computeVisibleSet(
        nodes: [RKNodeSnapshot],
        edges: [RKEdgeSnapshot],
        positions: [UUID: SIMD3<Float>],
        positionArray: [SIMD3<Float>] = [],
        cameraPosition: SIMD3<Float>,
        selectedNode: UUID?,
        glowingNodes: [UUID: Float],
        hubs: Set<UUID>
    ) -> VisibleSet {
        // Compute per-node distances and classify into tiers
        var nearNodes: [(index: Int, distance: Float, priority: Int)] = []
        var midNodes: [(index: Int, distance: Float, priority: Int)] = []
        var farNodes: [(index: Int, distance: Float, priority: Int)] = []

        let useArray = positionArray.count == nodes.count
        for i in 0..<nodes.count {
            let node = nodes[i]
            let pos: SIMD3<Float>
            if useArray {
                pos = positionArray[i]
            } else {
                guard let p = positions[node.id] else { continue }
                pos = p
            }
            let dist = simd_length(pos - cameraPosition)
            let tier = LODTier.from(distance: dist)

            // Priority: selected > glowing > hub > importance > distance
            var priority = 0
            if node.id == selectedNode { priority = 1000 }
            else if glowingNodes[node.id] != nil { priority = 500 }
            else if node.isHub { priority = 100 }
            else { priority = node.importance }

            switch tier {
            case .near: nearNodes.append((i, dist, priority))
            case .mid: midNodes.append((i, dist, priority))
            case .far: farNodes.append((i, dist, priority))
            case .culled: break
            }
        }

        // Sort each tier by priority descending, then distance ascending
        nearNodes.sort { $0.priority != $1.priority ? $0.priority > $1.priority : $0.distance < $1.distance }
        midNodes.sort { $0.priority != $1.priority ? $0.priority > $1.priority : $0.distance < $1.distance }
        farNodes.sort { $0.priority != $1.priority ? $0.priority > $1.priority : $0.distance < $1.distance }

        // Apply render budget
        var remaining = maxNodeInstances
        let nearCapped = Array(nearNodes.prefix(remaining).map(\.index))
        remaining -= nearCapped.count
        let midCapped = Array(midNodes.prefix(max(0, remaining)).map(\.index))
        remaining -= midCapped.count
        let farCapped = Array(farNodes.prefix(max(0, remaining)).map(\.index))

        // Build visible node ID set for edge filtering
        var visibleNodeIds = Set<UUID>(minimumCapacity: nearCapped.count + midCapped.count + farCapped.count)
        for idx in nearCapped { visibleNodeIds.insert(nodes[idx].id) }
        for idx in midCapped { visibleNodeIds.insert(nodes[idx].id) }
        for idx in farCapped { visibleNodeIds.insert(nodes[idx].id) }

        // Filter edges: both endpoints must be visible
        var visibleEdges: [Int] = []
        visibleEdges.reserveCapacity(min(edges.count, maxEdgeInstances))
        for i in 0..<edges.count {
            guard visibleEdges.count < maxEdgeInstances else { break }
            let edge = edges[i]
            if visibleNodeIds.contains(edge.sourceId) && visibleNodeIds.contains(edge.targetId) {
                visibleEdges.append(i)
            }
        }

        // Labels: near nodes + important mid nodes
        var visibleLabels: [Int] = []
        visibleLabels.reserveCapacity(min(nearCapped.count + midCapped.count, maxLabelInstances))
        for idx in nearCapped {
            guard visibleLabels.count < maxLabelInstances else { break }
            visibleLabels.append(idx)
        }
        // Add hub/important mid nodes
        for idx in midCapped {
            guard visibleLabels.count < maxLabelInstances else { break }
            if nodes[idx].isHub || nodes[idx].importance >= 3 {
                visibleLabels.append(idx)
            }
        }

        return VisibleSet(
            nearNodes: nearCapped,
            midNodes: midCapped,
            farNodes: farCapped,
            visibleEdgeIndices: visibleEdges,
            visibleLabelIndices: visibleLabels
        )
    }
}
