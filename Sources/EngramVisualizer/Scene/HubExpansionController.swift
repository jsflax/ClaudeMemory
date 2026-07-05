import Foundation
import simd

/// Manages hub expand/collapse orbit animations.
/// Extracted from MetalSceneManager — owns the expansion state and lerps
/// child node positions each frame when a hub is expanded or collapsing.
@MainActor
final class HubExpansionController {

    var expandedHubs: Set<UUID> = []
    var expansionProgress: [UUID: Float] = [:]
    var expansionDirection: [UUID: Bool] = [:]
    var preExpansionPositions: [UUID: SIMD3<Float>] = [:]
    var expandedChildPositions: [UUID: SIMD3<Float>] = [:]
    var pendingHubToggles: [(hubId: UUID, expanding: Bool)] = []

    // MARK: - Queries

    func childrenOfHub(
        _ hubId: UUID,
        edges: [(sourceId: UUID, targetId: UUID, relation: String)]
    ) -> [UUID] {
        edges.filter { $0.relation == "part_of" && $0.targetId == hubId }.map(\.sourceId)
    }

    func computeOrbitPositions(
        hubId: UUID,
        children: [UUID],
        positions: [UUID: SIMD3<Float>]
    ) -> [UUID: SIMD3<Float>] {
        guard let hubPos = positions[hubId] else { return [:] }
        let radius: Float = 80
        var result: [UUID: SIMD3<Float>] = [:]
        let n = children.count
        guard n > 0 else { return result }
        let goldenRatio: Float = (1 + sqrt(5)) / 2
        for (idx, childId) in children.enumerated() {
            let i = Float(idx)
            let theta = acos(1 - 2 * (i + 0.5) / Float(n))
            let phi = 2 * Float.pi * i / goldenRatio
            let x = radius * sin(theta) * cos(phi)
            let y = radius * sin(theta) * sin(phi)
            let z = radius * cos(theta)
            result[childId] = hubPos + SIMD3(x, y, z)
        }
        return result
    }

    // MARK: - Toggle

    func toggleHubExpansion(
        hubId: UUID,
        positions: [UUID: SIMD3<Float>],
        edges: [(sourceId: UUID, targetId: UUID, relation: String)]
    ) {
        if expandedHubs.contains(hubId) {
            expansionDirection[hubId] = false
        } else {
            expandedHubs.insert(hubId)
            let children = childrenOfHub(hubId, edges: edges)
            for childId in children {
                preExpansionPositions[childId] = positions[childId] ?? .zero
            }
            expansionProgress[hubId] = 0
            expansionDirection[hubId] = true
        }
    }

    // MARK: - Per-frame update

    /// Advance all active expansions/collapses and write lerped positions
    /// into both `positions` (in-out) and `expandedChildPositions`.
    func updateExpansions(
        dt: Float,
        positions: inout [UUID: SIMD3<Float>],
        edges: [(sourceId: UUID, targetId: UUID, relation: String)]
    ) {
        var toRemove: [UUID] = []
        var allExpandedPositions: [UUID: SIMD3<Float>] = [:]

        for hubId in expandedHubs {
            let expanding = expansionDirection[hubId] ?? true
            var progress = expansionProgress[hubId] ?? 0

            if expanding { progress = min(1.0, progress + dt * 3.0) }
            else { progress = max(0.0, progress - dt * 3.0) }
            expansionProgress[hubId] = progress

            let t = progress * progress * (3 - 2 * progress)
            let children = childrenOfHub(hubId, edges: edges)
            let orbitPositions = computeOrbitPositions(hubId: hubId, children: children, positions: positions)

            for childId in children {
                let startPos = preExpansionPositions[childId] ?? positions[childId] ?? .zero
                let endPos = orbitPositions[childId] ?? startPos
                let lerpedPos = startPos + (endPos - startPos) * t
                positions[childId] = lerpedPos
                allExpandedPositions[childId] = lerpedPos
            }

            if !expanding && progress <= 0 {
                toRemove.append(hubId)
                for childId in children {
                    if let original = preExpansionPositions[childId] {
                        positions[childId] = original
                    }
                    preExpansionPositions.removeValue(forKey: childId)
                }
            }
        }

        expandedChildPositions = allExpandedPositions
        for hubId in toRemove {
            expandedHubs.remove(hubId)
            expansionProgress.removeValue(forKey: hubId)
            expansionDirection.removeValue(forKey: hubId)
        }
    }
}
