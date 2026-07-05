// NebulaFrameBuilder — pure function converting project centroids → NebulaGroupInput[].

import simd
import CEngramSceneTypes

/// Build nebula group inputs from project centroid data.
/// Returns empty array if fewer than 2 members in a group.
public func buildNebulaGroups(
    projectCentroids: [String: (centroid: SIMD3<Float>, maxY: Float, count: Int)],
    projectColors: [String: SIMD4<Float>],
    scaleFactor: Float
) -> [NebulaGroupInput] {
    var groups: [NebulaGroupInput] = []
    groups.reserveCapacity(projectCentroids.count)

    for (project, data) in projectCentroids where data.count >= 2 {
        let worldCentroid = data.centroid * scaleFactor
        let radius = max(30.0, Float(data.count) * 2.0) * scaleFactor
        let color = projectColors[project] ?? SIMD4<Float>(0.5, 0.5, 0.5, 0.25)
        let noisePhase = Float(project.hashValue & 0xFFFF) / Float(0xFFFF) * 10.0

        groups.append(NebulaGroupInput(
            centroid: worldCentroid,
            radius: radius,
            color: color,
            noisePhase: noisePhase,
            _pad0: 0, _pad1: 0, _pad2: 0
        ))
    }

    return groups
}
