// SceneLabelFrameBuilder — pure function converting label data → GPU-ready LabelInstance[].

import Foundation
import simd
import CEngramSceneTypes

// MARK: - Input / Output Types

/// Atlas rect from texture packing.
public struct AtlasRect: Sendable {
    public let u0: Float, v0: Float, u1: Float, v1: Float
    public init(u0: Float, v0: Float, u1: Float, v1: Float) {
        self.u0 = u0; self.v0 = v0; self.u1 = u1; self.v1 = v1
    }
}

public struct SceneLabelFrameInput: Sendable {
    public let nodeLabels: [(id: UUID, position: SIMD3<Float>, isHub: Bool,
                             importance: Int, isSelected: Bool,
                             isSearchMatch: Bool, searchDimmed: Bool,
                             isExpandedChild: Bool)]
    public let atlasRects: [UUID: AtlasRect]
    public let projectCentroids: [String: (centroid: SIMD3<Float>, maxY: Float, count: Int)]
    public let projectLabelRects: [String: AtlasRect]
    public let projectColors: [String: SIMD3<Float>]
    public let galaxyLabels: [(name: String, center: SIMD3<Float>)]
    public let galaxyLabelRects: [String: AtlasRect]
    public let topicGroups: [(topic: String, project: String, memberPositions: [SIMD3<Float>])]
    public let topicLabelRects: [String: AtlasRect]
    public let scaleFactor: Float
    public let nodeRadius: Float
    public let aspectCorrection: Float
    public let maxLabels: Int

    public init(
        nodeLabels: [(id: UUID, position: SIMD3<Float>, isHub: Bool,
                       importance: Int, isSelected: Bool,
                       isSearchMatch: Bool, searchDimmed: Bool,
                       isExpandedChild: Bool)],
        atlasRects: [UUID: AtlasRect],
        projectCentroids: [String: (centroid: SIMD3<Float>, maxY: Float, count: Int)],
        projectLabelRects: [String: AtlasRect],
        projectColors: [String: SIMD3<Float>],
        galaxyLabels: [(name: String, center: SIMD3<Float>)],
        galaxyLabelRects: [String: AtlasRect],
        topicGroups: [(topic: String, project: String, memberPositions: [SIMD3<Float>])],
        topicLabelRects: [String: AtlasRect],
        scaleFactor: Float,
        nodeRadius: Float,
        aspectCorrection: Float,
        maxLabels: Int = 2048
    ) {
        self.nodeLabels = nodeLabels
        self.atlasRects = atlasRects
        self.projectCentroids = projectCentroids
        self.projectLabelRects = projectLabelRects
        self.projectColors = projectColors
        self.galaxyLabels = galaxyLabels
        self.galaxyLabelRects = galaxyLabelRects
        self.topicGroups = topicGroups
        self.topicLabelRects = topicLabelRects
        self.scaleFactor = scaleFactor
        self.nodeRadius = nodeRadius
        self.aspectCorrection = aspectCorrection
        self.maxLabels = maxLabels
    }
}

public struct SceneLabelFrame: Sendable {
    public var instances: [LabelInstance]
    public var count: Int { instances.count }

    public init() {
        self.instances = []
    }
}

// MARK: - Builder

public func buildSceneLabelFrame(_ input: SceneLabelFrameInput) -> SceneLabelFrame {
    var frame = SceneLabelFrame()
    let sf = input.scaleFactor
    let ac = input.aspectCorrection

    let projectLabelCount = input.projectCentroids.count
    let topicLabelCount = input.topicGroups.count
    let maxNodeLabels = max(0, input.maxLabels - projectLabelCount - topicLabelCount)

    frame.instances.reserveCapacity(min(
        input.nodeLabels.count + projectLabelCount + topicLabelCount + input.galaxyLabels.count,
        input.maxLabels
    ))

    // Node labels
    for node in input.nodeLabels {
        guard frame.instances.count < maxNodeLabels,
              let rect = input.atlasRects[node.id] else { continue }

        let maxVisible: Float = (node.isSelected || node.isExpandedChild || node.isSearchMatch)
            ? .greatestFiniteMagnitude
            : (node.isHub ? 1200 : (200 + Float(max(1, node.importance)) * 80))

        let baseOpacity: Float = node.isSelected ? 0.95 : (node.isHub ? 0.8 : 0.6)
        let halfH: Float = node.isSelected ? 0.025 : (node.isHub ? 0.022 : 0.018)
        let textAspect = (rect.u1 - rect.u0) / max(0.001, rect.v1 - rect.v0) * ac

        let anchor = node.position * sf + SIMD3<Float>(0, input.nodeRadius * 1.8, 0)

        var flags: UInt32 = 0
        if node.isSelected { flags |= 1 }
        if node.isSearchMatch { flags |= 2 }
        if node.searchDimmed { flags |= 4 }

        frame.instances.append(LabelInstance(
            anchor: anchor,
            halfH: halfH,
            uvRect: SIMD4<Float>(rect.u0, rect.v0, rect.u1, rect.v1),
            color: SIMD4<Float>(1, 1, 1, baseOpacity),
            textAspect: textAspect,
            maxVisible: maxVisible,
            forwardBias: 0,
            flags: flags
        ))
    }

    // Project labels
    for (project, centroidData) in input.projectCentroids {
        guard let rect = input.projectLabelRects[project] else { continue }

        let labelY = centroidData.maxY * sf + 0.15
        let anchor = SIMD3<Float>(centroidData.centroid.x * sf, labelY, centroidData.centroid.z * sf)
        let color = input.projectColors[project] ?? SIMD3<Float>(1, 1, 1)
        let textAspect = (rect.u1 - rect.u0) / max(0.001, rect.v1 - rect.v0) * ac

        frame.instances.append(LabelInstance(
            anchor: anchor,
            halfH: 0.20,
            uvRect: SIMD4<Float>(rect.u0, rect.v0, rect.u1, rect.v1),
            color: SIMD4<Float>(color.x, color.y, color.z, 0.9),
            textAspect: textAspect,
            maxVisible: 12000,
            forwardBias: 0.25,
            flags: 0
        ))
    }

    // Galaxy labels
    for galaxy in input.galaxyLabels {
        guard let rect = input.galaxyLabelRects[galaxy.name] else { continue }

        let center = galaxy.center * sf
        let anchor = SIMD3<Float>(center.x, center.y + 0.6, center.z)
        let textAspect = (rect.u1 - rect.u0) / max(0.001, rect.v1 - rect.v0) * ac

        frame.instances.append(LabelInstance(
            anchor: anchor,
            halfH: 0.25,
            uvRect: SIMD4<Float>(rect.u0, rect.v0, rect.u1, rect.v1),
            color: SIMD4<Float>(0.8, 0.85, 1.0, 0.85),
            textAspect: textAspect,
            maxVisible: .greatestFiniteMagnitude,
            forwardBias: 0.3,
            flags: 0
        ))
    }

    // Topic group labels
    for group in input.topicGroups {
        guard group.memberPositions.count >= 3,
              let rect = input.topicLabelRects[group.topic] else { continue }

        var sum = SIMD3<Float>.zero
        var maxY: Float = -.greatestFiniteMagnitude
        for pos in group.memberPositions {
            sum += pos
            maxY = max(maxY, pos.y)
        }
        let centroid = sum / Float(group.memberPositions.count)

        let labelY = maxY * sf + 0.06
        let anchor = SIMD3<Float>(centroid.x * sf, labelY, centroid.z * sf)
        let textAspect = (rect.u1 - rect.u0) / max(0.001, rect.v1 - rect.v0) * ac
        let color = input.projectColors[group.project] ?? SIMD3<Float>(0.75, 0.82, 0.95)

        frame.instances.append(LabelInstance(
            anchor: anchor,
            halfH: 0.06,
            uvRect: SIMD4<Float>(rect.u0, rect.v0, rect.u1, rect.v1),
            color: SIMD4<Float>(color.x, color.y, color.z, 0.6),
            textAspect: textAspect,
            maxVisible: 1500,
            forwardBias: 0.15,
            flags: 0
        ))
    }

    return frame
}
