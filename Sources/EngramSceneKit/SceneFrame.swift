// EngramSceneKit — Pure functions for converting domain data → GPU-ready structs.
// No Metal imports, no @MainActor. Fully testable without a GPU.

import Foundation
import simd
import CEngramSceneTypes

/// Complete frame of GPU-ready scene data, built from domain inputs.
public struct SceneFrame: Sendable {
    public let nodes: SceneNodeFrame
    public let edges: SceneEdgeFrame
    public let labels: SceneLabelFrame
    public let nebulae: [NebulaGroupInput]

    public init(nodes: SceneNodeFrame, edges: SceneEdgeFrame, labels: SceneLabelFrame, nebulae: [NebulaGroupInput]) {
        self.nodes = nodes
        self.edges = edges
        self.labels = labels
        self.nebulae = nebulae
    }
}
