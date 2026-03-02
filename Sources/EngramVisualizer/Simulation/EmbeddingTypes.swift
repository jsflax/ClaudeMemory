import Foundation
import SwiftUI
import Lattice
import simd

// MARK: - Data Types

@LatticeEnum enum LayoutMode: String, CaseIterable, Equatable {
    case forceDirected = "Force"
    case embedding = "Semantic"
}

@LatticeEnum enum DimensionMode: String, CaseIterable, Equatable {
    case twoD = "2D"
    case threeD = "3D"
}

enum ProjectionState: Equatable {
    case idle
    case loadingEmbeddings
    case computing(progress: Double)
    case ready
    case failed(String)
}

struct KnowledgeVoid: Identifiable {
    let id = UUID()
    let position: CGPoint
    let radius: CGFloat
    let nearestTopics: [String]
    let sparsity: CGFloat  // 0..1
}

struct SemanticCluster: Identifiable {
    let id = UUID()
    let nodeIds: [UUID]
    let centroid: CGPoint
    let hullPoints: [CGPoint]  // convex hull for drawing
    let label: String          // dominant topics/projects
    let projectBreakdown: [(project: String, count: Int)]
    var subClusters: [SemanticCluster] = []
}

struct SemanticCluster3D: Identifiable {
    let id = UUID()
    let nodeIds: [UUID]
    let centroid: SIMD3<Float>
    let boundingRadius: Float
    let label: String
    let projectBreakdown: [(project: String, count: Int)]
}
