import Lattice
import Foundation

@Model
final class VisualizerConfig {
    var hiddenProjects: Set<String> = []
    var hiddenRelations: Set<String> = []
    var layoutMode: LayoutMode = .forceDirected
    var dimensionMode: DimensionMode = .twoD
    var showVoids: Bool = false
    var soundEnabled: Bool = false
}
