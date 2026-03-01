import Lattice
import Foundation

@Model
final class VisualizerConfig {
    var hiddenProjects: Set<String> = []
    var hiddenRelations: Set<String> = []
    var layoutMode: LayoutMode = .forceDirected
    var dimensionMode: DimensionMode = .threeD
    var showVoids: Bool = false
    var soundEnabled: Bool = false
    var notificationsEnabled: Bool = false
    var showActivityLog: Bool = true
    var showStatsOverlay: Bool = false
}
