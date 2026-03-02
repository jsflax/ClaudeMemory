import SwiftUI
import Lattice
import EngramKit
import simd

// MARK: - 3D Layout & View

extension GraphView {

    /// 3D positions for the current mode. Used by Graph3DView.
    var positions3D: [UUID: SIMD3<Float>] {
        if config.layoutMode == .embedding {
            let tsne3D = embeddingProjection.projectedPositions3D
            guard !tsne3D.isEmpty else { return simulation3D.positions }
            if transitionProgress >= 1.0 { return tsne3D }
            // Lerp between force snapshot and t-SNE
            var blended: [UUID: SIMD3<Float>] = [:]
            let allIds = Set(forcePositionSnapshot3D.keys).union(tsne3D.keys)
            for id in allIds {
                let forcePos = forcePositionSnapshot3D[id] ?? simulation3D.positions[id] ?? .zero
                let tsnePos = tsne3D[id] ?? forcePos
                blended[id] = forcePos + (tsnePos - forcePos) * Float(transitionProgress)
            }
            return blended
        }
        return simulation3D.positions
    }

    // MARK: - Dimension Mode Switching

    func switchDimensionMode(to mode: DimensionMode, viewSize: CGSize) {
        guard mode != config.dimensionMode else { return }
        config.dimensionMode = mode

        switch mode {
        case .threeD:
            // Seed 3D simulation from current 2D positions with z-jitter proportional to spread
            let currentPositions = effectivePositions ?? simulation.positions
            let xs = currentPositions.values.map { Float($0.x) }
            let ys = currentPositions.values.map { Float($0.y) }
            let spreadX = (xs.max() ?? 0) - (xs.min() ?? 0)
            let spreadY = (ys.max() ?? 0) - (ys.min() ?? 0)
            let zRange = max(spreadX, spreadY) * 0.4  // z spread = 40% of largest 2D axis
            var pos3D: [UUID: SIMD3<Float>] = [:]
            for (id, pt) in currentPositions {
                pos3D[id] = SIMD3(Float(pt.x), Float(pt.y), Float.random(in: -zRange...zRange))
            }

            // Rebuild 3D simulation topology
            let filtered = renderStore.nodes
            let currentIds = Set(filtered.map(\.id))
            let edgePairs = renderStore.edges.map { ($0.sourceId, $0.targetId) }
            var projectMap: [UUID: String] = [:]
            var topicMap: [UUID: String] = [:]
            for node in filtered { projectMap[node.id] = node.project; topicMap[node.id] = node.topic }
            simulation3D.updateGraph(nodeIds: currentIds, edges: edgePairs,
                                     projectForNode: projectMap, topicForNode: topicMap)
            simulation3D.setPositions(pos3D)
            simulation3D.isActive = (config.layoutMode == .forceDirected)

            // If in semantic mode, recompute t-SNE in 3D
            if config.layoutMode == .embedding {
                forcePositionSnapshot3D = pos3D
                embeddingProjection.invalidate()
                let nodeIds = renderStore.visibleNodeIds
                embeddingProjection.loadEmbeddings(for: nodeIds, from: lattice)
                let nodeScale = max(1.0, sqrt(Float(nodeIds.count) / 30.0))
                let spread = max(Float(viewSize.width), Float(viewSize.height)) * 1.2 * nodeScale

                Task {
                    await embeddingProjection.computeProjection3D(
                        nodeIds: nodeIds, spread: spread, initialPositions: pos3D
                    )
                    var topics: [UUID: String] = [:]
                    var projects: [UUID: String] = [:]
                    var labels: [UUID: String] = [:]
                    for node in renderStore.nodes {
                        topics[node.id] = node.topic
                        projects[node.id] = node.project
                        labels[node.id] = node.label
                    }
                    embeddingProjection.detectClusters3D(nodeTopics: topics, nodeProjects: projects, nodeLabels: labels)
                }
            }

        case .twoD:
            // Project 3D positions to 2D (drop z) and inject into 2D sim
            let current3D = positions3D
            var pos2D: [UUID: CGPoint] = [:]
            for (id, p) in current3D {
                pos2D[id] = CGPoint(x: CGFloat(p.x), y: CGFloat(p.y))
            }

            simulation3D.isActive = false

            if config.layoutMode == .forceDirected {
                simulation.setPositions(pos2D)
                simulation.isActive = true
            } else {
                // Back in 2D semantic — recompute 2D t-SNE
                embeddingProjection.invalidate()
                let nodeIds = renderStore.visibleNodeIds
                embeddingProjection.loadEmbeddings(for: nodeIds, from: lattice)
                let center = simulation.center
                let nodeScale = max(1.0, sqrt(CGFloat(nodeIds.count) / 30.0))
                let spread = max(viewSize.width, viewSize.height) * 1.2 * nodeScale

                projectedPositions2DFromSwitch(pos2D)

                Task {
                    await embeddingProjection.computeProjection(
                        nodeIds: nodeIds, center: center, spread: spread,
                        initialPositions: pos2D
                    )
                    var topics: [UUID: String] = [:]
                    var projects: [UUID: String] = [:]
                    var labels: [UUID: String] = [:]
                    for node in renderStore.nodes {
                        topics[node.id] = node.topic
                        projects[node.id] = node.project
                        labels[node.id] = node.label
                    }
                    embeddingProjection.detectClusters(nodeTopics: topics, nodeProjects: projects, nodeLabels: labels)
                    embeddingProjection.detectVoids(nodeTopics: topics, nodeProjects: projects)
                }
            }
        }
    }

    // MARK: - Layout Mode Switching (3D)

    /// Layout mode switching when in 3D dimension mode.
    func switchLayoutMode3D(to mode: LayoutMode, viewSize: CGSize) {
        switch mode {
        case .embedding:
            forcePositionSnapshot3D = simulation3D.positions
            simulation3D.isActive = false
            embeddingProjection.invalidate()

            let nodeIds = renderStore.visibleNodeIds
            embeddingProjection.loadEmbeddings(for: nodeIds, from: lattice)
            let nodeScale = max(1.0, sqrt(Float(nodeIds.count) / 30.0))
            let spread = max(Float(viewSize.width), Float(viewSize.height)) * 1.2 * nodeScale

            withAnimation(.easeInOut(duration: 0.8)) { transitionProgress = 1.0 }

            Task {
                await embeddingProjection.computeProjection3D(
                    nodeIds: nodeIds, spread: spread,
                    initialPositions: forcePositionSnapshot3D
                )
                var topics: [UUID: String] = [:]
                var projects: [UUID: String] = [:]
                var labels: [UUID: String] = [:]
                for node in renderStore.nodes {
                    topics[node.id] = node.topic
                    projects[node.id] = node.project
                    labels[node.id] = node.label
                }
                embeddingProjection.detectClusters3D(nodeTopics: topics, nodeProjects: projects, nodeLabels: labels)
                projectionTopologyVersion = 0
            }

        case .forceDirected:
            let currentPositions = positions3D
            if !currentPositions.isEmpty {
                simulation3D.setPositions(currentPositions)
            }
            simulation3D.isActive = true
            config.showVoids = false

            withAnimation(.easeInOut(duration: 0.8)) { transitionProgress = 0 }

            Task { @MainActor in
                try? await Task.sleep(for: .seconds(0.9))
                forcePositionSnapshot3D = [:]
            }
        }
    }

    // MARK: - Graph 3D View

    func graph3DView(colorMap: [String: Color]) -> some View {
        Graph3DView(
            layoutMode: config.layoutMode,
            showMascots: config.showMascots,
            selectedNode: $selectedMemoryId,
            semanticClusters3D: embeddingProjection.semanticClusters3D,
            simulation3D: simulation3D,
            embeddingProjection: embeddingProjection,
            camera3DState: camera3DState,
            forcePositionSnapshot3D: forcePositionSnapshot3D,
            transitionProgress: transitionProgress,
            renderStore: renderStore,
            galaxyRegistry: galaxyRegistry,
            cameraProjectTarget: $cameraProjectTarget
        )
        .transaction { $0.animation = nil }  // prevent overlay animations from resizing the 3D view
    }
}
