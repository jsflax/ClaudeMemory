import SwiftUI
import Lattice
import EngramKit
import AppKit

// MARK: - 2D Layout & Canvas

extension GraphView {

    // MARK: - Embedding Projection

    /// Blended positions: lerp between force layout and t-SNE based on transitionProgress.
    var effectivePositions: [UUID: CGPoint]? {
        guard transitionProgress > 0 else { return nil }  // pure force mode
        let tsne = embeddingProjection.projectedPositions
        guard !tsne.isEmpty else { return nil }

        if transitionProgress >= 1.0 {
            return tsne  // pure embedding mode
        }

        // Lerp between snapshot and t-SNE
        var blended: [UUID: CGPoint] = [:]
        let allIds = Set(forcePositionSnapshot.keys).union(tsne.keys)
        for id in allIds {
            let forcePos = forcePositionSnapshot[id] ?? simulation.positions[id] ?? .zero
            let tsnePos = tsne[id] ?? forcePos
            blended[id] = CGPoint(
                x: forcePos.x + (tsnePos.x - forcePos.x) * transitionProgress,
                y: forcePos.y + (tsnePos.y - forcePos.y) * transitionProgress
            )
        }
        return blended
    }

    // MARK: - Layout Mode Switching (2D)

    /// Helper to set initial 2D positions when switching back from 3D
    func projectedPositions2DFromSwitch(_ positions: [UUID: CGPoint]) {
        forcePositionSnapshot = positions
        withAnimation(.easeInOut(duration: 0.8)) { transitionProgress = 1.0 }
    }

    func switchLayoutMode(to mode: LayoutMode, viewSize: CGSize) {
        guard mode != config.layoutMode else { return }
        config.layoutMode = mode

        if config.dimensionMode == .threeD {
            switchLayoutMode3D(to: mode, viewSize: viewSize)
            return
        }

        switch mode {
        case .embedding:
            // Snapshot current force positions
            forcePositionSnapshot = simulation.positions
            // Pause force simulation
            simulation.isActive = false

            // Clear stale positions from previous run so effectivePositions returns nil
            // (nodes stay at force positions) until new t-SNE emissions arrive.
            embeddingProjection.invalidate()

            // Load embeddings and compute projection
            let nodeIds = renderStore.visibleNodeIds
            embeddingProjection.loadEmbeddings(for: nodeIds, from: lattice)

            let center = simulation.center
            // t-SNE clusters tend to be very tight — use generous spread relative to node count
            let nodeScale = max(1.0, sqrt(CGFloat(nodeIds.count) / 30.0))
            let spread = max(viewSize.width, viewSize.height) * 1.2 * nodeScale

            // Start transition immediately — nodes will drift as t-SNE iterates
            withAnimation(.easeInOut(duration: 0.8)) {
                transitionProgress = 1.0
            }

            Task {
                await embeddingProjection.computeProjection(
                    nodeIds: nodeIds,
                    center: center,
                    spread: spread,
                    initialPositions: forcePositionSnapshot
                )
                // Detect clusters and voids after projection converges
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

                projectionTopologyVersion = 0  // reset — projection is fresh
            }

        case .forceDirected:
            // Inject current t-SNE positions into force sim so it resumes smoothly
            let currentPositions = effectivePositions ?? embeddingProjection.projectedPositions
            if !currentPositions.isEmpty {
                simulation.setPositions(currentPositions)
            }
            simulation.isActive = true
            config.showVoids = false

            // Animate transition back
            withAnimation(.easeInOut(duration: 0.8)) {
                transitionProgress = 0
            }

            // Clean up after animation
            Task { @MainActor in
                try? await Task.sleep(for: .seconds(0.9))
                forcePositionSnapshot = [:]
            }
        }
    }

    // MARK: - Graph Canvas

    func graphCanvas(colorMap: [String: Color]) -> some View {
        GraphCanvas(
            simulation: simulation,
            nodes: renderStore.nodes,
            edges: renderStore.edges,
            hubs: renderStore.hubs,
            glowingNodes: renderStore.glowingNodes,
            newNodes: renderStore.newNodeGlows,
            dyingNodes: renderStore.dyingNodes,
            searchMatchIds: searchMatchIds,
            isSearchActive: isSearchActive,
            viewport: viewport,
            selectedNode: $selectedMemoryId,
            clusters: renderStore.clusterGroups,
            topicGroups: renderStore.topicGroups,
            colorMap: colorMap,
            layoutMode: config.layoutMode,
            overridePositions: effectivePositions,
            knowledgeVoids: config.showVoids ? embeddingProjection.knowledgeVoids : [],
            semanticClusters: embeddingProjection.semanticClusters,
            projectionState: embeddingProjection.state,
            onAnimationTick: { embeddingProjection.tickAnimation() }
        )
    }

    // MARK: - Scroll Monitor

    func installScrollMonitor() {
        scrollMonitor = NSEvent.addLocalMonitorForEvents(matching: .scrollWheel) { event in
            // In 3D mode, Graph3DView handles its own scroll-to-pan
            if config.dimensionMode == .threeD { return event }

            if let window = event.window {
                let loc = event.locationInWindow
                let rightEdge = window.frame.width
                // Let the activity log or detail panel scroll normally
                if loc.x > rightEdge - 240 { return event }
                if selectedMemoryId != nil && loc.x > rightEdge * 0.6 { return event }
            }
            // Two-finger scroll / trackpad → pan (consume event to prevent ScrollView conflicts)
            viewport.offset = CGPoint(
                x: viewport.offset.x + event.scrollingDeltaX,
                y: viewport.offset.y + event.scrollingDeltaY
            )
            return nil
        }
    }

    func removeScrollMonitor() {
        if let monitor = scrollMonitor {
            NSEvent.removeMonitor(monitor)
            scrollMonitor = nil
        }
    }

    func handleSelectionChange(oldId: UUID?, newId: UUID?, viewSize: CGSize) {
        guard let new = newId,
              let worldPos = simulation.positions[new] else { return }

        // Where the node currently sits on screen
        let screenX = worldPos.x * viewport.scale + viewport.offset.x
        let screenY = worldPos.y * viewport.scale + viewport.offset.y

        // Visible region is the left ~60% (right 40% is the detail panel)
        let safeRight = viewSize.width * 0.58
        let margin: CGFloat = 60
        let safeLeft = margin
        let safeTop = margin
        let safeBottom = viewSize.height - margin

        // Only pan if the node is outside the safe region
        var dx: CGFloat = 0, dy: CGFloat = 0
        if screenX > safeRight { dx = safeRight - screenX - margin }
        else if screenX < safeLeft { dx = safeLeft - screenX + margin }
        if screenY > safeBottom { dy = safeBottom - screenY - margin }
        else if screenY < safeTop { dy = safeTop - screenY + margin }

        if dx != 0 || dy != 0 {
            viewport.animateTo(offset: CGPoint(
                x: viewport.offset.x + dx,
                y: viewport.offset.y + dy
            ))
        }
    }
}
