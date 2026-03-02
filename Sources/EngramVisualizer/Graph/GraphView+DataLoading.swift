import SwiftUI
import Lattice
import Combine
import EngramKit
import UserNotifications
import os

private let flushLog = Logger(subsystem: "io.engram.app", category: "FlushInsert")

// MARK: - Data Loading, Observation & Batched Flush

extension GraphView {

    // MARK: - Derived Data (cached, recomputed on structural changes)

    /// Recompute all derived data for all galaxies.
    func recomputeDerivedData() {
        for galaxy in galaxyRegistry.galaxies.values {
            GalaxyDataLoader.recomputeDerivedData(for: galaxy)
        }
    }

    func recomputeClusters() {
        for galaxy in galaxyRegistry.galaxies.values {
            let store = galaxy.renderStore
            let visibleProjects = Set(store.nodes.map(\.project))
            var all: [[UUID]] = []
            for project in visibleProjects {
                let pkClusters = findMemoryClusters(in: galaxy.lattice, project: project, minClusterSize: 2, neighborLimit: 20).clusters
                for pkCluster in pkClusters {
                    let uuidCluster = pkCluster.compactMap { store.pkToGlobalId[$0] }
                    if uuidCluster.count >= 2 { all.append(uuidCluster) }
                }
            }
            store.clusterGroups = all
        }
    }

    // MARK: - Data Loading & Observation

    func loadData() {
        let is3D = config.dimensionMode == .threeD

        // Create the personal galaxy from this view's lattice
        if galaxyRegistry.galaxies["personal"] == nil {
            let galaxy = Galaxy(id: "personal", displayName: "Personal",
                                lattice: lattice, hierarchyLevel: 0)

            // When sync is active, filter personal galaxy to exclude synced non-private memories
            // (those live in the synced galaxy to prevent visual duplication)
            if syncManager.isSyncing {
                let syncedProjects = syncManager.syncedProjectNames
                if !syncedProjects.isEmpty {
                    galaxy.nodeFilter = { @Sendable memory in
                        !syncedProjects.contains(memory.project) || memory.isPrivate
                    }
                }
            }

            galaxyRegistry.register(galaxy)
        }

        if let personal = galaxyRegistry.galaxies["personal"] {
            GalaxyDataLoader.loadData(
                into: personal,
                hiddenProjects: config.hiddenProjects,
                hiddenRelations: config.hiddenRelations,
                timeFilter: debouncedTimeSliderDate,
                is3D: is3D,
                soundEnabled: config.soundEnabled,
                notificationsEnabled: config.notificationsEnabled
            )
        }

        // Create synced galaxy when IPC relay sync is active
        if syncManager.isSyncing,
           let syncedLattice = syncManager.syncedLattice,
           galaxyRegistry.galaxies["synced"] == nil {
            let synced = Galaxy(id: "synced", displayName: "Synced",
                                lattice: syncedLattice, hierarchyLevel: 0)
            // No nodeFilter needed — memory_synced.db already contains only filtered data
            galaxyRegistry.register(synced)

            GalaxyDataLoader.loadData(
                into: synced,
                hiddenProjects: config.hiddenProjects,
                hiddenRelations: config.hiddenRelations,
                timeFilter: debouncedTimeSliderDate,
                is3D: is3D,
                soundEnabled: config.soundEnabled,
                notificationsEnabled: config.notificationsEnabled
            )
        }

        // Observe SyncConfig changes to migrate nodes between galaxies
        galaxyRegistry.syncManager = syncManager
        galaxyRegistry.setupSyncConfigObserver(
            hiddenProjects: config.hiddenProjects,
            hiddenRelations: config.hiddenRelations,
            timeFilter: debouncedTimeSliderDate,
            is3D: is3D,
            soundEnabled: config.soundEnabled,
            notificationsEnabled: config.notificationsEnabled
        )
    }

    // Node/edge change handlers and flush methods are now in GalaxyDataLoader.
    // GraphView delegates all data operations through the Galaxy pipeline.

    func recomputeFilteredData() {
        for galaxy in galaxyRegistry.galaxies.values {
            let store = galaxy.renderStore
            store.nodes = store.allNodes.values.filter { node in
                !config.hiddenProjects.contains(node.project) &&
                (debouncedTimeSliderDate == nil || node.createdAt <= debouncedTimeSliderDate!)
            }
            // Recompute hubs
            var hubs = Set<UUID>()
            for edge in store.allEdges.values where edge.relation == "part_of" {
                hubs.insert(edge.targetId)
            }
            store.hubs = hubs

            let nodeIds = Set(store.nodes.map(\.id))
            store.edges = store.allEdges.values.filter { edge in
                nodeIds.contains(edge.sourceId) && nodeIds.contains(edge.targetId) &&
                !config.hiddenRelations.contains(edge.relation)
            }
            store.filteredEdgeIds = Set(store.edges.map(\.id))
            GalaxyDataLoader.recomputeDerivedData(for: galaxy)
            store.bumpTopology()
        }
    }

    func recomputeHubs() {
        for galaxy in galaxyRegistry.galaxies.values {
            var hubs = Set<UUID>()
            for edge in galaxy.renderStore.allEdges.values where edge.relation == "part_of" {
                hubs.insert(edge.targetId)
            }
            galaxy.renderStore.hubs = hubs
        }
    }

    func rebuildSimulationGraph() {
        for galaxy in galaxyRegistry.galaxies.values {
            let store = galaxy.renderStore
            let filtered = store.nodes
            let currentIds = Set(filtered.map(\.id))
            let edgePairs = store.edges.map { ($0.sourceId, $0.targetId) }
            var projectMap: [UUID: String] = [:]
            var topicMap: [UUID: String] = [:]
            for node in filtered {
                projectMap[node.id] = node.project
                topicMap[node.id] = node.topic
            }

            // 2D simulation only on primary galaxy
            if galaxy.id == "personal" {
                simulation.updateGraph(nodeIds: currentIds, edges: edgePairs, projectForNode: projectMap, topicForNode: topicMap)
            }

            // Update 3D simulation when in 3D mode
            if config.dimensionMode == .threeD {
                galaxy.simulation3D.updateGraph(nodeIds: currentIds, edges: edgePairs,
                                                 projectForNode: projectMap, topicForNode: topicMap)
            }
        }

        // Mark embedding projection stale if topology changed while in embedding mode
        if config.layoutMode == .embedding {
            projectionTopologyVersion &+= 1
        }
    }

    // Flush methods moved to GalaxyDataLoader.

    func sendMemoryNotification(title: String, body: String, id: UUID) {
        if ProcessInfo.processInfo.environment["ENGRAM_TEST_NO_NOTIFY"] != nil { return }
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        content.userInfo = ["memoryId": id]
        let request = UNNotificationRequest(
            identifier: "memory-\(id)-\(UUID().uuidString)",
            content: content,
            trigger: nil
        )
        Task {
            try? await UNUserNotificationCenter.current().add(request)
        }
    }
}
