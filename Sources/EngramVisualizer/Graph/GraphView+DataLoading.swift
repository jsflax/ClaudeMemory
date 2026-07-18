import SwiftUI
import EngramSceneKit
import Lattice
import EngramSceneKit
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
            galaxy.recomputeDerivedData()
        }
    }

//    func recomputeClusters() {
//        for galaxy in galaxyRegistry.galaxies.values {
//            let store = galaxy.renderStore
//            let visibleProjects = Set(store.nodes.map(\.project))
//            var all: [[UUID]] = []
//            for project in visibleProjects {
//                let projectClusters = findMemoryClusters(in: galaxy.lattice, project: project, minClusterSize: 2, neighborLimit: 20).clusters
//                all.append(contentsOf: projectClusters)
//            }
//            store.clusterGroups = all
//        }
//    }

    // MARK: - Data Loading & Observation

    /// Set up galaxies reactively — no async, no temporal dependencies.
    /// `onLatticeAvailable` is idempotent so this is safe to call from both
    /// `onAppear` and `onReceive(syncManager.didConnect)`.
    func loadData() {
        // Propagate initial filter config to registry (onChange only fires on subsequent changes)
        galaxyRegistry.hiddenProjects = config.hiddenProjects
        galaxyRegistry.hiddenRelations = config.hiddenRelations

        // Personal galaxy filter
        var personalFilter: (@Sendable (Memory) -> Bool)?
        if syncManager.isSyncing {
            let syncedProjects = syncManager.syncedProjectNames
            if !syncedProjects.isEmpty {
                personalFilter = { !syncedProjects.contains($0.project) || $0.isPrivate }
            }
        }

        // Register ALL galaxies synchronously FIRST — computeWorldLayout runs with
        // the correct galaxy count, so worldCenter is final before any data arrives.
        let personalRef = lattice.sendableReference
        if galaxyRegistry.galaxies["personal"] == nil {
            let personal = Galaxy(id: "personal", displayName: "Personal",
                                  lattice: personalRef, hierarchyLevel: 0)
            galaxyRegistry.register(personal)
        }

        var syncedRef: LatticeThreadSafeReference?
        if syncManager.isSyncing, let ref = syncManager.actor.syncedLatticeRef {
            syncedRef = ref
            if galaxyRegistry.galaxies["synced"] == nil {
                let synced = Galaxy(id: "synced", displayName: "Synced",
                                    lattice: ref, hierarchyLevel: 0)
                galaxyRegistry.register(synced)
            }
        }

        // NOW start loading — worldCenter is already correct for all galaxies.
        if let personal = galaxyRegistry.galaxies["personal"] {
            Task {
                if let filter = personalFilter { personal.setNodeFilter(filter) }
                await personal.loadData()
                await personal.startObservers()
            }
        }
        if let synced = galaxyRegistry.galaxies["synced"], syncedRef != nil {
            Task {
                await synced.loadData()
                await synced.startObservers()
            }
        }

        galaxyRegistry.syncManager = syncManager
        galaxyRegistry.setupSyncConfigObserver()
    }

    /// Handle late-arriving synced galaxy when daemon connects.
    func handleSyncConnect() {
        if let ref = syncManager.actor.syncedLatticeRef {
            galaxyRegistry.onLatticeAvailable(
                id: "synced", displayName: "Synced", latticeRef: ref
            )
        }
        galaxyRegistry.rebuildPersonalNodeFilter()
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
            galaxy.recomputeDerivedData()
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

            galaxy.simulation3D?.updateGraph(nodeIds: currentIds, edges: edgePairs,
                                              projectForNode: projectMap, topicForNode: topicMap)
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
