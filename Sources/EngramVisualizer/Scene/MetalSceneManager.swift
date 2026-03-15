import Metal
import CEngramSceneTypes
import simd
import SwiftUI
import os
import GameController
import EngramSceneKit

private let frameLog = Logger(subsystem: "io.engram.app", category: "MetalScene")

/// Per-frame scene manager for the Metal renderer path.
/// Ports the essential logic from Graph3DScene.renderTick() to write into
/// MetalGraphRenderer's MTLBuffers instead of RealityKit LowLevelMesh.
@MainActor
final class MetalSceneManager {

    let renderer: MetalGraphRenderer
    let camera: CameraController
    let nebulaFog: NebulaFogSystem
    let flowParticles: FlowParticleSystem
    let mascotSharedResources: MascotSharedResources

    /// Spatial audio engine — nil when sound is disabled.
    var spatialAudio: SpatialAudioEngine?

    // External references (set by Graph3DView)
    weak var camera3DState: Camera3DState?
    // Pushed from SwiftUI onChange handlers — never read from @Model in renderTick
    var soundEnabled: Bool = false
    var notificationsEnabled: Bool = false

    /// Galaxy registry — ticks all galaxy simulations and provides merged render data.
    var galaxyRegistry: GalaxyRegistry? {
        didSet { renderer.galaxyRegistryRef = galaxyRegistry }
    }

    // Legacy singular refs — kept for RealityKit path (deprecated) and as fallback.
    // Metal path reads from galaxyRegistry; these are set by Graph3DViewHost for compat.
    weak var simulation3D: ForceSimulation3D?
    weak var embeddingProjection: EmbeddingProjection?
    weak var renderStore: GraphRenderStore?

    // Maintenance mode state
    var isMaintenanceActive: Bool = false
    private var maintenancePulse: Float = 0  // 0..1 lerp for atmosphere effect

    // Data pushed from SwiftUI (only layout/transition state — visual data comes from registry)
    var showMascots: Bool = true
    var layoutMode: LayoutMode = .forceDirected
    var semanticClusters3D: [SemanticCluster3D] = []
    var forcePositionSnapshot3D: [UUID: SIMD3<Float>] = [:]
    var transitionProgress: CGFloat = 0

    // Per-frame cached visual data — refreshed once at the start of renderTick
    // to avoid rebuilding merged dictionaries on every access inside per-node loops.
    private(set) var glowingNodes: [UUID: Date] = [:]
    private(set) var newNodes: [UUID: Date] = [:]
    private(set) var dyingNodes: [UUID: DyingNode] = [:]
    private var topicGroups: [TopicGroupInfo] = []
    private var clusters: [[UUID]] = []
    private var searchMatchIds: Set<UUID> = []
    private var isSearchActive: Bool = false

    // Mutable render state
    var positions: [UUID: SIMD3<Float>] = [:]
    var selectedNode: UUID?
    var selectionCallback: ((UUID?) -> Void)?

    // Hub expansion state
    var expandedHubs: Set<UUID> = []
    var expansionProgress: [UUID: Float] = [:]
    var expansionDirection: [UUID: Bool] = [:]
    var preExpansionPositions: [UUID: SIMD3<Float>] = [:]
    var expandedChildPositions: [UUID: SIMD3<Float>] = [:]
    var pendingHubToggles: [(hubId: UUID, expanding: Bool)] = []

    // Input
    var heldKeys: Set<String> = []

    // Internal state
    private var renderFrameCount: UInt64 = 0
    private var hasCenteredCamera = false
    private var cameraStartTime: Date?
    private var centerTickCount: Int = 0
    private var isDragging = false
    private var lastSelectedNode: UUID?
    private var lastSearchActive = false
    private var lastSearchMatchIds: Set<UUID> = []

    // Reticle (gamepad targeting)
    var reticleTarget: UUID?
    /// Callback for SwiftUI to observe reticle target changes (MetalSceneManager is not @Observable).
    var reticleCallback: ((UUID?) -> Void)?
    /// Callback for SwiftUI to observe teleport label/counter changes (MetalSceneManager is not @Observable).
    var teleportCallback: ((String?, Int) -> Void)?
    // Teleport state
    var teleportLabel: String?
    var teleportCounter: Int = 0
    var teleportProjectIndex: Int = 0
    // Gamepad state
    private var prevButtonA = false
    private var prevButtonB = false
    private var prevLB = false
    private var prevRB = false
    private var prevButtonY = false

    // Color caches
    private var nodeColorCache: [String: SIMD3<Float>] = [:]
    private var edgeColorCache: [String: SIMD3<Float>] = [:]
    private var lastColorMapVersion: UInt64 = 0

    // Node instance staging
    private var instanceArray: [NodeInstance] = []

    // Cached per-node data rebuilt only on topology change
    private var cachedNodeRadii: [UUID: Float] = [:]
    private var cachedNodeProject: [UUID: String] = [:]
    private var lastTopologyNodeCount: Int = 0

    // GPU edge packing state
    private var nodeIndexMap: [UUID: UInt32] = [:]
    private var cachedResolvedEdges: [ResolvedEdge] = []
    private var lastEdgeDescriptorTopologyCount: Int = 0
    private var lastEdgeDescriptorSelection: UUID? = nil
    private var lastEdgeDescriptorSearchActive: Bool = false
    private var lastEdgeDescriptorSearchMatchIds: Set<UUID> = []
    private var edgeDescriptorsDirty: Bool = true
    private var nodePositionsDirty: Bool = true

    // Label atlas state
    var labelAtlasRects: [UUID: (u0: Float, v0: Float, u1: Float, v1: Float)] = [:]
    var projectLabelAtlasRects: [String: (u0: Float, v0: Float, u1: Float, v1: Float)] = [:]
    var galaxyLabelAtlasRects: [String: (u0: Float, v0: Float, u1: Float, v1: Float)] = [:]
    var topicLabelAtlasRects: [String: (u0: Float, v0: Float, u1: Float, v1: Float)] = [:]
    var projectCentroids: [String: (centroid: SIMD3<Float>, radius: Float, maxY: Float, count: Int)] = [:]
    // Cached by packNodeInstances for packLabelInstances — avoids redundant position passes
    private var cachedMinDepth: Float = .greatestFiniteMagnitude
    private var cachedMaxDepth: Float = 0
    var labelAtlasNodeIds: Set<UUID> = []
    var labelAtlasHubIds: Set<UUID> = []
    var labelAtlasProjects: Set<String> = []
    var labelAtlasGalaxyNames: [String] = []
    private var lastAtlasTopicGroupCount: Int = 0
    private var labelAtlasAspectCorrection: Float = 1.0
    private var labelAtlasAllocW: Int = 0
    private var labelAtlasAllocH: Int = 0
    private var labelAtlasRegenFrame: UInt64 = 0
    private var lastAtlasTopologyVersion: UInt64 = 0
    private var isAtlasGenerating = false
    private var pendingAtlasRegen = false

    // View properties
    var renderViewSize: CGSize = .zero

    private let scaleFactor: Float = 1.0 / 200.0
    private let nodeRadius: Float = 0.04
    private let edgeRadius: Float = 0.004

    // Precomputed cylinder trig (6-sided) — avoids 12 sin/cos per edge per frame

    private(set) var renderNodes: [NodeData] = []
    private(set) var renderEdges: [EdgeData] = []
    private(set) var renderHubs: Set<UUID> = []
    private(set) var renderColorMap: [String: Color] = [:]

    // Mascot inspection change tracking — avoids re-packing all nodes when
    // the set of inspected nodes hasn't changed between frames.
    private var lastInspectingNodeIds: Set<UUID> = []
    // Cached mascot per-galaxy data — only rebuilt when project topology changes
    private var cachedMascotColorMap: [String: NSColor] = [:]
    private var lastMascotColorMapVersion: UInt64 = 0

    init?(renderer: MetalGraphRenderer) {
        self.renderer = renderer
        self.camera = CameraController()
        self.nebulaFog = NebulaFogSystem(device: renderer.device)
        self.flowParticles = FlowParticleSystem(device: renderer.device)
        let templateMascot = MascotSystem(device: renderer.device)
        self.mascotSharedResources = templateMascot.extractSharedResources()

        renderer.camera = camera
        renderer.onFrameCallback = { [weak self] dt in
            self?.renderTick(dt: dt)
        }
    }

    // MARK: - Render Tick

    #if ENGRAM_INSTRUMENTATION
    private var lastFrameWallTime: CFAbsoluteTime = 0
    #endif

    func renderTick(dt: Float) {
        renderFrameCount &+= 1
        nodePositionsDirty = true  // will be cleared by packNodeInstances or uploadNodePositions
        let frameStart = CFAbsoluteTimeGetCurrent()
        #if ENGRAM_INSTRUMENTATION
        let wallDt = lastFrameWallTime > 0 ? (frameStart - lastFrameWallTime) * 1000.0 : 0
        lastFrameWallTime = frameStart
        #endif

        // Input + camera
        let preInputSelection = selectedNode
        camera.heldKeys = heldKeys
        camera.pollKeyboard(dt: dt)
        pollGamepad(dt: dt)
        camera.updateCamera(dt: dt)

        if selectedNode != preInputSelection {
            selectionCallback?(selectedNode)
            #if ENGRAM_INSTRUMENTATION
            jitterCallbackFired = true
            #endif
        }

        // Tick ALL galaxy simulations + compute merged positions
        let simStart = CFAbsoluteTimeGetCurrent()
        var didUpdatePositions = false
        #if ENGRAM_INSTRUMENTATION
        var drainMs: Double = 0
        #endif
        if let registry = galaxyRegistry {
            // Build DrainConfig from VisualizerConfig + layout mode each frame.
            let drainConfig = DrainConfig(
                hiddenProjects: registry.hiddenProjects,
                hiddenRelations: registry.hiddenRelations,
                timeFilter: nil,
                is3D: layoutMode == .forceDirected,
                soundEnabled: soundEnabled,
                notificationsEnabled: notificationsEnabled
            )
            registry.currentDrainConfig = drainConfig

            // Drain pending updates from Galaxy actors — atomic, before sim tick.
            // All store + simulation mutations happen here, ensuring consistency.
            #if ENGRAM_INSTRUMENTATION
            let drainStart = CFAbsoluteTimeGetCurrent()
            #endif
            for galaxy in registry.galaxies.values {
                galaxy.drainPendingUpdate(config: drainConfig)
            }
            #if ENGRAM_INSTRUMENTATION
            drainMs = (CFAbsoluteTimeGetCurrent() - drainStart) * 1000.0
            #endif

            var anyActive = false
            for galaxy in registry.galaxies.values {
                // Skip simulation during initial batch loading — nodes are still arriving
                // in batches of 50 and the sim would waste GPU on partial graphs while
                // alpha decays uselessly. wake() fires after all batches complete.
                if !galaxy.isInitialLoad {
                    galaxy.simulation3D.tick()
                }
                galaxy.embeddingProjection.tickAnimation3D()
                if !galaxy.simulation3D.isSettled || galaxy.simulation3D.isLocalWake || !galaxy.embeddingProjection.is3DAnimationSettled {
                    anyActive = true
                }
            }

            #if ENGRAM_INSTRUMENTATION
            // Diagnostic: log per-galaxy sim state once per second
            if renderFrameCount % 60 == 0 && registry.galaxies.count > 1 {
                for galaxy in registry.galaxies.values {
                    let sim = galaxy.simulation3D
                    print("SIM[\(galaxy.id)] frame=\(renderFrameCount) initLoad=\(galaxy.isInitialLoad) settled=\(sim.isSettled) nodes=\(sim.nodeCount) positions=\(sim.positions.count) alpha=\(String(format: "%.4f", sim.alpha)) maxSpeedSq=\(String(format: "%.4f", sim.lastMaxSpeedSq)) atten=\(String(format: "%.4f", sim.smoothedAttenuation)) forceAge=\(sim.forceAge) tickInFlight=\(sim.tickInFlight) framesSinceWake=\(sim.framesSinceWake) center=(\(String(format: "%.0f", sim.center.x)),\(String(format: "%.0f", sim.center.y)),\(String(format: "%.0f", sim.center.z)))")
                }
            }
            #endif
            
            registry.mergeRenderData()

            // Prune expired glow/arrival entries so visualOnlyChanged goes false.
            // Only allocate new dictionaries when there are entries to filter.
            let now = Date()
            let recallTotalDuration: TimeInterval = 1.0 + 1.5 + 2.0  // fadeIn + hold + fadeOut
            let arrivalTotalDuration: TimeInterval = 0.8 + 2.0 + 3.0
            for galaxy in registry.galaxies.values {
                if !galaxy.renderStore.glowingNodes.isEmpty {
                    galaxy.renderStore.glowingNodes = galaxy.renderStore.glowingNodes.filter {
                        now.timeIntervalSince($0.value) < recallTotalDuration
                    }
                }
                if !galaxy.renderStore.newNodeGlows.isEmpty {
                    galaxy.renderStore.newNodeGlows = galaxy.renderStore.newNodeGlows.filter {
                        now.timeIntervalSince($0.value) < arrivalTotalDuration
                    }
                }
            }

            // Cache merged data for this frame — one read per frame,
            // not per-access (avoids rebuilding dictionaries inside per-node loops)
            renderNodes = registry.mergedNodes
            renderEdges = registry.mergedEdges
            renderHubs = registry.mergedHubs
            renderColorMap = registry.mergedColorMap
            glowingNodes = registry.mergedGlowingNodes
            newNodes = registry.mergedNewNodeGlows
            dyingNodes = registry.mergedDyingNodes
            topicGroups = registry.mergedTopicGroups
            clusters = registry.mergedClusterGroups
            searchMatchIds = registry.mergedSearchMatchIds
            isSearchActive = registry.mergedIsSearchActive

            if anyActive {
                // Compute merged positions — each galaxy's positions are already in world space
                // because ForceSimulation3D.center is set to galaxy.worldCenter
                if layoutMode == .forceDirected {
                    registry.updateMergedPositions()
                    let simPositions = registry.mergedPositions
                    // Smooth render positions toward sim positions to prevent jitter
                    // from async force delivery. Adaptive blend: when any sim is
                    // actively converging (high velocity), use aggressive smoothing
                    // to hide force arrival discontinuities. When nearly settled,
                    // use responsive blend so the last few pixels snap into place.
                    if positions.isEmpty {
                        positions = simPositions
                    } else {
                        let maxVelSq = registry.galaxies.values.map { $0.simulation3D.lastMaxSpeedSq }.max() ?? 0
                        let blend: Float = maxVelSq > 4.0 ? 0.10 : (maxVelSq > 0.5 ? 0.18 : 0.35)
                        for (id, simPos) in simPositions {
                            if let curPos = positions[id] {
                                positions[id] = curPos + (simPos - curPos) * blend
                            } else {
                                positions[id] = simPos  // new node — snap immediately
                            }
                        }
                        // Remove positions for nodes that no longer exist
                        for id in positions.keys where simPositions[id] == nil {
                            positions.removeValue(forKey: id)
                        }
                    }
                } else {
                    // Embedding mode: blend per-galaxy force snapshots with t-SNE projections
                    var merged: [UUID: SIMD3<Float>] = [:]
                    for galaxy in registry.galaxies.values {
                        let proj = galaxy.embeddingProjection
                        let sim = galaxy.simulation3D
                        let tsne3D = proj.projectedPositions3D
                        if tsne3D.isEmpty {
                            merged.merge(sim.positions) { _, new in new }
                        } else if transitionProgress >= 1.0 {
                            merged.merge(tsne3D) { _, new in new }
                        } else {
                            let allIds = Set(forcePositionSnapshot3D.keys.filter { registry.nodeToGalaxy[$0] == galaxy.id })
                                .union(tsne3D.keys)
                            for id in allIds {
                                let forcePos = forcePositionSnapshot3D[id] ?? sim.positions[id] ?? .zero
                                let tsnePos = tsne3D[id] ?? forcePos
                                merged[id] = forcePos + (tsnePos - forcePos) * Float(transitionProgress)
                            }
                        }
                    }
                    positions = merged
                }
                didUpdatePositions = true

                if cameraStartTime == nil { cameraStartTime = Date() }
                let elapsed = Date().timeIntervalSince(cameraStartTime!)
                if (!hasCenteredCamera || elapsed < 3.0) && !isDragging && heldKeys.isEmpty {
                    centerTickCount += 1
                    if centerTickCount % 6 == 0 {
                        #if ENGRAM_INSTRUMENTATION
                        if Self.centerLogFile == nil {
                            Self.centerLogFile = fopen("/tmp/center-log.csv", "w")
                            if let f = Self.centerLogFile {
                                fputs("frame,elapsed_s,keys_held,positions,cam_target_x,cam_target_y,cam_target_z,target_pos_x,target_pos_y,target_pos_z\n", f)
                            }
                        }
                        if let f = Self.centerLogFile {
                            let ct = self.camera.cameraTarget
                            let tp = self.camera.targetCameraPos
                            let line = "\(self.renderFrameCount),\(String(format: "%.2f", elapsed)),\(self.heldKeys.count),\(self.positions.count),\(String(format: "%.1f", ct.x)),\(String(format: "%.1f", ct.y)),\(String(format: "%.1f", ct.z)),\(String(format: "%.1f", tp.x)),\(String(format: "%.1f", tp.y)),\(String(format: "%.1f", tp.z))\n"
                            fputs(line, f)
                            fflush(f)
                        }
                        #endif
                        camera.centerOnGraph(positions: positions)
                    }
                    if elapsed >= 3.0 { hasCenteredCamera = true }
                }
            }
        }
        let simMs = (CFAbsoluteTimeGetCurrent() - simStart) * 1000.0

        // Consume hub toggles — route to the correct galaxy's simulation
        if !pendingHubToggles.isEmpty, let registry = galaxyRegistry {
            for toggle in pendingHubToggles {
                if let galaxy = registry.galaxyForNode(toggle.hubId) {
                    let children = galaxy.renderStore.edges.filter { $0.relation == "part_of" && $0.targetId == toggle.hubId }.map(\.sourceId)
                    for childId in children {
                        if toggle.expanding { galaxy.simulation3D.pin(childId) } else { galaxy.simulation3D.unpin(childId) }
                    }
                }
            }
            pendingHubToggles.removeAll()
        }

        // Invalidate color caches when colorMap changes
        let colorVersion = galaxyRegistry?.mergedColorMapVersion ?? 0
        if colorVersion != lastColorMapVersion {
            nodeColorCache.removeAll(keepingCapacity: true)
            edgeColorCache.removeAll(keepingCapacity: true)
            lastColorMapVersion = colorVersion
        }

        // Determine update needs — split geometry changes from visual-only changes
        // to avoid running expensive edge/nebula packing for glow animations.
        let positionsChanged = didUpdatePositions
        let selectionChanged = selectedNode != lastSelectedNode
        let searchChanged = isSearchActive != lastSearchActive || searchMatchIds != lastSearchMatchIds
        let hasExpansions = !expandedHubs.isEmpty
        let cameraMoving = camera.isMoving
        let hasInput = !heldKeys.isEmpty
        let geometryChanged = positionsChanged || selectionChanged || searchChanged || hasExpansions
        // Track which nodes are being inspected — only flag scene dirty when the set changes,
        // not when mascots are just holding steady on the same node.
        var currentInspectingIds = Set<UUID>()
        if let registry = galaxyRegistry {
            for galaxy in registry.galaxies.values {
                if let fleet = galaxy.mascotFleet {
                    for mascot in fleet.mascots.values where mascot.arcaneIntensity > 0.01 {
                        if let targetId = mascot.currentTargetId {
                            currentInspectingIds.insert(targetId)
                        }
                    }
                }
            }
        }
        let inspectionChanged = currentInspectingIds != lastInspectingNodeIds
        lastInspectingNodeIds = currentInspectingIds
        let visualOnlyChanged = !glowingNodes.isEmpty || !newNodes.isEmpty || !dyingNodes.isEmpty || inspectionChanged
        let sceneNeedsUpdate = geometryChanged || visualOnlyChanged

        // Always advance animation time (shaders need it for scan lines, flicker, etc.)
        renderer.animationTime += dt

        // Maintenance pulse: lerp toward target (2s ramp up/down)
        let maintenanceTarget: Float = isMaintenanceActive ? 1.0 : 0.0
        let pulseRate: Float = 0.5 * dt  // 1/2s = 2s full transition
        if maintenancePulse < maintenanceTarget {
            maintenancePulse = min(maintenancePulse + pulseRate, maintenanceTarget)
        } else if maintenancePulse > maintenanceTarget {
            maintenancePulse = max(maintenancePulse - pulseRate, maintenanceTarget)
        }
        renderer.maintenancePulse = maintenancePulse

        // Mascot update — they patrol independently of scene changes.
        // Color map conversion is cached and only rebuilt when topology changes.
        let mascotStart = CFAbsoluteTimeGetCurrent()
        renderer.galaxyRegistryRef = showMascots ? galaxyRegistry : nil
        if showMascots, let registry = galaxyRegistry {
            let nodeByIdForMascot = registry.mergedNodeById

            // Cache NSColor conversion — only rebuild when color map changes
            let cmVersion = registry.mergedColorMapVersion
            if cmVersion != lastMascotColorMapVersion {
                cachedMascotColorMap = renderColorMap.compactMapValues { NSColor($0) }
                lastMascotColorMapVersion = cmVersion
            }

            for galaxy in registry.galaxies.values {
                if galaxy.mascotFleet == nil {
                    galaxy.mascotFleet = MascotFleet(
                        galaxyId: galaxy.id,
                        device: renderer.device,
                        sharedResources: mascotSharedResources,
                        library: renderer.library
                    )
                }

                // Only rebuild nodesByProject when positions have changed
                if didUpdatePositions, let fleet = galaxy.mascotFleet {
                    var nodesByProject: [String: [UUID: SIMD3<Float>]] = [:]
                    for node in galaxy.renderStore.nodes {
                        if let pos = positions[node.id] {
                            nodesByProject[node.project, default: [:]][node.id] = pos
                        }
                    }
                    fleet.cachedNodesByProject = nodesByProject
                    let activeProjects = Set(nodesByProject.keys)
                    fleet.syncProjects(active: activeProjects, colorMap: cachedMascotColorMap)
                }

                // Build node info only for mascots that have targets
                var nodeInfo: [UUID: MascotNodeInfo] = [:]
                for mascot in galaxy.mascotFleet?.mascots.values ?? [:].values {
                    if let targetId = mascot.currentTargetId, let nd = nodeByIdForMascot[targetId] {
                        nodeInfo[targetId] = MascotNodeInfo(
                            content: nd.content, project: nd.project,
                            topic: nd.topic, importance: nd.importance,
                            createdAt: nd.createdAt, lastAccessedAt: nd.lastAccessedAt
                        )
                    }
                }

                galaxy.mascotFleet?.update(
                    dt: dt, camera: camera, positions: positions,
                    nodesByProject: galaxy.mascotFleet?.cachedNodesByProject ?? [:],
                    nodeInfo: nodeInfo,
                    maintenanceActive: isMaintenanceActive
                )
            }
        }
        let mascotMs = (CFAbsoluteTimeGetCurrent() - mascotStart) * 1000.0

        let anyMascotActive = galaxyRegistry?.galaxies.values.contains { galaxy in
            galaxy.mascotFleet?.mascots.values.contains { !$0.isSettled } ?? false
        } ?? false
        let maintenanceTransitioning = maintenancePulse > 0.001 && maintenancePulse < 0.999
        let isActive = sceneNeedsUpdate || cameraMoving || hasInput || anyMascotActive || maintenanceTransitioning

        if !isActive {
            if renderFrameCount > 10 && renderFrameCount % 30 != 0 {
                #if ENGRAM_INSTRUMENTATION
                writeJitterLine(
                    wallDt: wallDt, totalMs: (CFAbsoluteTimeGetCurrent() - frameStart) * 1000.0,
                    simMs: simMs, mascotMs: mascotMs, nodesMs: 0, edgesMs: 0, labelsMs: 0,
                    skipped: true, reason: "idle_skip",
                    positionsChanged: positionsChanged, geometryChanged: geometryChanged,
                    cameraMoving: cameraMoving, anyMascotActive: anyMascotActive
                )
                #endif
                return
            }
        }

        var nodesMs = 0.0, edgesMs = 0.0, nebMs = 0.0, labelsMs = 0.0, flowMs = 0.0

        if sceneNeedsUpdate {
            updateExpansions(dt: dt)

            let t0 = CFAbsoluteTimeGetCurrent()
            packNodeInstances()
            nodesMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0

            // Edge and nebula packing only needed when geometry changed
            // (positions, selection, search), not for glow/arrival visual-only changes.
            if geometryChanged {
                let positionOnly = positionsChanged && !selectionChanged && !searchChanged && !hasExpansions
                // Edges: throttle to every 2nd frame when only positions change.
                // At 60fps, 16ms of edge lag is sub-pixel during smooth simulation drift.
                if !positionOnly || renderFrameCount % 2 == 0 {
                    let t1 = CFAbsoluteTimeGetCurrent()
                    packEdgeVertices()
                    edgesMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000.0
                }

                if !positionOnly || renderFrameCount % 6 == 0 {
                    let t2 = CFAbsoluteTimeGetCurrent()
                    updateNebulae()
                    nebMs = (CFAbsoluteTimeGetCurrent() - t2) * 1000.0
                }
            }

            if selectedNode != nil || renderer.actualFlowParticleCount > 0 {
                let t3 = CFAbsoluteTimeGetCurrent()
                flowParticles.update(
                    dt: dt,
                    selectedNode: selectedNode,
                    edges: renderEdges,
                    positions: positions,
                    expandedHubs: expandedHubs,
                    renderer: renderer
                )
                flowMs = (CFAbsoluteTimeGetCurrent() - t3) * 1000.0
            }

            lastSelectedNode = selectedNode
            lastSearchActive = isSearchActive
            lastSearchMatchIds = searchMatchIds

        }

        if sceneNeedsUpdate || cameraMoving {
            let t4 = CFAbsoluteTimeGetCurrent()
            packLabelInstances()
            labelsMs = (CFAbsoluteTimeGetCurrent() - t4) * 1000.0
        }

        let frameTotalMs = (CFAbsoluteTimeGetCurrent() - frameStart) * 1000.0
        if renderFrameCount % 60 == 1 || frameTotalMs > 20 {
            print("[engram:frame] #\(renderFrameCount) total=\(String(format: "%.1f", frameTotalMs))ms sim=\(String(format: "%.1f", simMs))ms mascot=\(String(format: "%.1f", mascotMs))ms nodes=\(String(format: "%.1f", nodesMs))ms edges=\(String(format: "%.1f", edgesMs))ms labels=\(String(format: "%.1f", labelsMs))ms neb=\(String(format: "%.1f", nebMs))ms nodeCount=\(positions.count) edgeCount=\(renderEdges.count)")
        }

        // Report camera state
        if let camState = camera3DState {
            if camera.azimuth != camState.azimuth { camState.azimuth = camera.azimuth }
            if camera.cameraPosition != camState.position { camState.position = camera.cameraPosition }
            if camera.cameraTarget != camState.target { camState.target = camera.cameraTarget }
            if didUpdatePositions { camState.positions = positions }
        }

        // Spatial audio tick — reads all state computed above
        #if ENGRAM_INSTRUMENTATION
        let audioStart = CFAbsoluteTimeGetCurrent()
        #endif
        spatialAudio?.tick(dt: dt, scene: self)
        #if ENGRAM_INSTRUMENTATION
        let audioMs = (CFAbsoluteTimeGetCurrent() - audioStart) * 1000.0
        #endif

        let totalMs = (CFAbsoluteTimeGetCurrent() - frameStart) * 1000.0
        let nodeCount = positions.count
        let edgeCount = renderEdges.count

        // Write CSV for analysis — every frame when active, every 30th when idle
        #if ENGRAM_INSTRUMENTATION
        let reason: String
        if positionsChanged && geometryChanged { reason = "sim" }
        else if selectionChanged { reason = "select" }
        else if searchChanged { reason = "search" }
        else if visualOnlyChanged {
            if inspectionChanged { reason = "glow_mascot" }
            else if !dyingNodes.isEmpty { reason = "glow_dying" }
            else if !glowingNodes.isEmpty { reason = "glow_recall" }
            else { reason = "glow_arrival" }
        }
        else if cameraMoving { reason = "camera" }
        else { reason = "idle" }

        if metalTimingFile == nil {
            metalTimingFile = fopen("/tmp/metal-frame-timing.csv", "w")
            if let f = metalTimingFile {
                fputs("frame,dt_ms,wall_dt_ms,total_ms,drain_ms,sim_ms,mascot_ms,nodes_ms,edges_ms,neb_ms,labels_ms,flow_ms,node_count,edge_count,reason,audio_ms\n", f)
            }
        }
        if let f = metalTimingFile, (isActive || renderFrameCount % 30 == 0) {
            let line = "\(renderFrameCount),\(String(format: "%.2f", dt * 1000)),\(String(format: "%.2f", wallDt)),\(String(format: "%.2f", totalMs)),\(String(format: "%.2f", drainMs)),\(String(format: "%.2f", simMs)),\(String(format: "%.2f", mascotMs)),\(String(format: "%.2f", nodesMs)),\(String(format: "%.2f", edgesMs)),\(String(format: "%.2f", nebMs)),\(String(format: "%.2f", labelsMs)),\(String(format: "%.2f", flowMs)),\(nodeCount),\(edgeCount),\(reason),\(String(format: "%.2f", audioMs))\n"
            fputs(line, f)
            fflush(f)
        }

        writeJitterLine(
            wallDt: wallDt, totalMs: totalMs,
            simMs: simMs, mascotMs: mascotMs, nodesMs: nodesMs, edgesMs: edgesMs, labelsMs: labelsMs,
            skipped: false, reason: reason,
            positionsChanged: positionsChanged, geometryChanged: geometryChanged,
            cameraMoving: cameraMoving, anyMascotActive: anyMascotActive
        )
        #endif
    }

    #if ENGRAM_INSTRUMENTATION
    private var metalTimingFile: UnsafeMutablePointer<FILE>? = nil
    static var centerLogFile: UnsafeMutablePointer<FILE>? = nil
    private var labelDiagFile: UnsafeMutablePointer<FILE>? = nil
    private var jitterFile: UnsafeMutablePointer<FILE>? = nil
    private var jitterCallbackFired: Bool = false
    private var jitterReticleCallbackFired: Bool = false
    private var jitterTeleportCallbackFired: Bool = false

    private func writeJitterLine(
        wallDt: Double, totalMs: Double,
        simMs: Double, mascotMs: Double, nodesMs: Double, edgesMs: Double, labelsMs: Double,
        skipped: Bool, reason: String,
        positionsChanged: Bool, geometryChanged: Bool,
        cameraMoving: Bool, anyMascotActive: Bool
    ) {
        if jitterFile == nil {
            jitterFile = fopen("/tmp/swiftui-jitter.csv", "w")
            if let f = jitterFile {
                fputs("frame,wall_dt_ms,total_ms,sim_ms,mascot_ms,nodes_ms,edges_ms,labels_ms,skipped,reason,selection_cb,reticle_cb,teleport_cb,positions_changed,geometry_changed,camera_moving,mascot_active,body_eval_count\n", f)
            }
        }
        if let f = jitterFile {
            let bodyCount = Graph3DView.bodyEvalCount
            let line = "\(renderFrameCount),\(String(format: "%.2f", wallDt)),\(String(format: "%.2f", totalMs)),\(String(format: "%.2f", simMs)),\(String(format: "%.2f", mascotMs)),\(String(format: "%.2f", nodesMs)),\(String(format: "%.2f", edgesMs)),\(String(format: "%.2f", labelsMs)),\(skipped ? 1 : 0),\(reason),\(jitterCallbackFired ? 1 : 0),\(jitterReticleCallbackFired ? 1 : 0),\(jitterTeleportCallbackFired ? 1 : 0),\(positionsChanged ? 1 : 0),\(geometryChanged ? 1 : 0),\(cameraMoving ? 1 : 0),\(anyMascotActive ? 1 : 0),\(bodyCount)\n"
            fputs(line, f)
            fflush(f)
        }
        // Reset per-frame callback flags
        jitterCallbackFired = false
        jitterReticleCallbackFired = false
        jitterTeleportCallbackFired = false
    }
    #endif

    // MARK: - Node Instance Packing (GPU-accelerated)

    private func packNodeInstances() {
        let nodes = renderNodes
        guard !nodes.isEmpty else {
            renderer.actualNodeCount = 0
            return
        }

        #if ENGRAM_INSTRUMENTATION
        let packStart = CFAbsoluteTimeGetCurrent()
        #endif

        let colorMap = renderColorMap
        let now = Date()

        // Pre-compute inspecting nodes and birthing elapsed times across all galaxies
        var inspectingIntensity: [UUID: Float] = [:]
        var birthingElapsed: [UUID: Float] = [:]
        var birthingGalaxyMap: [UUID: String] = [:]
        if let registry = galaxyRegistry {
            for galaxy in registry.galaxies.values {
                if let fleet = galaxy.mascotFleet {
                    for mascot in fleet.mascots.values where mascot.arcaneIntensity > 0.01 {
                        if let targetId = mascot.currentTargetId {
                            inspectingIntensity[targetId] = mascot.arcaneIntensity
                        }
                    }
                }
                for (nodeId, start) in galaxy.renderStore.birthingNodes {
                    birthingElapsed[nodeId] = Float(now.timeIntervalSince(start))
                    birthingGalaxyMap[nodeId] = galaxy.id
                }
            }
        }

        // Convert glow/arrival Dates → elapsed Floats for the pure builder
        var glowElapsed: [UUID: Float] = [:]
        for (id, date) in glowingNodes { glowElapsed[id] = Float(now.timeIntervalSince(date)) }
        var arrivalElapsed: [UUID: Float] = [:]
        for (id, date) in newNodes { arrivalElapsed[id] = Float(now.timeIntervalSince(date)) }

        // Pre-convert project colors to SIMD3<Float>
        var projectColors: [String: SIMD3<Float>] = [:]
        for (project, _) in colorMap {
            projectColors[project] = nodeColorFloat3(for: project, colorMap: colorMap)
        }

        #if ENGRAM_INSTRUMENTATION
        let packPrecomputeMs = (CFAbsoluteTimeGetCurrent() - packStart) * 1000.0
        let packLoopStart = CFAbsoluteTimeGetCurrent()
        #endif

        // Build the NodeFrame using the pure builder (testable, no GPU dependency)
        let t_input = CFAbsoluteTimeGetCurrent()
        let hubs = renderHubs
        let nodeCount = nodes.count

        // Rebuild cached data only on topology change
        if nodeCount != lastTopologyNodeCount {
            nodeIndexMap.removeAll(keepingCapacity: true)
            cachedNodeProject.removeAll(keepingCapacity: true)
            cachedNodeRadii.removeAll(keepingCapacity: true)
            for (i, node) in nodes.enumerated() {
                nodeIndexMap[node.id] = UInt32(i)
                cachedNodeProject[node.id] = node.project
                let isHub = hubs.contains(node.id)
                let importance = max(1, node.importance)
                let baseR: Float = isHub ? nodeRadius * 1.6 : nodeRadius
                cachedNodeRadii[node.id] = baseR * (1.0 + Float(importance - 1) * 0.08)
            }
            lastTopologyNodeCount = nodeCount
        }

        // Build parallel arrays for the builder (no UUID dict lookups in hot loop)
        var nodeIds = [UUID](repeating: UUID(), count: nodeCount)
        var nodeProj = [String](repeating: "", count: nodeCount)
        var nodeImp = [Int](repeating: 1, count: nodeCount)
        var nodePos = [SIMD3<Float>](repeating: .zero, count: nodeCount)
        var hasPos = [Bool](repeating: false, count: nodeCount)
        var nodeRad = [Float](repeating: nodeRadius, count: nodeCount)
        var nodeHub = [Bool](repeating: false, count: nodeCount)
        var selectedIdx: Int? = nil
        var searchMatchIdx = Set<Int>()

        for (i, node) in nodes.enumerated() {
            nodeIds[i] = node.id
            nodeProj[i] = node.project
            nodeImp[i] = node.importance
            if let pos = positions[node.id] {
                nodePos[i] = pos
                hasPos[i] = true
            }
            nodeRad[i] = cachedNodeRadii[node.id] ?? nodeRadius
            nodeHub[i] = hubs.contains(node.id)
            if node.id == selectedNode { selectedIdx = i }
            if isSearchActive && searchMatchIds.contains(node.id) { searchMatchIdx.insert(i) }
        }

        let frameInput = SceneNodeFrameInput(
            nodeIds: nodeIds,
            nodeProjects: nodeProj,
            nodeImportances: nodeImp,
            nodePositions: nodePos,
            hasPosition: hasPos,
            nodeRadii: nodeRad,
            nodeIsHub: nodeHub,
            projectColors: projectColors,
            selectedNodeIndex: selectedIdx,
            glowingNodes: glowElapsed,
            newNodes: arrivalElapsed,
            isSearchActive: isSearchActive,
            searchMatchIndices: searchMatchIdx,
            inspectingIntensity: inspectingIntensity,
            birthingElapsed: birthingElapsed,
            nodeRadius: nodeRadius,
            cameraPosition: camera.cameraPosition
        )
        let t_build = CFAbsoluteTimeGetCurrent()
        let nodeFrame = buildSceneNodeFrame(frameInput)
        let t_built = CFAbsoluteTimeGetCurrent()
        let actualCount = nodeFrame.packInputs.count

        // Apply side effect: remove completed births from render stores
        for nodeId in nodeFrame.completedBirths {
            if let galaxyId = birthingGalaxyMap[nodeId] {
                galaxyRegistry?.galaxies[galaxyId]?.renderStore.birthingNodes.removeValue(forKey: nodeId)
            }
        }

        // Upload NodeFrame to GPU buffers
        let t_upload = CFAbsoluteTimeGetCurrent()
        renderer.ensureNodeBuffers(nodeCount: nodeCount)
        renderer.ensureNodePackBuffers(count: nodeCount)
        renderer.ensureNodePositionBuffer(count: nodeCount)

        guard let packBuf = renderer.nodePackInputBuffer,
              let posBuf = renderer.nodePositionBuffer else {
            renderer.actualNodeCount = 0
            return
        }

        let packPtr = packBuf.contents().bindMemory(to: NodePackInput.self, capacity: nodeCount)
        let posPtr = posBuf.contents().bindMemory(to: SIMD3<Float>.self, capacity: nodeCount)

        nodeFrame.packInputs.withUnsafeBufferPointer { ptr in
            guard let base = ptr.baseAddress else { return }
            packPtr.update(from: base, count: actualCount)
        }
        nodeFrame.nodePositions.withUnsafeBufferPointer { ptr in
            guard let base = ptr.baseAddress else { return }
            posPtr.update(from: base, count: actualCount)
        }

        // Finalize centroid + depth caches (consumed by packLabelInstances)
        projectCentroids.removeAll(keepingCapacity: true)
        for (project, data) in nodeFrame.projectCentroids {
            projectCentroids[project] = (centroid: data.centroid, radius: 0, maxY: data.maxY, count: data.count)
        }
        cachedMinDepth = nodeFrame.minDepth
        cachedMaxDepth = nodeFrame.maxDepth

        // Set GPU pack params — the pack_node_instances kernel runs in the compute pass
        if let paramsBuf = renderer.nodePackParamsBuffer {
            let paramsPtr = paramsBuf.contents().bindMemory(to: NodePackParams.self, capacity: 1)
            paramsPtr.pointee = NodePackParams(
                nodeCount: UInt32(actualCount),
                scaleFactor: scaleFactor,
                nodeRadius: nodeRadius,
                animationTime: renderer.animationTime,
                projectCount: UInt32(projectCentroids.count),
                _pad0: 0, _pad1: 0, _pad2: 0
            )
        }

        // Set stamp params for stamp_node_spheres kernel
        if let paramsBuf = renderer.nodeStampParamsBuffer {
            let paramsPtr = paramsBuf.contents().bindMemory(to: StampParams.self, capacity: 1)
            paramsPtr.pointee = StampParams(
                vertsPerNode: UInt32(renderer.vertsPerSphere),
                nodeCount: UInt32(actualCount)
            )
        }

        renderer.actualNodeCount = actualCount

        // Read back point lights from GPU after the compute pass completes (1-frame latency).
        if let lightCountBuf = renderer.pointLightCountBuffer,
           let lightBuf = renderer.pointLightOutputBuffer {
            let count = min(Int(lightCountBuf.contents().bindMemory(to: UInt32.self, capacity: 1).pointee), 16)
            let lights = lightBuf.contents().bindMemory(to: PointLightEntry.self, capacity: count)
            for i in 0..<count {
                setPointLight(index: i, position: lights[i].position, color: lights[i].color,
                              intensity: lights[i].intensity, attenuation: lights[i].attenuation)
            }
            renderer.lightingUniforms.pointLightCount = UInt32(count)
        }

        if !nodeIndexMap.isEmpty {
            nodePositionsDirty = false
            renderer.edgeDataDirty = true
        }

        let t_done = CFAbsoluteTimeGetCurrent()
        let packTotalCheck = (t_done - t_input) * 1000.0
        if packTotalCheck > 10 {
            print("[engram:pack] n=\(actualCount) input=\(String(format: "%.1f", (t_build - t_input) * 1000))ms build=\(String(format: "%.1f", (t_built - t_build) * 1000))ms upload=\(String(format: "%.1f", (t_done - t_upload) * 1000))ms total=\(String(format: "%.1f", packTotalCheck))ms")
        }

        #if ENGRAM_INSTRUMENTATION
        let packLoopMs = (CFAbsoluteTimeGetCurrent() - packLoopStart) * 1000.0
        let packTotalMs = (CFAbsoluteTimeGetCurrent() - packStart) * 1000.0
        if metalTimingFile == nil {
            metalTimingFile = fopen("/tmp/metal-frame-timing.csv", "w")
            if let f = metalTimingFile {
                fputs("frame,dt_ms,wall_dt_ms,total_ms,sim_ms,mascot_ms,nodes_ms,edges_ms,neb_ms,labels_ms,flow_ms,node_count,edge_count,reason\n", f)
            }
        }
        do {
            struct PackTiming { nonisolated(unsafe) static var file: UnsafeMutablePointer<FILE>? = nil }
            if PackTiming.file == nil {
                PackTiming.file = fopen("/tmp/pack-timing.csv", "w")
                if let f = PackTiming.file {
                    fputs("frame,pack_precompute_ms,pack_loop_ms,pack_total_ms,node_count\n", f)
                }
            }
            if let f = PackTiming.file {
                let line = "\(renderFrameCount),\(String(format: "%.2f", packPrecomputeMs)),\(String(format: "%.2f", packLoopMs)),\(String(format: "%.2f", packTotalMs)),\(actualCount)\n"
                fputs(line, f)
                fflush(f)
            }
        }
        #endif
    }

    private func setPointLight(index: Int, position: SIMD3<Float>, color: SIMD3<Float>, intensity: Float, attenuation: Float) {
        let light = PointLightData(
            position: position,
            intensity: intensity,
            color: color,
            attenuationRadius: attenuation
        )
        withUnsafeMutablePointer(to: &renderer.lightingUniforms.pointLights) { tuple in
            let ptr = UnsafeMutableRawPointer(tuple).bindMemory(to: PointLightData.self, capacity: 16)
            ptr[index] = light
        }
    }

    // MARK: - Edge Vertex Packing

    /// Uploads node positions to the GPU position buffer. Called from packNodeInstances()
    /// which already iterates all positions — avoids a redundant dict pass.
    /// Falls back to dict iteration if called standalone (e.g., edge-only update).
    private func uploadNodePositions() {
        // If nodePositionsDirty is false, packNodeInstances already uploaded this frame
        guard nodePositionsDirty else { return }
        let nodeCount = nodeIndexMap.count
        guard nodeCount > 0 else { return }
        renderer.ensureNodePositionBuffer(count: nodeCount)
        guard let posBuf = renderer.nodePositionBuffer else { return }

        let posPtr = posBuf.contents().bindMemory(to: SIMD3<Float>.self, capacity: nodeCount)
        for (id, idx) in nodeIndexMap {
            posPtr[Int(idx)] = positions[id] ?? .zero
        }
        nodePositionsDirty = false
        renderer.edgeDataDirty = true
    }

    /// Rebuilds edge descriptors when topology, selection, or search state changes.
    /// Rebuilds edge descriptors using pre-resolved edges + per-frame state.
    private func rebuildEdgeDescriptors() {
        let interGalaxyConns = galaxyRegistry?.interGalaxyConnections ?? []
        let totalEdgeCapacity = cachedResolvedEdges.count + interGalaxyConns.count
        guard totalEdgeCapacity > 0 else {
            renderer.actualEdgeCount = 0
            renderer.actualEdgeVertexCount = 0
            return
        }

        // Build per-frame input (lightweight — no UUID lookups)
        let selectedIdx = selectedNode.flatMap { nodeIndexMap[$0] }
        var dimmedIndices = Set<UInt32>()
        if isSearchActive {
            for (id, idx) in nodeIndexMap where !searchMatchIds.contains(id) {
                dimmedIndices.insert(idx)
            }
        }

        let edgeInput = SceneEdgeFrameInput(
            resolvedEdges: cachedResolvedEdges,
            selectedIdx: selectedIdx,
            isSearchActive: isSearchActive,
            searchDimmedSourceIdx: dimmedIndices,
            isSemanticMode: layoutMode == .embedding,
            edgeRadius: edgeRadius,
            nodeCount: nodeIndexMap.count,
            interGalaxyConnections: interGalaxyConns.map { (from: $0.from, to: $0.to) }
        )
        let edgeFrame = buildSceneEdgeFrame(edgeInput)

        renderer.ensureEdgeBuffers(edgeCount: totalEdgeCapacity)
        renderer.ensureEdgeDescriptorBuffer(count: totalEdgeCapacity)
        guard let descBuf = renderer.edgeDescriptorBuffer else { return }

        let descriptors = descBuf.contents().bindMemory(to: EdgeDescriptor.self, capacity: totalEdgeCapacity)
        edgeFrame.descriptors.withUnsafeBufferPointer { ptr in
            guard let base = ptr.baseAddress else { return }
            descriptors.update(from: base, count: edgeFrame.count)
        }

        // Upload inter-galaxy virtual positions
        if !edgeFrame.interGalaxyPositions.isEmpty {
            let interGalaxyBase = Int(nodeIndexMap.count)
            let totalPositions = interGalaxyBase + edgeFrame.interGalaxyPositions.count
            renderer.ensureNodePositionBuffer(count: totalPositions)
            if let posBuf = renderer.nodePositionBuffer {
                let posPtr = posBuf.contents().bindMemory(to: SIMD3<Float>.self, capacity: totalPositions)
                for (i, pos) in edgeFrame.interGalaxyPositions.enumerated() {
                    posPtr[interGalaxyBase + i] = pos
                }
            }
        }

        renderer.actualEdgeCount = edgeFrame.count

        if let paramsBuf = renderer.packEdgeParamsBuffer {
            let paramsPtr = paramsBuf.contents().bindMemory(to: PackEdgeParams.self, capacity: 1)
            paramsPtr.pointee = PackEdgeParams(
                edgeCount: UInt32(edgeFrame.count),
                scaleFactor: scaleFactor,
                _pad0: 0, _pad1: 0
            )
        }

        edgeDescriptorsDirty = false
        renderer.edgeDataDirty = true
        lastEdgeDescriptorSelection = selectedNode
        lastEdgeDescriptorSearchActive = isSearchActive
        lastEdgeDescriptorSearchMatchIds = searchMatchIds
        lastEdgeDescriptorTopologyCount = renderNodes.count
    }

    /// Updates edge GPU data: uploads positions every frame, rebuilds descriptors only when needed.
    private func packEdgeVertices() {
        let interGalaxyConns = galaxyRegistry?.interGalaxyConnections ?? []
        guard renderEdges.count + interGalaxyConns.count > 0 else {
            renderer.actualEdgeCount = 0
            renderer.actualEdgeVertexCount = 0
            return
        }

        // Re-resolve edges on topology change (one-time UUID work)
        if renderNodes.count != lastEdgeDescriptorTopologyCount || edgeDescriptorsDirty {
            let colorMap = renderColorMap
            var projectColors: [String: SIMD3<Float>] = [:]
            for project in cachedNodeProject.values {
                if projectColors[project] == nil {
                    projectColors[project] = nodeColorFloat3(for: project, colorMap: colorMap)
                }
            }
            cachedResolvedEdges = resolveEdges(
                edges: renderEdges.map { (sourceId: $0.sourceId, targetId: $0.targetId, relation: $0.relation) },
                nodeIndexMap: nodeIndexMap,
                nodeRadii: cachedNodeRadii,
                nodeProjects: cachedNodeProject,
                projectColors: projectColors,
                defaultEdgeRadius: edgeRadius
            )
        }

        // Check if descriptors need rebuilding (per-frame state changes)
        let descriptorsNeedRebuild = edgeDescriptorsDirty
            || renderNodes.count != lastEdgeDescriptorTopologyCount
            || selectedNode != lastEdgeDescriptorSelection
            || isSearchActive != lastEdgeDescriptorSearchActive
            || searchMatchIds != lastEdgeDescriptorSearchMatchIds

        if descriptorsNeedRebuild {
            rebuildEdgeDescriptors()
        }

        // Upload positions every frame (O(nodes) contiguous write — fast)
        uploadNodePositions()
    }

    // MARK: - Label Instance Packing

    private func packLabelInstances() {
        #if ENGRAM_INSTRUMENTATION
        if labelDiagFile == nil {
            labelDiagFile = fopen("/tmp/label-diag.csv", "w")
            if let f = labelDiagFile {
                fputs("frame,positions,atlasRects,missingRects,instances,atlasRegen,depthRange,minDepth,maxDepth,camX,camY,camZ,projLabels,topicLabels,galaxyLabels,pendingRegen,isGenerating\n", f)
            }
        }
        var diagAtlasRegen = false
        var diagMissingRects = 0
        #endif

        let nodeCount = positions.count
        guard nodeCount > 0 else {
            renderer.actualLabelCount = 0
            return
        }

        let nodes = renderNodes
        let hubs = renderHubs
        let storeVersion = galaxyRegistry?.mergedTopologyVersion ?? 0

        // Regen atlas if needed — O(1) version checks.
        // On topology change, set pendingAtlasRegen and dispatch on the NEXT frame
        // to avoid stacking atlas dispatch overhead on the same frame as the insert.
        let currentTopicGroupCount = topicGroups.count
        let atlasNeedsRegen = renderer.labelAtlasTexture == nil
            || storeVersion != lastAtlasTopologyVersion
            || currentTopicGroupCount != lastAtlasTopicGroupCount
        if atlasNeedsRegen {
            let isFirstAtlas = renderer.labelAtlasTexture == nil
            if isFirstAtlas {
                // First atlas must be synchronous — nothing to show until it's ready
                let currentProjects = Set(nodes.map(\.project))
                let currentGalaxyNames = galaxyRegistry?.galaxies.values.map(\.displayName) ?? []
                generateLabelAtlas(nodes: nodes, hubs: hubs, projects: currentProjects,
                                    galaxyNames: currentGalaxyNames)
                labelAtlasRegenFrame = renderFrameCount
                #if ENGRAM_INSTRUMENTATION
                diagAtlasRegen = true
                #endif
            } else {
                // Defer dispatch to next frame — just record that regen is needed
                pendingAtlasRegen = true
            }
            lastAtlasTopologyVersion = storeVersion
            lastAtlasTopicGroupCount = currentTopicGroupCount
        } else if pendingAtlasRegen && !isAtlasGenerating {
            // Dispatch the deferred atlas regen (runs on the frame after topology change)
            let framesSinceRegen = renderFrameCount &- labelAtlasRegenFrame
            if framesSinceRegen >= 60 {
                let currentProjects = Set(nodes.map(\.project))
                let currentGalaxyNames = galaxyRegistry?.galaxies.values.map(\.displayName) ?? []
                dispatchAtlasRegen(nodes: nodes, hubs: hubs, projects: currentProjects,
                                    galaxyNames: currentGalaxyNames)
                labelAtlasRegenFrame = renderFrameCount
                pendingAtlasRegen = false
                #if ENGRAM_INSTRUMENTATION
                diagAtlasRegen = true
                #endif
            }
            // If throttle not met, keep pendingAtlasRegen = true for the next frame
        }
        guard renderer.labelAtlasTexture != nil else { return }

        // Build label frame using pure builder
        let sf = scaleFactor

        // projectCentroids already computed by packNodeInstances — just build color lookup
        var projectColors: [String: SIMD3<Float>] = [:]
        for project in projectCentroids.keys {
            projectColors[project] = nodeColorFloat3(for: project, colorMap: renderColorMap)
        }

        let nodeById = galaxyRegistry?.mergedNodeById ?? [:]

        let hasExpansions = !expandedChildPositions.isEmpty
        let allPositions: [UUID: SIMD3<Float>]
        if hasExpansions {
            var merged = positions
            for (id, pos) in expandedChildPositions { merged[id] = pos }
            allPositions = merged
        } else {
            allPositions = positions
        }
        var expandedChildren = Set<UUID>()
        if hasExpansions {
            for hubId in expandedHubs {
                for childId in childrenOfHub(hubId) {
                    expandedChildren.insert(childId)
                }
            }
        }

        // Build node label entries from allPositions
        var nodeLabels: [(id: UUID, position: SIMD3<Float>, isHub: Bool,
                          importance: Int, isSelected: Bool,
                          isSearchMatch: Bool, searchDimmed: Bool,
                          isExpandedChild: Bool)] = []
        nodeLabels.reserveCapacity(allPositions.count)
        #if ENGRAM_INSTRUMENTATION
        var diagMissingRects = 0
        #endif
        for (id, pos3D) in allPositions {
            guard let nodeData = nodeById[id] else { continue }
            #if ENGRAM_INSTRUMENTATION
            if labelAtlasRects[id] == nil { diagMissingRects += 1 }
            #endif
            nodeLabels.append((
                id: id, position: pos3D,
                isHub: hubs.contains(id),
                importance: nodeData.importance,
                isSelected: id == selectedNode,
                isSearchMatch: isSearchActive && searchMatchIds.contains(id),
                searchDimmed: isSearchActive && !searchMatchIds.contains(id),
                isExpandedChild: expandedChildren.contains(id)
            ))
        }

        // Convert atlas rects
        var atlasRects: [UUID: AtlasRect] = [:]
        for (id, rect) in labelAtlasRects {
            atlasRects[id] = AtlasRect(u0: rect.u0, v0: rect.v0, u1: rect.u1, v1: rect.v1)
        }
        var projLabelRects: [String: AtlasRect] = [:]
        for (k, rect) in projectLabelAtlasRects {
            projLabelRects[k] = AtlasRect(u0: rect.u0, v0: rect.v0, u1: rect.u1, v1: rect.v1)
        }
        var galLabelRects: [String: AtlasRect] = [:]
        for (k, rect) in galaxyLabelAtlasRects {
            galLabelRects[k] = AtlasRect(u0: rect.u0, v0: rect.v0, u1: rect.u1, v1: rect.v1)
        }
        var topLabelRects: [String: AtlasRect] = [:]
        for (k, rect) in topicLabelAtlasRects {
            topLabelRects[k] = AtlasRect(u0: rect.u0, v0: rect.v0, u1: rect.u1, v1: rect.v1)
        }

        // Build galaxy labels
        var galaxyLabels: [(name: String, center: SIMD3<Float>)] = []
        if let registry = galaxyRegistry, registry.galaxies.count > 1 {
            for galaxy in registry.galaxies.values {
                galaxyLabels.append((name: galaxy.displayName, center: galaxy.worldCenter))
            }
        }

        // Build topic groups with member positions
        var topicGroupsInput: [(topic: String, project: String, memberPositions: [SIMD3<Float>])] = []
        for group in topicGroups {
            guard group.ids.count >= 3 else { continue }
            let memberPositions = group.ids.compactMap { allPositions[$0] }
            guard memberPositions.count >= 3 else { continue }
            topicGroupsInput.append((topic: group.topic, project: group.project, memberPositions: memberPositions))
        }

        // Build centroids in the format the label builder expects
        var labelCentroids: [String: (centroid: SIMD3<Float>, maxY: Float, count: Int)] = [:]
        for (project, data) in projectCentroids {
            labelCentroids[project] = (centroid: data.centroid, maxY: data.maxY, count: data.count)
        }

        let labelInput = SceneLabelFrameInput(
            nodeLabels: nodeLabels,
            atlasRects: atlasRects,
            projectCentroids: labelCentroids,
            projectLabelRects: projLabelRects,
            projectColors: projectColors,
            galaxyLabels: galaxyLabels,
            galaxyLabelRects: galLabelRects,
            topicGroups: topicGroupsInput,
            topicLabelRects: topLabelRects,
            scaleFactor: sf,
            nodeRadius: nodeRadius,
            aspectCorrection: labelAtlasAspectCorrection
        )
        let labelFrame = buildSceneLabelFrame(labelInput)
        let instances = labelFrame.instances
        let actualLabelCount = instances.count

        renderer.ensureLabelBuffers(labelCount: actualLabelCount)
        if let instanceBuf = renderer.labelInstanceBuffer {
            instances.withUnsafeBufferPointer { ptr in
                guard let base = ptr.baseAddress else { return }
                instanceBuf.contents().copyMemory(
                    from: base,
                    byteCount: actualLabelCount * MemoryLayout<LabelInstance>.stride
                )
            }
        }

        // Set stamp params
        let depthRange = max(50.0, cachedMaxDepth - cachedMinDepth)
        let scaledCamPos = camera.cameraPosition * sf
        if let paramsBuf = renderer.labelStampParamsBuffer {
            let paramsPtr = paramsBuf.contents().bindMemory(to: LabelStampParams.self, capacity: 1)
            paramsPtr.pointee = LabelStampParams(
                cameraPos: scaledCamPos,
                minDepth: cachedMinDepth * sf,
                depthRange: depthRange * sf,
                labelCount: UInt32(actualLabelCount),
                _pad0: 0, _pad1: 0
            )
        }

        renderer.actualLabelCount = actualLabelCount

        #if ENGRAM_INSTRUMENTATION
        if let f = labelDiagFile {
            let line = String(format: "%llu,%d,%d,%d,%d,%d,%.2f,%.2f,%.2f,%.3f,%.3f,%.3f,%d,%d,%d,%d,%d\n",
                renderFrameCount,
                allPositions.count,
                labelAtlasRects.count,
                diagMissingRects,
                actualLabelCount,
                diagAtlasRegen ? 1 : 0,
                depthRange, cachedMinDepth, cachedMaxDepth,
                camera.cameraPosition.x, camera.cameraPosition.y, camera.cameraPosition.z,
                projectCentroids.count,
                topicGroups.count,
                (galaxyRegistry?.galaxies.count ?? 0) > 1 ? galaxyRegistry!.galaxies.count : 0,
                pendingAtlasRegen ? 1 : 0,
                isAtlasGenerating ? 1 : 0)
            fputs(line, f)
            if renderFrameCount % 10 == 0 { fflush(f) }
        }
        #endif
    }

    // MARK: - Label Atlas Generation (MTLTexture)

    private struct AtlasLabelEntry: Sendable {
        let id: UUID; let label: String; let isHub: Bool
    }

    private struct AtlasResult: @unchecked Sendable {
        let texture: MTLTexture
        let nodeRects: [UUID: (u0: Float, v0: Float, u1: Float, v1: Float)]
        let projectRects: [String: (u0: Float, v0: Float, u1: Float, v1: Float)]
        let galaxyRects: [String: (u0: Float, v0: Float, u1: Float, v1: Float)]
        let topicRects: [String: (u0: Float, v0: Float, u1: Float, v1: Float)]
        let aspectCorrection: Float
        let allocW: Int; let allocH: Int
    }

    /// Synchronous atlas generation (used for the first atlas only).
    private func generateLabelAtlas(nodes: [NodeData], hubs: Set<UUID>, projects: Set<String>,
                                     galaxyNames: [String]) {
        let entries = nodes.map { AtlasLabelEntry(id: $0.id, label: $0.label, isHub: hubs.contains($0.id)) }
        let topicLabels = Array(Set(topicGroups.map(\.topic))).sorted()
        let monoMedium: CTFont = NSFont.monospacedSystemFont(ofSize: 28, weight: .medium) as CTFont
        let monoBold: CTFont = NSFont.monospacedSystemFont(ofSize: 28, weight: .bold) as CTFont
        let projCTFont: CTFont = NSFont.systemFont(ofSize: 40, weight: .bold) as CTFont
        guard let result = Self.renderLabelAtlas(
            entries: entries, sortedProjects: projects.sorted(), galaxyNames: galaxyNames,
            topicLabels: topicLabels,
            device: renderer.device,
            monoMedium: monoMedium, monoBold: monoBold, projFont: projCTFont
        ) else { return }
        applyAtlasResult(result, nodeIds: Set(nodes.map(\.id)), hubs: hubs, projects: projects,
                          galaxyNames: galaxyNames)
        frameLog.info("[labelAtlas] \(result.nodeRects.count) node + \(result.projectRects.count) project + \(result.galaxyRects.count) galaxy labels, \(result.allocW/2)x\(result.allocH/2)")
    }

    /// Dispatch label atlas regeneration to a background thread. Old atlas stays visible until complete.
    private func dispatchAtlasRegen(nodes: [NodeData], hubs: Set<UUID>, projects: Set<String>,
                                     galaxyNames: [String]) {
        isAtlasGenerating = true
        // Minimal main-thread prep: only sorts + font creation (~0.3ms).
        // O(n) entries mapping and Set creation are deferred to the background thread.
        let sortedProjects = projects.sorted()
        let topicLabels = Array(Set(topicGroups.map(\.topic))).sorted()
        let device = renderer.device
        let capturedNodes = nodes
        let capturedHubs = hubs
        let capturedProjects = projects

        // Create fonts on the main thread (NSFont is safe here), then pass CTFont refs
        // to the background. CTFont is immutable and thread-safe; this avoids NSFont
        // shared cache lock contention from the background thread.
        let monoMedium: CTFont = NSFont.monospacedSystemFont(ofSize: 28, weight: .medium) as CTFont
        let monoBold: CTFont = NSFont.monospacedSystemFont(ofSize: 28, weight: .bold) as CTFont
        let projCTFont: CTFont = NSFont.systemFont(ofSize: 40, weight: .bold) as CTFont

        #if ENGRAM_INSTRUMENTATION
        let dispatchFrame = renderFrameCount
        #endif

        Task.detached(priority: .userInitiated) { [weak self] in
            #if ENGRAM_INSTRUMENTATION
            let bgStart = CFAbsoluteTimeGetCurrent()
            #endif
            // O(n) work done on background thread to reduce insert-frame jitter
            let entries = capturedNodes.map { AtlasLabelEntry(id: $0.id, label: $0.label, isHub: capturedHubs.contains($0.id)) }
            let capturedNodeIds = Set(capturedNodes.map(\.id))

            let result = Self.renderLabelAtlas(
                entries: entries, sortedProjects: sortedProjects, galaxyNames: galaxyNames,
                topicLabels: topicLabels,
                device: device,
                monoMedium: monoMedium, monoBold: monoBold, projFont: projCTFont
            )
            #if ENGRAM_INSTRUMENTATION
            let bgMs = (CFAbsoluteTimeGetCurrent() - bgStart) * 1000.0
            #endif
            await MainActor.run { [weak self] in
                #if ENGRAM_INSTRUMENTATION
                let applyStart = CFAbsoluteTimeGetCurrent()
                #endif
                guard let self else { return }
                if let result {
                    self.applyAtlasResult(result, nodeIds: capturedNodeIds, hubs: capturedHubs,
                                           projects: capturedProjects, galaxyNames: galaxyNames)
                }
                self.isAtlasGenerating = false
                #if ENGRAM_INSTRUMENTATION
                let applyMs = (CFAbsoluteTimeGetCurrent() - applyStart) * 1000.0
                let msg = "dispatched=frame\(dispatchFrame) bg=\(String(format: "%.1f", bgMs))ms apply=\(String(format: "%.1f", applyMs))ms currentFrame=\(self.renderFrameCount)\n"
                if let data = msg.data(using: .utf8) {
                    let url = URL(fileURLWithPath: "/tmp/atlas-timing.log")
                    if let fh = try? FileHandle(forWritingTo: url) {
                        fh.seekToEndOfFile(); fh.write(data); fh.closeFile()
                    } else { try? data.write(to: url) }
                }
                #endif
            }
        }
    }

    private func applyAtlasResult(_ result: AtlasResult, nodeIds: Set<UUID>, hubs: Set<UUID>,
                                    projects: Set<String>, galaxyNames: [String]) {
        renderer.labelAtlasTexture = result.texture
        labelAtlasRects = result.nodeRects
        projectLabelAtlasRects = result.projectRects
        galaxyLabelAtlasRects = result.galaxyRects
        topicLabelAtlasRects = result.topicRects
        labelAtlasAspectCorrection = result.aspectCorrection
        labelAtlasAllocW = result.allocW
        labelAtlasAllocH = result.allocH
        labelAtlasNodeIds = nodeIds
        labelAtlasHubIds = hubs
        labelAtlasProjects = projects
        labelAtlasGalaxyNames = galaxyNames
    }

    /// Pure rendering work — can run on any thread.
    /// Uses CoreText (CTLine) + CGContext directly — no AppKit (NSGraphicsContext/NSString.draw)
    /// to avoid internal AppKit lock contention with the main thread.
    /// Fonts are pre-created on the main thread and passed in (CTFont is immutable, thread-safe).
    nonisolated private static func renderLabelAtlas(
        entries: [AtlasLabelEntry], sortedProjects: [String], galaxyNames: [String],
        topicLabels: [String],
        device: MTLDevice,
        monoMedium: CTFont, monoBold: CTFont, projFont: CTFont
    ) -> AtlasResult? {
        let atlasW = 4096
        let padding: CGFloat = 4

        // --- Pure CoreText measurement + line creation (no AppKit calls) ---
        let white = CGColor(red: 1, green: 1, blue: 1, alpha: 1)

        func makeLine(_ text: String, font: CTFont) -> (line: CTLine, width: CGFloat, ascent: CGFloat, descent: CGFloat, leading: CGFloat) {
            let attrs: [CFString: Any] = [
                kCTFontAttributeName: font,
                kCTForegroundColorAttributeName: white
            ]
            let attrStr = CFAttributedStringCreate(nil, text as CFString, attrs as CFDictionary)!
            let line = CTLineCreateWithAttributedString(attrStr)
            var ascent: CGFloat = 0, descent: CGFloat = 0, leading: CGFloat = 0
            let width = CGFloat(CTLineGetTypographicBounds(line, &ascent, &descent, &leading))
            return (line, width, ascent, descent, leading)
        }

        struct LabelSize { let width: CGFloat; let height: CGFloat }
        struct LabelInfo { let line: CTLine; let size: LabelSize; let ascent: CGFloat }

        // Measure node labels
        var nodeInfos: [LabelInfo] = []
        nodeInfos.reserveCapacity(entries.count)
        for entry in entries {
            let font = entry.isHub ? monoBold : monoMedium
            let m = makeLine(entry.label, font: font)
            let w = ceil(m.width) + padding * 2
            let h = ceil(m.ascent + m.descent + m.leading) + padding
            nodeInfos.append(LabelInfo(line: m.line, size: LabelSize(width: w, height: h), ascent: m.ascent))
        }

        // Measure project labels
        var projInfos: [LabelInfo] = []
        for project in sortedProjects {
            let m = makeLine(project, font: projFont)
            let w = ceil(m.width) + padding * 2
            let h = ceil(m.ascent + m.descent + m.leading) + padding
            projInfos.append(LabelInfo(line: m.line, size: LabelSize(width: w, height: h), ascent: m.ascent))
        }

        // Measure galaxy labels (larger than project labels)
        var galaxyInfos: [LabelInfo] = []
        for name in galaxyNames {
            let m = makeLine(name.uppercased(), font: projFont)
            let w = ceil(m.width) + padding * 2
            let h = ceil(m.ascent + m.descent + m.leading) + padding
            galaxyInfos.append(LabelInfo(line: m.line, size: LabelSize(width: w, height: h), ascent: m.ascent))
        }

        // Measure topic labels (force-directed topic group labels)
        let topicFont: CTFont = projFont
        var topicInfos: [LabelInfo] = []
        for label in topicLabels {
            let m = makeLine(label, font: topicFont)
            let w = ceil(m.width) + padding * 2
            let h = ceil(m.ascent + m.descent + m.leading) + padding
            topicInfos.append(LabelInfo(line: m.line, size: LabelSize(width: w, height: h), ascent: m.ascent))
        }

        // Simulate packing
        var cursorX: CGFloat = 2
        var cursorY: CGFloat = 0
        var rowHeight: CGFloat = 0
        for info in nodeInfos {
            let s = info.size
            if cursorX + s.width > CGFloat(atlasW) { cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0 }
            cursorX += s.width + padding; rowHeight = max(rowHeight, s.height)
        }
        cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0
        for info in projInfos {
            let s = info.size
            if cursorX + s.width > CGFloat(atlasW) { cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0 }
            cursorX += s.width + padding; rowHeight = max(rowHeight, s.height)
        }
        cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0
        for info in galaxyInfos {
            let s = info.size
            if cursorX + s.width > CGFloat(atlasW) { cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0 }
            cursorX += s.width + padding; rowHeight = max(rowHeight, s.height)
        }
        cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0
        for info in topicInfos {
            let s = info.size
            if cursorX + s.width > CGFloat(atlasW) { cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0 }
            cursorX += s.width + padding; rowHeight = max(rowHeight, s.height)
        }
        let neededH = Int(ceil(cursorY + rowHeight + padding))

        // Fall back to 1x scale if 2x would exceed Metal's max texture dimension (16384).
        // Labels that still don't fit are gracefully skipped (break guards in drawing loops).
        let maxDim = 16384
        let scale = (max(512, neededH + 32) * 2 <= maxDim) ? 2 : 1
        let allocW = atlasW * scale
        let allocH = min(max(512, neededH + 32) * scale, maxDim)

        let uvAtlasW = allocW / scale
        let uvAtlasH = allocH / scale
        let aspectCorrection = Float(uvAtlasW) / (2.0 * Float(uvAtlasH))

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil, width: allocW, height: allocH,
            bitsPerComponent: 8, bytesPerRow: allocW * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }

        ctx.translateBy(x: 0, y: CGFloat(allocH))
        ctx.scaleBy(x: CGFloat(scale), y: CGFloat(-scale))

        // Draw node labels using CTLineDraw (no AppKit locks)
        var rects: [UUID: (u0: Float, v0: Float, u1: Float, v1: Float)] = [:]
        cursorX = 2; cursorY = 0; rowHeight = 0

        for (i, entry) in entries.enumerated() {
            let info = nodeInfos[i]
            let s = info.size
            if cursorX + s.width > CGFloat(atlasW) { cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0 }
            if cursorY + s.height > CGFloat(uvAtlasH) { break }

            // CTLineDraw renders glyphs upward from baseline, but the CGContext has
            // a negative y-scale (flipped). Locally unflip at the draw point so text
            // renders right-side up, then restore.
            ctx.saveGState()
            ctx.translateBy(x: cursorX + padding, y: cursorY + padding * 0.5 + info.ascent)
            ctx.scaleBy(x: 1, y: -1)
            ctx.textPosition = .zero
            CTLineDraw(info.line, ctx)
            ctx.restoreGState()

            rects[entry.id] = (
                Float(cursorX) / Float(uvAtlasW), Float(cursorY) / Float(uvAtlasH),
                Float(cursorX + s.width) / Float(uvAtlasW), Float(cursorY + s.height) / Float(uvAtlasH)
            )
            cursorX += s.width + padding; rowHeight = max(rowHeight, s.height)
        }

        // Draw project labels using CTLineDraw (no AppKit locks)
        var projRects: [String: (u0: Float, v0: Float, u1: Float, v1: Float)] = [:]
        cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0

        for (i, project) in sortedProjects.enumerated() {
            let info = projInfos[i]
            let s = info.size
            if cursorX + s.width > CGFloat(atlasW) { cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0 }
            if cursorY + s.height > CGFloat(uvAtlasH) { break }

            ctx.saveGState()
            ctx.translateBy(x: cursorX + padding, y: cursorY + padding * 0.5 + info.ascent)
            ctx.scaleBy(x: 1, y: -1)
            ctx.textPosition = .zero
            CTLineDraw(info.line, ctx)
            ctx.restoreGState()

            projRects[project] = (
                Float(cursorX) / Float(uvAtlasW), Float(cursorY) / Float(uvAtlasH),
                Float(cursorX + s.width) / Float(uvAtlasW), Float(cursorY + s.height) / Float(uvAtlasH)
            )
            cursorX += s.width + padding; rowHeight = max(rowHeight, s.height)
        }

        // Draw galaxy labels using CTLineDraw
        var galaxyRects: [String: (u0: Float, v0: Float, u1: Float, v1: Float)] = [:]
        cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0

        for (i, name) in galaxyNames.enumerated() {
            let info = galaxyInfos[i]
            let s = info.size
            if cursorX + s.width > CGFloat(atlasW) { cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0 }
            if cursorY + s.height > CGFloat(uvAtlasH) { break }

            ctx.saveGState()
            ctx.translateBy(x: cursorX + padding, y: cursorY + padding * 0.5 + info.ascent)
            ctx.scaleBy(x: 1, y: -1)
            ctx.textPosition = .zero
            CTLineDraw(info.line, ctx)
            ctx.restoreGState()

            galaxyRects[name] = (
                Float(cursorX) / Float(uvAtlasW), Float(cursorY) / Float(uvAtlasH),
                Float(cursorX + s.width) / Float(uvAtlasW), Float(cursorY + s.height) / Float(uvAtlasH)
            )
            cursorX += s.width + padding; rowHeight = max(rowHeight, s.height)
        }

        // Draw topic labels using CTLineDraw
        var topicRects: [String: (u0: Float, v0: Float, u1: Float, v1: Float)] = [:]
        cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0

        for (i, label) in topicLabels.enumerated() {
            let info = topicInfos[i]
            let s = info.size
            if cursorX + s.width > CGFloat(atlasW) { cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0 }
            if cursorY + s.height > CGFloat(uvAtlasH) { break }

            ctx.saveGState()
            ctx.translateBy(x: cursorX + padding, y: cursorY + padding * 0.5 + info.ascent)
            ctx.scaleBy(x: 1, y: -1)
            ctx.textPosition = .zero
            CTLineDraw(info.line, ctx)
            ctx.restoreGState()

            topicRects[label] = (
                Float(cursorX) / Float(uvAtlasW), Float(cursorY) / Float(uvAtlasH),
                Float(cursorX + s.width) / Float(uvAtlasW), Float(cursorY + s.height) / Float(uvAtlasH)
            )
            cursorX += s.width + padding; rowHeight = max(rowHeight, s.height)
        }

        // Create MTLTexture from CGContext
        guard let pixelData = ctx.data else { return nil }
        let bytesPerRow = allocW * 4

        // Validate dimensions before Metal texture creation (max 16384 on Apple Silicon)
        guard allocW > 0, allocH > 0, allocW <= 16384, allocH <= 16384 else {
            frameLog.error("[labelAtlas] invalid texture dimensions: \(allocW)x\(allocH)")
            return nil
        }

        let texDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: allocW,
            height: allocH,
            mipmapped: false
        )
        texDesc.usage = [.shaderRead]
        texDesc.storageMode = .shared

        guard let texture = device.makeTexture(descriptor: texDesc) else { return nil }
        texture.replace(
            region: MTLRegion(origin: MTLOrigin(), size: MTLSize(width: allocW, height: allocH, depth: 1)),
            mipmapLevel: 0,
            withBytes: pixelData,
            bytesPerRow: bytesPerRow
        )

        return AtlasResult(
            texture: texture, nodeRects: rects, projectRects: projRects,
            galaxyRects: galaxyRects, topicRects: topicRects,
            aspectCorrection: aspectCorrection, allocW: allocW, allocH: allocH
        )
    }

    // MARK: - Nebulae

    // Cached nebula color conversion — only rebuild when color map changes
    private var cachedNebulaColors: [String: SIMD4<Float>] = [:]
    private var lastNebulaColorMapVersion: UInt64 = 0

    private func updateNebulae() {
        // Rebuild nebula color cache when color map changes (avoids per-frame NSColor conversion)
        let colorVersion = galaxyRegistry?.mergedColorMapVersion ?? 0
        if colorVersion != lastNebulaColorMapVersion {
            cachedNebulaColors.removeAll(keepingCapacity: true)
            for (project, color) in renderColorMap {
                let nsColor = NSColor(color).usingColorSpace(.sRGB) ?? NSColor(color)
                var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
                nsColor.getRed(&r, green: &g, blue: &b, alpha: &a)
                cachedNebulaColors[project] = SIMD4<Float>(
                    Float(min(1.0, r * 1.3 + 0.1)),
                    Float(min(1.0, g * 1.3 + 0.1)),
                    Float(min(1.0, b * 1.3 + 0.1)),
                    0.25
                )
            }
            lastNebulaColorMapVersion = colorVersion
        }

        if layoutMode == .embedding {
            // Embedding mode: use semantic clusters (fall back to original CPU path)
            let groups = nebulaFog.nebulaGroupsForCurrentMode(
                layoutMode: layoutMode, positions: positions,
                nodes: renderNodes, semanticClusters3D: semanticClusters3D
            )
            nebulaFog.updateRenderer(renderer: renderer, groups: groups, colorMap: renderColorMap)
            renderer.nebulaGroupCount = 0  // disable GPU nebula path
            return
        }

        // Force-directed mode: use projectCentroids from packNodeInstances (saves O(N) recomputation).
        // Build NebulaGroupInput[] using the pure builder.
        var nebulaCentroids: [String: (centroid: SIMD3<Float>, maxY: Float, count: Int)] = [:]
        for (project, data) in projectCentroids {
            nebulaCentroids[project] = (centroid: data.centroid, maxY: data.maxY, count: data.count)
        }
        let groupInputs = buildNebulaGroups(
            projectCentroids: nebulaCentroids,
            projectColors: cachedNebulaColors,
            scaleFactor: scaleFactor
        )

        let groupCount = groupInputs.count
        guard groupCount > 0 else {
            renderer.actualNebulaVertexCount = 0
            renderer.nebulaGroupCount = 0
            return
        }

        // Ensure GPU buffers
        renderer.ensureNebulaGroupInputBuffer(count: groupCount)
        let quadCount = groupCount * 3  // 3 quads per group
        if renderer.nebulaVertexCapacity < quadCount {
            let cap = max(quadCount * 2, 64)
            renderer.nebulaVertexBuffer = renderer.device.makeBuffer(
                length: cap * 4 * MemoryLayout<NebulaQuadVertex>.stride,
                options: .storageModeShared
            )
            // Index buffer for quads
            let indexCount = cap * 6
            renderer.nebulaIndexBuffer = renderer.device.makeBuffer(
                length: indexCount * MemoryLayout<UInt32>.stride,
                options: .storageModeShared
            )
            if let buf = renderer.nebulaIndexBuffer {
                let indices = buf.contents().bindMemory(to: UInt32.self, capacity: indexCount)
                for i in 0..<cap {
                    let vBase = UInt32(i * 4)
                    let iBase = i * 6
                    indices[iBase]     = vBase
                    indices[iBase + 1] = vBase + 2
                    indices[iBase + 2] = vBase + 1
                    indices[iBase + 3] = vBase + 1
                    indices[iBase + 4] = vBase + 2
                    indices[iBase + 5] = vBase + 3
                }
            }
            renderer.nebulaVertexCapacity = cap
        }

        // Upload group inputs to GPU buffer
        if let inputBuf = renderer.nebulaGroupInputBuffer {
            groupInputs.withUnsafeBufferPointer { ptr in
                guard let base = ptr.baseAddress else { return }
                inputBuf.contents().copyMemory(
                    from: base,
                    byteCount: groupCount * MemoryLayout<NebulaGroupInput>.stride
                )
            }
        }

        // Set pack params
        if let paramsBuf = renderer.nebulaPackParamsBuffer {
            let paramsPtr = paramsBuf.contents().bindMemory(to: NebulaPackParams.self, capacity: 1)
            paramsPtr.pointee = NebulaPackParams(
                groupCount: UInt32(groupCount),
                quadsPerGroup: 3,
                _pad0: 0, _pad1: 0
            )
        }

        renderer.nebulaGroupCount = groupCount
        renderer.actualNebulaVertexCount = groupCount * 3 * 4  // 3 quads × 4 vertices each
    }

    // MARK: - Hub Expansion

    func childrenOfHub(_ hubId: UUID) -> [UUID] {
        renderEdges.filter { $0.relation == "part_of" && $0.targetId == hubId }.map(\.sourceId)
    }

    func computeOrbitPositions(hubId: UUID, children: [UUID]) -> [UUID: SIMD3<Float>] {
        guard let hubPos = positions[hubId] else { return [:] }
        let radius: Float = 80
        var result: [UUID: SIMD3<Float>] = [:]
        let n = children.count
        guard n > 0 else { return result }
        let goldenRatio: Float = (1 + sqrt(5)) / 2
        for (idx, childId) in children.enumerated() {
            let i = Float(idx)
            let theta = acos(1 - 2 * (i + 0.5) / Float(n))
            let phi = 2 * Float.pi * i / goldenRatio
            let x = radius * sin(theta) * cos(phi)
            let y = radius * sin(theta) * sin(phi)
            let z = radius * cos(theta)
            result[childId] = hubPos + SIMD3(x, y, z)
        }
        return result
    }

    func toggleHubExpansion(hubId: UUID) {
        if expandedHubs.contains(hubId) {
            expansionDirection[hubId] = false
        } else {
            expandedHubs.insert(hubId)
            let children = childrenOfHub(hubId)
            for childId in children {
                preExpansionPositions[childId] = positions[childId] ?? .zero
            }
            expansionProgress[hubId] = 0
            expansionDirection[hubId] = true
        }
    }

    private func updateExpansions(dt: Float) {
        var toRemove: [UUID] = []
        var allExpandedPositions: [UUID: SIMD3<Float>] = [:]

        for hubId in expandedHubs {
            let expanding = expansionDirection[hubId] ?? true
            var progress = expansionProgress[hubId] ?? 0

            if expanding { progress = min(1.0, progress + dt * 3.0) }
            else { progress = max(0.0, progress - dt * 3.0) }
            expansionProgress[hubId] = progress

            let t = progress * progress * (3 - 2 * progress)
            let children = childrenOfHub(hubId)
            let orbitPositions = computeOrbitPositions(hubId: hubId, children: children)

            for childId in children {
                let startPos = preExpansionPositions[childId] ?? positions[childId] ?? .zero
                let endPos = orbitPositions[childId] ?? startPos
                let lerpedPos = startPos + (endPos - startPos) * t
                positions[childId] = lerpedPos
                allExpandedPositions[childId] = lerpedPos
            }

            if !expanding && progress <= 0 {
                toRemove.append(hubId)
                for childId in children {
                    if let original = preExpansionPositions[childId] {
                        positions[childId] = original
                    }
                    preExpansionPositions.removeValue(forKey: childId)
                }
            }
        }

        expandedChildPositions = allExpandedPositions
        for hubId in toRemove {
            expandedHubs.remove(hubId)
            expansionProgress.removeValue(forKey: hubId)
            expansionDirection.removeValue(forKey: hubId)
        }
    }

    // MARK: - Gamepad

    private func pollGamepad(dt: Float) {
        guard let gp = GCController.current?.extendedGamepad else { return }
        let sprint: Float = (gp.leftThumbstickButton?.isPressed ?? false) || (gp.rightThumbstickButton?.isPressed ?? false) ? 3.0 : 1.0

        // Movement
        let lx = gp.leftThumbstick.xAxis.value
        let ly = gp.leftThumbstick.yAxis.value
        if abs(ly) > 0.1 { camera.dolly(amount: ly * 200 * dt * sprint) }
        if abs(lx) > 0.1 { camera.pan(dx: -lx * 300 * dt * sprint, dy: 0) }

        // Look
        let rx = gp.rightThumbstick.xAxis.value
        let ry = gp.rightThumbstick.yAxis.value
        if abs(rx) > 0.1 || abs(ry) > 0.1 {
            camera.lookRotate(deltaAz: -rx * 2.0 * dt, deltaEl: ry * 2.0 * dt)
        }

        // Triggers
        let lt = gp.leftTrigger.value
        let rt = gp.rightTrigger.value
        if rt > 0.05 { camera.pan(dx: 0, dy: rt * 300 * dt * sprint) }
        if lt > 0.05 { camera.pan(dx: 0, dy: -lt * 300 * dt * sprint) }

        // Buttons (rising-edge)
        let aPressed = gp.buttonA.isPressed
        if aPressed && !prevButtonA {
            if let target = reticleTarget {
                if renderHubs.contains(target) {
                    toggleHubExpansion(hubId: target)
                }
                selectedNode = target
                selectionCallback?(selectedNode)
            }
        }
        prevButtonA = aPressed

        let bPressed = gp.buttonB.isPressed
        if bPressed && !prevButtonB {
            for hubId in expandedHubs { toggleHubExpansion(hubId: hubId) }
            selectedNode = nil
            selectionCallback?(nil)
        }
        prevButtonB = bPressed

        // Teleport
        let yPressed = gp.buttonY.isPressed
        if yPressed && !prevButtonY {
            teleportToNextProject(direction: 1)
        }
        prevButtonY = yPressed

        // Reticle hit test — guard write to avoid per-frame callback spam
        if renderViewSize.width > 0 {
            let center = CGPoint(x: renderViewSize.width / 2, y: renderViewSize.height / 2)
            let newTarget = camera.hitTest(at: center, viewSize: renderViewSize, positions: positions,
                                              currentTarget: reticleTarget)
            if newTarget != reticleTarget {
                reticleTarget = newTarget
                reticleCallback?(newTarget)
                #if ENGRAM_INSTRUMENTATION
                jitterReticleCallbackFired = true
                #endif
            }
        }
    }

    // MARK: - Teleport

    func teleportToNextProject(direction: Int) {
        camera.teleportToNextProject(positions: positions, nodes: renderNodes,
                                     hubs: renderHubs, direction: direction)
        teleportLabel = camera.teleportLabel
        teleportCounter = camera.teleportCounter
        teleportCallback?(teleportLabel, teleportCounter)
    }

    /// Smoothly drive the camera to a named project using cached render data.
    func driveToProject(_ project: String) {
        camera.driveToProject(project, positions: positions, nodes: renderNodes, hubs: renderHubs)
        teleportLabel = camera.teleportLabel
        teleportCounter = camera.teleportCounter
        teleportCallback?(teleportLabel, teleportCounter)
    }

    var teleportGalaxyIndex: Int = 0

    func teleportToNextGalaxy(direction: Int) {
        guard let registry = galaxyRegistry else { return }
        let sorted = registry.galaxies.values.sorted(by: { $0.id < $1.id })
        guard !sorted.isEmpty else { return }
        teleportGalaxyIndex = (teleportGalaxyIndex + direction + sorted.count) % sorted.count
        let galaxy = sorted[teleportGalaxyIndex]

        // Compute radius from galaxy's node spread
        var maxSpread: Float = 200
        let galPositions = galaxy.simulation3D.positions
        let center = galaxy.worldCenter
        for (_, pos) in galPositions {
            maxSpread = max(maxSpread, simd_length(pos - center))
        }

        camera.teleportToGalaxy(center: center, radius: maxSpread * 0.5, label: galaxy.displayName)
        teleportLabel = camera.teleportLabel
        teleportCounter = camera.teleportCounter
        teleportCallback?(teleportLabel, teleportCounter)
    }

    // MARK: - Hit Testing

    func hitTest(at location: CGPoint, viewSize: CGSize) -> UUID? {
        camera.hitTest(at: location, viewSize: viewSize, positions: positions)
    }

    /// Hit test any mascot in any fleet — returns true if the tap is within 50px of a mascot's screen position.
    func hitTestMascot(at location: CGPoint, viewSize: CGSize) -> Bool {
        let sf = camera.scaleFactor
        guard sf > 0, let registry = galaxyRegistry else { return false }
        for galaxy in registry.galaxies.values {
            for mascot in galaxy.mascotFleet?.mascots.values ?? [:].values {
                let pos = mascot.currentPosition / sf
                guard let screenPos = camera.project(point3D: pos, viewSize: viewSize) else { continue }
                let dist = hypot(location.x - screenPos.x, location.y - screenPos.y)
                if dist < 50 { return true }
            }
        }
        return false
    }

    // MARK: - Fleet Chat

    func enterFleetChat() {
        let camPos = camera.cameraPosition * camera.scaleFactor
        galaxyRegistry?.galaxies.values.forEach { $0.mascotFleet?.enterChat(cameraPosition: camPos) }
    }

    func exitFleetChat() {
        galaxyRegistry?.galaxies.values.forEach { $0.mascotFleet?.exitChat() }
    }

    // MARK: - Drag Support

    func captureLookState() -> (azimuth: Float, elevation: Float, camPos: SIMD3<Float>) {
        camera.captureLookState()
    }

    func applyLookDrag(start: (azimuth: Float, elevation: Float, camPos: SIMD3<Float>), dx: Float, dy: Float) {
        camera.applyLookDrag(start: start, dx: dx, dy: dy)
    }

    func endDrag() {
        isDragging = false
    }

    // MARK: - Color Helpers

    private func nodeColorFloat3(for project: String, colorMap: [String: Color]) -> SIMD3<Float> {
        if let cached = nodeColorCache[project] { return cached }
        let color = colorMap[project] ?? .gray
        let nsColor = NSColor(color).usingColorSpace(.sRGB) ?? NSColor(color)
        var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
        nsColor.getRed(&r, green: &g, blue: &b, alpha: &a)
        let c = SIMD3<Float>(Float(r), Float(g), Float(b))
        nodeColorCache[project] = c
        return c
    }

    private func edgeColorFloat3(for project: String?, colorMap: [String: Color]) -> SIMD3<Float> {
        let key = project ?? "__default"
        if let cached = edgeColorCache[key] { return cached }
        if let project, let swiftColor = colorMap[project] {
            let nsColor = NSColor(swiftColor).usingColorSpace(.sRGB) ?? NSColor(swiftColor)
            var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
            nsColor.getRed(&r, green: &g, blue: &b, alpha: &a)
            let c = SIMD3<Float>(Float(r) * 0.6, Float(g) * 0.6, Float(b) * 0.6)
            edgeColorCache[key] = c
            return c
        }
        let c = SIMD3<Float>(0.35, 0.35, 0.4)
        edgeColorCache[key] = c
        return c
    }
}
