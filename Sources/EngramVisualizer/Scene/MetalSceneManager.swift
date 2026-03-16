import Metal
import CEngramSceneTypes
import simd
import SwiftUI
import os
import GameController
import EngramSceneKit

private let frameLog = Logger(subsystem: "io.engram.app", category: "MetalScene")

/// Per-frame scene manager for the Metal renderer path.
/// Coordinates scene data (input, simulation, data packing) each frame.
/// FrameOrchestrator (MTKViewDelegate) calls renderTick via onFrameCallback.
@MainActor
final class MetalSceneManager {

    let orchestrator: FrameOrchestrator
    let camera: CameraController
    let nebulaFog: NebulaFogSystem
    let flowParticles: FlowParticleSystem
    let mascotSharedResources: MascotSharedResources

    /// Mascot draw pass — registered as ExtraDrawPass on FrameOrchestrator.
    let mascotDrawPass: MascotDrawPass

    /// Input handler — keyboard, gamepad, selection, hit-test, drag, teleport.
    let inputHandler: InputHandler

    /// Label atlas generator — owns atlas texture generation + rect state.
    let labelAtlas: LabelAtlasGenerator

    /// Nebula packing — extracted stage that owns nebula color cache.
    let nebulaPackingStage = NebulaPackingStage()

    /// Node packing — extracted stage that owns node topology caches.
    let nodePacking = NodePackingStage()

    /// Edge packing — extracted stage that owns edge descriptor caches.
    let edgePacking = EdgePackingStage()

    /// Label packing — extracted stage that owns label diagnostic state.
    let labelPacking = LabelPackingStage()

    /// Convenience — shortcut to orchestrator.bufferManager
    var bufferManager: InstanceBufferManager { orchestrator.bufferManager }
    /// Convenience — shortcut to orchestrator.device
    var device: MTLDevice { orchestrator.device }

    /// Spatial audio engine — nil when sound is disabled.
    var spatialAudio: SpatialAudioEngine?

    // External references (set by Graph3DView)
    weak var camera3DState: Camera3DState?
    // Pushed from SwiftUI onChange handlers — never read from @Model in renderTick
    var soundEnabled: Bool = false
    var notificationsEnabled: Bool = false

    /// Galaxy registry — ticks all galaxy simulations and provides merged render data.
    var galaxyRegistry: GalaxyRegistry? {
        didSet { mascotDrawPass.galaxyRegistry = galaxyRegistry }
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

    /// Forwarding — selection state lives in inputHandler.
    var selectedNode: UUID? {
        get { inputHandler.selectedNode }
        set { inputHandler.selectedNode = newValue }
    }
    /// Forwarding — selection callback lives in inputHandler.
    var selectionCallback: ((UUID?) -> Void)? {
        get { inputHandler.selectionCallback }
        set { inputHandler.selectionCallback = newValue }
    }

    // Hub expansion
    let hubExpansion = HubExpansionController()

    /// Forwarding — held keys live in inputHandler.
    var heldKeys: Set<String> {
        get { inputHandler.heldKeys }
        set { inputHandler.heldKeys = newValue }
    }

    // Internal state
    private var renderFrameCount: UInt64 = 0
    private var hasCenteredCamera = false
    private var cameraStartTime: Date?
    private var centerTickCount: Int = 0
    private var lastSelectedNode: UUID?
    private var lastSearchActive = false
    private var lastSearchMatchIds: Set<UUID> = []

    // Forwarding — reticle/teleport/callback state lives in inputHandler.
    var reticleTarget: UUID? {
        get { inputHandler.reticleTarget }
        set { inputHandler.reticleTarget = newValue }
    }
    var reticleCallback: ((UUID?) -> Void)? {
        get { inputHandler.reticleCallback }
        set { inputHandler.reticleCallback = newValue }
    }
    var teleportCallback: ((String?, Int) -> Void)? {
        get { inputHandler.teleportCallback }
        set { inputHandler.teleportCallback = newValue }
    }
    var teleportLabel: String? {
        get { inputHandler.teleportLabel }
        set { inputHandler.teleportLabel = newValue }
    }
    var teleportCounter: Int {
        get { inputHandler.teleportCounter }
        set { inputHandler.teleportCounter = newValue }
    }

    // Color caches
    private var nodeColorCache: [String: SIMD3<Float>] = [:]
    private var edgeColorCache: [String: SIMD3<Float>] = [:]
    private var lastColorMapVersion: UInt64 = 0

    // nodePositionsDirty tracking — coordinated between renderTick and packing stages
    private var nodePositionsDirty: Bool = true

    // Project centroids — computed by NodePackingStage, consumed by LabelPackingStage + nebulae
    var projectCentroids: [String: (centroid: SIMD3<Float>, radius: Float, maxY: Float, count: Int)] = [:]
    // Depth range — computed by NodePackingStage, consumed by LabelPackingStage
    private var cachedMinDepth: Float = .greatestFiniteMagnitude
    private var cachedMaxDepth: Float = 0
    // nodeIndexMap — computed by NodePackingStage, consumed by EdgePackingStage
    private var nodeIndexMap: [UUID: UInt32] = [:]

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

    init?(orchestrator: FrameOrchestrator) {
        self.orchestrator = orchestrator
        self.camera = CameraController()
        self.inputHandler = InputHandler(camera: camera)
        self.nebulaFog = NebulaFogSystem(device: orchestrator.device)
        self.flowParticles = FlowParticleSystem(device: orchestrator.device)
        let templateMascot = MascotSystem(device: orchestrator.device)
        self.mascotSharedResources = templateMascot.extractSharedResources()
        self.labelAtlas = LabelAtlasGenerator(device: orchestrator.device)
        self.mascotDrawPass = MascotDrawPass()
        mascotDrawPass.cameraController = camera
        mascotDrawPass.opaqueDepthState = orchestrator.opaqueDepthState

        orchestrator.onFrameCallback = { [weak self] dt in
            self?.renderTick(dt: dt)
        }
        orchestrator.extraDrawPasses = [mascotDrawPass]
    }

    // MARK: - Render Tick

    #if ENGRAM_INSTRUMENTATION
    private var lastFrameWallTime: CFAbsoluteTime = 0
    #endif

    func renderTick(dt: Float) {
        renderFrameCount &+= 1
        nodePositionsDirty = true  // will be cleared by nodePacking or edgePacking.uploadNodePositions
        edgePacking.markPositionsDirty()
        let frameStart = CFAbsoluteTimeGetCurrent()
        var checkpoint = frameStart
        func mark(_ label: String) {
            let now = CFAbsoluteTimeGetCurrent()
            let ms = (now - checkpoint) * 1000.0
            if ms > 3 { GPULog.log("TICK \(label) \(String(format: "%.1f", ms))ms frame=\(renderFrameCount)") }
            checkpoint = now
        }
        #if ENGRAM_INSTRUMENTATION
        let wallDt = lastFrameWallTime > 0 ? (frameStart - lastFrameWallTime) * 1000.0 : 0
        lastFrameWallTime = frameStart
        #endif

        // Input + camera
        let preInputSelection = selectedNode
        camera.heldKeys = inputHandler.heldKeys
        camera.pollKeyboard(dt: dt)
        inputHandler.pollGamepad(
            dt: dt,
            positions: positions,
            renderNodes: renderNodes,
            renderEdges: renderEdges,
            renderHubs: renderHubs,
            renderViewSize: renderViewSize,
            hubExpansion: hubExpansion
        )
        camera.updateCamera(dt: dt)
        orchestrator.camera = camera.state
        mark("input")

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
            mark("drain")
            #if ENGRAM_INSTRUMENTATION
            drainMs = (CFAbsoluteTimeGetCurrent() - drainStart) * 1000.0
            #endif

            // Tick the unified simulation once (not per-galaxy).
            // Always tick even during initial load — nodes already drained into the sim
            // should start converging. The old per-galaxy code ticked each galaxy
            // independently; blocking the unified sim starves all nodes.
            // useGPUForces is managed by drawFrame() — set true when GPU forces are
            // encoded, false when skipped (gpuInFlight or unavailable). This tells tick()
            // whether to dispatch CPU fallback forces.
            let sim = registry.unifiedSimulation
            if sim.nodeCount > 0 { sim.tick() }
            mark("tick")
            for galaxy in registry.galaxies.values {
                galaxy.embeddingProjection.tickAnimation3D()
            }
            let anyEmbeddingActive = registry.galaxies.values.contains { !$0.embeddingProjection.is3DAnimationSettled }
            let anyActive = !sim.isSettled || sim.isLocalWake || anyEmbeddingActive

            #if ENGRAM_INSTRUMENTATION
            // Diagnostic: log unified sim state once per second
            if renderFrameCount % 60 == 0 {
                print("SIM[unified] frame=\(renderFrameCount) anyInitLoad=\(anyInitialLoad) settled=\(sim.isSettled) nodes=\(sim.nodeCount) positions=\(sim.positions.count) alpha=\(String(format: "%.4f", sim.alpha)) maxSpeedSq=\(String(format: "%.4f", sim.lastMaxSpeedSq)) atten=\(String(format: "%.4f", sim.smoothedAttenuation)) forceAge=\(sim.forceAge) tickInFlight=\(sim.tickInFlight) framesSinceWake=\(sim.framesSinceWake)")
            }
            #endif
            
            registry.mergeRenderData()
            mark("merge")

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
            mark("cache")

            if anyActive {
                if layoutMode == .forceDirected {
                    // Unified sim positions are already in world space — read directly.
                    positions = sim.positions
                } else {
                    // Embedding mode: blend unified force positions with t-SNE projections
                    var merged: [UUID: SIMD3<Float>] = [:]
                    for galaxy in registry.galaxies.values {
                        let proj = galaxy.embeddingProjection
                        let tsne3D = proj.projectedPositions3D
                        if tsne3D.isEmpty {
                            // No t-SNE yet — use force positions for this galaxy's nodes
                            for (id, pos) in sim.positions where registry.nodeToGalaxy[id] == galaxy.id {
                                merged[id] = pos
                            }
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
                if (!hasCenteredCamera || elapsed < 3.0) && !inputHandler.isDragging && heldKeys.isEmpty {
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
        mark("sim-block")

        // Consume hub toggles — pin/unpin in unified simulation
        if !hubExpansion.pendingHubToggles.isEmpty, let registry = galaxyRegistry {
            let sim = registry.unifiedSimulation
            for toggle in hubExpansion.pendingHubToggles {
                if let galaxy = registry.galaxyForNode(toggle.hubId) {
                    let children = galaxy.renderStore.edges.filter { $0.relation == "part_of" && $0.targetId == toggle.hubId }.map(\.sourceId)
                    for childId in children {
                        if toggle.expanding { sim.pin(childId) } else { sim.unpin(childId) }
                    }
                }
            }
            hubExpansion.pendingHubToggles.removeAll()
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
        let hasExpansions = !hubExpansion.expandedHubs.isEmpty
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
        orchestrator.animationTime += dt

        // Maintenance pulse: lerp toward target (2s ramp up/down)
        let maintenanceTarget: Float = isMaintenanceActive ? 1.0 : 0.0
        let pulseRate: Float = 0.5 * dt  // 1/2s = 2s full transition
        if maintenancePulse < maintenanceTarget {
            maintenancePulse = min(maintenancePulse + pulseRate, maintenanceTarget)
        } else if maintenancePulse > maintenanceTarget {
            maintenancePulse = max(maintenancePulse - pulseRate, maintenanceTarget)
        }
        orchestrator.maintenancePulse = maintenancePulse

        // Mascot update — they patrol independently of scene changes.
        // Color map conversion is cached and only rebuilt when topology changes.
        let mascotStart = CFAbsoluteTimeGetCurrent()
        mascotDrawPass.galaxyRegistry = showMascots ? galaxyRegistry : nil
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
                        device: device,
                        sharedResources: mascotSharedResources,
                        library: orchestrator.library
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
        mark("mascot")

        let anyMascotActive = galaxyRegistry?.galaxies.values.contains { galaxy in
            galaxy.mascotFleet?.mascots.values.contains { !$0.isSettled } ?? false
        } ?? false
        let maintenanceTransitioning = maintenancePulse > 0.001 && maintenancePulse < 0.999
        let isActive = sceneNeedsUpdate || cameraMoving || hasInput || anyMascotActive || maintenanceTransitioning

        mark("flags")
        if !isActive {
            if renderFrameCount > 10 && renderFrameCount % 30 != 0 {
                GPULog.log("TICK idle_skip frame=\(renderFrameCount)")

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

        GPULog.log("TICK update scene=\(sceneNeedsUpdate) geo=\(geometryChanged) vis=\(visualOnlyChanged) cam=\(cameraMoving) mascot=\(anyMascotActive) frame=\(renderFrameCount)")
        if sceneNeedsUpdate {
            let edgeTuples = renderEdges.map { (sourceId: $0.sourceId, targetId: $0.targetId, relation: $0.relation) }
            hubExpansion.updateExpansions(dt: dt, positions: &positions, edges: edgeTuples)

            let t0 = CFAbsoluteTimeGetCurrent()
            let nodeResult = nodePacking.pack(
                nodes: renderNodes,
                positions: positions,
                selectedNode: selectedNode,
                expandedHubs: hubExpansion.expandedHubs,
                expandedChildPositions: hubExpansion.expandedChildPositions,
                glowingNodes: glowingNodes,
                newNodes: newNodes,
                dyingNodes: dyingNodes,
                searchMatchIds: searchMatchIds,
                isSearchActive: isSearchActive,
                renderColorMap: renderColorMap,
                topicGroups: topicGroups,
                nodeColorCache: &nodeColorCache,
                edgeColorCache: &edgeColorCache,
                bufferManager: bufferManager,
                lightingUniforms: &orchestrator.lightingUniforms,
                animationTime: orchestrator.animationTime,
                scaleFactor: scaleFactor,
                nodeRadius: nodeRadius,
                cameraPosition: camera.cameraPosition,
                hubs: renderHubs,
                galaxyRegistry: galaxyRegistry,
                renderFrameCount: renderFrameCount
            )
            nodeIndexMap = nodeResult.nodeIndexMap
            projectCentroids = nodeResult.projectCentroids
            cachedMinDepth = nodeResult.minDepth
            cachedMaxDepth = nodeResult.maxDepth
            if !nodeIndexMap.isEmpty {
                nodePositionsDirty = false
                edgePacking.nodePositionsDirty = false
            }
            nodesMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
            mark("packNodes")

            // Edge and nebula packing only needed when geometry changed
            // (positions, selection, search), not for glow/arrival visual-only changes.
            if geometryChanged {
                let positionOnly = positionsChanged && !selectionChanged && !searchChanged && !hasExpansions
                // Edges: throttle to every 2nd frame when only positions change.
                // At 60fps, 16ms of edge lag is sub-pixel during smooth simulation drift.
                if !positionOnly || renderFrameCount % 2 == 0 {
                    let t1 = CFAbsoluteTimeGetCurrent()
                    edgePacking.pack(
                        edges: renderEdges,
                        nodes: renderNodes,
                        positions: positions,
                        nodeIndexMap: nodeIndexMap,
                        selectedNode: selectedNode,
                        expandedHubs: hubExpansion.expandedHubs,
                        expandedChildPositions: hubExpansion.expandedChildPositions,
                        hubs: renderHubs,
                        searchMatchIds: searchMatchIds,
                        isSearchActive: isSearchActive,
                        renderColorMap: renderColorMap,
                        edgeColorCache: &edgeColorCache,
                        nodeColorCache: &nodeColorCache,
                        bufferManager: bufferManager,
                        scaleFactor: scaleFactor,
                        edgeRadius: edgeRadius,
                        nodeRadius: nodeRadius,
                        layoutMode: layoutMode,
                        galaxyRegistry: galaxyRegistry,
                        cachedNodeRadii: nodeResult.cachedNodeRadii,
                        cachedNodeProject: nodeResult.cachedNodeProject
                    )
                    edgesMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000.0
                    mark("packEdges")
                }

                if !positionOnly || renderFrameCount % 6 == 0 {
                    let t2 = CFAbsoluteTimeGetCurrent()
                    updateNebulae()
                    nebMs = (CFAbsoluteTimeGetCurrent() - t2) * 1000.0
                    mark("nebulae")
                }
            }

            if selectedNode != nil || bufferManager.actualFlowParticleCount > 0 {
                let t3 = CFAbsoluteTimeGetCurrent()
                flowParticles.update(
                    dt: dt,
                    selectedNode: selectedNode,
                    edges: renderEdges,
                    positions: positions,
                    expandedHubs: hubExpansion.expandedHubs,
                    bufferManager: bufferManager
                )
                flowMs = (CFAbsoluteTimeGetCurrent() - t3) * 1000.0
            }

            lastSelectedNode = selectedNode
            lastSearchActive = isSearchActive
            lastSearchMatchIds = searchMatchIds

        }

        if sceneNeedsUpdate || cameraMoving {
            let t4 = CFAbsoluteTimeGetCurrent()
            labelPacking.pack(
                positions: positions,
                nodes: renderNodes,
                hubs: renderHubs,
                edges: renderEdges,
                selectedNode: selectedNode,
                camera: camera,
                hubExpansion: hubExpansion,
                labelAtlas: labelAtlas,
                projectCentroids: projectCentroids,
                cachedMinDepth: cachedMinDepth,
                cachedMaxDepth: cachedMaxDepth,
                renderColorMap: renderColorMap,
                topicGroups: topicGroups,
                isSearchActive: isSearchActive,
                searchMatchIds: searchMatchIds,
                bufferManager: bufferManager,
                galaxyRegistry: galaxyRegistry,
                scaleFactor: scaleFactor,
                nodeRadius: nodeRadius,
                renderFrameCount: renderFrameCount,
                nodeColorFloat3: { [self] project, colorMap in
                    self.nodeColorFloat3(for: project, colorMap: colorMap)
                }
            )
            labelsMs = (CFAbsoluteTimeGetCurrent() - t4) * 1000.0
            mark("packLabels")
        }

        let frameTotalMs = (CFAbsoluteTimeGetCurrent() - frameStart) * 1000.0
        if renderFrameCount % 60 == 1 || frameTotalMs > 20 {
            print("[engram:frame] #\(renderFrameCount) total=\(String(format: "%.1f", frameTotalMs))ms sim=\(String(format: "%.1f", simMs))ms mascot=\(String(format: "%.1f", mascotMs))ms nodes=\(String(format: "%.1f", nodesMs))ms edges=\(String(format: "%.1f", edgesMs))ms labels=\(String(format: "%.1f", labelsMs))ms neb=\(String(format: "%.1f", nebMs))ms nodeCount=\(positions.count) edgeCount=\(renderEdges.count)")
        }

        mark("print")
        // Report camera state
        if let camState = camera3DState {
            if camera.azimuth != camState.azimuth { camState.azimuth = camera.azimuth }
            if camera.cameraPosition != camState.position { camState.position = camera.cameraPosition }
            if camera.cameraTarget != camState.target { camState.target = camera.cameraTarget }
            if didUpdatePositions {
                GPULog.log("TICK camState.positions = \(positions.count) frame=\(renderFrameCount)")
                camState.positions = positions
            }
        }

        // Spatial audio tick — reads all state computed above
        #if ENGRAM_INSTRUMENTATION
        let audioStart = CFAbsoluteTimeGetCurrent()
        #endif
        mark("camState")
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
    private var jitterFile: UnsafeMutablePointer<FILE>? = nil
    private var jitterCallbackFired: Bool = false
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
            let line = "\(renderFrameCount),\(String(format: "%.2f", wallDt)),\(String(format: "%.2f", totalMs)),\(String(format: "%.2f", simMs)),\(String(format: "%.2f", mascotMs)),\(String(format: "%.2f", nodesMs)),\(String(format: "%.2f", edgesMs)),\(String(format: "%.2f", labelsMs)),\(skipped ? 1 : 0),\(reason),\(jitterCallbackFired ? 1 : 0),\(inputHandler.jitterReticleCallbackFired ? 1 : 0),\(jitterTeleportCallbackFired ? 1 : 0),\(positionsChanged ? 1 : 0),\(geometryChanged ? 1 : 0),\(cameraMoving ? 1 : 0),\(anyMascotActive ? 1 : 0),\(bodyCount)\n"
            fputs(line, f)
            fflush(f)
        }
        // Reset per-frame callback flags
        jitterCallbackFired = false
        inputHandler.jitterReticleCallbackFired = false
        jitterTeleportCallbackFired = false
    }
    #endif

    // MARK: - Nebulae

    private func updateNebulae() {
        nebulaPackingStage.update(
            device: device,
            nebulaFog: nebulaFog,
            bufferManager: bufferManager,
            positions: positions,
            nodes: renderNodes,
            renderColorMap: renderColorMap,
            semanticClusters3D: semanticClusters3D,
            layoutMode: layoutMode,
            scaleFactor: scaleFactor,
            projectCentroids: projectCentroids,
            colorMapVersion: galaxyRegistry?.mergedColorMapVersion ?? 0
        )
    }

    // MARK: - Hub Expansion (delegated to hubExpansion controller)

    // MARK: - Input Forwarding (delegated to inputHandler)

    func teleportToNextProject(direction: Int) {
        inputHandler.teleportToNextProject(direction: direction, positions: positions, nodes: renderNodes, hubs: renderHubs)
    }

    func driveToProject(_ project: String) {
        inputHandler.driveToProject(project, positions: positions, nodes: renderNodes, hubs: renderHubs)
    }

    func teleportToNextGalaxy(direction: Int) {
        inputHandler.teleportToNextGalaxy(direction: direction, galaxyRegistry: galaxyRegistry)
    }

    func hitTest(at location: CGPoint, viewSize: CGSize) -> UUID? {
        inputHandler.hitTest(at: location, viewSize: viewSize, positions: positions)
    }

    func hitTestMascot(at location: CGPoint, viewSize: CGSize) -> Bool {
        inputHandler.hitTestMascot(at: location, viewSize: viewSize, galaxyRegistry: galaxyRegistry)
    }

    func enterFleetChat() {
        inputHandler.enterFleetChat(galaxyRegistry: galaxyRegistry)
    }

    func exitFleetChat() {
        inputHandler.exitFleetChat(galaxyRegistry: galaxyRegistry)
    }

    func captureLookState() -> (azimuth: Float, elevation: Float, camPos: SIMD3<Float>) {
        inputHandler.captureLookState()
    }

    func applyLookDrag(start: (azimuth: Float, elevation: Float, camPos: SIMD3<Float>), dx: Float, dy: Float) {
        inputHandler.applyLookDrag(start: start, dx: dx, dy: dy)
    }

    func endDrag() {
        inputHandler.endDrag()
    }

    // MARK: - Color Helpers (delegated to ColorHelpers enum)

    private func nodeColorFloat3(for project: String, colorMap: [String: Color]) -> SIMD3<Float> {
        ColorHelpers.nodeColorFloat3(for: project, colorMap: colorMap, cache: &nodeColorCache)
    }

    private func edgeColorFloat3(for project: String?, colorMap: [String: Color]) -> SIMD3<Float> {
        ColorHelpers.edgeColorFloat3(for: project, colorMap: colorMap, cache: &edgeColorCache)
    }
}
