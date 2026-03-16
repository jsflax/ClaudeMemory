import SwiftUI
import EngramSceneKit
import EngramMetalShaders
import RealityKit
import GameController
import Metal
import simd
import os
import Lattice
import EngramModels

private let frameLog = Logger(subsystem: "com.claudememory.visualizer", category: "3DFrameTiming")

// MARK: - Graph3DView (SwiftUI host)

struct Graph3DView: View {
    let layoutMode: LayoutMode
    let showMascots: Bool
    let soundEnabled: Bool
    @Binding var selectedNode: UUID?
    let semanticClusters3D: [SemanticCluster3D]
    /// Direct references for renderTick — avoids closures crossing SwiftUI observation boundary.
    let simulation3D: ForceSimulation3D
    let embeddingProjection: EmbeddingProjection
    let camera3DState: Camera3DState
    let forcePositionSnapshot3D: [UUID: SIMD3<Float>]
    let transitionProgress: CGFloat
    let renderStore: GraphRenderStore
    let galaxyRegistry: GalaxyRegistry
    @Binding var cameraProjectTarget: String?

    /// Feature flag: set to true to use raw Metal renderer instead of RealityKit.
    static let useMetalRenderer = true

    @State private var scene = Graph3DScene()
    @State private var metalScene: MetalSceneManager?
    @State private var metalOrchestrator: FrameOrchestrator?
    @State private var scrollMonitor: Any?
    @State private var updateClosureCount: UInt64 = 0
    /// Mirrors metalScene.reticleTarget for SwiftUI (MetalSceneManager is not @Observable).
    @State private var metalReticleTarget: UUID?
    /// Mirrors metalScene teleport state for SwiftUI (MetalSceneManager is not @Observable).
    @State private var metalTeleportLabel: String?
    @State private var metalTeleportCounter: Int = 0

    // Mascot chat state
    @State private var isMascotChatOpen = false
    @State private var mascotChatEngine: AnyObject?  // type-erased for @available gating
    @Environment(\.lattice) private var lattice

    // Maintenance mode observation
    @State private var maintenanceObserver: AnyObject?

    /// Active scene object for input forwarding (Metal or RealityKit).
    private var activeSceneForInput: AnyObject? {
        if Self.useMetalRenderer { return metalScene } else { return scene }
    }

    #if ENGRAM_INSTRUMENTATION
    static var bodyEvalCount: UInt64 = 0
    static var bodyEvalFile: UnsafeMutablePointer<FILE>? = nil
    #endif

    var body: some View {
        #if ENGRAM_INSTRUMENTATION
        let _ = {
            Self.bodyEvalCount &+= 1
            if Self.bodyEvalFile == nil {
                Self.bodyEvalFile = fopen("/tmp/swiftui-body-eval.csv", "w")
                if let f = Self.bodyEvalFile {
                    fputs("timestamp,body_eval_count\n", f)
                }
            }
            if let f = Self.bodyEvalFile {
                let line = "\(String(format: "%.3f", CFAbsoluteTimeGetCurrent())),\(Self.bodyEvalCount)\n"
                fputs(line, f)
                fflush(f)
            }
        }()
        #endif
        GeometryReader { geo in
            ZStack {
                Group {
                    if Self.useMetalRenderer {
                        metalViewContent
                            .gesture(metalOrbitGesture)
                            .simultaneousGesture(metalTapGesture(viewSize: geo.size))
                    } else {
                        realityViewContent
                            .gesture(orbitGesture)
                            .simultaneousGesture(tapGesture(viewSize: geo.size))
                    }
                }
                .onGeometryChange(for: CGRect.self) { proxy in
                    proxy.frame(in: .global)
                } action: { frame in
                    viewFrame = frame
                    if Self.useMetalRenderer {
                        metalScene?.renderViewSize = frame.size
                    } else {
                        scene.renderViewSize = frame.size
                    }
                }

                // Center reticle (only when gamepad connected)
                if GCController.current != nil {
                    reticleOverlay
                }

                // Teleport project label (fades after 2s)
                if let label = Self.useMetalRenderer ? metalTeleportLabel : scene.teleportLabel {
                    let counter = Self.useMetalRenderer ? metalTeleportCounter : scene.teleportCounter
                    Text(label)
                        .font(.system(size: 24, weight: .bold, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.8))
                        .padding(.horizontal, 16)
                        .padding(.vertical, 8)
                        .background(.black.opacity(0.5), in: .capsule)
                        .transition(.opacity)
                        .allowsHitTesting(false)
                        .task(id: counter) {
                            let myCounter = counter
                            try? await Task.sleep(for: .seconds(2))
                            if Self.useMetalRenderer {
                                if metalTeleportCounter == myCounter {
                                    metalTeleportLabel = nil
                                }
                            } else {
                                if scene.teleportCounter == myCounter {
                                    scene.teleportLabel = nil
                                }
                            }
                        }
                }

                // Mascot chat panel
                if isMascotChatOpen {
                    // Click-outside-to-dismiss backdrop
                    Color.clear
                        .contentShape(Rectangle())
                        .onTapGesture { withAnimation(.easeOut(duration: 0.2)) { toggleMascotChat() } }

                    if #available(macOS 26.0, *), let engine = mascotChatEngine as? MascotChatEngine {
                        MascotChatPanel(engine: engine) {
                            toggleMascotChat()
                        }
                        .transition(.move(edge: .leading).combined(with: .opacity))
                        .padding(.leading, 16)
                        .padding(.top, 16)
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                        .onKeyPress(.escape) {
                            withAnimation(.easeOut(duration: 0.2)) { toggleMascotChat() }
                            return .handled
                        }
                    }
                }

                // Mode indicator + navigation hints
                VStack {
                    if layoutMode == .embedding {
                        Text("3D SEMANTIC VIEW — proximity = embedding similarity")
                            .font(.system(size: 11, weight: .medium, design: .monospaced))
                            .foregroundStyle(.cyan.opacity(0.4))
                            .padding(.top, 12)
                    }
                    Spacer()
                    VStack(spacing: 4) {
                        HStack(spacing: 16) {
                            Label("Drag to look", systemImage: "arrow.triangle.2.circlepath")
                            Label("Scroll to strafe", systemImage: "hand.draw")
                            Label("Pinch to fly", systemImage: "arrow.up.left.and.arrow.down.right")
                            Label("Twist to rotate", systemImage: "rotate.left")
                            Label("Click to select", systemImage: "cursorarrow.click")
                        }
                        HStack(spacing: 16) {
                            Label("WASD move", systemImage: "keyboard")
                            Label("IJKL look", systemImage: "keyboard")
                            Label("QE rise/descend", systemImage: "keyboard")
                            Label("Shift sprint", systemImage: "keyboard")
                            Label("T/R teleport", systemImage: "keyboard")
                            Label("[/] galaxy", systemImage: "keyboard")
                        }
                        if GCController.current != nil {
                            HStack(spacing: 16) {
                                Label("L-Stick move", systemImage: "l.joystick")
                                Label("R-Stick look", systemImage: "r.joystick")
                                Label("Triggers rise/descend", systemImage: "l2.button.roundedtop.horizontal")
                                Label("A select", systemImage: "a.button.roundedtop.horizontal.fill")
                                Label("X prev project", systemImage: "x.button.roundedtop.horizontal.fill")
                                Label("Y next project", systemImage: "y.button.roundedtop.horizontal.fill")
                                Label("Bumpers cycle", systemImage: "l1.button.roundedtop.horizontal")
                                Label("L3/R3 sprint", systemImage: "l.joystick.press.down")
                            }
                        }
                    }
                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.4))
                    .padding(.horizontal, 16)
                    .padding(.vertical, 6)
                    .background(.black.opacity(0.3), in: .capsule)
                    .padding(.bottom, 10)
                }
            }
        }
        .accessibilityIdentifier("graph-3d-view")
        .onAppear { if !Self.useMetalRenderer { installInputMonitor() } }
        .onDisappear { removeInputMonitor() }
        .onChange(of: soundEnabled) { _, enabled in
            if Self.useMetalRenderer, let ms = metalScene {
                ms.soundEnabled = enabled
                if enabled {
                    let audio = SpatialAudioEngine()
                    ms.spatialAudio = audio
                    audio.start()
                } else {
                    ms.spatialAudio?.stop()
                    ms.spatialAudio = nil
                }
            }
        }
        .onChange(of: selectedNode) { _, newValue in
            if Self.useMetalRenderer {
                if newValue != metalScene?.selectedNode {
                    metalScene?.selectedNode = newValue
                    if newValue == nil { collapseAllHubs() }
                }
            } else {
                if newValue != scene.renderSelectedNode {
                    scene.renderSelectedNode = newValue
                    if newValue == nil { collapseAllHubs() }
                }
            }
        }
        .onChange(of: cameraProjectTarget) { _, project in
            if let project {
                if Self.useMetalRenderer {
                    metalScene?.driveToProject(project)
                } else {
                    scene.driveToProject(project)
                }
                cameraProjectTarget = nil
            }
        }
    }

    /// Push slow-changing data from SwiftUI to the scene.
    /// Called from RealityView make/update — only fires when SwiftUI inputs actually change.
    /// NOTE: nodes/edges/hubs/colorMap are NOT pushed here — they're read directly from
    /// renderStore by the render tick, eliminating the SwiftUI timing mismatch.
    private func pushDataToScene() {
        scene.renderLayoutMode = layoutMode
        scene.renderGlowingNodes = renderStore.glowingNodes
        scene.renderNewNodes = renderStore.newNodeGlows
        scene.renderDyingNodes = renderStore.dyingNodes
        scene.renderSemanticClusters3D = semanticClusters3D
        scene.renderTopicGroups = renderStore.topicGroups
        scene.renderClusters = renderStore.clusterGroups
        scene.renderSearchMatchIds = renderStore.searchMatchIds
        scene.renderIsSearchActive = renderStore.isSearchActive
    }

    private var realityViewContent: some View {
        RealityView { content in
            let (root, camera) = scene.setup()
            #if ENGRAM_INSTRUMENTATION
            scene.setupProfiling()
            scene.setupLabelDiag()
            #endif
            content.add(root)
            content.add(camera)

            // Inject direct references — renderTick calls methods on these directly,
            // bypassing closures that would cross the SwiftUI observation boundary.
            scene.simulation3D = simulation3D
            scene.embeddingProjection = embeddingProjection
            scene.camera3DState = camera3DState
            scene.renderStore = renderStore
            // galaxyRegistry wired only for Metal path (RealityKit deprecated)
            scene.forcePositionSnapshot3D = forcePositionSnapshot3D
            scene.transitionProgress = transitionProgress
            scene.selectionCallback = { newSelection in
                selectedNode = newSelection
            }

            // Single render loop: SceneEvents.Update → renderTick (like a game engine).
            // No Timer. renderTick ticks the simulation, updates entities, triggers labels.
            scene.renderSubscription = content.subscribe(to: SceneEvents.Update.self) { _ in
                scene.renderTick()
            }

            pushDataToScene()
        } update: { content in
            updateClosureCount &+= 1
            let t0 = CFAbsoluteTimeGetCurrent()
            // Update snapshot/transition state (these change when layout mode switches)
            scene.forcePositionSnapshot3D = forcePositionSnapshot3D
            scene.transitionProgress = transitionProgress

            // Push slow-changing data when SwiftUI inputs change
            pushDataToScene()
            let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
            if updateClosureCount % 30 == 0 || elapsed > 2 {
                frameLog.error("[RV-update] count=\(updateClosureCount) elapsed=\(elapsed, format: .fixed(precision: 2))ms")
            }
        }
        .background(Color(red: 0.051, green: 0.067, blue: 0.09))
    }

    // MARK: - Metal Renderer View

    @ViewBuilder
    private var metalViewContent: some View {
        if let orch = metalOrchestrator {
            MetalViewRepresentable(orchestrator: orch)
                .onChange(of: layoutMode) { _, _ in pushDataToMetalScene() }
                .onChange(of: showMascots) { _, _ in pushDataToMetalScene() }
                .onChange(of: transitionProgress) { _, _ in
                    metalScene?.forcePositionSnapshot3D = forcePositionSnapshot3D
                    metalScene?.transitionProgress = transitionProgress
                }
        } else {
            Color(red: 0.051, green: 0.067, blue: 0.09)
                .onAppear { setupMetalRenderer() }
        }
    }

    private func setupMetalRenderer() {
        guard metalOrchestrator == nil else { return }
        guard let device = MTLCreateSystemDefaultDevice(),
              let library = try? EngramMetalShaders.makeLibrary(device: device),
              let orch = FrameOrchestrator(device: device, library: library) else { return }
        metalOrchestrator = orch

        guard let mgr = MetalSceneManager(orchestrator: orch) else { return }
        metalScene = mgr

        // Build mascot pipelines (app-target only — after MetalSceneManager creates the MascotDrawPass)
        setupMascotPipelines(pass: mgr.mascotDrawPass, orchestrator: orch)

        // Wire force snapshot provider — FrameOrchestrator dispatches forces via ForceEngine
        orch.forceSnapshotProvider = { [weak mgr] in
            guard let mgr, let registry = mgr.galaxyRegistry else { return nil }
            let sim = registry.unifiedSimulation
            guard !sim.isSettled, sim.nodeCount > 1 else { return nil }
            sim.useGPUForces = true
            return ForceEngine.SimulationSnapshot(
                nodeCount: sim.nodeCount, isSettled: sim.isSettled,
                posX: sim.posX, posY: sim.posY, posZ: sim.posZ,
                projectGroups: sim.projectGroupPublic, topicGroups: sim.topicGroupPublic,
                galaxyGroups: sim.galaxyGroupPublic, galaxyCenters: sim.galaxyCentersPublic,
                edgeIndices: sim.edgeIndicesPublic, topologyDirty: sim.topologyDirtyForGPU,
                chargeStrength: sim.chargeStrength, crossChargeMultiplier: sim.crossChargeMultiplier,
                sameTopicChargeScale: sim.sameTopicChargeScale, sameProjectChargeScale: sim.sameProjectChargeScale,
                springLength: sim.springLength, crossProjectSpringLength: sim.crossProjectSpringLength,
                springStrength: sim.springStrength,
                cohesionStrength: sim.cohesionStrength, centroidRepulsion: sim.centroidRepulsion,
                topicCohesionStrength: sim.topicCohesionStrength, topicCentroidRepulsion: sim.topicCentroidRepulsion,
                centerStrength: sim.centerStrength, center: sim.center,
                alpha: sim.alpha, damping: 0.78, maxSpeed: 12.0
            )
        }
        orch.onForceResult = { [weak mgr] result in
            guard let mgr, let registry = mgr.galaxyRegistry else { return }
            let sim = registry.unifiedSimulation
            sim.topologyDirtyForGPU = false
            sim.applyGPUForces(result)
        }

        // Galaxy registry provides all simulation/render data for the Metal path.
        mgr.galaxyRegistry = galaxyRegistry
        mgr.soundEnabled = soundEnabled
        mgr.notificationsEnabled = false
        mgr.camera3DState = camera3DState
        mgr.simulation3D = simulation3D
        mgr.embeddingProjection = embeddingProjection
        mgr.renderStore = renderStore
        mgr.forcePositionSnapshot3D = forcePositionSnapshot3D
        mgr.transitionProgress = transitionProgress
        mgr.selectionCallback = { newSelection in
            selectedNode = newSelection
        }
        mgr.reticleCallback = { newTarget in
            if newTarget != metalReticleTarget {
                metalReticleTarget = newTarget
            }
        }
        mgr.teleportCallback = { label, counter in
            if label != metalTeleportLabel || counter != metalTeleportCounter {
                metalTeleportLabel = label
                metalTeleportCounter = counter
            }
        }
        pushDataToMetalScene()
        installInputMonitor()
        setupMaintenanceObserver(mgr: mgr)

        // Start spatial audio if already enabled
        if soundEnabled {
            let audio = SpatialAudioEngine()
            mgr.spatialAudio = audio
            audio.start()
        }
    }

    private func setupMascotPipelines(pass: MascotDrawPass, orchestrator: FrameOrchestrator) {
        let device = orchestrator.device
        let library = orchestrator.library
        do {
            pass.mascotPipeline = try MetalPipelineBuilder.buildMascotPipeline(device: device, library: library)
            pass.arcaneCirclePipeline = try MetalPipelineBuilder.buildArcaneCirclePipeline(device: device, library: library)
            pass.nodeRingPipeline = try MetalPipelineBuilder.buildNodeRingPipeline(device: device, library: library)
            pass.holoScreenPipeline = try MetalPipelineBuilder.buildHoloScreenPipeline(device: device, library: library)
            pass.conjureScanPipeline = try MetalPipelineBuilder.buildConjureScanPipeline(device: device, library: library)
            pass.conjureOrbPipeline = try MetalPipelineBuilder.buildConjureOrbPipeline(device: device, library: library)
            pass.flowPipeline = orchestrator.flowPipeline
        } catch {
            // Mascot pipelines are non-critical — graph still renders without them
        }
    }

    private func setupMaintenanceObserver(mgr: MetalSceneManager) {
        // Check initial state
        updateMaintenanceState(mgr: mgr)

        // Observe HookState changes
        let ref = lattice.sendableReference
        maintenanceObserver = lattice.objects(HookState.self).observe { [weak mgr] change in
            guard let bgLattice = ref.resolve() else { return }
            // Check if this is the maintenanceActive key
            switch change {
            case .insert(let pk), .update(let pk):
                guard let state = bgLattice.object(HookState.self, primaryKey: pk),
                      state.key == .maintenanceActive else { return }
                let isActive = state.value == "1"
                Task { @MainActor [weak mgr] in
                    mgr?.isMaintenanceActive = isActive
                }
            case .delete:
                Task { @MainActor [weak mgr] in
                    mgr?.isMaintenanceActive = false
                }
            }
        } as AnyObject
    }

    private func updateMaintenanceState(mgr: MetalSceneManager) {
        if let state = lattice.objects(HookState.self)
            .where({ $0.key == .maintenanceActive }).first {
            mgr.isMaintenanceActive = state.value == "1"
        }
    }

    private func pushDataToMetalScene() {
        guard let ms = metalScene else { return }
        ms.layoutMode = layoutMode
        ms.showMascots = showMascots
        ms.semanticClusters3D = semanticClusters3D
    }

    // MARK: - Metal Gestures

    @GestureState private var metalDragStart: (azimuth: Float, elevation: Float, camPos: SIMD3<Float>)?

    private var metalOrbitGesture: some Gesture {
        DragGesture(minimumDistance: 2)
            .updating($metalDragStart) { value, state, _ in
                guard let ms = metalScene else { return }
                if state == nil {
                    state = ms.captureLookState()
                }
                guard let start = state else { return }
                let sensitivity: Float = 0.005
                let dw = Float(value.translation.width)
                let dh = Float(value.translation.height)
                ms.applyLookDrag(start: start, dx: dw * sensitivity, dy: -dh * sensitivity)
            }
            .onEnded { _ in
                metalScene?.endDrag()
            }
    }

    private func metalTapGesture(viewSize: CGSize) -> some Gesture {
        SpatialTapGesture()
            .onEnded { value in
                guard let ms = metalScene else { return }
                // Check mascot tap first — mascot takes priority over nodes
                if ms.hitTestMascot(at: value.location, viewSize: viewSize) {
                    toggleMascotChat()
                    return
                }
                if let nodeId = ms.hitTest(at: value.location, viewSize: viewSize) {
                    if renderStore.hubs.contains(nodeId) {
                        if selectedNode == nodeId {
                            metalCollapseAllHubs()
                            selectedNode = nil
                        } else {
                            let expanding = !ms.hubExpansion.expandedHubs.contains(nodeId)
                            let edgeTuples = ms.renderEdges.map { (sourceId: $0.sourceId, targetId: $0.targetId, relation: $0.relation) }
                            for hubId in ms.hubExpansion.expandedHubs where hubId != nodeId {
                                ms.hubExpansion.toggleHubExpansion(hubId: hubId, positions: ms.positions, edges: edgeTuples)
                                pinUnpinHubChildren(hubId: hubId, expanding: false)
                            }
                            ms.hubExpansion.toggleHubExpansion(hubId: nodeId, positions: ms.positions, edges: edgeTuples)
                            pinUnpinHubChildren(hubId: nodeId, expanding: expanding)
                            selectedNode = nodeId
                        }
                    } else {
                        metalCollapseAllHubs()
                        selectedNode = selectedNode == nodeId ? nil : nodeId
                    }
                } else {
                    metalCollapseAllHubs()
                    selectedNode = nil
                }
            }
    }

    private func toggleMascotChat() {
        if isMascotChatOpen {
            isMascotChatOpen = false
            metalScene?.exitFleetChat()
            // Clear sessions so next open starts fresh — prevents
            // response session context from accumulating across conversations
            if #available(macOS 26.0, *),
               let engine = mascotChatEngine as? MascotChatEngine {
                engine.reset()
            }
            mascotChatEngine = nil
        } else {
            isMascotChatOpen = true
            metalScene?.enterFleetChat()
            if #available(macOS 26.0, *) {
                setupChatEngineIfNeeded()
            }
        }
    }

    private func setupChatEngineIfNeeded() {
        if #available(macOS 26.0, *) {
            if mascotChatEngine == nil {
                let engine = MascotChatEngine()
                mascotChatEngine = engine
                let store = renderStore
                let binding = $selectedNode
                Task {
                    await engine.setup(
                        lattice: lattice,
                        renderStore: store,
                        selectedMemoryId: { binding.wrappedValue }
                    )
                }
            }
        }
    }

    private func metalCollapseAllHubs() {
        guard let ms = metalScene else { return }
        let edgeTuples = ms.renderEdges.map { (sourceId: $0.sourceId, targetId: $0.targetId, relation: $0.relation) }
        for hubId in ms.hubExpansion.expandedHubs {
            ms.hubExpansion.toggleHubExpansion(hubId: hubId, positions: ms.positions, edges: edgeTuples)
            pinUnpinHubChildren(hubId: hubId, expanding: false)
        }
    }

    // MARK: - Gestures

    // MARK: - Reticle (gamepad targeting)

    @ViewBuilder
    private var reticleOverlay: some View {
        let reticleTarget = Self.useMetalRenderer ? metalReticleTarget : scene.reticleTarget
        let hasTarget = reticleTarget != nil
        let nodeById = galaxyRegistry.mergedNodeById

        ZStack {
            // Crosshair lines
            let size: CGFloat = hasTarget ? 20 : 14
            let color: Color = hasTarget ? .white : .white.opacity(0.4)
            let thickness: CGFloat = hasTarget ? 1.5 : 1

            // Horizontal line with gap
            HStack(spacing: hasTarget ? 8 : 6) {
                Rectangle().frame(width: size, height: thickness)
                Rectangle().frame(width: size, height: thickness)
            }
            .foregroundStyle(color)

            // Vertical line with gap
            VStack(spacing: hasTarget ? 8 : 6) {
                Rectangle().frame(width: thickness, height: size)
                Rectangle().frame(width: thickness, height: size)
            }
            .foregroundStyle(color)

            // Center dot when targeting
            if hasTarget {
                Circle()
                    .frame(width: 4, height: 4)
                    .foregroundStyle(.white)
            }

            // Target label below reticle
            if let targetId = reticleTarget, let node = nodeById[targetId] {
                Text(node.label)
                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.9))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(.black.opacity(0.5), in: .capsule)
                    .offset(y: 30)
            }
        }
        .allowsHitTesting(false)
        .animation(.easeOut(duration: 0.15), value: hasTarget)
    }

    @GestureState private var dragStart: (azimuth: Float, elevation: Float, camPos: SIMD3<Float>)?

    /// Drag to look around (camera stays in place, view direction changes).
    private var orbitGesture: some Gesture {
        DragGesture(minimumDistance: 2)
            .updating($dragStart) { value, state, _ in
                if state == nil {
                    state = scene.captureLookState()
                }
                guard let start = state else { return }
                let sensitivity: Float = 0.005
                let dw = Float(value.translation.width)
                let dh = Float(value.translation.height)
                scene.applyLookDrag(start: start, dx: dw * sensitivity, dy: -dh * sensitivity)
            }
            .onEnded { _ in
                scene.endDrag()
            }
    }

    // MARK: - Trackpad input (scroll → strafe, pinch → dolly toward cursor)

    @State private var viewFrame: NSRect = .zero

    /// Keys tracked for continuous WASD/IJKL/QE movement.
    private static let movementKeys: Set<String> = ["w","a","s","d","i","j","k","l","q","e"]

    private func installInputMonitor() {
        // Scene-specific closures — capture the right scene reference once
        let heldKeys: (Bool, String) -> Void   // (insert, key) — insert=true inserts, false removes
        let clearKeys: () -> Void
        let teleport: (Int) -> Void
        let galaxyTeleport: (Int) -> Void
        let lookRotate: (Float, Float) -> Void
        let dolly: (Float) -> Void
        let dollyAt: (Float, Float, Float) -> Void  // (amount, cursorNX, cursorNY)
        let pan: (Float, Float) -> Void

        if Self.useMetalRenderer, let ms = metalScene {
            heldKeys = { insert, key in if insert { ms.heldKeys.insert(key) } else { ms.heldKeys.remove(key) } }
            clearKeys = { ms.heldKeys.removeAll() }
            teleport = { direction in ms.teleportToNextProject(direction: direction) }
            galaxyTeleport = { direction in ms.teleportToNextGalaxy(direction: direction) }
            lookRotate = { dAz, dEl in ms.camera.lookRotate(deltaAz: dAz, deltaEl: dEl) }
            dolly = { amount in ms.camera.dolly(amount: amount) }
            dollyAt = { amount, nx, ny in ms.camera.dolly(amount: amount, cursorNX: nx, cursorNY: ny) }
            pan = { dx, dy in ms.camera.pan(dx: dx, dy: dy) }
        } else {
            let s = scene
            heldKeys = { insert, key in if insert { s.heldKeys.insert(key) } else { s.heldKeys.remove(key) } }
            clearKeys = { s.heldKeys.removeAll() }
            teleport = { direction in s.teleportToNextProject(positions: s.renderPositions, nodes: s.renderNodes, direction: direction) }
            galaxyTeleport = { _ in } // Galaxy teleport only supported on Metal path
            lookRotate = { dAz, dEl in s.lookRotate(deltaAz: dAz, deltaEl: dEl) }
            dolly = { amount in s.dolly(amount: amount) }
            dollyAt = { amount, nx, ny in s.dolly(amount: amount, cursorNX: nx, cursorNY: ny) }
            pan = { dx, dy in s.pan(dx: dx, dy: dy) }
        }

        scrollMonitor = NSEvent.addLocalMonitorForEvents(matching: [.scrollWheel, .magnify, .rotate, .keyDown, .keyUp, .flagsChanged]) { event in
            // Shift tracking
            if event.type == .flagsChanged {
                heldKeys(event.modifierFlags.contains(.shift), "shift")
                return event
            }

            // Key events
            if event.type == .keyDown || event.type == .keyUp {
                if let responder = event.window?.firstResponder,
                   responder is NSTextView || responder is NSTextField {
                    clearKeys()
                    return event
                }

                let key = event.charactersIgnoringModifiers?.lowercased() ?? ""
                let noMods = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
                    .subtracting(.shift).isEmpty

                if event.type == .keyDown {
                    if noMods && Self.movementKeys.contains(key) {
                        heldKeys(true, key)
                        return nil
                    }
                    if key == "t" && noMods { teleport(1); return nil }
                    if key == "r" && noMods { teleport(-1); return nil }
                    if key == "]" && noMods { galaxyTeleport(1); return nil }
                    if key == "[" && noMods { galaxyTeleport(-1); return nil }
                } else {
                    if Self.movementKeys.contains(key) {
                        heldKeys(false, key)
                        return nil
                    }
                }
                return event
            }

            // Let scroll events pass through to overlay ScrollViews
            if let contentView = event.window?.contentView {
                let hitView = contentView.hitTest(event.locationInWindow)
                if hitView is NSScrollView || hitView?.enclosingScrollView != nil {
                    return event
                }
            }

            if event.type == .rotate {
                lookRotate(-Float(event.rotation) * 0.02, 0)
                return nil
            }
            if event.type == .magnify {
                let amount = Float(event.magnification) * 400
                guard event.window != nil else { dolly(amount); return nil }
                let loc = event.locationInWindow
                let nx = Float((loc.x - self.viewFrame.origin.x) / self.viewFrame.width - 0.5) * 2
                let ny = Float((loc.y - self.viewFrame.origin.y) / self.viewFrame.height - 0.5) * 2
                dollyAt(amount, nx, ny)
                return nil
            }

            pan(Float(event.scrollingDeltaX), Float(event.scrollingDeltaY))
            return nil
        }
    }

    private func removeInputMonitor() {
        if let monitor = scrollMonitor {
            NSEvent.removeMonitor(monitor)
            scrollMonitor = nil
        }
        if Self.useMetalRenderer {
            metalScene?.heldKeys.removeAll()
        } else {
            scene.heldKeys.removeAll()
        }
    }

    private func collapseAllHubs() {
        if Self.useMetalRenderer {
            metalCollapseAllHubs()
        } else {
            for hubId in scene.expandedHubs {
                scene.toggleHubExpansion(hubId: hubId)
                pinUnpinHubChildren(hubId: hubId, expanding: false)
            }
        }
    }

    /// Pin/unpin hub children in the force simulation directly (no closure crossing SwiftUI boundary).
    private func pinUnpinHubChildren(hubId: UUID, expanding: Bool) {
        let children = renderStore.edges.filter { $0.relation == "part_of" && $0.targetId == hubId }.map(\.sourceId)
        for childId in children {
            if expanding { simulation3D.pin(childId) } else { simulation3D.unpin(childId) }
        }
    }

    private func tapGesture(viewSize: CGSize) -> some Gesture {
        SpatialTapGesture()
            .onEnded { value in
                if let nodeId = scene.hitTest(at: value.location, viewSize: viewSize, positions: scene.renderPositions) {
                    if renderStore.hubs.contains(nodeId) {
                        if selectedNode == nodeId {
                            collapseAllHubs()
                            selectedNode = nil
                        } else {
                            let expanding = !scene.expandedHubs.contains(nodeId)
                            // Collapse other expanded hubs first
                            for hubId in scene.expandedHubs where hubId != nodeId {
                                scene.toggleHubExpansion(hubId: hubId)
                                pinUnpinHubChildren(hubId: hubId, expanding: false)
                            }
                            scene.toggleHubExpansion(hubId: nodeId)
                            pinUnpinHubChildren(hubId: nodeId, expanding: expanding)
                            selectedNode = nodeId
                        }
                    } else {
                        collapseAllHubs()
                        selectedNode = selectedNode == nodeId ? nil : nodeId
                    }
                } else {
                    collapseAllHubs()
                    selectedNode = nil
                }
            }
    }

}
