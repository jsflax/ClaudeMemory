import Metal
import simd
import SwiftUI
import os
import GameController

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
    let mascot: MascotSystem

    // External references (set by Graph3DView)
    weak var simulation3D: ForceSimulation3D?
    weak var embeddingProjection: EmbeddingProjection?
    weak var camera3DState: Camera3DState?
    weak var renderStore: GraphRenderStore?

    // Data pushed from SwiftUI (only layout/transition state — visual data comes from renderStore)
    var layoutMode: LayoutMode = .forceDirected
    var semanticClusters3D: [SemanticCluster3D] = []
    var forcePositionSnapshot3D: [UUID: SIMD3<Float>] = [:]
    var transitionProgress: CGFloat = 0

    // Convenience accessors — read from renderStore (no local copies needed)
    private var glowingNodes: [UUID: Date] { renderStore?.glowingNodes ?? [:] }
    private var newNodes: [UUID: Date] { renderStore?.newNodeGlows ?? [:] }
    private var dyingNodes: [UUID: DyingNode] { renderStore?.dyingNodes ?? [:] }
    private var topicGroups: [TopicGroupInfo] { renderStore?.topicGroups ?? [] }
    private var clusters: [[UUID]] { renderStore?.clusterGroups ?? [] }
    private var searchMatchIds: Set<UUID> { renderStore?.searchMatchIds ?? [] }
    private var isSearchActive: Bool { renderStore?.isSearchActive ?? false }

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

    // Edge packing: cached source/target IDs + radii for fast position-only updates

    // Label atlas state
    var labelAtlasRects: [UUID: (u0: Float, v0: Float, u1: Float, v1: Float)] = [:]
    var projectLabelAtlasRects: [String: (u0: Float, v0: Float, u1: Float, v1: Float)] = [:]
    var projectCentroids: [String: (centroid: SIMD3<Float>, radius: Float, maxY: Float)] = [:]
    var labelAtlasNodeIds: Set<UUID> = []
    var labelAtlasHubIds: Set<UUID> = []
    var labelAtlasProjects: Set<String> = []
    private var labelAtlasAspectCorrection: Float = 1.0
    private var labelAtlasAllocW: Int = 0
    private var labelAtlasAllocH: Int = 0
    private var labelAtlasRegenFrame: UInt64 = 0
    private var lastAtlasTopologyVersion: UInt64 = 0
    private var isAtlasGenerating = false

    // View properties
    var renderViewSize: CGSize = .zero

    private let scaleFactor: Float = 1.0 / 200.0
    private let nodeRadius: Float = 0.04
    private let edgeRadius: Float = 0.004

    // Precomputed cylinder trig (6-sided) — avoids 12 sin/cos per edge per frame

    private var renderNodes: [NodeData] { renderStore?.nodes ?? [] }
    private var renderEdges: [EdgeData] { renderStore?.edges ?? [] }
    private var renderHubs: Set<UUID> { renderStore?.hubs ?? [] }
    private var renderColorMap: [String: Color] { renderStore?.colorMap ?? [:] }

    init?(renderer: MetalGraphRenderer) {
        self.renderer = renderer
        self.camera = CameraController()
        self.nebulaFog = NebulaFogSystem(device: renderer.device)
        self.flowParticles = FlowParticleSystem(device: renderer.device)
        self.mascot = MascotSystem(device: renderer.device)

        renderer.camera = camera
        renderer.mascotSystem = mascot
        renderer.onFrameCallback = { [weak self] dt in
            self?.renderTick(dt: dt)
        }
    }

    // MARK: - Render Tick

    func renderTick(dt: Float) {
        renderFrameCount &+= 1

        // Input + camera
        let preInputSelection = selectedNode
        camera.heldKeys = heldKeys
        camera.pollKeyboard(dt: dt)
        pollGamepad(dt: dt)
        camera.updateCamera(dt: dt)

        if selectedNode != preInputSelection {
            selectionCallback?(selectedNode)
        }

        // Tick simulation + compute positions
        var didUpdatePositions = false
        if let sim = simulation3D, let proj = embeddingProjection {
            sim.tick()
            proj.tickAnimation3D()

            let newPositions: [UUID: SIMD3<Float>]?
            if sim.isSettled && proj.is3DAnimationSettled {
                newPositions = nil
            } else if layoutMode == .embedding {
                let tsne3D = proj.projectedPositions3D
                if tsne3D.isEmpty {
                    newPositions = sim.positions
                } else if transitionProgress >= 1.0 {
                    newPositions = tsne3D
                } else {
                    var blended: [UUID: SIMD3<Float>] = [:]
                    let allIds = Set(forcePositionSnapshot3D.keys).union(tsne3D.keys)
                    for id in allIds {
                        let forcePos = forcePositionSnapshot3D[id] ?? sim.positions[id] ?? .zero
                        let tsnePos = tsne3D[id] ?? forcePos
                        blended[id] = forcePos + (tsnePos - forcePos) * Float(transitionProgress)
                    }
                    newPositions = blended
                }
            } else {
                newPositions = sim.positions
            }
            if let p = newPositions {
                positions = p
                didUpdatePositions = true

                if cameraStartTime == nil { cameraStartTime = Date() }
                let elapsed = Date().timeIntervalSince(cameraStartTime!)
                if (!hasCenteredCamera || elapsed < 3.0) && !isDragging {
                    centerTickCount += 1
                    if centerTickCount % 6 == 0 {
                        camera.centerOnGraph(positions: positions)
                    }
                    if elapsed >= 3.0 { hasCenteredCamera = true }
                }
            }
        }

        // Consume hub toggles
        if !pendingHubToggles.isEmpty, let sim = simulation3D {
            for toggle in pendingHubToggles {
                let children = renderEdges.filter { $0.relation == "part_of" && $0.targetId == toggle.hubId }.map(\.sourceId)
                for childId in children {
                    if toggle.expanding { sim.pin(childId) } else { sim.unpin(childId) }
                }
            }
            pendingHubToggles.removeAll()
        }

        // Invalidate color caches when colorMap changes
        if let version = renderStore?.colorMapVersion, version != lastColorMapVersion {
            nodeColorCache.removeAll(keepingCapacity: true)
            edgeColorCache.removeAll(keepingCapacity: true)
            lastColorMapVersion = version
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
        let mascotInspecting = mascot.arcaneIntensity > 0.01
        let visualOnlyChanged = !glowingNodes.isEmpty || !newNodes.isEmpty || !dyingNodes.isEmpty || mascotInspecting
        let sceneNeedsUpdate = geometryChanged || visualOnlyChanged

        // Always advance animation time (shaders need it for scan lines, flicker, etc.)
        renderer.animationTime += dt

        // Always update mascot — it patrols independently of scene changes
        let nodeByIdForMascot = renderStore?.nodeById ?? [:]
        var mascotNodeInfo: [UUID: MascotNodeInfo] = [:]
        if let targetId = mascot.currentTargetId, let nd = nodeByIdForMascot[targetId] {
            mascotNodeInfo[targetId] = MascotNodeInfo(
                content: nd.content, project: nd.project,
                topic: nd.topic, importance: nd.importance,
                createdAt: nd.createdAt, lastAccessedAt: nd.lastAccessedAt
            )
        }
        mascot.update(dt: dt, camera: camera, nodePositions: positions, nodeInfo: mascotNodeInfo)

        let isActive = sceneNeedsUpdate || cameraMoving || hasInput || !mascot.isSettled

        if !isActive {
            if renderFrameCount > 10 && renderFrameCount % 30 != 0 {
                return
            }
        }

        if sceneNeedsUpdate {
            updateExpansions(dt: dt)
            packNodeInstances()

            // Edge and nebula packing only needed when geometry changed
            // (positions, selection, search), not for glow/arrival visual-only changes.
            if geometryChanged {
                packEdgeVertices()
                let positionOnly = positionsChanged && !selectionChanged && !searchChanged && !hasExpansions
                if !positionOnly || renderFrameCount % 6 == 0 {
                    updateNebulae()
                }
            }

            if selectedNode != nil || renderer.actualFlowParticleCount > 0 {
                flowParticles.update(
                    dt: dt,
                    selectedNode: selectedNode,
                    edges: renderEdges,
                    positions: positions,
                    expandedHubs: expandedHubs,
                    renderer: renderer
                )
            }

            lastSelectedNode = selectedNode
            lastSearchActive = isSearchActive
            lastSearchMatchIds = searchMatchIds

        }

        if sceneNeedsUpdate || cameraMoving {
            packLabelInstances()
        }

        // Report camera state
        if let camState = camera3DState {
            if camera.azimuth != camState.azimuth { camState.azimuth = camera.azimuth }
            if camera.cameraPosition != camState.position { camState.position = camera.cameraPosition }
            if camera.cameraTarget != camState.target { camState.target = camera.cameraTarget }
            if didUpdatePositions { camState.positions = positions }
        }
    }

    // MARK: - Node Instance Packing

    private func packNodeInstances() {
        let nodeCount = positions.count
        guard nodeCount > 0 else {
            renderer.actualNodeCount = 0
            return
        }

        let hubs = renderHubs
        let colorMap = renderColorMap
        let nodeById = renderStore?.nodeById ?? [:]
        let now = Date()

        if instanceArray.count < nodeCount {
            instanceArray = [NodeInstance](repeating: NodeInstance(position: .zero, scale: 0, color: .zero), count: max(nodeCount * 2, 512))
        }

        // Point lights
        var pointLightCount: Int = 0

        var idx = 0
        for (id, pos) in positions {
            guard let nodeData = nodeById[id] else { continue }

            let worldPos = pos * scaleFactor
            let isHub = hubs.contains(id)
            let importance = max(1, nodeData.importance)
            let baseRadius: Float = isHub ? nodeRadius * 1.6 : nodeRadius
            let r = baseRadius * (1.0 + Float(importance - 1) * 0.08)

            // Recall glow
            let ri: Float = {
                guard let glowStart = glowingNodes[id], id != selectedNode else { return 0 }
                let elapsed = Float(now.timeIntervalSince(glowStart))
                let fadeIn: Float = 1.0, hold: Float = 1.5, fadeOut: Float = 2.0
                if elapsed < fadeIn { let t = elapsed / fadeIn; return t * t * t }
                else if elapsed < fadeIn + hold { return 1.0 }
                else if elapsed < fadeIn + hold + fadeOut {
                    let t = 1.0 - (elapsed - fadeIn - hold) / fadeOut; return t * t
                }
                return 0
            }()

            // Arrival intensity
            let ai: Float = {
                guard let arrivalTime = newNodes[id] else { return 0 }
                let elapsed = Float(now.timeIntervalSince(arrivalTime))
                let fadeIn: Float = 0.8, hold: Float = 2.0, fadeOut: Float = 3.0
                if elapsed < fadeIn { let t = elapsed / fadeIn; return t * t * t }
                else if elapsed < fadeIn + hold { return 1.0 }
                else if elapsed < fadeIn + hold + fadeOut {
                    let t = 1.0 - (elapsed - fadeIn - hold) / fadeOut; return t * t
                }
                return 0
            }()

            let searchDimmed = isSearchActive && !searchMatchIds.contains(id)
            let searchMatched = isSearchActive && searchMatchIds.contains(id) && id != selectedNode
            let isInspecting = mascot.ringTargetId == id && mascot.arcaneIntensity > 0.01

            let curState: Float
            let curIntensity: Float
            if id == selectedNode {
                curState = 1; curIntensity = 0
            } else if searchMatched {
                curState = 4; curIntensity = 0
            } else if searchDimmed {
                // When search is active, suppress recall/arrival glow on non-matched nodes
                curState = 0; curIntensity = 0
            } else if isInspecting {
                curState = 5; curIntensity = mascot.arcaneIntensity
            } else if ri > 0 {
                curState = 2; curIntensity = ri
            } else if ai > 0 {
                curState = 3; curIntensity = ai
            } else {
                curState = 0; curIntensity = 0
            }

            let packedState: Float = curState + (searchDimmed ? 10.0 : 0.0) + curIntensity * 0.01

            let color: SIMD3<Float> = id == selectedNode
                ? SIMD3<Float>(1, 1, 1)
                : nodeColorFloat3(for: nodeData.project, colorMap: colorMap)

            instanceArray[idx] = NodeInstance(
                position: worldPos,
                scale: r,
                color: SIMD4<Float>(color.x, color.y, color.z, packedState)
            )
            idx += 1

            // Pack point lights into uniform buffer
            if (curState >= 1 && curState <= 5) && pointLightCount < 16 {
                let lightColor: SIMD3<Float>
                let intensity: Float
                let atten: Float
                if curState == 2 {
                    let pulse = 1.0 + sin(renderer.animationTime * 3.0) * 0.4
                    lightColor = SIMD3(0.7, 0.9, 1.0)
                    intensity = 3.0 * curIntensity * pulse
                    atten = 0.5
                } else if curState == 3 {
                    let pulse = 1.0 + sin(renderer.animationTime * 2.5) * 0.4
                    lightColor = SIMD3(1.0, 0.8, 0.3)
                    intensity = 3.0 * curIntensity * pulse
                    atten = 0.5
                } else if curState == 4 {
                    let pulse = 1.0 + sin(renderer.animationTime * 4.0) * 0.3
                    lightColor = SIMD3(0.0, 0.9, 1.0)
                    intensity = 4.0 * pulse
                    atten = 0.4
                } else if curState == 5 {
                    let pulse = 1.0 + sin(renderer.animationTime * 2.0) * 0.2
                    lightColor = SIMD3(0.2, 0.6, 0.9)
                    intensity = 2.0 * curIntensity * pulse
                    atten = 0.4
                } else {
                    lightColor = SIMD3(1, 1, 1)
                    intensity = 2.0
                    atten = 0.3
                }
                setPointLight(index: pointLightCount, position: worldPos, color: lightColor, intensity: intensity, attenuation: atten)
                pointLightCount += 1
            }
        }

        let actualNodeCount = idx
        renderer.lightingUniforms.pointLightCount = UInt32(pointLightCount)

        // Upload instance data to renderer
        renderer.ensureNodeBuffers(nodeCount: actualNodeCount)
        if let instanceBuf = renderer.nodeInstanceBuffer {
            instanceArray.withUnsafeBufferPointer { ptr in
                guard let base = ptr.baseAddress else { return }
                instanceBuf.contents().copyMemory(
                    from: base,
                    byteCount: actualNodeCount * MemoryLayout<NodeInstance>.stride
                )
            }
        }

        // Set stamp params
        if let paramsBuf = renderer.nodeStampParamsBuffer {
            let paramsPtr = paramsBuf.contents().bindMemory(to: StampParams.self, capacity: 1)
            paramsPtr.pointee = StampParams(
                vertsPerNode: UInt32(renderer.vertsPerSphere),
                nodeCount: UInt32(actualNodeCount)
            )
        }

        renderer.actualNodeCount = actualNodeCount
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

    private func packEdgeVertices() {
        let edges = renderEdges
        let edgeCount = edges.count
        guard edgeCount > 0 else {
            renderer.actualEdgeCount = 0
            renderer.actualEdgeVertexCount = 0
            return
        }

        let nodes = renderNodes
        let hubs = renderHubs
        let colorMap = renderColorMap
        let isSemanticMode = layoutMode == .embedding

        // Rebuild cached per-node data only when topology changes
        if nodes.count != lastTopologyNodeCount {
            cachedNodeProject.removeAll(keepingCapacity: true)
            cachedNodeRadii.removeAll(keepingCapacity: true)
            for node in nodes {
                cachedNodeProject[node.id] = node.project
                let isHub = hubs.contains(node.id)
                let importance = max(1, node.importance)
                let baseR: Float = isHub ? nodeRadius * 1.6 : nodeRadius
                cachedNodeRadii[node.id] = baseR * (1.0 + Float(importance - 1) * 0.08)
            }
            lastTopologyNodeCount = nodes.count
        }

        renderer.ensureEdgeBuffers(edgeCount: edgeCount)
        guard let instanceBuf = renderer.edgeInstanceBuffer else { return }

        let instances = instanceBuf.contents().bindMemory(to: EdgeInstance.self, capacity: edgeCount)

        let nodeProject = cachedNodeProject
        let nodeRadii = cachedNodeRadii
        var ei = 0

        for edge in edges {
            guard let from = positions[edge.sourceId],
                  let to = positions[edge.targetId] else { continue }

            let p1 = from * scaleFactor
            let p2 = to * scaleFactor
            let delta = p2 - p1
            let len = simd_length(delta)

            let r1 = nodeRadii[edge.sourceId] ?? nodeRadius
            let r2 = nodeRadii[edge.targetId] ?? nodeRadius
            guard len > r1 + r2 else { continue }

            let dir = delta / len
            let p1inset = p1 + dir * r1
            let p2inset = p2 - dir * r2

            let connected = edge.sourceId == selectedNode || edge.targetId == selectedNode
            let radius = connected ? edgeRadius * 2.5 : edgeRadius * 1.3

            let searchDimmed = isSearchActive
                && !searchMatchIds.contains(edge.sourceId)
                && !searchMatchIds.contains(edge.targetId)
            let state: Float
            if connected { state = 1 }
            else if searchDimmed { state = 2 }
            else if isSemanticMode { state = 3 }
            else { state = 0 }

            let color: SIMD3<Float>
            if connected {
                color = SIMD3<Float>(1, 1, 1)
            } else {
                color = edgeColorFloat3(for: nodeProject[edge.sourceId], colorMap: colorMap)
            }

            instances[ei] = EdgeInstance(
                sourcePos: p1inset,
                radius: radius,
                targetPos: p2inset,
                state: state,
                color: SIMD4<Float>(color.x, color.y, color.z, 1)
            )
            ei += 1
        }

        renderer.actualEdgeCount = ei
    }

    // MARK: - Label Instance Packing

    private func packLabelInstances() {
        let nodeCount = positions.count
        guard nodeCount > 0 else {
            renderer.actualLabelCount = 0
            return
        }

        let nodes = renderNodes
        let hubs = renderHubs
        let storeVersion = renderStore?.topologyVersion ?? 0

        // Regen atlas if needed — O(1) version check instead of O(n) Set comparison
        let atlasNeedsRegen = renderer.labelAtlasTexture == nil || storeVersion != lastAtlasTopologyVersion
        if atlasNeedsRegen {
            let currentNodeIds = Set(positions.keys)
            let currentProjects = Set(nodes.map(\.project))
            let isFirstAtlas = renderer.labelAtlasTexture == nil
            let framesSinceRegen = renderFrameCount &- labelAtlasRegenFrame
            if isFirstAtlas || framesSinceRegen >= 60 {
                if isFirstAtlas {
                    // First atlas must be synchronous — nothing to show until it's ready
                    generateLabelAtlas(nodes: nodes, hubs: hubs, projects: currentProjects)
                } else if !isAtlasGenerating {
                    // Subsequent rebuilds run async — old atlas stays visible until new one is ready
                    dispatchAtlasRegen(nodes: nodes, hubs: hubs, projects: currentProjects)
                }
                labelAtlasRegenFrame = renderFrameCount
                labelAtlasNodeIds = currentNodeIds
                lastAtlasTopologyVersion = storeVersion
            }
        }
        guard renderer.labelAtlasTexture != nil else { return }

        // Per-project centroids
        var projectNodePositions: [String: [SIMD3<Float>]] = [:]
        for node in nodes {
            guard let pos = positions[node.id] else { continue }
            projectNodePositions[node.project, default: []].append(pos)
        }
        projectCentroids.removeAll(keepingCapacity: true)
        for (project, pts) in projectNodePositions where pts.count >= 2 {
            var sum = SIMD3<Float>.zero
            var maxY: Float = -.greatestFiniteMagnitude
            for p in pts { sum += p; maxY = max(maxY, p.y) }
            let centroid = sum / Float(pts.count)
            var maxDist: Float = 0
            for p in pts { maxDist = max(maxDist, simd_length(p - centroid)) }
            projectCentroids[project] = (centroid: centroid, radius: maxDist, maxY: maxY)
        }

        var projectColors: [String: SIMD3<Float>] = [:]
        for project in projectCentroids.keys {
            projectColors[project] = nodeColorFloat3(for: project, colorMap: renderColorMap)
        }

        let nodeById = renderStore?.nodeById ?? [:]
        let sf = scaleFactor
        let aspectCorr = labelAtlasAspectCorrection

        var allPositions = positions
        for (id, pos) in expandedChildPositions {
            allPositions[id] = pos
        }
        var expandedChildren = Set<UUID>()
        for hubId in expandedHubs {
            for childId in childrenOfHub(hubId) {
                expandedChildren.insert(childId)
            }
        }

        let camPos = camera.cameraPosition
        var minDepth: Float = .greatestFiniteMagnitude
        var maxDepth: Float = 0
        for (_, pos) in allPositions {
            let d = simd_length(pos - camPos)
            minDepth = min(minDepth, d)
            maxDepth = max(maxDepth, d)
        }
        let depthRange = max(1.0, maxDepth - minDepth)

        let projectLabelCount = projectCentroids.count
        let maxNodeLabels = max(0, 2048 - projectLabelCount)
        var instances = [LabelInstance]()
        instances.reserveCapacity(min(allPositions.count + projectLabelCount, 2048))

        for (id, pos3D) in allPositions {
            guard instances.count < maxNodeLabels,
                  let nodeData = nodeById[id],
                  let rect = labelAtlasRects[id] else {
                continue
            }

            let isSelected = id == selectedNode
            let isHub = hubs.contains(id)
            let importance = Float(max(1, nodeData.importance))
            let isSearchMatch = isSearchActive && searchMatchIds.contains(id)
            let searchDimmed = isSearchActive && !searchMatchIds.contains(id)

            let maxVisible: Float = (isSelected || expandedChildren.contains(id) || isSearchMatch)
                ? .greatestFiniteMagnitude
                : (isHub ? 1200 : (200 + importance * 80))

            let baseOpacity: Float = isSelected ? 0.95 : (isHub ? 0.8 : 0.6)
            let halfH: Float = isSelected ? 0.025 : (isHub ? 0.022 : 0.018)
            let textAspect = (rect.u1 - rect.u0) / max(0.001, rect.v1 - rect.v0) * aspectCorr

            let anchor = pos3D * sf + SIMD3<Float>(0, nodeRadius * 1.8, 0)

            var flags: UInt32 = 0
            if isSelected { flags |= 1 }
            if isSearchMatch { flags |= 2 }
            if searchDimmed { flags |= 4 }

            instances.append(LabelInstance(
                anchor: anchor,
                halfH: halfH,
                uvRect: SIMD4<Float>(rect.u0, rect.v0, rect.u1, rect.v1),
                color: SIMD4<Float>(1, 1, 1, baseOpacity),
                textAspect: textAspect,
                maxVisible: maxVisible,
                forwardBias: 0,
                flags: flags
            ))
        }

        // Project labels
        for (project, centroidData) in projectCentroids {
            guard let rect = projectLabelAtlasRects[project] else { continue }

            let labelY = centroidData.maxY * sf + 0.15
            let anchor = SIMD3<Float>(centroidData.centroid.x * sf, labelY, centroidData.centroid.z * sf)
            let color = projectColors[project] ?? SIMD3<Float>(1, 1, 1)
            let textAspect = (rect.u1 - rect.u0) / max(0.001, rect.v1 - rect.v0) * aspectCorr

            instances.append(LabelInstance(
                anchor: anchor,
                halfH: 0.15,
                uvRect: SIMD4<Float>(rect.u0, rect.v0, rect.u1, rect.v1),
                color: SIMD4<Float>(color.x, color.y, color.z, 0.9),
                textAspect: textAspect,
                maxVisible: 12000,
                forwardBias: 0.25,
                flags: 0
            ))
        }

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
        let scaledCamPos = camPos * sf
        if let paramsBuf = renderer.labelStampParamsBuffer {
            let paramsPtr = paramsBuf.contents().bindMemory(to: LabelStampParams.self, capacity: 1)
            paramsPtr.pointee = LabelStampParams(
                cameraPos: scaledCamPos,
                minDepth: minDepth * sf,
                depthRange: depthRange * sf,
                labelCount: UInt32(actualLabelCount),
                _pad0: 0, _pad1: 0
            )
        }

        renderer.actualLabelCount = actualLabelCount
    }

    // MARK: - Label Atlas Generation (MTLTexture)

    private struct AtlasLabelEntry: Sendable {
        let id: UUID; let label: String; let isHub: Bool
    }

    private struct AtlasResult: @unchecked Sendable {
        let texture: MTLTexture
        let nodeRects: [UUID: (u0: Float, v0: Float, u1: Float, v1: Float)]
        let projectRects: [String: (u0: Float, v0: Float, u1: Float, v1: Float)]
        let aspectCorrection: Float
        let allocW: Int; let allocH: Int
    }

    /// Synchronous atlas generation (used for the first atlas only).
    private func generateLabelAtlas(nodes: [NodeData], hubs: Set<UUID>, projects: Set<String>) {
        let entries = nodes.map { AtlasLabelEntry(id: $0.id, label: $0.label, isHub: hubs.contains($0.id)) }
        guard let result = Self.renderLabelAtlas(
            entries: entries, sortedProjects: projects.sorted(), device: renderer.device
        ) else { return }
        applyAtlasResult(result, nodeIds: Set(nodes.map(\.id)), hubs: hubs, projects: projects)
        frameLog.info("[labelAtlas] \(result.nodeRects.count) node + \(result.projectRects.count) project labels, \(result.allocW/2)x\(result.allocH/2)")
    }

    /// Dispatch label atlas regeneration to a background thread. Old atlas stays visible until complete.
    private func dispatchAtlasRegen(nodes: [NodeData], hubs: Set<UUID>, projects: Set<String>) {
        isAtlasGenerating = true
        let entries = nodes.map { AtlasLabelEntry(id: $0.id, label: $0.label, isHub: hubs.contains($0.id)) }
        let sortedProjects = projects.sorted()
        let device = renderer.device
        let capturedNodeIds = Set(nodes.map(\.id))
        let capturedHubs = hubs
        let capturedProjects = projects

        Task.detached(priority: .userInitiated) { [weak self] in
            let result = Self.renderLabelAtlas(entries: entries, sortedProjects: sortedProjects, device: device)
            await MainActor.run { [weak self] in
                guard let self else { return }
                if let result {
                    self.applyAtlasResult(result, nodeIds: capturedNodeIds, hubs: capturedHubs, projects: capturedProjects)
                    frameLog.info("[labelAtlas] async regen: \(result.nodeRects.count) node + \(result.projectRects.count) project labels")
                }
                self.isAtlasGenerating = false
            }
        }
    }

    private func applyAtlasResult(_ result: AtlasResult, nodeIds: Set<UUID>, hubs: Set<UUID>, projects: Set<String>) {
        renderer.labelAtlasTexture = result.texture
        labelAtlasRects = result.nodeRects
        projectLabelAtlasRects = result.projectRects
        labelAtlasAspectCorrection = result.aspectCorrection
        labelAtlasAllocW = result.allocW
        labelAtlasAllocH = result.allocH
        labelAtlasNodeIds = nodeIds
        labelAtlasHubIds = hubs
        labelAtlasProjects = projects
    }

    /// Pure rendering work — can run on any thread. CoreText (via NSString.draw) is thread-safe.
    nonisolated private static func renderLabelAtlas(
        entries: [AtlasLabelEntry], sortedProjects: [String], device: MTLDevice
    ) -> AtlasResult? {
        let atlasW = 4096
        let padding: CGFloat = 4
        let fontSize: CGFloat = 28
        let projFontSize: CGFloat = 40

        struct LabelSize { let width: CGFloat; let height: CGFloat }
        var nodeSizes: [LabelSize] = []
        nodeSizes.reserveCapacity(entries.count)
        for entry in entries {
            let weight: NSFont.Weight = entry.isHub ? .bold : .medium
            let font = NSFont.monospacedSystemFont(ofSize: fontSize, weight: weight)
            let attrs: [NSAttributedString.Key: Any] = [.font: font, .foregroundColor: NSColor.white]
            let size = (entry.label as NSString).size(withAttributes: attrs)
            nodeSizes.append(LabelSize(width: ceil(size.width) + padding * 2, height: ceil(size.height) + padding))
        }

        let projFont = NSFont.systemFont(ofSize: projFontSize, weight: .bold)
        let projAttrs: [NSAttributedString.Key: Any] = [.font: projFont, .foregroundColor: NSColor.white]
        var projSizes: [LabelSize] = []
        for project in sortedProjects {
            let size = (project as NSString).size(withAttributes: projAttrs)
            projSizes.append(LabelSize(width: ceil(size.width) + padding * 2, height: ceil(size.height) + padding))
        }

        // Simulate packing
        var cursorX: CGFloat = 2
        var cursorY: CGFloat = 0
        var rowHeight: CGFloat = 0
        for s in nodeSizes {
            if cursorX + s.width > CGFloat(atlasW) { cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0 }
            cursorX += s.width + padding; rowHeight = max(rowHeight, s.height)
        }
        cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0
        for s in projSizes {
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

        // Draw node labels
        var rects: [UUID: (u0: Float, v0: Float, u1: Float, v1: Float)] = [:]
        cursorX = 2; cursorY = 0; rowHeight = 0

        for (i, entry) in entries.enumerated() {
            let s = nodeSizes[i]
            if cursorX + s.width > CGFloat(atlasW) { cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0 }
            if cursorY + s.height > CGFloat(uvAtlasH) { break }

            let weight: NSFont.Weight = entry.isHub ? .bold : .medium
            let font = NSFont.monospacedSystemFont(ofSize: fontSize, weight: weight)
            let attrs: [NSAttributedString.Key: Any] = [.font: font, .foregroundColor: NSColor.white]

            let drawPoint = CGPoint(x: cursorX + padding, y: cursorY + padding * 0.5)
            NSGraphicsContext.saveGraphicsState()
            NSGraphicsContext.current = NSGraphicsContext(cgContext: ctx, flipped: true)
            (entry.label as NSString).draw(at: drawPoint, withAttributes: attrs)
            NSGraphicsContext.restoreGraphicsState()

            rects[entry.id] = (
                Float(cursorX) / Float(uvAtlasW), Float(cursorY) / Float(uvAtlasH),
                Float(cursorX + s.width) / Float(uvAtlasW), Float(cursorY + s.height) / Float(uvAtlasH)
            )
            cursorX += s.width + padding; rowHeight = max(rowHeight, s.height)
        }

        // Draw project labels
        var projRects: [String: (u0: Float, v0: Float, u1: Float, v1: Float)] = [:]
        cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0

        for (i, project) in sortedProjects.enumerated() {
            let s = projSizes[i]
            if cursorX + s.width > CGFloat(atlasW) { cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0 }
            if cursorY + s.height > CGFloat(uvAtlasH) { break }

            let drawPoint = CGPoint(x: cursorX + padding, y: cursorY + padding * 0.5)
            NSGraphicsContext.saveGraphicsState()
            NSGraphicsContext.current = NSGraphicsContext(cgContext: ctx, flipped: true)
            (project as NSString).draw(at: drawPoint, withAttributes: projAttrs)
            NSGraphicsContext.restoreGraphicsState()

            projRects[project] = (
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
            aspectCorrection: aspectCorrection, allocW: allocW, allocH: allocH
        )
    }

    // MARK: - Nebulae

    private func updateNebulae() {
        let groups = nebulaFog.nebulaGroupsForCurrentMode(
            layoutMode: layoutMode,
            positions: positions,
            nodes: renderNodes,
            semanticClusters3D: semanticClusters3D
        )
        nebulaFog.updateRenderer(renderer: renderer, groups: groups, colorMap: renderColorMap)
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
            let newTarget = camera.hitTest(at: center, viewSize: renderViewSize, positions: positions)
            if newTarget != reticleTarget {
                reticleTarget = newTarget
                reticleCallback?(newTarget)
            }
        }
    }

    // MARK: - Teleport

    func teleportToNextProject(direction: Int) {
        camera.teleportToNextProject(positions: positions, nodes: renderNodes,
                                     hubs: renderHubs, direction: direction)
        teleportLabel = camera.teleportLabel
        teleportCounter = camera.teleportCounter
    }

    // MARK: - Hit Testing

    func hitTest(at location: CGPoint, viewSize: CGSize) -> UUID? {
        camera.hitTest(at: location, viewSize: viewSize, positions: positions)
    }

    /// Hit test the mascot — returns true if the tap is within 50px of the mascot's screen position.
    /// Note: mascot.currentPosition is already in scaled world space, but camera.project()
    /// applies scaleFactor internally, so we must unscale first to avoid double-scaling.
    func hitTestMascot(at location: CGPoint, viewSize: CGSize) -> Bool {
        let sf = camera.scaleFactor
        guard sf > 0 else { return false }
        let pos = mascot.currentPosition / sf
        guard let screenPos = camera.project(point3D: pos, viewSize: viewSize) else { return false }
        let dist = hypot(location.x - screenPos.x, location.y - screenPos.y)
        return dist < 50
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
