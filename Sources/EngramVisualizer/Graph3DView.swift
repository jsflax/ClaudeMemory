import SwiftUI
import RealityKit
import GameController
import Metal
import simd
import os

private let frameLog = Logger(subsystem: "com.claudememory.visualizer", category: "3DFrameTiming")

// MARK: - 3D Scene Manager

/// Manages RealityKit entity hierarchy for the 3D graph.
@Observable
@MainActor
final class Graph3DScene {
    private var rootEntity = Entity()
    private var cameraEntity: PerspectiveCamera?
    private var edgeContainer = Entity()

    // Camera state — inputs write to target* values, updateCamera() lerps toward them.
    // This smooths out irregular input event timing for silky camera motion.

    // Target state (written by gestures/events)
    var targetAzimuth: Float = 0
    var targetElevation: Float = 0.3
    var targetCameraPos: SIMD3<Float> = .zero  // the look-at point

    // Smoothed state (lerped each frame, used for rendering)
    // @ObservationIgnored: written every frame by updateCamera(), only read within
    // renderTick and gesture handlers — never by SwiftUI body evaluation.
    @ObservationIgnored private(set) var azimuth: Float = 0
    @ObservationIgnored private(set) var elevation: Float = 0.3
    @ObservationIgnored private(set) var cameraTarget: SIMD3<Float> = .zero

    var orbitRadius: Float = 2.0 // orbit arm length — set by centerOnGraph to fit the scene
    private let smoothing: Float = 0.2  // lerp factor per frame (0=frozen, 1=instant)
    var isDragging = false

    /// Camera position in internal coordinates (cameraTransform applies scaleFactor).
    var cameraPosition: SIMD3<Float> {
        let x = orbitRadius * cos(elevation) * sin(azimuth)
        let y = orbitRadius * sin(elevation)
        let z = orbitRadius * cos(elevation) * cos(azimuth)
        return cameraTarget + SIMD3(x, y, z)
    }

    /// Camera forward direction (from camera toward target).
    var forward: SIMD3<Float> {
        normalize(cameraTarget - cameraPosition)
    }

    /// Camera right vector.
    var right: SIMD3<Float> {
        normalize(cross(forward, SIMD3<Float>(0, 1, 0)))
    }

    /// Camera up vector.
    var up: SIMD3<Float> {
        cross(right, forward)
    }

    // Metal shaders for CustomMaterial — all visual effects run on GPU
    private var metalDevice: MTLDevice?
    private var metalLibrary: MTLLibrary?
    private var edgeSurfaceShader: CustomMaterial.SurfaceShader?

    // Batched node rendering — all nodes as a single LowLevelMesh (1 draw call).
    // Metal compute kernel stamps template sphere vertices at each node position.
    private var nodeBatchMesh: LowLevelMesh?
    private var nodeBatchEntity: ModelEntity?
    private var nodeBatchMaterial: CustomMaterial?
    private var nodeBatchCapacity: Int = 0
    private var nodeColorCache: [String: SIMD3<Float>] = [:]

    // Metal compute pipeline for sphere instancing
    // stamp_node_spheres compute pipeline removed — CPU stamping is faster for <1K nodes
    // (avoids command buffer encode + submit + fence wait overhead)
    // sphereTemplateBuffer removed — using sphereTemplateVertices Swift array instead     // template sphere vertices
    private var sphereTemplateIndices: [UInt32] = [] // template sphere index data
    private var sphereTemplateVertices: [(pos: SIMD3<Float>, norm: SIMD3<Float>)] = []
    private var vertsPerSphere: Int = 0
    private var indicesPerSphere: Int = 0
    private var instanceArray: [NodeInstance] = []     // per-node instance data (CPU staging)

    // Point lights for glowing/arriving/search-matched nodes
    private var pointLightEntities: [Int64: Entity] = [:]

    // Dying node entities — individual PBR entities with red flash animation (max ~5 at a time)
    private var dyingNodeEntities: [Int64: ModelEntity] = [:]
    private var dyingNodePos3D: [Int64: SIMD3<Float>] = [:]

    // Batched edge rendering — all edges as a single LowLevelMesh (1 draw call)
    private var edgeBatchMesh: LowLevelMesh?
    private var edgeBatchEntity: ModelEntity?
    private var edgeBatchMaterial: CustomMaterial?
    private var edgeBatchCapacity: Int = 0
    private var edgeColorCache: [String: SIMD3<Float>] = [:]

    // Track last mesh part index counts to avoid calling parts.replaceAll() every frame.
    // Calling replaceAll forces RealityKit to rebuild internal structures — can cause 1-frame blanks.
    // Using capacity-based counts so replaceAll only fires on mesh creation/resize.
    @ObservationIgnored private var lastNodePartIndexCount: Int = -1
    @ObservationIgnored private var lastEdgePartIndexCount: Int = -1

    // Batched label rendering — all labels as billboard quads in a single LowLevelMesh (1 draw call)
    private var labelBatchMesh: LowLevelMesh?
    private var labelBatchEntity: ModelEntity?
    private var labelBatchMaterial: CustomMaterial?
    private var labelBatchCapacity: Int = 0
    @ObservationIgnored private var lastLabelPartIndexCount: Int = -1
    private var labelAtlasTexture: TextureResource?
    private var labelAtlasRects: [Int64: (u0: Float, v0: Float, u1: Float, v1: Float)] = [:]
    private var labelAtlasNodeIds: Set<Int64> = []
    private var labelAtlasHubIds: Set<Int64> = []
    // Project labels — larger billboard quads floating above each project cluster
    private var projectLabelAtlasRects: [String: (u0: Float, v0: Float, u1: Float, v1: Float)] = [:]
    private var labelAtlasProjects: Set<String> = []
    @ObservationIgnored private var projectCentroids: [String: (centroid: SIMD3<Float>, radius: Float, maxY: Float)] = [:]
    /// Correction factor for text aspect ratio: atlasW / (2 * atlasH).
    /// Quad spans -halfW..+halfW (2x) horizontally but 0..halfH (1x) vertically.
    @ObservationIgnored private var labelAtlasAspectCorrection: Float = 1.0
    /// Debounce atlas regeneration to prevent flicker from texture/UV mismatch
    /// in RealityKit's triple-buffered LowLevelMesh vertex buffers.
    @ObservationIgnored private var labelAtlasRegenFrame: UInt64 = 0

    /// 48-byte vertex layout shared by batched nodes and edges.
    private struct BatchVertex {
        var px: Float, py: Float, pz: Float     // position
        var nx: Float, ny: Float, nz: Float     // normal
        var u: Float, v: Float                   // uv0
        var cr: Float, cg: Float, cb: Float, ca: Float  // color
    }

    /// Per-node instance data for the Metal compute stamping kernel (32 bytes).
    private struct NodeInstance {
        var px: Float, py: Float, pz: Float  // world position
        var scale: Float                      // sphere radius
        var cr: Float, cg: Float, cb: Float  // color
        var packed: Float                     // packedState
    }

    /// Generate a UV sphere template. Returns (vertices, indices).
    /// vertices: array of (position, normal) pairs for a unit sphere.
    private static func generateSphereTemplate(segments: Int = 10, rings: Int = 6)
        -> (vertices: [(SIMD3<Float>, SIMD3<Float>)], indices: [UInt32])
    {
        var vertices: [(SIMD3<Float>, SIMD3<Float>)] = []
        var indices: [UInt32] = []

        // Top pole
        vertices.append((.init(0, 1, 0), .init(0, 1, 0)))

        // Ring vertices
        for ring in 1..<rings {
            let phi = Float.pi * Float(ring) / Float(rings)
            let sinPhi = sin(phi)
            let cosPhi = cos(phi)
            for seg in 0..<segments {
                let theta = 2.0 * Float.pi * Float(seg) / Float(segments)
                let pos = SIMD3<Float>(sinPhi * cos(theta), cosPhi, sinPhi * sin(theta))
                vertices.append((pos, pos)) // unit sphere: normal = position
            }
        }

        // Bottom pole
        vertices.append((.init(0, -1, 0), .init(0, -1, 0)))

        // Top cap
        for seg in 0..<segments {
            let next = (seg + 1) % segments
            indices.append(0)
            indices.append(UInt32(1 + seg))
            indices.append(UInt32(1 + next))
        }

        // Body quads
        for ring in 0..<(rings - 2) {
            let ringStart = 1 + ring * segments
            let nextStart = 1 + (ring + 1) * segments
            for seg in 0..<segments {
                let next = (seg + 1) % segments
                let a = UInt32(ringStart + seg)
                let b = UInt32(ringStart + next)
                let c = UInt32(nextStart + seg)
                let d = UInt32(nextStart + next)
                indices.append(contentsOf: [a, c, b, b, c, d])
            }
        }

        // Bottom cap
        let bottomPole = UInt32(vertices.count - 1)
        let lastStart = 1 + (rings - 2) * segments
        for seg in 0..<segments {
            let next = (seg + 1) % segments
            indices.append(bottomPole)
            indices.append(UInt32(lastStart + next))
            indices.append(UInt32(lastStart + seg))
        }

        return (vertices, indices)
    }

    @ObservationIgnored var animationTime: Float = 0

    // Shared meshes — created once, reused for all entities
    private let nodeMesh = MeshResource.generateSphere(radius: 1.0)

    // Nebula particle emitter container
    private var nebulaContainer = Entity()
    /// Soft gaussian circle texture for nebula particles — generated once, reused.
    private var softParticleTexture: TextureResource?

    private let scaleFactor: Float = 1.0 / 200.0
    private let nodeRadius: Float = 0.04
    private let edgeRadius: Float = 0.004

    private struct EdgeKey: Hashable {
        let source: Int64
        let target: Int64
    }

    /// Generate a soft radial gradient circle texture for nebula particles.
    /// Uses raw RGBA bitmap to avoid pre-multiplied alpha issues that cause dark fringes.
    private func generateSoftCircleTexture(size: Int = 64) -> TextureResource? {
        var pixels = [UInt8](repeating: 0, count: size * size * 4)  // RGBA, all zeros = transparent black
        let center = Float(size) / 2.0
        for y in 0..<size {
            for x in 0..<size {
                let dx = Float(x) - center + 0.5
                let dy = Float(y) - center + 0.5
                let dist = sqrt(dx * dx + dy * dy)
                let norm = dist / center  // 0 at center, 1 at edge
                // Quartic falloff: (1-d²)² — much softer edges than quadratic.
                // Spends more time near zero, making individual particle edges invisible.
                let q = max(0.0, 1.0 - norm * norm)
                let alpha = q * q  // quartic: very soft boundary
                let idx = (y * size + x) * 4
                // Non-premultiplied: RGB = white, A = alpha
                let a8 = UInt8(min(255, alpha * 255))
                pixels[idx + 0] = 255  // R
                pixels[idx + 1] = 255  // G
                pixels[idx + 2] = 255  // B
                pixels[idx + 3] = a8   // A
            }
        }
        // Create CGImage from raw pixel data
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let provider = CGDataProvider(data: Data(pixels) as CFData),
              let cgImage = CGImage(
                  width: size, height: size,
                  bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: size * 4,
                  space: colorSpace,
                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                  provider: provider, decode: nil, shouldInterpolate: true,
                  intent: .defaultIntent
              ) else { return nil }
        return try? TextureResource.generate(from: cgImage, options: .init(semantic: .raw))
    }

    func setup() -> (root: Entity, camera: PerspectiveCamera) {
        rootEntity = Entity()
        softParticleTexture = generateSoftCircleTexture()
        nebulaContainer = Entity()
        nebulaContainer.name = "nebulae"
        rootEntity.addChild(nebulaContainer)
        edgeContainer = Entity()
        edgeContainer.name = "edges"
        rootEntity.addChild(edgeContainer)

        flowParticleContainer = Entity()
        flowParticleContainer.name = "flow_particles"
        rootEntity.addChild(flowParticleContainer)

        // Initialize Metal device, shaders, and compute pipelines
        if let device = MTLCreateSystemDefaultDevice(),
           let library = device.makeDefaultLibrary() {
            metalDevice = device
            metalLibrary = library
            frameLog.error("[setup] Metal device=\(device.name), library functionNames=\(library.functionNames)")
            edgeSurfaceShader = CustomMaterial.SurfaceShader(named: "edge_surface", in: library)

            // Node batch: CustomMaterial with lit PBR + geometry modifier for scale pulse.
            // Real sphere geometry in a single LowLevelMesh — per-vertex color carries state.
            let nodeBatchSurf = CustomMaterial.SurfaceShader(named: "node_batch_lit_surface", in: library)
            let nodeBatchGeom = CustomMaterial.GeometryModifier(named: "node_batch_geometry", in: library)
            do {
                var mat = try CustomMaterial(surfaceShader: nodeBatchSurf,
                                             geometryModifier: nodeBatchGeom,
                                             lightingModel: .lit)
                // OPAQUE — no transparent blending. Fog/dimming baked into base color
                // by the shader. Enables early-Z rejection and avoids per-fragment sorting.
                mat.baseColor.tint = .white  // color comes from vertex data
                nodeBatchMaterial = mat
                frameLog.error("[setup] ✅ nodeBatchMaterial created (.lit, OPAQUE)")
            } catch {
                frameLog.error("[setup] ❌ nodeBatchMaterial FAILED: \(error)")
            }

            // Generate sphere template — stored as Swift array for CPU stamping
            let template = Self.generateSphereTemplate(segments: 16, rings: 10)
            vertsPerSphere = template.vertices.count
            indicesPerSphere = template.indices.count
            sphereTemplateIndices = template.indices
            sphereTemplateVertices = template.vertices
            frameLog.error("[setup] sphere template: \(self.vertsPerSphere) verts, \(self.indicesPerSphere) indices")

            // Edge batch material
            if let surf = edgeSurfaceShader {
                do {
                    var mat = try CustomMaterial(surfaceShader: surf, lightingModel: .unlit)
                    mat.blending = .transparent(opacity: .init(floatLiteral: 1.0))
                    mat.baseColor.tint = .white
                    mat.faceCulling = .none
                    edgeBatchMaterial = mat
                    frameLog.error("[setup] ✅ edgeBatchMaterial created (TRANSPARENT)")
                } catch {
                    frameLog.error("[setup] ❌ edgeBatchMaterial FAILED: \(error)")
                }
            }

            // Label batch material — unlit billboard quads with texture atlas
            let labelSurf = CustomMaterial.SurfaceShader(named: "label_surface", in: library)
            let labelGeom = CustomMaterial.GeometryModifier(named: "label_geometry", in: library)
            do {
                var mat = try CustomMaterial(
                    surfaceShader: labelSurf,
                    geometryModifier: labelGeom,
                    lightingModel: .unlit)
                mat.blending = .transparent(opacity: .init(floatLiteral: 1.0))
                mat.faceCulling = .none
                labelBatchMaterial = mat
                frameLog.error("[setup] ✅ labelBatchMaterial created (TRANSPARENT)")
            } catch {
                frameLog.error("[setup] ❌ labelBatchMaterial FAILED: \(error)")
            }
        } else {
            frameLog.error("[setup] ❌ Metal init FAILED: no device or no default library")
        }

        // World-space lighting — dramatic, fixed in space so shading changes
        // as you orbit around nodes. High contrast: strong key, dim fill.

        // Key light: bright, from upper-right. Creates strong lit/shadow split.
        let keyLight = Entity()
        keyLight.components.set(DirectionalLightComponent(
            color: .white, intensity: 4000, isRealWorldProxy: false
        ))
        keyLight.look(at: .zero, from: SIMD3(3, 5, 2), relativeTo: nil)
        rootEntity.addChild(keyLight)

        // Fill light: very dim, opposite side. Prevents pure black shadows.
        let fillLight = Entity()
        fillLight.components.set(DirectionalLightComponent(
            color: .init(red: 0.4, green: 0.5, blue: 0.7, alpha: 1), intensity: 600, isRealWorldProxy: false
        ))
        fillLight.look(at: .zero, from: SIMD3(-4, 0, -1), relativeTo: nil)
        rootEntity.addChild(fillLight)

        // Rim light: from below-behind for edge definition.
        let rimLight = Entity()
        rimLight.components.set(DirectionalLightComponent(
            color: .init(red: 0.6, green: 0.6, blue: 0.8, alpha: 1), intensity: 1000, isRealWorldProxy: false
        ))
        rimLight.look(at: .zero, from: SIMD3(0, -3, -4), relativeTo: nil)
        rootEntity.addChild(rimLight)

        let camera = PerspectiveCamera()
        camera.camera.fieldOfViewInDegrees = 60
        cameraEntity = camera

        return (rootEntity, camera)
    }

    /// Strafe: move camera + target in the screen plane (two-finger scroll).
    func pan(dx: Float, dy: Float) {
        let speed: Float = 1.5
        let offset = -right * dx * speed + up * dy * speed
        targetCameraPos += offset
    }

    /// Dolly: move camera + target forward/backward along the view direction.
    /// Toward the cursor if `cursorNX`/`cursorNY` are provided (normalized -1…1).
    func dolly(amount: Float, cursorNX: Float = 0, cursorNY: Float = 0) {
        let fovTan: Float = tan(Float.pi / 6)  // half of 60° FOV
        let dir = normalize(forward + right * cursorNX * fovTan + up * cursorNY * fovTan)
        targetCameraPos += dir * amount
    }

    /// Rotate the view direction while keeping the camera position fixed.
    /// Adjusts the target point so the orbit offset change is compensated.
    /// This gives "look around" (turn head) behavior instead of "orbit around target."
    func lookRotate(deltaAz: Float, deltaEl: Float) {
        // Compute current camera position from target-space values
        let oldOx = orbitRadius * cos(targetElevation) * sin(targetAzimuth)
        let oldOy = orbitRadius * sin(targetElevation)
        let oldOz = orbitRadius * cos(targetElevation) * cos(targetAzimuth)
        let camPos = targetCameraPos + SIMD3(oldOx, oldOy, oldOz)

        // Apply angle changes
        targetAzimuth += deltaAz
        targetElevation = max(-.pi/2 + 0.1, min(.pi/2 - 0.1, targetElevation + deltaEl))

        // Recompute orbit offset with new angles
        let newOx = orbitRadius * cos(targetElevation) * sin(targetAzimuth)
        let newOy = orbitRadius * sin(targetElevation)
        let newOz = orbitRadius * cos(targetElevation) * cos(targetAzimuth)

        // Move target so camera stays at the same position
        targetCameraPos = camPos - SIMD3(newOx, newOy, newOz)
    }

    /// Snapshot current look state for drag-based look-around.
    func captureLookState() -> (azimuth: Float, elevation: Float, camPos: SIMD3<Float>) {
        isDragging = true
        let az = targetAzimuth
        let el = targetElevation
        let offset = orbitOffset(azimuth: az, elevation: el)
        return (azimuth: az, elevation: el, camPos: targetCameraPos + offset)
    }

    /// End drag — re-enable camera lerp and centering.
    func endDrag() {
        isDragging = false
    }

    /// Apply a drag-based look-around from a captured start state.
    /// Only sets TARGET values — smoothed values are updated atomically with the
    /// camera entity transform inside updateCamera(dt:) to keep labels and entities in sync.
    func applyLookDrag(start: (azimuth: Float, elevation: Float, camPos: SIMD3<Float>),
                       dx: Float, dy: Float) {
        let newAz = start.azimuth + dx
        let newEl = max(-.pi/2 + 0.1, min(.pi/2 - 0.1, start.elevation + dy))
        let offset = orbitOffset(azimuth: newAz, elevation: newEl)
        let newTarget = start.camPos - offset
        targetAzimuth = newAz
        targetElevation = newEl
        targetCameraPos = newTarget
    }

    /// Compute the orbit offset vector for given angles.
    private func orbitOffset(azimuth az: Float, elevation el: Float) -> SIMD3<Float> {
        SIMD3(orbitRadius * cos(el) * sin(az),
              orbitRadius * sin(el),
              orbitRadius * cos(el) * cos(az))
    }

    /// Distance from camera to a world-space point.
    func cameraDistance(to point: SIMD3<Float>) -> Float {
        simd_length(point - cameraPosition)
    }

    // MARK: - Gamepad

    /// Node currently under the center reticle (nil when no controller or nothing targeted).
    var reticleTarget: Int64?

    /// Tracks whether A was pressed last frame to detect rising edge.
    private var prevButtonA = false
    /// Tracks whether B was pressed last frame to detect rising edge.
    private var prevButtonB = false
    /// Tracks bumper state for rising-edge cycle detection.
    private var prevLB = false
    private var prevRB = false
    /// Tracks X/Y buttons for project teleport (Y = next, X = previous).
    private var prevButtonX = false
    private var prevButtonY = false
    /// Held keyboard keys for WASD/IJKL continuous movement.
    var heldKeys: Set<String> = []
    /// Current project index for teleport cycling.
    private var teleportProjectIndex = 0
    var teleportCounter: Int = 0
    /// Name of the project we last teleported to (shown briefly in overlay).
    var teleportLabel: String?

    /// Poll held keyboard keys and apply continuous camera movement. Called each render frame.
    /// WASD = move (dolly/strafe), IJKL = look, QE = rise/descend, Shift = sprint.
    func pollKeyboard(dt: Float) {
        guard !heldKeys.isEmpty else { return }

        let sprint: Float = heldKeys.contains("shift") ? 3.0 : 1.0
        let moveSpeed: Float = 200 * dt * sprint
        let strafeSpeed: Float = 300 * dt * sprint
        let vertSpeed: Float = 300 * dt * sprint
        let lookSpeed: Float = 2.0 * dt

        // WASD → dolly forward/back, strafe left/right
        if heldKeys.contains("w") { dolly(amount: moveSpeed) }
        if heldKeys.contains("s") { dolly(amount: -moveSpeed) }
        if heldKeys.contains("a") { pan(dx: strafeSpeed, dy: 0) }
        if heldKeys.contains("d") { pan(dx: -strafeSpeed, dy: 0) }

        // QE → rise / descend
        if heldKeys.contains("e") { pan(dx: 0, dy: vertSpeed) }
        if heldKeys.contains("q") { pan(dx: 0, dy: -vertSpeed) }

        // IJKL → look around
        if heldKeys.contains("i") { lookRotate(deltaAz: 0, deltaEl: lookSpeed) }
        if heldKeys.contains("k") { lookRotate(deltaAz: 0, deltaEl: -lookSpeed) }
        if heldKeys.contains("j") { lookRotate(deltaAz: lookSpeed, deltaEl: 0) }
        if heldKeys.contains("l") { lookRotate(deltaAz: -lookSpeed, deltaEl: 0) }
    }

    /// Poll gamepad and apply inputs to camera. Called each frame before updateCamera.
    /// FPS-style: L-stick = move, R-stick = look, triggers = rise/descend, bumpers = cycle nodes.
    func pollGamepad(dt: Float, selectedNode: inout Int64?,
                     positions: [Int64: SIMD3<Float>], viewSize: CGSize) {
        guard let pad = GCController.current?.extendedGamepad else {
            reticleTarget = nil
            return
        }

        // L3 or R3 (stick click) → sprint modifier (3x movement speed)
        let sprint: Float = (pad.leftThumbstickButton?.isPressed == true ||
                             pad.rightThumbstickButton?.isPressed == true) ? 3.0 : 1.0

        // Left stick → FPS move: Y = dolly forward/back, X = strafe left/right
        let lx = pad.leftThumbstick.xAxis.value
        let ly = pad.leftThumbstick.yAxis.value
        if abs(lx) > 0.1 || abs(ly) > 0.1 {
            let strafeSpeed: Float = 300 * dt * sprint
            let dollySpeed: Float = 200 * dt * sprint
            pan(dx: -lx * strafeSpeed, dy: 0)
            dolly(amount: ly * dollySpeed)
        }

        // Right stick → look around (camera stays in place, view direction changes)
        let rx = pad.rightThumbstick.xAxis.value
        let ry = pad.rightThumbstick.yAxis.value
        if abs(rx) > 0.1 || abs(ry) > 0.1 {
            let lookSpeed: Float = 2.0 * dt
            lookRotate(deltaAz: -rx * lookSpeed, deltaEl: ry * lookSpeed)
        }

        // Triggers → rise / descend (vertical movement in screen plane)
        let rt = pad.rightTrigger.value
        let lt = pad.leftTrigger.value
        let vertInput = rt - lt
        if abs(vertInput) > 0.05 {
            let vertSpeed: Float = 300 * dt
            pan(dx: 0, dy: vertInput * vertSpeed)
        }

        // Reticle targeting — hit test at screen center each frame
        // Guard the write to avoid @Observable spam (every write triggers SwiftUI re-evaluation
        // of reticleOverlay, which builds a nodeById dictionary from all nodes)
        let center = CGPoint(x: viewSize.width / 2, y: viewSize.height / 2)
        let newTarget = hitTest(at: center, viewSize: viewSize, positions: positions)
        if newTarget != reticleTarget { reticleTarget = newTarget }

        // A / Cross → select targeted node, or toggle hub expansion (rising edge only)
        let buttonA = pad.buttonA.isPressed
        if buttonA && !prevButtonA {
            if let target = reticleTarget {
                if renderHubs.contains(target) {
                    // Collapse other expanded hubs first
                    for hubId in expandedHubs where hubId != target {
                        toggleHubExpansion(hubId: hubId)
                        pendingHubToggles.append((hubId: hubId, expanding: false))
                    }
                    let expanding = !expandedHubs.contains(target)
                    toggleHubExpansion(hubId: target)
                    pendingHubToggles.append((hubId: target, expanding: expanding))
                    selectedNode = target
                } else {
                    // Collapse all expanded hubs when selecting non-hub
                    for hubId in expandedHubs {
                        toggleHubExpansion(hubId: hubId)
                        pendingHubToggles.append((hubId: hubId, expanding: false))
                    }
                    selectedNode = selectedNode == target ? nil : target
                }
            }
        }
        prevButtonA = buttonA

        // B / Circle → deselect + collapse all hubs (rising edge only)
        let buttonB = pad.buttonB.isPressed
        if buttonB && !prevButtonB {
            for hubId in expandedHubs {
                toggleHubExpansion(hubId: hubId)
                pendingHubToggles.append((hubId: hubId, expanding: false))
            }
            selectedNode = nil
        }
        prevButtonB = buttonB

        // Bumpers → cycle through nearby visible nodes (rising edge)
        let lb = pad.leftShoulder.isPressed
        let rb = pad.rightShoulder.isPressed
        if (lb && !prevLB) || (rb && !prevRB) {
            let direction: Int = (rb && !prevRB) ? 1 : -1
            selectedNode = cycleNode(current: selectedNode, direction: direction,
                                     positions: positions, viewSize: viewSize)
        }
        prevLB = lb
        prevRB = rb

        // X / Square → teleport to previous project (rising edge)
        let buttonX = pad.buttonX.isPressed
        if buttonX && !prevButtonX {
            teleportToNextProject(positions: positions, nodes: renderNodes, direction: -1)
        }
        prevButtonX = buttonX

        // Y / Triangle → teleport to next project (rising edge)
        let buttonY = pad.buttonY.isPressed
        if buttonY && !prevButtonY {
            teleportToNextProject(positions: positions, nodes: renderNodes, direction: 1)
        }
        prevButtonY = buttonY
    }

    /// Teleport camera to the hub node of the next project (falls back to centroid).
    func teleportToNextProject(positions: [Int64: SIMD3<Float>], nodes: [NodeData], direction: Int = 1) {
        // Group node IDs and positions by project
        var projectNodeIds: [String: [Int64]] = [:]
        var projectPositions: [String: [SIMD3<Float>]] = [:]
        let nodeById = Dictionary(nodes.map { ($0.id, $0) }, uniquingKeysWith: { _, last in last })
        for (id, pos) in positions {
            guard let node = nodeById[id] else { continue }
            projectNodeIds[node.project, default: []].append(id)
            projectPositions[node.project, default: []].append(pos)
        }

        // Sort projects alphabetically for stable cycling
        let projects = projectPositions.keys.sorted()
        guard !projects.isEmpty else { return }

        // Advance to next/previous project
        teleportProjectIndex = (teleportProjectIndex + direction + projects.count) % projects.count
        let project = projects[teleportProjectIndex]
        let pts = projectPositions[project]!
        let nodeIds = projectNodeIds[project]!

        // Prefer hub node (target of part_of edges) over centroid
        let hubId = nodeIds.first(where: { renderHubs.contains($0) })

        let targetPos: SIMD3<Float>
        if let hubId, let pos = positions[hubId] {
            targetPos = pos
        } else {
            // Fallback: centroid
            var sum = SIMD3<Float>.zero
            for p in pts { sum += p }
            targetPos = sum / Float(pts.count)
        }

        // Compute the project's bounding sphere to set an appropriate orbit radius
        var maxSpread: Float = 0
        for p in pts {
            maxSpread = max(maxSpread, simd_length(p - targetPos))
        }
        // Orbit radius: close to the hub so we're "among" the nodes.
        // Cap at 250 so even large clusters don't zoom too far out.
        // Minimum 60 for single-node projects (0.3 RealityKit units ≈ 7.5 node radii).
        orbitRadius = max(min(maxSpread * 0.4, 250), 60)

        // Teleport: set camera target to UNSCALED position.
        // cameraTransform() applies scaleFactor once, placing the orbit center at
        // targetPos * scaleFactor — exactly where the node entity lives in RealityKit.
        // (Previously this was targetPos * scaleFactor, which got double-scaled by
        // cameraTransform, putting the orbit center near the origin.)
        targetCameraPos = targetPos
        cameraTarget = targetPos

        teleportCounter += 1
        teleportLabel = project

        #if DEBUG
        // Write teleport verification data for UI tests
        let camRK = (targetPos + SIMD3(orbitRadius, 0, 0)) * scaleFactor
        let nodeRK = targetPos * scaleFactor
        let dist = simd_length(camRK - nodeRK)
        let line = "\(project),\(targetPos.x),\(targetPos.y),\(targetPos.z),\(orbitRadius),\(dist),\(pts.count)\n"
        let logPath = "/tmp/teleport-log.csv"
        if !FileManager.default.fileExists(atPath: logPath) {
            let header = "project,target_x,target_y,target_z,orbit_radius,rk_distance,node_count\n"
            FileManager.default.createFile(atPath: logPath, contents: header.data(using: .utf8))
        }
        if let fh = FileHandle(forWritingAtPath: logPath) {
            fh.seekToEndOfFile()
            fh.write(line.data(using: .utf8)!)
            fh.closeFile()
        }
        #endif
    }

    /// Smoothly drive the camera to a named project's cluster.
    func driveToProject(_ project: String) {
        let positions = renderPositions
        let nodes = renderNodes
        let nodeById = Dictionary(nodes.map { ($0.id, $0) }, uniquingKeysWith: { _, last in last })

        var pts: [SIMD3<Float>] = []
        var nodeIds: [Int64] = []
        for (id, pos) in positions {
            guard let node = nodeById[id], node.project == project else { continue }
            pts.append(pos)
            nodeIds.append(id)
        }
        guard !pts.isEmpty else { return }

        // Prefer hub node over centroid
        let hubId = nodeIds.first(where: { renderHubs.contains($0) })
        let targetPos: SIMD3<Float>
        if let hubId, let pos = positions[hubId] {
            targetPos = pos
        } else {
            var sum = SIMD3<Float>.zero
            for p in pts { sum += p }
            targetPos = sum / Float(pts.count)
        }

        // Bounding sphere for orbit radius
        var maxSpread: Float = 0
        for p in pts { maxSpread = max(maxSpread, simd_length(p - targetPos)) }
        orbitRadius = max(min(maxSpread * 0.4, 250), 60)

        // Only set targetCameraPos — cameraTarget lerps toward it for a smooth drive
        targetCameraPos = targetPos

        teleportCounter += 1
        teleportLabel = project
    }

    /// Cycle through nodes sorted by distance from camera.
    /// direction: +1 = next farther, -1 = next closer.
    private func cycleNode(current: Int64?, direction: Int,
                           positions: [Int64: SIMD3<Float>], viewSize: CGSize) -> Int64? {
        // Only consider nodes visible on screen
        let visible = positions.compactMap { (id, pos) -> (Int64, Float)? in
            guard project(point3D: pos, viewSize: viewSize) != nil else { return nil }
            return (id, cameraDistance(to: pos))
        }.sorted { $0.1 < $1.1 }

        guard !visible.isEmpty else { return current }

        if let current, let idx = visible.firstIndex(where: { $0.0 == current }) {
            let next = idx + direction
            if next >= 0 && next < visible.count {
                return visible[next].0
            }
            // Wrap around
            return direction > 0 ? visible.first?.0 : visible.last?.0
        }
        // Nothing selected — pick the closest
        return visible.first?.0
    }

    /// Smooth camera state toward targets and apply transform. Called each frame.
    /// Update camera with delta-time-aware smoothing.
    /// `dt` is seconds since last render frame. Normalizes lerp so camera speed
    /// is consistent regardless of frame rate variance (25ms–35ms observed).
    func updateCamera(dt: Float) {
        let targetDt: Float = 1.0 / 60.0
        // During drag: instant snap (factor=1.0) so smoothed values match targets exactly.
        // Otherwise: gradual lerp for smooth camera motion.
        // Updating smoothed values HERE (same callsite as cameraEntity transform) keeps
        // @Observable-driven label reprojection in sync with entity rendering.
        let s: Float = isDragging ? 1.0 : smoothing
        let factor = 1.0 - pow(1.0 - s, dt / targetDt)
        let newAz = azimuth + (targetAzimuth - azimuth) * factor
        let newEl = elevation + (targetElevation - elevation) * factor
        let newTarget = cameraTarget + (targetCameraPos - cameraTarget) * factor
        // Guard @Observable writes — only mutate when delta is perceptible.
        // Without this, every tiny lerp residual triggers SwiftUI Canvas re-render (O(n) projections).
        let threshold: Float = 0.0001
        if abs(newAz - azimuth) > threshold { azimuth = newAz }
        else if azimuth != targetAzimuth { azimuth = targetAzimuth }  // snap to stop cameraMoving
        if abs(newEl - elevation) > threshold { elevation = newEl }
        else if elevation != targetElevation { elevation = targetElevation }
        if simd_length(newTarget - cameraTarget) > threshold { cameraTarget = newTarget }
        else if cameraTarget != targetCameraPos { cameraTarget = targetCameraPos }
        cameraEntity?.transform = cameraTransform()
    }

    /// Center camera on the given positions (call once when graph first loads).
    func centerOnGraph(positions: [Int64: SIMD3<Float>]) {
        guard !positions.isEmpty else { return }
        var sum = SIMD3<Float>.zero
        for (_, pos) in positions { sum += pos }
        let centroid = sum / Float(positions.count)

        // Compute bounding sphere radius so camera fits the whole graph
        var maxDist: Float = 0
        for (_, pos) in positions {
            maxDist = max(maxDist, simd_length(pos - centroid))
        }
        // cameraTransform() applies an additional scaleFactor to cameraPosition,
        // so the effective distance in the transform = orbitRadius * scaleFactor.
        // To get an effective distance of ~3.5 * scaledRadius for the initial zoom:
        //   orbitRadius * scaleFactor = 3.5 * maxDist * scaleFactor
        //   orbitRadius = maxDist * 3.5
        orbitRadius = max(maxDist * 3.5, 800)

        // Set both target and current to avoid lerping from origin
        let scaledCentroid = centroid * scaleFactor
        targetCameraPos = scaledCentroid
        cameraTarget = scaledCentroid
    }

    /// Get cached NSColor tint for a project.
    /// Get cached node color as SIMD3<Float> for vertex data.
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

    // MARK: - Node Batch Mesh

    /// Create or resize the LowLevelMesh for batched node rendering (real sphere geometry).
    private func ensureNodeBatchMesh(capacity: Int) {
        guard capacity > nodeBatchCapacity else { return }
        let newCapacity = max(capacity * 2, 512)

        let totalVerts = newCapacity * vertsPerSphere
        let totalIndices = newCapacity * indicesPerSphere

        var desc = LowLevelMesh.Descriptor()
        desc.vertexCapacity = totalVerts
        desc.indexCapacity = totalIndices
        desc.vertexAttributes = [
            .init(semantic: .position, format: .float3, offset: 0),
            .init(semantic: .normal, format: .float3, offset: 12),
            .init(semantic: .uv0, format: .float2, offset: 24),
            .init(semantic: .color, format: .float4, offset: 32),
        ]
        desc.vertexLayouts = [.init(bufferIndex: 0, bufferStride: 48)]
        desc.indexType = .uint32

        guard let mesh = try? LowLevelMesh(descriptor: desc) else {
            frameLog.error("[nodeBatch] ❌ LowLevelMesh creation failed")
            return
        }

        // Pre-fill index buffer: stamp template indices for each node instance
        let templateIndices = sphereTemplateIndices
        let vpn = UInt32(vertsPerSphere)
        mesh.withUnsafeMutableIndices { raw in
            let indices = raw.bindMemory(to: UInt32.self)
            for node in 0..<newCapacity {
                let baseVert = UInt32(node) * vpn
                let baseIdx = node * indicesPerSphere
                for i in 0..<indicesPerSphere {
                    indices[baseIdx + i] = baseVert + templateIndices[i]
                }
            }
        }

        // Set parts to full capacity BEFORE assigning to entity — ensures the mesh
        // is never visible with zero parts. Vertices beyond actual node count are zero
        // from allocation (degenerate triangles, zero-area, GPU discards instantly).
        // This eliminates the need for parts.replaceAll during updateNodeBatch, which
        // caused 1-frame blanks when RealityKit rebuilt internal structures.
        let generousBounds = BoundingBox(min: SIMD3(-10, -10, -10), max: SIMD3(10, 10, 10))
        mesh.parts.replaceAll([
            LowLevelMesh.Part(
                indexCount: totalIndices,
                topology: .triangle,
                materialIndex: 0,
                bounds: generousBounds
            )
        ])

        nodeBatchMesh = mesh
        nodeBatchCapacity = newCapacity
        lastNodePartIndexCount = totalIndices  // Match capacity — prevents per-frame parts.replaceAll

        guard let resource = try? MeshResource(from: mesh),
              let material = nodeBatchMaterial else { return }

        if let entity = nodeBatchEntity {
            entity.model = ModelComponent(mesh: resource, materials: [material])
        } else {
            let entity = ModelEntity(mesh: resource, materials: [material])
            entity.name = "node_batch"
            rootEntity.addChild(entity)
            nodeBatchEntity = entity
        }
        frameLog.error("[nodeBatch] ✅ mesh created: \(newCapacity) nodes, \(totalVerts) verts, \(totalIndices) indices")
    }

    /// Get cached edge color as SIMD3<Float> for vertex data.
    private func edgeColorFloat3(for project: String?, colorMap: [String: Color]) -> SIMD3<Float> {
        let key = project ?? "__default"
        if let cached = edgeColorCache[key] { return cached }
        if let project, let swiftColor = colorMap[project] {
            let nsColor = NSColor(swiftColor).usingColorSpace(.sRGB) ?? NSColor(swiftColor)
            var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
            nsColor.getRed(&r, green: &g, blue: &b, alpha: &a)
            let c = SIMD3<Float>(
                Float(min(1.0, r * 1.4 + 0.15)),
                Float(min(1.0, g * 1.4 + 0.15)),
                Float(min(1.0, b * 1.4 + 0.15))
            )
            edgeColorCache[key] = c
            return c
        } else {
            let c = SIMD3<Float>(0.6, 0.6, 0.6)
            edgeColorCache[key] = c
            return c
        }
    }

    // MARK: - Edge Batch Mesh

    /// Create or resize the LowLevelMesh for batched edge rendering.
    private func ensureEdgeBatchMesh(capacity: Int) {
        guard capacity > edgeBatchCapacity else { return }
        let newCapacity = max(capacity * 2, 2048)

        var desc = LowLevelMesh.Descriptor()
        desc.vertexCapacity = newCapacity * 12   // 12 vertices per cylinder (6-segment tube)
        desc.indexCapacity = newCapacity * 36    // 36 indices per cylinder (6 quad faces × 2 triangles × 3)
        desc.vertexAttributes = [
            .init(semantic: .position, format: .float3, offset: 0),
            .init(semantic: .normal, format: .float3, offset: 12),
            .init(semantic: .uv0, format: .float2, offset: 24),
            .init(semantic: .color, format: .float4, offset: 32),
        ]
        desc.vertexLayouts = [.init(bufferIndex: 0, bufferStride: 48)]
        desc.indexType = .uint32

        guard let mesh = try? LowLevelMesh(descriptor: desc) else { return }

        // Pre-fill index buffer (6-sided cylinder topology: 6 quad faces per edge)
        mesh.withUnsafeMutableIndices { raw in
            let indices = raw.bindMemory(to: UInt32.self)
            for i in 0..<newCapacity {
                let vBase = UInt32(i * 12)    // 12 verts per edge (bottom ring 0..5, top ring 6..11)
                let iBase = i * 36            // 36 indices per edge
                for seg in 0..<6 {
                    let next = (seg + 1) % 6
                    let b0 = vBase + UInt32(seg)        // bottom ring
                    let b1 = vBase + UInt32(next)
                    let t0 = vBase + UInt32(seg + 6)    // top ring
                    let t1 = vBase + UInt32(next + 6)
                    let idx = iBase + seg * 6
                    // Two triangles per quad face (CCW from outside)
                    indices[idx]     = b0; indices[idx + 1] = t0; indices[idx + 2] = t1
                    indices[idx + 3] = b0; indices[idx + 4] = t1; indices[idx + 5] = b1
                }
            }
        }

        // Set parts to full capacity before assigning to entity — same pattern as node batch.
        // Eliminates per-frame parts.replaceAll that caused 1-frame blanks.
        let generousBounds = BoundingBox(min: SIMD3(-10, -10, -10), max: SIMD3(10, 10, 10))
        mesh.parts.replaceAll([
            LowLevelMesh.Part(
                indexCount: newCapacity * 36,
                topology: .triangle,
                materialIndex: 0,
                bounds: generousBounds
            )
        ])

        edgeBatchMesh = mesh
        edgeBatchCapacity = newCapacity
        lastEdgePartIndexCount = newCapacity * 36  // Match capacity — prevents per-frame parts.replaceAll

        guard let resource = try? MeshResource(from: mesh),
              let material = edgeBatchMaterial else { return }

        if let entity = edgeBatchEntity {
            entity.model = ModelComponent(mesh: resource, materials: [material])
        } else {
            let entity = ModelEntity(mesh: resource, materials: [material])
            entity.name = "edge_batch"
            edgeContainer.addChild(entity)
            edgeBatchEntity = entity
        }
    }

    // MARK: - Label Batch (GPU billboard quads)

    /// Generate a texture atlas containing all node labels and project labels.
    /// Two-pass: first measures all label sizes to compute exact atlas height, then draws.
    /// Returns the atlas texture resource; populates `labelAtlasRects` and `projectLabelAtlasRects`.
    private func generateLabelAtlas(nodes: [NodeData], hubs: Set<Int64>, projects: Set<String>) {
        let atlasW = 4096
        let padding: CGFloat = 4
        let fontSize: CGFloat = 28
        let projFontSize: CGFloat = 40

        // Pre-compute all label sizes (measurement pass)
        struct LabelSize { let width: CGFloat; let height: CGFloat }
        var nodeSizes: [LabelSize] = []
        nodeSizes.reserveCapacity(nodes.count)
        for node in nodes {
            let isHub = hubs.contains(node.id)
            let weight: NSFont.Weight = isHub ? .bold : .medium
            let font = NSFont.monospacedSystemFont(ofSize: fontSize, weight: weight)
            let attrs: [NSAttributedString.Key: Any] = [.font: font, .foregroundColor: NSColor.white]
            let size = (node.label as NSString).size(withAttributes: attrs)
            nodeSizes.append(LabelSize(width: ceil(size.width) + padding * 2, height: ceil(size.height) + padding))
        }

        let projFont = NSFont.systemFont(ofSize: projFontSize, weight: .bold)
        let projAttrs: [NSAttributedString.Key: Any] = [.font: projFont, .foregroundColor: NSColor.white]
        let sortedProjects = projects.sorted()
        var projSizes: [LabelSize] = []
        for project in sortedProjects {
            let size = (project as NSString).size(withAttributes: projAttrs)
            projSizes.append(LabelSize(width: ceil(size.width) + padding * 2, height: ceil(size.height) + padding))
        }

        // Simulate packing to compute needed height
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
        let atlasH = max(512, neededH + 32)  // small margin
        labelAtlasAspectCorrection = Float(atlasW) / (2.0 * Float(atlasH))

        let scale = 2  // retina
        let pixelW = atlasW * scale
        let pixelH = atlasH * scale

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil, width: pixelW, height: pixelH,
            bitsPerComponent: 8, bytesPerRow: pixelW * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            frameLog.error("[labelAtlas] ❌ CGContext creation failed")
            return
        }

        // Flip for correct text rendering (Core Graphics is bottom-up)
        ctx.translateBy(x: 0, y: CGFloat(pixelH))
        ctx.scaleBy(x: CGFloat(scale), y: CGFloat(-scale))

        // Drawing pass — node labels
        var rects: [Int64: (u0: Float, v0: Float, u1: Float, v1: Float)] = [:]
        cursorX = 2; cursorY = 0; rowHeight = 0

        for (i, node) in nodes.enumerated() {
            let s = nodeSizes[i]
            if cursorX + s.width > CGFloat(atlasW) { cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0 }
            if cursorY + s.height > CGFloat(atlasH) { break }

            let isHub = hubs.contains(node.id)
            let weight: NSFont.Weight = isHub ? .bold : .medium
            let font = NSFont.monospacedSystemFont(ofSize: fontSize, weight: weight)
            let attrs: [NSAttributedString.Key: Any] = [.font: font, .foregroundColor: NSColor.white]

            let drawPoint = CGPoint(x: cursorX + padding, y: cursorY + padding * 0.5)
            NSGraphicsContext.saveGraphicsState()
            NSGraphicsContext.current = NSGraphicsContext(cgContext: ctx, flipped: true)
            (node.label as NSString).draw(at: drawPoint, withAttributes: attrs)
            NSGraphicsContext.restoreGraphicsState()

            rects[node.id] = (
                Float(cursorX) / Float(atlasW), Float(cursorY) / Float(atlasH),
                Float(cursorX + s.width) / Float(atlasW), Float(cursorY + s.height) / Float(atlasH)
            )
            cursorX += s.width + padding; rowHeight = max(rowHeight, s.height)
        }

        // Drawing pass — project labels
        var projRects: [String: (u0: Float, v0: Float, u1: Float, v1: Float)] = [:]
        cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0

        for (i, project) in sortedProjects.enumerated() {
            let s = projSizes[i]
            if cursorX + s.width > CGFloat(atlasW) { cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0 }
            if cursorY + s.height > CGFloat(atlasH) { break }

            let drawPoint = CGPoint(x: cursorX + padding, y: cursorY + padding * 0.5)
            NSGraphicsContext.saveGraphicsState()
            NSGraphicsContext.current = NSGraphicsContext(cgContext: ctx, flipped: true)
            (project as NSString).draw(at: drawPoint, withAttributes: projAttrs)
            NSGraphicsContext.restoreGraphicsState()

            projRects[project] = (
                Float(cursorX) / Float(atlasW), Float(cursorY) / Float(atlasH),
                Float(cursorX + s.width) / Float(atlasW), Float(cursorY + s.height) / Float(atlasH)
            )
            cursorX += s.width + padding; rowHeight = max(rowHeight, s.height)
        }

        guard let cgImage = ctx.makeImage() else {
            frameLog.error("[labelAtlas] ❌ CGImage creation failed")
            return
        }
        guard let texture = try? TextureResource.generate(from: cgImage, options: .init(semantic: .raw)) else {
            frameLog.error("[labelAtlas] ❌ TextureResource generation failed")
            return
        }

        labelAtlasTexture = texture
        labelAtlasRects = rects
        projectLabelAtlasRects = projRects
        labelAtlasNodeIds = Set(nodes.map(\.id))
        labelAtlasHubIds = hubs
        labelAtlasProjects = projects

        // Update material texture binding
        if var mat = labelBatchMaterial {
            mat.custom.texture = .init(CustomMaterial.Texture(texture))
            labelBatchMaterial = mat
            labelBatchEntity?.model?.materials = [mat]
        }

        frameLog.error("[labelAtlas] ✅ atlas generated: \(rects.count) node + \(projRects.count) project labels, \(atlasW)×\(atlasH), cursorY=\(cursorY) projects=\(projects.sorted())")
    }

    /// Create or resize the LowLevelMesh for batched label rendering.
    /// 4 vertices per label (billboard quad), 6 indices per quad.
    private func ensureLabelBatchMesh(capacity: Int) {
        guard capacity > labelBatchCapacity else { return }
        let newCapacity = max(capacity * 2, 512)

        var desc = LowLevelMesh.Descriptor()
        desc.vertexCapacity = newCapacity * 4   // 4 verts per quad
        desc.indexCapacity = newCapacity * 6    // 6 indices per quad (2 triangles)
        desc.vertexAttributes = [
            .init(semantic: .position, format: .float3, offset: 0),
            .init(semantic: .normal, format: .float3, offset: 12),
            .init(semantic: .uv0, format: .float2, offset: 24),
            .init(semantic: .color, format: .float4, offset: 32),
        ]
        desc.vertexLayouts = [.init(bufferIndex: 0, bufferStride: 48)]
        desc.indexType = .uint32

        guard let mesh = try? LowLevelMesh(descriptor: desc) else {
            frameLog.error("[labelBatch] ❌ LowLevelMesh creation failed")
            return
        }

        // Pre-fill index buffer: 2 triangles per quad (CCW winding for billboard facing camera)
        mesh.withUnsafeMutableIndices { raw in
            let indices = raw.bindMemory(to: UInt32.self)
            for i in 0..<newCapacity {
                let vBase = UInt32(i * 4)
                let iBase = i * 6
                // v0=TL, v1=TR, v2=BL, v3=BR
                // Tri 1: v0→v2→v1 (CCW from camera)
                // Tri 2: v1→v2→v3 (CCW from camera)
                indices[iBase]     = vBase
                indices[iBase + 1] = vBase + 2
                indices[iBase + 2] = vBase + 1
                indices[iBase + 3] = vBase + 1
                indices[iBase + 4] = vBase + 2
                indices[iBase + 5] = vBase + 3
            }
        }

        // Set parts to full capacity (same pattern as node/edge batch)
        let generousBounds = BoundingBox(min: SIMD3(-10, -10, -10), max: SIMD3(10, 10, 10))
        mesh.parts.replaceAll([
            LowLevelMesh.Part(
                indexCount: newCapacity * 6,
                topology: .triangle,
                materialIndex: 0,
                bounds: generousBounds
            )
        ])

        labelBatchMesh = mesh
        labelBatchCapacity = newCapacity
        lastLabelPartIndexCount = newCapacity * 6

        guard let resource = try? MeshResource(from: mesh),
              let material = labelBatchMaterial else { return }

        if let entity = labelBatchEntity {
            entity.model = ModelComponent(mesh: resource, materials: [material])
        } else {
            let entity = ModelEntity(mesh: resource, materials: [material])
            entity.name = "label_batch"
            rootEntity.addChild(entity)
            labelBatchEntity = entity
        }
        frameLog.error("[labelBatch] ✅ mesh created: \(newCapacity) labels capacity")
    }

    /// Per-frame vertex stamping for label billboard quads.
    /// Encodes position (anchor), normal (corner offset), UV (atlas), color (opacity).
    func updateLabelBatch(positions: [Int64: SIMD3<Float>],
                          nodes: [NodeData], hubs: Set<Int64>,
                          selectedNode: Int64?) {
        let nodeCount = positions.count
        guard nodeCount > 0 else {
            if lastLabelPartIndexCount != 0 {
                frameLog.error("[labelBatch] ⚠️ ZERO-GUARD: clearing parts")
                labelBatchMesh?.parts.replaceAll([])
                lastLabelPartIndexCount = 0
            }
            return
        }

        // Regenerate atlas if node set, hub set, or project set changed.
        // Debounce: wait 60 frames (~1s) between regens to prevent flicker from
        // texture/UV mismatch in RealityKit's triple-buffered vertex data.
        // Nodes without atlas rects just skip rendering until the next regen.
        let currentNodeIds = Set(positions.keys)
        let currentProjects = Set(nodes.map(\.project))
        let atlasNeedsRegen = labelAtlasTexture == nil || currentNodeIds != labelAtlasNodeIds || hubs != labelAtlasHubIds || currentProjects != labelAtlasProjects
        if atlasNeedsRegen {
            let isFirstAtlas = labelAtlasTexture == nil
            let framesSinceRegen = renderFrameCount &- labelAtlasRegenFrame
            if isFirstAtlas || framesSinceRegen >= 60 {
                generateLabelAtlas(nodes: nodes, hubs: hubs, projects: currentProjects)
                labelAtlasRegenFrame = renderFrameCount
                // Sync labelAtlasNodeIds to match the comparison source (positions.keys)
                // to prevent perpetual regen when nodes and positions sets differ.
                labelAtlasNodeIds = currentNodeIds
            }
        }
        guard labelAtlasTexture != nil else { return }

        // Compute per-project centroids directly from node positions
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

        // Pre-compute project colors outside the unsafe buffer closure
        var projectColors: [String: SIMD3<Float>] = [:]
        for project in projectCentroids.keys {
            projectColors[project] = nodeColorFloat3(for: project, colorMap: renderColorMap)
        }

        let projectLabelCount = projectCentroids.count
        ensureLabelBatchMesh(capacity: nodeCount + projectLabelCount)
        guard let mesh = labelBatchMesh else { return }

        let nodeById = Dictionary(nodes.map { ($0.id, $0) }, uniquingKeysWith: { _, last in last })
        let camPos = cameraPosition
        let sf = scaleFactor
        let aspectCorr = labelAtlasAspectCorrection

        // Merge expanded child positions
        var allPositions = positions
        for (id, pos) in expandedChildPositions {
            allPositions[id] = pos
        }
        var expandedChildren = Set<Int64>()
        for hubId in expandedHubs {
            for childId in childrenOfHub(hubId) {
                expandedChildren.insert(childId)
            }
        }

        // Depth range for normalization
        var minDepth: Float = .greatestFiniteMagnitude
        var maxDepth: Float = 0
        for (_, pos) in allPositions {
            let d = simd_length(pos - camPos)
            minDepth = min(minDepth, d)
            maxDepth = max(maxDepth, d)
        }
        let depthRange = max(1.0, maxDepth - minDepth)

        mesh.withUnsafeMutableBytes(bufferIndex: 0) { raw in
            let verts = raw.bindMemory(to: BatchVertex.self)
            var quadIdx = 0

            let maxQuads = labelBatchCapacity - projectLabelCount
            for (id, pos3D) in allPositions {
                guard quadIdx < maxQuads,
                      let nodeData = nodeById[id],
                      let rect = labelAtlasRects[id] else {
                    continue
                }

                let depth = simd_length(pos3D - camPos)
                let isSelected = id == selectedNode
                let isHub = hubs.contains(id)
                let importance = Float(max(1, nodeData.importance))

                // LOD check — search-matched nodes visible from much further
                let isSearchMatch = renderIsSearchActive && renderSearchMatchIds.contains(id)
                if !isSelected && !expandedChildren.contains(id) && !isSearchMatch {
                    let maxVisible: Float = isHub ? 1200 : (200 + importance * 80)
                    guard depth < maxVisible else {
                        // Write 4 degenerate zero vertices
                        let base = quadIdx * 4
                        for c in 0..<4 {
                            verts[base + c] = BatchVertex(px: 0, py: 0, pz: 0, nx: 0, ny: 0, nz: 0, u: 0, v: 0, cr: 0, cg: 0, cb: 0, ca: 0)
                        }
                        quadIdx += 1
                        continue
                    }
                }

                // Opacity — same formula as old overlay
                let depthNorm = (depth - minDepth) / depthRange
                let depthFade: Float = isSelected ? 1.0 : (1.0 - depthNorm * 0.7)

                let maxVisible: Float = isHub ? 1200 : (200 + importance * 80)
                let fadeRange = maxVisible * 0.3
                let fadeT: Float = isSelected ? 1.0 : min(1.0, max(0.0, (maxVisible - depth) / fadeRange))

                let baseOpacity: Float = isSelected ? 0.95 : (isHub ? 0.8 : 0.6)
                let searchDimmed = renderIsSearchActive && !renderSearchMatchIds.contains(id)
                let searchFade: Float = searchDimmed ? 0.15 : 1.0
                let opacity = baseOpacity * fadeT * depthFade * searchFade

                if opacity < 0.02 {
                    let base = quadIdx * 4
                    for c in 0..<4 {
                        verts[base + c] = BatchVertex(px: 0, py: 0, pz: 0, nx: 0, ny: 0, nz: 0, u: 0, v: 0, cr: 0, cg: 0, cb: 0, ca: 0)
                    }
                    quadIdx += 1
                    continue
                }

                // Quad dimensions — halfH from node type, halfW from text aspect ratio
                let halfH: Float = isSelected ? 0.025 : (isHub ? 0.022 : 0.018)
                let textAspect = (rect.u1 - rect.u0) / max(0.001, rect.v1 - rect.v0) * aspectCorr
                let halfW = halfH * textAspect

                // Anchor position: node world pos scaled + Y offset above sphere
                let anchor = pos3D * sf + SIMD3<Float>(0, nodeRadius * 1.8, 0)

                // Corners: TL(0), TR(1), BL(2), BR(3)
                // normal carries pre-multiplied billboard offset (cx*halfW, cy*halfH, 0)
                let base = quadIdx * 4

                // v0 = TL: corner (-1, +1)
                verts[base] = BatchVertex(
                    px: anchor.x, py: anchor.y, pz: anchor.z,
                    nx: -halfW, ny: halfH, nz: 0,
                    u: rect.u0, v: rect.v0,
                    cr: 1, cg: 1, cb: 1, ca: opacity)
                // v1 = TR: corner (+1, +1)
                verts[base + 1] = BatchVertex(
                    px: anchor.x, py: anchor.y, pz: anchor.z,
                    nx: halfW, ny: halfH, nz: 0,
                    u: rect.u1, v: rect.v0,
                    cr: 1, cg: 1, cb: 1, ca: opacity)
                // v2 = BL: corner (-1, 0)
                verts[base + 2] = BatchVertex(
                    px: anchor.x, py: anchor.y, pz: anchor.z,
                    nx: -halfW, ny: 0, nz: 0,
                    u: rect.u0, v: rect.v1,
                    cr: 1, cg: 1, cb: 1, ca: opacity)
                // v3 = BR: corner (+1, 0)
                verts[base + 3] = BatchVertex(
                    px: anchor.x, py: anchor.y, pz: anchor.z,
                    nx: halfW, ny: 0, nz: 0,
                    u: rect.u1, v: rect.v1,
                    cr: 1, cg: 1, cb: 1, ca: opacity)

                quadIdx += 1
            }

            // Project labels — larger quads floating above each project cluster
            let projCentroids = projectCentroids
            let projAtlasRects = projectLabelAtlasRects
            for (project, centroidData) in projCentroids {
                guard let rect = projAtlasRects[project] else { continue }

                // Position label above the topmost node in the cluster, not at centroid
                let labelY = centroidData.maxY * sf + 0.15  // above highest node
                let anchor = SIMD3<Float>(centroidData.centroid.x * sf, labelY, centroidData.centroid.z * sf)
                let depth = simd_length(centroidData.centroid - camPos)

                // LOD — visible from much farther than node labels
                let maxVisible: Float = 12000
                guard depth < maxVisible else {
                    let base = quadIdx * 4
                    for c in 0..<4 {
                        verts[base + c] = BatchVertex(px: 0, py: 0, pz: 0, nx: 0, ny: 0, nz: 0, u: 0, v: 0, cr: 0, cg: 0, cb: 0, ca: 0)
                    }
                    quadIdx += 1
                    continue
                }

                let fadeRange = maxVisible * 0.3
                let fadeT = min(1.0, max(0.0, (maxVisible - depth) / fadeRange))
                let opacity: Float = 0.9 * fadeT

                if opacity < 0.02 {
                    let base = quadIdx * 4
                    for c in 0..<4 {
                        verts[base + c] = BatchVertex(px: 0, py: 0, pz: 0, nx: 0, ny: 0, nz: 0, u: 0, v: 0, cr: 0, cg: 0, cb: 0, ca: 0)
                    }
                    quadIdx += 1
                    continue
                }

                // ~7x node label size for prominent project labels
                let halfH: Float = 0.15
                let textAspect = (rect.u1 - rect.u0) / max(0.001, rect.v1 - rect.v0) * aspectCorr
                let halfW = halfH * textAspect

                // Pre-computed project color
                let color = projectColors[project] ?? SIMD3<Float>(1, 1, 1)

                // nz encodes forward bias toward camera (rendered in front of nodes and edges)
                let fwd: Float = 0.25
                let base = quadIdx * 4
                verts[base] = BatchVertex(
                    px: anchor.x, py: anchor.y, pz: anchor.z,
                    nx: -halfW, ny: halfH, nz: fwd,
                    u: rect.u0, v: rect.v0,
                    cr: color.x, cg: color.y, cb: color.z, ca: opacity)
                verts[base + 1] = BatchVertex(
                    px: anchor.x, py: anchor.y, pz: anchor.z,
                    nx: halfW, ny: halfH, nz: fwd,
                    u: rect.u1, v: rect.v0,
                    cr: color.x, cg: color.y, cb: color.z, ca: opacity)
                verts[base + 2] = BatchVertex(
                    px: anchor.x, py: anchor.y, pz: anchor.z,
                    nx: -halfW, ny: 0, nz: fwd,
                    u: rect.u0, v: rect.v1,
                    cr: color.x, cg: color.y, cb: color.z, ca: opacity)
                verts[base + 3] = BatchVertex(
                    px: anchor.x, py: anchor.y, pz: anchor.z,
                    nx: halfW, ny: 0, nz: fwd,
                    u: rect.u1, v: rect.v1,
                    cr: color.x, cg: color.y, cb: color.z, ca: opacity)

                quadIdx += 1
            }

            // Zero remaining vertices
            let totalUsed = quadIdx * 4
            let totalCapacity = labelBatchCapacity * 4
            if totalUsed < totalCapacity {
                for i in totalUsed..<totalCapacity {
                    verts[i] = BatchVertex(px: 0, py: 0, pz: 0, nx: 0, ny: 0, nz: 0, u: 0, v: 0, cr: 0, cg: 0, cb: 0, ca: 0)
                }
            }
        }

        // Capacity-based recovery (same pattern as node/edge)
        let capacityIndexCount = labelBatchCapacity * 6
        if capacityIndexCount != lastLabelPartIndexCount {
            let generousBounds = BoundingBox(min: SIMD3(-10, -10, -10), max: SIMD3(10, 10, 10))
            mesh.parts.replaceAll([
                LowLevelMesh.Part(
                    indexCount: capacityIndexCount,
                    topology: .triangle,
                    materialIndex: 0,
                    bounds: generousBounds
                )
            ])
            lastLabelPartIndexCount = capacityIndexCount
        }
    }

    /// Rebuild the batched node mesh using Metal compute to stamp sphere instances.
    /// All alive nodes rendered as real sphere triangles in a single LowLevelMesh (1 draw call).
    /// Point lights managed as separate lightweight entities (~5-10 at most).
    func updateNodeBatch(positions: [Int64: SIMD3<Float>],
                         nodes: [NodeData], hubs: Set<Int64>,
                         colorMap: [String: Color],
                         selectedNode: Int64?,
                         glowingNodes: [Int64: Date],
                         newNodes: [Int64: Date]) {
        // Invalidate SIMD color caches when the color map changes
        if let version = renderStore?.colorMapVersion, version != lastColorMapVersion {
            nodeColorCache.removeAll(keepingCapacity: true)
            edgeColorCache.removeAll(keepingCapacity: true)
            lastColorMapVersion = version
        }

        let nodeCount = positions.count
        guard nodeCount > 0, vertsPerSphere > 0 else {
            if lastNodePartIndexCount != 0 {
                frameLog.error("[nodeBatch] ⚠️ ZERO-GUARD: clearing parts (nodeCount=\(nodeCount), vps=\(self.vertsPerSphere))")
                nodeBatchMesh?.parts.replaceAll([])
                lastNodePartIndexCount = 0
            }
            return
        }

        let prevCapacity = nodeBatchCapacity
        ensureNodeBatchMesh(capacity: nodeCount)
        if nodeBatchCapacity != prevCapacity {
            frameLog.error("[nodeBatch] ⚠️ CAPACITY GREW: \(prevCapacity) → \(self.nodeBatchCapacity) (nodeCount=\(nodeCount))")
        }
        guard let mesh = nodeBatchMesh else { return }

        let nodeById = Dictionary(nodes.map { ($0.id, $0) }, uniquingKeysWith: { _, last in last })
        let now = Date()
        let camPos = cameraPosition
        let fogNear: Float = 100
        let fogFar: Float = 1200

        // Ensure instance array is large enough
        if instanceArray.count < nodeCount {
            instanceArray = [NodeInstance](repeating: NodeInstance(px: 0, py: 0, pz: 0, scale: 0, cr: 0, cg: 0, cb: 0, packed: 0), count: max(nodeCount * 2, 512))
        }

        // Fill instance array + track point lights
        var activeLightIds = Set<Int64>()
        var idx = 0

        for (id, pos) in positions {
            guard let nodeData = nodeById[id] else { continue }

            let worldPos = pos * scaleFactor
            let isHub = hubs.contains(id)
            let importance = max(1, nodeData.importance)
            let baseRadius: Float = isHub ? nodeRadius * 1.6 : nodeRadius
            let r = baseRadius * (1.0 + Float(importance - 1) * 0.08)

            // Recall glow intensity
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

            let searchDimmed = renderIsSearchActive && !renderSearchMatchIds.contains(id)
            let searchMatched = renderIsSearchActive && renderSearchMatchIds.contains(id) && id != selectedNode

            let curState: Float
            let curIntensity: Float
            if id == selectedNode {
                curState = 1; curIntensity = 0
            } else if ri > 0 {
                curState = 2; curIntensity = ri
            } else if ai > 0 {
                curState = 3; curIntensity = ai
            } else if searchMatched {
                curState = 4; curIntensity = 0
            } else {
                curState = 0; curIntensity = 0
            }

            let packedState: Float = curState + (searchDimmed ? 10.0 : 0.0) + curIntensity * 0.01

            let color: SIMD3<Float> = id == selectedNode
                ? SIMD3<Float>(1, 1, 1)
                : nodeColorFloat3(for: nodeData.project, colorMap: colorMap)

            instanceArray[idx] = NodeInstance(
                px: worldPos.x, py: worldPos.y, pz: worldPos.z,
                scale: r,
                cr: color.x, cg: color.y, cb: color.z,
                packed: packedState
            )

            idx += 1

            // Point lights for glowing/arriving/search-matched nodes.
            // Reuses deactivated lights from the pool to avoid addChild scene graph churn.
            if !Self.DIAG_SKIP_POINT_LIGHTS, ri > 0 || ai > 0 || searchMatched {
                activeLightIds.insert(id)
                let lightEntity: Entity
                if let existing = pointLightEntities[id] {
                    lightEntity = existing
                } else if let recycled = pointLightEntities.first(where: { !activeLightIds.contains($0.key) && $0.key != id }) {
                    // Recycle a deactivated light entity — avoids addChild()
                    pointLightEntities.removeValue(forKey: recycled.key)
                    lightEntity = recycled.value
                    lightEntity.name = "ptlight_\(id)"
                    pointLightEntities[id] = lightEntity
                } else {
                    lightEntity = Entity()
                    lightEntity.name = "ptlight_\(id)"
                    rootEntity.addChild(lightEntity)
                    pointLightEntities[id] = lightEntity
                }
                lightEntity.position = worldPos

                let dist = simd_length(pos - camPos)
                let fogT = max(Float(0), min(Float(1), (dist - fogNear) / (fogFar - fogNear)))
                let depthFade = max(Float(0.0), 1.0 - fogT * 0.95)

                if ri > 0 {
                    let lightPulse = 1.0 + sin(animationTime * 3.0) * 0.4
                    lightEntity.components.set(PointLightComponent(
                        color: NSColor(red: 0.7, green: 0.9, blue: 1.0, alpha: 1),
                        intensity: 20000 * ri * depthFade * lightPulse,
                        attenuationRadius: 2.0 * depthFade + 0.1
                    ))
                } else if ai > 0 {
                    let lightPulse = 1.0 + sin(animationTime * 2.5) * 0.4
                    lightEntity.components.set(PointLightComponent(
                        color: NSColor(red: 1.0, green: 0.8, blue: 0.3, alpha: 1),
                        intensity: 20000 * ai * depthFade * lightPulse,
                        attenuationRadius: 2.0 * depthFade + 0.1
                    ))
                } else {
                    // Search-matched: distance-independent glow
                    let lightPulse = 1.0 + sin(animationTime * 4.0) * 0.3
                    lightEntity.components.set(PointLightComponent(
                        color: NSColor(red: 0.0, green: 0.9, blue: 1.0, alpha: 1),
                        intensity: 20000 * lightPulse,
                        attenuationRadius: 2.5
                    ))
                }
            }
        }
        let actualNodeCount = idx
        if actualNodeCount != nodeCount && renderFrameCount % 60 == 0 {
            frameLog.error("[nodeBatch] ⚠️ MISMATCH: actualNodeCount=\(actualNodeCount) vs positions.count=\(nodeCount) nodes.count=\(nodes.count)")
        }

        // --- CPU stamp: directly write sphere vertices into mesh buffer ---
        // For ~500 nodes × 52 verts = 26K vertices, CPU is faster than GPU dispatch
        // overhead (command buffer encode + submit + fence wait).
        let templateVerts = sphereTemplateVertices
        let vps = vertsPerSphere
        let instances = instanceArray
        mesh.withUnsafeMutableBytes(bufferIndex: 0) { raw in
            let outPtr = raw.baseAddress!.assumingMemoryBound(to: Float.self)
            for nodeIdx in 0..<actualNodeCount {
                let inst = instances[nodeIdx]
                let instPos = SIMD3<Float>(inst.px, inst.py, inst.pz)
                let scale = inst.scale
                let cr = inst.cr, cg = inst.cg, cb = inst.cb, packed = inst.packed
                for vertIdx in 0..<vps {
                    let tv = templateVerts[vertIdx]
                    let wp = instPos + tv.pos * scale
                    let base = (nodeIdx * vps + vertIdx) * 12
                    outPtr[base + 0]  = wp.x
                    outPtr[base + 1]  = wp.y
                    outPtr[base + 2]  = wp.z
                    outPtr[base + 3]  = tv.norm.x
                    outPtr[base + 4]  = tv.norm.y
                    outPtr[base + 5]  = tv.norm.z
                    outPtr[base + 6]  = 0  // uv0.x (reserved — nonzero causes lit pipeline artifacts)
                    outPtr[base + 7]  = 0
                    outPtr[base + 8]  = cr
                    outPtr[base + 9]  = cg
                    outPtr[base + 10] = cb
                    outPtr[base + 11] = packed
                }
            }
            // Zero stale vertices beyond actualNodeCount to prevent ghost rendering.
            // Part indexCount covers full capacity, so unwritten slots must be degenerate.
            let writtenFloats = actualNodeCount * vps * 12
            let capacityFloats = nodeBatchCapacity * vps * 12
            if writtenFloats < capacityFloats {
                let startPtr = outPtr + writtenFloats
                startPtr.update(repeating: 0, count: capacityFloats - writtenFloats)
            }
        }

        // Use capacity-based index count — parts.replaceAll only fires when recovering
        // from the zero-guard (which clears parts and sets lastNodePartIndexCount=0).
        // During normal operation, capacity doesn't change so this never fires.
        let capacityIndexCount = nodeBatchCapacity * indicesPerSphere
        if capacityIndexCount != lastNodePartIndexCount {
            frameLog.error("[nodeBatch] ⚠️ PARTS RECOVERY: \(self.lastNodePartIndexCount) → \(capacityIndexCount)")
            let generousBounds = BoundingBox(min: SIMD3(-10, -10, -10), max: SIMD3(10, 10, 10))
            mesh.parts.replaceAll([
                LowLevelMesh.Part(
                    indexCount: capacityIndexCount,
                    topology: .triangle,
                    materialIndex: 0,
                    bounds: generousBounds
                )
            ])
            lastNodePartIndexCount = capacityIndexCount
        }

        // Deactivate stale point lights instead of removing them from the scene graph.
        // removeFromParent() triggers scene graph rebuilds → flicker. Setting intensity to 0
        // keeps the entity in the tree (no rebuild) but makes it invisible/free.
        // Deactivated lights accumulate in pointLightEntities and are reused when needed.
        if Self.DIAG_SKIP_POINT_LIGHTS {
            for (_, e) in pointLightEntities {
                e.components.set(PointLightComponent(color: .black, intensity: 0, attenuationRadius: 0))
            }
        } else {
            for id in pointLightEntities.keys where !activeLightIds.contains(id) {
                pointLightEntities[id]?.components.set(
                    PointLightComponent(color: .black, intensity: 0, attenuationRadius: 0)
                )
            }
        }
    }

    /// Update dying nodes — individual PBR entities with red flash animation (max ~5 at a time).
    func updateDyingNodes(positions: [Int64: SIMD3<Float>],
                          dyingNodes: [Int64: DyingNode]) {
        let now = Date()

        // Capture 3D positions for dying nodes while they're still in positions
        for id in dyingNodes.keys {
            if dyingNodePos3D[id] == nil, let pos = positions[id] {
                dyingNodePos3D[id] = pos
            }
        }

        for (id, dying) in dyingNodes {
            if positions[id] != nil { continue }  // still live, handled in batch

            let elapsed = Float(now.timeIntervalSince(dying.startTime))
            let flashIn: Float = 0.3, hold: Float = 1.2, fadeOut: Float = 1.5
            let total = flashIn + hold + fadeOut

            guard elapsed < total else {
                if let entity = dyingNodeEntities[id] {
                    entity.removeFromParent()
                    dyingNodeEntities.removeValue(forKey: id)
                }
                dyingNodePos3D.removeValue(forKey: id)
                continue
            }

            // Create entity on first frame if needed
            if dyingNodeEntities[id] == nil {
                let pos3D = dyingNodePos3D[id] ?? .zero
                let entity = ModelEntity(mesh: nodeMesh, materials: [PhysicallyBasedMaterial()])
                entity.position = pos3D * scaleFactor
                entity.name = "dying_\(id)"
                rootEntity.addChild(entity)
                dyingNodeEntities[id] = entity
            }
            guard let entity = dyingNodeEntities[id] else { continue }

            let di: Float
            if elapsed < flashIn {
                let t = elapsed / flashIn; di = t * t
            } else if elapsed < flashIn + hold {
                di = 1.0
            } else {
                let t = 1.0 - (elapsed - flashIn - hold) / fadeOut; di = t * t
            }

            let darkening = elapsed > flashIn ? min(1.0, (elapsed - flashIn) / (hold + fadeOut)) : 0.0
            var mat = PhysicallyBasedMaterial()
            mat.baseColor = .init(tint: NSColor(
                red: CGFloat(0.9 * (1 - darkening * 0.8)),
                green: CGFloat(0.15 * (1 - darkening)),
                blue: CGFloat(0.1 * (1 - darkening)),
                alpha: 1
            ))
            mat.emissiveColor = .init(color: NSColor(red: 1, green: 0.2, blue: 0.1, alpha: 1))
            mat.emissiveIntensity = 2.5 * di
            mat.roughness = .init(floatLiteral: 0.3)
            entity.model?.materials = [mat]

            entity.components.set(PointLightComponent(
                color: NSColor(red: 1.0, green: 0.15, blue: 0.05, alpha: 1),
                intensity: 15000 * di,
                attenuationRadius: 1.5
            ))

            let shrink: Float = elapsed > flashIn + hold ? 1.0 - (1.0 - di) * 0.5 : 1.0
            let baseR: Float = dying.isHub ? nodeRadius * 1.6 : nodeRadius
            let impR = baseR * (1.0 + Float(max(1, dying.importance) - 1) * 0.08)
            entity.scale = SIMD3<Float>(repeating: impR * shrink)
            entity.components.set(OpacityComponent(opacity: di))
        }

        // Clean up finished dying entities
        for id in dyingNodeEntities.keys where dyingNodes[id] == nil {
            dyingNodeEntities[id]?.removeFromParent()
            dyingNodeEntities.removeValue(forKey: id)
        }
    }

    /// Rebuild the batched edge mesh. All edges are rendered as a single LowLevelMesh
    /// with per-edge state encoded in vertex UV and color in vertex color attribute.
    /// This produces 1 draw call instead of 1238 separate entities.
    func updateEdgeBatch(positions: [Int64: SIMD3<Float>],
                         edges: [EdgeData],
                         nodes: [NodeData],
                         hubs: Set<Int64>,
                         layoutMode: LayoutMode,
                         colorMap: [String: Color],
                         selectedNode: Int64?) {
        let edgeCount = edges.count
        guard edgeCount > 0 else {
            if lastEdgePartIndexCount != 0 {
                frameLog.error("[edgeBatch] ⚠️ ZERO-GUARD: clearing parts (edgeCount=0)")
                edgeBatchMesh?.parts.replaceAll([])
                lastEdgePartIndexCount = 0
            }
            return
        }

        ensureEdgeBatchMesh(capacity: edgeCount)
        guard let mesh = edgeBatchMesh else { return }

        let isSemanticMode = layoutMode == .embedding
        let nodeProject: [Int64: String] = Dictionary(
            nodes.map { ($0.id, $0.project) }, uniquingKeysWith: { _, last in last }
        )

        // Build per-node radius lookup for endpoint inset (same formula as updateNodeBatch)
        var nodeRadii: [Int64: Float] = [:]
        for node in nodes {
            let isHub = hubs.contains(node.id)
            let importance = max(1, node.importance)
            let baseR: Float = isHub ? nodeRadius * 1.6 : nodeRadius
            nodeRadii[node.id] = baseR * (1.0 + Float(importance - 1) * 0.08)
        }

        var actualEdges = 0

        mesh.withUnsafeMutableBytes(bufferIndex: 0) { raw in
            let vertices = raw.bindMemory(to: BatchVertex.self)
            var vi = 0

            for edge in edges {
                guard let from = positions[edge.sourceId],
                      let to = positions[edge.targetId] else {
                    // Degenerate: write 12 zero vertices (cylinder has 12 verts)
                    let zero = BatchVertex(px: 0, py: 0, pz: 0, nx: 0, ny: 1, nz: 0,
                                               u: 0, v: 0, cr: 0, cg: 0, cb: 0, ca: 0)
                    for _ in 0..<12 { vertices[vi] = zero; vi += 1 }
                    continue
                }

                let p1 = from * scaleFactor
                let p2 = to * scaleFactor
                let delta = p2 - p1
                let length = simd_length(delta)

                // Inset endpoints by node radii so edges stop at sphere surfaces
                let r1 = nodeRadii[edge.sourceId] ?? nodeRadius
                let r2 = nodeRadii[edge.targetId] ?? nodeRadius
                guard length > r1 + r2 else {
                    // Edge too short — nodes overlap, write 12 degenerate vertices
                    let zero = BatchVertex(px: 0, py: 0, pz: 0, nx: 0, ny: 1, nz: 0,
                                               u: 0, v: 0, cr: 0, cg: 0, cb: 0, ca: 0)
                    for _ in 0..<12 { vertices[vi] = zero; vi += 1 }
                    continue
                }

                let dir = delta / length
                let p1inset = p1 + dir * r1
                let p2inset = p2 - dir * r2

                // Build perpendicular basis vectors for cylinder cross-section
                let basisUp: SIMD3<Float> = abs(dir.y) < 0.99 ? SIMD3(0, 1, 0) : SIMD3(1, 0, 0)
                let basisRight = simd_normalize(cross(dir, basisUp))
                let basisForward = simd_normalize(cross(basisRight, dir))

                let connected = edge.sourceId == selectedNode || edge.targetId == selectedNode
                let radius = connected ? edgeRadius * 2.5 : edgeRadius * 1.3

                // State encoding for shader
                let searchDimmed = renderIsSearchActive
                    && !renderSearchMatchIds.contains(edge.sourceId)
                    && !renderSearchMatchIds.contains(edge.targetId)
                let state: Float
                if connected { state = 1 }
                else if searchDimmed { state = 2 }
                else if isSemanticMode { state = 3 }
                else { state = 0 }

                // Per-edge color
                let color: SIMD3<Float>
                if connected {
                    color = SIMD3<Float>(1, 1, 1)
                } else {
                    color = edgeColorFloat3(for: nodeProject[edge.sourceId], colorMap: colorMap)
                }

                // Cylinder: 12 vertices (bottom ring 0..5, top ring 6..11)
                // Bottom ring at p1inset
                for seg in 0..<6 {
                    let angle = Float(seg) * (.pi * 2 / 6)
                    let c = cos(angle), s = sin(angle)
                    let offset = (basisRight * c + basisForward * s) * radius
                    let normal = simd_normalize(basisRight * c + basisForward * s)
                    let bp = p1inset + offset
                    vertices[vi] = BatchVertex(px: bp.x, py: bp.y, pz: bp.z,
                                                   nx: normal.x, ny: normal.y, nz: normal.z,
                                                   u: state, v: 0,
                                                   cr: color.x, cg: color.y, cb: color.z, ca: 1)
                    vi += 1
                }
                // Top ring at p2inset
                for seg in 0..<6 {
                    let angle = Float(seg) * (.pi * 2 / 6)
                    let c = cos(angle), s = sin(angle)
                    let offset = (basisRight * c + basisForward * s) * radius
                    let normal = simd_normalize(basisRight * c + basisForward * s)
                    let tp = p2inset + offset
                    vertices[vi] = BatchVertex(px: tp.x, py: tp.y, pz: tp.z,
                                                   nx: normal.x, ny: normal.y, nz: normal.z,
                                                   u: state, v: 1,
                                                   cr: color.x, cg: color.y, cb: color.z, ca: 1)
                    vi += 1
                }
                actualEdges += 1
            }

            // Zero stale vertices beyond actual edges to prevent ghost rendering.
            let vertsPerEdge = 12
            let writtenVerts = actualEdges * vertsPerEdge
            let capacityVerts = edgeBatchCapacity * vertsPerEdge
            if writtenVerts < capacityVerts {
                let zero = BatchVertex(px: 0, py: 0, pz: 0, nx: 0, ny: 1, nz: 0,
                                       u: 0, v: 0, cr: 0, cg: 0, cb: 0, ca: 0)
                for i in writtenVerts..<capacityVerts {
                    vertices[i] = zero
                }
            }
        }

        // Use capacity-based index count — parts.replaceAll only fires when recovering
        // from the zero-guard (which clears parts and sets lastEdgePartIndexCount=0).
        let capacityIndexCount = edgeBatchCapacity * 36
        if capacityIndexCount != lastEdgePartIndexCount {
            frameLog.error("[edgeBatch] ⚠️ PARTS RECOVERY: \(self.lastEdgePartIndexCount) → \(capacityIndexCount)")
            let generousBounds = BoundingBox(min: SIMD3(-10, -10, -10), max: SIMD3(10, 10, 10))
            mesh.parts.replaceAll([
                LowLevelMesh.Part(
                    indexCount: capacityIndexCount,
                    topology: .triangle,
                    materialIndex: 0,
                    bounds: generousBounds
                )
            ])
            lastEdgePartIndexCount = capacityIndexCount
        }
    }

    // MARK: - Diagnostics: disable expensive features to isolate bottleneck
    // Toggle these to measure the impact of each subsystem.
    // RESULTS: record dt after each toggle, then remove this block.
    private static let DIAG_SKIP_NEBULAE = false      // skip ALL particle emitters
    private static let DIAG_SKIP_POINT_LIGHTS = false  // skip per-node point lights
    private static let DIAG_SKIP_GEOM_MOD = false     // skip geometry modifier on node batch
    private static let DIAG_SKIP_NODE_BATCH = false   // skip node LowLevelMesh update
    private static let DIAG_SKIP_EDGE_BATCH = false   // skip edge LowLevelMesh update
    private static let DIAG_SKIP_LABEL_BATCH = false  // skip label LowLevelMesh update

    // MARK: - Nebulae (particle-based gaseous hulls)

    /// One particle emitter entity per cluster group. Uses ParticleEmitterComponent
    /// with sphere-shaped volume emission and additive blending — particles are
    /// billboard sprites that never write to the depth buffer, so they can't occlude nodes.
    private var nebulaEmitters: [String: Entity] = [:]

    private struct NebulaGroup {
        let key: String
        let centroid: SIMD3<Float>
        let radius: Float
        let project: String
    }

    /// Cached per-project start/end colors for particle evolving tint.
    private var nebulaColorCache: [String: (start: NSColor, end: NSColor)] = [:]

    private func nebulaColors(for project: String, colorMap: [String: Color]) -> (start: NSColor, end: NSColor) {
        if let cached = nebulaColorCache[project] { return cached }
        let color = colorMap[project] ?? .gray
        let nsColor = NSColor(color).usingColorSpace(.sRGB) ?? NSColor(color)
        var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
        nsColor.getRed(&r, green: &g, blue: &b, alpha: &a)
        // Start: visible but subtle. End: barely there.
        // Lower per-particle alpha — more particles, each nearly invisible individually,
        // but accumulating into a smooth colored fog. Quartic texture falloff helps blend.
        let start = NSColor(
            red: min(1.0, r * 1.3 + 0.1),
            green: min(1.0, g * 1.3 + 0.1),
            blue: min(1.0, b * 1.3 + 0.1),
            alpha: 0.035
        )
        let end = NSColor(
            red: min(1.0, r * 0.7 + 0.05),
            green: min(1.0, g * 0.7 + 0.05),
            blue: min(1.0, b * 0.7 + 0.05),
            alpha: 0.005
        )
        let result = (start: start, end: end)
        nebulaColorCache[project] = result
        return result
    }

    private func nebulaGroupsForCurrentMode() -> [NebulaGroup] {
        let positions = renderPositions
        guard !positions.isEmpty else { return [] }

        let isSemanticMode = renderLayoutMode == .embedding

        var groups: [NebulaGroup] = []

        if isSemanticMode {
            for cluster in renderSemanticClusters3D {
                let memberPositions = cluster.nodeIds.compactMap { positions[$0] }
                guard memberPositions.count >= 3 else { continue }
                var sum = SIMD3<Float>.zero
                for p in memberPositions { sum += p }
                let centroid = sum / Float(memberPositions.count)
                var maxDist: Float = 0
                for p in memberPositions { maxDist = max(maxDist, simd_length(p - centroid)) }
                let dominantProject = cluster.projectBreakdown.first?.project ?? "global"
                groups.append(NebulaGroup(
                    key: "sem_\(cluster.id)", centroid: centroid, radius: maxDist + 15,
                    project: dominantProject
                ))
            }
        } else {
            // Per-project grouping — every project with 2+ nodes gets a nebula.
            var projectNodes: [String: [SIMD3<Float>]] = [:]
            for node in renderNodes {
                guard let pos = positions[node.id] else { continue }
                projectNodes[node.project, default: []].append(pos)
            }
            for (project, pts) in projectNodes where pts.count >= 2 {
                var sum = SIMD3<Float>.zero
                for p in pts { sum += p }
                let centroid = sum / Float(pts.count)
                var maxDist: Float = 0
                for p in pts { maxDist = max(maxDist, simd_length(p - centroid)) }
                groups.append(NebulaGroup(
                    key: "proj_\(project)", centroid: centroid, radius: maxDist + 40,
                    project: project
                ))
            }
        }

        return groups
    }

    /// Build a ParticleEmitterComponent configured as a gaseous nebula.
    private func makeNebulaEmitter(radius: Float, startColor: NSColor, endColor: NSColor) -> ParticleEmitterComponent {
        var emitter = ParticleEmitterComponent()
        let scaledR = radius * scaleFactor

        // Sphere volume — generously larger than cluster for soft boundary
        emitter.emitterShape = .sphere
        emitter.birthLocation = .volume
        emitter.emitterShapeSize = SIMD3<Float>(repeating: scaledR * 2.4)

        // Nearly stationary — particles hover in place like fog
        emitter.speed = 0.0001
        emitter.speedVariation = 0.00005

        // Warm-up matches lifespan for full steady-state on first frame
        emitter.timing = .repeating(
            warmUp: 15.0,
            emit: .init(duration: .infinity, variation: nil),
            idle: nil
        )

        // More particles, each nearly invisible — smooth accumulation hides circles
        emitter.mainEmitter.birthRate = max(5, min(15, radius * 0.05))
        emitter.mainEmitter.lifeSpan = 15.0
        emitter.mainEmitter.lifeSpanVariation = 5.0

        // 55% of cluster radius — overlapping enough to blend, manageable for GPU
        emitter.mainEmitter.size = max(0.03, scaledR * 0.55)
        emitter.mainEmitter.sizeVariation = emitter.mainEmitter.size * 0.4
        // Grow slightly over lifetime
        emitter.mainEmitter.sizeMultiplierAtEndOfLifespan = 1.2
        emitter.mainEmitter.sizeMultiplierAtEndOfLifespanPower = 0.4

        // Evolving color: bright start → dim end for organic lifecycle
        emitter.mainEmitter.color = .evolving(
            start: .single(startColor),
            end: .single(endColor)
        )
        emitter.mainEmitter.colorEvolutionPower = 0.5
        // Alpha blending preserves color identity — additive summed all colors to white
        emitter.mainEmitter.blendMode = .alpha

        // Soft gaussian texture — the single biggest visual improvement
        if let texture = softParticleTexture {
            emitter.mainEmitter.image = texture
        }

        // Billboard — always face camera
        emitter.mainEmitter.billboardMode = .billboard

        // Minimal noise — near-static glow, not morphing blobs
        emitter.mainEmitter.noiseStrength = 0.0003
        emitter.mainEmitter.noiseScale = 6.0
        emitter.mainEmitter.noiseAnimationSpeed = 0.03

        // Fade in and out smoothly
        emitter.mainEmitter.opacityCurve = .quickFadeInOut

        // Maximum damping — particles stay exactly where they spawn
        emitter.mainEmitter.dampingFactor = 0.98

        return emitter
    }

    /// Full nebula update — creates/removes/repositions particle emitter entities.
    func updateNebulae() {
        if Self.DIAG_SKIP_NEBULAE {
            // Remove any existing emitters for clean measurement
            for (_, e) in nebulaEmitters { e.removeFromParent() }
            nebulaEmitters.removeAll()
            return
        }
        let groups = nebulaGroupsForCurrentMode()
        let currentKeys = Set(groups.map(\.key))

        if renderFrameCount % 180 == 0 {
            frameLog.info("[nebula] groups=\(groups.count) emitters=\(self.nebulaEmitters.count) mode=\(self.renderLayoutMode == .embedding ? "semantic" : "force") clusters=\(self.renderClusters.count)")
        }

        // Remove stale emitters
        for key in nebulaEmitters.keys where !currentKeys.contains(key) {
            nebulaEmitters[key]?.removeFromParent()
            nebulaEmitters.removeValue(forKey: key)
        }

        for group in groups {
            let R = max(group.radius, 30.0)
            let worldCentroid = group.centroid * scaleFactor

            let colors = nebulaColors(for: group.project, colorMap: renderColorMap)

            if let entity = nebulaEmitters[group.key] {
                // Update position
                entity.position = worldCentroid
                // Update emitter shape size if radius changed significantly
                if var emitter = entity.components[ParticleEmitterComponent.self] {
                    let scaledR = R * scaleFactor
                    let newSize = SIMD3<Float>(repeating: scaledR * 2)
                    if abs(emitter.emitterShapeSize.x - newSize.x) > 0.01 {
                        emitter.emitterShapeSize = SIMD3<Float>(repeating: scaledR * 2.4)
                        emitter.mainEmitter.size = max(0.03, scaledR * 0.55)
                        emitter.mainEmitter.sizeVariation = emitter.mainEmitter.size * 0.4
                        emitter.mainEmitter.birthRate = max(5, min(15, R * 0.05))
                        emitter.mainEmitter.color = .evolving(
                            start: .single(colors.start), end: .single(colors.end)
                        )
                        entity.components.set(emitter)
                    }
                }
            } else {
                // Create new emitter entity
                let entity = Entity()
                entity.position = worldCentroid
                entity.name = "nebula_\(group.key)"
                let emitter = makeNebulaEmitter(radius: R, startColor: colors.start, endColor: colors.end)
                entity.components.set(emitter)
                nebulaContainer.addChild(entity)
                nebulaEmitters[group.key] = entity
            }
        }
    }

    /// Per-frame position update for existing emitters (particles animate themselves).
    func updateNebulaBreathing() {
        if Self.DIAG_SKIP_NEBULAE { return }
        let groups = nebulaGroupsForCurrentMode()
        let groupByKey = Dictionary(groups.map { ($0.key, $0) }, uniquingKeysWith: { _, last in last })

        for (key, entity) in nebulaEmitters {
            guard let group = groupByKey[key] else { continue }
            entity.position = group.centroid * scaleFactor
        }
    }

    func cameraTransform() -> Transform {
        let pos = cameraPosition
        let fwd = forward
        let rt = right
        let u = up
        let rotation = simd_quatf(simd_float3x3(columns: (rt, u, -fwd)))

        var transform = Transform()
        transform.translation = pos * scaleFactor
        transform.rotation = rotation
        return transform
    }

    func project(point3D: SIMD3<Float>, viewSize: CGSize) -> CGPoint? {
        let camPos = cameraPosition * scaleFactor
        let pointScaled = point3D * scaleFactor

        let toPoint = pointScaled - camPos
        let fwd = forward
        let rt = right
        let u = up

        let depth = dot(toPoint, fwd)
        guard depth > 0.01 * scaleFactor else { return nil }

        let projX = dot(toPoint, rt) / depth
        let projY = dot(toPoint, u) / depth

        let fovTan = tan(Float.pi / 6)  // half of 60° FOV
        let aspect = Float(viewSize.width / viewSize.height)

        let screenX = CGFloat((projX / (fovTan * aspect) * 0.5 + 0.5)) * viewSize.width
        let screenY = CGFloat((0.5 - projY / fovTan * 0.5)) * viewSize.height

        return CGPoint(x: screenX, y: screenY)
    }

    func hitTest(at location: CGPoint, viewSize: CGSize, positions: [Int64: SIMD3<Float>]) -> Int64? {
        var closest: Int64?
        var closestDist: CGFloat = 30

        for (id, pos) in positions {
            guard let screenPos = project(point3D: pos, viewSize: viewSize) else { continue }
            let dist = hypot(location.x - screenPos.x, location.y - screenPos.y)
            if dist < closestDist {
                closestDist = dist
                closest = id
            }
        }
        return closest
    }

    func clearAll() {
        // Clear batched node mesh
        nodeBatchEntity?.removeFromParent()
        nodeBatchEntity = nil
        nodeBatchMesh = nil
        nodeBatchCapacity = 0
        nodeColorCache.removeAll()
        // Clear point lights
        for (_, entity) in pointLightEntities { entity.removeFromParent() }
        pointLightEntities.removeAll()
        // Clear dying node entities
        for (_, entity) in dyingNodeEntities { entity.removeFromParent() }
        dyingNodeEntities.removeAll()
        dyingNodePos3D.removeAll()
        // Clear batched edge mesh
        edgeBatchEntity?.removeFromParent()
        edgeBatchEntity = nil
        edgeBatchMesh = nil
        edgeBatchCapacity = 0
        edgeColorCache.removeAll()
        // Clear batched label mesh
        labelBatchEntity?.removeFromParent()
        labelBatchEntity = nil
        labelBatchMesh = nil
        labelBatchCapacity = 0
        labelAtlasTexture = nil
        labelAtlasRects.removeAll()
        projectLabelAtlasRects.removeAll()
        labelAtlasNodeIds.removeAll()
        labelAtlasHubIds.removeAll()
        labelAtlasProjects.removeAll()
        projectCentroids.removeAll()
        for (_, entity) in nebulaEmitters { entity.removeFromParent() }
        nebulaEmitters.removeAll()
        nebulaColorCache.removeAll()
        // Clean up flow particles (container swap — single entity removal)
        flowParticleContainer.removeFromParent()
        flowParticleContainer = Entity()
        flowParticleContainer.name = "flow_particles"
        rootEntity.addChild(flowParticleContainer)
        flowParticles.removeAll()
        flowSpawnTimers.removeAll()
        // Clean up expansion state
        expandedHubs.removeAll()
        preExpansionPositions.removeAll()
        expansionProgress.removeAll()
        expansionDirection.removeAll()
        expandedChildPositions.removeAll()
    }

    // MARK: - Hub Expansion

    /// Return IDs of children connected to a hub via part_of edges.
    func childrenOfHub(_ hubId: Int64) -> [Int64] {
        renderEdges.filter { $0.relation == "part_of" && $0.targetId == hubId }.map(\.sourceId)
    }

    /// Compute Fibonacci sphere orbit positions for children around a hub.
    func computeOrbitPositions(hubId: Int64, children: [Int64]) -> [Int64: SIMD3<Float>] {
        guard let hubPos = renderPositions[hubId] else { return [:] }
        let radius: Float = 80
        var result: [Int64: SIMD3<Float>] = [:]
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

    /// Toggle hub expansion: start expanding if collapsed, start collapsing if expanded.
    func toggleHubExpansion(hubId: Int64) {
        if expandedHubs.contains(hubId) {
            expansionDirection[hubId] = false
        } else {
            expandedHubs.insert(hubId)
            let children = childrenOfHub(hubId)
            for childId in children {
                preExpansionPositions[childId] = renderPositions[childId] ?? .zero
            }
            expansionProgress[hubId] = 0
            expansionDirection[hubId] = true
        }
    }

    /// Per-frame expansion animation: lerp children between original and orbit positions.
    func updateExpansions(dt: Float) {
        var toRemove: [Int64] = []
        var allExpandedPositions: [Int64: SIMD3<Float>] = [:]

        for hubId in expandedHubs {
            let expanding = expansionDirection[hubId] ?? true
            var progress = expansionProgress[hubId] ?? 0

            if expanding {
                progress = min(1.0, progress + dt * 3.0)
            } else {
                progress = max(0.0, progress - dt * 3.0)
            }
            expansionProgress[hubId] = progress

            // Smooth-step ease
            let t = progress * progress * (3 - 2 * progress)

            let children = childrenOfHub(hubId)
            let orbitPositions = computeOrbitPositions(hubId: hubId, children: children)

            for childId in children {
                let startPos = preExpansionPositions[childId] ?? renderPositions[childId] ?? .zero
                let endPos = orbitPositions[childId] ?? startPos
                let lerpedPos = startPos + (endPos - startPos) * t
                renderPositions[childId] = lerpedPos
                allExpandedPositions[childId] = lerpedPos
                // Position updated in renderPositions — batch mesh picks it up automatically
            }

            // When collapse finishes
            if !expanding && progress <= 0 {
                toRemove.append(hubId)
                for childId in children {
                    if let original = preExpansionPositions[childId] {
                        renderPositions[childId] = original
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

    // MARK: - Edge Flow Particles

    private func flowColorForRelation(_ relation: String) -> SIMD3<Float> {
        switch relation {
        case "part_of":       return SIMD3(0.3, 0.6, 1.0)
        case "contradicts":   return SIMD3(1.0, 0.2, 0.2)
        case "supersedes":    return SIMD3(1.0, 0.6, 0.15)
        case "derived_from":  return SIMD3(0.65, 0.3, 1.0)
        case "summarized_by": return SIMD3(0.2, 0.9, 0.5)
        default:              return SIMD3(0.8, 0.8, 0.9)
        }
    }

    func updateFlowParticles(dt: Float) {
        let sel = renderSelectedNode

        // Hard cleanup on selection change — flush ALL particles so old paths don't linger.
        // Swap the container entity instead of removing children individually.
        // N individual removeFromParent() calls cause N scene graph rebuilds → flicker.
        // One container swap = 1 removal + 1 addition regardless of particle count.
        if sel != flowLastSelectedNode {
            flowParticleContainer.removeFromParent()
            flowParticleContainer = Entity()
            flowParticleContainer.name = "flow_particles"
            rootEntity.addChild(flowParticleContainer)
            flowParticles.removeAll()
            flowSpawnTimers.removeAll()
            flowLastSelectedNode = sel
        }

        // Determine active edges: connected to selected node or expanded hub children
        var activeEdges: [EdgeData] = []

        for edge in renderEdges {
            let isConnectedToSelected = (sel != nil) && (edge.sourceId == sel || edge.targetId == sel)
            let isExpandedPartOf = edge.relation == "part_of" && expandedHubs.contains(edge.targetId)
            if isConnectedToSelected || isExpandedPartOf {
                activeEdges.append(edge)
            }
        }

        // Cap active edges for performance
        if activeEdges.count > 30 {
            activeEdges = Array(activeEdges.prefix(30))
        }

        let activeKeys = Set(activeEdges.map { "\($0.sourceId)-\($0.targetId)" })

        // Clean up particles for inactive edges — hide instead of removing.
        // Set scale to zero (invisible, degenerate) to avoid scene graph churn.
        // Entity cleanup deferred to the container swap on next selection change.
        for key in flowParticles.keys where !activeKeys.contains(key) {
            if let particles = flowParticles[key] {
                for p in particles {
                    p.entity.scale = .zero
                    p.entity.components.set(OpacityComponent(opacity: 0))
                }
            }
            flowParticles.removeValue(forKey: key)
            flowSpawnTimers.removeValue(forKey: key)
        }

        // Update/spawn particles for active edges
        for edge in activeEdges {
            let key = "\(edge.sourceId)-\(edge.targetId)"
            guard let p1 = renderPositions[edge.sourceId],
                  let p2 = renderPositions[edge.targetId] else { continue }

            let scaledP1 = p1 * scaleFactor
            let scaledP2 = p2 * scaleFactor
            let color = flowColorForRelation(edge.relation)

            // Spawn timer
            var timer = flowSpawnTimers[key] ?? 0
            timer += dt
            if timer >= 0.4 {
                timer -= 0.4
                var mat = UnlitMaterial()
                mat.color = .init(tint: NSColor(
                    red: CGFloat(color.x), green: CGFloat(color.y),
                    blue: CGFloat(color.z), alpha: 1.0
                ))
                let entity = ModelEntity(mesh: flowParticleMesh, materials: [mat])
                entity.scale = SIMD3<Float>(repeating: 0.006)
                entity.position = scaledP1
                entity.components.set(OpacityComponent(opacity: 0))
                flowParticleContainer.addChild(entity)

                var particles = flowParticles[key] ?? []
                particles.append(FlowParticle(entity: entity, t: 0, speed: 0.8))
                flowParticles[key] = particles
            }
            flowSpawnTimers[key] = timer

            // Advance existing particles
            if var particles = flowParticles[key] {
                var toRemove: [Int] = []
                for i in particles.indices {
                    particles[i].t += dt * particles[i].speed
                    let t = particles[i].t

                    if t >= 1.0 {
                        // Hide instead of removeFromParent — deferred to container swap
                        particles[i].entity.scale = .zero
                        particles[i].entity.components.set(OpacityComponent(opacity: 0))
                        toRemove.append(i)
                        continue
                    }

                    // Position: lerp along edge
                    let pos = scaledP1 + (scaledP2 - scaledP1) * t
                    particles[i].entity.position = pos

                    // Opacity: fade in first 10%, fade out last 15%
                    let opacity: Float
                    if t < 0.1 {
                        opacity = t / 0.1
                    } else if t > 0.85 {
                        opacity = (1.0 - t) / 0.15
                    } else {
                        opacity = 1.0
                    }
                    particles[i].entity.components.set(OpacityComponent(opacity: opacity))
                }

                for i in toRemove.reversed() {
                    particles.remove(at: i)
                }
                flowParticles[key] = particles
            }
        }
    }

    // MARK: - Display-synced rendering via SceneEvents.Update

    /// Retained subscription for RealityKit scene update events (vsync-synced).
    var renderSubscription: EventSubscription?

    // Direct references — owned by GraphView, injected once during setup.
    // renderTick calls methods directly on these objects, bypassing SwiftUI observation.
    // CRITICAL: these must be @ObservationIgnored so reading them in renderTick
    // does NOT trigger SwiftUI body re-evaluation (the root cause of the 50%+ slow frames).
    @ObservationIgnored var simulation3D: ForceSimulation3D?
    @ObservationIgnored var embeddingProjection: EmbeddingProjection?
    @ObservationIgnored var camera3DState: Camera3DState?
    @ObservationIgnored var forcePositionSnapshot3D: [Int64: SIMD3<Float>] = [:]
    @ObservationIgnored var transitionProgress: CGFloat = 0

    // Selection callback — still needed because it writes to a @Binding in Graph3DView.
    @ObservationIgnored var selectionCallback: ((Int64?) -> Void)?

    // Camera centering state (driven by renderTick)
    @ObservationIgnored private var hasCenteredCamera = false
    @ObservationIgnored private var cameraStartTime: Date?
    @ObservationIgnored private var centerTickCount: Int = 0

    /// Render inputs — read by renderTick (SceneEvents.Update).
    /// @ObservationIgnored: internal data channel, must NOT trigger SwiftUI body re-evaluation.
    @ObservationIgnored var renderPositions: [Int64: SIMD3<Float>] = [:]
    /// Shared render store — written directly by GraphView alongside simulation updates.
    /// Eliminates timing mismatch between SwiftUI push and render tick.
    @ObservationIgnored var renderStore: GraphRenderStore?

    /// Convenience accessors — redirect to renderStore so call sites don't change.
    var renderNodes: [NodeData] { renderStore?.nodes ?? [] }
    var renderEdges: [EdgeData] { renderStore?.edges ?? [] }
    var renderHubs: Set<Int64> { renderStore?.hubs ?? [] }
    var renderColorMap: [String: Color] { renderStore?.colorMap ?? [:] }

    @ObservationIgnored var renderSelectedNode: Int64?
    @ObservationIgnored var renderLayoutMode: LayoutMode = .forceDirected
    @ObservationIgnored var renderGlowingNodes: [Int64: Date] = [:]
    @ObservationIgnored var renderNewNodes: [Int64: Date] = [:]
    @ObservationIgnored var renderDyingNodes: [Int64: DyingNode] = [:]
    @ObservationIgnored var renderSemanticClusters3D: [SemanticCluster3D] = []
    @ObservationIgnored var renderTopicGroups: [TopicGroupInfo] = []
    @ObservationIgnored var renderClusters: [[Int64]] = []
    @ObservationIgnored var renderViewSize: CGSize = CGSize(width: 800, height: 600)

    // Search spotlight
    @ObservationIgnored var renderSearchMatchIds: Set<Int64> = []
    @ObservationIgnored var renderIsSearchActive: Bool = false

    // Hub expansion state
    @ObservationIgnored var expandedHubs: Set<Int64> = []
    @ObservationIgnored private var preExpansionPositions: [Int64: SIMD3<Float>] = [:]
    @ObservationIgnored private var expansionProgress: [Int64: Float] = [:]
    @ObservationIgnored private var expansionDirection: [Int64: Bool] = [:]  // true=expanding, false=collapsing
    /// Expansion-adjusted positions for labels (read by Graph3DView).
    @ObservationIgnored private(set) var expandedChildPositions: [Int64: SIMD3<Float>] = [:]
    /// Pending hub toggles from gamepad — consumed by Graph3DView for pinning callback.
    @ObservationIgnored var pendingHubToggles: [(hubId: Int64, expanding: Bool)] = []

    // Edge flow particles
    private struct FlowParticle {
        let entity: ModelEntity
        var t: Float
        let speed: Float
    }
    private var flowParticles: [String: [FlowParticle]] = [:]
    private var flowParticleContainer = Entity()
    private let flowParticleMesh = MeshResource.generateSphere(radius: 1.0)
    private var flowSpawnTimers: [String: Float] = [:]
    private var flowLastSelectedNode: Int64?

    private var renderFrameCount: UInt64 = 0
    private var renderLastSelectedNode: Int64?
    private var renderLastSearchActive: Bool = false
    private var renderLastSearchMatchIds: Set<Int64> = []
    /// Tracks renderStore.colorMapVersion to invalidate nodeColorCache when colors change.
    @ObservationIgnored private var lastColorMapVersion: UInt64 = 0
    /// Dirty counter: after any position change, write node/edge buffers for N consecutive
    /// frames to flush all of RealityKit's internal LowLevelMesh buffers (double/triple-buffered).
    /// Prevents stale/zero buffer from being rendered when we skip a frame.
    private var meshDirtyFrames: Int = 0

    // Frame timing diagnostics for render tick
    private var renderLogCounter: UInt64 = 0
    private var lastRenderTime: CFAbsoluteTime = 0

    #if DEBUG
    /// File-based profiling: accumulates frame data, flushes periodically to /tmp/frame-timing.csv
    private var profilingLines: [String] = []
    private var profilingFileHandle: FileHandle?
    private var profilingReady = false

    func setupProfiling() {
        let path = "/tmp/frame-timing.csv"
        FileManager.default.createFile(atPath: path, contents: nil)
        profilingFileHandle = FileHandle(forWritingAtPath: path)
        let header = "frame,dt_ms,work_ms,expand_ms,nodes_ms,edges_ms,neb_ms,node_count,edge_count,labels_ms,flow_ms,idle,entities,reason\n"
        profilingFileHandle?.write(header.data(using: .utf8)!)
        profilingReady = true
    }

    /// Canvas label timing — written by the SwiftUI overlay, read by renderTick
    var lastCanvasLabelMs: Double = 0

    private func flushProfiling() {
        guard profilingReady, !profilingLines.isEmpty else { return }
        let data = profilingLines.joined().data(using: .utf8)!
        profilingFileHandle?.write(data)
        profilingLines.removeAll(keepingCapacity: true)
    }
    #endif

    /// Called every RealityKit render frame (vsync-synchronized).
    /// Consecutive frames with no meaningful work — used to detect idle state.
    @ObservationIgnored private(set) var idleFrameCount: UInt64 = 0

    func renderTick() {
        let now = CFAbsoluteTimeGetCurrent()
        let dtSec = lastRenderTime > 0 ? Float(now - lastRenderTime) : Float(1.0 / 60.0)
        let dt = Double(dtSec) * 1000.0

        // Always poll inputs + update camera (cheap, needed for responsiveness)
        let preInputSelection = renderSelectedNode
        pollKeyboard(dt: dtSec)
        pollGamepad(dt: dtSec, selectedNode: &renderSelectedNode,
                    positions: renderPositions, viewSize: renderViewSize)
        updateCamera(dt: dtSec)

        // Detect selection change from gamepad/keyboard → notify parent
        if renderSelectedNode != preInputSelection {
            selectionCallback?(renderSelectedNode)
        }

        // --- Single-loop: tick simulation + projection directly (no closures crossing SwiftUI boundary) ---
        var didUpdatePositions = false
        if let sim = simulation3D, let proj = embeddingProjection {
            sim.tick()
            proj.tickAnimation3D()

            // Compute positions (same logic as GraphView.positions3D, but reads objects directly)
            let newPositions: [Int64: SIMD3<Float>]?
            if sim.isSettled && proj.is3DAnimationSettled {
                newPositions = nil  // settled — no position update needed
            } else if renderLayoutMode == .embedding {
                let tsne3D = proj.projectedPositions3D
                if tsne3D.isEmpty {
                    newPositions = sim.positions
                } else if transitionProgress >= 1.0 {
                    newPositions = tsne3D
                } else {
                    // Lerp between force snapshot and t-SNE
                    var blended: [Int64: SIMD3<Float>] = [:]
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
            if let positions = newPositions {
                renderPositions = positions
                didUpdatePositions = true

                // Camera centering (first 3 seconds after positions appear)
                if cameraStartTime == nil { cameraStartTime = Date() }
                let elapsed = Date().timeIntervalSince(cameraStartTime!)
                if (!hasCenteredCamera || elapsed < 3.0) && !isDragging {
                    centerTickCount += 1
                    if centerTickCount % 6 == 0 {
                        centerOnGraph(positions: renderPositions)
                    }
                    if elapsed >= 3.0 { hasCenteredCamera = true }
                }
            }
        }

        // Consume pending hub toggles — pin/unpin children in simulation directly
        if !pendingHubToggles.isEmpty, let sim = simulation3D {
            for toggle in pendingHubToggles {
                let children = renderEdges.filter { $0.relation == "part_of" && $0.targetId == toggle.hubId }.map(\.sourceId)
                for childId in children {
                    if toggle.expanding { sim.pin(childId) } else { sim.unpin(childId) }
                }
            }
            pendingHubToggles.removeAll()
        }

        // --- Idle detection ---
        // Camera still lerping?
        let cameraMoving = abs(targetAzimuth - azimuth) > 0.0001
            || abs(targetElevation - elevation) > 0.0001
            || simd_length(targetCameraPos - cameraTarget) > 0.01
        let hasInput = !heldKeys.isEmpty || GCController.current?.extendedGamepad != nil
            && (abs(GCController.current?.extendedGamepad?.leftThumbstick.xAxis.value ?? 0) > 0.1
             || abs(GCController.current?.extendedGamepad?.leftThumbstick.yAxis.value ?? 0) > 0.1
             || abs(GCController.current?.extendedGamepad?.rightThumbstick.xAxis.value ?? 0) > 0.1
             || abs(GCController.current?.extendedGamepad?.rightThumbstick.yAxis.value ?? 0) > 0.1)
        let hasExpansions = !expandedHubs.isEmpty
        let hasParticles = !flowParticles.isEmpty
        let hasGlowing = !renderGlowingNodes.isEmpty || !renderNewNodes.isEmpty || !renderDyingNodes.isEmpty
        let selectionChanged = renderSelectedNode != renderLastSelectedNode
        let searchChanged = renderIsSearchActive != renderLastSearchActive
            || renderSearchMatchIds != renderLastSearchMatchIds

        // Position change detection: use the authoritative flag from the simulation
        // update path. The old hash-based approach was unreliable — Dictionary iteration
        // order is non-deterministic, so the 3-value hash could collide even when
        // positions changed, causing skipped buffer writes and stale-buffer flicker.
        let positionsChanged = didUpdatePositions

        // When positions change, set dirty counter to flush all RealityKit internal
        // LowLevelMesh buffers (double/triple-buffered). This ensures the GPU never
        // renders from a stale or zero-initialized buffer.
        if positionsChanged || selectionChanged || hasGlowing || searchChanged || hasExpansions {
            meshDirtyFrames = 10
        }

        // Scene-level changes that require entity updates (nodes, edges, nebulae).
        // meshDirtyFrames > 0 means we still need to write buffers to flush internal
        // double/triple-buffering, even if nothing changed THIS frame.
        let sceneNeedsUpdate = meshDirtyFrames > 0
        // Camera-only activity (orbit, keyboard) doesn't need entity work —
        // GPU shaders compute fog/lighting from camera position autonomously.
        let isActive = sceneNeedsUpdate || cameraMoving || hasInput || hasParticles

        if !isActive {
            idleFrameCount += 1
            // When idle: skip all heavy work. Only wake every 30th frame
            // for a cheap maintenance pass (nebula breathing, etc.)
            if idleFrameCount > 10 && idleFrameCount % 30 != 0 {
                lastRenderTime = now
                renderFrameCount &+= 1
                return
            }
        } else {
            idleFrameCount = 0
        }

        animationTime += dtSec

        if renderFrameCount == 10 {
            frameLog.error("[DIAG] SKIP_NEBULAE=\(Self.DIAG_SKIP_NEBULAE) SKIP_POINT_LIGHTS=\(Self.DIAG_SKIP_POINT_LIGHTS) SKIP_GEOM_MOD=\(Self.DIAG_SKIP_GEOM_MOD) SKIP_NODE_BATCH=\(Self.DIAG_SKIP_NODE_BATCH) SKIP_EDGE_BATCH=\(Self.DIAG_SKIP_EDGE_BATCH) SKIP_LABEL_BATCH=\(Self.DIAG_SKIP_LABEL_BATCH)")
        }

        let t0 = CFAbsoluteTimeGetCurrent()
        var msExpand = 0.0, msNodes = 0.0, msEdges = 0.0, msNeb = 0.0, msFlow = 0.0

        // Only do entity updates when scene state changed — camera orbit is free
        // because GPU shaders compute fog/emissive from camera distance autonomously.
        if sceneNeedsUpdate {
            // Hub expansion animation (before node/edge updates so positions are current)
            updateExpansions(dt: dtSec)
            let tExpand = CFAbsoluteTimeGetCurrent()
            msExpand = (tExpand - t0) * 1000

            // Batched nodes: Metal compute stamps sphere instances → single LowLevelMesh.
            if !Self.DIAG_SKIP_NODE_BATCH {
                updateNodeBatch(
                    positions: renderPositions, nodes: renderNodes, hubs: renderHubs,
                    colorMap: renderColorMap, selectedNode: renderSelectedNode,
                    glowingNodes: renderGlowingNodes, newNodes: renderNewNodes
                )
            }
            let tNodes = CFAbsoluteTimeGetCurrent()
            msNodes = (tNodes - tExpand) * 1000

            // Dying nodes: individual PBR entities (max ~5, temporary)
            if !renderDyingNodes.isEmpty || !dyingNodeEntities.isEmpty {
                updateDyingNodes(positions: renderPositions, dyingNodes: renderDyingNodes)
            }

            // Batched edge mesh: rebuild when positions, selection, or search changes.
            // Also re-write during dirty flush frames to flush all internal LowLevelMesh buffers.
            // Camera movement does NOT require rebuild — GPU shader handles fog/pulse.
            if !Self.DIAG_SKIP_EDGE_BATCH,
               positionsChanged || selectionChanged || searchChanged || hasExpansions || meshDirtyFrames > 0 {
                updateEdgeBatch(
                    positions: renderPositions, edges: renderEdges,
                    nodes: renderNodes, hubs: renderHubs,
                    layoutMode: renderLayoutMode,
                    colorMap: renderColorMap, selectedNode: renderSelectedNode
                )
                renderLastSelectedNode = renderSelectedNode
                renderLastSearchActive = renderIsSearchActive
                renderLastSearchMatchIds = renderSearchMatchIds
            }
            let tEdges = CFAbsoluteTimeGetCurrent()
            msEdges = (tEdges - tNodes) * 1000

            // Edge flow particles — only when selection exists
            if renderSelectedNode != nil || !flowParticles.isEmpty {
                updateFlowParticles(dt: dtSec)
            }
            let tFlow = CFAbsoluteTimeGetCurrent()
            msFlow = (tFlow - tEdges) * 1000

            // Nebulae
            if renderFrameCount > 180 {
                if renderFrameCount % 30 == 0 {
                    updateNebulae()
                } else if positionsChanged && renderFrameCount % 4 == 0 {
                    updateNebulaBreathing()
                }
            }
            msNeb = (CFAbsoluteTimeGetCurrent() - tFlow) * 1000

            // Decrement dirty counter — ensures we write buffers for N consecutive
            // frames to flush all RealityKit internal double/triple-buffers.
            meshDirtyFrames = max(0, meshDirtyFrames - 1)
        }

        // GPU billboard labels — update when scene changes or camera moves (for LOD/opacity).
        // Billboarding itself is handled by the geometry modifier on GPU.
        // Must update every frame (no throttle) to avoid stale data in triple-buffered mesh.
        if !Self.DIAG_SKIP_LABEL_BATCH,
           sceneNeedsUpdate || cameraMoving {
            updateLabelBatch(positions: renderPositions, nodes: renderNodes,
                             hubs: renderHubs, selectedNode: renderSelectedNode)
        }

        let elapsed = (CFAbsoluteTimeGetCurrent() - now) * 1000.0

        // Count total entities in scene (used for both logging and CSV)
        func countEntities(_ e: Entity) -> Int {
            1 + e.children.reduce(0) { $0 + countEntities($1) }
        }
        let totalEntities = countEntities(rootEntity)

        renderLogCounter &+= 1
        if renderLogCounter % 60 == 0 || dt > 25 || elapsed > 10 {
            frameLog.error("[3D-render] dt=\(dt, format: .fixed(precision: 1))ms work=\(elapsed, format: .fixed(precision: 2))ms expand=\(msExpand, format: .fixed(precision: 2)) nodes=\(msNodes, format: .fixed(precision: 2)) edges=\(msEdges, format: .fixed(precision: 2)) flow=\(msFlow, format: .fixed(precision: 2)) neb=\(msNeb, format: .fixed(precision: 2)) | n=\(self.renderPositions.count) e=\(self.renderEdges.count) entities=\(totalEntities) nebEmitters=\(self.nebulaEmitters.count) ptLights=\(self.pointLightEntities.count) frame=\(self.renderFrameCount) idle=\(self.idleFrameCount)")
        }

        #if DEBUG
        // Write every frame to CSV for profiling analysis
        if profilingReady {
            let idleFlag = isActive ? 0 : 1
            // Build reason flags: P=posChanged S=selChanged E=expansions G=glowing X=searchChanged C=cameraMoving K=keyInput
            var reason = ""
            if positionsChanged { reason += "P" }
            if selectionChanged { reason += "S" }
            if hasExpansions { reason += "E" }
            if hasGlowing { reason += "G" }
            if searchChanged { reason += "X" }
            if cameraMoving { reason += "C" }
            if hasInput { reason += "K" }
            if reason.isEmpty { reason = "-" }
            profilingLines.append("\(renderFrameCount),\(String(format: "%.2f", dt)),\(String(format: "%.2f", elapsed)),\(String(format: "%.2f", msExpand)),\(String(format: "%.2f", msNodes)),\(String(format: "%.2f", msEdges)),\(String(format: "%.2f", msNeb)),\(renderPositions.count),\(renderEdges.count),\(String(format: "%.2f", lastCanvasLabelMs)),\(String(format: "%.2f", msFlow)),\(idleFlag),\(totalEntities),\(reason)\n")
            if renderLogCounter % 60 == 0 { flushProfiling() }
        }
        #endif

        // Report camera state directly (writes to Camera3DState @Observable —
        // only MinimapView re-renders, GraphView body is NOT re-evaluated).
        // Only write when values differ to avoid triggering SwiftUI observation every frame.
        if let camState = camera3DState {
            if azimuth != camState.azimuth { camState.azimuth = azimuth }
            if cameraPosition != camState.position { camState.position = cameraPosition }
            if cameraTarget != camState.target { camState.target = cameraTarget }
            if didUpdatePositions { camState.positions = renderPositions }
        }

        lastRenderTime = now
        renderFrameCount &+= 1

        // DIAGNOSTIC: main queue saturation probe — measures how much main-thread work
        // is queued between renderTick calls. If this delta is large (~100ms), something
        // on the main queue (SwiftUI body evals, Canvas draws, etc.) is starving SceneEvents.Update.
        let probePostTime = CFAbsoluteTimeGetCurrent()
        let probeFrame = renderFrameCount
        DispatchQueue.main.async {
            let probeDelta = (CFAbsoluteTimeGetCurrent() - probePostTime) * 1000.0
            if probeDelta > 5.0 {
                frameLog.error("[PROBE] frame=\(probeFrame) mainQ-delay=\(probeDelta, format: .fixed(precision: 1))ms")
            }
        }
    }
}

// MARK: - Graph3DView (SwiftUI host)

struct Graph3DView: View {
    let nodes: [NodeData]
    let edges: [EdgeData]
    let hubs: Set<Int64>
    let colorMap: [String: Color]
    let layoutMode: LayoutMode
    @Binding var selectedNode: Int64?
    let glowingNodes: [Int64: Date]
    let newNodes: [Int64: Date]
    let dyingNodes: [Int64: DyingNode]
    let semanticClusters3D: [SemanticCluster3D]
    let topicGroups: [TopicGroupInfo]
    let clusters: [[Int64]]
    let searchMatchIds: Set<Int64>
    let isSearchActive: Bool
    /// Direct references for renderTick — avoids closures crossing SwiftUI observation boundary.
    let simulation3D: ForceSimulation3D
    let embeddingProjection: EmbeddingProjection
    let camera3DState: Camera3DState
    let forcePositionSnapshot3D: [Int64: SIMD3<Float>]
    let transitionProgress: CGFloat
    let renderStore: GraphRenderStore
    @Binding var cameraProjectTarget: String?

    @State private var scene = Graph3DScene()
    @State private var scrollMonitor: Any?
    @State private var updateClosureCount: UInt64 = 0


    var body: some View {
        GeometryReader { geo in
            ZStack {
                realityViewContent
                    .gesture(orbitGesture)
                    .simultaneousGesture(tapGesture(viewSize: geo.size))
                    .onGeometryChange(for: CGRect.self) { proxy in
                        proxy.frame(in: .global)
                    } action: { frame in
                        viewFrame = frame
                        scene.renderViewSize = frame.size
                    }

                // Center reticle (only when gamepad connected)
                if GCController.current != nil {
                    reticleOverlay
                }

                // Teleport project label (fades after 2s)
                // Uses teleportCounter to avoid stomping: each teleport increments
                // the counter, and the dismiss task only clears if counter hasn't changed.
                if let label = scene.teleportLabel {
                    Text(label)
                        .font(.system(size: 24, weight: .bold, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.8))
                        .padding(.horizontal, 16)
                        .padding(.vertical, 8)
                        .background(.black.opacity(0.5), in: .capsule)
                        .transition(.opacity)
                        .allowsHitTesting(false)
                        .task(id: scene.teleportCounter) {
                            let myCounter = scene.teleportCounter
                            try? await Task.sleep(for: .seconds(2))
                            if scene.teleportCounter == myCounter {
                                scene.teleportLabel = nil
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
        .onAppear { installInputMonitor() }
        .onDisappear { removeInputMonitor() }
        .onChange(of: selectedNode) { _, newValue in
            // SwiftUI → scene: push selection when binding changes from parent
            if newValue != scene.renderSelectedNode {
                scene.renderSelectedNode = newValue
                if newValue == nil {
                    collapseAllHubs()
                }
            }
        }
        .onChange(of: cameraProjectTarget) { _, project in
            if let project {
                scene.driveToProject(project)
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
        scene.renderGlowingNodes = glowingNodes
        scene.renderNewNodes = newNodes
        scene.renderDyingNodes = dyingNodes
        scene.renderSemanticClusters3D = semanticClusters3D
        scene.renderTopicGroups = topicGroups
        scene.renderClusters = clusters
        scene.renderSearchMatchIds = searchMatchIds
        scene.renderIsSearchActive = isSearchActive
    }

    private var realityViewContent: some View {
        RealityView { content in
            let (root, camera) = scene.setup()
            #if DEBUG
            scene.setupProfiling()
            #endif
            content.add(root)
            content.add(camera)

            // Inject direct references — renderTick calls methods on these directly,
            // bypassing closures that would cross the SwiftUI observation boundary.
            scene.simulation3D = simulation3D
            scene.embeddingProjection = embeddingProjection
            scene.camera3DState = camera3DState
            scene.renderStore = renderStore
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

    // MARK: - Gestures

    // MARK: - Reticle (gamepad targeting)

    @ViewBuilder
    private var reticleOverlay: some View {
        let hasTarget = scene.reticleTarget != nil
        let nodeById = Dictionary(nodes.map { ($0.id, $0) }, uniquingKeysWith: { _, last in last })

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
            if let targetId = scene.reticleTarget, let node = nodeById[targetId] {
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
        scrollMonitor = NSEvent.addLocalMonitorForEvents(matching: [.scrollWheel, .magnify, .rotate, .keyDown, .keyUp, .flagsChanged]) { [scene] event in
            // --- Shift tracking (flagsChanged) ---
            if event.type == .flagsChanged {
                if event.modifierFlags.contains(.shift) {
                    scene.heldKeys.insert("shift")
                } else {
                    scene.heldKeys.remove("shift")
                }
                return event
            }

            // --- Key events ---
            if event.type == .keyDown || event.type == .keyUp {
                // Skip if a text field has focus (search bar, etc.)
                if let responder = event.window?.firstResponder,
                   responder is NSTextView || responder is NSTextField {
                    scene.heldKeys.removeAll()
                    return event
                }

                let key = event.charactersIgnoringModifiers?.lowercased() ?? ""
                let noMods = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
                    .subtracting(.shift).isEmpty

                if event.type == .keyDown {
                    // Track continuous movement keys
                    if noMods && Self.movementKeys.contains(key) {
                        scene.heldKeys.insert(key)
                        return nil  // consume
                    }
                    // T/R → teleport next/previous project
                    if key == "t" && noMods {
                        scene.teleportToNextProject(positions: scene.renderPositions, nodes: scene.renderNodes, direction: 1)
                        return nil
                    }
                    if key == "r" && noMods {
                        scene.teleportToNextProject(positions: scene.renderPositions, nodes: scene.renderNodes, direction: -1)
                        return nil
                    }
                } else {
                    // keyUp — release held key
                    if Self.movementKeys.contains(key) {
                        scene.heldKeys.remove(key)
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
                // Two-finger twist → look around (camera stays, view rotates)
                let sensitivity: Float = 0.02
                scene.lookRotate(deltaAz: -Float(event.rotation) * sensitivity, deltaEl: 0)
                return nil
            }
            if event.type == .magnify {
                // Pinch → dolly forward/backward toward cursor position
                let amount = Float(event.magnification) * 400
                guard event.window != nil else {
                    scene.dolly(amount: amount)
                    return nil
                }
                // Convert window location to view-local normalized coords (-1…1)
                let locInWindow = event.locationInWindow
                let nx = Float((locInWindow.x - viewFrame.origin.x) / viewFrame.width - 0.5) * 2
                let ny = Float((locInWindow.y - viewFrame.origin.y) / viewFrame.height - 0.5) * 2
                scene.dolly(amount: amount, cursorNX: nx, cursorNY: ny)
                return nil
            }
            // Two-finger scroll → strafe (move camera in screen plane)
            scene.pan(dx: Float(event.scrollingDeltaX), dy: Float(event.scrollingDeltaY))
            return nil
        }
    }

    private func removeInputMonitor() {
        if let monitor = scrollMonitor {
            NSEvent.removeMonitor(monitor)
            scrollMonitor = nil
        }
        scene.heldKeys.removeAll()
    }

    private func collapseAllHubs() {
        for hubId in scene.expandedHubs {
            scene.toggleHubExpansion(hubId: hubId)
            pinUnpinHubChildren(hubId: hubId, expanding: false)
        }
    }

    /// Pin/unpin hub children in the force simulation directly (no closure crossing SwiftUI boundary).
    private func pinUnpinHubChildren(hubId: Int64, expanding: Bool) {
        let children = edges.filter { $0.relation == "part_of" && $0.targetId == hubId }.map(\.sourceId)
        for childId in children {
            if expanding { simulation3D.pin(childId) } else { simulation3D.unpin(childId) }
        }
    }

    private func tapGesture(viewSize: CGSize) -> some Gesture {
        SpatialTapGesture()
            .onEnded { value in
                if let nodeId = scene.hitTest(at: value.location, viewSize: viewSize, positions: scene.renderPositions) {
                    if hubs.contains(nodeId) {
                        let expanding = !scene.expandedHubs.contains(nodeId)
                        // Collapse other expanded hubs first
                        for hubId in scene.expandedHubs where hubId != nodeId {
                            scene.toggleHubExpansion(hubId: hubId)
                            pinUnpinHubChildren(hubId: hubId, expanding: false)
                        }
                        scene.toggleHubExpansion(hubId: nodeId)
                        pinUnpinHubChildren(hubId: nodeId, expanding: expanding)
                        selectedNode = nodeId
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
