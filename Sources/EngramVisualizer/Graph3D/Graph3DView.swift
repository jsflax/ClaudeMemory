import SwiftUI
import RealityKit
import GameController
import Metal
import simd
import os
import Lattice

private let frameLog = Logger(subsystem: "com.claudememory.visualizer", category: "3DFrameTiming")

// MARK: - 3D Scene Manager

/// Manages RealityKit entity hierarchy for the 3D graph.
@Observable
@MainActor
final class Graph3DScene {
    var rootEntity = Entity()
    var cameraEntity: PerspectiveCamera?
    var edgeContainer = Entity()

    // Camera state — inputs write to target* values, updateCamera() lerps toward them.
    // This smooths out irregular input event timing for silky camera motion.

    // Target state (written by gestures/events)
    var targetAzimuth: Float = 0
    var targetElevation: Float = 0.3
    var targetCameraPos: SIMD3<Float> = .zero  // the look-at point

    // Smoothed state (lerped each frame, used for rendering)
    // @ObservationIgnored: written every frame by updateCamera(), only read within
    // renderTick and gesture handlers — never by SwiftUI body evaluation.
    @ObservationIgnored var azimuth: Float = 0
    @ObservationIgnored var elevation: Float = 0.3
    @ObservationIgnored var cameraTarget: SIMD3<Float> = .zero

    var orbitRadius: Float = 2.0 // orbit arm length — set by centerOnGraph to fit the scene
    let smoothing: Float = 0.2  // lerp factor per frame (0=frozen, 1=instant)
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
    var metalDevice: MTLDevice?
    var metalLibrary: MTLLibrary?
    var metalCommandQueue: MTLCommandQueue?
    var edgeSurfaceShader: CustomMaterial.SurfaceShader?

    // Batched node rendering — all nodes as a single LowLevelMesh (1 draw call).
    // Metal compute kernel stamps template sphere vertices at each node position.
    var nodeBatchMesh: LowLevelMesh?
    var nodeBatchEntity: ModelEntity?
    var nodeBatchMaterial: CustomMaterial?
    var nodeBatchCapacity: Int = 0
    var nodeColorCache: [String: SIMD3<Float>] = [:]

    // Metal compute pipeline for GPU sphere stamping (stamp_node_spheres kernel)
    var stampComputePipeline: MTLComputePipelineState?
    var sphereTemplateBuffer: MTLBuffer?     // static GPU buffer: template sphere vertices
    var stampInstanceBuffer: MTLBuffer?       // shared GPU buffer: per-node instance data (per-frame)
    var stampParamsBuffer: MTLBuffer?          // shared GPU buffer: dispatch parameters
    var stampInstanceCapacity: Int = 0         // current instance buffer capacity (node count)

    var sphereTemplateIndices: [UInt32] = []
    var sphereTemplateVertices: [(pos: SIMD3<Float>, norm: SIMD3<Float>)] = []
    var vertsPerSphere: Int = 0
    var indicesPerSphere: Int = 0
    var instanceArray: [NodeInstance] = []     // per-node instance data (CPU staging)

    /// GPU sphere vertex for template buffer. Layout matches Metal SphereTemplateVertex.
    struct GPUSphereVertex {
        var px: Float, py: Float, pz: Float
        var nx: Float, ny: Float, nz: Float
    }

    /// Dispatch parameters for stamp_node_spheres. Layout matches Metal StampParams.
    struct StampParams {
        var vertsPerNode: UInt32
        var nodeCount: UInt32
    }

    // Metal compute pipeline for GPU label quad stamping (stamp_label_quads kernel)
    var labelStampPipeline: MTLComputePipelineState?
    var labelInstanceBuffer: MTLBuffer?
    var labelParamsBuffer: MTLBuffer?
    var labelInstanceCapacity: Int = 0

    /// Per-label instance data for the Metal compute stamping kernel (64 bytes).
    /// Layout matches Metal LabelInstance: packed_float3(12) + float(4) + float4(16) + float4(16) + 4×float(16) = 64.
    struct LabelInstance {
        var ax: Float, ay: Float, az: Float  // anchor (12)
        var halfH: Float                      // (4)
        var u0: Float, v0: Float, u1: Float, v1: Float  // uvRect (16)
        var cr: Float, cg: Float, cb: Float, baseOpacity: Float  // color (16)
        var textAspect: Float                 // (4)
        var maxVisible: Float                 // (4)
        var forwardBias: Float                // (4)
        var flags: UInt32                     // (4)
    }  // Total: 64 bytes

    /// Dispatch parameters for stamp_label_quads. Layout matches Metal LabelStampParams.
    struct LabelStampParams {
        var cx: Float, cy: Float, cz: Float  // cameraPos (12)
        var minDepth: Float                   // (4)
        var depthRange: Float                 // (4)
        var labelCount: UInt32                // (4)
        var _pad0: Float = 0                  // (4) — pad to 28 bytes
        var _pad1: Float = 0                  // (4) — pad to 32 bytes
    }

    // Staging buffer for edge batch GPU-synchronized writes
    var edgeStagingBuffer: MTLBuffer?
    var edgeStagingCapacity: Int = 0

    // Point lights for glowing/arriving/search-matched nodes
    var pointLightEntities: [UUID: Entity] = [:]

    // Dying node entities — individual PBR entities with red flash animation (max ~5 at a time)
    var dyingNodeEntities: [UUID: ModelEntity] = [:]
    var dyingNodePos3D: [UUID: SIMD3<Float>] = [:]

    // Batched edge rendering — all edges as a single LowLevelMesh (1 draw call)
    var edgeBatchMesh: LowLevelMesh?
    var edgeBatchEntity: ModelEntity?
    var edgeBatchMaterial: CustomMaterial?
    var edgeBatchCapacity: Int = 0
    var edgeColorCache: [String: SIMD3<Float>] = [:]

    // Track last mesh part index counts to avoid calling parts.replaceAll() every frame.
    // Calling replaceAll forces RealityKit to rebuild internal structures — can cause 1-frame blanks.
    // Using capacity-based counts so replaceAll only fires on mesh creation/resize.
    @ObservationIgnored var lastNodePartIndexCount: Int = -1
    @ObservationIgnored var lastEdgePartIndexCount: Int = -1

    // Batched label rendering — all labels as billboard quads in a single LowLevelMesh (1 draw call)
    var labelBatchMesh: LowLevelMesh?
    var labelBatchEntity: ModelEntity?
    var labelBatchMaterial: CustomMaterial?
    var labelBatchCapacity: Int = 0
    @ObservationIgnored var lastLabelPartIndexCount: Int = -1
    var labelAtlasTexture: TextureResource?
    var labelAtlasRects: [UUID: (u0: Float, v0: Float, u1: Float, v1: Float)] = [:]
    var labelAtlasNodeIds: Set<UUID> = []
    var labelAtlasHubIds: Set<UUID> = []
    // Project labels — larger billboard quads floating above each project cluster
    var projectLabelAtlasRects: [String: (u0: Float, v0: Float, u1: Float, v1: Float)] = [:]
    var labelAtlasProjects: Set<String> = []
    @ObservationIgnored var projectCentroids: [String: (centroid: SIMD3<Float>, radius: Float, maxY: Float)] = [:]
    /// Correction factor for text aspect ratio: atlasW / (2 * atlasH).
    /// Quad spans -halfW..+halfW (2x) horizontally but 0..halfH (1x) vertically.
    @ObservationIgnored var labelAtlasAspectCorrection: Float = 1.0
    /// Debounce atlas regeneration to prevent flicker from texture/UV mismatch
    /// in RealityKit's triple-buffered LowLevelMesh vertex buffers.
    @ObservationIgnored var labelAtlasRegenFrame: UInt64 = 0
    /// LowLevelTexture backing the atlas — allows in-place pixel updates without
    /// creating a new TextureResource or swapping the material on the entity.
    /// Material swap was the root cause of label flicker (RealityKit rebuild).
    @ObservationIgnored var labelAtlasLLT: LowLevelTexture?
    @ObservationIgnored var labelAtlasAllocW: Int = 0  // allocated pixel width
    @ObservationIgnored var labelAtlasAllocH: Int = 0  // allocated pixel height

    /// 48-byte vertex layout shared by batched nodes and edges.
    struct BatchVertex {
        var px: Float, py: Float, pz: Float     // position
        var nx: Float, ny: Float, nz: Float     // normal
        var u: Float, v: Float                   // uv0
        var cr: Float, cg: Float, cb: Float, ca: Float  // color
    }

    /// Per-node instance data for the Metal compute stamping kernel (32 bytes).
    struct NodeInstance {
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
    let nodeMesh = MeshResource.generateSphere(radius: 1.0)

    // Nebula particle emitter container
    var nebulaContainer = Entity()
    /// Soft gaussian circle texture for nebula particles — generated once, reused.
    var softParticleTexture: TextureResource?

    let scaleFactor: Float = 1.0 / 200.0
    let nodeRadius: Float = 0.04
    let edgeRadius: Float = 0.004

    private struct EdgeKey: Hashable {
        let source: UUID
        let target: UUID
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
            metalCommandQueue = device.makeCommandQueue()
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

            // Generate sphere template and upload to GPU
            let template = Self.generateSphereTemplate(segments: 16, rings: 10)
            vertsPerSphere = template.vertices.count
            indicesPerSphere = template.indices.count
            sphereTemplateIndices = template.indices
            sphereTemplateVertices = template.vertices
            frameLog.error("[setup] sphere template: \(self.vertsPerSphere) verts, \(self.indicesPerSphere) indices")

            // Create compute pipeline for GPU vertex stamping
            if let stampFn = library.makeFunction(name: "stamp_node_spheres"),
               let pipeline = try? device.makeComputePipelineState(function: stampFn) {
                stampComputePipeline = pipeline

                // Upload sphere template to static GPU buffer
                var gpuVerts = template.vertices.map { tv in
                    GPUSphereVertex(px: tv.0.x, py: tv.0.y, pz: tv.0.z,
                                    nx: tv.1.x, ny: tv.1.y, nz: tv.1.z)
                }
                let templateBytes = MemoryLayout<GPUSphereVertex>.stride * gpuVerts.count
                sphereTemplateBuffer = gpuVerts.withUnsafeMutableBufferPointer { ptr in
                    device.makeBuffer(bytes: ptr.baseAddress!, length: templateBytes, options: .storageModeShared)
                }

                // Pre-allocate params buffer (reused every frame)
                stampParamsBuffer = device.makeBuffer(length: MemoryLayout<StampParams>.stride, options: .storageModeShared)
                frameLog.error("[setup] ✅ stamp_node_spheres compute pipeline created")
            } else {
                frameLog.error("[setup] ❌ stamp_node_spheres compute pipeline FAILED — falling back to CPU stamping")
            }

            // Create compute pipeline for GPU label quad stamping
            if let labelFn = library.makeFunction(name: "stamp_label_quads"),
               let pipeline = try? device.makeComputePipelineState(function: labelFn) {
                labelStampPipeline = pipeline
                labelParamsBuffer = device.makeBuffer(length: MemoryLayout<LabelStampParams>.stride, options: .storageModeShared)
                frameLog.error("[setup] ✅ stamp_label_quads compute pipeline created")
            } else {
                frameLog.error("[setup] ❌ stamp_label_quads compute pipeline FAILED")
            }

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
    var reticleTarget: UUID?

    /// Tracks whether A was pressed last frame to detect rising edge.
    var prevButtonA = false
    /// Tracks whether B was pressed last frame to detect rising edge.
    var prevButtonB = false
    /// Tracks bumper state for rising-edge cycle detection.
    var prevLB = false
    var prevRB = false
    /// Tracks X/Y buttons for project teleport (Y = next, X = previous).
    var prevButtonX = false
    var prevButtonY = false
    /// Held keyboard keys for WASD/IJKL continuous movement.
    var heldKeys: Set<String> = []
    /// Current project index for teleport cycling.
    var teleportProjectIndex = 0
    var teleportCounter: Int = 0
    /// Name of the project we last teleported to (shown briefly in overlay).
    var teleportLabel: String?

    /// Poll held keyboard keys and apply continuous camera movement. Called each render frame.

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
    func centerOnGraph(positions: [UUID: SIMD3<Float>]) {
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

    // MARK: - Label Batch Header (comment: see Graph3DScene+Labels.swift)

    // MARK: - Nebulae stored properties

    var nebulaEmitters: [String: Entity] = [:]


    /// Cached per-project start/end colors for particle evolving tint.
    var nebulaColorCache: [String: (start: NSColor, end: NSColor)] = [:]

    // MARK: - Display-synced rendering properties

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
    @ObservationIgnored var forcePositionSnapshot3D: [UUID: SIMD3<Float>] = [:]
    @ObservationIgnored var transitionProgress: CGFloat = 0

    // Selection callback — still needed because it writes to a @Binding in Graph3DView.
    @ObservationIgnored var selectionCallback: ((UUID?) -> Void)?

    // Camera centering state (driven by renderTick)
    @ObservationIgnored var hasCenteredCamera = false
    @ObservationIgnored var cameraStartTime: Date?
    @ObservationIgnored var centerTickCount: Int = 0

    /// Render inputs — read by renderTick (SceneEvents.Update).
    /// @ObservationIgnored: internal data channel, must NOT trigger SwiftUI body re-evaluation.
    @ObservationIgnored var renderPositions: [UUID: SIMD3<Float>] = [:]
    /// Shared render store — written directly by GraphView alongside simulation updates.
    /// Eliminates timing mismatch between SwiftUI push and render tick.
    @ObservationIgnored var renderStore: GraphRenderStore?

    /// Convenience accessors — redirect to renderStore so call sites don't change.
    var renderNodes: [NodeData] { renderStore?.nodes ?? [] }
    var renderEdges: [EdgeData] { renderStore?.edges ?? [] }
    var renderHubs: Set<UUID> { renderStore?.hubs ?? [] }
    var renderColorMap: [String: Color] { renderStore?.colorMap ?? [:] }

    @ObservationIgnored var renderSelectedNode: UUID?
    @ObservationIgnored var renderLayoutMode: LayoutMode = .forceDirected
    @ObservationIgnored var renderGlowingNodes: [UUID: Date] = [:]
    @ObservationIgnored var renderNewNodes: [UUID: Date] = [:]
    @ObservationIgnored var renderDyingNodes: [UUID: DyingNode] = [:]
    @ObservationIgnored var renderSemanticClusters3D: [SemanticCluster3D] = []
    @ObservationIgnored var renderTopicGroups: [TopicGroupInfo] = []
    @ObservationIgnored var renderClusters: [[UUID]] = []
    @ObservationIgnored var renderViewSize: CGSize = CGSize(width: 800, height: 600)

    // Search spotlight
    @ObservationIgnored var renderSearchMatchIds: Set<UUID> = []
    @ObservationIgnored var renderIsSearchActive: Bool = false

    // Hub expansion state
    @ObservationIgnored var expandedHubs: Set<UUID> = []
    @ObservationIgnored var preExpansionPositions: [UUID: SIMD3<Float>] = [:]
    @ObservationIgnored var expansionProgress: [UUID: Float] = [:]
    @ObservationIgnored var expansionDirection: [UUID: Bool] = [:]  // true=expanding, false=collapsing
    /// Expansion-adjusted positions for labels (read by Graph3DView).
    @ObservationIgnored var expandedChildPositions: [UUID: SIMD3<Float>] = [:]
    /// Pending hub toggles from gamepad — consumed by Graph3DView for pinning callback.
    @ObservationIgnored var pendingHubToggles: [(hubId: UUID, expanding: Bool)] = []

    // Edge flow particles
    struct FlowParticle {
        let entity: ModelEntity
        var t: Float
        let speed: Float
    }
    var flowParticles: [String: [FlowParticle]] = [:]
    var flowParticleContainer = Entity()
    let flowParticleMesh = MeshResource.generateSphere(radius: 1.0)
    var flowSpawnTimers: [String: Float] = [:]
    var flowLastSelectedNode: UUID?

    var renderFrameCount: UInt64 = 0
    var renderLastSelectedNode: UUID?
    var renderLastSearchActive: Bool = false
    var renderLastSearchMatchIds: Set<UUID> = []
    /// Tracks renderStore.colorMapVersion to invalidate nodeColorCache when colors change.
    @ObservationIgnored var lastColorMapVersion: UInt64 = 0
    /// Dirty counter: after any position change, write node/edge buffers for N consecutive
    /// frames to flush all of RealityKit's internal LowLevelMesh buffers (double/triple-buffered).
    /// Prevents stale/zero buffer from being rendered when we skip a frame.
    var meshDirtyFrames: Int = 0

    // Frame timing diagnostics for render tick
    var renderLogCounter: UInt64 = 0
    var lastRenderTime: CFAbsoluteTime = 0

    #if ENGRAM_INSTRUMENTATION
    /// File-based profiling: accumulates frame data, flushes periodically to /tmp/frame-timing.csv
    var profilingLines: [String] = []
    var profilingFileHandle: FileHandle?
    var profilingReady = false

    /// Canvas label timing — written by the SwiftUI overlay, read by renderTick
    var lastCanvasLabelMs: Double = 0

    /// Label flicker diagnostics — written by updateLabelBatch, flushed to /tmp/label-diag.csv
    var labelDiagLines: [String] = []
    var labelDiagFileHandle: FileHandle?
    var labelDiagReady = false
    #endif

    /// Called every RealityKit render frame (vsync-synchronized).
    /// Consecutive frames with no meaningful work — used to detect idle state.
    @ObservationIgnored var idleFrameCount: UInt64 = 0
}
