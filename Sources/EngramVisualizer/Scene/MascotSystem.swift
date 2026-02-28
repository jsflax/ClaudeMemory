import Metal
import MetalKit
import simd
import AppKit

/// Lightweight info about the node the mascot is visiting.
struct MascotNodeInfo {
    let content: String
    let project: String
    let topic: String
    let importance: Int
    let createdAt: Date
    let lastAccessedAt: Date
}

/// Loads, animates, and draws the robot mascot companion.
/// The mascot floats near the camera and has 5 independently animated parts:
/// body, left arm, right arm, eyes, and bottom thruster.
@MainActor
final class MascotSystem {

    // Part indices matching the binary file order
    private enum Part: Int, CaseIterable {
        case body = 0, leftArm, rightArm, eye, bottom
    }

    private struct PartMesh {
        let vertexBuffer: MTLBuffer
        let indexBuffer: MTLBuffer
        let vertexCount: Int
        let indexCount: Int
        let pivot: SIMD3<Float>
    }

    private let device: MTLDevice
    private var parts: [PartMesh] = []
    private var uniformBuffer: MTLBuffer?
    private var baseColorTexture: MTLTexture?
    private var metalRoughTexture: MTLTexture?
    private var sampler: MTLSamplerState?

    // Animation state
    private var time: Float = 0
    private(set) var currentPosition: SIMD3<Float> = .zero
    private var initialized = false
    private var currentDynamicScale: Float = 0

    // Chat state — when chatting, mascot stops patrolling and faces the camera
    private(set) var isChatting = false

    // Fixed render-space scale — mascot is about 3x node size (nodes are radius 0.04)
    private let mascotScale: Float = 0.06

    // Patrol state — mascot flies between nodes
    private var patrolTarget: SIMD3<Float> = .zero
    private var patrolHoverTimer: Float = 0
    private let patrolHoverDuration: Float = 8.0  // total seconds at a node (includes fade-out)
    private let holoDelay: Float = 1.5  // seconds after hover starts before holo appears
    private let holoFadeOutLead: Float = 1.0  // seconds before departure to start fading holo out
    private(set) var isHovering = false
    private var patrolSpeed: Float = 0.3  // render-space units/sec

    // Smooth yaw — lerps toward target direction instead of snapping
    private var currentYaw: Float = 0

    // Thruster particles
    private struct ThrusterParticle {
        var position: SIMD3<Float>
        var velocity: SIMD3<Float>
        var life: Float      // 0..maxLife
        var maxLife: Float
        var size: Float
        var color: SIMD3<Float>
    }
    private var thrusterParticles: [ThrusterParticle] = []
    private var thrusterSpawnTimer: Float = 0
    private(set) var thrusterVertexBuffer: MTLBuffer?
    private(set) var thrusterIndexBuffer: MTLBuffer?
    private(set) var thrusterVertexCapacity: Int = 0
    private(set) var actualThrusterParticleCount: Int = 0

    // Node awareness — which node the mascot is currently visiting
    private(set) var currentTargetId: UUID?

    // Arcane circle
    private var arcaneUniformBuffer: MTLBuffer?
    private var arcaneIndexBuffer: MTLBuffer?
    private(set) var arcaneIntensity: Float = 0  // smoothly lerps to 1 when hovering
    private(set) var arcaneVisible: Bool = false

    // Node inspection rings (3 tilted rings orbiting the inspected node)
    private var ringUniformBuffers: [MTLBuffer] = []  // 3 buffers, one per ring
    private var ringIndexBuffer: MTLBuffer?
    private var ringIntensity: Float = 0  // fades synced with arcaneIntensity
    private(set) var ringTargetId: UUID?   // persists while fading out (unlike currentTargetId)
    private var ringCenter: SIMD3<Float> = .zero  // cached node position for fade-out
    private(set) var ringsVisible: Bool = false

    // Holo info screen
    private var holoUniformBuffer: MTLBuffer?
    private var holoIndexBuffer: MTLBuffer?
    private var holoTexture: MTLTexture?
    private var holoIntensity: Float = 0  // smoothly lerps to 1 when hovering
    private(set) var holoVisible: Bool = false
    private var lastHoloTargetId: UUID?  // tracks which node's texture is currently rendered
    private var holoRevealProgress: Float = 0  // typewriter effect 0→1

    /// Always false — mascot is always animating (bob, patrol, particles).
    var isSettled: Bool { false }

    /// Toggle chat mode. When chatting, mascot stops patrolling and faces the camera.
    func setChatting(_ chatting: Bool) {
        isChatting = chatting
        if chatting {
            isHovering = true
            patrolHoverTimer = 0
        } else {
            isHovering = false
        }
    }

    init(device: MTLDevice) {
        self.device = device
        loadMesh()
        loadTextures()
        createSampler()
        uniformBuffer = device.makeBuffer(
            length: MemoryLayout<MascotUniforms>.stride,
            options: .storageModeShared
        )
        setupArcaneBuffers()
        setupRingBuffers()
        setupHoloBuffers()
    }

    private func setupArcaneBuffers() {
        arcaneUniformBuffer = device.makeBuffer(
            length: MemoryLayout<ArcaneCircleUniforms>.stride,
            options: .storageModeShared
        )
        // 6 indices for 1 quad (4 vertices expanded from center in vertex shader)
        var indices: [UInt32] = [0, 2, 1, 1, 2, 3]
        arcaneIndexBuffer = device.makeBuffer(
            bytes: &indices,
            length: indices.count * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )
    }

    private func setupRingBuffers() {
        for _ in 0..<3 {
            if let buf = device.makeBuffer(
                length: MemoryLayout<ArcaneCircleUniforms>.stride,
                options: .storageModeShared
            ) {
                ringUniformBuffers.append(buf)
            }
        }
        // Shared 6-index buffer for quad (same as arcane circle)
        var indices: [UInt32] = [0, 2, 1, 1, 2, 3]
        ringIndexBuffer = device.makeBuffer(
            bytes: &indices,
            length: indices.count * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )
    }

    private func setupHoloBuffers() {
        holoUniformBuffer = device.makeBuffer(
            length: MemoryLayout<HoloScreenUniforms>.stride,
            options: .storageModeShared
        )
        // 6 indices for 1 quad (4 vertices expanded in vertex shader)
        var indices: [UInt32] = [0, 2, 1, 1, 2, 3]
        holoIndexBuffer = device.makeBuffer(
            bytes: &indices,
            length: indices.count * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )
    }

    // MARK: - Asset Loading

    private func loadMesh() {
        guard let url = Bundle.main.url(forResource: "mascot_mesh", withExtension: "bin") else {
            print("[MascotSystem] mascot_mesh.bin not found in bundle")
            return
        }
        guard let data = try? Data(contentsOf: url) else {
            print("[MascotSystem] Failed to read mascot_mesh.bin")
            return
        }

        data.withUnsafeBytes { raw in
            let base = raw.baseAddress!
            var offset = 0

            // Header: part count
            let partCount = base.load(fromByteOffset: offset, as: UInt32.self)
            offset += 4
            guard partCount == 5 else {
                print("[MascotSystem] Unexpected part count: \(partCount)")
                return
            }

            // Per-part headers (20 bytes each)
            struct PartHeader {
                let vertexCount: UInt32
                let indexCount: UInt32
                let pivotX: Float
                let pivotY: Float
                let pivotZ: Float
            }

            var headers: [PartHeader] = []
            for _ in 0..<5 {
                let vc = base.load(fromByteOffset: offset, as: UInt32.self); offset += 4
                let ic = base.load(fromByteOffset: offset, as: UInt32.self); offset += 4
                let px = base.load(fromByteOffset: offset, as: Float.self); offset += 4
                let py = base.load(fromByteOffset: offset, as: Float.self); offset += 4
                let pz = base.load(fromByteOffset: offset, as: Float.self); offset += 4
                headers.append(PartHeader(vertexCount: vc, indexCount: ic, pivotX: px, pivotY: py, pivotZ: pz))
            }

            // Per-part vertex + index data
            for i in 0..<5 {
                let h = headers[i]
                let vertBytes = Int(h.vertexCount) * 36  // 9 floats: pos(3)+normal(3)+uv(2)+skinWeight(1)
                let idxBytes = Int(h.indexCount) * 4      // uint32

                guard let vtxBuf = device.makeBuffer(
                    bytes: base + offset, length: vertBytes, options: .storageModeShared
                ) else { offset += vertBytes + idxBytes; continue }
                offset += vertBytes

                guard let idxBuf = device.makeBuffer(
                    bytes: base + offset, length: idxBytes, options: .storageModeShared
                ) else { offset += idxBytes; continue }
                offset += idxBytes

                parts.append(PartMesh(
                    vertexBuffer: vtxBuf,
                    indexBuffer: idxBuf,
                    vertexCount: Int(h.vertexCount),
                    indexCount: Int(h.indexCount),
                    pivot: SIMD3(h.pivotX, h.pivotY, h.pivotZ)
                ))
            }
        }

        if parts.count == 5 {
            print("[MascotSystem] Loaded 5 mesh parts")
        }
    }

    private func loadTextures() {
        let loader = MTKTextureLoader(device: device)
        let options: [MTKTextureLoader.Option: Any] = [
            .SRGB: false,
            .generateMipmaps: true,
        ]

        if let url = Bundle.main.url(forResource: "mascot_basecolor", withExtension: "jpg") {
            baseColorTexture = try? loader.newTexture(URL: url, options: options)
        }
        if let url = Bundle.main.url(forResource: "mascot_metalrough", withExtension: "png") {
            metalRoughTexture = try? loader.newTexture(URL: url, options: options)
        }

        if baseColorTexture != nil && metalRoughTexture != nil {
            print("[MascotSystem] Loaded textures")
        } else {
            print("[MascotSystem] Warning: missing textures (base=\(baseColorTexture != nil), mr=\(metalRoughTexture != nil))")
        }
    }

    private func createSampler() {
        let desc = MTLSamplerDescriptor()
        desc.minFilter = .linear
        desc.magFilter = .linear
        desc.mipFilter = .linear
        desc.sAddressMode = .repeat
        desc.tAddressMode = .repeat
        sampler = device.makeSamplerState(descriptor: desc)
    }

    // MARK: - Animation

    /// Pick a new random node to fly to, keeping track of which node UUID it is.
    private func pickNewPatrolTarget(positions: [UUID: SIMD3<Float>], scaleFactor: Float) {
        let scaled = positions.map { (id: $0.key, pos: $0.value * scaleFactor) }
        guard !scaled.isEmpty else { return }
        // Pick a random node, but try to avoid the one we're already near
        var candidate = scaled.randomElement()!
        for _ in 0..<3 {
            let next = scaled.randomElement()!
            if simd_length(next.pos - currentPosition) > simd_length(candidate.pos - currentPosition) {
                candidate = next
                break
            }
        }
        currentTargetId = candidate.id
        // Hover slightly above the node
        patrolTarget = candidate.pos + SIMD3<Float>(0, mascotScale * 2.0, 0)
    }

    /// Update mascot position and animation. Call once per frame.
    func update(dt: Float, camera: CameraController, nodePositions: [UUID: SIMD3<Float>], nodeInfo: [UUID: MascotNodeInfo] = [:]) {
        time += dt
        guard parts.count == 5 else { return }

        let sf: Float = camera.scaleFactor
        let dynamicScale: Float = mascotScale

        // ── Patrol behavior: fly between nodes ──
        if !initialized {
            // Start at graph center
            let graphCenter: SIMD3<Float> = camera.cameraTarget * sf
            currentPosition = graphCenter
            pickNewPatrolTarget(positions: nodePositions, scaleFactor: sf)
            initialized = true
            isHovering = false
        }

        if !isChatting {
            let distToTarget: Float = simd_length(patrolTarget - currentPosition)

            if isHovering {
                // Hover at current node
                patrolHoverTimer += dt
                if patrolHoverTimer >= patrolHoverDuration {
                    patrolHoverTimer = 0
                    isHovering = false
                    pickNewPatrolTarget(positions: nodePositions, scaleFactor: sf)
                }
            } else {
                // Flying toward target
                if distToTarget < mascotScale * 0.5 {
                    // Arrived — start hovering
                    isHovering = true
                    patrolHoverTimer = 0
                } else {
                    // Smooth flight with ease-in/ease-out
                    let dir: SIMD3<Float> = simd_normalize(patrolTarget - currentPosition)
                    // Speed ramps down as we approach target for smooth arrival
                    let approachFactor: Float = min(distToTarget / (mascotScale * 3.0), 1.0)
                    let speed: Float = patrolSpeed * approachFactor
                    let step: SIMD3<Float> = dir * speed * dt
                    if simd_length(step) < distToTarget {
                        currentPosition += step
                    } else {
                        currentPosition = patrolTarget
                    }
                }
            }
        }
        // else: chatting — stay at current position, keep hovering

        // Face direction: camera when chatting, travel direction when patrolling
        let desiredYaw: Float
        if isChatting {
            // Face toward camera
            let camPos = camera.cameraPosition * sf
            let toCamera = camPos - currentPosition
            if simd_length(toCamera) > 0.001 {
                desiredYaw = atan2(toCamera.x, toCamera.z)
            } else {
                desiredYaw = currentYaw
            }
        } else {
            // Face the direction of travel
            let toTarget: SIMD3<Float> = patrolTarget - currentPosition
            if simd_length(toTarget) > 0.001 {
                desiredYaw = atan2(toTarget.x, toTarget.z)
            } else {
                desiredYaw = currentYaw  // hold current facing when hovering
            }
        }
        // Shortest-path angle difference
        var yawDiff = desiredYaw - currentYaw
        if yawDiff > .pi { yawDiff -= 2 * .pi }
        if yawDiff < -.pi { yawDiff += 2 * .pi }
        let yawSpeed: Float = isHovering ? 1.5 : 3.0  // slower turn when hovering
        currentYaw += yawDiff * min(dt * yawSpeed, 1.0)

        // Build per-part model matrices
        guard let uniformBuf = uniformBuffer else { return }
        let ptr = uniformBuf.contents().bindMemory(to: MascotUniforms.self, capacity: 1)

        // Idle animation: bob + lean (more when hovering, less when flying)
        let bobY: Float = sin(time * 2.0) * dynamicScale * 0.15
        let idleYaw: Float = currentYaw
        let flyLean: Float = isHovering ? 0.0 : 0.2  // tilt forward when flying
        let leanZ: Float = sin(time * 1.3) * 0.08 + flyLean
        let leanX: Float = sin(time * 0.9) * 0.06

        let bodyTranslation: simd_float4x4 = translationMatrix(currentPosition + SIMD3<Float>(0, bobY, 0))
        let bodyYaw: simd_float4x4 = rotationY(idleYaw)
        let bodyLeanZ: simd_float4x4 = rotationZ(leanZ)
        let bodyLeanX: simd_float4x4 = rotationX(leanX)
        let bodyScale: simd_float4x4 = scaleMatrix(dynamicScale)
        let bodyRot: simd_float4x4 = bodyYaw * bodyLeanZ * bodyLeanX
        let bodyRotScale: simd_float4x4 = bodyRot * bodyScale
        let bodyMatrix: simd_float4x4 = bodyTranslation * bodyRotScale

        let zp = SIMD3<Float>(0, 0, 0)

        // Arm swing animation — pendulum rotation around shoulder pivots.
        // skinWeight in the mesh data (0 at shoulder → 1 at hand) blends
        // between parentMatrix (body) and modelMatrix (arm) in the shader,
        // so vertices near the shoulder stay with the body and the hand
        // follows the arm rotation fully.
        // Dampen swing during flight to prevent stretching from rapid yaw changes.
        let swingAmplitude: Float = isHovering ? 0.4 : 0.15
        let leftSwing: Float = sin(time * 2.5) * swingAmplitude
        let rightSwing: Float = sin(time * 2.5 + .pi) * swingAmplitude  // opposite phase

        // Left arm pivot at ball joint center (from the Python script)
        let leftPivot = SIMD3<Float>(-0.68, -0.21, 0.10)
        let toPivotL: simd_float4x4 = translationMatrix(-leftPivot)
        let fromPivotL: simd_float4x4 = translationMatrix(leftPivot)
        let leftArmRot: simd_float4x4 = rotationX(leftSwing)
        // Arm matrix: body * fromPivot * rotation * toPivot
        // (moves to pivot origin, rotates, moves back, then applies body transform)
        let leftArmLocal: simd_float4x4 = fromPivotL * leftArmRot * toPivotL
        // Pivot rotation in model space (before scale), so pivot units match vertex units
        let leftArmMatrix: simd_float4x4 = bodyTranslation * bodyRot * bodyScale * leftArmLocal

        // Right arm pivot at ball joint center (from the Python script)
        let rightPivot = SIMD3<Float>(0.67, -0.21, 0.10)
        let toPivotR: simd_float4x4 = translationMatrix(-rightPivot)
        let fromPivotR: simd_float4x4 = translationMatrix(rightPivot)
        let rightArmRot: simd_float4x4 = rotationX(rightSwing)
        let rightArmLocal: simd_float4x4 = fromPivotR * rightArmRot * toPivotR
        let rightArmMatrix: simd_float4x4 = bodyTranslation * bodyRot * bodyScale * rightArmLocal

        // Body
        ptr.pointee.parts.0 = MascotPartUniforms(
            modelMatrix: bodyMatrix, parentMatrix: bodyMatrix,
            emissive: SIMD4<Float>(0, 0, 0, 0), blendPivot: zp, blendRadius: 0
        )

        // Left arm — skinned: modelMatrix = arm transform, parentMatrix = body,
        // blendRadius > 0 activates per-vertex skinning in the shader
        ptr.pointee.parts.1 = MascotPartUniforms(
            modelMatrix: leftArmMatrix, parentMatrix: bodyMatrix,
            emissive: SIMD4<Float>(0, 0, 0, 0), blendPivot: leftPivot, blendRadius: 1.0
        )

        // Right arm — skinned
        ptr.pointee.parts.2 = MascotPartUniforms(
            modelMatrix: rightArmMatrix, parentMatrix: bodyMatrix,
            emissive: SIMD4<Float>(0, 0, 0, 0), blendPivot: rightPivot, blendRadius: 1.0
        )

        // Eye: cyan emissive pulse
        let eyePulse: Float = 0.6 + 0.4 * sin(time * 3.0)
        ptr.pointee.parts.3 = MascotPartUniforms(
            modelMatrix: bodyMatrix, parentMatrix: bodyMatrix,
            emissive: SIMD4<Float>(0, 0.8, 1.0, eyePulse), blendPivot: zp, blendRadius: 0
        )

        // Bottom: blue thruster glow
        let bottomPulse: Float = 0.4 + 0.3 * sin(time * 4.0)
        ptr.pointee.parts.4 = MascotPartUniforms(
            modelMatrix: bodyMatrix, parentMatrix: bodyMatrix,
            emissive: SIMD4<Float>(0.2, 0.4, 1.0, bottomPulse), blendPivot: zp, blendRadius: 0
        )

        // ── Thruster particles ──
        updateThrusterParticles(dt: dt, bodyMatrix: bodyMatrix, dynamicScale: dynamicScale)

        // ── Arcane circle ──
        updateArcaneCircle(dt: dt, bodyMatrix: bodyMatrix, bodyRot: bodyRot, dynamicScale: dynamicScale)

        // ── Node inspection rings ──
        updateNodeRings(dt: dt, nodePositions: nodePositions, scaleFactor: sf, dynamicScale: dynamicScale)

        // ── Holo info screen ──
        updateHoloScreen(dt: dt, bodyMatrix: bodyMatrix, bodyRot: bodyRot, dynamicScale: dynamicScale, nodeInfo: nodeInfo)
    }

    // MARK: - Thruster Particles

    private func updateThrusterParticles(dt: Float, bodyMatrix: simd_float4x4, dynamicScale: Float) {
        // Bottom of the mascot in model space is approximately Y = -0.8
        // Transform to world space using the body matrix
        let bottomLocal = SIMD4<Float>(0, -0.85, 0, 1)
        let bottomWorld4: SIMD4<Float> = bodyMatrix * bottomLocal
        let emitPos = SIMD3<Float>(bottomWorld4.x, bottomWorld4.y, bottomWorld4.z)

        // Spawn particles — thick plume
        let spawnRate: Float = 1.0 / 120.0  // 120 particles/sec
        thrusterSpawnTimer += dt
        while thrusterSpawnTimer >= spawnRate {
            thrusterSpawnTimer -= spawnRate

            // Wide cone pointing down with some horizontal spread
            let spreadX: Float = Float.random(in: -0.3...0.3) * dynamicScale
            let spreadZ: Float = Float.random(in: -0.3...0.3) * dynamicScale
            let speedY: Float = -Float.random(in: 0.4...0.9) * dynamicScale
            let vel = SIMD3<Float>(spreadX, speedY, spreadZ)

            let life: Float = Float.random(in: 0.6...1.2)
            let size: Float = dynamicScale * Float.random(in: 0.10...0.22)

            // Blue-cyan color with slight variation
            let r: Float = Float.random(in: 0.1...0.35)
            let g: Float = Float.random(in: 0.4...0.75)
            let b: Float = Float.random(in: 0.8...1.0)

            thrusterParticles.append(ThrusterParticle(
                position: emitPos, velocity: vel,
                life: 0, maxLife: life, size: size,
                color: SIMD3<Float>(r, g, b)
            ))
        }

        // Advance particles
        thrusterParticles.removeAll { $0.life >= $0.maxLife }
        for i in thrusterParticles.indices {
            thrusterParticles[i].life += dt
            thrusterParticles[i].position += thrusterParticles[i].velocity * dt
            // Gentle slowdown
            thrusterParticles[i].velocity *= (1.0 - dt * 1.5)
        }

        // Pack into FlowParticleVertex buffer for GPU rendering
        let count = thrusterParticles.count
        actualThrusterParticleCount = count
        guard count > 0 else { return }

        // Ensure buffers (4 verts per particle quad, 6 indices per quad)
        if thrusterVertexCapacity < count {
            let cap = max(count * 2, 64)
            thrusterVertexBuffer = device.makeBuffer(
                length: cap * 4 * MemoryLayout<FlowParticleVertex>.stride,
                options: .storageModeShared
            )
            let idxCount = cap * 6
            thrusterIndexBuffer = device.makeBuffer(
                length: idxCount * MemoryLayout<UInt32>.stride,
                options: .storageModeShared
            )
            if let buf = thrusterIndexBuffer {
                let indices = buf.contents().bindMemory(to: UInt32.self, capacity: idxCount)
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
            thrusterVertexCapacity = cap
        }

        // Write one FlowParticleVertex per particle (flow_vertex expands to 4 corners)
        if let buf = thrusterVertexBuffer {
            let ptr = buf.contents().bindMemory(to: FlowParticleVertex.self, capacity: count)
            for (i, p) in thrusterParticles.enumerated() {
                let t: Float = p.life / p.maxLife
                // Fade in quickly, fade out over life
                let opacity: Float
                if t < 0.05 { opacity = t / 0.05 }
                else { opacity = (1.0 - t) / 0.95 }
                let finalOpacity: Float = opacity * 0.85

                ptr[i] = FlowParticleVertex(
                    position: p.position,
                    uv: SIMD2<Float>(0, 0),
                    color: SIMD4<Float>(p.color.x, p.color.y, p.color.z, finalOpacity),
                    size: p.size * (1.0 - t * 0.5),  // shrink over life
                    _pad0: 0, _pad1: 0, _pad2: 0
                )
            }
        }
    }

    // MARK: - Arcane Circle

    private func updateArcaneCircle(dt: Float, bodyMatrix: simd_float4x4, bodyRot: simd_float4x4, dynamicScale: Float) {
        // Fade in when hovering, fade out when flying
        let target: Float = isHovering ? 1.0 : 0.0
        arcaneIntensity += (target - arcaneIntensity) * min(dt * 4.0, 1.0)
        arcaneVisible = arcaneIntensity > 0.01

        guard arcaneVisible, parts.count == 5, let buf = arcaneUniformBuffer else { return }

        // Eye centroid in model space → world space, offset forward (+Z in model space)
        let eyePivot = parts[Part.eye.rawValue].pivot
        let eyeForward = SIMD4<Float>(eyePivot.x, eyePivot.y, eyePivot.z + 0.35, 1.0)
        let eyeWorld4 = bodyMatrix * eyeForward
        let center = SIMD3<Float>(eyeWorld4.x, eyeWorld4.y, eyeWorld4.z)

        // Mascot orientation vectors from body rotation (columns of the rotation matrix)
        let right = simd_normalize(SIMD3<Float>(bodyRot.columns.0.x, bodyRot.columns.0.y, bodyRot.columns.0.z))
        let up    = simd_normalize(SIMD3<Float>(bodyRot.columns.1.x, bodyRot.columns.1.y, bodyRot.columns.1.z))

        let ptr = buf.contents().bindMemory(to: ArcaneCircleUniforms.self, capacity: 1)
        ptr.pointee = ArcaneCircleUniforms(
            center: center,
            size: dynamicScale * 1.8,
            right: right,
            opacity: arcaneIntensity,
            up: up,
            _pad: 0
        )
    }

    // MARK: - Node Inspection Rings

    private func updateNodeRings(dt: Float, nodePositions: [UUID: SIMD3<Float>], scaleFactor: Float, dynamicScale: Float) {
        // Fade in/out synced with arcane circle
        ringIntensity = arcaneIntensity
        ringsVisible = ringIntensity > 0.01

        // Latch the target node when hovering starts; keep it while fading out
        if isHovering, let targetId = currentTargetId, let nodePos = nodePositions[targetId] {
            ringTargetId = targetId
            ringCenter = nodePos * scaleFactor
        }
        // Clear once fully faded
        if ringIntensity <= 0.01 {
            ringTargetId = nil
        }

        guard ringsVisible,
              ringUniformBuffers.count == 3,
              ringTargetId != nil else {
            ringsVisible = false
            return
        }

        let center = ringCenter
        let ringSize = dynamicScale * 2.5

        // 3 rings at different tilt angles and rotation speeds
        let ringConfigs: [(tiltAngle: Float, speed: Float, offset: Float)] = [
            (0.52, 0.4, 0.0),       // ~30° tilt, slow rotation
            (1.05, -0.3, 2.094),     // ~60° tilt, reverse rotation
            (1.40, 0.2, 4.189),      // ~80° tilt (near-equatorial), slowest
        ]

        for (i, config) in ringConfigs.enumerated() {
            let angle = time * config.speed + config.offset
            let tilt = config.tiltAngle

            // Compute right/up vectors for a tilted ring plane
            let cosA = cos(angle), sinA = sin(angle)
            let cosT = cos(tilt), sinT = sin(tilt)

            // Right vector rotates around Y axis
            let right = SIMD3<Float>(cosA, 0, sinA)

            // Up vector: tilt away from Y axis
            let up = SIMD3<Float>(
                -sinA * sinT,
                cosT,
                cosA * sinT
            )

            let ptr = ringUniformBuffers[i].contents().bindMemory(to: ArcaneCircleUniforms.self, capacity: 1)
            ptr.pointee = ArcaneCircleUniforms(
                center: center,
                size: ringSize,
                right: right,
                opacity: ringIntensity,
                up: up,
                _pad: 0
            )
        }
    }

    // MARK: - Holo Info Screen

    private func updateHoloScreen(dt: Float, bodyMatrix: simd_float4x4, bodyRot: simd_float4x4, dynamicScale: Float, nodeInfo: [UUID: MascotNodeInfo]) {
        // Fade in after arcane circle, fade out before mascot departs
        let timeUntilDepart = patrolHoverDuration - patrolHoverTimer
        let holoReady = isHovering && patrolHoverTimer >= holoDelay && timeUntilDepart > holoFadeOutLead
        let target: Float = holoReady ? 1.0 : 0.0
        let fadeSpeed: Float = holoReady ? 2.5 : 4.0  // fade out faster than fade in
        holoIntensity += (target - holoIntensity) * min(dt * fadeSpeed, 1.0)
        holoVisible = holoIntensity > 0.01

        // Generate/regenerate texture only when we arrive at a new node
        if isHovering, let targetId = currentTargetId, targetId != lastHoloTargetId {
            lastHoloTargetId = targetId
            holoRevealProgress = 0  // reset typewriter
            if let info = nodeInfo[targetId] {
                renderHoloTexture(info: info)
            }
        }
        if !isHovering {
            lastHoloTargetId = nil
            holoRevealProgress = 0
        }

        // Advance typewriter reveal (~5.5s to fully type out)
        if holoReady && holoRevealProgress < 1.0 {
            holoRevealProgress = min(holoRevealProgress + dt * 0.18, 1.0)
        }

        guard holoVisible, let buf = holoUniformBuffer else { return }

        // Mascot orientation vectors from body rotation (same as arcane circle).
        // Negate right so billboard text reads left-to-right from the viewer's perspective
        // (mascot's right is viewer's left when looking at the front).
        let right   = -simd_normalize(SIMD3<Float>(bodyRot.columns.0.x, bodyRot.columns.0.y, bodyRot.columns.0.z))
        let up      = simd_normalize(SIMD3<Float>(bodyRot.columns.1.x, bodyRot.columns.1.y, bodyRot.columns.1.z))
        let forward = simd_normalize(SIMD3<Float>(bodyRot.columns.2.x, bodyRot.columns.2.y, bodyRot.columns.2.z))

        // Position: centered in front of the mascot, slightly above eye level
        let bodyPos = SIMD3<Float>(bodyMatrix.columns.3.x, bodyMatrix.columns.3.y, bodyMatrix.columns.3.z)
        let center = bodyPos + forward * dynamicScale * 3.0 + up * dynamicScale * 1.5

        let ptr = buf.contents().bindMemory(to: HoloScreenUniforms.self, capacity: 1)
        ptr.pointee = HoloScreenUniforms(
            center: center,
            width: dynamicScale * 2.2,
            right: right,
            height: dynamicScale * 1.7,
            up: up,
            opacity: holoIntensity,
            revealProgress: holoRevealProgress,
            _pad0: 0, _pad1: 0, _pad2: 0
        )
    }

    private static let holoDateFormatter: DateFormatter = {
        let df = DateFormatter()
        df.dateFormat = "yyyy-MM-dd HH:mm"
        return df
    }()

    /// Render info card texture using AppKit text rendering. Called only when target node changes.
    private func renderHoloTexture(info: MascotNodeInfo) {
        let texW: CGFloat = 512
        let texH: CGFloat = 400
        let charsPerLine = 50

        // Use a flipped NSImage so drawing coordinates match UV (origin top-left)
        let image = NSImage(size: NSSize(width: texW, height: texH))
        image.lockFocusFlipped(true)

        // Clear to transparent
        NSColor.clear.set()
        NSRect(x: 0, y: 0, width: texW, height: texH).fill()

        let margin: CGFloat = 32
        var y: CGFloat = 28

        // ── Header: Topic / Project ──
        let topicFont = NSFont.monospacedSystemFont(ofSize: 18, weight: .bold)
        let topicAttrs: [NSAttributedString.Key: Any] = [
            .font: topicFont,
            .foregroundColor: NSColor(red: 0.3, green: 0.9, blue: 1.0, alpha: 1.0)
        ]
        let topicStr = "\(info.topic.prefix(24)) / \(info.project.prefix(16))"
        (topicStr as NSString).draw(at: NSPoint(x: margin, y: y), withAttributes: topicAttrs)
        y += 26

        // ── Meta: dates ──
        let metaFont = NSFont.monospacedSystemFont(ofSize: 10, weight: .regular)
        let metaAttrs: [NSAttributedString.Key: Any] = [
            .font: metaFont,
            .foregroundColor: NSColor(red: 0.45, green: 0.6, blue: 0.7, alpha: 0.7)
        ]
        let metaStr = "created \(Self.holoDateFormatter.string(from: info.createdAt))  |  accessed \(Self.holoDateFormatter.string(from: info.lastAccessedAt))"
        (metaStr as NSString).draw(at: NSPoint(x: margin, y: y), withAttributes: metaAttrs)
        y += 16

        // ── Separator ──
        let sepAttrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.monospacedSystemFont(ofSize: 8, weight: .regular),
            .foregroundColor: NSColor(red: 0.3, green: 0.5, blue: 0.6, alpha: 0.4)
        ]
        let sep = String(repeating: "\u{2500}", count: 46)
        (sep as NSString).draw(at: NSPoint(x: margin, y: y), withAttributes: sepAttrs)
        y += 12

        // ── Memory content: fill remaining space ──
        let contentFont = NSFont.monospacedSystemFont(ofSize: 10, weight: .regular)
        let contentAttrs: [NSAttributedString.Key: Any] = [
            .font: contentFont,
            .foregroundColor: NSColor(red: 0.65, green: 0.8, blue: 0.9, alpha: 0.85)
        ]
        let lineHeight: CGFloat = 14
        let usableWidth = texW - margin * 2
        let contentCharsPerLine = Int(usableWidth / 6)  // ~6pt per monospace char at size 10
        let maxLines = Int((texH - y - 20) / lineHeight)
        let content = info.content
        // Word-wrap into lines
        var lines: [String] = []
        var currentLine = ""
        for word in content.split(separator: " ", omittingEmptySubsequences: false) {
            let candidate = currentLine.isEmpty ? String(word) : currentLine + " " + word
            if candidate.count > contentCharsPerLine && !currentLine.isEmpty {
                lines.append(currentLine)
                currentLine = String(word)
            } else {
                currentLine = candidate
            }
        }
        if !currentLine.isEmpty { lines.append(currentLine) }

        for line in lines.prefix(maxLines) {
            (line as NSString).draw(at: NSPoint(x: margin, y: y), withAttributes: contentAttrs)
            y += lineHeight
        }

        image.unlockFocus()

        // Convert NSImage → CGImage → BGRA pixel buffer → MTLTexture
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return }

        let pixelW = cgImage.width
        let pixelH = cgImage.height
        let bytesPerRow = pixelW * 4

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil,
            width: pixelW, height: pixelH,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
        ) else { return }

        // CGImage from flipped NSImage is already top-down; draw without flipping
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: pixelW, height: pixelH))

        guard let data = ctx.data else { return }

        let texDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: pixelW, height: pixelH,
            mipmapped: false
        )
        texDesc.usage = .shaderRead
        texDesc.storageMode = .shared

        guard let texture = device.makeTexture(descriptor: texDesc) else { return }
        texture.replace(
            region: MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                              size: MTLSize(width: pixelW, height: pixelH, depth: 1)),
            mipmapLevel: 0,
            withBytes: data,
            bytesPerRow: bytesPerRow
        )
        holoTexture = texture
    }

    // MARK: - Drawing

    /// Draw the mascot: 5 indexed draw calls, one per part.
    func draw(
        encoder: MTLRenderCommandEncoder,
        frameUniformBuf: MTLBuffer,
        lightUniformBuf: MTLBuffer,
        pipeline: MTLRenderPipelineState,
        depthState: MTLDepthStencilState
    ) {
        guard parts.count == 5,
              let uniformBuf = uniformBuffer,
              let baseTex = baseColorTexture,
              let mrTex = metalRoughTexture,
              let samp = sampler else { return }

        encoder.setRenderPipelineState(pipeline)
        encoder.setDepthStencilState(depthState)
        encoder.setCullMode(.front)

        // Shared textures and sampler
        encoder.setFragmentTexture(baseTex, index: 0)
        encoder.setFragmentTexture(mrTex, index: 1)
        encoder.setFragmentSamplerState(samp, index: 0)

        let uniformsPtr = uniformBuf.contents().bindMemory(to: MascotUniforms.self, capacity: 1)

        for i in 0..<5 {
            let part = parts[i]

            // Vertex buffers
            encoder.setVertexBuffer(part.vertexBuffer, offset: 0, index: 0)
            encoder.setVertexBuffer(frameUniformBuf, offset: 0, index: 1)

            // Per-part uniforms: pass as bytes
            let partOffset = i * MemoryLayout<MascotPartUniforms>.stride
            encoder.setVertexBuffer(uniformBuf, offset: partOffset, index: 2)

            // Part index
            var partIdx = UInt32(i)
            encoder.setVertexBytes(&partIdx, length: 4, index: 3)

            // Fragment buffers
            encoder.setFragmentBuffer(frameUniformBuf, offset: 0, index: 0)
            encoder.setFragmentBuffer(lightUniformBuf, offset: 0, index: 1)
            encoder.setFragmentBuffer(uniformBuf, offset: partOffset, index: 2)

            // Draw indexed
            encoder.drawIndexedPrimitives(
                type: .triangle,
                indexCount: part.indexCount,
                indexType: .uint32,
                indexBuffer: part.indexBuffer,
                indexBufferOffset: 0
            )
        }
    }

    /// Draw the arcane circle (transparent pass, additive blending).
    func drawArcaneCircle(
        encoder: MTLRenderCommandEncoder,
        frameUniformBuf: MTLBuffer,
        pipeline: MTLRenderPipelineState
    ) {
        guard arcaneVisible,
              let uniformBuf = arcaneUniformBuffer,
              let idxBuf = arcaneIndexBuffer else { return }

        encoder.setRenderPipelineState(pipeline)
        encoder.setVertexBuffer(uniformBuf, offset: 0, index: 0)
        encoder.setVertexBuffer(frameUniformBuf, offset: 0, index: 1)
        encoder.setFragmentBuffer(frameUniformBuf, offset: 0, index: 0)
        encoder.drawIndexedPrimitives(
            type: .triangle,
            indexCount: 6,
            indexType: .uint32,
            indexBuffer: idxBuf,
            indexBufferOffset: 0
        )
    }

    /// Draw the 3 node inspection rings (transparent pass, additive blending).
    func drawNodeRings(
        encoder: MTLRenderCommandEncoder,
        frameUniformBuf: MTLBuffer,
        pipeline: MTLRenderPipelineState
    ) {
        guard ringsVisible,
              ringUniformBuffers.count == 3,
              let idxBuf = ringIndexBuffer else { return }

        encoder.setRenderPipelineState(pipeline)
        encoder.setFragmentBuffer(frameUniformBuf, offset: 0, index: 0)

        for i in 0..<3 {
            encoder.setVertexBuffer(ringUniformBuffers[i], offset: 0, index: 0)
            encoder.setVertexBuffer(frameUniformBuf, offset: 0, index: 1)
            encoder.drawIndexedPrimitives(
                type: .triangle,
                indexCount: 6,
                indexType: .uint32,
                indexBuffer: idxBuf,
                indexBufferOffset: 0
            )
        }
    }

    /// Draw the holographic info screen (transparent pass, alpha blended).
    func drawHoloScreen(
        encoder: MTLRenderCommandEncoder,
        frameUniformBuf: MTLBuffer,
        pipeline: MTLRenderPipelineState
    ) {
        guard holoVisible,
              let uniformBuf = holoUniformBuffer,
              let idxBuf = holoIndexBuffer,
              let tex = holoTexture,
              let samp = sampler else { return }

        encoder.setRenderPipelineState(pipeline)
        encoder.setVertexBuffer(uniformBuf, offset: 0, index: 0)
        encoder.setVertexBuffer(frameUniformBuf, offset: 0, index: 1)
        encoder.setFragmentBuffer(frameUniformBuf, offset: 0, index: 0)
        encoder.setFragmentTexture(tex, index: 0)
        encoder.setFragmentSamplerState(samp, index: 0)
        encoder.drawIndexedPrimitives(
            type: .triangle,
            indexCount: 6,
            indexType: .uint32,
            indexBuffer: idxBuf,
            indexBufferOffset: 0
        )
    }

    /// Draw thruster particles using the flow particle pipeline (transparent pass).
    func drawThrusterParticles(
        encoder: MTLRenderCommandEncoder,
        frameUniformBuf: MTLBuffer,
        pipeline: MTLRenderPipelineState
    ) {
        guard actualThrusterParticleCount > 0,
              let vtxBuf = thrusterVertexBuffer,
              let idxBuf = thrusterIndexBuffer else { return }

        encoder.setRenderPipelineState(pipeline)
        encoder.setVertexBuffer(vtxBuf, offset: 0, index: 0)
        encoder.setVertexBuffer(frameUniformBuf, offset: 0, index: 1)
        let indexCount = actualThrusterParticleCount * 6
        encoder.drawIndexedPrimitives(
            type: .triangle,
            indexCount: indexCount,
            indexType: .uint32,
            indexBuffer: idxBuf,
            indexBufferOffset: 0
        )
    }

    // MARK: - Matrix Helpers

    private func translationMatrix(_ t: SIMD3<Float>) -> simd_float4x4 {
        simd_float4x4(columns: (
            SIMD4(1, 0, 0, 0),
            SIMD4(0, 1, 0, 0),
            SIMD4(0, 0, 1, 0),
            SIMD4(t.x, t.y, t.z, 1)
        ))
    }

    private func scaleMatrix(_ s: Float) -> simd_float4x4 {
        simd_float4x4(columns: (
            SIMD4(s, 0, 0, 0),
            SIMD4(0, s, 0, 0),
            SIMD4(0, 0, s, 0),
            SIMD4(0, 0, 0, 1)
        ))
    }

    private func rotationX(_ angle: Float) -> simd_float4x4 {
        let c = cos(angle), s = sin(angle)
        return simd_float4x4(columns: (
            SIMD4(1, 0, 0, 0),
            SIMD4(0, c, s, 0),
            SIMD4(0, -s, c, 0),
            SIMD4(0, 0, 0, 1)
        ))
    }

    private func rotationY(_ angle: Float) -> simd_float4x4 {
        let c = cos(angle), s = sin(angle)
        return simd_float4x4(columns: (
            SIMD4(c, 0, -s, 0),
            SIMD4(0, 1, 0, 0),
            SIMD4(s, 0, c, 0),
            SIMD4(0, 0, 0, 1)
        ))
    }

    private func rotationZ(_ angle: Float) -> simd_float4x4 {
        let c = cos(angle), s = sin(angle)
        return simd_float4x4(columns: (
            SIMD4(c, s, 0, 0),
            SIMD4(-s, c, 0, 0),
            SIMD4(0, 0, 1, 0),
            SIMD4(0, 0, 0, 1)
        ))
    }
}
