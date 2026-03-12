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

/// Shared GPU resources for all mascot instances (loaded once, shared across fleet).
@MainActor
struct MascotSharedResources {
    let parts: [MascotSystem.PartMesh]
    let baseColorTexture: MTLTexture?
    let metalRoughTexture: MTLTexture?
    let sampler: MTLSamplerState?
}

/// Loads, animates, and draws the robot mascot companion.
/// The mascot floats near the camera and has 5 independently animated parts:
/// body, left arm, right arm, eyes, and bottom thruster.
@MainActor
final class MascotSystem {

    // MARK: - State Machine

    enum MascotState: Equatable {
        case idle(IdleSubState)
        case patrol(targetId: UUID, targetPos: SIMD3<Float>)
        case hover(nodeId: UUID, timer: Float)
        case conjure(nodeId: UUID, phase: Float)
        case absorb(nodeId: UUID, phase: Float)
        case react(nodeId: UUID, ReactionType)
        case chatting
    }

    enum IdleSubState: Equatable { case awake, drowsy, sleeping }
    enum ReactionType: Equatable { case headTurn, inspect, importanceBoost }

    enum MascotTask: Comparable {
        case create(nodeId: UUID, position: SIMD3<Float>)
        case delete(nodeId: UUID, position: SIMD3<Float>)
        case react(nodeId: UUID, type: ReactionType)
        case patrol
        case idle

        /// Priority ordering: create > delete > react > patrol > idle
        private var priority: Int {
            switch self {
            case .create: return 4
            case .delete: return 3
            case .react: return 2
            case .patrol: return 1
            case .idle: return 0
            }
        }

        static func < (lhs: MascotTask, rhs: MascotTask) -> Bool {
            lhs.priority < rhs.priority
        }
    }

    // Part indices matching the binary file order
    private enum Part: Int, CaseIterable {
        case body = 0, leftArm, rightArm, eye, bottom
    }

    struct PartMesh {
        let vertexBuffer: MTLBuffer
        let indexBuffer: MTLBuffer
        let vertexCount: Int
        let indexCount: Int
        let pivot: SIMD3<Float>
    }

    private let device: MTLDevice
    private(set) var parts: [PartMesh] = []
    private(set) var uniformBuffer: MTLBuffer?
    private(set) var baseColorTexture: MTLTexture?
    private(set) var metalRoughTexture: MTLTexture?
    private(set) var sampler: MTLSamplerState?

    // State machine
    private(set) var currentState: MascotState = .idle(.awake)
    private(set) var taskQueue: [MascotTask] = []  // sorted descending by priority, max depth 8
    private let maxTaskQueueDepth = 8
    private var idleTimer: Float = 0  // time spent in idle state
    private var nextPatrolThreshold: Float = Float.random(in: 12...30)  // set once on idle entry

    /// Maintenance busy mode — visual overlay only, does NOT block state machine.
    private(set) var isBusyMode: Bool = false

    // Project identity
    var projectId: String = ""
    var projectTint: SIMD3<Float> = SIMD3<Float>(0, 0.8, 1.0)  // default cyan

    // Animation state
    private var time: Float = 0
    private(set) var currentPosition: SIMD3<Float> = .zero
    private var initialized = false
    private var currentDynamicScale: Float = 0

    /// Last computed body matrix (for GPU particle spawn generation)
    private(set) var lastBodyMatrix: simd_float4x4 = matrix_identity_float4x4
    /// Last computed dynamic scale
    private(set) var lastDynamicScale: Float = 0

    // Backward-compat computed properties
    var isChatting: Bool { if case .chatting = currentState { return true } else { return false } }
    var isHovering: Bool {
        switch currentState {
        case .hover, .chatting: return true
        default: return false
        }
    }

    /// Whether the arcane circle should be active (hovering OR conjuring).
    private var wantsArcaneCircle: Bool {
        switch currentState {
        case .hover, .chatting, .conjure: return true
        default: return false
        }
    }

    /// Whether the mascot is in conjure state (for eye glow boost).
    private var isConjuring: Bool {
        if case .conjure = currentState { return true } else { return false }
    }

    // Fixed render-space scale — mascot is about 3x node size (nodes are radius 0.04)
    private let mascotScale: Float = 0.06

    // Patrol/hover timing constants
    private var patrolTarget: SIMD3<Float> = .zero
    private var patrolHoverTimer: Float = 0
    private let patrolHoverDuration: Float = 8.0  // total seconds at a node (includes fade-out)
    private let holoDelay: Float = 1.5  // seconds after hover starts before holo appears
    private let holoFadeOutLead: Float = 1.0  // seconds before departure to start fading holo out
    private var patrolSpeed: Float = 0.3  // render-space units/sec

    // Smooth yaw — lerps toward target direction instead of snapping
    private var currentYaw: Float = 0
    private var idleYawDrift: Float = 0  // random wander in idle state

    // Conjure/absorb arm animation angle (radians, 0 = resting, positive = raised)
    private var conjureArmAngle: Float = 0

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

    // Conjure scan rings (horizontal rings sweeping up/down mascot during conjure)
    private var scanRingUniformBuffers: [MTLBuffer] = []  // 3 buffers
    private var scanRingIndexBuffer: MTLBuffer?
    private(set) var scanRingsVisible: Bool = false
    private var scanRingIntensity: Float = 0

    // Conjure orb (energy sphere forming between hands)
    private var conjureOrbUniformBuffer: MTLBuffer?
    private var conjureOrbIndexBuffer: MTLBuffer?
    private(set) var conjureOrbVisible: Bool = false
    private var conjureOrbIntensity: Float = 0
    private var conjureOrbPosition: SIMD3<Float> = .zero
    private var conjureOrbScale: Float = 0
    private var conjureNodeTarget: SIMD3<Float> = .zero  // final node position for orb flight

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
        if chatting {
            currentState = .chatting
            patrolHoverTimer = 0
        } else {
            enterIdle()
        }
    }

    /// Enter busy (maintenance) mode. Visual overlay only — does not block patrol/movement.
    func enterBusyMode() {
        isBusyMode = true
    }

    /// Exit busy (maintenance) mode.
    func exitBusyMode() {
        isBusyMode = false
    }

    /// Enqueue a task. Maintains sorted order (highest priority first) and caps at maxTaskQueueDepth.
    func enqueueTask(_ task: MascotTask) {
        taskQueue.append(task)
        taskQueue.sort(by: >)  // highest priority first
        // Drop lowest-priority tasks if over capacity
        while taskQueue.count > maxTaskQueueDepth {
            taskQueue.removeLast()
        }
    }

    /// Dequeue the next highest-priority task, or nil if empty.
    private func dequeueTask() -> MascotTask? {
        taskQueue.isEmpty ? nil : taskQueue.removeFirst()
    }

    /// Enter idle state with a fresh patrol threshold.
    private func enterIdle(shortPause: Bool = false) {
        currentState = .idle(.awake)
        idleTimer = 0
        // After a patrol cycle, use a short pause (3-8s) before next patrol.
        // Full idle threshold (12-30s) applies only for initial idle or explicit idle tasks.
        nextPatrolThreshold = shortPause ? Float.random(in: 3...8) : Float.random(in: 12...30)
        idleYawDrift = 0
    }

    /// Transition to the next task from the queue, or idle if empty.
    private func transitionToNextTask(positions: [UUID: SIMD3<Float>], scaleFactor: Float) {
        if let task = dequeueTask() {
            switch task {
            case .create(let nodeId, let position):
                currentState = .conjure(nodeId: nodeId, phase: 0)
                patrolTarget = position
            case .delete(let nodeId, let position):
                currentState = .absorb(nodeId: nodeId, phase: 0)
                patrolTarget = position
            case .react(let nodeId, let type):
                currentState = .react(nodeId: nodeId, type)
            case .patrol:
                pickNewPatrolTarget(positions: positions, scaleFactor: scaleFactor)
            case .idle:
                enterIdle()
            }
        } else {
            // Short pause between patrols so mascot doesn't feel frozen
            enterIdle(shortPause: true)
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
        setupConjureBuffers()
    }

    /// Init from shared resources — used by MascotFleet to avoid per-instance mesh duplication.
    init(device: MTLDevice, shared: MascotSharedResources, projectId: String, tint: SIMD3<Float>) {
        self.device = device
        self.parts = shared.parts
        self.baseColorTexture = shared.baseColorTexture
        self.metalRoughTexture = shared.metalRoughTexture
        self.sampler = shared.sampler
        self.projectId = projectId
        self.projectTint = tint
        uniformBuffer = device.makeBuffer(
            length: MemoryLayout<MascotUniforms>.stride,
            options: .storageModeShared
        )
        setupArcaneBuffers()
        setupRingBuffers()
        setupHoloBuffers()
        setupConjureBuffers()
    }

    /// Extract shared resources from this instance (call on the "template" mascot).
    func extractSharedResources() -> MascotSharedResources {
        MascotSharedResources(
            parts: parts,
            baseColorTexture: baseColorTexture,
            metalRoughTexture: metalRoughTexture,
            sampler: sampler
        )
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

    private func setupConjureBuffers() {
        // 3 scan ring uniform buffers (reuse ArcaneCircleUniforms layout)
        for _ in 0..<3 {
            if let buf = device.makeBuffer(
                length: MemoryLayout<ArcaneCircleUniforms>.stride,
                options: .storageModeShared
            ) {
                scanRingUniformBuffers.append(buf)
            }
        }
        var indices: [UInt32] = [0, 2, 1, 1, 2, 3]
        scanRingIndexBuffer = device.makeBuffer(
            bytes: &indices,
            length: indices.count * MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        )

        // Conjure orb uniform buffer
        conjureOrbUniformBuffer = device.makeBuffer(
            length: MemoryLayout<ConjureOrbUniforms>.stride,
            options: .storageModeShared
        )
        conjureOrbIndexBuffer = device.makeBuffer(
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
    func pickNewPatrolTarget(positions: [UUID: SIMD3<Float>], scaleFactor: Float) {
        let scaled = positions.map { (id: $0.key, pos: $0.value * scaleFactor) }
        guard !scaled.isEmpty else {
            enterIdle()
            return
        }
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
        currentState = .patrol(targetId: candidate.id, targetPos: patrolTarget)
    }

    // MARK: - State-Dispatched Update Methods

    private func updateIdle(dt: Float, camera: CameraController, positions: [UUID: SIMD3<Float>], sf: Float) {
        idleTimer += dt

        // Sub-state transitions based on idle duration
        switch currentState {
        case .idle(.awake):
            if idleTimer >= 30 {
                currentState = .idle(.drowsy)
            }
            // Random yaw drift — curious look-around
            idleYawDrift += Float.random(in: -0.1...0.1) * dt
            idleYawDrift = max(-0.3, min(0.3, idleYawDrift))  // clamp drift range
        case .idle(.drowsy):
            if idleTimer >= 60 {
                currentState = .idle(.sleeping)
            }
        case .idle(.sleeping):
            break
        default:
            break
        }

        // Check for queued tasks — any task wakes from sleep
        if !taskQueue.isEmpty {
            transitionToNextTask(positions: positions, scaleFactor: sf)
            return
        }

        // Patrol after stored threshold. Skip for single-node projects (nowhere to go).
        if idleTimer >= nextPatrolThreshold && positions.count > 1 {
            idleTimer = 0
            pickNewPatrolTarget(positions: positions, scaleFactor: sf)
        } else if positions.count == 1, let (_, pos) = positions.first {
            // Single node: idle near the node
            let targetPos = pos * sf + SIMD3<Float>(0, mascotScale * 2.0, 0)
            let dist = simd_length(targetPos - currentPosition)
            if dist > mascotScale * 4.0 {
                // Too far — fly to it
                currentPosition = currentPosition + simd_normalize(targetPos - currentPosition) * min(dt * 2.0, dist)
            }
        }
    }

    private func updatePatrol(dt: Float, targetId: UUID, targetPos: SIMD3<Float>, positions: [UUID: SIMD3<Float>], sf: Float) {
        // Check for higher-priority tasks
        if let first = taskQueue.first {
            switch first {
            case .create, .delete:
                transitionToNextTask(positions: positions, scaleFactor: sf)
                return
            default: break
            }
        }

        let distToTarget = simd_length(patrolTarget - currentPosition)

        if distToTarget < mascotScale * 0.5 {
            // Arrived — start hovering
            patrolHoverTimer = 0
            currentState = .hover(nodeId: targetId, timer: 0)
        } else {
            // Smooth flight with ease-in/ease-out
            let dir = simd_normalize(patrolTarget - currentPosition)
            let approachFactor = min(distToTarget / (mascotScale * 3.0), 1.0)
            let speed = patrolSpeed * approachFactor
            let step = dir * speed * dt
            if simd_length(step) < distToTarget {
                currentPosition += step
            } else {
                currentPosition = patrolTarget
            }
        }
    }

    private func updateHover(dt: Float, nodeId: UUID, timer: Float, positions: [UUID: SIMD3<Float>], sf: Float) {
        // Check for higher-priority tasks
        if let first = taskQueue.first {
            switch first {
            case .create, .delete:
                transitionToNextTask(positions: positions, scaleFactor: sf)
                return
            default: break
            }
        }

        patrolHoverTimer += dt
        if patrolHoverTimer >= patrolHoverDuration {
            patrolHoverTimer = 0
            transitionToNextTask(positions: positions, scaleFactor: sf)
        } else {
            currentState = .hover(nodeId: nodeId, timer: patrolHoverTimer)
        }
    }

    /// Conjure animation: dramatic memory creation sequence.
    /// Phase 0→0.15: fast approach to target position
    /// Phase 0.15→0.30: arms raise to 100°, scan rings activate, arcane circle lights up
    /// Phase 0.30→0.65: orb forms between hands, scanning rings sweep, eye glow intensifies
    /// Phase 0.65→0.80: orb detaches and flies to node position
    /// Phase 0.80→1.0: effects fade, arms lower
    private func updateConjure(dt: Float, nodeId: UUID, phase: Float, positions: [UUID: SIMD3<Float>], sf: Float) {
        let totalDuration: Float = 4.5
        let newPhase = phase + dt / totalDuration

        if newPhase >= 1.0 {
            // Complete — restore normal state
            conjureArmAngle = 0
            scanRingsVisible = false
            scanRingIntensity = 0
            conjureOrbVisible = false
            conjureOrbIntensity = 0
            conjureOrbScale = 0
            transitionToNextTask(positions: positions, scaleFactor: sf)
            return
        }

        // Track the node's simulation position for orb flight target
        if let pos = positions[nodeId] {
            patrolTarget = pos * sf + SIMD3<Float>(0, mascotScale * 2.0, 0)
            conjureNodeTarget = pos * sf
        }

        if newPhase < 0.15 {
            // Phase 1: Fast approach
            let distToTarget = simd_length(patrolTarget - currentPosition)
            if distToTarget > mascotScale * 0.3 {
                let dir = simd_normalize(patrolTarget - currentPosition)
                let speed = patrolSpeed * 2.0
                let step = dir * speed * dt
                if simd_length(step) < distToTarget {
                    currentPosition += step
                } else {
                    currentPosition = patrolTarget
                }
            }
            // Arms begin raising
            conjureArmAngle = (newPhase / 0.15) * 0.5  // 0→~29 degrees
        } else if newPhase < 0.30 {
            // Phase 2: Arms raise to 100°, scan rings activate
            let t = (newPhase - 0.15) / 0.15  // 0→1
            conjureArmAngle = 0.5 + t * 1.25  // ~29°→100°
            scanRingIntensity = t
            scanRingsVisible = true
        } else if newPhase < 0.65 {
            // Phase 3: Orb forms between hands, scan rings sweep
            let t = (newPhase - 0.30) / 0.35  // 0→1
            conjureArmAngle = 1.75  // hold at 100°
            scanRingIntensity = 1.0
            scanRingsVisible = true
            conjureOrbScale = t
            conjureOrbIntensity = 0.5 + t * 0.5
            conjureOrbVisible = true
            // Orb stays between hands
            conjureOrbPosition = computeOrbHandPosition()
        } else if newPhase < 0.80 {
            // Phase 4: Orb detaches and flies to node
            let t = (newPhase - 0.65) / 0.15  // 0→1
            let eased = t * t * (3.0 - 2.0 * t)  // smoothstep easing
            conjureArmAngle = 1.75 * (1.0 - t * 0.3)  // arms begin lowering
            scanRingIntensity = 1.0 - t
            scanRingsVisible = scanRingIntensity > 0.01
            conjureOrbScale = 1.0
            conjureOrbIntensity = 1.0 - t * 0.3
            conjureOrbVisible = true
            // Lerp orb from hands to node position
            let handPos = computeOrbHandPosition()
            conjureOrbPosition = simd_mix(handPos, conjureNodeTarget, SIMD3<Float>(repeating: eased))
        } else {
            // Phase 5: Fade out
            let t = (newPhase - 0.80) / 0.20  // 0→1
            conjureArmAngle = 1.75 * 0.7 * (1.0 - t)  // finish lowering
            scanRingsVisible = false
            scanRingIntensity = 0
            conjureOrbVisible = false
            conjureOrbIntensity = 0
            conjureOrbScale = 0
        }

        currentState = .conjure(nodeId: nodeId, phase: newPhase)
    }

    /// Compute the world-space midpoint between the mascot's raised hands.
    private func computeOrbHandPosition() -> SIMD3<Float> {
        guard parts.count == 5 else { return currentPosition }
        // Hand tips are at the far end of each arm — approximate offset from pivot
        let leftPivot = SIMD3<Float>(-0.68, -0.21, 0.10)
        let rightPivot = SIMD3<Float>(0.67, -0.21, 0.10)
        // When arms are raised, hands move upward. Approximate hand position
        // as pivot + rotated offset along the arm (arm length ~0.5 in model space)
        let armLength: Float = 0.5
        let leftHandLocal = leftPivot + SIMD3<Float>(0, -armLength * cos(conjureArmAngle), armLength * sin(conjureArmAngle))
        let rightHandLocal = rightPivot + SIMD3<Float>(0, -armLength * cos(conjureArmAngle), armLength * sin(conjureArmAngle))
        let midpoint = (leftHandLocal + rightHandLocal) * 0.5

        // Transform to world space via body matrix
        let world4 = lastBodyMatrix * SIMD4<Float>(midpoint.x, midpoint.y, midpoint.z, 1.0)
        return SIMD3<Float>(world4.x, world4.y, world4.z)
    }

    /// Absorb animation: fly to condemned node, node shrinks, mascot absorbs.
    /// Phase 0→0.2: flyTo, 0.2→1.0: perform (node scales 1→0)
    private func updateAbsorb(dt: Float, nodeId: UUID, phase: Float, positions: [UUID: SIMD3<Float>], sf: Float) {
        let totalDuration: Float = 3.0
        let newPhase = phase + dt / totalDuration

        if newPhase >= 1.0 {
            conjureArmAngle = 0
            transitionToNextTask(positions: positions, scaleFactor: sf)
            return
        }

        // Update target position from simulation if available
        if let pos = positions[nodeId] {
            patrolTarget = pos * sf + SIMD3<Float>(0, mascotScale * 2.0, 0)
        }

        if newPhase < 0.2 {
            // FlyTo phase
            let distToTarget = simd_length(patrolTarget - currentPosition)
            if distToTarget > mascotScale * 0.3 {
                let dir = simd_normalize(patrolTarget - currentPosition)
                let speed = patrolSpeed * 1.5
                let step = dir * speed * dt
                if simd_length(step) < distToTarget {
                    currentPosition += step
                } else {
                    currentPosition = patrolTarget
                }
            }
            conjureArmAngle = (newPhase / 0.2) * 1.57  // 0→90 degrees (T-pose)
        } else {
            // Perform phase: arms in T-pose, node shrinks
            conjureArmAngle = 1.57  // 90 degrees (T-pose)
        }

        currentState = .absorb(nodeId: nodeId, phase: newPhase)
    }

    /// Reaction timer for transient reactions (headTurn, importanceBoost).
    private var reactTimer: Float = 0
    private var reactYawTarget: Float?

    private func updateReact(dt: Float, nodeId: UUID, type: ReactionType, positions: [UUID: SIMD3<Float>], sf: Float) {
        reactTimer += dt

        switch type {
        case .headTurn:
            // Yaw toward target node over 0.3s, hold 1s, return. Duration ~2s.
            let duration: Float = 2.0
            if reactTimer >= duration {
                reactTimer = 0
                reactYawTarget = nil
                transitionToNextTask(positions: positions, scaleFactor: sf)
                return
            }
            // Turn toward the target node
            if let pos = positions[nodeId] {
                let targetWorld = pos * sf
                let toTarget = targetWorld - currentPosition
                if simd_length(toTarget) > 0.001 {
                    reactYawTarget = atan2(toTarget.x, toTarget.z)
                }
            }

        case .inspect:
            // If idle/patrolling, fly to node, short hover (3s)
            if reactTimer < 0.1 {
                // On first tick, set up flight toward node
                if let pos = positions[nodeId] {
                    patrolTarget = pos * sf + SIMD3<Float>(0, mascotScale * 2.0, 0)
                    currentTargetId = nodeId
                }
            }
            let distToTarget = simd_length(patrolTarget - currentPosition)
            if distToTarget > mascotScale * 0.5 {
                let dir = simd_normalize(patrolTarget - currentPosition)
                let speed = patrolSpeed * 0.8
                let step = dir * speed * dt
                if simd_length(step) < distToTarget {
                    currentPosition += step
                }
            }
            // Short hover for 3s then done
            if reactTimer >= 3.5 {
                reactTimer = 0
                reactYawTarget = nil
                transitionToNextTask(positions: positions, scaleFactor: sf)
            }

        case .importanceBoost:
            // Arm raise for 0.5s then release. Duration ~1.5s.
            let duration: Float = 1.5
            if reactTimer >= duration {
                reactTimer = 0
                conjureArmAngle = 0
                transitionToNextTask(positions: positions, scaleFactor: sf)
                return
            }
            // Brief arm raise (both arms +45 deg for 0.5s)
            if reactTimer < 0.5 {
                conjureArmAngle = 0.785 * min(reactTimer / 0.3, 1.0)  // 45 degrees
            } else if reactTimer < 1.0 {
                conjureArmAngle = 0.785 * (1.0 - (reactTimer - 0.5) / 0.5)  // lower back
            } else {
                conjureArmAngle = 0
            }
        }
    }

    /// Update mascot position and animation. Call once per frame.
    func update(dt: Float, camera: CameraController, nodePositions: [UUID: SIMD3<Float>], nodeInfo: [UUID: MascotNodeInfo] = [:]) {
        time += dt
        guard parts.count == 5 else { return }

        let sf: Float = camera.scaleFactor
        let dynamicScale: Float = mascotScale

        // ── Initialize on first frame with positions ──
        if !initialized {
            // Wait until we have node positions before initializing
            guard !nodePositions.isEmpty else { return }
            let centroid = nodePositions.values.reduce(SIMD3<Float>.zero, +) / Float(nodePositions.count)
            currentPosition = centroid * sf
            // Immediately start patrolling instead of idling
            pickNewPatrolTarget(positions: nodePositions, scaleFactor: sf)
            initialized = true
        }

        // ── State-dispatched behavior ──
        switch currentState {
        case .idle:
            updateIdle(dt: dt, camera: camera, positions: nodePositions, sf: sf)
        case .patrol(let targetId, let targetPos):
            updatePatrol(dt: dt, targetId: targetId, targetPos: targetPos, positions: nodePositions, sf: sf)
        case .hover(let nodeId, let timer):
            updateHover(dt: dt, nodeId: nodeId, timer: timer, positions: nodePositions, sf: sf)
        case .conjure(let nodeId, let phase):
            updateConjure(dt: dt, nodeId: nodeId, phase: phase, positions: nodePositions, sf: sf)
        case .absorb(let nodeId, let phase):
            updateAbsorb(dt: dt, nodeId: nodeId, phase: phase, positions: nodePositions, sf: sf)
        case .react(let nodeId, let type):
            updateReact(dt: dt, nodeId: nodeId, type: type, positions: nodePositions, sf: sf)
        case .chatting:
            break  // stay at current position
        }

        // Face direction: camera when chatting, travel direction otherwise
        let desiredYaw: Float
        if isChatting {
            let camPos = camera.cameraPosition * sf
            let toCamera = camPos - currentPosition
            if simd_length(toCamera) > 0.001 {
                desiredYaw = atan2(toCamera.x, toCamera.z)
            } else {
                desiredYaw = currentYaw
            }
        } else if case .idle = currentState {
            // Idle: gentle random yaw drift
            desiredYaw = currentYaw + idleYawDrift
        } else {
            let toTarget = patrolTarget - currentPosition
            if simd_length(toTarget) > 0.001 {
                desiredYaw = atan2(toTarget.x, toTarget.z)
            } else {
                desiredYaw = currentYaw
            }
        }
        var yawDiff = desiredYaw - currentYaw
        if yawDiff > .pi { yawDiff -= 2 * .pi }
        if yawDiff < -.pi { yawDiff += 2 * .pi }
        let yawSpeed: Float = isHovering ? 1.5 : 3.0
        currentYaw += yawDiff * min(dt * yawSpeed, 1.0)

        // Build per-part model matrices
        guard let uniformBuf = uniformBuffer else { return }
        let ptr = uniformBuf.contents().bindMemory(to: MascotUniforms.self, capacity: 1)

        // Idle animation: bob + lean with state-dependent intensity
        let idleSubState: IdleSubState? = {
            if case .idle(let s) = currentState { return s } else { return nil }
        }()
        let isBusy = isBusyMode
        let bobSpeed: Float = {
            if isBusy { return 6.0 }  // 3x normal during maintenance
            switch idleSubState {
            case .drowsy: return 1.0
            case .sleeping: return 0.6
            default: return 2.0
            }
        }()
        let bobAmplitude: Float = {
            if isBusy { return 0.12 }
            switch idleSubState {
            case .sleeping: return 0.08
            default: return 0.15
            }
        }()
        let bobY: Float = sin(time * bobSpeed) * dynamicScale * bobAmplitude
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
        lastBodyMatrix = bodyMatrix
        lastDynamicScale = dynamicScale

        let zp = SIMD3<Float>(0, 0, 0)

        // Arm swing animation — pendulum rotation around shoulder pivots.
        // skinWeight in the mesh data (0 at shoulder → 1 at hand) blends
        // between parentMatrix (body) and modelMatrix (arm) in the shader,
        // so vertices near the shoulder stay with the body and the hand
        // follows the arm rotation fully.
        // Dampen swing during flight to prevent stretching from rapid yaw changes.
        let swingAmplitude: Float = {
            switch idleSubState {
            case .sleeping: return 0.05  // arms droop in sleep
            case .drowsy: return 0.2
            default: return isHovering ? 0.4 : 0.15
            }
        }()
        // Override arm angle for conjure/absorb animations (arms raise)
        let leftSwing: Float
        let rightSwing: Float
        if conjureArmAngle > 0.01 {
            // Both arms raise symmetrically during conjure/absorb
            leftSwing = -conjureArmAngle   // negative = raise upward
            rightSwing = -conjureArmAngle
        } else if isBusy {
            // Busy mode: arms at working angle (45 degrees outward)
            let busyAngle: Float = -.pi / 4.0
            leftSwing = busyAngle + sin(time * 4.0) * 0.1   // subtle working motion
            rightSwing = busyAngle + sin(time * 4.0 + .pi) * 0.1
        } else {
            leftSwing = sin(time * 2.5) * swingAmplitude
            rightSwing = sin(time * 2.5 + .pi) * swingAmplitude  // opposite phase
        }

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

        // Eye: cyan emissive pulse (dims in drowsy/sleeping idle states)
        let eyeMax: Float = {
            if isConjuring { return 1.8 }  // intense glow during conjure
            if isBusy { return 1.2 }  // brighter during maintenance
            switch idleSubState {
            case .drowsy: return 0.5   // 50% brightness
            case .sleeping: return 0.1  // 10% brightness
            default: return 1.0
            }
        }()
        let eyeFreq: Float = isConjuring ? 8.0 : (isBusy ? 6.0 : 3.0)
        let eyePulse: Float = eyeMax * (0.6 + 0.4 * sin(time * eyeFreq))
        ptr.pointee.parts.3 = MascotPartUniforms(
            modelMatrix: bodyMatrix, parentMatrix: bodyMatrix,
            emissive: SIMD4<Float>(projectTint.x, projectTint.y, projectTint.z, eyePulse), blendPivot: zp, blendRadius: 0
        )

        // Bottom: blue thruster glow
        let bottomPulse: Float = 0.4 + 0.3 * sin(time * 4.0)
        ptr.pointee.parts.4 = MascotPartUniforms(
            modelMatrix: bodyMatrix, parentMatrix: bodyMatrix,
            emissive: SIMD4<Float>(0.2, 0.4, 1.0, bottomPulse), blendPivot: zp, blendRadius: 0
        )

        // Thruster particles are now managed by MascotFleet's GPU particle system.
        // Spawn generation happens in MascotFleet.collectAndDispatchParticles().

        // ── Arcane circle ──
        updateArcaneCircle(dt: dt, bodyMatrix: bodyMatrix, bodyRot: bodyRot, dynamicScale: dynamicScale)

        // ── Node inspection rings ──
        updateNodeRings(dt: dt, nodePositions: nodePositions, scaleFactor: sf, dynamicScale: dynamicScale)

        // ── Conjure scan rings ──
        updateScanRings(dt: dt, bodyMatrix: bodyMatrix, dynamicScale: dynamicScale)

        // ── Conjure orb ──
        updateConjureOrb(dt: dt, camera: camera, dynamicScale: dynamicScale)

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

    // MARK: - GPU Animation State Packing

    /// Pack scalar animation state for the GPU matrix compute kernel.
    /// CPU computes state machine decisions and scalar params; GPU does the matrix math.
    func packAnimState() -> MascotAnimState {
        let idleSubState: IdleSubState? = {
            if case .idle(let s) = currentState { return s } else { return nil }
        }()
        let isBusy = isBusyMode

        let bobSpeed: Float = {
            if isBusy { return 6.0 }
            switch idleSubState {
            case .drowsy: return 1.0
            case .sleeping: return 0.6
            default: return 2.0
            }
        }()
        let bobAmplitude: Float = {
            if isBusy { return 0.12 }
            switch idleSubState {
            case .sleeping: return 0.08
            default: return 0.15
            }
        }()

        let flyLean: Float = isHovering ? 0.0 : 0.2

        // Arm swing
        let swingAmplitude: Float = {
            switch idleSubState {
            case .sleeping: return 0.05
            case .drowsy: return 0.2
            default: return isHovering ? 0.4 : 0.15
            }
        }()
        let leftSwing: Float
        let rightSwing: Float
        if conjureArmAngle > 0.01 {
            leftSwing = -conjureArmAngle
            rightSwing = -conjureArmAngle
        } else if isBusy {
            let busyAngle: Float = -.pi / 4.0
            leftSwing = busyAngle + sin(time * 4.0) * 0.1
            rightSwing = busyAngle + sin(time * 4.0 + .pi) * 0.1
        } else {
            leftSwing = sin(time * 2.5) * swingAmplitude
            rightSwing = sin(time * 2.5 + .pi) * swingAmplitude
        }

        let eyeMax: Float = {
            if isConjuring { return 1.8 }
            if isBusy { return 1.2 }
            switch idleSubState {
            case .drowsy: return 0.5
            case .sleeping: return 0.1
            default: return 1.0
            }
        }()
        let eyeFreq: Float = isConjuring ? 8.0 : (isBusy ? 6.0 : 3.0)

        return MascotAnimState(
            currentPosition: currentPosition,
            currentYaw: currentYaw,
            time: time,
            dynamicScale: lastDynamicScale,
            bobSpeed: bobSpeed,
            bobAmplitude: bobAmplitude,
            flyLean: flyLean,
            leftSwing: leftSwing,
            rightSwing: rightSwing,
            eyePulseMax: eyeMax,
            eyeFreq: eyeFreq,
            bottomPulse: 0,  // computed on GPU
            projectTint: projectTint,
            leftPivot: SIMD3<Float>(-0.68, -0.21, 0.10),
            rightPivot: SIMD3<Float>(0.67, -0.21, 0.10),
            _pad0: 0, _pad1: 0
        )
    }

    // MARK: - GPU Thruster Spawn Generation

    /// Generate spawn requests for the GPU particle system (replaces CPU particle sim).
    /// Returns an array of spawn requests to be submitted to the fleet's GPU particle buffer.
    func generateThrusterSpawns(dt: Float, bodyMatrix: simd_float4x4, dynamicScale: Float) -> [ThrusterSpawnRequest] {
        // Bottom of the mascot in model space → world space
        let bottomLocal = SIMD4<Float>(0, -0.85, 0, 1)
        let bottomWorld4: SIMD4<Float> = bodyMatrix * bottomLocal
        let emitPos = SIMD3<Float>(bottomWorld4.x, bottomWorld4.y, bottomWorld4.z)

        var spawns: [ThrusterSpawnRequest] = []
        let spawnRate: Float = 1.0 / 120.0  // 120 particles/sec
        thrusterSpawnTimer += dt
        while thrusterSpawnTimer >= spawnRate {
            thrusterSpawnTimer -= spawnRate

            let spreadX: Float = Float.random(in: -0.3...0.3) * dynamicScale
            let spreadZ: Float = Float.random(in: -0.3...0.3) * dynamicScale
            let speedY: Float = -Float.random(in: 0.4...0.9) * dynamicScale
            let vel = SIMD3<Float>(spreadX, speedY, spreadZ)

            let life: Float = Float.random(in: 0.6...1.2)
            let size: Float = dynamicScale * Float.random(in: 0.10...0.22)

            let r: Float = Float.random(in: 0.1...0.35)
            let g: Float = Float.random(in: 0.4...0.75)
            let b: Float = Float.random(in: 0.8...1.0)

            spawns.append(ThrusterSpawnRequest(
                position: emitPos, velocity: vel,
                maxLife: life, size: size,
                color: SIMD3<Float>(r, g, b), _pad0: 0
            ))
        }
        return spawns
    }

    // MARK: - Arcane Circle

    private func updateArcaneCircle(dt: Float, bodyMatrix: simd_float4x4, bodyRot: simd_float4x4, dynamicScale: Float) {
        // Fade in when hovering or conjuring, fade out when flying
        let target: Float = wantsArcaneCircle ? 1.0 : 0.0
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
            _pad0: 0,
            tintColor: projectTint,
            _pad1: 0
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
                _pad0: 0,
                tintColor: projectTint,
                _pad1: 0
            )
        }
    }

    // MARK: - Conjure Scan Rings

    private func updateScanRings(dt: Float, bodyMatrix: simd_float4x4, dynamicScale: Float) {
        guard scanRingsVisible, scanRingUniformBuffers.count == 3 else {
            scanRingsVisible = false
            return
        }

        let bodyPos = SIMD3<Float>(bodyMatrix.columns.3.x, bodyMatrix.columns.3.y, bodyMatrix.columns.3.z)
        let ringSize = dynamicScale * 1.2  // slightly wider than mascot body

        // 3 horizontal rings at different Y offsets, sweeping up and down
        let sweepSpeed: Float = 3.0
        let sweepRange = dynamicScale * 1.5  // vertical sweep range
        let offsets: [Float] = [0.0, 0.33, 0.66]  // phase offsets for stagger

        for (i, offset) in offsets.enumerated() {
            // Ping-pong sweep: triangle wave from -1 to +1
            let phase = time * sweepSpeed / sweepRange + offset
            let rawT = phase - floor(phase)  // fract equivalent
            let sweep: Float = rawT < 0.5 ? rawT * 4.0 - 1.0 : 3.0 - rawT * 4.0
            let yOffset = sweep * sweepRange * 0.5

            let center = bodyPos + SIMD3<Float>(0, yOffset, 0)

            // Horizontal ring: right = world X, up = world Z (ring lies flat)
            let yaw = atan2(bodyMatrix.columns.2.x, bodyMatrix.columns.2.z)
            let right = SIMD3<Float>(cos(yaw), 0, sin(yaw))
            let up = SIMD3<Float>(-sin(yaw), 0, cos(yaw))

            let ptr = scanRingUniformBuffers[i].contents().bindMemory(to: ArcaneCircleUniforms.self, capacity: 1)
            ptr.pointee = ArcaneCircleUniforms(
                center: center,
                size: ringSize,
                right: right,
                opacity: scanRingIntensity * (0.7 + 0.3 * (1.0 - abs(sweep))),  // brighter near center
                up: up,
                _pad0: 0,
                tintColor: projectTint,
                _pad1: 0
            )
        }
    }

    // MARK: - Conjure Orb

    private func updateConjureOrb(dt: Float, camera: CameraController, dynamicScale: Float) {
        guard conjureOrbVisible, let buf = conjureOrbUniformBuffer else {
            conjureOrbVisible = false
            return
        }

        // Camera-facing billboard (orb always faces viewer)
        let vm = camera.viewMatrix()
        let right = SIMD3<Float>(vm.columns.0.x, vm.columns.1.x, vm.columns.2.x)
        let up = SIMD3<Float>(vm.columns.0.y, vm.columns.1.y, vm.columns.2.y)

        let ptr = buf.contents().bindMemory(to: ConjureOrbUniforms.self, capacity: 1)
        ptr.pointee = ConjureOrbUniforms(
            center: conjureOrbPosition,
            size: dynamicScale * 1.0,
            right: right,
            opacity: conjureOrbIntensity,
            up: up,
            orbScale: conjureOrbScale,
            tintColor: projectTint,
            glowIntensity: conjureOrbIntensity
        )
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

    /// Render info card texture using CoreText directly. Called only when target node changes.
    /// Uses CTLine + CGContext — no NSGraphicsContext/NSString.draw to avoid AppKit lock
    /// contention during Metal draw callbacks (same fix applied to MetalSceneManager label atlas).
    private func renderHoloTexture(info: MascotNodeInfo) {
        let texW = 512
        let texH = 400
        let scale = 2  // retina
        let allocW = texW * scale
        let allocH = texH * scale
        let bytesPerRow = allocW * 4

        // Draw directly into a CGContext — bypasses NSImage entirely to avoid
        // stale bitmap backing memory that NSImage.lockFocusFlipped can recycle.
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil,
            width: allocW, height: allocH,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
        ) else { return }

        // Guarantee zeroed buffer
        ctx.clear(CGRect(x: 0, y: 0, width: allocW, height: allocH))

        // Scale for retina — CoreText uses bottom-left origin natively,
        // we convert top-left Y to baseline Y per draw call.
        ctx.scaleBy(x: CGFloat(scale), y: CGFloat(scale))

        // Create fonts via NSFont.monospacedSystemFont (resolves SF Mono correctly),
        // then cast to CTFont (immutable, thread-safe) for CoreText drawing.
        let topicFont: CTFont = NSFont.monospacedSystemFont(ofSize: 18, weight: .bold) as CTFont
        let metaFont: CTFont = NSFont.monospacedSystemFont(ofSize: 10, weight: .regular) as CTFont
        let sepFont: CTFont = NSFont.monospacedSystemFont(ofSize: 8, weight: .regular) as CTFont
        let contentFont: CTFont = NSFont.monospacedSystemFont(ofSize: 10, weight: .regular) as CTFont

        // Colors (CGColor for CoreText — no NSColor)
        let topicColor = CGColor(red: 0.3, green: 0.9, blue: 1.0, alpha: 1.0)
        let metaColor = CGColor(red: 0.45, green: 0.6, blue: 0.7, alpha: 0.7)
        let sepColor = CGColor(red: 0.3, green: 0.5, blue: 0.6, alpha: 0.4)
        let contentColor = CGColor(red: 0.65, green: 0.8, blue: 0.9, alpha: 0.85)

        /// Create a CTLine and measure its typographic bounds
        func makeLine(_ text: String, font: CTFont, color: CGColor) -> (line: CTLine, ascent: CGFloat) {
            let attrs: [CFString: Any] = [
                kCTFontAttributeName: font,
                kCTForegroundColorAttributeName: color
            ]
            let attrStr = CFAttributedStringCreate(nil, text as CFString, attrs as CFDictionary)!
            let line = CTLineCreateWithAttributedString(attrStr)
            var ascent: CGFloat = 0, descent: CGFloat = 0, leading: CGFloat = 0
            CTLineGetTypographicBounds(line, &ascent, &descent, &leading)
            return (line, ascent)
        }

        /// Draw a CTLine at top-left-origin (x, topY) — converts to CoreText's bottom-left baseline
        func drawLine(_ line: CTLine, ascent: CGFloat, x: CGFloat, topY: CGFloat) {
            ctx.textPosition = CGPoint(x: x, y: CGFloat(texH) - topY - ascent)
            CTLineDraw(line, ctx)
        }

        let margin: CGFloat = 32
        var y: CGFloat = 28

        // ── Header: Topic / Project ──
        let topicStr = "\(info.topic.prefix(24)) / \(info.project.prefix(16))"
        let topic = makeLine(topicStr, font: topicFont, color: topicColor)
        drawLine(topic.line, ascent: topic.ascent, x: margin, topY: y)
        y += 26

        // ── Meta: dates ──
        let metaStr = "created \(Self.holoDateFormatter.string(from: info.createdAt))  |  accessed \(Self.holoDateFormatter.string(from: info.lastAccessedAt))"
        let meta = makeLine(metaStr, font: metaFont, color: metaColor)
        drawLine(meta.line, ascent: meta.ascent, x: margin, topY: y)
        y += 16

        // ── Separator ──
        let sep = String(repeating: "\u{2500}", count: 46)
        let sepLine = makeLine(sep, font: sepFont, color: sepColor)
        drawLine(sepLine.line, ascent: sepLine.ascent, x: margin, topY: y)
        y += 12

        // ── Memory content: fill remaining space ──
        let lineHeight: CGFloat = 14
        let usableWidth = CGFloat(texW) - margin * 2
        let contentCharsPerLine = Int(usableWidth / 6)  // ~6pt per monospace char at size 10
        let maxLines = Int((CGFloat(texH) - y - 20) / lineHeight)
        // Collapse newlines to spaces for word-wrapping
        let content = info.content
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\r", with: " ")
        // Word-wrap into lines
        var lines: [String] = []
        var currentLine = ""
        for word in content.split(separator: " ", omittingEmptySubsequences: true) {
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
            let ct = makeLine(line, font: contentFont, color: contentColor)
            drawLine(ct.line, ascent: ct.ascent, x: margin, topY: y)
            y += lineHeight
        }

        // Create MTLTexture directly from CGContext pixel data
        guard let data = ctx.data else { return }

        let texDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: allocW, height: allocH,
            mipmapped: false
        )
        texDesc.usage = .shaderRead
        texDesc.storageMode = .shared

        guard let texture = device.makeTexture(descriptor: texDesc) else { return }
        texture.replace(
            region: MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                              size: MTLSize(width: allocW, height: allocH, depth: 1)),
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

    /// Draw the 3 conjure scan rings (transparent pass, additive blending).
    func drawScanRings(
        encoder: MTLRenderCommandEncoder,
        frameUniformBuf: MTLBuffer,
        pipeline: MTLRenderPipelineState
    ) {
        guard scanRingsVisible,
              scanRingUniformBuffers.count == 3,
              let idxBuf = scanRingIndexBuffer else { return }

        encoder.setRenderPipelineState(pipeline)
        encoder.setFragmentBuffer(frameUniformBuf, offset: 0, index: 0)

        for i in 0..<3 {
            encoder.setVertexBuffer(scanRingUniformBuffers[i], offset: 0, index: 0)
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

    /// Draw the conjure orb (transparent pass, additive blending).
    func drawConjureOrb(
        encoder: MTLRenderCommandEncoder,
        frameUniformBuf: MTLBuffer,
        pipeline: MTLRenderPipelineState
    ) {
        guard conjureOrbVisible,
              let uniformBuf = conjureOrbUniformBuffer,
              let idxBuf = conjureOrbIndexBuffer else { return }

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
