import SwiftUI
import RealityKit
import GameController
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
    private var nodeEntities: [Int64: ModelEntity] = [:]
    private var edgeContainer = Entity()
    private var edgeEntities: [EdgeKey: ModelEntity] = [:]
    private var edgeLastOpacity: [EdgeKey: Float] = [:]
    private var nodeLastOpacity: [Int64: Float] = [:]

    // Camera state — inputs write to target* values, updateCamera() lerps toward them.
    // This smooths out irregular input event timing for silky camera motion.

    // Target state (written by gestures/events)
    var targetAzimuth: Float = 0
    var targetElevation: Float = 0.3
    var targetCameraPos: SIMD3<Float> = .zero  // the look-at point

    // Smoothed state (lerped each frame, used for rendering)
    private(set) var azimuth: Float = 0
    private(set) var elevation: Float = 0.3
    private(set) var cameraTarget: SIMD3<Float> = .zero

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

    // Material cache — PBR for nodes (lit, 3D shading), unlit for edges
    private var materialCache: [String: PhysicallyBasedMaterial] = [:]
    private var edgeMaterialCache: [String: UnlitMaterial] = [:]
    private var edgeMaterialConnected: UnlitMaterial?
    private var edgeMaterialSemantic: UnlitMaterial?
    var animationTime: Float = 0

    // Shared meshes — created once, reused for all entities
    private let nodeMesh = MeshResource.generateSphere(radius: 1.0)
    private let edgeMesh = MeshResource.generateCylinder(height: 1.0, radius: 1.0)

    // Nebula particle emitter container
    private var nebulaContainer = Entity()
    /// Soft gaussian circle texture for nebula particles — generated once, reused.
    private var softParticleTexture: TextureResource?

    private let scaleFactor: Float = 1.0 / 200.0
    private let nodeRadius: Float = 0.04
    private let edgeRadius: Float = 0.002

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
        if abs(newEl - elevation) > threshold { elevation = newEl }
        if simd_length(newTarget - cameraTarget) > threshold { cameraTarget = newTarget }
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

    private func nodeMaterial(for project: String, colorMap: [String: Color]) -> PhysicallyBasedMaterial {
        if let cached = materialCache[project] { return cached }
        let color = colorMap[project] ?? .gray
        let resolved = NSColor(color)
        var mat = PhysicallyBasedMaterial()
        mat.baseColor = .init(tint: resolved.withAlphaComponent(0.95))
        mat.roughness = .init(floatLiteral: 0.15)    // very glossy — sharp specular highlights
        mat.metallic = .init(floatLiteral: 0.05)     // mostly dielectric, natural specular
        mat.emissiveColor = .init(color: resolved.withAlphaComponent(0.1))
        mat.emissiveIntensity = 0.15  // subtle self-glow so shadow side isn't pure black
        materialCache[project] = mat
        return mat
    }

    /// Edge materials use full alpha — opacity is controlled entirely via OpacityComponent
    /// so the pulse animation can swing through a wide visible range.
    private func getEdgeMaterial(connected: Bool, semantic: Bool,
                                project: String? = nil,
                                colorMap: [String: Color] = [:]) -> UnlitMaterial {
        if connected {
            if edgeMaterialConnected == nil {
                var mat = UnlitMaterial()
                mat.color = .init(tint: .white)
                edgeMaterialConnected = mat
            }
            return edgeMaterialConnected!
        }
        // Project-colored edge (both force and semantic modes)
        if let project, let swiftColor = colorMap[project] {
            let key = semantic ? "sem_\(project)" : project
            if let cached = edgeMaterialCache[key] { return cached }
            let nsColor = NSColor(swiftColor).usingColorSpace(.sRGB) ?? NSColor(swiftColor)
            // Brighten the color so it reads well at low opacity
            var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
            nsColor.getRed(&r, green: &g, blue: &b, alpha: &a)
            let brightened = NSColor(
                red: min(1.0, r * 1.4 + 0.15),
                green: min(1.0, g * 1.4 + 0.15),
                blue: min(1.0, b * 1.4 + 0.15),
                alpha: 1.0
            )
            var mat = UnlitMaterial()
            mat.color = .init(tint: brightened)
            edgeMaterialCache[key] = mat
            return mat
        }
        var mat = UnlitMaterial()
        mat.color = .init(tint: NSColor(white: 0.6, alpha: 1.0))
        return mat
    }

    func updateNodes(positions: [Int64: SIMD3<Float>],
                     nodes: [NodeData], hubs: Set<Int64>,
                     colorMap: [String: Color],
                     selectedNode: Int64?,
                     glowingNodes: [Int64: Date],
                     newNodes: [Int64: Date],
                     dyingNodes: [Int64: DyingNode]) {
        let nodeById = Dictionary(nodes.map { ($0.id, $0) }, uniquingKeysWith: { _, last in last })
        let camPos = cameraPosition  // scaled coordinates
        let now = Date()

        // Keep entities for visible nodes + active dying nodes
        let visibleIds = Set(nodeById.keys)
        let dyingIds = Set(dyingNodes.keys)
        let keepIds = visibleIds.union(dyingIds)
        for (id, entity) in nodeEntities where !keepIds.contains(id) {
            entity.removeFromParent()
            nodeEntities.removeValue(forKey: id)
            nodeLastOpacity.removeValue(forKey: id)
        }

        // Fog parameters (in scaled coordinates — world values / 200)
        let fogNear: Float = 100
        let fogFar: Float = 1200
        let fogMinOpacity: Float = 0.08


        // Add/update live nodes
        for (id, pos) in positions {
            let worldPos = pos * scaleFactor
            guard let nodeData = nodeById[id] else { continue }

            let isHub = hubs.contains(id)
            let importance = max(1, nodeData.importance)
            let baseRadius: Float = isHub ? nodeRadius * 1.6 : nodeRadius
            var importanceRadius = baseRadius * (1.0 + Float(importance - 1) * 0.08)

            // Recall glow intensity (slow fade in → hold → fade out)
            let ri: Float = {
                guard let glowStart = glowingNodes[id], id != selectedNode else { return 0 }
                let elapsed = Float(now.timeIntervalSince(glowStart))
                let fadeIn: Float = 1.0, hold: Float = 1.5, fadeOut: Float = 2.0
                if elapsed < fadeIn {
                    // Smooth ease-in: cubic for gentle build-up
                    let t = elapsed / fadeIn; return t * t * t
                }
                else if elapsed < fadeIn + hold { return 1.0 }
                else if elapsed < fadeIn + hold + fadeOut {
                    let t = 1.0 - (elapsed - fadeIn - hold) / fadeOut; return t * t
                }
                return 0
            }()

            // Arrival (new node) intensity
            let ai: Float = {
                guard let arrivalTime = newNodes[id] else { return 0 }
                let elapsed = Float(now.timeIntervalSince(arrivalTime))
                let fadeIn: Float = 0.8, hold: Float = 2.0, fadeOut: Float = 3.0
                if elapsed < fadeIn {
                    let t = elapsed / fadeIn; return t * t * t
                }
                else if elapsed < fadeIn + hold { return 1.0 }
                else if elapsed < fadeIn + hold + fadeOut {
                    let t = 1.0 - (elapsed - fadeIn - hold) / fadeOut; return t * t
                }
                return 0
            }()

            // Depth fog (shared by node and point light effects)
            let dist = simd_length(pos - camPos)
            let fogT = max(0, min(1, (dist - fogNear) / (fogFar - fogNear)))
            let baseFogOpacity = id == selectedNode ? 1.0 : max(fogMinOpacity, 1.0 - fogT * 0.92)
            let depthFade = max(Float(0.0), 1.0 - fogT * 0.95)  // 1.0 near → 0.05 far

            // Search spotlight dimming
            let searchDimmed = renderIsSearchActive && !renderSearchMatchIds.contains(id)
            let searchMatched = renderIsSearchActive && renderSearchMatchIds.contains(id) && id != selectedNode
            let fogOpacity = searchDimmed ? min(baseFogOpacity, 0.12) : baseFogOpacity

            // Pulse scale for glow/arrival/search effects (more dramatic in 3D)
            if ri > 0 {
                importanceRadius *= 1.0 + sin(animationTime * 4.0) * 0.2 * ri
            } else if ai > 0 {
                importanceRadius *= 1.0 + sin(animationTime * 3.0) * 0.2 * ai
            } else if searchMatched {
                importanceRadius *= 1 + sin(animationTime * 4) * 0.12
            }

            // Choose material based on state
            let material: PhysicallyBasedMaterial
            if id == selectedNode {
                var mat = PhysicallyBasedMaterial()
                mat.baseColor = .init(tint: .white)
                mat.roughness = .init(floatLiteral: 0.15)
                mat.emissiveColor = .init(color: .white)
                mat.emissiveIntensity = 1.0
                material = mat
            } else if searchMatched {
                // Search match: cyan emissive glow
                var mat = nodeMaterial(for: nodeData.project, colorMap: colorMap)
                mat.emissiveColor = .init(color: NSColor(red: 0.0, green: 0.9, blue: 1.0, alpha: 1))
                mat.emissiveIntensity = 4.0 * (1 + sin(animationTime * 4) * 0.3) * depthFade
                material = mat
            } else if ri > 0 {
                // Recalled: intense blue-white emissive, depth-attenuated
                var mat = nodeMaterial(for: nodeData.project, colorMap: colorMap)
                mat.emissiveColor = .init(color: NSColor(red: 0.7, green: 0.9, blue: 1.0, alpha: 1))
                let emPulse = 1.0 + sin(animationTime * 5.0) * 0.3
                mat.emissiveIntensity = 8.0 * ri * emPulse * depthFade
                material = mat
            } else if ai > 0 {
                // New arrival: intense golden emissive, depth-attenuated
                var mat = nodeMaterial(for: nodeData.project, colorMap: colorMap)
                mat.emissiveColor = .init(color: NSColor(red: 1.0, green: 0.85, blue: 0.4, alpha: 1))
                let emPulse = 1.0 + sin(animationTime * 3.5) * 0.3
                mat.emissiveIntensity = 8.0 * ai * emPulse * depthFade
                material = mat
            } else {
                material = nodeMaterial(for: nodeData.project, colorMap: colorMap)
            }

            if let entity = nodeEntities[id] {
                entity.position = worldPos
                entity.scale = SIMD3<Float>(repeating: importanceRadius)
                if abs(fogOpacity - (nodeLastOpacity[id] ?? -1)) > 0.02 {
                    entity.components.set(OpacityComponent(opacity: fogOpacity))
                    nodeLastOpacity[id] = fogOpacity
                }
                entity.model?.materials = [material]
            } else {
                let entity = ModelEntity(mesh: nodeMesh, materials: [material])
                entity.position = worldPos
                entity.scale = SIMD3<Float>(repeating: importanceRadius)
                entity.name = "node_\(id)"
                entity.components.set(OpacityComponent(opacity: fogOpacity))
                nodeLastOpacity[id] = fogOpacity
                entity.components.set(
                    CollisionComponent(shapes: [.generateSphere(radius: 1.0)])
                )
                rootEntity.addChild(entity)
                nodeEntities[id] = entity
            }

            // --- Pulsing point light (depth-attenuated, no geometry) ---
            // High intensity + large attenuation radius so light visibly washes
            // over neighboring nodes. Nodes are ~0.04 units, spaced ~0.1–0.3 apart
            // in scaled coords, so attenuationRadius needs to be 0.5+ to reach neighbors.
            if let entity = nodeEntities[id] {
                if ri > 0 {
                    let lightPulse = 1.0 + sin(animationTime * 3.0) * 0.4
                    entity.components.set(PointLightComponent(
                        color: NSColor(red: 0.7, green: 0.9, blue: 1.0, alpha: 1),
                        intensity: 20000 * ri * depthFade * lightPulse,
                        attenuationRadius: 2.0 * depthFade + 0.1
                    ))
                } else if ai > 0 {
                    let lightPulse = 1.0 + sin(animationTime * 2.5) * 0.4
                    entity.components.set(PointLightComponent(
                        color: NSColor(red: 1.0, green: 0.8, blue: 0.3, alpha: 1),
                        intensity: 20000 * ai * depthFade * lightPulse,
                        attenuationRadius: 2.0 * depthFade + 0.1
                    ))
                } else if searchMatched {
                    let lightPulse = 1.0 + sin(animationTime * 4.0) * 0.3
                    entity.components.set(PointLightComponent(
                        color: NSColor(red: 0.0, green: 0.9, blue: 1.0, alpha: 1),
                        intensity: 12000 * depthFade * lightPulse,
                        attenuationRadius: 1.5 * depthFade + 0.1
                    ))
                } else {
                    entity.components.remove(PointLightComponent.self)
                }
            }
        }

        // Dying nodes: red flash → dark fade → remove
        for (id, dying) in dyingNodes {
            if positions[id] != nil { continue }  // still live, handled above
            guard let entity = nodeEntities[id] else { continue }

            let elapsed = Float(now.timeIntervalSince(dying.startTime))
            let flashIn: Float = 0.3, hold: Float = 1.2, fadeOut: Float = 1.5
            let total = flashIn + hold + fadeOut
            guard elapsed < total else {
                entity.removeFromParent()
                nodeEntities.removeValue(forKey: id)
                continue
            }

            let di: Float
            if elapsed < flashIn {
                let t = elapsed / flashIn; di = t * t
            } else if elapsed < flashIn + hold {
                di = 1.0
            } else {
                let t = 1.0 - (elapsed - flashIn - hold) / fadeOut; di = t * t
            }

            // Red material that darkens over time
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

            // Red point light during death — bright enough to illuminate neighbors
            entity.components.set(PointLightComponent(
                color: NSColor(red: 1.0, green: 0.15, blue: 0.05, alpha: 1),
                intensity: 15000 * di,
                attenuationRadius: 1.5
            ))

            // Shrink during fade-out phase
            let shrink: Float = elapsed > flashIn + hold ? 1.0 - (1.0 - di) * 0.5 : 1.0
            let baseR: Float = dying.isHub ? nodeRadius * 1.6 : nodeRadius
            let impR = baseR * (1.0 + Float(max(1, dying.importance) - 1) * 0.08)
            entity.scale = SIMD3<Float>(repeating: impR * shrink)
            entity.components.set(OpacityComponent(opacity: di))
        }
    }

    /// Create or update a dual pulse sphere pair for a node effect.
    /// Two rings offset by half-period create a seamless loop — one is always
    /// mid-expansion when the other resets, eliminating the visible pop.
    ///
    /// `period`: seconds per full cycle.  `phase`: 0→1 progress through one cycle.
    /// Caller passes the raw `animationTime`; this method computes both ring phases.
    /// Repositions existing edge entities with pulse animation (no material/topology changes).
    func repositionEdgesAnimated(positions: [Int64: SIMD3<Float>], edges: [EdgeData],
                                 selectedNode: Int64?, positionsChanged: Bool) {
        let camPos = cameraPosition  // scaled coordinates (= unscaled, since camPos comes from cameraPosition which uses unscaled coords)
        let fogFar: Float = 1200
        let isSemanticMode = renderLayoutMode == .embedding

        for edge in edges {
            let key = EdgeKey(source: edge.sourceId, target: edge.targetId)
            guard let entity = edgeEntities[key],
                  let from = positions[edge.sourceId],
                  let to = positions[edge.targetId] else { continue }

            let connected = edge.sourceId == selectedNode || edge.targetId == selectedNode

            // Cull edges entirely beyond fog distance (both endpoints far away and not connected)
            if !connected {
                let distFrom = simd_length(from - camPos)
                let distTo = simd_length(to - camPos)
                if distFrom > fogFar && distTo > fogFar {
                    // Hide it and skip all work
                    if edgeLastOpacity[key] != 0 {
                        entity.components.set(OpacityComponent(opacity: 0))
                        edgeLastOpacity[key] = 0
                    }
                    continue
                }
            }

            // Only reposition geometry when node positions actually changed
            if positionsChanged {
                let p1 = from * scaleFactor
                let p2 = to * scaleFactor
                let delta = p2 - p1
                let length = simd_length(delta)
                guard length > 0.001 else { continue }
                let midpoint = (p1 + p2) / 2
                let radius = connected ? edgeRadius * 2.5 : edgeRadius * 1.3

                entity.position = midpoint
                entity.orientation = simd_quatf(from: SIMD3<Float>(0, 1, 0), to: normalize(delta))
                entity.scale = SIMD3<Float>(radius, length, radius)
            }

            // Pulse opacity — update every frame but skip ECS mutation if unchanged
            let edgeMid = (from + to) / 2
            let edgeDist = simd_length(edgeMid - camPos)
            let fogNear: Float = 100
            let fogT = max(0, min(1, (edgeDist - fogNear) / (fogFar - fogNear)))
            let depthFade = max(Float(0.0), 1.0 - fogT * 0.95)
            let scaledMid = edgeMid * scaleFactor
            let phase = (scaledMid.x + scaledMid.y + scaledMid.z) * 8.0
            let pulse = (sin(animationTime * 3.0 + phase) + 1.0) * 0.5

            let searchDimmedEdge = renderIsSearchActive
                && !renderSearchMatchIds.contains(edge.sourceId)
                && !renderSearchMatchIds.contains(edge.targetId)

            let edgeFog: Float
            if searchDimmedEdge && !connected {
                edgeFog = 0.02 * depthFade
            } else if connected {
                edgeFog = 0.5 + pulse * 0.3
            } else if isSemanticMode {
                edgeFog = (0.02 + pulse * 0.04) * depthFade
            } else {
                edgeFog = (0.06 + pulse * 0.16) * depthFade
            }
            if abs(edgeFog - (edgeLastOpacity[key] ?? -1)) > 0.02 {
                entity.components.set(OpacityComponent(opacity: edgeFog))
                edgeLastOpacity[key] = edgeFog
            }
        }
    }

    func updateEdges(positions: [Int64: SIMD3<Float>],
                     edges: [EdgeData],
                     nodes: [NodeData],
                     layoutMode: LayoutMode,
                     colorMap: [String: Color],
                     selectedNode: Int64?) {
        let isSemanticMode = layoutMode == .embedding
        let camPos = cameraPosition  // scaled coordinates
        let fogNear: Float = 100
        let fogFar: Float = 1200

        // Build node project lookup
        let nodeProject: [Int64: String] = Dictionary(
            nodes.map { ($0.id, $0.project) }, uniquingKeysWith: { _, last in last }
        )

        // Build current edge set
        var currentKeys = Set<EdgeKey>()
        for edge in edges {
            currentKeys.insert(EdgeKey(source: edge.sourceId, target: edge.targetId))
        }

        // Remove entities for edges no longer present
        for (key, entity) in edgeEntities where !currentKeys.contains(key) {
            entity.removeFromParent()
            edgeEntities.removeValue(forKey: key)
            edgeLastOpacity.removeValue(forKey: key)
        }

        // Update or create edge entities
        for edge in edges {
            guard let from = positions[edge.sourceId],
                  let to = positions[edge.targetId] else { continue }

            let p1 = from * scaleFactor
            let p2 = to * scaleFactor
            let delta = p2 - p1
            let length = simd_length(delta)
            guard length > 0.001 else { continue }

            let midpoint = (p1 + p2) / 2
            let dir = normalize(delta)
            let rotation = simd_quatf(from: SIMD3<Float>(0, 1, 0), to: dir)
            let connected = edge.sourceId == selectedNode || edge.targetId == selectedNode
            let project = nodeProject[edge.sourceId]
            let mat = getEdgeMaterial(connected: connected, semantic: isSemanticMode,
                                      project: project, colorMap: colorMap)

            // Depth fog
            let edgeMid = (from + to) / 2
            let edgeDist = simd_length(edgeMid - camPos)
            let fogT = max(0, min(1, (edgeDist - fogNear) / (fogFar - fogNear)))
            let depthFade = max(Float(0.0), 1.0 - fogT * 0.95)

            // Pulse animation — opacity swings from dim to bright
            let phase = (midpoint.x + midpoint.y + midpoint.z) * 8.0
            let pulse = (sin(animationTime * 3.0 + phase) + 1.0) * 0.5  // 0..1

            let searchDimmedEdge = renderIsSearchActive
                && !renderSearchMatchIds.contains(edge.sourceId)
                && !renderSearchMatchIds.contains(edge.targetId)

            let edgeFog: Float
            if searchDimmedEdge && !connected {
                edgeFog = 0.02 * depthFade
            } else if connected {
                edgeFog = 0.5 + pulse * 0.3  // 0.5–0.8
            } else if isSemanticMode {
                edgeFog = (0.02 + pulse * 0.04) * depthFade  // very subtle
            } else {
                edgeFog = (0.06 + pulse * 0.16) * depthFade  // 0.06–0.22, clearly visible pulse
            }

            let key = EdgeKey(source: edge.sourceId, target: edge.targetId)

            // Connected edges get slightly thicker
            let radius = connected ? edgeRadius * 2.5 : edgeRadius * 1.3

            if let entity = edgeEntities[key] {
                entity.position = midpoint
                entity.orientation = rotation
                entity.scale = SIMD3<Float>(radius, length, radius)
                entity.model?.materials = [mat]
                if abs(edgeFog - (edgeLastOpacity[key] ?? -1)) > 0.02 {
                    entity.components.set(OpacityComponent(opacity: edgeFog))
                    edgeLastOpacity[key] = edgeFog
                }
            } else {
                let entity = ModelEntity(mesh: edgeMesh, materials: [mat])
                entity.position = midpoint
                entity.orientation = rotation
                entity.scale = SIMD3<Float>(radius, length, radius)
                entity.components.set(OpacityComponent(opacity: edgeFog))
                edgeLastOpacity[key] = edgeFog
                edgeContainer.addChild(entity)
                edgeEntities[key] = entity
            }
        }
    }

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
        for (_, entity) in nodeEntities { entity.removeFromParent() }
        nodeEntities.removeAll()
        nodeLastOpacity.removeAll()
        for (_, entity) in edgeEntities { entity.removeFromParent() }
        edgeEntities.removeAll()
        edgeLastOpacity.removeAll()
        for (_, entity) in nebulaEmitters { entity.removeFromParent() }
        nebulaEmitters.removeAll()
        nebulaColorCache.removeAll()
        // Clean up flow particles
        for (_, particles) in flowParticles {
            for p in particles { p.entity.removeFromParent() }
        }
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
                if let entity = nodeEntities[childId] {
                    entity.position = lerpedPos * scaleFactor
                }
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

        // Hard cleanup on selection change — flush ALL particles so old paths don't linger
        if sel != flowLastSelectedNode {
            for (_, particles) in flowParticles {
                for p in particles { p.entity.removeFromParent() }
            }
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

        // Clean up particles for inactive edges
        for key in flowParticles.keys where !activeKeys.contains(key) {
            if let particles = flowParticles[key] {
                for p in particles { p.entity.removeFromParent() }
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
                        particles[i].entity.removeFromParent()
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

    /// Render inputs — written by Timer (force sim), read by SceneEvents.Update (render).
    /// Both run on @MainActor so no races.
    var renderPositions: [Int64: SIMD3<Float>] = [:]
    var renderNodes: [NodeData] = []
    var renderEdges: [EdgeData] = []
    var renderHubs: Set<Int64> = []
    var renderColorMap: [String: Color] = [:]
    var renderSelectedNode: Int64?
    var renderLayoutMode: LayoutMode = .forceDirected
    var renderGlowingNodes: [Int64: Date] = [:]
    var renderNewNodes: [Int64: Date] = [:]
    var renderDyingNodes: [Int64: DyingNode] = [:]
    var renderSemanticClusters3D: [SemanticCluster3D] = []
    var renderTopicGroups: [TopicGroupInfo] = []
    var renderClusters: [[Int64]] = []
    var renderViewSize: CGSize = CGSize(width: 800, height: 600)

    // Search spotlight
    var renderSearchMatchIds: Set<Int64> = []
    var renderIsSearchActive: Bool = false

    // Hub expansion state
    var expandedHubs: Set<Int64> = []
    private var preExpansionPositions: [Int64: SIMD3<Float>] = [:]
    private var expansionProgress: [Int64: Float] = [:]
    private var expansionDirection: [Int64: Bool] = [:]  // true=expanding, false=collapsing
    /// Expansion-adjusted positions for labels (read by Graph3DView).
    private(set) var expandedChildPositions: [Int64: SIMD3<Float>] = [:]
    /// Pending hub toggles from gamepad — consumed by Graph3DView for pinning callback.
    var pendingHubToggles: [(hubId: Int64, expanding: Bool)] = []

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
    /// Tracks previous positions to detect when they've changed (avoids redundant edge work).
    private var prevPositionHash: Int = 0

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
        let header = "frame,dt_ms,work_ms,repos_ms,nodes_ms,edges_ms,neb_ms,node_count,edge_count,labels_ms\n"
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

    /// Lightweight per-frame node update: position + depth fog only.
    /// Skips material selection, glow/arrival animations, and point light management.
    func repositionNodes(positions: [Int64: SIMD3<Float>], selectedNode: Int64?) {
        let camPos = cameraPosition
        for (id, pos) in positions {
            guard let entity = nodeEntities[id] else { continue }
            entity.position = pos * scaleFactor
            let dist = simd_length(pos - camPos)
            let fogT = max(0, min(1, (dist - 100) / 1100))
            let baseFogOpacity = id == selectedNode ? 1.0 : max(Float(0.08), 1.0 - fogT * 0.92)
            let searchDimmed = renderIsSearchActive && !renderSearchMatchIds.contains(id)
            let fogOpacity = searchDimmed ? min(baseFogOpacity, 0.12) : baseFogOpacity
            // Skip ECS mutation when opacity hasn't changed visibly
            if abs(fogOpacity - (nodeLastOpacity[id] ?? -1)) > 0.02 {
                entity.components.set(OpacityComponent(opacity: fogOpacity))
                nodeLastOpacity[id] = fogOpacity
            }
        }
    }

    /// Called every RealityKit render frame (vsync-synchronized).
    func renderTick() {
        let now = CFAbsoluteTimeGetCurrent()
        let dtSec = lastRenderTime > 0 ? Float(now - lastRenderTime) : Float(1.0 / 60.0)
        let dt = Double(dtSec) * 1000.0

        pollKeyboard(dt: dtSec)
        pollGamepad(dt: dtSec, selectedNode: &renderSelectedNode,
                    positions: renderPositions, viewSize: renderViewSize)
        updateCamera(dt: dtSec)
        animationTime += dtSec

        // Hub expansion animation (before node/edge updates so positions are current)
        updateExpansions(dt: dtSec)

        let selectionChanged = renderSelectedNode != renderLastSelectedNode

        // Detect if node positions actually changed since last frame
        // Use a cheap hash (count + sample positions) rather than comparing all values
        let posHash: Int = {
            var h = renderPositions.count
            // Sample a few positions to detect changes
            if let first = renderPositions.first {
                h ^= first.value.x.bitPattern.hashValue
                h ^= first.value.y.bitPattern.hashValue
            }
            if renderPositions.count > 10 {
                let mid = renderPositions.index(renderPositions.startIndex,
                                                 offsetBy: renderPositions.count / 2)
                h ^= renderPositions[mid].value.x.bitPattern.hashValue
            }
            return h
        }()
        let positionsChanged = posHash != prevPositionHash
        prevPositionHash = posHash

        // Every frame: cheap position + fog update (skip when nothing moved)
        let t0 = CFAbsoluteTimeGetCurrent()
        if positionsChanged {
            repositionNodes(positions: renderPositions, selectedNode: renderSelectedNode)
        }
        let tRepos = CFAbsoluteTimeGetCurrent()

        // Every 3 frames (or on state change): full material/light/creation/removal
        var tNodes = tRepos
        if renderFrameCount % 3 == 0 || selectionChanged {
            updateNodes(
                positions: renderPositions, nodes: renderNodes, hubs: renderHubs,
                colorMap: renderColorMap, selectedNode: renderSelectedNode,
                glowingNodes: renderGlowingNodes, newNodes: renderNewNodes,
                dyingNodes: renderDyingNodes
            )
            tNodes = CFAbsoluteTimeGetCurrent()
        }

        let edgeInterval: UInt64 = renderFrameCount < 180 ? 12 : 6
        if selectionChanged || renderFrameCount % edgeInterval == 0 {
            updateEdges(
                positions: renderPositions, edges: renderEdges,
                nodes: renderNodes, layoutMode: renderLayoutMode,
                colorMap: renderColorMap, selectedNode: renderSelectedNode
            )
            renderLastSelectedNode = renderSelectedNode
        } else {
            // Lightweight: reposition existing edge entities + update pulse opacity
            // Passes positionsChanged so geometry is only rebuilt when nodes move
            repositionEdgesAnimated(positions: renderPositions, edges: renderEdges,
                                    selectedNode: renderSelectedNode,
                                    positionsChanged: positionsChanged)
        }
        let tEdges = CFAbsoluteTimeGetCurrent()

        // Edge flow particles
        updateFlowParticles(dt: dtSec)

        // Nebulae: skip during initial settle (first 180 frames) to avoid
        // particle system warm-up competing with heavy force layout work
        if renderFrameCount > 180 {
            if renderFrameCount % 30 == 0 {
                updateNebulae()
            } else if renderFrameCount % 2 == 0 {
                updateNebulaBreathing()
            }
        }
        let tNeb = CFAbsoluteTimeGetCurrent()

        let elapsed = (tNeb - now) * 1000.0

        let msRepos = (tRepos - t0) * 1000
        let msNodes = (tNodes - tRepos) * 1000
        let msEdges = (tEdges - tNodes) * 1000
        let msNeb = (tNeb - tEdges) * 1000

        renderLogCounter &+= 1
        if renderLogCounter % 60 == 0 || dt > 25 || elapsed > 10 {
            frameLog.error("[3D-render] dt=\(dt, format: .fixed(precision: 1))ms work=\(elapsed, format: .fixed(precision: 2))ms repos=\(msRepos, format: .fixed(precision: 2)) nodes=\(msNodes, format: .fixed(precision: 2)) edges=\(msEdges, format: .fixed(precision: 2)) neb=\(msNeb, format: .fixed(precision: 2)) | n=\(self.renderPositions.count) e=\(self.renderEdges.count) frame=\(self.renderFrameCount)")
        }

        #if DEBUG
        // Write every frame to CSV for profiling analysis
        if profilingReady {
            profilingLines.append("\(renderFrameCount),\(String(format: "%.2f", dt)),\(String(format: "%.2f", elapsed)),\(String(format: "%.2f", msRepos)),\(String(format: "%.2f", msNodes)),\(String(format: "%.2f", msEdges)),\(String(format: "%.2f", msNeb)),\(renderPositions.count),\(renderEdges.count),\(String(format: "%.2f", lastCanvasLabelMs))\n")
            if renderLogCounter % 60 == 0 { flushProfiling() }
        }
        #endif

        lastRenderTime = now
        renderFrameCount &+= 1
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
    /// Called each frame. Must return the CURRENT live positions.
    var onAnimationTick: (() -> [Int64: SIMD3<Float>])?
    /// Called each frame with current camera state for minimap: (azimuth, cameraPosition, cameraTarget).
    var onCameraUpdate: ((Float, SIMD3<Float>, SIMD3<Float>) -> Void)?
    /// Called when a hub is expanded/collapsed (for pinning children in force simulation).
    var onHubToggle: ((Int64, Bool) -> Void)?

    @State private var scene = Graph3DScene()
    @State private var livePositions: [Int64: SIMD3<Float>] = [:]
    @State private var scrollMonitor: Any?
    @State private var hasCenteredCamera = false
    @State private var cameraStartTime: Date?
    @State private var centerTickCount: Int = 0
    /// Tracks the last synced selection value to detect which side (binding vs gamepad) changed.
    @State private var lastSyncedSelection: Int64? = nil
    private let timer = Timer.publish(every: 1.0 / 60.0, on: .main, in: .common).autoconnect()


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
                    }

                labelsOverlay(viewSize: geo.size)

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
        .onReceive(timer) { _ in
            // Force simulation tick — compute new positions
            let newPos = onAnimationTick?() ?? scene.renderPositions

            // Keep re-centering camera for the first 3 seconds so force simulation
            // has time to spread nodes out — initial positions are often clustered at origin.
            if !newPos.isEmpty {
                if cameraStartTime == nil { cameraStartTime = Date() }
                let elapsed = Date().timeIntervalSince(cameraStartTime ?? Date())
                if (!hasCenteredCamera || elapsed < 3.0) && !scene.isDragging {
                    centerTickCount += 1
                    if centerTickCount % 6 == 0 {
                        scene.centerOnGraph(positions: newPos)
                    }
                    if elapsed >= 3.0 { hasCenteredCamera = true }
                }
            }

            // Two-way selection sync: detect which side changed and propagate
            let sceneChanged = scene.renderSelectedNode != lastSyncedSelection
            let bindingChanged = selectedNode != lastSyncedSelection
            if sceneChanged {
                // Gamepad/reticle changed selection — push scene → binding
                selectedNode = scene.renderSelectedNode
            } else if bindingChanged {
                // UI (activity panel/tap/Escape) changed selection — push binding → scene
                scene.renderSelectedNode = selectedNode
                // Collapse all expanded hubs when deselected from any source
                if selectedNode == nil {
                    collapseAllHubs()
                }
            }
            lastSyncedSelection = selectedNode

            // Push data to scene for vsync-synced rendering
            scene.renderPositions = newPos
            scene.renderNodes = nodes
            scene.renderEdges = edges
            scene.renderHubs = hubs
            scene.renderColorMap = colorMap
            scene.renderLayoutMode = layoutMode
            scene.renderGlowingNodes = glowingNodes
            scene.renderNewNodes = newNodes
            scene.renderDyingNodes = dyingNodes
            scene.renderSemanticClusters3D = semanticClusters3D
            scene.renderTopicGroups = topicGroups
            scene.renderClusters = clusters
            scene.renderViewSize = viewFrame.size
            scene.renderSearchMatchIds = searchMatchIds
            scene.renderIsSearchActive = isSearchActive

            // Consume pending hub toggles from gamepad
            for toggle in scene.pendingHubToggles {
                onHubToggle?(toggle.hubId, toggle.expanding)
            }
            scene.pendingHubToggles.removeAll()

            livePositions = newPos
            // Override with expansion positions for accurate label placement
            for (id, pos) in scene.expandedChildPositions {
                livePositions[id] = pos
            }

            // Report camera state to minimap
            onCameraUpdate?(scene.azimuth, scene.cameraPosition, scene.cameraTarget)
        }
    }

    private var realityViewContent: some View {
        RealityView { content in
            let (root, camera) = scene.setup()
            #if DEBUG
            scene.setupProfiling()
            #endif
            content.add(root)
            content.add(camera)

            // Subscribe to RealityKit's display-synced update event.
            // This fires at vsync, eliminating phase mismatch between
            // Timer and RealityKit's render loop.
            scene.renderSubscription = content.subscribe(to: SceneEvents.Update.self) { _ in
                scene.renderTick()
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
            onHubToggle?(hubId, false)
        }
    }

    private func tapGesture(viewSize: CGSize) -> some Gesture {
        SpatialTapGesture()
            .onEnded { value in
                if let nodeId = scene.hitTest(at: value.location, viewSize: viewSize, positions: livePositions) {
                    if hubs.contains(nodeId) {
                        let expanding = !scene.expandedHubs.contains(nodeId)
                        // Collapse other expanded hubs first
                        for hubId in scene.expandedHubs where hubId != nodeId {
                            scene.toggleHubExpansion(hubId: hubId)
                            onHubToggle?(hubId, false)
                        }
                        scene.toggleHubExpansion(hubId: nodeId)
                        onHubToggle?(nodeId, expanding)
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

    // MARK: - Labels Overlay

    @ViewBuilder
    private func labelsOverlay(viewSize: CGSize) -> some View {
        let nodeById = Dictionary(nodes.map { ($0.id, $0) }, uniquingKeysWith: { _, last in last })
        Canvas { [scene] context, size in
            #if DEBUG
            let canvasStart = CFAbsoluteTimeGetCurrent()
            defer {
                let canvasMs = (CFAbsoluteTimeGetCurrent() - canvasStart) * 1000
                Task { @MainActor in scene.lastCanvasLabelMs = canvasMs }
            }
            #endif
            struct LabelEntry {
                let id: Int64
                let screenPos: CGPoint
                let nodeData: NodeData
                let depthFromCam: Float
                let isSelected: Bool
                let isHub: Bool
            }

            // Precompute children of expanded hubs for visibility bypass
            var expandedChildren = Set<Int64>()
            for hubId in scene.expandedHubs {
                for childId in scene.childrenOfHub(hubId) {
                    expandedChildren.insert(childId)
                }
            }

            var entries: [LabelEntry] = []
            var minDepth: Float = .greatestFiniteMagnitude
            var maxDepth: Float = 0

            for (id, pos3D) in livePositions {
                guard let screenPos = scene.project(point3D: pos3D, viewSize: size) else { continue }
                guard let nodeData = nodeById[id] else { continue }

                let depth = scene.cameraDistance(to: pos3D)
                let isSelected = id == selectedNode
                let isHub = hubs.contains(id)
                let importance = CGFloat(max(1, nodeData.importance))

                // Visibility by actual distance from camera — not a global "zoom level"
                // Closer nodes are visible; farther nodes culled by tier.
                // Children of expanded hubs always visible for inspection.
                if !isSelected && !expandedChildren.contains(id) {
                    let maxVisible: CGFloat = isHub ? 1200 : (200 + importance * 80)
                    guard CGFloat(depth) < maxVisible else { continue }
                }

                minDepth = min(minDepth, depth)
                maxDepth = max(maxDepth, depth)

                entries.append(LabelEntry(
                    id: id, screenPos: screenPos, nodeData: nodeData,
                    depthFromCam: depth, isSelected: isSelected, isHub: isHub
                ))
            }

            // Sort front-to-back first for cap, then reverse for draw order
            entries.sort { $0.depthFromCam < $1.depthFromCam }

            // Cap at 80 labels max — nearest labels are most important.
            // Selected and hub nodes always survive the cap.
            let maxLabels = 80
            if entries.count > maxLabels {
                let prioritized = entries.prefix(while: { $0.isSelected || $0.isHub })
                let rest = entries.dropFirst(prioritized.count)
                entries = Array(prioritized) + Array(rest.prefix(maxLabels - prioritized.count))
            }

            // Reverse to back-to-front: far labels drawn first, near labels on top
            entries.reverse()

            let depthRange = max(1, maxDepth - minDepth)

            for entry in entries {
                let importance = CGFloat(max(1, entry.nodeData.importance))
                let nodeDist = CGFloat(entry.depthFromCam)

                // Depth factor: 1.0 for nearest, fades toward farthest visible
                let depthNorm = CGFloat((entry.depthFromCam - minDepth) / depthRange)
                let depthFade = entry.isSelected ? 1.0 : (1.0 - depthNorm * 0.7)

                // Distance-based fade: smooth fade-in as camera approaches
                let maxVisible: CGFloat = entry.isHub ? 1200 : (200 + importance * 80)
                let fadeRange: CGFloat = maxVisible * 0.3
                let fadeT = entry.isSelected ? 1.0 : min(1.0, max(0.0, (maxVisible - nodeDist) / fadeRange))

                // Font size: larger when close, smaller when far — per-node distance
                let baseSize: CGFloat = entry.isSelected ? 12 : (entry.isHub ? 10 : 8.5)
                let distScale = CGFloat(400.0 / max(entry.depthFromCam, 50))
                let fontSize = max(7, min(20, baseSize * pow(distScale, 0.5)))

                let baseOpacity: CGFloat = entry.isSelected ? 0.95 : (entry.isHub ? 0.8 : 0.6)
                let searchDimmedLabel = scene.renderIsSearchActive && !scene.renderSearchMatchIds.contains(entry.id)
                let searchFade: CGFloat = searchDimmedLabel ? 0.15 : 1.0
                let opacity = baseOpacity * fadeT * depthFade * searchFade

                guard opacity > 0.02 else { continue }

                let labelPos = CGPoint(x: entry.screenPos.x, y: entry.screenPos.y - fontSize - 4)
                let fontWeight: Font.Weight = entry.isSelected || entry.isHub ? .bold : .medium
                let font: Font = .system(size: fontSize, weight: fontWeight, design: .monospaced)

                // Single draw with shadow for outline — 5x cheaper than 4-offset stroke
                let text = Text(entry.nodeData.label).font(font)
                    .foregroundStyle(.white.opacity(opacity))
                context.drawLayer { layerCtx in
                    layerCtx.addFilter(.shadow(color: .black.opacity(min(1.0, opacity * 1.5)),
                                               radius: 1.5, x: 0, y: 0))
                    layerCtx.draw(text, at: labelPos)
                }
            }
        }
        .allowsHitTesting(false)
    }
}
