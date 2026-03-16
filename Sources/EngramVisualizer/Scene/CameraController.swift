import simd
import GameController
import os
import EngramSceneKit

private let cameraLog = Logger(subsystem: "io.engram.app", category: "CameraController")

/// Camera state and math extracted from Graph3DScene.
/// Shared by both the Metal renderer and the RealityKit renderer.
/// Input handling, smoothing, and animation live here; pure math delegates to CameraState.
@Observable
@MainActor
final class CameraController {

    // Target state (written by gestures/events)
    var targetAzimuth: Float = 0
    var targetElevation: Float = 0.3
    var targetCameraPos: SIMD3<Float> = .zero

    // Smoothed state (lerped each frame, used for rendering)
    @ObservationIgnored private(set) var azimuth: Float = 0
    @ObservationIgnored private(set) var elevation: Float = 0.3
    @ObservationIgnored private(set) var cameraTarget: SIMD3<Float> = .zero

    var orbitRadius: Float = 2.0
    private let smoothing: Float = 0.2
    var isDragging = false

    let scaleFactor: Float = 1.0 / 200.0
    let fovDegrees: Float = 60

    // MARK: - CameraState snapshot

    /// Build a CameraState value from current smoothed state.
    var state: CameraState {
        CameraState(
            azimuth: azimuth, elevation: elevation,
            cameraTarget: cameraTarget, orbitRadius: orbitRadius,
            scaleFactor: scaleFactor, fovDegrees: fovDegrees
        )
    }

    // MARK: - Computed Properties (delegate to CameraState)

    var cameraPosition: SIMD3<Float> { state.cameraPosition }
    var forward: SIMD3<Float> { state.forward }
    var right: SIMD3<Float> { state.right }
    var up: SIMD3<Float> { state.up }

    // MARK: - View / Projection Matrices

    func viewMatrix() -> simd_float4x4 { state.viewMatrix() }
    func projectionMatrix(aspect: Float) -> simd_float4x4 { state.projectionMatrix(aspect: aspect) }

    // MARK: - Camera Movement

    func pan(dx: Float, dy: Float) {
        let speed: Float = 1.5
        let offset = -right * dx * speed + up * dy * speed
        targetCameraPos += offset
    }

    func dolly(amount: Float, cursorNX: Float = 0, cursorNY: Float = 0) {
        let fovTan: Float = tan(Float.pi / 6)
        let dir = normalize(forward + right * cursorNX * fovTan + up * cursorNY * fovTan)
        targetCameraPos += dir * amount
    }

    func lookRotate(deltaAz: Float, deltaEl: Float) {
        let oldOx = orbitRadius * cos(targetElevation) * sin(targetAzimuth)
        let oldOy = orbitRadius * sin(targetElevation)
        let oldOz = orbitRadius * cos(targetElevation) * cos(targetAzimuth)
        let camPos = targetCameraPos + SIMD3(oldOx, oldOy, oldOz)

        targetAzimuth += deltaAz
        targetElevation = max(-.pi/2 + 0.1, min(.pi/2 - 0.1, targetElevation + deltaEl))

        let newOx = orbitRadius * cos(targetElevation) * sin(targetAzimuth)
        let newOy = orbitRadius * sin(targetElevation)
        let newOz = orbitRadius * cos(targetElevation) * cos(targetAzimuth)
        targetCameraPos = camPos - SIMD3(newOx, newOy, newOz)
    }

    func captureLookState() -> (azimuth: Float, elevation: Float, camPos: SIMD3<Float>) {
        isDragging = true
        let az = targetAzimuth
        let el = targetElevation
        let offset = orbitOffset(azimuth: az, elevation: el)
        return (azimuth: az, elevation: el, camPos: targetCameraPos + offset)
    }

    func endDrag() {
        isDragging = false
    }

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

    private func orbitOffset(azimuth az: Float, elevation el: Float) -> SIMD3<Float> {
        SIMD3(orbitRadius * cos(el) * sin(az),
              orbitRadius * sin(el),
              orbitRadius * cos(el) * cos(az))
    }

    func cameraDistance(to point: SIMD3<Float>) -> Float {
        simd_length(point - cameraPosition)
    }

    // MARK: - Camera Update (per frame)

    func updateCamera(dt: Float) {
        let targetDt: Float = 1.0 / 60.0
        let s: Float = isDragging ? 1.0 : smoothing
        let factor = 1.0 - pow(1.0 - s, dt / targetDt)
        let newAz = azimuth + (targetAzimuth - azimuth) * factor
        let newEl = elevation + (targetElevation - elevation) * factor
        let newTarget = cameraTarget + (targetCameraPos - cameraTarget) * factor
        let threshold: Float = 0.0001
        if abs(newAz - azimuth) > threshold { azimuth = newAz }
        else if azimuth != targetAzimuth { azimuth = targetAzimuth }
        if abs(newEl - elevation) > threshold { elevation = newEl }
        else if elevation != targetElevation { elevation = targetElevation }
        if simd_length(newTarget - cameraTarget) > threshold { cameraTarget = newTarget }
        else if cameraTarget != targetCameraPos { cameraTarget = targetCameraPos }
    }

    func centerOnGraph(positions: [UUID: SIMD3<Float>]) {
        guard !positions.isEmpty else { return }
        var sum = SIMD3<Float>.zero
        for (_, pos) in positions { sum += pos }
        let centroid = sum / Float(positions.count)

        var maxDist: Float = 0
        for (_, pos) in positions {
            maxDist = max(maxDist, simd_length(pos - centroid))
        }
        orbitRadius = max(maxDist * 3.5, 800)
        let scaledCentroid = centroid * scaleFactor
        targetCameraPos = scaledCentroid
        cameraTarget = scaledCentroid
    }

    /// Whether camera is still lerping toward targets.
    var isMoving: Bool {
        abs(targetAzimuth - azimuth) > 0.0001
            || abs(targetElevation - elevation) > 0.0001
            || simd_length(targetCameraPos - cameraTarget) > 0.01
    }

    // MARK: - Projection / Hit Testing

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

        let fovTan = tan(Float.pi / 6)
        let aspect = Float(viewSize.width / viewSize.height)

        let screenX = CGFloat((projX / (fovTan * aspect) * 0.5 + 0.5)) * viewSize.width
        let screenY = CGFloat((0.5 - projY / fovTan * 0.5)) * viewSize.height

        return CGPoint(x: screenX, y: screenY)
    }

    func hitTest(at location: CGPoint, viewSize: CGSize, positions: [UUID: SIMD3<Float>],
                 currentTarget: UUID? = nil) -> UUID? {
        var closest: UUID?
        var closestDist: CGFloat = 30

        for (id, pos) in positions {
            guard let screenPos = project(point3D: pos, viewSize: viewSize) else { continue }
            let dist = hypot(location.x - screenPos.x, location.y - screenPos.y)
            if dist < closestDist {
                closestDist = dist
                closest = id
            }
        }

        // Hysteresis: keep the current target if it's still within a larger radius,
        // even if nothing new was acquired within the tighter 30px radius.
        if closest == nil, let current = currentTarget,
           let pos = positions[current],
           let screenPos = project(point3D: pos, viewSize: viewSize) {
            let dist = hypot(location.x - screenPos.x, location.y - screenPos.y)
            if dist < 45 {
                return current
            }
        }

        return closest
    }

    // MARK: - Keyboard Input

    var heldKeys: Set<String> = []

    func pollKeyboard(dt: Float) {
        guard !heldKeys.isEmpty else { return }

        let sprint: Float = heldKeys.contains("shift") ? 3.0 : 1.0
        let moveSpeed: Float = 200 * dt * sprint
        let strafeSpeed: Float = 300 * dt * sprint
        let vertSpeed: Float = 300 * dt * sprint
        let lookSpeed: Float = 2.0 * dt

        if heldKeys.contains("w") { dolly(amount: moveSpeed) }
        if heldKeys.contains("s") { dolly(amount: -moveSpeed) }
        if heldKeys.contains("a") { pan(dx: strafeSpeed, dy: 0) }
        if heldKeys.contains("d") { pan(dx: -strafeSpeed, dy: 0) }
        if heldKeys.contains("e") { pan(dx: 0, dy: vertSpeed) }
        if heldKeys.contains("q") { pan(dx: 0, dy: -vertSpeed) }
        if heldKeys.contains("i") { lookRotate(deltaAz: 0, deltaEl: lookSpeed) }
        if heldKeys.contains("k") { lookRotate(deltaAz: 0, deltaEl: -lookSpeed) }
        if heldKeys.contains("j") { lookRotate(deltaAz: lookSpeed, deltaEl: 0) }
        if heldKeys.contains("l") { lookRotate(deltaAz: -lookSpeed, deltaEl: 0) }
    }

    // MARK: - Teleport

    var teleportProjectIndex = 0
    var teleportCounter: Int = 0
    var teleportLabel: String?

    func teleportToNextProject(positions: [UUID: SIMD3<Float>], nodes: [NodeData],
                                hubs: Set<UUID>, direction: Int = 1) {
        var projectNodeIds: [String: [UUID]] = [:]
        var projectPositions: [String: [SIMD3<Float>]] = [:]
        let nodeById = Dictionary(nodes.map { ($0.id, $0) }, uniquingKeysWith: { _, last in last })
        for (id, pos) in positions {
            guard let node = nodeById[id] else { continue }
            projectNodeIds[node.project, default: []].append(id)
            projectPositions[node.project, default: []].append(pos)
        }

        let projects = projectPositions.keys.sorted()
        guard !projects.isEmpty else { return }

        teleportProjectIndex = (teleportProjectIndex + direction + projects.count) % projects.count
        let project = projects[teleportProjectIndex]
        let pts = projectPositions[project]!
        let nodeIds = projectNodeIds[project]!

        let hubId = nodeIds.first(where: { hubs.contains($0) })

        let targetPos: SIMD3<Float>
        if let hubId, let pos = positions[hubId] {
            targetPos = pos
        } else {
            var sum = SIMD3<Float>.zero
            for p in pts { sum += p }
            targetPos = sum / Float(pts.count)
        }

        var maxSpread: Float = 0
        for p in pts {
            maxSpread = max(maxSpread, simd_length(p - targetPos))
        }
        orbitRadius = max(min(maxSpread * 0.4, 250), 60)
        targetCameraPos = targetPos
        cameraTarget = targetPos
        teleportCounter += 1
        teleportLabel = project
    }

    func driveToProject(_ project: String, positions: [UUID: SIMD3<Float>],
                        nodes: [NodeData], hubs: Set<UUID>) {
        let nodeById = Dictionary(nodes.map { ($0.id, $0) }, uniquingKeysWith: { _, last in last })

        var pts: [SIMD3<Float>] = []
        var nodeIds: [UUID] = []
        for (id, pos) in positions {
            guard let node = nodeById[id], node.project == project else { continue }
            pts.append(pos)
            nodeIds.append(id)
        }
        guard !pts.isEmpty else { return }

        let hubId = nodeIds.first(where: { hubs.contains($0) })
        let targetPos: SIMD3<Float>
        if let hubId, let pos = positions[hubId] {
            targetPos = pos
        } else {
            var sum = SIMD3<Float>.zero
            for p in pts { sum += p }
            targetPos = sum / Float(pts.count)
        }

        var maxSpread: Float = 0
        for p in pts { maxSpread = max(maxSpread, simd_length(p - targetPos)) }
        orbitRadius = max(min(maxSpread * 0.4, 250), 60)
        targetCameraPos = targetPos
        teleportCounter += 1
        teleportLabel = project
    }

    /// Teleport camera to a galaxy's world center with appropriate orbit radius.
    func teleportToGalaxy(center: SIMD3<Float>, radius: Float, label: String) {
        orbitRadius = max(radius, 60)
        targetCameraPos = center
        cameraTarget = center
        teleportCounter += 1
        teleportLabel = label
    }

}
