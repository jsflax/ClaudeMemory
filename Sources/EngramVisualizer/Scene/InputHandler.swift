import simd
import GameController
import SwiftUI
import os

/// Handles keyboard, gamepad, and selection input extracted from MetalSceneManager.
/// Pure input state + polling — no render data, no Metal dependencies.
@MainActor
final class InputHandler {

    // MARK: - References

    unowned let camera: CameraController

    // MARK: - Selection State

    var selectedNode: UUID?
    var selectionCallback: ((UUID?) -> Void)?

    // MARK: - Reticle (gamepad targeting)

    var reticleTarget: UUID?
    /// Callback for SwiftUI to observe reticle target changes (MetalSceneManager is not @Observable).
    var reticleCallback: ((UUID?) -> Void)?

    // MARK: - Teleport State

    /// Callback for SwiftUI to observe teleport label/counter changes (MetalSceneManager is not @Observable).
    var teleportCallback: ((String?, Int) -> Void)?
    var teleportLabel: String?
    var teleportCounter: Int = 0
    var teleportProjectIndex: Int = 0
    var teleportGalaxyIndex: Int = 0

    // MARK: - Keyboard Input

    var heldKeys: Set<String> = []

    // MARK: - Drag State

    var isDragging = false

    // MARK: - Gamepad Button State (rising-edge detection)

    private var prevButtonA = false
    private var prevButtonB = false
    private var prevLB = false
    private var prevRB = false
    private var prevButtonY = false

    // MARK: - Instrumentation

    #if ENGRAM_INSTRUMENTATION
    var jitterReticleCallbackFired = false
    #endif

    // MARK: - Init

    init(camera: CameraController) {
        self.camera = camera
    }

    // MARK: - Gamepad

    func pollGamepad(
        dt: Float,
        positions: [UUID: SIMD3<Float>],
        renderNodes: [NodeData],
        renderEdges: [EdgeData],
        renderHubs: Set<UUID>,
        renderViewSize: CGSize,
        hubExpansion: HubExpansionController
    ) {
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
                    let edgeTuples = renderEdges.map { (sourceId: $0.sourceId, targetId: $0.targetId, relation: $0.relation) }
                    hubExpansion.toggleHubExpansion(hubId: target, positions: positions, edges: edgeTuples)
                }
                selectedNode = target
                selectionCallback?(selectedNode)
            }
        }
        prevButtonA = aPressed

        let bPressed = gp.buttonB.isPressed
        if bPressed && !prevButtonB {
            let edgeTuples = renderEdges.map { (sourceId: $0.sourceId, targetId: $0.targetId, relation: $0.relation) }
            for hubId in hubExpansion.expandedHubs { hubExpansion.toggleHubExpansion(hubId: hubId, positions: positions, edges: edgeTuples) }
            selectedNode = nil
            selectionCallback?(nil)
        }
        prevButtonB = bPressed

        // Teleport
        let yPressed = gp.buttonY.isPressed
        if yPressed && !prevButtonY {
            teleportToNextProject(direction: 1, positions: positions, nodes: renderNodes, hubs: renderHubs)
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

    func teleportToNextProject(direction: Int, positions: [UUID: SIMD3<Float>],
                                nodes: [NodeData], hubs: Set<UUID>) {
        camera.teleportToNextProject(positions: positions, nodes: nodes,
                                     hubs: hubs, direction: direction)
        teleportLabel = camera.teleportLabel
        teleportCounter = camera.teleportCounter
        teleportCallback?(teleportLabel, teleportCounter)
    }

    /// Smoothly drive the camera to a named project using cached render data.
    func driveToProject(_ project: String, positions: [UUID: SIMD3<Float>],
                        nodes: [NodeData], hubs: Set<UUID>) {
        camera.driveToProject(project, positions: positions, nodes: nodes, hubs: hubs)
        teleportLabel = camera.teleportLabel
        teleportCounter = camera.teleportCounter
        teleportCallback?(teleportLabel, teleportCounter)
    }

    func teleportToNextGalaxy(direction: Int, galaxyRegistry: GalaxyRegistry?) {
        guard let registry = galaxyRegistry else { return }
        let sorted = registry.galaxies.values.sorted(by: { $0.id < $1.id })
        guard !sorted.isEmpty else { return }
        teleportGalaxyIndex = (teleportGalaxyIndex + direction + sorted.count) % sorted.count
        let galaxy = sorted[teleportGalaxyIndex]

        // Compute radius from galaxy's node spread
        var maxSpread: Float = 200
        let galPositions = registry.unifiedSimulation.positions
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

    func hitTest(at location: CGPoint, viewSize: CGSize, positions: [UUID: SIMD3<Float>]) -> UUID? {
        camera.hitTest(at: location, viewSize: viewSize, positions: positions)
    }

    /// Hit test any mascot in any fleet — returns true if the tap is within 50px of a mascot's screen position.
    func hitTestMascot(at location: CGPoint, viewSize: CGSize, galaxyRegistry: GalaxyRegistry?) -> Bool {
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

    func enterFleetChat(galaxyRegistry: GalaxyRegistry?) {
        let camPos = camera.cameraPosition * camera.scaleFactor
        galaxyRegistry?.galaxies.values.forEach { $0.mascotFleet?.enterChat(cameraPosition: camPos) }
    }

    func exitFleetChat(galaxyRegistry: GalaxyRegistry?) {
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
}
