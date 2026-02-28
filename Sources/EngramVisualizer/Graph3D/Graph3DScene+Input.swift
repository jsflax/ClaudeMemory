import SwiftUI
import RealityKit
import GameController
import simd

extension Graph3DScene {

    // MARK: - Keyboard Input

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

    // MARK: - Gamepad Input

    func pollGamepad(dt: Float, selectedNode: inout UUID?,
                     positions: [UUID: SIMD3<Float>], viewSize: CGSize) {
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

    // MARK: - Project Teleport

    func teleportToNextProject(positions: [UUID: SIMD3<Float>], nodes: [NodeData], direction: Int = 1) {
        // Group node IDs and positions by project
        var projectNodeIds: [String: [UUID]] = [:]
        var projectPositions: [String: [SIMD3<Float>]] = [:]
        let nodeById = renderStore?.nodeById ?? [:]
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
        let nodeById = renderStore?.nodeById ?? [:]

        var pts: [SIMD3<Float>] = []
        var nodeIds: [UUID] = []
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

    // MARK: - Node Cycling

    func cycleNode(current: UUID?, direction: Int,
                           positions: [UUID: SIMD3<Float>], viewSize: CGSize) -> UUID? {
        // Only consider nodes visible on screen
        let visible = positions.compactMap { (id, pos) -> (UUID, Float)? in
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
}
