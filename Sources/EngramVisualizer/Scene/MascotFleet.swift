import Metal
import simd
import SwiftUI

/// Notification protocol for memory lifecycle events → mascot animations.
@MainActor
protocol MascotNotifier: AnyObject {
    func onNodeCreated(nodeId: UUID, project: String)
    func onNodeDeleted(nodeId: UUID, project: String, lastPosition: SIMD3<Float>?)
    func onNodeUpdated(nodeId: UUID, project: String, changes: NodeChangeInfo)
}

/// Describes what changed on a node update.
struct NodeChangeInfo {
    let importanceChanged: Bool
    let topicChanged: Bool
    let isAccessOnly: Bool
}

/// Manages a fleet of per-project mascots within a single galaxy.
/// Each project gets its own MascotSystem that patrols only that project's nodes.
@MainActor
final class MascotFleet: MascotNotifier {
    let galaxyId: String
    private(set) var mascots: [String: MascotSystem] = [:]  // project -> mascot
    private let device: MTLDevice
    private let sharedResources: MascotSharedResources
    private let maxMascots = 10

    /// Which project's mascot is currently in chat mode (only one at a time).
    private(set) var chattingMascotProject: String?

    init(galaxyId: String, device: MTLDevice, sharedResources: MascotSharedResources) {
        self.galaxyId = galaxyId
        self.device = device
        self.sharedResources = sharedResources
    }

    // MARK: - Project Sync

    /// Add/remove mascots to match the active project set.
    /// `colorMap` provides tint colors per project.
    func syncProjects(active: Set<String>, colorMap: [String: NSColor]) {
        // Remove mascots for projects that are no longer active
        for project in mascots.keys where !active.contains(project) {
            mascots.removeValue(forKey: project)
            if chattingMascotProject == project { chattingMascotProject = nil }
        }

        // Add mascots for new projects (up to max)
        for project in active {
            guard mascots[project] == nil else { continue }
            guard mascots.count < maxMascots else { break }

            let tint = colorMap[project].map { nsColorToTint($0) } ?? SIMD3<Float>(0, 0.8, 1.0)
            let mascot = MascotSystem(
                device: device, shared: sharedResources,
                projectId: project, tint: tint
            )
            mascots[project] = mascot
        }
    }

    // MARK: - Per-Frame Update

    func update(
        dt: Float,
        camera: CameraController,
        positions: [UUID: SIMD3<Float>],
        nodesByProject: [String: [UUID: SIMD3<Float>]],
        nodeInfo: [UUID: MascotNodeInfo],
        maintenanceActive: Bool
    ) {
        for (project, mascot) in mascots {
            let projectPositions = nodesByProject[project] ?? [:]

            // Build node info for current target only
            var info: [UUID: MascotNodeInfo] = [:]
            if let targetId = mascot.currentTargetId, let ni = nodeInfo[targetId] {
                info[targetId] = ni
            }

            // Handle maintenance state
            if maintenanceActive {
                if case .busy = mascot.currentState {
                    // Already busy
                } else {
                    // Enter busy mode unless doing higher-priority work
                    switch mascot.currentState {
                    case .conjure, .absorb: break  // don't interrupt
                    default: mascot.enterBusyMode()
                    }
                }
            } else if case .busy = mascot.currentState {
                mascot.exitBusyMode()
            }

            mascot.update(dt: dt, camera: camera, nodePositions: projectPositions, nodeInfo: info)
        }
    }

    // MARK: - Drawing (opaque pass)

    func drawAll(
        encoder: MTLRenderCommandEncoder,
        frameUniformBuf: MTLBuffer,
        lightUniformBuf: MTLBuffer,
        pipeline: MTLRenderPipelineState,
        depthState: MTLDepthStencilState
    ) {
        for mascot in mascots.values {
            mascot.draw(
                encoder: encoder,
                frameUniformBuf: frameUniformBuf,
                lightUniformBuf: lightUniformBuf,
                pipeline: pipeline,
                depthState: depthState
            )
        }
    }

    // MARK: - Drawing (transparent pass — grouped by effect type)

    func drawAllThrusterParticles(
        encoder: MTLRenderCommandEncoder,
        frameUniformBuf: MTLBuffer,
        pipeline: MTLRenderPipelineState
    ) {
        for mascot in mascots.values {
            mascot.drawThrusterParticles(
                encoder: encoder,
                frameUniformBuf: frameUniformBuf,
                pipeline: pipeline
            )
        }
    }

    func drawAllArcaneCircles(
        encoder: MTLRenderCommandEncoder,
        frameUniformBuf: MTLBuffer,
        pipeline: MTLRenderPipelineState
    ) {
        for mascot in mascots.values where mascot.arcaneVisible {
            mascot.drawArcaneCircle(
                encoder: encoder,
                frameUniformBuf: frameUniformBuf,
                pipeline: pipeline
            )
        }
    }

    func drawAllNodeRings(
        encoder: MTLRenderCommandEncoder,
        frameUniformBuf: MTLBuffer,
        pipeline: MTLRenderPipelineState
    ) {
        for mascot in mascots.values where mascot.ringsVisible {
            mascot.drawNodeRings(
                encoder: encoder,
                frameUniformBuf: frameUniformBuf,
                pipeline: pipeline
            )
        }
    }

    func drawAllHoloScreens(
        encoder: MTLRenderCommandEncoder,
        frameUniformBuf: MTLBuffer,
        pipeline: MTLRenderPipelineState
    ) {
        for mascot in mascots.values where mascot.holoVisible {
            mascot.drawHoloScreen(
                encoder: encoder,
                frameUniformBuf: frameUniformBuf,
                pipeline: pipeline
            )
        }
    }

    // MARK: - Chat Mode

    /// Enter chat mode for the mascot nearest to the camera.
    func enterChat(cameraPosition: SIMD3<Float>) {
        // Find nearest mascot
        var bestProject: String?
        var bestDist: Float = .greatestFiniteMagnitude
        for (project, mascot) in mascots {
            let dist = simd_length(mascot.currentPosition - cameraPosition)
            if dist < bestDist {
                bestDist = dist
                bestProject = project
            }
        }
        if let project = bestProject {
            mascots[project]?.setChatting(true)
            chattingMascotProject = project
        }
    }

    /// Exit chat mode.
    func exitChat() {
        if let project = chattingMascotProject {
            mascots[project]?.setChatting(false)
            chattingMascotProject = nil
        }
    }

    // MARK: - MascotNotifier

    func onNodeCreated(nodeId: UUID, project: String) {
        guard let mascot = mascots[project] else { return }
        // Position will be set when the node appears in the simulation
        mascot.enqueueTask(.create(nodeId: nodeId, position: .zero))
    }

    func onNodeDeleted(nodeId: UUID, project: String, lastPosition: SIMD3<Float>?) {
        guard let mascot = mascots[project] else { return }
        mascot.enqueueTask(.delete(nodeId: nodeId, position: lastPosition ?? mascot.currentPosition))
    }

    func onNodeUpdated(nodeId: UUID, project: String, changes: NodeChangeInfo) {
        guard let mascot = mascots[project] else { return }
        guard !changes.isAccessOnly else { return }  // too frequent, no visual meaning

        // Limit pending reactions to 3 — drop oldest if overflow
        let pendingReactions = mascot.taskQueue.filter {
            if case .react = $0 { return true } else { return false }
        }.count
        guard pendingReactions < 3 else { return }

        let type: MascotSystem.ReactionType
        if changes.importanceChanged {
            type = .importanceBoost
        } else if changes.topicChanged {
            type = .inspect
        } else {
            type = .headTurn
        }
        mascot.enqueueTask(.react(nodeId: nodeId, type: type))
    }

    // MARK: - Mascot Transfer (migration between galaxies)

    /// Remove and return a project's mascot (for transfer to another fleet).
    func extractMascot(for project: String) -> MascotSystem? {
        let mascot = mascots.removeValue(forKey: project)
        if chattingMascotProject == project { chattingMascotProject = nil }
        return mascot
    }

    /// Insert an already-initialized mascot (transferred from another fleet).
    func insertMascot(_ mascot: MascotSystem, for project: String) {
        guard mascots.count < maxMascots else { return }
        mascots[project] = mascot
    }

    // MARK: - Query

    /// Whether any mascot is currently inspecting a node (for node instance packing).
    var anyInspecting: Bool {
        mascots.values.contains { $0.arcaneIntensity > 0.01 }
    }

    /// Find the mascot inspecting a given node, if any.
    func inspectingMascot(for nodeId: UUID) -> MascotSystem? {
        mascots.values.first { $0.currentTargetId == nodeId && $0.arcaneIntensity > 0.01 }
    }

    // MARK: - Helpers

    private func nsColorToTint(_ color: NSColor) -> SIMD3<Float> {
        let c = color.usingColorSpace(.sRGB) ?? color
        return SIMD3<Float>(Float(c.redComponent), Float(c.greenComponent), Float(c.blueComponent))
    }
}
