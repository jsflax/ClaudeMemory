import AVFoundation
import simd
import os

private let audioLog = Logger(subsystem: "io.engram.app", category: "SpatialAudio")

/// Core spatial audio engine for the 3D memory graph.
/// Uses AVAudioEngine + AVAudioEnvironmentNode for HRTF-based 3D spatialization.
/// Listener position/orientation follows the camera each frame.
@MainActor
final class SpatialAudioEngine {

    let engine = AVAudioEngine()
    let environment = AVAudioEnvironmentNode()

    let ambientBed: AmbientBedController
    let clusterDrones: ClusterDroneManager
    let proximityPool: ProximityTonePool
    let eventPlayer: EventSoundPlayer

    // State tracking for change detection
    private var lastCameraPosition: SIMD3<Float> = .zero
    private var lastSelectedNode: UUID?
    private var lastExpandedHubs: Set<UUID> = []
    private var lastTeleportCounter: Int = 0
    private var smoothedVelocity: Float = 0
    private var isRunning = false

    // Coordinate scale: world units → audio meters
    // World space is ~100-1000 units, render space divides by 200.
    // Audio space works in meters — we use render-space coords directly,
    // which puts the graph in a ~0.5-5m radius sphere. Good for HRTF.
    private let scaleFactor: Float = 1.0 / 200.0

    init() {
        ambientBed = AmbientBedController(engine: engine)
        clusterDrones = ClusterDroneManager(environment: environment)
        proximityPool = ProximityTonePool(environment: environment, poolSize: 8)
        eventPlayer = EventSoundPlayer(environment: environment)

        setupAudioGraph()
    }

    private func setupAudioGraph() {
        engine.attach(environment)
        engine.connect(environment, to: engine.mainMixerNode, format: AudioSynthesis.monoFormat)

        // Environment settings
        environment.distanceAttenuationParameters.distanceAttenuationModel = .inverse
        environment.distanceAttenuationParameters.referenceDistance = 0.5
        environment.distanceAttenuationParameters.maximumDistance = 10.0
        environment.distanceAttenuationParameters.rolloffFactor = 2.0

        // Reverb for spacey feel
        environment.reverbParameters.enable = true
        environment.reverbParameters.loadFactoryReverbPreset(.cathedral)
        environment.reverbParameters.level = -20 // subtle

        // Let subsystems attach their nodes
        ambientBed.setup()
        clusterDrones.setup()
        proximityPool.setup()
        eventPlayer.setup()
    }

    // MARK: - Lifecycle

    func start() {
        guard !isRunning else { return }
        do {
            try engine.start()
            isRunning = true
            ambientBed.start()
            audioLog.info("Spatial audio engine started")
        } catch {
            audioLog.error("Failed to start audio engine: \(error)")
        }
    }

    func stop() {
        guard isRunning else { return }
        ambientBed.stop()
        clusterDrones.stopAll()
        proximityPool.stopAll()
        engine.stop()
        isRunning = false
        audioLog.info("Spatial audio engine stopped")
    }

    // MARK: - Per-Frame Tick

    func tick(dt: Float, scene: MetalSceneManager) {
        guard isRunning else { return }

        // Update listener from camera
        updateListener(camera: scene.camera)

        // Compute camera velocity
        let camPos = scene.camera.cameraPosition * scaleFactor
        let velocity = simd_length(camPos - lastCameraPosition) / max(dt, 0.001)
        smoothedVelocity = smoothedVelocity * 0.9 + velocity * 0.1
        lastCameraPosition = camPos

        // Ambient bed: velocity + simulation alpha (tension/resolution)
        let simAlpha = scene.simulation3D?.alpha ?? 0.0
        ambientBed.update(velocity: smoothedVelocity, simAlpha: simAlpha)

        // Cluster drones: update positions from nebula system
        let nebulaGroups = scene.nebulaFog.nebulaGroupsForCurrentMode(
            layoutMode: scene.layoutMode,
            positions: scene.positions,
            nodes: scene.renderNodes,
            semanticClusters3D: scene.semanticClusters3D
        )
        clusterDrones.update(groups: nebulaGroups, colorMap: scene.renderColorMap, scaleFactor: scaleFactor)

        // Proximity tones: nearest nodes
        proximityPool.update(
            positions: scene.positions,
            cameraPosition: scene.camera.cameraPosition,
            selectedNode: scene.selectedNode,
            glowingNodes: scene.glowingNodes,
            newNodes: scene.newNodes,
            scaleFactor: scaleFactor
        )

        // Event detection
        detectEvents(scene: scene)

        // Mascot audio
        updateMascotAudio(scene: scene)
    }

    // MARK: - Listener

    private func updateListener(camera: CameraController) {
        let pos = camera.cameraPosition * scaleFactor
        environment.listenerPosition = AVAudio3DPoint(x: pos.x, y: pos.y, z: pos.z)

        // Convert forward/up to angular orientation
        let fwd = camera.forward
        let yaw = atan2(fwd.x, fwd.z) * 180.0 / Float.pi
        let pitch = asin(-fwd.y) * 180.0 / Float.pi
        environment.listenerAngularOrientation = AVAudio3DAngularOrientation(yaw: yaw, pitch: pitch, roll: 0)
    }

    // MARK: - Event Detection

    private func detectEvents(scene: MetalSceneManager) {
        // Selection change
        if scene.selectedNode != lastSelectedNode {
            if let nodeId = scene.selectedNode, let pos = scene.positions[nodeId] {
                let audioPos = pos * scaleFactor
                eventPlayer.playSelection(at: audioPos)
            } else if lastSelectedNode != nil {
                eventPlayer.playDeselection()
            }
            lastSelectedNode = scene.selectedNode
        }

        // Hub expansion
        let currentExpanded = scene.expandedHubs
        let newlyExpanded = currentExpanded.subtracting(lastExpandedHubs)
        let newlyCollapsed = lastExpandedHubs.subtracting(currentExpanded)
        for hubId in newlyExpanded {
            if let pos = scene.positions[hubId] {
                eventPlayer.playExpansion(at: pos * scaleFactor)
            }
        }
        for hubId in newlyCollapsed {
            if let pos = scene.positions[hubId] {
                eventPlayer.playCollapse(at: pos * scaleFactor)
            }
        }
        lastExpandedHubs = currentExpanded

        // Teleport
        if scene.teleportCounter != lastTeleportCounter {
            eventPlayer.playTeleport()
            lastTeleportCounter = scene.teleportCounter
        }
    }

    // MARK: - Mascot Audio

    private func updateMascotAudio(scene: MetalSceneManager) {
        guard let registry = scene.galaxyRegistry else { return }
        for galaxy in registry.galaxies.values {
            guard let fleet = galaxy.mascotFleet else { continue }
            for (_, mascot) in fleet.mascots {
                eventPlayer.updateMascotThruster(
                    mascot: mascot,
                    scaleFactor: scaleFactor
                )
            }
        }
    }
}
