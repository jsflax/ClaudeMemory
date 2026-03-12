import AVFoundation
import simd
import os

private let audioLog = Logger(subsystem: "io.engram.app", category: "SpatialAudio")

/// Snapshot of scene state captured on main thread, consumed on audio queue.
struct AudioSceneSnapshot: Sendable {
    let positions: [UUID: SIMD3<Float>]
    let cameraPosition: SIMD3<Float>
    let cameraForward: SIMD3<Float>
    let selectedNode: UUID?
    let glowingNodes: [UUID: Date]
    let newNodes: [UUID: Date]
    let expandedHubs: Set<UUID>
    let teleportCounter: Int
    let topologyVersion: UInt64
    let renderEdges: [EdgeData]
    let mascots: [MascotAudioSnapshot]
    let scaleFactor: Float
}

struct MascotAudioSnapshot: Sendable {
    let projectId: String
    let position: SIMD3<Float>
    let volume: Float
}

/// Core spatial audio engine for the 3D memory graph.
/// Uses AVAudioEngine + AVAudioEnvironmentNode for HRTF-based 3D spatialization.
///
/// Architecture: tick() snapshots scene data on the main thread (~0.01ms),
/// then dispatches all audio processing to a dedicated serial queue.
/// This keeps audio work off the render frame entirely.
final class SpatialAudioEngine: @unchecked Sendable {

    let engine = AVAudioEngine()
    let environment = AVAudioEnvironmentNode()

    let edgeHumPool: EdgeHumPool
    let proximityPool: ProximityTonePool
    let eventPlayer: EventSoundPlayer

    // State tracking for change detection (accessed only from audioQueue)
    private var lastSelectedNode: UUID?
    private var lastExpandedHubs: Set<UUID> = []
    private var lastTeleportCounter: Int = 0
    private var lastEdgeTopologyVersion: UInt64 = 0

    private var isRunning = false

    // Coordinate scale: world units → audio meters
    // World space is ~100-1000 units, render space divides by 200.
    // Audio space works in meters — we use render-space coords directly,
    // which puts the graph in a ~0.5-5m radius sphere. Good for HRTF.
    private let scaleFactor: Float = 1.0 / 200.0

    // Background audio processing — serial queue, at most one tick in flight
    private let audioQueue = DispatchQueue(label: "io.engram.audio", qos: .userInteractive)
    private var tickInFlight = false // only read/written on main thread

    init() {
        edgeHumPool = EdgeHumPool(environment: environment, poolSize: 10)
        proximityPool = ProximityTonePool(environment: environment, poolSize: 8)
        eventPlayer = EventSoundPlayer(environment: environment)

        setupAudioGraph()
    }

    private func setupAudioGraph() {
        engine.attach(environment)
        engine.connect(environment, to: engine.mainMixerNode, format: AudioSynthesis.stereoFormat)

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
        edgeHumPool.setup()
        proximityPool.setup()
        eventPlayer.setup()
    }

    // MARK: - Lifecycle

    func start() {
        guard !isRunning else { return }
        do {
            try engine.start()
            isRunning = true
            audioLog.info("Spatial audio engine started")
        } catch {
            audioLog.error("Failed to start audio engine: \(error)")
        }
    }

    func stop() {
        guard isRunning else { return }
        // Synchronous stop — wait for in-flight audio tick to finish
        audioQueue.sync {
            edgeHumPool.stopAll()
            proximityPool.stopAll()
        }
        engine.stop()
        isRunning = false
        audioLog.info("Spatial audio engine stopped")
    }

    // MARK: - Per-Frame Tick (called from main thread)

    #if ENGRAM_INSTRUMENTATION
    nonisolated(unsafe) private var audioTimingFile: UnsafeMutablePointer<FILE>?
    nonisolated(unsafe) private var audioFrameCounter: Int = 0
    #endif

    @MainActor
    func tick(dt: Float, scene: MetalSceneManager) {
        guard isRunning, !tickInFlight else { return }

        // Snapshot scene data on main thread (~0.01ms — value-type copies)
        let snapshot = captureSnapshot(scene: scene)

        tickInFlight = true
        audioQueue.async { [self] in
            processAudio(snapshot: snapshot)
            DispatchQueue.main.async { self.tickInFlight = false }
        }
    }

    // MARK: - Snapshot (main thread)

    @MainActor
    private func captureSnapshot(scene: MetalSceneManager) -> AudioSceneSnapshot {
        var mascots: [MascotAudioSnapshot] = []
        if let registry = scene.galaxyRegistry {
            for galaxy in registry.galaxies.values {
                guard let fleet = galaxy.mascotFleet else { continue }
                for (_, mascot) in fleet.mascots {
                    mascots.append(MascotAudioSnapshot(
                        projectId: mascot.projectId,
                        position: mascot.currentPosition * scaleFactor,
                        volume: Self.thrusterVolume(for: mascot.currentState)
                    ))
                }
            }
        }

        return AudioSceneSnapshot(
            positions: scene.positions,
            cameraPosition: scene.camera.cameraPosition,
            cameraForward: scene.camera.forward,
            selectedNode: scene.selectedNode,
            glowingNodes: scene.glowingNodes,
            newNodes: scene.newNodes,
            expandedHubs: scene.expandedHubs,
            teleportCounter: scene.teleportCounter,
            topologyVersion: scene.galaxyRegistry?.mergedTopologyVersion ?? 0,
            renderEdges: scene.renderEdges,
            mascots: mascots,
            scaleFactor: scaleFactor
        )
    }

    @MainActor
    private static func thrusterVolume(for state: MascotSystem.MascotState) -> Float {
        switch state {
        case .patrol: return 0.25
        case .hover: return 0.08
        case .conjure, .absorb: return 0.15
        case .chatting: return 0.05
        case .idle(.sleeping): return 0.02
        case .idle(.drowsy): return 0.05
        case .idle(.awake): return 0.1
        case .react: return 0.12
        }
    }

    // MARK: - Audio Processing (runs on audioQueue)

    private func processAudio(snapshot: AudioSceneSnapshot) {
        #if ENGRAM_INSTRUMENTATION
        let t0 = CFAbsoluteTimeGetCurrent()
        #endif

        // Update listener position/orientation
        let pos = snapshot.cameraPosition * snapshot.scaleFactor
        environment.listenerPosition = AVAudio3DPoint(x: pos.x, y: pos.y, z: pos.z)
        let fwd = snapshot.cameraForward
        let yaw = atan2(fwd.x, fwd.z) * 180.0 / Float.pi
        let pitch = asin(-fwd.y) * 180.0 / Float.pi
        environment.listenerAngularOrientation = AVAudio3DAngularOrientation(yaw: yaw, pitch: pitch, roll: 0)

        #if ENGRAM_INSTRUMENTATION
        let t1 = CFAbsoluteTimeGetCurrent()
        #endif

        // Edge hum: rebuild topology on edge add/remove, then per-frame update
        if snapshot.topologyVersion != lastEdgeTopologyVersion {
            edgeHumPool.rebuildTopology(
                edges: snapshot.renderEdges,
                positions: snapshot.positions,
                version: snapshot.topologyVersion
            )
            lastEdgeTopologyVersion = snapshot.topologyVersion
        }

        edgeHumPool.update(
            positions: snapshot.positions,
            cameraPosition: snapshot.cameraPosition,
            scaleFactor: snapshot.scaleFactor
        )

        #if ENGRAM_INSTRUMENTATION
        let t2 = CFAbsoluteTimeGetCurrent()
        #endif

        // Proximity tones: nearest nodes
        proximityPool.update(
            positions: snapshot.positions,
            cameraPosition: snapshot.cameraPosition,
            selectedNode: snapshot.selectedNode,
            glowingNodes: snapshot.glowingNodes,
            newNodes: snapshot.newNodes,
            scaleFactor: snapshot.scaleFactor
        )

        #if ENGRAM_INSTRUMENTATION
        let t3 = CFAbsoluteTimeGetCurrent()
        #endif

        // Event detection
        detectEvents(snapshot: snapshot)

        #if ENGRAM_INSTRUMENTATION
        let t4 = CFAbsoluteTimeGetCurrent()
        #endif

        // Mascot thruster audio
        for mascot in snapshot.mascots {
            eventPlayer.updateMascotThruster(
                projectId: mascot.projectId,
                position: mascot.position,
                volume: mascot.volume
            )
        }

        #if ENGRAM_INSTRUMENTATION
        let t5 = CFAbsoluteTimeGetCurrent()

        if audioTimingFile == nil {
            audioTimingFile = fopen("/tmp/audio-timing.csv", "w")
            if let f = audioTimingFile {
                fputs("frame,listener_ms,edge_ms,proximity_ms,events_ms,mascot_ms,total_ms\n", f)
            }
        }
        audioFrameCounter += 1
        if let f = audioTimingFile {
            let line = "\(audioFrameCounter),\(String(format: "%.3f", (t1 - t0) * 1000)),\(String(format: "%.3f", (t2 - t1) * 1000)),\(String(format: "%.3f", (t3 - t2) * 1000)),\(String(format: "%.3f", (t4 - t3) * 1000)),\(String(format: "%.3f", (t5 - t4) * 1000)),\(String(format: "%.3f", (t5 - t0) * 1000))\n"
            fputs(line, f)
            fflush(f)
        }
        #endif
    }

    // MARK: - Event Detection (runs on audioQueue)

    private func detectEvents(snapshot: AudioSceneSnapshot) {
        // Selection change
        if snapshot.selectedNode != lastSelectedNode {
            if let nodeId = snapshot.selectedNode, let pos = snapshot.positions[nodeId] {
                let audioPos = pos * snapshot.scaleFactor
                eventPlayer.playSelection(at: audioPos)
            } else if lastSelectedNode != nil {
                eventPlayer.playDeselection()
            }
            lastSelectedNode = snapshot.selectedNode
        }

        // Hub expansion
        let currentExpanded = snapshot.expandedHubs
        let newlyExpanded = currentExpanded.subtracting(lastExpandedHubs)
        let newlyCollapsed = lastExpandedHubs.subtracting(currentExpanded)
        for hubId in newlyExpanded {
            if let pos = snapshot.positions[hubId] {
                eventPlayer.playExpansion(at: pos * snapshot.scaleFactor)
            }
        }
        for hubId in newlyCollapsed {
            if let pos = snapshot.positions[hubId] {
                eventPlayer.playCollapse(at: pos * snapshot.scaleFactor)
            }
        }
        lastExpandedHubs = currentExpanded

        // Teleport
        if snapshot.teleportCounter != lastTeleportCounter {
            eventPlayer.playTeleport()
            lastTeleportCounter = snapshot.teleportCounter
        }
    }
}
