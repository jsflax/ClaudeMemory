import AVFoundation
import simd

/// Pool of spatialized voices dynamically assigned to the closest nodes.
/// Creates a pointillist texture as you fly through — stars singing as you pass.
///
/// Two-tier update:
/// - Rescan (6Hz): iterate all positions with simd_length_squared, bounded insertion — O(n)
/// - Voice update (60Hz): 8 dict lookups, 8 sqrts for volume — ~0.01ms
final class ProximityTonePool: @unchecked Sendable {

    private let environment: AVAudioEnvironmentNode
    private let poolSize: Int

    private struct Voice {
        let player: AVAudioPlayerNode
        var assignedNode: UUID?
        var targetVolume: Float = 0
    }

    private var voices: [Voice] = []
    /// Buffers keyed by frequency, pre-generated for each pentatonic note × 3 octaves.
    private var bufferCache: [Int: AVAudioPCMBuffer] = [:]

    private let maxDistance: Float = 300
    private let maxDistanceSq: Float = 300 * 300

    // --- Rescan throttle ---
    private var frameCounter: Int = 0
    private let rescanInterval: Int = 10 // every 10 frames at 60fps ≈ 6Hz

    // --- Cached results ---
    private struct NodeResult {
        let id: UUID
        let distanceSq: Float
    }
    private var cachedResults: [NodeResult] = []

    init(environment: AVAudioEnvironmentNode, poolSize: Int) {
        self.environment = environment
        self.poolSize = poolSize
    }

    func setup() {
        guard let eng = environment.engine else { return }

        // Pre-generate ping buffers for pentatonic notes across 3 octaves
        for (i, baseFreq) in AudioSynthesis.pentatonicFrequencies.enumerated() {
            for octave in 0..<3 {
                let freq = baseFreq * Float(1 << octave) // 1x, 2x, 4x
                let key = i * 3 + octave
                bufferCache[key] = AudioSynthesis.sinePing(
                    frequency: freq, duration: 1.2, amplitude: 0.3,
                    attack: 0.05, decay: 0.2, sustain: 0.2, release: 0.6
                )
            }
        }

        // Create voice pool
        for _ in 0..<poolSize {
            let player = AVAudioPlayerNode()
            player.renderingAlgorithm = .HRTFHQ
            player.volume = 0.0 // start silent
            player.reverbBlend = 0.3
            eng.attach(player)
            eng.connect(player, to: environment, format: AudioSynthesis.monoFormat)
            voices.append(Voice(player: player))
        }
    }

    // MARK: - Per-Frame Entry Point

    func update(
        positions: [UUID: SIMD3<Float>],
        cameraPosition: SIMD3<Float>,
        selectedNode: UUID?,
        glowingNodes: [UUID: Date],
        newNodes: [UUID: Date],
        scaleFactor: Float
    ) {
        guard !positions.isEmpty else { return }

        // Throttled rescan at ~6Hz
        frameCounter += 1
        if frameCounter >= rescanInterval {
            frameCounter = 0
            rescan(positions: positions, cameraPosition: cameraPosition, selectedNode: selectedNode)
        }

        // Every frame: update assigned voice positions from live data
        updateVoicePositions(
            positions: positions,
            cameraPosition: cameraPosition,
            glowingNodes: glowingNodes,
            newNodes: newNodes,
            scaleFactor: scaleFactor
        )
    }

    func stopAll() {
        for i in voices.indices {
            voices[i].player.stop()
            voices[i].assignedNode = nil
            voices[i].targetVolume = 0
        }
        cachedResults.removeAll()
    }

    // MARK: - Rescan (~6Hz)

    func rescan(
        positions: [UUID: SIMD3<Float>],
        cameraPosition: SIMD3<Float>,
        selectedNode: UUID?
    ) {
        cachedResults.removeAll(keepingCapacity: true)
        var worstDistSq: Float = 0
        var worstIdx: Int = 0

        for (id, pos) in positions {
            if id == selectedNode { continue }
            let distSq = simd_length_squared(pos - cameraPosition)
            guard distSq < maxDistanceSq else { continue }

            if cachedResults.count < poolSize {
                cachedResults.append(NodeResult(id: id, distanceSq: distSq))
                if distSq > worstDistSq {
                    worstDistSq = distSq
                    worstIdx = cachedResults.count - 1
                }
            } else if distSq < worstDistSq {
                // Swap-remove the worst, insert this one
                cachedResults[worstIdx] = NodeResult(id: id, distanceSq: distSq)
                // Recompute worst
                worstDistSq = 0
                for j in cachedResults.indices {
                    if cachedResults[j].distanceSq > worstDistSq {
                        worstDistSq = cachedResults[j].distanceSq
                        worstIdx = j
                    }
                }
            }
        }

        assignVoices()
    }

    // MARK: - Voice Assignment

    func assignVoices() {
        let resultIds = Set(cachedResults.map(\.id))

        // Free voices whose nodes left the closest set
        for i in voices.indices {
            guard let assigned = voices[i].assignedNode else { continue }
            if !resultIds.contains(assigned) {
                voices[i].assignedNode = nil
                voices[i].targetVolume = 0
            }
        }

        // Assign unvoiced nodes to free voices
        for result in cachedResults {
            if voices.contains(where: { $0.assignedNode == result.id }) { continue }
            guard let freeIdx = voices.firstIndex(where: { $0.assignedNode == nil }) else { break }

            voices[freeIdx].assignedNode = result.id

            // Choose buffer based on node characteristics
            let bufferKey = (result.id.hashValue & Int.max) % bufferCache.count
            if let buffer = bufferCache[bufferKey] {
                // Stop before reschedule (player may be paused with old buffer)
                voices[freeIdx].player.stop()
                voices[freeIdx].player.scheduleBuffer(buffer, at: nil, options: .loops)
                voices[freeIdx].player.play()
            }
        }
    }

    // MARK: - Per-Frame Voice Update (60Hz, O(poolSize))

    func updateVoicePositions(
        positions: [UUID: SIMD3<Float>],
        cameraPosition: SIMD3<Float>,
        glowingNodes: [UUID: Date],
        newNodes: [UUID: Date],
        scaleFactor: Float
    ) {
        for i in voices.indices {
            guard let nodeId = voices[i].assignedNode,
                  let pos = positions[nodeId] else {
                // Fade unassigned voices to silence, then pause to free HRTF processing
                let vol = voices[i].player.volume * 0.85
                voices[i].player.volume = vol
                if vol < 0.001 { voices[i].player.pause() }
                continue
            }

            let audioPos = pos * scaleFactor
            voices[i].player.position = AVAudio3DPoint(x: audioPos.x, y: audioPos.y, z: audioPos.z)

            // Volume based on distance (sqrt only for 8 assigned voices)
            let dist = simd_length(pos - cameraPosition)
            let normalizedDist = min(dist / maxDistance, 1.0)
            let distVolume = (1.0 - normalizedDist * normalizedDist) // inverse square falloff

            // Boost for glowing/arriving nodes
            let glowBoost: Float = glowingNodes[nodeId] != nil ? 1.5 : 1.0
            let arrivalBoost: Float = newNodes[nodeId] != nil ? 1.3 : 1.0

            voices[i].targetVolume = min(0.6, distVolume * 0.35 * glowBoost * arrivalBoost)

            // Smooth volume transition
            let current = voices[i].player.volume
            voices[i].player.volume = current + (voices[i].targetVolume - current) * 0.15
        }
    }
}
