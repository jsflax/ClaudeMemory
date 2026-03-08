import AVFoundation
import simd

/// Pool of spatialized voices dynamically assigned to the closest nodes.
/// Creates a pointillist texture as you fly through — stars singing as you pass.
@MainActor
final class ProximityTonePool {

    private let environment: AVAudioEnvironmentNode
    private let poolSize: Int

    private struct Voice {
        let player: AVAudioPlayerNode
        var assignedNode: UUID?
        var isPlaying: Bool = false
    }

    private var voices: [Voice] = []
    /// Buffers keyed by frequency, pre-generated for each pentatonic note × 3 octaves.
    private var bufferCache: [Int: AVAudioPCMBuffer] = [:]
    /// Hysteresis: a new node must be this much closer than the current assignment to steal a voice.
    private let hysteresisMargin: Float = 30.0

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

    func update(
        positions: [UUID: SIMD3<Float>],
        cameraPosition: SIMD3<Float>,
        selectedNode: UUID?,
        glowingNodes: [UUID: Date],
        newNodes: [UUID: Date],
        scaleFactor: Float
    ) {
        guard !positions.isEmpty else { return }

        // Find N closest nodes (excluding selected — that gets its own event sound)
        let camPos = cameraPosition
        let maxDistance: Float = 300 // world units — beyond this, no proximity tone

        struct NodeDist: Comparable {
            let id: UUID
            let distance: Float
            static func < (lhs: NodeDist, rhs: NodeDist) -> Bool { lhs.distance < rhs.distance }
        }

        var candidates: [NodeDist] = []
        candidates.reserveCapacity(min(positions.count, poolSize * 4))

        for (id, pos) in positions {
            if id == selectedNode { continue }
            let dist = simd_length(pos - camPos)
            if dist < maxDistance {
                candidates.append(NodeDist(id: id, distance: dist))
            }
        }

        candidates.sort()
        let closest = Array(candidates.prefix(poolSize))

        // Assign voices to closest nodes
        let closestIds = Set(closest.map(\.id))
        let currentAssignments = Set(voices.compactMap(\.assignedNode))

        // First pass: free voices whose nodes are no longer in closest set
        for i in voices.indices {
            if let assigned = voices[i].assignedNode, !closestIds.contains(assigned) {
                voices[i].assignedNode = nil
                voices[i].player.volume = 0
            }
        }

        // Second pass: assign unassigned closest nodes to free voices
        for nd in closest {
            // Already assigned?
            if voices.contains(where: { $0.assignedNode == nd.id }) { continue }

            // Find a free voice
            guard let freeIdx = voices.firstIndex(where: { $0.assignedNode == nil }) else { break }

            voices[freeIdx].assignedNode = nd.id

            // Choose buffer based on node characteristics
            let bufferKey = (nd.id.hashValue & Int.max) % bufferCache.count
            if let buffer = bufferCache[bufferKey] {
                voices[freeIdx].player.stop()
                voices[freeIdx].player.scheduleBuffer(buffer, at: nil, options: .loops)
                voices[freeIdx].player.play()
                voices[freeIdx].isPlaying = true
            }
        }

        // Update positions and volume for all assigned voices
        for i in voices.indices {
            guard let nodeId = voices[i].assignedNode,
                  let pos = positions[nodeId] else { continue }

            let audioPos = pos * scaleFactor
            voices[i].player.position = AVAudio3DPoint(x: audioPos.x, y: audioPos.y, z: audioPos.z)

            // Volume based on distance
            let dist = simd_length(pos - camPos)
            let normalizedDist = min(dist / maxDistance, 1.0)
            let distVolume = (1.0 - normalizedDist * normalizedDist) // inverse square falloff

            // Boost for glowing/arriving nodes
            let glowBoost: Float = glowingNodes[nodeId] != nil ? 1.5 : 1.0
            let arrivalBoost: Float = newNodes[nodeId] != nil ? 1.3 : 1.0

            voices[i].player.volume = min(0.6, distVolume * 0.35 * glowBoost * arrivalBoost)
        }
    }

    func stopAll() {
        for i in voices.indices {
            voices[i].player.stop()
            voices[i].isPlaying = false
            voices[i].assignedNode = nil
        }
    }
}
