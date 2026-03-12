import AVFoundation
import simd

/// Pool of spatialized drone voices positioned at the nearest edges.
/// Uses adjacency-based lookup + explicit bridge edge index to avoid
/// scanning all 29K edges every rescan.
///
/// Two-tier candidate discovery:
/// - Tier 1: Adjacency dict — nearby nodes → their edges → score with nearestPointOnSegment
/// - Tier 2: Bridge edges (span > 1500 units) — always checked, never filtered by node proximity
///
/// Performance (29K edges, 3500 nodes):
/// - Topology rebuild: ~1ms on edge add/remove (dict insert)
/// - Rescan (6Hz): ~0.3ms — 3500 nodes + ~850 candidates + ~50 bridges
/// - Voice update (60Hz): ~0.01ms — 10 voices, 20 dict lookups
final class EdgeHumPool: @unchecked Sendable {

    private let environment: AVAudioEnvironmentNode
    private let poolSize: Int

    private struct Voice {
        let player: AVAudioPlayerNode
        var assignedEdge: UUID?
        var assignedRelation: String = ""
        var targetVolume: Float = 0
    }

    private var voices: [Voice] = []

    private var buffersByRelation: [String: AVAudioPCMBuffer] = [:]
    private var defaultBuffer: AVAudioPCMBuffer?

    private let maxDistance: Float = 500
    private let maxDistanceSq: Float = 500 * 500

    // --- Adjacency index (Tier 1) ---
    private struct Neighbor {
        let edgeId: UUID
        let peerId: UUID
        let relation: String
    }
    private var neighbors: [UUID: [Neighbor]] = [:]

    // --- Bridge edge index (Tier 2) ---
    private struct BridgeEdge {
        let edgeId: UUID
        let sourceId: UUID
        let targetId: UUID
        let relation: String
    }
    private var bridgeEdges: [BridgeEdge] = []
    private let bridgeThreshold: Float = 1500 // units — roughly half galaxy spacing

    private(set) var lastTopologyVersion: UInt64 = 0

    // --- Rescan throttle ---
    private var frameCounter: Int = 0
    private let rescanInterval: Int = 10 // every 10 frames at 60fps = 6Hz

    // --- Cached results for voice assignment ---
    private struct EdgeResult {
        let edgeId: UUID
        let relation: String
        let sourceId: UUID
        let targetId: UUID
        let distanceSq: Float
    }
    private var cachedResults: [EdgeResult] = []

    init(environment: AVAudioEnvironmentNode, poolSize: Int = 10) {
        self.environment = environment
        self.poolSize = poolSize
    }

    func setup() {
        guard let eng = environment.engine else { return }

        let relations: [(name: String, freq: Float, harmonicColor: Float)] = [
            ("relates_to",    130.81, 0.3),
            ("part_of",       146.83, 0.15),
            ("derived_from",  164.81, 0.25),
            ("supersedes",    196.00, 0.4),
            ("contradicts",   110.00, 0.6),
            ("summarized_by", 174.61, 0.2),
        ]

        for rel in relations {
            if let buf = Self.edgeDrone(frequency: rel.freq, harmonicColor: rel.harmonicColor) {
                buffersByRelation[rel.name] = buf
            }
        }

        defaultBuffer = Self.edgeDrone(frequency: 155.56, harmonicColor: 0.25)

        for _ in 0..<poolSize {
            let player = AVAudioPlayerNode()
            player.renderingAlgorithm = .HRTFHQ
            player.volume = 0.0
            player.reverbBlend = 0.45
            eng.attach(player)
            eng.connect(player, to: environment, format: AudioSynthesis.monoFormat)
            voices.append(Voice(player: player))
        }
    }

    // MARK: - Topology Rebuild (on edge add/remove only)

    func rebuildTopology(edges: [EdgeData], positions: [UUID: SIMD3<Float>], version: UInt64) {
        neighbors.removeAll(keepingCapacity: true)
        bridgeEdges.removeAll(keepingCapacity: true)

        let bridgeThresholdSq = bridgeThreshold * bridgeThreshold

        for edge in edges {
            // Build adjacency: both endpoints → this edge
            let fwd = Neighbor(edgeId: edge.id, peerId: edge.targetId, relation: edge.relation)
            let rev = Neighbor(edgeId: edge.id, peerId: edge.sourceId, relation: edge.relation)
            neighbors[edge.sourceId, default: []].append(fwd)
            neighbors[edge.targetId, default: []].append(rev)

            // Identify bridge edges by endpoint distance
            if let srcPos = positions[edge.sourceId],
               let tgtPos = positions[edge.targetId] {
                let spanSq = simd_length_squared(tgtPos - srcPos)
                if spanSq > bridgeThresholdSq {
                    bridgeEdges.append(BridgeEdge(
                        edgeId: edge.id, sourceId: edge.sourceId,
                        targetId: edge.targetId, relation: edge.relation
                    ))
                }
            }
        }

        lastTopologyVersion = version
    }

    // MARK: - Per-Frame Entry Point

    func update(
        positions: [UUID: SIMD3<Float>],
        cameraPosition: SIMD3<Float>,
        scaleFactor: Float
    ) {
        guard !neighbors.isEmpty, !positions.isEmpty else {
            fadeAllToSilence()
            return
        }

        // Throttled rescan at ~6Hz
        frameCounter += 1
        if frameCounter >= rescanInterval {
            frameCounter = 0
            rescan(positions: positions, cameraPosition: cameraPosition)
        }

        // Every frame: update assigned voice positions from live data
        updateVoicePositions(
            positions: positions,
            cameraPosition: cameraPosition,
            scaleFactor: scaleFactor
        )
    }

    func stopAll() {
        for i in voices.indices {
            voices[i].player.stop()
            voices[i].assignedEdge = nil
            voices[i].targetVolume = 0
        }
        cachedResults.removeAll()
    }

    // MARK: - Rescan (~6Hz)

    private func rescan(positions: [UUID: SIMD3<Float>], cameraPosition: SIMD3<Float>) {
        // Tier 1: Find nearby nodes, collect their edges via adjacency
        var seen = Set<UUID>()
        var candidates: [EdgeResult] = []

        for (nodeId, pos) in positions {
            let distSq = simd_length_squared(pos - cameraPosition)
            guard distSq < maxDistanceSq else { continue }

            // This node is nearby — collect all its edges
            guard let adj = neighbors[nodeId] else { continue }
            for neighbor in adj {
                guard seen.insert(neighbor.edgeId).inserted else { continue }
                guard let tgtPos = positions[neighbor.peerId] else { continue }

                let nearest = Self.nearestPointOnSegment(pos, tgtPos, to: cameraPosition)
                let nearDistSq = simd_length_squared(nearest - cameraPosition)
                guard nearDistSq < maxDistanceSq else { continue }

                candidates.append(EdgeResult(
                    edgeId: neighbor.edgeId, relation: neighbor.relation,
                    sourceId: nodeId, targetId: neighbor.peerId,
                    distanceSq: nearDistSq
                ))
            }
        }

        // Tier 2: Always check bridge edges regardless of node proximity
        for bridge in bridgeEdges {
            guard seen.insert(bridge.edgeId).inserted else { continue }
            guard let srcPos = positions[bridge.sourceId],
                  let tgtPos = positions[bridge.targetId] else { continue }

            let nearest = Self.nearestPointOnSegment(srcPos, tgtPos, to: cameraPosition)
            let nearDistSq = simd_length_squared(nearest - cameraPosition)
            guard nearDistSq < maxDistanceSq else { continue }

            candidates.append(EdgeResult(
                edgeId: bridge.edgeId, relation: bridge.relation,
                sourceId: bridge.sourceId, targetId: bridge.targetId,
                distanceSq: nearDistSq
            ))
        }

        // Keep top N by distance
        if candidates.count > poolSize {
            candidates.sort { $0.distanceSq < $1.distanceSq }
            candidates = Array(candidates.prefix(poolSize))
        }

        cachedResults = candidates
        assignVoices()
    }

    // MARK: - Voice Assignment

    private func assignVoices() {
        let resultIds = Set(cachedResults.map(\.edgeId))

        // Free voices whose edges left the closest set
        for i in voices.indices {
            guard let assigned = voices[i].assignedEdge else { continue }
            if !resultIds.contains(assigned) {
                voices[i].assignedEdge = nil
                voices[i].targetVolume = 0
            }
        }

        // Assign unvoiced edges to free voices
        for result in cachedResults {
            if voices.contains(where: { $0.assignedEdge == result.edgeId }) { continue }
            guard let freeIdx = voices.firstIndex(where: { $0.assignedEdge == nil }) else { break }

            let relation = result.relation
            if voices[freeIdx].assignedRelation != relation {
                let buffer = buffersByRelation[relation] ?? defaultBuffer
                guard let buffer else { continue }
                voices[freeIdx].player.stop()
                voices[freeIdx].player.scheduleBuffer(buffer, at: nil, options: .loops)
                voices[freeIdx].player.play()
                voices[freeIdx].assignedRelation = relation
            }

            voices[freeIdx].assignedEdge = result.edgeId
        }
    }

    // MARK: - Per-Frame Voice Update (60Hz, O(poolSize))

    private func updateVoicePositions(
        positions: [UUID: SIMD3<Float>],
        cameraPosition: SIMD3<Float>,
        scaleFactor: Float
    ) {
        for i in voices.indices {
            guard let edgeId = voices[i].assignedEdge else {
                // Fade unassigned voices to silence, then pause to free HRTF processing
                let vol = voices[i].player.volume * 0.85
                voices[i].player.volume = vol
                if vol < 0.001 { voices[i].player.pause() }
                continue
            }

            var found = false
            for result in cachedResults where result.edgeId == edgeId {
                guard let srcPos = positions[result.sourceId],
                      let tgtPos = positions[result.targetId] else { break }

                let nearest = Self.nearestPointOnSegment(srcPos, tgtPos, to: cameraPosition)
                let audioPos = nearest * scaleFactor
                voices[i].player.position = AVAudio3DPoint(x: audioPos.x, y: audioPos.y, z: audioPos.z)

                let dist = simd_length(nearest - cameraPosition)
                let normalizedDist = min(dist / maxDistance, 1.0)
                voices[i].targetVolume = (1.0 - normalizedDist * normalizedDist) * 0.3

                found = true
                break
            }

            if !found {
                voices[i].assignedEdge = nil
                voices[i].targetVolume = 0
            }

            let current = voices[i].player.volume
            voices[i].player.volume = current + (voices[i].targetVolume - current) * 0.15
        }
    }

    private func fadeAllToSilence() {
        for i in voices.indices {
            voices[i].targetVolume = 0
            let vol = voices[i].player.volume * 0.85
            voices[i].player.volume = vol
            if vol < 0.001 { voices[i].player.pause() }
        }
    }

    // MARK: - Geometry

    private static func nearestPointOnSegment(
        _ a: SIMD3<Float>, _ b: SIMD3<Float>, to p: SIMD3<Float>
    ) -> SIMD3<Float> {
        let ab = b - a
        let lengthSq = simd_length_squared(ab)
        guard lengthSq > 1e-8 else { return a }
        let t = simd_clamp(simd_dot(p - a, ab) / lengthSq, 0, 1)
        return a + t * ab
    }

    // MARK: - Buffer Synthesis

    private static func edgeDrone(
        frequency: Float,
        harmonicColor: Float,
        duration: Float = 3.0,
        amplitude: Float = 0.2
    ) -> AVAudioPCMBuffer? {
        let sr = Float(AudioSynthesis.sampleRate)
        let frameCount = AVAudioFrameCount(Double(duration) * AudioSynthesis.sampleRate)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: AudioSynthesis.monoFormat, frameCapacity: frameCount) else { return nil }
        buffer.frameLength = frameCount
        guard let data = buffer.floatChannelData?[0] else { return nil }

        let partials: [(harmonic: Float, baseAmp: Float)] = [
            (1.0, 1.0),
            (2.0, 0.5),
            (3.0, 0.25),
            (4.0, 0.12),
            (5.0, 0.06),
        ]

        let lfoRates: [Float] = [0.07, 0.11, 0.13, 0.17, 0.19]

        for i in 0..<Int(frameCount) {
            let fi = Float(i)
            var sample: Float = 0

            for (pIdx, p) in partials.enumerated() {
                let harmonicGate: Float = pIdx == 0 ? 1.0 : min(1.0, harmonicColor * Float(4 - min(pIdx, 3)))
                let omega = 2.0 * Float.pi * frequency * p.harmonic / sr
                let lfo = 0.7 + 0.3 * sin(fi * 2.0 * Float.pi * lfoRates[pIdx] / sr)
                sample += p.baseAmp * harmonicGate * lfo * sin(fi * omega)
            }

            data[i] = amplitude * sample
        }

        return buffer
    }
}
