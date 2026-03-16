import Metal
import CEngramSceneTypes
import EngramSceneKit
import simd
import SwiftUI

/// Manages batched flow particle billboard quads for edge traversal animation.
/// Replaces RealityKit ModelEntity-based particles with GPU-instanced billboards.
@MainActor
final class FlowParticleSystem {

    private let device: MTLDevice
    private let scaleFactor: Float = 1.0 / 200.0
    private let particleSize: Float = 0.006

    struct Particle {
        var t: Float         // 0..1 progress along edge
        let speed: Float
        let edgeKey: String
        let sourcePos: SIMD3<Float>  // scaled world pos
        let targetPos: SIMD3<Float>  // scaled world pos
        let color: SIMD3<Float>
    }

    private var particles: [Particle] = []
    private var spawnTimers: [String: Float] = [:]
    private var lastSelectedNode: UUID?

    init(device: MTLDevice) {
        self.device = device
    }

    func flowColorForRelation(_ relation: String) -> SIMD3<Float> {
        switch relation {
        case "part_of":       return SIMD3(0.3, 0.6, 1.0)
        case "contradicts":   return SIMD3(1.0, 0.2, 0.2)
        case "supersedes":    return SIMD3(1.0, 0.6, 0.15)
        case "derived_from":  return SIMD3(0.65, 0.3, 1.0)
        case "summarized_by": return SIMD3(0.2, 0.9, 0.5)
        default:              return SIMD3(0.8, 0.8, 0.9)
        }
    }

    /// Update flow particles and pack vertices for GPU rendering.
    func update(
        dt: Float,
        selectedNode: UUID?,
        edges: [EdgeData],
        positions: [UUID: SIMD3<Float>],
        expandedHubs: Set<UUID>,
        bufferManager: InstanceBufferManager
    ) {
        // Selection change → flush all particles
        if selectedNode != lastSelectedNode {
            particles.removeAll()
            spawnTimers.removeAll()
            lastSelectedNode = selectedNode
        }

        // Determine active edges
        var activeEdges: [EdgeData] = []
        for edge in edges {
            let isConnected = (selectedNode != nil) && (edge.sourceId == selectedNode || edge.targetId == selectedNode)
            let isExpanded = edge.relation == "part_of" && expandedHubs.contains(edge.targetId)
            if isConnected || isExpanded {
                activeEdges.append(edge)
            }
        }
        if activeEdges.count > 30 {
            activeEdges = Array(activeEdges.prefix(30))
        }

        let activeKeys = Set(activeEdges.map { "\($0.sourceId)-\($0.targetId)" })

        // Remove particles for inactive edges
        particles.removeAll { !activeKeys.contains($0.edgeKey) }
        for key in spawnTimers.keys where !activeKeys.contains(key) {
            spawnTimers.removeValue(forKey: key)
        }

        // Spawn + advance
        for edge in activeEdges {
            let key = "\(edge.sourceId)-\(edge.targetId)"
            guard let p1 = positions[edge.sourceId],
                  let p2 = positions[edge.targetId] else { continue }

            let scaledP1 = p1 * scaleFactor
            let scaledP2 = p2 * scaleFactor
            let color = flowColorForRelation(edge.relation)

            // Spawn timer
            var timer = spawnTimers[key] ?? 0
            timer += dt
            if timer >= 0.4 {
                timer -= 0.4
                particles.append(Particle(
                    t: 0, speed: 0.8,
                    edgeKey: key,
                    sourcePos: scaledP1, targetPos: scaledP2,
                    color: color
                ))
            }
            spawnTimers[key] = timer
        }

        // Advance existing particles
        var toRemove: [Int] = []
        for i in particles.indices {
            particles[i].t += dt * particles[i].speed
            if particles[i].t >= 1.0 {
                toRemove.append(i)
            }
        }
        for i in toRemove.reversed() {
            particles.remove(at: i)
        }

        // Pack into FlowParticleVertex buffer
        let particleCount = particles.count
        guard particleCount > 0 else {
            bufferManager.actualFlowParticleCount = 0
            return
        }

        // Ensure buffers
        if bufferManager.flowVertexCapacity < particleCount {
            let cap = max(particleCount * 2, 64)
            bufferManager.flowVertexBuffer = device.makeBuffer(
                length: cap * MemoryLayout<FlowParticleVertex>.stride,
                options: .storageModeShared
            )
            let indexCount = cap * 6
            bufferManager.flowIndexBuffer = device.makeBuffer(
                length: indexCount * MemoryLayout<UInt32>.stride,
                options: .storageModeShared
            )
            if let buf = bufferManager.flowIndexBuffer {
                let indices = buf.contents().bindMemory(to: UInt32.self, capacity: indexCount)
                for i in 0..<cap {
                    let vBase = UInt32(i * 4)
                    let iBase = i * 6
                    indices[iBase]     = vBase
                    indices[iBase + 1] = vBase + 2
                    indices[iBase + 2] = vBase + 1
                    indices[iBase + 3] = vBase + 1
                    indices[iBase + 4] = vBase + 2
                    indices[iBase + 5] = vBase + 3
                }
            }
            bufferManager.flowVertexCapacity = cap
        }

        // Pack one FlowParticleVertex per particle (vertex shader expands to 4 corners)
        if let buf = bufferManager.flowVertexBuffer {
            let ptr = buf.contents().bindMemory(to: FlowParticleVertex.self, capacity: particleCount)
            for (i, p) in particles.enumerated() {
                let pos = p.sourcePos + (p.targetPos - p.sourcePos) * p.t

                let opacity: Float
                if p.t < 0.1 { opacity = p.t / 0.1 }
                else if p.t > 0.85 { opacity = (1.0 - p.t) / 0.15 }
                else { opacity = 1.0 }

                ptr[i] = FlowParticleVertex(
                    position: pos,
                    uv: SIMD2(0, 0),  // unused — vertex shader sets per-corner
                    color: SIMD4(p.color.x, p.color.y, p.color.z, opacity),
                    size: particleSize,
                    _pad0: 0, _pad1: 0, _pad2: 0
                )
            }
        }

        bufferManager.actualFlowParticleCount = particleCount
    }
}
