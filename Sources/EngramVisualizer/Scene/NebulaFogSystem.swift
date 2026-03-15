import Metal
import CEngramSceneTypes
import simd
import SwiftUI

/// Manages GPU fog billboard quads for nebula effects.
/// Replaces RealityKit's ParticleEmitterComponent with procedural fBm noise in fragment shader.
@MainActor
final class NebulaFogSystem {

    private let device: MTLDevice
    private let scaleFactor: Float = 1.0 / 200.0

    // Per-cluster: 3 billboard quads at different depth offsets for parallax
    private let quadsPerCluster = 3

    struct NebulaGroup {
        let key: String
        let centroid: SIMD3<Float>
        let radius: Float
        let project: String
    }

    init(device: MTLDevice) {
        self.device = device
    }

    /// Build nebula groups from current render state.
    func nebulaGroupsForCurrentMode(
        layoutMode: LayoutMode,
        positions: [UUID: SIMD3<Float>],
        nodes: [NodeData],
        semanticClusters3D: [SemanticCluster3D]
    ) -> [NebulaGroup] {
        guard !positions.isEmpty else { return [] }

        var groups: [NebulaGroup] = []

        if layoutMode == .embedding {
            for cluster in semanticClusters3D {
                let memberPositions = cluster.nodeIds.compactMap { positions[$0] }
                guard memberPositions.count >= 3 else { continue }
                var sum = SIMD3<Float>.zero
                for p in memberPositions { sum += p }
                let centroid = sum / Float(memberPositions.count)
                var maxDist: Float = 0
                for p in memberPositions { maxDist = max(maxDist, simd_length(p - centroid)) }
                let dominantProject = cluster.projectBreakdown.first?.project ?? "global"
                groups.append(NebulaGroup(
                    key: "sem_\(cluster.id)", centroid: centroid, radius: maxDist + 15,
                    project: dominantProject
                ))
            }
        } else {
            var projectNodes: [String: [SIMD3<Float>]] = [:]
            for node in nodes {
                guard let pos = positions[node.id] else { continue }
                projectNodes[node.project, default: []].append(pos)
            }
            for (project, pts) in projectNodes where pts.count >= 2 {
                var sum = SIMD3<Float>.zero
                for p in pts { sum += p }
                let centroid = sum / Float(pts.count)
                var maxDist: Float = 0
                for p in pts { maxDist = max(maxDist, simd_length(p - centroid)) }
                groups.append(NebulaGroup(
                    key: "proj_\(project)", centroid: centroid, radius: maxDist + 40,
                    project: project
                ))
            }
        }

        return groups
    }

    /// Pack nebula billboard vertices for the GPU.
    /// Returns (vertex data, vertex count) to be uploaded to the nebula vertex buffer.
    func packNebulaVertices(
        groups: [NebulaGroup],
        colorMap: [String: Color]
    ) -> [NebulaQuadVertex] {
        var vertices: [NebulaQuadVertex] = []
        vertices.reserveCapacity(groups.count * quadsPerCluster * 4)

        for group in groups {
            let R = max(group.radius, 30.0) * scaleFactor
            let worldCentroid = group.centroid * scaleFactor

            // Get project color
            let color = colorMap[group.project] ?? .gray
            let nsColor = NSColor(color).usingColorSpace(.sRGB) ?? NSColor(color)
            var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
            nsColor.getRed(&r, green: &g, blue: &b, alpha: &a)
            let tint = SIMD4<Float>(
                Float(min(1.0, r * 1.3 + 0.1)),
                Float(min(1.0, g * 1.3 + 0.1)),
                Float(min(1.0, b * 1.3 + 0.1)),
                0.25  // per-quad alpha
            )

            // 3 quads per cluster at different depth offsets
            let depthOffsets: [Float] = [-0.3, 0.0, 0.3]
            let layerWeights: [Float] = [0.7, 1.0, 0.6]

            for (layerIdx, depthZ) in depthOffsets.enumerated() {
                let noisePhase = Float(group.key.hashValue & 0xFFFF) / Float(0xFFFF) * 10.0 + Float(layerIdx) * 3.7
                let quadAlpha = tint.w * layerWeights[layerIdx]

                // Quad expansion size
                let halfSize = R * 1.2

                // 4 corners: TL, TR, BL, BR
                let corners: [(SIMD3<Float>, SIMD2<Float>)] = [
                    (SIMD3(-halfSize, halfSize, depthZ * R), SIMD2(0, 0)),   // TL
                    (SIMD3(halfSize, halfSize, depthZ * R), SIMD2(1, 0)),    // TR
                    (SIMD3(-halfSize, -halfSize, depthZ * R), SIMD2(0, 1)),  // BL
                    (SIMD3(halfSize, -halfSize, depthZ * R), SIMD2(1, 1)),   // BR
                ]

                for (offset, uv) in corners {
                    vertices.append(NebulaQuadVertex(
                        position: worldCentroid,
                        cornerOffset: offset,
                        uv: uv,
                        color: SIMD4(tint.x, tint.y, tint.z, quadAlpha),
                        noisePhase: noisePhase,
                        radius: R,
                        _pad0: 0, _pad1: 0
                    ))
                }
            }
        }

        return vertices
    }

    /// Upload nebula vertices and update index buffer on the renderer.
    func updateRenderer(
        renderer: MetalGraphRenderer,
        groups: [NebulaGroup],
        colorMap: [String: Color]
    ) {
        let vertices = packNebulaVertices(groups: groups, colorMap: colorMap)
        let vertexCount = vertices.count
        let quadCount = vertexCount / 4

        guard quadCount > 0 else {
            renderer.actualNebulaVertexCount = 0
            return
        }

        // Ensure buffers
        if renderer.nebulaVertexCapacity < quadCount {
            let cap = max(quadCount * 2, 64)
            renderer.nebulaVertexBuffer = device.makeBuffer(
                length: cap * 4 * MemoryLayout<NebulaQuadVertex>.stride,
                options: .storageModeShared
            )
            // Index buffer for quads
            let indexCount = cap * 6
            renderer.nebulaIndexBuffer = device.makeBuffer(
                length: indexCount * MemoryLayout<UInt32>.stride,
                options: .storageModeShared
            )
            if let buf = renderer.nebulaIndexBuffer {
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
            renderer.nebulaVertexCapacity = cap
        }

        // Upload vertices
        if let buf = renderer.nebulaVertexBuffer {
            vertices.withUnsafeBufferPointer { ptr in
                guard let base = ptr.baseAddress else { return }
                buf.contents().copyMemory(
                    from: base,
                    byteCount: vertexCount * MemoryLayout<NebulaQuadVertex>.stride
                )
            }
        }

        renderer.actualNebulaVertexCount = vertexCount
    }
}
