import Metal
import CEngramSceneTypes
import simd
import SwiftUI
import EngramSceneKit

/// Packs nebula fog data each frame.
/// Extracted from MetalSceneManager to keep that file focused on orchestration.
@MainActor
final class NebulaPackingStage {

    // Cached nebula color conversion — only rebuild when color map changes
    private var cachedNebulaColors: [String: SIMD4<Float>] = [:]
    private var lastNebulaColorMapVersion: UInt64 = 0

    func update(
        device: MTLDevice,
        nebulaFog: NebulaFogSystem,
        bufferManager: InstanceBufferManager,
        positions: [UUID: SIMD3<Float>],
        nodes: [NodeData],
        renderColorMap: [String: Color],
        semanticClusters3D: [SemanticCluster3D],
        layoutMode: LayoutMode,
        scaleFactor: Float,
        projectCentroids: [String: (centroid: SIMD3<Float>, radius: Float, maxY: Float, count: Int)],
        colorMapVersion: UInt64
    ) {
        // Rebuild nebula color cache when color map changes (avoids per-frame NSColor conversion)
        if colorMapVersion != lastNebulaColorMapVersion {
            cachedNebulaColors.removeAll(keepingCapacity: true)
            for (project, color) in renderColorMap {
                let nsColor = NSColor(color).usingColorSpace(.sRGB) ?? NSColor(color)
                var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
                nsColor.getRed(&r, green: &g, blue: &b, alpha: &a)
                cachedNebulaColors[project] = SIMD4<Float>(
                    Float(min(1.0, r * 1.3 + 0.1)),
                    Float(min(1.0, g * 1.3 + 0.1)),
                    Float(min(1.0, b * 1.3 + 0.1)),
                    0.25
                )
            }
            lastNebulaColorMapVersion = colorMapVersion
        }

        if layoutMode == .embedding {
            // Embedding mode: use semantic clusters (fall back to original CPU path)
            let groups = nebulaFog.nebulaGroupsForCurrentMode(
                layoutMode: layoutMode, positions: positions,
                nodes: nodes, semanticClusters3D: semanticClusters3D
            )
            nebulaFog.updateRenderer(bufferManager: bufferManager, groups: groups, colorMap: renderColorMap)
            bufferManager.nebulaGroupCount = 0  // disable GPU nebula path
            return
        }

        // Force-directed mode: use projectCentroids from packNodeInstances (saves O(N) recomputation).
        // Build NebulaGroupInput[] using the pure builder.
        var nebulaCentroids: [String: (centroid: SIMD3<Float>, maxY: Float, count: Int)] = [:]
        for (project, data) in projectCentroids {
            nebulaCentroids[project] = (centroid: data.centroid, maxY: data.maxY, count: data.count)
        }
        let groupInputs = buildNebulaGroups(
            projectCentroids: nebulaCentroids,
            projectColors: cachedNebulaColors,
            scaleFactor: scaleFactor
        )

        let groupCount = groupInputs.count
        guard groupCount > 0 else {
            bufferManager.actualNebulaVertexCount = 0
            bufferManager.nebulaGroupCount = 0
            return
        }

        // Ensure GPU buffers
        bufferManager.ensureNebulaGroupInputBuffer(count: groupCount)
        let quadCount = groupCount * 3  // 3 quads per group
        if bufferManager.nebulaVertexCapacity < quadCount {
            let cap = max(quadCount * 2, 64)
            bufferManager.nebulaVertexBuffer = device.makeBuffer(
                length: cap * 4 * MemoryLayout<NebulaQuadVertex>.stride,
                options: .storageModeShared
            )
            // Index buffer for quads
            let indexCount = cap * 6
            bufferManager.nebulaIndexBuffer = device.makeBuffer(
                length: indexCount * MemoryLayout<UInt32>.stride,
                options: .storageModeShared
            )
            if let buf = bufferManager.nebulaIndexBuffer {
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
            bufferManager.nebulaVertexCapacity = cap
        }

        // Upload group inputs to GPU buffer
        if let inputBuf = bufferManager.nebulaGroupInputBuffer {
            groupInputs.withUnsafeBufferPointer { ptr in
                guard let base = ptr.baseAddress else { return }
                inputBuf.contents().copyMemory(
                    from: base,
                    byteCount: groupCount * MemoryLayout<NebulaGroupInput>.stride
                )
            }
        }

        // Set pack params
        if let paramsBuf = bufferManager.nebulaPackParamsBuffer {
            let paramsPtr = paramsBuf.contents().bindMemory(to: NebulaPackParams.self, capacity: 1)
            paramsPtr.pointee = NebulaPackParams(
                groupCount: UInt32(groupCount),
                quadsPerGroup: 3,
                _pad0: 0, _pad1: 0
            )
        }

        bufferManager.nebulaGroupCount = groupCount
        bufferManager.actualNebulaVertexCount = groupCount * 3 * 4  // 3 quads x 4 vertices each
    }
}
