import SwiftUI
import RealityKit
import simd
import os

private let frameLog = Logger(subsystem: "com.claudememory.visualizer", category: "3DFrameTiming")

extension Graph3DScene {

    // MARK: - Nebulae (particle-based gaseous hulls)

    struct NebulaGroup {
        let key: String
        let centroid: SIMD3<Float>
        let radius: Float
        let project: String
    }

    func nebulaColors(for project: String, colorMap: [String: Color]) -> (start: NSColor, end: NSColor) {
        if let cached = nebulaColorCache[project] { return cached }
        let color = colorMap[project] ?? .gray
        let nsColor = NSColor(color).usingColorSpace(.sRGB) ?? NSColor(color)
        var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
        nsColor.getRed(&r, green: &g, blue: &b, alpha: &a)
        // Start: visible but subtle. End: barely there.
        // Lower per-particle alpha — more particles, each nearly invisible individually,
        // but accumulating into a smooth colored fog. Quartic texture falloff helps blend.
        let start = NSColor(
            red: min(1.0, r * 1.3 + 0.1),
            green: min(1.0, g * 1.3 + 0.1),
            blue: min(1.0, b * 1.3 + 0.1),
            alpha: 0.035
        )
        let end = NSColor(
            red: min(1.0, r * 0.7 + 0.05),
            green: min(1.0, g * 0.7 + 0.05),
            blue: min(1.0, b * 0.7 + 0.05),
            alpha: 0.005
        )
        let result = (start: start, end: end)
        nebulaColorCache[project] = result
        return result
    }

    func nebulaGroupsForCurrentMode() -> [NebulaGroup] {
        let positions = renderPositions
        guard !positions.isEmpty else { return [] }

        let isSemanticMode = renderLayoutMode == .embedding

        var groups: [NebulaGroup] = []

        if isSemanticMode {
            for cluster in renderSemanticClusters3D {
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
            // Per-project grouping — every project with 2+ nodes gets a nebula.
            var projectNodes: [String: [SIMD3<Float>]] = [:]
            for node in renderNodes {
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

    /// Build a ParticleEmitterComponent configured as a gaseous nebula.
    func makeNebulaEmitter(radius: Float, startColor: NSColor, endColor: NSColor) -> ParticleEmitterComponent {
        var emitter = ParticleEmitterComponent()
        let scaledR = radius * scaleFactor

        // Sphere volume — generously larger than cluster for soft boundary
        emitter.emitterShape = .sphere
        emitter.birthLocation = .volume
        emitter.emitterShapeSize = SIMD3<Float>(repeating: scaledR * 2.4)

        // Nearly stationary — particles hover in place like fog
        emitter.speed = 0.0001
        emitter.speedVariation = 0.00005

        // Warm-up matches lifespan for full steady-state on first frame
        emitter.timing = .repeating(
            warmUp: 15.0,
            emit: .init(duration: .infinity, variation: nil),
            idle: nil
        )

        // More particles, each nearly invisible — smooth accumulation hides circles
        emitter.mainEmitter.birthRate = max(5, min(15, radius * 0.05))
        emitter.mainEmitter.lifeSpan = 15.0
        emitter.mainEmitter.lifeSpanVariation = 5.0

        // 55% of cluster radius — overlapping enough to blend, manageable for GPU
        emitter.mainEmitter.size = max(0.03, scaledR * 0.55)
        emitter.mainEmitter.sizeVariation = emitter.mainEmitter.size * 0.4
        // Grow slightly over lifetime
        emitter.mainEmitter.sizeMultiplierAtEndOfLifespan = 1.2
        emitter.mainEmitter.sizeMultiplierAtEndOfLifespanPower = 0.4

        // Evolving color: bright start → dim end for organic lifecycle
        emitter.mainEmitter.color = .evolving(
            start: .single(startColor),
            end: .single(endColor)
        )
        emitter.mainEmitter.colorEvolutionPower = 0.5
        // Alpha blending preserves color identity — additive summed all colors to white
        emitter.mainEmitter.blendMode = .alpha

        // Soft gaussian texture — the single biggest visual improvement
        if let texture = softParticleTexture {
            emitter.mainEmitter.image = texture
        }

        // Billboard — always face camera
        emitter.mainEmitter.billboardMode = .billboard

        // Minimal noise — near-static glow, not morphing blobs
        emitter.mainEmitter.noiseStrength = 0.0003
        emitter.mainEmitter.noiseScale = 6.0
        emitter.mainEmitter.noiseAnimationSpeed = 0.03

        // Fade in and out smoothly
        emitter.mainEmitter.opacityCurve = .quickFadeInOut

        // Maximum damping — particles stay exactly where they spawn
        emitter.mainEmitter.dampingFactor = 0.98

        return emitter
    }

    // MARK: - Nebula Update

    /// Full nebula update — creates/removes/repositions particle emitter entities.
    func updateNebulae() {
        if Self.DIAG_SKIP_NEBULAE {
            // Remove any existing emitters for clean measurement
            for (_, e) in nebulaEmitters { e.removeFromParent() }
            nebulaEmitters.removeAll()
            return
        }
        let groups = nebulaGroupsForCurrentMode()
        let currentKeys = Set(groups.map(\.key))

        if renderFrameCount % 180 == 0 {
            frameLog.info("[nebula] groups=\(groups.count) emitters=\(self.nebulaEmitters.count) mode=\(self.renderLayoutMode == .embedding ? "semantic" : "force") clusters=\(self.renderClusters.count)")
        }

        // Remove stale emitters
        for key in nebulaEmitters.keys where !currentKeys.contains(key) {
            nebulaEmitters[key]?.removeFromParent()
            nebulaEmitters.removeValue(forKey: key)
        }

        for group in groups {
            let R = max(group.radius, 30.0)
            let worldCentroid = group.centroid * scaleFactor

            let colors = nebulaColors(for: group.project, colorMap: renderColorMap)

            if let entity = nebulaEmitters[group.key] {
                // Update position
                entity.position = worldCentroid
                // Update emitter shape size if radius changed significantly
                if var emitter = entity.components[ParticleEmitterComponent.self] {
                    let scaledR = R * scaleFactor
                    let newSize = SIMD3<Float>(repeating: scaledR * 2)
                    if abs(emitter.emitterShapeSize.x - newSize.x) > 0.01 {
                        emitter.emitterShapeSize = SIMD3<Float>(repeating: scaledR * 2.4)
                        emitter.mainEmitter.size = max(0.03, scaledR * 0.55)
                        emitter.mainEmitter.sizeVariation = emitter.mainEmitter.size * 0.4
                        emitter.mainEmitter.birthRate = max(5, min(15, R * 0.05))
                        emitter.mainEmitter.color = .evolving(
                            start: .single(colors.start), end: .single(colors.end)
                        )
                        entity.components.set(emitter)
                    }
                }
            } else {
                // Create new emitter entity
                let entity = Entity()
                entity.position = worldCentroid
                entity.name = "nebula_\(group.key)"
                let emitter = makeNebulaEmitter(radius: R, startColor: colors.start, endColor: colors.end)
                entity.components.set(emitter)
                nebulaContainer.addChild(entity)
                nebulaEmitters[group.key] = entity
            }
        }
    }

    /// Per-frame position update for existing emitters (particles animate themselves).

    func updateNebulaBreathing() {
        if Self.DIAG_SKIP_NEBULAE { return }
        let groups = nebulaGroupsForCurrentMode()
        let groupByKey = Dictionary(groups.map { ($0.key, $0) }, uniquingKeysWith: { _, last in last })

        for (key, entity) in nebulaEmitters {
            guard let group = groupByKey[key] else { continue }
            entity.position = group.centroid * scaleFactor
        }
    }

    // MARK: - Camera Transform & Hit Testing

    func cameraTransform() -> Transform {
        let pos = cameraPosition
        let fwd = forward
        let rt = right
        let u = up
        let rotation = simd_quatf(simd_float3x3(columns: (rt, u, -fwd)))

        var transform = Transform()
        transform.translation = pos * scaleFactor
        transform.rotation = rotation
        return transform
    }

    func project(point3D: SIMD3<Float>, viewSize: CGSize) -> CGPoint? {
        let camPos = cameraPosition * scaleFactor
        let pointScaled = point3D * scaleFactor

        let toPoint = pointScaled - camPos
        let fwd = forward
        let rt = right
        let u = up

        let depth = dot(toPoint, fwd)
        guard depth > 0.01 * scaleFactor else { return nil }

        let projX = dot(toPoint, rt) / depth
        let projY = dot(toPoint, u) / depth

        let fovTan = tan(Float.pi / 6)  // half of 60° FOV
        let aspect = Float(viewSize.width / viewSize.height)

        let screenX = CGFloat((projX / (fovTan * aspect) * 0.5 + 0.5)) * viewSize.width
        let screenY = CGFloat((0.5 - projY / fovTan * 0.5)) * viewSize.height

        return CGPoint(x: screenX, y: screenY)
    }

    func hitTest(at location: CGPoint, viewSize: CGSize, positions: [UUID: SIMD3<Float>]) -> UUID? {
        var closest: UUID?
        var closestDist: CGFloat = 30

        for (id, pos) in positions {
            guard let screenPos = project(point3D: pos, viewSize: viewSize) else { continue }
            let dist = hypot(location.x - screenPos.x, location.y - screenPos.y)
            if dist < closestDist {
                closestDist = dist
                closest = id
            }
        }
        return closest
    }


    func clearAll() {
        // Clear batched node mesh + GPU stamp buffers
        nodeBatchEntity?.removeFromParent()
        nodeBatchEntity = nil
        nodeBatchMesh = nil
        nodeBatchCapacity = 0
        nodeColorCache.removeAll()
        stampInstanceBuffer = nil
        stampInstanceCapacity = 0
        labelInstanceBuffer = nil
        labelInstanceCapacity = 0
        edgeStagingBuffer = nil
        edgeStagingCapacity = 0
        // Clear point lights
        for (_, entity) in pointLightEntities { entity.removeFromParent() }
        pointLightEntities.removeAll()
        // Clear dying node entities
        for (_, entity) in dyingNodeEntities { entity.removeFromParent() }
        dyingNodeEntities.removeAll()
        dyingNodePos3D.removeAll()
        // Clear batched edge mesh
        edgeBatchEntity?.removeFromParent()
        edgeBatchEntity = nil
        edgeBatchMesh = nil
        edgeBatchCapacity = 0
        edgeColorCache.removeAll()
        // Clear batched label mesh
        labelBatchEntity?.removeFromParent()
        labelBatchEntity = nil
        labelBatchMesh = nil
        labelBatchCapacity = 0
        labelAtlasTexture = nil
        labelAtlasLLT = nil
        labelAtlasAllocW = 0
        labelAtlasAllocH = 0
        labelAtlasRects.removeAll()
        projectLabelAtlasRects.removeAll()
        labelAtlasNodeIds.removeAll()
        labelAtlasHubIds.removeAll()
        labelAtlasProjects.removeAll()
        projectCentroids.removeAll()
        for (_, entity) in nebulaEmitters { entity.removeFromParent() }
        nebulaEmitters.removeAll()
        nebulaColorCache.removeAll()
        // Clean up flow particles (container swap — single entity removal)
        flowParticleContainer.removeFromParent()
        flowParticleContainer = Entity()
        flowParticleContainer.name = "flow_particles"
        rootEntity.addChild(flowParticleContainer)
        flowParticles.removeAll()
        flowSpawnTimers.removeAll()
        // Clean up expansion state
        expandedHubs.removeAll()
        preExpansionPositions.removeAll()
        expansionProgress.removeAll()
        expansionDirection.removeAll()
        expandedChildPositions.removeAll()
    }
}
