import RealityKit
import AppKit
import simd

/// Per-project nebula particle emitter system.
///
/// Creates/updates/destroys nebula entities as projects appear/disappear.
/// Uses RealityKit ParticleEmitterComponent for gaseous fog effect.
@MainActor
public final class NebulaBatchSystem {
    private var activeNebulae: [String: Entity] = [:]
    private var colorCache: [String: (start: NSColor, end: NSColor)] = [:]
    private var lastColorMapHash: Int = 0

    // Per-project node counts cached on topologyVersion — the inline
    // full-node histogram cost O(n) dict ops per frame at 40k nodes.
    private var cachedProjectSizes: [String: Int] = [:]
    private var cachedSizesTopologyVersion: UInt64 = .max

    public init() {}

    public func update(
        container: Entity,
        dataProvider: SceneDataProvider,
        topologyChanged: Bool,
        scaleFactor: Float
    ) {
        let centroids = dataProvider.projectCentroids
        let colorMap = dataProvider.projectColorMap

        // Invalidate color cache when colors change
        let colorHash = colorMap.hashValue
        if colorHash != lastColorMapHash {
            colorCache.removeAll()
            lastColorMapHash = colorHash
        }

        // Determine which projects have nebulae (2+ nodes)
        if dataProvider.topologyVersion != cachedSizesTopologyVersion {
            var projectSizes: [String: Int] = [:]
            for node in dataProvider.nodes {
                projectSizes[node.project, default: 0] += 1
            }
            cachedProjectSizes = projectSizes
            cachedSizesTopologyVersion = dataProvider.topologyVersion
        }
        let projectSizes = cachedProjectSizes

        let activeProjects = Set(centroids.keys.filter { (projectSizes[$0] ?? 0) >= 2 })

        // Remove nebulae for projects no longer active
        for (project, entity) in activeNebulae where !activeProjects.contains(project) {
            entity.removeFromParent()
        }
        for project in activeNebulae.keys where !activeProjects.contains(project) {
            activeNebulae.removeValue(forKey: project)
        }

        // Create/update nebulae for active projects
        for project in activeProjects {
            guard let centroid = centroids[project] else { continue }
            let color = colorMap[project] ?? SIMD3<Float>(0.5, 0.5, 0.5)

            if let entity = activeNebulae[project] {
                // Update position
                entity.position = centroid * scaleFactor
            } else {
                // Create new nebula. The cluster radius (a full node scan per
                // project) is only needed here — computing it in the shared
                // path cost O(nodes × projects) per frame, 427ms at 40k×70,
                // while existing nebulae never resize.
                let radius = computeClusterRadius(
                    project: project,
                    centroid: centroid,
                    nodes: dataProvider.nodes,
                    positions: dataProvider.positions
                )
                let entity = Entity()
                entity.name = "Nebula_\(project)"
                entity.position = centroid * scaleFactor

                let colors = nebulaColors(for: project, rgb: color)
                let emitter = makeNebulaEmitter(
                    radius: radius,
                    scaleFactor: scaleFactor,
                    startColor: colors.start,
                    endColor: colors.end
                )
                entity.components.set(emitter)
                entity.components.set(NebulaComponent(
                    project: project,
                    color: SIMD4(color.x, color.y, color.z, 0.3),
                    radius: radius
                ))

                container.addChild(entity)
                activeNebulae[project] = entity
            }
        }
    }

    // MARK: - Helpers

    private func computeClusterRadius(
        project: String,
        centroid: SIMD3<Float>,
        nodes: [RKNodeSnapshot],
        positions: [UUID: SIMD3<Float>]
    ) -> Float {
        var maxDist: Float = 0
        for node in nodes where node.project == project {
            if let pos = positions[node.id] {
                maxDist = max(maxDist, simd_length(pos - centroid))
            }
        }
        return maxDist + 40
    }

    private func nebulaColors(for project: String, rgb: SIMD3<Float>) -> (start: NSColor, end: NSColor) {
        if let cached = colorCache[project] { return cached }

        let start = NSColor(
            red: CGFloat(min(1.0, rgb.x * 1.3 + 0.1)),
            green: CGFloat(min(1.0, rgb.y * 1.3 + 0.1)),
            blue: CGFloat(min(1.0, rgb.z * 1.3 + 0.1)),
            alpha: 0.035
        )
        let end = NSColor(
            red: CGFloat(min(1.0, rgb.x * 0.7 + 0.05)),
            green: CGFloat(min(1.0, rgb.y * 0.7 + 0.05)),
            blue: CGFloat(min(1.0, rgb.z * 0.7 + 0.05)),
            alpha: 0.005
        )
        let result = (start: start, end: end)
        colorCache[project] = result
        return result
    }

    private func makeNebulaEmitter(
        radius: Float,
        scaleFactor: Float,
        startColor: NSColor,
        endColor: NSColor
    ) -> ParticleEmitterComponent {
        var emitter = ParticleEmitterComponent()
        let scaledR = radius * scaleFactor

        emitter.emitterShape = .sphere
        emitter.birthLocation = .volume
        emitter.emitterShapeSize = SIMD3<Float>(repeating: scaledR * 2.4)

        emitter.speed = 0.0001
        emitter.speedVariation = 0.00005

        emitter.timing = .repeating(
            warmUp: 15.0,
            emit: .init(duration: .infinity),
            idle: nil
        )

        emitter.mainEmitter.birthRate = max(5, min(15, radius * 0.05))
        emitter.mainEmitter.lifeSpan = 15.0
        emitter.mainEmitter.lifeSpanVariation = 5.0
        emitter.mainEmitter.size = max(0.03, scaledR * 0.55)
        emitter.mainEmitter.sizeVariation = emitter.mainEmitter.size * 0.4
        emitter.mainEmitter.sizeMultiplierAtEndOfLifespan = 1.2

        emitter.mainEmitter.color = .evolving(
            start: .single(startColor),
            end: .single(endColor)
        )

        emitter.mainEmitter.blendMode = .alpha
        emitter.mainEmitter.billboardMode = .billboard
        emitter.mainEmitter.opacityCurve = .quickFadeInOut

        emitter.mainEmitter.noiseStrength = 0.0003
        emitter.mainEmitter.noiseScale = 6.0
        emitter.mainEmitter.noiseAnimationSpeed = 0.03
        emitter.mainEmitter.dampingFactor = 0.98

        return emitter
    }
}
