import SwiftUI
import RealityKit
import simd

extension Graph3DScene {

    // MARK: - Hub Expansion

    func childrenOfHub(_ hubId: UUID) -> [UUID] {
        renderEdges.filter { $0.relation == "part_of" && $0.targetId == hubId }.map(\.sourceId)
    }

    /// Compute Fibonacci sphere orbit positions for children around a hub.
    func computeOrbitPositions(hubId: UUID, children: [UUID]) -> [UUID: SIMD3<Float>] {
        guard let hubPos = renderPositions[hubId] else { return [:] }
        let radius: Float = 80
        var result: [UUID: SIMD3<Float>] = [:]
        let n = children.count
        guard n > 0 else { return result }
        let goldenRatio: Float = (1 + sqrt(5)) / 2
        for (idx, childId) in children.enumerated() {
            let i = Float(idx)
            let theta = acos(1 - 2 * (i + 0.5) / Float(n))
            let phi = 2 * Float.pi * i / goldenRatio
            let x = radius * sin(theta) * cos(phi)
            let y = radius * sin(theta) * sin(phi)
            let z = radius * cos(theta)
            result[childId] = hubPos + SIMD3(x, y, z)
        }
        return result
    }

    /// Toggle hub expansion: start expanding if collapsed, start collapsing if expanded.
    func toggleHubExpansion(hubId: UUID) {
        if expandedHubs.contains(hubId) {
            expansionDirection[hubId] = false
        } else {
            expandedHubs.insert(hubId)
            let children = childrenOfHub(hubId)
            for childId in children {
                preExpansionPositions[childId] = renderPositions[childId] ?? .zero
            }
            expansionProgress[hubId] = 0
            expansionDirection[hubId] = true
        }
    }

    /// Per-frame expansion animation: lerp children between original and orbit positions.
    func updateExpansions(dt: Float) {
        var toRemove: [UUID] = []
        var allExpandedPositions: [UUID: SIMD3<Float>] = [:]

        for hubId in expandedHubs {
            let expanding = expansionDirection[hubId] ?? true
            var progress = expansionProgress[hubId] ?? 0

            if expanding {
                progress = min(1.0, progress + dt * 3.0)
            } else {
                progress = max(0.0, progress - dt * 3.0)
            }
            expansionProgress[hubId] = progress

            // Smooth-step ease
            let t = progress * progress * (3 - 2 * progress)

            let children = childrenOfHub(hubId)
            let orbitPositions = computeOrbitPositions(hubId: hubId, children: children)

            for childId in children {
                let startPos = preExpansionPositions[childId] ?? renderPositions[childId] ?? .zero
                let endPos = orbitPositions[childId] ?? startPos
                let lerpedPos = startPos + (endPos - startPos) * t
                renderPositions[childId] = lerpedPos
                allExpandedPositions[childId] = lerpedPos
                // Position updated in renderPositions — batch mesh picks it up automatically
            }

            // When collapse finishes
            if !expanding && progress <= 0 {
                toRemove.append(hubId)
                for childId in children {
                    if let original = preExpansionPositions[childId] {
                        renderPositions[childId] = original
                    }
                    preExpansionPositions.removeValue(forKey: childId)
                }
            }
        }

        expandedChildPositions = allExpandedPositions

        for hubId in toRemove {
            expandedHubs.remove(hubId)
            expansionProgress.removeValue(forKey: hubId)
            expansionDirection.removeValue(forKey: hubId)
        }
    }

    // MARK: - Edge Flow Particles

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

    func updateFlowParticles(dt: Float) {
        let sel = renderSelectedNode

        // Hard cleanup on selection change — flush ALL particles so old paths don't linger.
        // Swap the container entity instead of removing children individually.
        // N individual removeFromParent() calls cause N scene graph rebuilds → flicker.
        // One container swap = 1 removal + 1 addition regardless of particle count.
        if sel != flowLastSelectedNode {
            flowParticleContainer.removeFromParent()
            flowParticleContainer = Entity()
            flowParticleContainer.name = "flow_particles"
            rootEntity.addChild(flowParticleContainer)
            flowParticles.removeAll()
            flowSpawnTimers.removeAll()
            flowLastSelectedNode = sel
        }

        // Determine active edges: connected to selected node or expanded hub children
        var activeEdges: [EdgeData] = []

        for edge in renderEdges {
            let isConnectedToSelected = (sel != nil) && (edge.sourceId == sel || edge.targetId == sel)
            let isExpandedPartOf = edge.relation == "part_of" && expandedHubs.contains(edge.targetId)
            if isConnectedToSelected || isExpandedPartOf {
                activeEdges.append(edge)
            }
        }

        // Cap active edges for performance
        if activeEdges.count > 30 {
            activeEdges = Array(activeEdges.prefix(30))
        }

        let activeKeys = Set(activeEdges.map { "\($0.sourceId)-\($0.targetId)" })

        // Clean up particles for inactive edges — hide instead of removing.
        // Set scale to zero (invisible, degenerate) to avoid scene graph churn.
        // Entity cleanup deferred to the container swap on next selection change.
        for key in flowParticles.keys where !activeKeys.contains(key) {
            if let particles = flowParticles[key] {
                for p in particles {
                    p.entity.scale = .zero
                    p.entity.components.set(OpacityComponent(opacity: 0))
                }
            }
            flowParticles.removeValue(forKey: key)
            flowSpawnTimers.removeValue(forKey: key)
        }

        // Update/spawn particles for active edges
        for edge in activeEdges {
            let key = "\(edge.sourceId)-\(edge.targetId)"
            guard let p1 = renderPositions[edge.sourceId],
                  let p2 = renderPositions[edge.targetId] else { continue }

            let scaledP1 = p1 * scaleFactor
            let scaledP2 = p2 * scaleFactor
            let color = flowColorForRelation(edge.relation)

            // Spawn timer
            var timer = flowSpawnTimers[key] ?? 0
            timer += dt
            if timer >= 0.4 {
                timer -= 0.4
                var mat = UnlitMaterial()
                mat.color = .init(tint: NSColor(
                    red: CGFloat(color.x), green: CGFloat(color.y),
                    blue: CGFloat(color.z), alpha: 1.0
                ))
                let entity = ModelEntity(mesh: flowParticleMesh, materials: [mat])
                entity.scale = SIMD3<Float>(repeating: 0.006)
                entity.position = scaledP1
                entity.components.set(OpacityComponent(opacity: 0))
                flowParticleContainer.addChild(entity)

                var particles = flowParticles[key] ?? []
                particles.append(FlowParticle(entity: entity, t: 0, speed: 0.8))
                flowParticles[key] = particles
            }
            flowSpawnTimers[key] = timer

            // Advance existing particles
            if var particles = flowParticles[key] {
                var toRemove: [Int] = []
                for i in particles.indices {
                    particles[i].t += dt * particles[i].speed
                    let t = particles[i].t

                    if t >= 1.0 {
                        // Hide instead of removeFromParent — deferred to container swap
                        particles[i].entity.scale = .zero
                        particles[i].entity.components.set(OpacityComponent(opacity: 0))
                        toRemove.append(i)
                        continue
                    }

                    // Position: lerp along edge
                    let pos = scaledP1 + (scaledP2 - scaledP1) * t
                    particles[i].entity.position = pos

                    // Opacity: fade in first 10%, fade out last 15%
                    let opacity: Float
                    if t < 0.1 {
                        opacity = t / 0.1
                    } else if t > 0.85 {
                        opacity = (1.0 - t) / 0.15
                    } else {
                        opacity = 1.0
                    }
                    particles[i].entity.components.set(OpacityComponent(opacity: opacity))
                }

                for i in toRemove.reversed() {
                    particles.remove(at: i)
                }
                flowParticles[key] = particles
            }
        }
    }
}
