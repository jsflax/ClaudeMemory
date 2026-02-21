import Foundation
import Accelerate
import simd

/// 3D force-directed graph layout engine. Same split architecture as ForceSimulation:
/// O(n²) force computation runs async on @ForceSimulatorActor, O(n) integration runs sync at 60fps.
@MainActor
final class ForceSimulation3D: ObservableObject {
    // SoA layout for cache-friendly iteration
    private var ids: [Int64] = []
    private var x: [Float] = []
    private var y: [Float] = []
    private var z: [Float] = []
    private var vx: [Float] = []
    private var vy: [Float] = []
    private var vz: [Float] = []
    private var pinned: [Bool] = []
    private var idToIndex: [Int64: Int] = [:]
    private var edgeIndices: [(Int, Int)] = []
    private var projectGroup: [Int] = []
    private var topicGroup: [Int] = []
    private var topicProjectGroup: [Int] = []

    // Reverse mappings for surgical insert (addNode)
    private var projectToGroup: [String: Int] = [:]
    private var topicKeyToGroup: [String: Int] = [:]

    // Stored per-node forces from last async computation
    private var storedFx: [Float] = []
    private var storedFy: [Float] = []
    private var storedFz: [Float] = []

    private(set) var positions: [Int64: SIMD3<Float>] = [:]

    // Force parameters (tuned for 3D — slightly stronger since space is larger)
    private let springLength: Float = 240
    private let crossProjectSpringLength: Float = 400
    private let springStrength: Float = 0.0004
    private let chargeStrength: Float = 500
    private let sameTopicChargeScale: Float = 0.35
    private let centerStrength: Float = 0.006
    private let cohesionStrength: Float = 0.0015
    private let centroidRepulsion: Float = 2500
    private let topicCohesionStrength: Float = 0.009
    private let topicCentroidRepulsion: Float = 3500
    private let damping: Float = 0.78
    private let maxSpeed: Float = 12.0

    private var alpha: Float = 1.0
    private let alphaDecay: Float = 0.997
    private let alphaFloor: Float = 0.01
    private var tickInFlight = false
    private var forceAge: Int = 100
    private var topologyVersion: UInt64 = 0

    var center: SIMD3<Float> = .zero
    var isActive: Bool = true

    // MARK: - Graph Management

    func updateGraph(nodeIds: Set<Int64>, edges: [(Int64, Int64)],
                     projectForNode: [Int64: String] = [:], topicForNode: [Int64: String] = [:]) {
        // Remove nodes no longer in the graph
        var keep = [Bool](repeating: false, count: ids.count)
        for (i, id) in ids.enumerated() { keep[i] = nodeIds.contains(id) }

        var newIds: [Int64] = []
        var newX: [Float] = [], newY: [Float] = [], newZ: [Float] = []
        var newVx: [Float] = [], newVy: [Float] = [], newVz: [Float] = []
        var newPinned: [Bool] = []
        let cap = nodeIds.count
        newIds.reserveCapacity(cap)
        newX.reserveCapacity(cap); newY.reserveCapacity(cap); newZ.reserveCapacity(cap)
        newVx.reserveCapacity(cap); newVy.reserveCapacity(cap); newVz.reserveCapacity(cap)
        newPinned.reserveCapacity(cap)

        for i in 0..<ids.count where keep[i] {
            newIds.append(ids[i])
            newX.append(x[i]); newY.append(y[i]); newZ.append(z[i])
            newVx.append(vx[i]); newVy.append(vy[i]); newVz.append(vz[i])
            newPinned.append(pinned[i])
        }

        // Add new nodes
        for id in nodeIds where idToIndex[id] == nil {
            let angle = Float.random(in: 0...(2 * .pi))
            let phi = Float.random(in: -.pi/2...(.pi/2))
            let r = Float.random(in: 50...250)
            newIds.append(id)
            newX.append(center.x + cos(angle) * cos(phi) * r)
            newY.append(center.y + sin(angle) * cos(phi) * r)
            newZ.append(center.z + sin(phi) * r)
            newVx.append(0); newVy.append(0); newVz.append(0)
            newPinned.append(false)
        }

        ids = newIds
        x = newX; y = newY; z = newZ
        vx = newVx; vy = newVy; vz = newVz
        pinned = newPinned

        // Rebuild index
        idToIndex.removeAll(keepingCapacity: true)
        for (i, id) in ids.enumerated() { idToIndex[id] = i }

        // Build project group indices
        var newProjectToGroup: [String: Int] = [:]
        projectGroup = ids.map { id in
            let proj = projectForNode[id] ?? ""
            if let g = newProjectToGroup[proj] { return g }
            let g = newProjectToGroup.count; newProjectToGroup[proj] = g; return g
        }
        self.projectToGroup = newProjectToGroup

        // Build topic group indices
        var newTopicKeyToGroup: [String: Int] = [:]
        var topicGroupToProjectGroup: [Int] = []
        topicGroup = ids.map { id in
            let proj = projectForNode[id] ?? ""
            let topic = topicForNode[id] ?? "general"
            let key = "\(proj)|\(topic)"
            if let g = newTopicKeyToGroup[key] { return g }
            let g = newTopicKeyToGroup.count; newTopicKeyToGroup[key] = g
            topicGroupToProjectGroup.append(newProjectToGroup[proj] ?? 0)
            return g
        }
        self.topicKeyToGroup = newTopicKeyToGroup
        topicProjectGroup = topicGroupToProjectGroup

        // Convert edges to index pairs
        edgeIndices = edges.compactMap { (src, tgt) in
            guard let si = idToIndex[src], let ti = idToIndex[tgt] else { return nil }
            return (si, ti)
        }

        storedFx = [Float](repeating: 0, count: ids.count)
        storedFy = [Float](repeating: 0, count: ids.count)
        storedFz = [Float](repeating: 0, count: ids.count)

        alpha = max(alpha, 0.05)
        topologyVersion &+= 1
        rebuildPositions()
    }

    /// Surgically insert a single node without rebuilding the entire graph.
    func addNode(_ id: Int64, project: String, topic: String) {
        guard idToIndex[id] == nil else { return }
        let angle = Float.random(in: 0...(2 * .pi))
        let phi = Float.random(in: -.pi/2...(.pi/2))
        let r = Float.random(in: 50...250)
        ids.append(id)
        x.append(center.x + cos(angle) * cos(phi) * r)
        y.append(center.y + sin(angle) * cos(phi) * r)
        z.append(center.z + sin(phi) * r)
        vx.append(0); vy.append(0); vz.append(0)
        pinned.append(false)
        idToIndex[id] = ids.count - 1

        // Project group
        let projGroup: Int
        if let g = projectToGroup[project] {
            projGroup = g
        } else {
            projGroup = projectToGroup.count
            projectToGroup[project] = projGroup
        }
        projectGroup.append(projGroup)

        // Topic group
        let topicKey = "\(project)|\(topic)"
        if let g = topicKeyToGroup[topicKey] {
            topicGroup.append(g)
        } else {
            let g = topicKeyToGroup.count
            topicKeyToGroup[topicKey] = g
            topicGroup.append(g)
            topicProjectGroup.append(projGroup)
        }

        storedFx.append(0); storedFy.append(0); storedFz.append(0)
        alpha = max(alpha, 0.05)
        topologyVersion &+= 1
        rebuildPositions()
    }

    /// Surgically insert a single edge without rebuilding the entire graph.
    func addEdge(from source: Int64, to target: Int64) {
        guard let si = idToIndex[source], let ti = idToIndex[target] else { return }
        guard !edgeIndices.contains(where: { $0.0 == si && $0.1 == ti }) else { return }
        edgeIndices.append((si, ti))
        alpha = max(alpha, 0.05)
        topologyVersion &+= 1
    }

    /// Write external positions (e.g. from 2D positions + z jitter) into internal arrays.
    func setPositions(_ positions: [Int64: SIMD3<Float>]) {
        for (id, point) in positions {
            guard let i = idToIndex[id] else { continue }
            x[i] = point.x; y[i] = point.y; z[i] = point.z
            vx[i] = 0; vy[i] = 0; vz[i] = 0
        }
        storedFx = [Float](repeating: 0, count: ids.count)
        storedFy = [Float](repeating: 0, count: ids.count)
        storedFz = [Float](repeating: 0, count: ids.count)
        forceAge = 5
        alpha = 0.5
        syncPositions()
    }

    // MARK: - Pinning (for node expansion)

    /// Pin a node so the force simulation won't move it.
    func pin(_ id: Int64) {
        guard let i = idToIndex[id] else { return }
        pinned[i] = true
        vx[i] = 0; vy[i] = 0; vz[i] = 0
    }

    /// Unpin a node so the force simulation resumes moving it.
    func unpin(_ id: Int64) {
        guard let i = idToIndex[id] else { return }
        pinned[i] = false
    }

    /// Set a node's position directly and zero its velocity.
    func setPosition(_ id: Int64, to pos: SIMD3<Float>) {
        guard let i = idToIndex[id] else { return }
        x[i] = pos.x; y[i] = pos.y; z[i] = pos.z
        vx[i] = 0; vy[i] = 0; vz[i] = 0
        positions[id] = pos
    }

    // MARK: - Per-frame tick

    func tick() {
        guard isActive else { return }
        let n = ids.count
        guard n > 1 else {
            if n == 1 { x[0] = center.x; y[0] = center.y; z[0] = center.z; syncPositions() }
            return
        }

        let hasForcesComputed = storedFx.count == n
        let attenuation = pow(damping, Float(forceAge))
        forceAge += 1

        for i in 0..<n where !pinned[i] {
            if hasForcesComputed {
                vx[i] += storedFx[i] * attenuation
                vy[i] += storedFy[i] * attenuation
                vz[i] += storedFz[i] * attenuation
            }
            vx[i] *= damping; vy[i] *= damping; vz[i] *= damping

            let speedSq = vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]
            if speedSq > maxSpeed * maxSpeed {
                let scale = maxSpeed / sqrt(speedSq)
                vx[i] *= scale; vy[i] *= scale; vz[i] *= scale
            }

            x[i] += vx[i]; y[i] += vy[i]; z[i] += vz[i]
        }
        alpha = max(alpha * alphaDecay, alphaFloor)
        syncPositions()

        if !tickInFlight { dispatchForceComputation() }
    }

    // MARK: - Async force computation

    private struct SimState: Sendable {
        let n: Int
        let x: [Float], y: [Float], z: [Float]
        let projectGroup: [Int], topicGroup: [Int]
        let edgeIndices: [(Int, Int)]
        let topicProjectGroup: [Int]
        let alpha: Float, center: SIMD3<Float>
        let chargeStrength: Float, sameTopicChargeScale: Float
        let centerStrength: Float
        let springLength: Float, crossProjectSpringLength: Float, springStrength: Float
        let cohesionStrength: Float, centroidRepulsion: Float
        let topicCohesionStrength: Float, topicCentroidRepulsion: Float
    }

    private struct ForceResult: Sendable {
        let fx: [Float], fy: [Float], fz: [Float]
    }

    private func dispatchForceComputation() {
        let n = ids.count
        guard n > 1 else { return }

        tickInFlight = true
        let version = topologyVersion

        let state = SimState(
            n: n, x: x, y: y, z: z,
            projectGroup: projectGroup, topicGroup: topicGroup,
            edgeIndices: edgeIndices, topicProjectGroup: topicProjectGroup,
            alpha: alpha, center: center,
            chargeStrength: chargeStrength, sameTopicChargeScale: sameTopicChargeScale,
            centerStrength: centerStrength,
            springLength: springLength, crossProjectSpringLength: crossProjectSpringLength,
            springStrength: springStrength,
            cohesionStrength: cohesionStrength, centroidRepulsion: centroidRepulsion,
            topicCohesionStrength: topicCohesionStrength, topicCentroidRepulsion: topicCentroidRepulsion
        )

        Task.detached(priority: .userInitiated) {
            let result = await Self.computeForces(state)
            await MainActor.run { [self] in
                guard topologyVersion == version else { tickInFlight = false; return }
                storedFx = result.fx; storedFy = result.fy; storedFz = result.fz
                forceAge = 2
                tickInFlight = false
            }
        }
    }

    @ForceSimulatorActor
    private static func computeForces(_ s: SimState) -> ForceResult {
        let n = s.n
        let x = s.x, y = s.y, z = s.z
        let projectGroup = s.projectGroup, topicGroup = s.topicGroup
        let alpha = s.alpha

        let hasProjects = !projectGroup.isEmpty
        let hasTopics = !topicGroup.isEmpty

        var fx = [Float](repeating: 0, count: n)
        var fy = [Float](repeating: 0, count: n)
        var fz = [Float](repeating: 0, count: n)

        // --- Charge repulsion (O(n²)) ---
        let chargeBase = s.chargeStrength
        let chargeCross = chargeBase * 3.0
        let chargeSameTopic = chargeBase * s.sameTopicChargeScale
        let cutoffSq: Float = 500 * 500

        for i in 0..<n {
            let xi = x[i], yi = y[i], zi = z[i]
            let pg_i = hasProjects ? projectGroup[i] : 0
            let tg_i = hasTopics ? topicGroup[i] : -1

            for j in (i + 1)..<n {
                let dx = xi - x[j], dy = yi - y[j], dz = zi - z[j]
                var distSq = dx * dx + dy * dy + dz * dz
                if distSq > cutoffSq { continue }
                if distSq < 1 { distSq = 1 }

                let charge: Float
                if hasProjects && projectGroup[j] != pg_i {
                    charge = chargeCross
                } else if hasTopics && topicGroup[j] == tg_i {
                    charge = chargeSameTopic
                } else {
                    charge = chargeBase
                }

                let dist = sqrt(distSq)
                let forceMag = charge / distSq
                let ux = dx / dist, uy = dy / dist, uz = dz / dist
                let ffx = ux * forceMag, ffy = uy * forceMag, ffz = uz * forceMag
                fx[i] += ffx; fy[i] += ffy; fz[i] += ffz
                fx[j] -= ffx; fy[j] -= ffy; fz[j] -= ffz
            }
        }

        // --- Spring attraction ---
        let springStr = s.springStrength
        let springLen = s.springLength
        let crossSpringLen = s.crossProjectSpringLength
        for (si, ti) in s.edgeIndices {
            let dx = x[ti] - x[si], dy = y[ti] - y[si], dz = z[ti] - z[si]
            var d = sqrt(dx * dx + dy * dy + dz * dz)
            if d < 1 { d = 1 }
            let cross = hasProjects && projectGroup[si] != projectGroup[ti]
            let rest = cross ? crossSpringLen : springLen
            let force = springStr * (d - rest)
            let efx = (dx / d) * force, efy = (dy / d) * force, efz = (dz / d) * force
            fx[si] += efx; fy[si] += efy; fz[si] += efz
            fx[ti] -= efx; fy[ti] -= efy; fz[ti] -= efz
        }

        // --- Topic centroid forces ---
        let topicN = (topicGroup.max() ?? -1) + 1
        if topicN > 1 {
            var tSumX = [Float](repeating: 0, count: topicN)
            var tSumY = [Float](repeating: 0, count: topicN)
            var tSumZ = [Float](repeating: 0, count: topicN)
            var tCount = [Int](repeating: 0, count: topicN)
            for i in 0..<n {
                let g = topicGroup[i]
                tSumX[g] += x[i]; tSumY[g] += y[i]; tSumZ[g] += z[i]; tCount[g] += 1
            }
            let topicCoh = s.topicCohesionStrength
            for i in 0..<n {
                let g = topicGroup[i]; let cnt = Float(tCount[g])
                if cnt < 2 { continue }
                let cx = tSumX[g] / cnt, cy = tSumY[g] / cnt, cz = tSumZ[g] / cnt
                fx[i] += (cx - x[i]) * topicCoh
                fy[i] += (cy - y[i]) * topicCoh
                fz[i] += (cz - z[i]) * topicCoh
            }
            let tpg = s.topicProjectGroup
            let topicCentRep = s.topicCentroidRepulsion
            for g1 in 0..<topicN {
                guard tCount[g1] > 0 else { continue }
                let c1x = tSumX[g1] / Float(tCount[g1])
                let c1y = tSumY[g1] / Float(tCount[g1])
                let c1z = tSumZ[g1] / Float(tCount[g1])
                for g2 in (g1 + 1)..<topicN {
                    guard tCount[g2] > 0 else { continue }
                    guard g1 < tpg.count && g2 < tpg.count, tpg[g1] == tpg[g2] else { continue }
                    let c2x = tSumX[g2] / Float(tCount[g2])
                    let c2y = tSumY[g2] / Float(tCount[g2])
                    let c2z = tSumZ[g2] / Float(tCount[g2])
                    var tdx = c1x - c2x, tdy = c1y - c2y, tdz = c1z - c2z
                    var tdist = sqrt(tdx * tdx + tdy * tdy + tdz * tdz)
                    if tdist < 1 { tdist = 1; tdx = .random(in: -1...1); tdy = .random(in: -1...1); tdz = .random(in: -1...1) }
                    let force = topicCentRep / (tdist * tdist)
                    let tfx = (tdx / tdist) * force, tfy = (tdy / tdist) * force, tfz = (tdz / tdist) * force
                    let f1 = 1.0 / Float(tCount[g1]), f2 = 1.0 / Float(tCount[g2])
                    for i in 0..<n where topicGroup[i] == g1 {
                        fx[i] += tfx * f1; fy[i] += tfy * f1; fz[i] += tfz * f1
                    }
                    for i in 0..<n where topicGroup[i] == g2 {
                        fx[i] -= tfx * f2; fy[i] -= tfy * f2; fz[i] -= tfz * f2
                    }
                }
            }
        }

        // --- Project centroid forces ---
        if hasProjects {
            let groupN = (projectGroup.max() ?? -1) + 1
            if groupN > 0 {
                var gSumX = [Float](repeating: 0, count: groupN)
                var gSumY = [Float](repeating: 0, count: groupN)
                var gSumZ = [Float](repeating: 0, count: groupN)
                var gCount = [Int](repeating: 0, count: groupN)
                for i in 0..<n {
                    let g = projectGroup[i]
                    gSumX[g] += x[i]; gSumY[g] += y[i]; gSumZ[g] += z[i]; gCount[g] += 1
                }
                let cohStr = s.cohesionStrength
                for i in 0..<n {
                    let g = projectGroup[i]; let cnt = Float(gCount[g])
                    if cnt < 2 { continue }
                    let cx = gSumX[g] / cnt, cy = gSumY[g] / cnt, cz = gSumZ[g] / cnt
                    fx[i] += (cx - x[i]) * cohStr
                    fy[i] += (cy - y[i]) * cohStr
                    fz[i] += (cz - z[i]) * cohStr
                }
                let centRep = s.centroidRepulsion
                for g1 in 0..<groupN {
                    guard gCount[g1] > 0 else { continue }
                    let c1 = SIMD3<Float>(gSumX[g1], gSumY[g1], gSumZ[g1]) / Float(gCount[g1])
                    for g2 in (g1 + 1)..<groupN {
                        guard gCount[g2] > 0 else { continue }
                        let c2 = SIMD3<Float>(gSumX[g2], gSumY[g2], gSumZ[g2]) / Float(gCount[g2])
                        var delta = c1 - c2
                        var pdist = simd_length(delta)
                        if pdist < 1 { pdist = 1; delta = SIMD3(.random(in: -1...1), .random(in: -1...1), .random(in: -1...1)) }
                        let force = centRep / (pdist * pdist)
                        let fVec = (delta / pdist) * force
                        let f1 = 1.0 / Float(gCount[g1]), f2 = 1.0 / Float(gCount[g2])
                        for i in 0..<n where projectGroup[i] == g1 {
                            fx[i] += fVec.x * f1; fy[i] += fVec.y * f1; fz[i] += fVec.z * f1
                        }
                        for i in 0..<n where projectGroup[i] == g2 {
                            fx[i] -= fVec.x * f2; fy[i] -= fVec.y * f2; fz[i] -= fVec.z * f2
                        }
                    }
                }
            }
        }

        // --- Center gravity ---
        let centerStr = s.centerStrength
        let c = s.center
        for i in 0..<n {
            fx[i] += (c.x - x[i]) * alpha * centerStr
            fy[i] += (c.y - y[i]) * alpha * centerStr
            fz[i] += (c.z - z[i]) * alpha * centerStr
        }

        return ForceResult(fx: fx, fy: fy, fz: fz)
    }

    // MARK: - Position sync

    private func rebuildPositions() {
        positions.removeAll(keepingCapacity: true)
        for i in 0..<ids.count {
            positions[ids[i]] = SIMD3(x[i], y[i], z[i])
        }
    }

    private func syncPositions() {
        for i in 0..<ids.count {
            positions[ids[i]] = SIMD3(x[i], y[i], z[i])
        }
    }
}
