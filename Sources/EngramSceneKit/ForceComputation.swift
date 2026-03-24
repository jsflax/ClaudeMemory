// ForceComputation — pure CPU force computation (Barnes-Hut O(n log n) + springs + cohesion).
// Extracted from ForceSimulation3D for testability. No @MainActor, no Metal, no shared state.

import Foundation
import simd

// MARK: - Input / Output

public struct ForceComputationInput: Sendable {
    public let n: Int
    public let x: [Float], y: [Float], z: [Float]
    public let projectGroup: [Int], topicGroup: [Int]
    public let edgeIndices: [(Int, Int)]
    public let topicProjectGroup: [Int]
    public let alpha: Float, center: SIMD3<Float>
    public let chargeStrength: Float, crossChargeMultiplier: Float
    public let sameProjectChargeScale: Float, sameTopicChargeScale: Float
    public let centerStrength: Float
    public let springLength: Float, crossProjectSpringLength: Float, springStrength: Float
    public let cohesionStrength: Float, centroidRepulsion: Float
    public let topicCohesionStrength: Float, topicCentroidRepulsion: Float

    /// Per-node galaxy group index. Empty = single-galaxy backward compat.
    public let galaxyGroup: [Int]
    /// Per-galaxy center position. Indexed by galaxyGroup[i]. Empty = use `center`.
    public let galaxyCenters: [SIMD3<Float>]

    public init(
        n: Int, x: [Float], y: [Float], z: [Float],
        projectGroup: [Int], topicGroup: [Int],
        edgeIndices: [(Int, Int)], topicProjectGroup: [Int],
        alpha: Float, center: SIMD3<Float>,
        chargeStrength: Float, crossChargeMultiplier: Float,
        sameProjectChargeScale: Float, sameTopicChargeScale: Float,
        centerStrength: Float,
        springLength: Float, crossProjectSpringLength: Float, springStrength: Float,
        cohesionStrength: Float, centroidRepulsion: Float,
        topicCohesionStrength: Float, topicCentroidRepulsion: Float,
        galaxyGroup: [Int] = [], galaxyCenters: [SIMD3<Float>] = []
    ) {
        self.n = n; self.x = x; self.y = y; self.z = z
        self.projectGroup = projectGroup; self.topicGroup = topicGroup
        self.edgeIndices = edgeIndices; self.topicProjectGroup = topicProjectGroup
        self.alpha = alpha; self.center = center
        self.chargeStrength = chargeStrength; self.crossChargeMultiplier = crossChargeMultiplier
        self.sameProjectChargeScale = sameProjectChargeScale; self.sameTopicChargeScale = sameTopicChargeScale
        self.centerStrength = centerStrength
        self.springLength = springLength; self.crossProjectSpringLength = crossProjectSpringLength
        self.springStrength = springStrength
        self.cohesionStrength = cohesionStrength; self.centroidRepulsion = centroidRepulsion
        self.topicCohesionStrength = topicCohesionStrength; self.topicCentroidRepulsion = topicCentroidRepulsion
        self.galaxyGroup = galaxyGroup; self.galaxyCenters = galaxyCenters
    }
}

public struct ForceComputationResult: Sendable {
    public let fx: [Float], fy: [Float], fz: [Float]
    public init(fx: [Float], fy: [Float], fz: [Float]) {
        self.fx = fx; self.fy = fy; self.fz = fz
    }
}

// MARK: - Barnes-Hut Octree


public struct OctreeNode: Sendable {
    public var cx: Float, cy: Float, cz: Float   // cell geometric center
    public var halfSize: Float
    public var comX: Float, comY: Float, comZ: Float  // center of mass
    public var mass: Float                         // body count (as Float for division)
    public var child0: Int32, child1: Int32, child2: Int32, child3: Int32
    public var child4: Int32, child5: Int32, child6: Int32, child7: Int32  // -1 = empty
    public var bodyIndex: Int32  // leaf with 1 body: its index; else -1

    public static func empty(cx: Float, cy: Float, cz: Float, halfSize: Float) -> OctreeNode {
        OctreeNode(cx: cx, cy: cy, cz: cz, halfSize: halfSize,
                   comX: 0, comY: 0, comZ: 0, mass: 0,
                   child0: -1, child1: -1, child2: -1, child3: -1,
                   child4: -1, child5: -1, child6: -1, child7: -1,
                   bodyIndex: -1)
    }

    public func child(_ octant: Int) -> Int32 {
        switch octant {
        case 0: return child0; case 1: return child1
        case 2: return child2; case 3: return child3
        case 4: return child4; case 5: return child5
        case 6: return child6; case 7: return child7
        default: return -1
        }
    }

    mutating func setChild(_ octant: Int, _ value: Int32) {
        switch octant {
        case 0: child0 = value; case 1: child1 = value
        case 2: child2 = value; case 3: child3 = value
        case 4: child4 = value; case 5: child5 = value
        case 6: child6 = value; case 7: child7 = value
        default: break
        }
    }
}

/// Build octree from positions. Returns the node array (root at index 0).
public func buildOctree(x: [Float], y: [Float], z: [Float], n: Int) -> [OctreeNode] {
    guard n > 0 else { return [] }

    // Find bounding box, make cubic
    var minX = x[0], maxX = x[0], minY = y[0], maxY = y[0], minZ = z[0], maxZ = z[0]
    for i in 1..<n {
        if x[i] < minX { minX = x[i] } else if x[i] > maxX { maxX = x[i] }
        if y[i] < minY { minY = y[i] } else if y[i] > maxY { maxY = y[i] }
        if z[i] < minZ { minZ = z[i] } else if z[i] > maxZ { maxZ = z[i] }
    }
    let sizeX = maxX - minX, sizeY = maxY - minY, sizeZ = maxZ - minZ
    let maxSize = max(sizeX, max(sizeY, sizeZ), 1.0)
    let half = maxSize * 0.5 + 1.0  // +1 padding to avoid boundary issues
    let rootCX = (minX + maxX) * 0.5
    let rootCY = (minY + maxY) * 0.5
    let rootCZ = (minZ + maxZ) * 0.5

    var nodes = [OctreeNode]()
    nodes.reserveCapacity(max(8 * n, 512))
    nodes.append(.empty(cx: rootCX, cy: rootCY, cz: rootCZ, halfSize: half))

    let maxDepth = 20  // halfSize / 2^20 ≈ 0.001 — sub-pixel, no point going deeper

    for i in 0..<n {
        let px = x[i], py = y[i], pz = z[i]
        var nodeIdx = 0
        var depth = 0

        while depth < maxDepth {
            // Update center of mass
            let oldMass = nodes[nodeIdx].mass
            let newMass = oldMass + 1
            nodes[nodeIdx].comX = (nodes[nodeIdx].comX * oldMass + px) / newMass
            nodes[nodeIdx].comY = (nodes[nodeIdx].comY * oldMass + py) / newMass
            nodes[nodeIdx].comZ = (nodes[nodeIdx].comZ * oldMass + pz) / newMass
            nodes[nodeIdx].mass = newMass

            if nodes[nodeIdx].mass == 1 && nodes[nodeIdx].bodyIndex == -1 {
                // Empty cell — place body here as leaf
                nodes[nodeIdx].bodyIndex = Int32(i)
                break
            }

            if nodes[nodeIdx].bodyIndex >= 0 {
                // Leaf with existing body — subdivide
                let existingBody = Int(nodes[nodeIdx].bodyIndex)
                nodes[nodeIdx].bodyIndex = -1  // no longer a leaf

                // Push existing body down
                let eCX = nodes[nodeIdx].cx, eCY = nodes[nodeIdx].cy, eCZ = nodes[nodeIdx].cz
                let eHalf = nodes[nodeIdx].halfSize * 0.5
                let eOctant = octant(px: x[existingBody], py: y[existingBody], pz: z[existingBody],
                                     cx: eCX, cy: eCY, cz: eCZ)
                let (oCX, oCY, oCZ) = childCenter(parentCX: eCX, parentCY: eCY, parentCZ: eCZ,
                                                    halfSize: eHalf, octant: eOctant)
                let childIdx = Int32(nodes.count)
                var childNode = OctreeNode.empty(cx: oCX, cy: oCY, cz: oCZ, halfSize: eHalf)
                childNode.bodyIndex = Int32(existingBody)
                childNode.comX = x[existingBody]; childNode.comY = y[existingBody]; childNode.comZ = z[existingBody]
                childNode.mass = 1
                nodes.append(childNode)
                nodes[nodeIdx].setChild(eOctant, childIdx)
            }

            // Descend into correct octant for new body
            let oct = octant(px: px, py: py, pz: pz,
                           cx: nodes[nodeIdx].cx, cy: nodes[nodeIdx].cy, cz: nodes[nodeIdx].cz)
            let childRef = nodes[nodeIdx].child(oct)
            if childRef >= 0 {
                nodeIdx = Int(childRef)
            } else {
                // Create new child
                let newHalf = nodes[nodeIdx].halfSize * 0.5
                let (nCX, nCY, nCZ) = childCenter(parentCX: nodes[nodeIdx].cx, parentCY: nodes[nodeIdx].cy,
                                                    parentCZ: nodes[nodeIdx].cz, halfSize: newHalf, octant: oct)
                let newIdx = Int32(nodes.count)
                nodes.append(.empty(cx: nCX, cy: nCY, cz: nCZ, halfSize: newHalf))
                nodes[nodeIdx].setChild(oct, newIdx)
                nodeIdx = Int(newIdx)
            }
            depth += 1
        }

        // Max depth reached — leave as multi-body leaf (mass > 1, bodyIndex == -1)
    }

    return nodes
}

@inline(__always)
func octant(px: Float, py: Float, pz: Float, cx: Float, cy: Float, cz: Float) -> Int {
    var o = 0
    if px >= cx { o |= 1 }
    if py >= cy { o |= 2 }
    if pz >= cz { o |= 4 }
    return o
}

@inline(__always)
func childCenter(parentCX: Float, parentCY: Float, parentCZ: Float,
                          halfSize: Float, octant: Int) -> (Float, Float, Float) {
    let qSize = halfSize * 0.5
    let cx = parentCX + ((octant & 1) != 0 ? qSize : -qSize)
    let cy = parentCY + ((octant & 2) != 0 ? qSize : -qSize)
    let cz = parentCZ + ((octant & 4) != 0 ? qSize : -qSize)
    return (cx, cy, cz)
}

/// Compute charge forces using Barnes-Hut O(n log n) approximation.
/// - theta: opening angle threshold (lower = more accurate, 0.7 is standard)
/// - cutoff: skip subtrees whose nearest point is beyond this distance
func computeChargeForcesBH(
    x: [Float], y: [Float], z: [Float], n: Int,
    projectGroup: [Int], topicGroup: [Int],
    galaxyGroup: [Int],
    hasProjects: Bool, hasTopics: Bool, hasGalaxies: Bool,
    chargeBase: Float, chargeCross: Float,
    chargeSameProject: Float, chargeSameTopic: Float,
    fx: inout [Float], fy: inout [Float], fz: inout [Float]
) {
    let nodes = buildOctree(x: x, y: y, z: z, n: n)
    guard !nodes.isEmpty else { return }

    let theta: Float = 0.7
    let thetaSq = theta * theta
    let cutoff: Float = 500
    let cutoffSq = cutoff * cutoff

    // Pre-allocate traversal stack
    var stack = [Int]()
    stack.reserveCapacity(64)

    for i in 0..<n {
        let xi = x[i], yi = y[i], zi = z[i]
        let pg_i = hasProjects ? projectGroup[i] : 0
        let tg_i = hasTopics ? topicGroup[i] : -1
        let gg_i = hasGalaxies ? galaxyGroup[i] : 0

        stack.removeAll(keepingCapacity: true)
        stack.append(0)  // start from root

        while !stack.isEmpty {
            let nIdx = stack.removeLast()
            let node = nodes[nIdx]

            if node.mass == 0 { continue }

            // Single-body leaf — exact pairwise
            if node.bodyIndex >= 0 {
                let j = Int(node.bodyIndex)
                if j == i { continue }
                // Skip charge between nodes in different galaxies
                if hasGalaxies && galaxyGroup[j] != gg_i { continue }

                let dx = xi - x[j], dy = yi - y[j], dz = zi - z[j]
                var distSq = dx * dx + dy * dy + dz * dz
                if distSq > cutoffSq { continue }
                if distSq < 1 { distSq = 1 }

                let charge: Float
                if hasProjects && projectGroup[j] != pg_i {
                    charge = chargeCross
                } else if hasTopics && topicGroup[j] == tg_i {
                    charge = chargeSameTopic
                } else if hasProjects {
                    charge = chargeSameProject
                } else {
                    charge = chargeBase
                }

                let dist = sqrt(distSq)
                let forceMag = charge / distSq
                fx[i] += (dx / dist) * forceMag
                fy[i] += (dy / dist) * forceMag
                fz[i] += (dz / dist) * forceMag
                continue
            }

            // Internal node — check opening angle
            let dx = xi - node.comX, dy = yi - node.comY, dz = zi - node.comZ
            var distSq = dx * dx + dy * dy + dz * dz
            if distSq < 1 { distSq = 1 }

            let cellSize = node.halfSize * 2
            let sSq = cellSize * cellSize

            // Far enough — approximate as single body with base charge
            if sSq < distSq * thetaSq {
                // Skip if center of mass is beyond cutoff
                if distSq > cutoffSq { continue }

                let dist = sqrt(distSq)
                let forceMag = chargeBase * node.mass / distSq
                fx[i] += (dx / dist) * forceMag
                fy[i] += (dy / dist) * forceMag
                fz[i] += (dz / dist) * forceMag
                continue
            }

            // Check if nearest point of cell is beyond cutoff
            let nearX = max(node.cx - node.halfSize, min(xi, node.cx + node.halfSize))
            let nearY = max(node.cy - node.halfSize, min(yi, node.cy + node.halfSize))
            let nearZ = max(node.cz - node.halfSize, min(zi, node.cz + node.halfSize))
            let nearDx = xi - nearX, nearDy = yi - nearY, nearDz = zi - nearZ
            if nearDx * nearDx + nearDy * nearDy + nearDz * nearDz > cutoffSq { continue }

            // Recurse into children
            for oct in 0..<8 {
                let c = node.child(oct)
                if c >= 0 { stack.append(Int(c)) }
            }
        }
    }
}

// MARK: - Public API

/// Compute all forces (charge + springs + cohesion + centroids + center gravity).
/// Pure function — safe to call from any thread. Uses Barnes-Hut O(n log n) for charge.
public func computeAllForces(_ s: ForceComputationInput) -> ForceComputationResult {
    let n = s.n
    let x = s.x, y = s.y, z = s.z
    let projectGroup = s.projectGroup, topicGroup = s.topicGroup
    let hasProjects = !projectGroup.isEmpty
    let hasTopics = !topicGroup.isEmpty
    var fx = [Float](repeating: 0, count: n)
    var fy = [Float](repeating: 0, count: n)
    var fz = [Float](repeating: 0, count: n)

    // --- Charge repulsion (exact or Barnes-Hut) ---
    let chargeBase = s.chargeStrength
    let chargeCross = chargeBase * s.crossChargeMultiplier
    let chargeSameProject = chargeBase * s.sameProjectChargeScale
    let chargeSameTopic = chargeBase * s.sameTopicChargeScale
    let cutoffSq: Float = 500 * 500
    var useBH = n > 256
    if useBH {
        var mnX = x[0], mxX = x[0], mnY = y[0], mxY = y[0], mnZ = z[0], mxZ = z[0]
        for i in 1..<n {
            if x[i] < mnX { mnX = x[i] } else if x[i] > mxX { mxX = x[i] }
            if y[i] < mnY { mnY = y[i] } else if y[i] > mxY { mxY = y[i] }
            if z[i] < mnZ { mnZ = z[i] } else if z[i] > mxZ { mxZ = z[i] }
        }
        if max(mxX - mnX, max(mxY - mnY, mxZ - mnZ)) < 1000 { useBH = false }
    }
    let hasGalaxies = !s.galaxyGroup.isEmpty && !s.galaxyCenters.isEmpty
    let galaxyGroup_ = s.galaxyGroup
    if !useBH {
        for i in 0..<n {
            let xi = x[i], yi = y[i], zi = z[i]
            let pg_i = hasProjects ? projectGroup[i] : 0
            let tg_i = hasTopics ? topicGroup[i] : -1
            let gg_i = hasGalaxies ? galaxyGroup_[i] : 0
            for j in (i+1)..<n {
                // Skip charge between nodes in different galaxies
                if hasGalaxies && galaxyGroup_[j] != gg_i { continue }
                let dx = xi - x[j], dy = yi - y[j], dz = zi - z[j]
                var dSq = dx*dx + dy*dy + dz*dz
                if dSq > cutoffSq { continue }
                if dSq < 1 { dSq = 1 }
                let charge: Float
                if hasProjects && projectGroup[j] != pg_i { charge = chargeCross }
                else if hasTopics && topicGroup[j] == tg_i { charge = chargeSameTopic }
                else if hasProjects { charge = chargeSameProject }
                else { charge = chargeBase }
                let d = sqrt(dSq), fm = charge / dSq
                let ux = dx/d, uy = dy/d, uz = dz/d
                fx[i] += ux*fm; fy[i] += uy*fm; fz[i] += uz*fm
                fx[j] -= ux*fm; fy[j] -= uy*fm; fz[j] -= uz*fm
            }
        }
    } else {
        computeChargeForcesBH(x: x, y: y, z: z, n: n,
            projectGroup: projectGroup, topicGroup: topicGroup,
            galaxyGroup: galaxyGroup_,
            hasProjects: hasProjects, hasTopics: hasTopics, hasGalaxies: hasGalaxies,
            chargeBase: chargeBase, chargeCross: chargeCross,
            chargeSameProject: chargeSameProject, chargeSameTopic: chargeSameTopic,
            fx: &fx, fy: &fy, fz: &fz)
    }

    // --- Springs (skip cross-galaxy edges) ---
    for (si, ti) in s.edgeIndices {
        // Cross-galaxy edges produce zero spring force — visual separation between galaxies
        if hasGalaxies && galaxyGroup_[si] != galaxyGroup_[ti] { continue }
        let dx = x[ti]-x[si], dy = y[ti]-y[si], dz = z[ti]-z[si]
        var d = sqrt(dx*dx + dy*dy + dz*dz); if d < 1 { d = 1 }
        let cross = hasProjects && projectGroup[si] != projectGroup[ti]
        let rest = cross ? s.crossProjectSpringLength : s.springLength
        let f = s.springStrength * (d - rest)
        let efx = (dx/d)*f, efy = (dy/d)*f, efz = (dz/d)*f
        fx[si] += efx; fy[si] += efy; fz[si] += efz
        fx[ti] -= efx; fy[ti] -= efy; fz[ti] -= efz
    }

    // --- Topic centroids ---
    // Structural forces (cohesion + centroid repulsion) decay as sqrt(alpha) instead of alpha.
    // Pre-multiply by 1/sqrt(alpha) so integration's `* alpha` yields `* sqrt(alpha)`.
    // At alpha=1.0: full strength. At alpha=0.01: 10% (vs 1% for other forces).
    // This maintains cluster separation while still allowing the sim to settle.
    let alphaInv = 1.0 / max(s.alpha, 0.001)
    let topicN = (topicGroup.max() ?? -1) + 1
    if topicN > 1 {
        var tSX = [Float](repeating: 0, count: topicN), tSY = [Float](repeating: 0, count: topicN), tSZ = [Float](repeating: 0, count: topicN)
        var tCnt = [Int](repeating: 0, count: topicN)
        var tMem = [[Int]](repeating: [], count: topicN)
        for i in 0..<n { let g = topicGroup[i]; tSX[g] += x[i]; tSY[g] += y[i]; tSZ[g] += z[i]; tCnt[g] += 1; tMem[g].append(i) }
        for i in 0..<n {
            let g = topicGroup[i]; let c = Float(tCnt[g]); if c < 2 { continue }
            fx[i] += (tSX[g]/c - x[i]) * s.topicCohesionStrength * alphaInv
            fy[i] += (tSY[g]/c - y[i]) * s.topicCohesionStrength * alphaInv
            fz[i] += (tSZ[g]/c - z[i]) * s.topicCohesionStrength * alphaInv
        }
        let tpg = s.topicProjectGroup
        var tFX = [Float](repeating: 0, count: topicN), tFY = [Float](repeating: 0, count: topicN), tFZ = [Float](repeating: 0, count: topicN)
        for g1 in 0..<topicN where tCnt[g1] > 0 {
            let c1 = SIMD3<Float>(tSX[g1], tSY[g1], tSZ[g1]) / Float(tCnt[g1])
            for g2 in (g1+1)..<topicN where tCnt[g2] > 0 {
                guard g1 < tpg.count && g2 < tpg.count, tpg[g1] == tpg[g2] else { continue }
                let c2 = SIMD3<Float>(tSX[g2], tSY[g2], tSZ[g2]) / Float(tCnt[g2])
                var d = c1 - c2; var pd = simd_length(d); if pd < 1 { pd = 1; d = .init(.random(in:-1...1),.random(in:-1...1),.random(in:-1...1)) }
                let fv = (d/pd) * (s.topicCentroidRepulsion * alphaInv / (pd*pd))
                let f1 = 1/sqrt(Float(tCnt[g1])), f2 = 1/sqrt(Float(tCnt[g2]))
                tFX[g1] += fv.x*f1; tFY[g1] += fv.y*f1; tFZ[g1] += fv.z*f1
                tFX[g2] -= fv.x*f2; tFY[g2] -= fv.y*f2; tFZ[g2] -= fv.z*f2
            }
        }
        for g in 0..<topicN where tCnt[g] > 0 && (tFX[g] != 0 || tFY[g] != 0 || tFZ[g] != 0) {
            for i in tMem[g] { fx[i] += tFX[g]; fy[i] += tFY[g]; fz[i] += tFZ[g] }
        }
    }

    // --- Project centroids (non-linear cohesion, same-galaxy only) ---
    if hasProjects {
        let gN = (projectGroup.max() ?? -1) + 1
        if gN > 0 {
            // Build project group → galaxy mapping
            var projGalaxy = [Int](repeating: -1, count: gN)
            if hasGalaxies { for i in 0..<n { let g = projectGroup[i]; if projGalaxy[g] < 0 { projGalaxy[g] = galaxyGroup_[i] } } }
            var gSX = [Float](repeating:0,count:gN), gSY = [Float](repeating:0,count:gN), gSZ = [Float](repeating:0,count:gN)
            var gCnt = [Int](repeating:0,count:gN); var pMem = [[Int]](repeating:[],count:gN)
            for i in 0..<n { let g=projectGroup[i]; gSX[g]+=x[i]; gSY[g]+=y[i]; gSZ[g]+=z[i]; gCnt[g]+=1; pMem[g].append(i) }
            var gRad = [[Float]](repeating:[],count:gN)
            for i in 0..<n { let g=projectGroup[i]; let c=Float(gCnt[g]); if c<2{continue}; let cx=gSX[g]/c,cy=gSY[g]/c,cz=gSZ[g]/c; let dx=x[i]-cx,dy=y[i]-cy,dz=z[i]-cz; gRad[g].append(sqrt(dx*dx+dy*dy+dz*dz)) }
            var gRefR = [Float](repeating:30,count:gN)
            for g in 0..<gN { guard !gRad[g].isEmpty else{continue}; gRad[g].sort(); gRefR[g]=max(30,gRad[g][gRad[g].count*3/4]) }
            for i in 0..<n { let g=projectGroup[i]; let c=Float(gCnt[g]); if c<2{continue}; let cx=gSX[g]/c,cy=gSY[g]/c,cz=gSZ[g]/c; let dx=cx-x[i],dy=cy-y[i],dz=cz-z[i]; let dist=sqrt(dx*dx+dy*dy+dz*dz); let r=max(1.0,dist/gRefR[g]); let sc=r*r; fx[i]+=dx*s.cohesionStrength*sc*alphaInv; fy[i]+=dy*s.cohesionStrength*sc*alphaInv; fz[i]+=dz*s.cohesionStrength*sc*alphaInv }
            var pFX=[Float](repeating:0,count:gN),pFY=[Float](repeating:0,count:gN),pFZ=[Float](repeating:0,count:gN)
            for g1 in 0..<gN where gCnt[g1]>0 { let c1=SIMD3<Float>(gSX[g1],gSY[g1],gSZ[g1])/Float(gCnt[g1]); for g2 in (g1+1)..<gN where gCnt[g2]>0 { if hasGalaxies && projGalaxy[g1] != projGalaxy[g2] { continue }; let c2=SIMD3<Float>(gSX[g2],gSY[g2],gSZ[g2])/Float(gCnt[g2]); var d=c1-c2; var pd=simd_length(d); if pd<1{pd=1;d = .init(.random(in:-1...1),.random(in:-1...1),.random(in:-1...1))}; let fv=(d/pd)*(s.centroidRepulsion*alphaInv/(pd*pd)); let f1=1/sqrt(Float(gCnt[g1])),f2=1/sqrt(Float(gCnt[g2])); pFX[g1]+=fv.x*f1;pFY[g1]+=fv.y*f1;pFZ[g1]+=fv.z*f1; pFX[g2]-=fv.x*f2;pFY[g2]-=fv.y*f2;pFZ[g2]-=fv.z*f2 } }
            for g in 0..<gN where gCnt[g]>0 && (pFX[g] != 0||pFY[g] != 0||pFZ[g] != 0) { for i in pMem[g]{fx[i]+=pFX[g];fy[i]+=pFY[g];fz[i]+=pFZ[g]} }
        }
    }

    // --- Center gravity (per-galaxy or single center) ---
    let galaxyCenters = s.galaxyCenters
    if hasGalaxies {
        for i in 0..<n {
            let c = galaxyCenters[galaxyGroup_[i]]
            fx[i]+=(c.x-x[i])*s.centerStrength
            fy[i]+=(c.y-y[i])*s.centerStrength
            fz[i]+=(c.z-z[i])*s.centerStrength
        }
    } else {
        let c = s.center
        for i in 0..<n { fx[i]+=(c.x-x[i])*s.centerStrength; fy[i]+=(c.y-y[i])*s.centerStrength; fz[i]+=(c.z-z[i])*s.centerStrength }
    }

    return ForceComputationResult(fx: fx, fy: fy, fz: fz)
}
