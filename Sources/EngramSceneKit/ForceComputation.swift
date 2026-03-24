// Barnes-Hut octree for GPU force computation (used by MetalForceCompute).

import Foundation
import simd

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

