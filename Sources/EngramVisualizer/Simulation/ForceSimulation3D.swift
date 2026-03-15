import Foundation
import Accelerate
import simd
import os
import EngramSceneKit

private let forceSimLog = Logger(subsystem: "com.claudememory.visualizer", category: "ForceSimulation3D")

// MARK: - Barnes-Hut Octree for O(n log n) charge computation

private struct OctreeNode {
    var cx: Float, cy: Float, cz: Float   // cell geometric center
    var halfSize: Float
    var comX: Float, comY: Float, comZ: Float  // center of mass
    var mass: Float                         // body count (as Float for division)
    var child0: Int32, child1: Int32, child2: Int32, child3: Int32
    var child4: Int32, child5: Int32, child6: Int32, child7: Int32  // -1 = empty
    var bodyIndex: Int32  // leaf with 1 body: its index; else -1

    static func empty(cx: Float, cy: Float, cz: Float, halfSize: Float) -> OctreeNode {
        OctreeNode(cx: cx, cy: cy, cz: cz, halfSize: halfSize,
                   comX: 0, comY: 0, comZ: 0, mass: 0,
                   child0: -1, child1: -1, child2: -1, child3: -1,
                   child4: -1, child5: -1, child6: -1, child7: -1,
                   bodyIndex: -1)
    }

    func child(_ octant: Int) -> Int32 {
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
private func buildOctree(x: [Float], y: [Float], z: [Float], n: Int) -> [OctreeNode] {
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

    let maxDepth = 40

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
private func octant(px: Float, py: Float, pz: Float, cx: Float, cy: Float, cz: Float) -> Int {
    var o = 0
    if px >= cx { o |= 1 }
    if py >= cy { o |= 2 }
    if pz >= cz { o |= 4 }
    return o
}

@inline(__always)
private func childCenter(parentCX: Float, parentCY: Float, parentCZ: Float,
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
private func computeChargeForcesBH(
    x: [Float], y: [Float], z: [Float], n: Int,
    projectGroup: [Int], topicGroup: [Int],
    hasProjects: Bool, hasTopics: Bool,
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

/// 3D force-directed graph layout engine. Same split architecture as ForceSimulation:
/// O(n²) force computation runs async on @ForceSimulatorActor, O(n) integration runs sync at 60fps.
@MainActor
final class ForceSimulation3D {
    // SoA layout for cache-friendly iteration
    private var ids: [UUID] = []
    private var x: [Float] = []
    private var y: [Float] = []
    private var z: [Float] = []
    private var vx: [Float] = []
    private var vy: [Float] = []
    private var vz: [Float] = []
    private var pinned: [Bool] = []
    private var idToIndex: [UUID: Int] = [:]
    private var edgeIndices: [(Int, Int)] = []
    private var edgeIndexSet: Set<UInt64> = []  // packed (si << 32 | ti) for O(1) duplicate check
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

    /// Pending force result from async computation — written off-MainActor, read by tick().
    /// Using OSAllocatedUnfairLock for safe cross-isolation handoff without MainActor hop.
    private let pendingForces = OSAllocatedUnfairLock<ForceResult?>(initialState: nil)

    private(set) var positions: [UUID: SIMD3<Float>] = [:]
    var nodeCount: Int { ids.count }

    // Force parameters (tuned for 3D — slightly stronger since space is larger)
    private let springLength: Float = 240
    private let crossProjectSpringLength: Float = 400
    private let springStrength: Float = 0.0004
    private let chargeStrength: Float = 500
    private let crossChargeMultiplier: Float = 3.0
    private let sameProjectChargeScale: Float = 1.0   // pre-GPU: no reduction for same-project different-topic
    private let sameTopicChargeScale: Float = 0.35     // pre-GPU value (was 0.25 post-GPU)
    private let centerStrength: Float = 0.006
    private let cohesionStrength: Float = 0.0015       // pre-GPU value (was 0.004 post-GPU)
    private let centroidRepulsion: Float = 2500         // pre-GPU value (was 4000 post-GPU)
    private let topicCohesionStrength: Float = 0.009    // pre-GPU value (was 0.005 post-GPU)
    private let topicCentroidRepulsion: Float = 3500    // pre-GPU value (was 6000 post-GPU)
    private let damping: Float = 0.78
    private let maxSpeed: Float = 12.0

    private(set) var alpha: Float = 1.0
    private let alphaDecay: Float = 0.995
    private let alphaFloor: Float = 0.01
    private(set) var tickInFlight = false
    private var dispatchedToGPU = false  // true when current dispatch is GPU (can hang), false for CPU (always completes)
    private(set) var forceAge: Int = 100
    private(set) var smoothedAttenuation: Float = 0.001
    private var topologyVersion: UInt64 = 0
    /// After a GPU timeout, skip GPU path for this many ticks to prevent concurrent dispatch.
    private var gpuCooldownTicks: Int = 0

    /// Set by addNode/addEdge, drained by tick(). Coalesces multiple topology changes
    /// into a single wake per tick instead of one per call.
    private var hasPendingTopologyChanges = false

    /// Set by topology changes, cleared after CSR rebuild in dispatchForceComputation.
    /// CSR (Compressed Sparse Row) structures are pre-built for GPU gather kernels.
    private var topologyDirtyForGPU = true

    // MARK: - Local wake state
    // When a node is inserted into a settled sim, we run a lightweight CPU-only
    // force computation on just the neighborhood (~50-100 nodes) instead of waking
    // the full O(n) simulation. This avoids the 300x per-frame cost cliff.

    /// Indices of nodes in the active local wake neighborhood.
    private var localWakeIndices: Set<Int> = []

    /// True when a local wake is active (node inserted while settled).
    private(set) var isLocalWake = false

    /// Frames remaining in local wake. Counts down from localWakeMaxFrames.
    private var localWakeFramesRemaining = 0
    private let localWakeMaxFrames = 15  // ~250ms at 60fps

    /// Index of the newly inserted node that triggered the local wake.
    private var localWakeNewNodeIndex: Int? = nil

    /// Whether the neighborhood has been expanded (done once on first local tick).
    private var localWakeExpanded = false

    /// Metal compute for GPU-accelerated charge force calculation.
    private var metalForceCompute: MetalForceCompute? = MetalForceCompute()

    var center: SIMD3<Float> = .zero
    var isActive: Bool = true

    /// True when the simulation has converged (velocities near-zero for consecutive frames).
    /// When settled, tick() skips force dispatch and position sync to save CPU/GPU.
    private(set) var isSettled = false
    private(set) var settledFrameCount = 0
    private(set) var framesSinceWake = 0
    /// Minimum frames after wake() before settle is allowed. Prevents premature settling
    /// when the first async force results haven't arrived yet.
    private let settleGuardFrames = 30  // ~0.5s at 60fps

    /// Max speed² from the last tick. Used by the Timer to throttle visual updates
    /// when nodes are barely moving (convergence tail).
    private(set) var lastMaxSpeedSq: Float = 0

    /// Wake the simulation from settled state (e.g. after topology change or user interaction).
    func wake() {
        // Cancel any active local wake — full wake supersedes it
        clearLocalWake()
        isSettled = false
        settledFrameCount = 0
        framesSinceWake = 0
        forceAge = 2
        // Reset alpha so center gravity and other alpha-scaled forces are meaningful.
        // Without this, alpha decays to 0.01 after ~15s and never recovers — making
        // migration animations crawl because center force is 100x weaker.
        alpha = 1.0
        // Reset smoothedAttenuation so forces apply at meaningful strength immediately.
        // Without this, smoothedAttenuation ramps from near-zero at 4%/frame — taking
        // ~30 frames to reach useful values — causing newly added nodes to appear stuck.
        smoothedAttenuation = pow(damping, Float(forceAge))
    }

    // MARK: - Graph Management

    func updateGraph(nodeIds: Set<UUID>, edges: [(UUID, UUID)],
                     projectForNode: [UUID: String] = [:], topicForNode: [UUID: String] = [:]) {
        // Remove nodes no longer in the graph
        var keep = [Bool](repeating: false, count: ids.count)
        for (i, id) in ids.enumerated() { keep[i] = nodeIds.contains(id) }

        var newIds: [UUID] = []
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
        edgeIndexSet = Set(edgeIndices.map { UInt64($0.0) << 32 | UInt64($0.1) })

        storedFx = [Float](repeating: 0, count: ids.count)
        storedFy = [Float](repeating: 0, count: ids.count)
        storedFz = [Float](repeating: 0, count: ids.count)

        topologyVersion &+= 1
        topologyDirtyForGPU = true
        isSettled = false; settledFrameCount = 0
        forceAge = 2
        rebuildPositions()
    }

    /// Surgically remove nodes without rebuilding the entire graph.
    /// Compacts SoA arrays and remaps edge indices.
    func removeNodes(_ idsToRemove: Set<UUID>) {
        guard !idsToRemove.isEmpty else { return }
        clearLocalWake()  // indices become invalid after compaction
        let oldCount = ids.count

        // Build old→new index mapping
        var oldToNew = [Int](repeating: -1, count: oldCount)
        var newIdx = 0
        for i in 0..<oldCount {
            if !idsToRemove.contains(ids[i]) {
                oldToNew[i] = newIdx
                newIdx += 1
            }
        }

        // Compact SoA arrays
        let cap = newIdx
        var newIds: [UUID] = [];      newIds.reserveCapacity(cap)
        var newX: [Float] = [];       newX.reserveCapacity(cap)
        var newY: [Float] = [];       newY.reserveCapacity(cap)
        var newZ: [Float] = [];       newZ.reserveCapacity(cap)
        var newVx: [Float] = [];      newVx.reserveCapacity(cap)
        var newVy: [Float] = [];      newVy.reserveCapacity(cap)
        var newVz: [Float] = [];      newVz.reserveCapacity(cap)
        var newPinned: [Bool] = [];   newPinned.reserveCapacity(cap)
        var newProjGroup: [Int] = []; newProjGroup.reserveCapacity(cap)
        var newTopicGrp: [Int] = [];  newTopicGrp.reserveCapacity(cap)

        for i in 0..<oldCount where oldToNew[i] >= 0 {
            newIds.append(ids[i])
            newX.append(x[i]); newY.append(y[i]); newZ.append(z[i])
            newVx.append(vx[i]); newVy.append(vy[i]); newVz.append(vz[i])
            newPinned.append(pinned[i])
            newProjGroup.append(projectGroup[i])
            newTopicGrp.append(topicGroup[i])
        }

        ids = newIds; x = newX; y = newY; z = newZ
        vx = newVx; vy = newVy; vz = newVz; pinned = newPinned
        projectGroup = newProjGroup; topicGroup = newTopicGrp

        // Rebuild index
        idToIndex.removeAll(keepingCapacity: true)
        for (i, id) in ids.enumerated() { idToIndex[id] = i }

        // Remap edge indices (drop edges that touch removed nodes)
        edgeIndices = edgeIndices.compactMap { (si, ti) in
            let nsi = oldToNew[si]; let nti = oldToNew[ti]
            guard nsi >= 0, nti >= 0 else { return nil }
            return (nsi, nti)
        }
        edgeIndexSet = Set(edgeIndices.map { UInt64($0.0) << 32 | UInt64($0.1) })

        // Compact stored forces
        storedFx = [Float](repeating: 0, count: cap)
        storedFy = [Float](repeating: 0, count: cap)
        storedFz = [Float](repeating: 0, count: cap)

        // Remove from positions dict
        for id in idsToRemove { positions.removeValue(forKey: id) }

        hasPendingTopologyChanges = true
        topologyDirtyForGPU = true
    }

    /// Surgically insert a single node without rebuilding the entire graph.
    /// When the sim is settled, activates local wake instead of full wake.
    func addNode(_ id: UUID, project: String, topic: String) {
        guard idToIndex[id] == nil else { return }

        let newIndex = ids.count

        // Smart positioning: place at project cluster centroid + jitter when settled,
        // random sphere position when sim is actively converging.
        let position: SIMD3<Float>
        if (isSettled || isLocalWake), let pg = projectToGroup[project] {
            position = projectCentroid(group: pg, jitter: 15.0)
        } else {
            position = randomSpherePosition()
        }

        ids.append(id)
        x.append(position.x); y.append(position.y); z.append(position.z)
        vx.append(0); vy.append(0); vz.append(0)
        pinned.append(false)
        idToIndex[id] = newIndex

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
        positions[id] = position
        topologyDirtyForGPU = true

        // Route to local wake or full wake
        if isSettled && !isLocalWake {
            // First insert into a settled sim → activate local wake
            isLocalWake = true
            localWakeFramesRemaining = localWakeMaxFrames
            localWakeNewNodeIndex = newIndex
            localWakeIndices = [newIndex]
            topologyVersion &+= 1  // so render pipeline picks up the new node
        } else if isLocalWake {
            // Additional insert during active local wake → extend the set
            localWakeIndices.insert(newIndex)
            topologyVersion &+= 1
        } else {
            // Sim already awake → normal full-wake path
            hasPendingTopologyChanges = true
        }
    }

    /// Surgically insert a single edge without rebuilding the entire graph.
    func addEdge(from source: UUID, to target: UUID) {
        guard let si = idToIndex[source], let ti = idToIndex[target] else { return }
        let key = UInt64(si) << 32 | UInt64(ti)
        guard edgeIndexSet.insert(key).inserted else { return }
        edgeIndices.append((si, ti))
        topologyDirtyForGPU = true

        if isLocalWake {
            // Extend neighborhood to include edge endpoints
            localWakeIndices.insert(si)
            localWakeIndices.insert(ti)
        } else {
            hasPendingTopologyChanges = true
        }
    }

    /// Write external positions (e.g. from 2D positions + z jitter) into internal arrays.
    func setPositions(_ positions: [UUID: SIMD3<Float>]) {
        clearLocalWake()
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
    func pin(_ id: UUID) {
        guard let i = idToIndex[id] else { return }
        pinned[i] = true
        vx[i] = 0; vy[i] = 0; vz[i] = 0
    }

    /// Unpin a node so the force simulation resumes moving it.
    func unpin(_ id: UUID) {
        guard let i = idToIndex[id] else { return }
        pinned[i] = false
    }

    /// Set a node's position directly and zero its velocity.
    func setPosition(_ id: UUID, to pos: SIMD3<Float>) {
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

        // Coalesce topology changes into a single wake per tick regardless of how many
        // addNode/addEdge calls arrived since the last tick.
        if hasPendingTopologyChanges {
            hasPendingTopologyChanges = false
            topologyVersion &+= 1
            isSettled = false
            settledFrameCount = 0
            // Full topology change during local wake → escalate to full wake
            if isLocalWake { clearLocalWake() }
        }

        // Local wake path: lightweight CPU-only integration for neighborhood only.
        // Runs instead of full sim when a node was inserted while settled.
        if isLocalWake {
            tickLocalWake()
            return
        }

        // Fast path: when settled, skip all work (no integration, no sync, no force dispatch)
        if isSettled { return }

        // Async force computation — CPU Barnes-Hut on background thread.
        // GPU reserved for rendering to avoid contention/hangs.
        let forceStart = CFAbsoluteTimeGetCurrent()
        let forceMethod = "cpu"

        // Drain pending force results from async dispatch
        if let result = pendingForces.withLock({ value -> ForceResult? in
            let r = value; value = nil; return r
        }) {
            storedFx = result.fx; storedFy = result.fy; storedFz = result.fz
            tickInFlight = false
        }

        let forceMs = (CFAbsoluteTimeGetCurrent() - forceStart) * 1000.0

        // Integrate with whatever forces we have (current or from previous frame)
        let integrateStart = CFAbsoluteTimeGetCurrent()
        let hasForcesComputed = storedFx.count == n
        var maxSpeedSq: Float = 0
        var totalKineticEnergy: Float = 0
        for i in 0..<n where !pinned[i] {
            if hasForcesComputed {
                vx[i] += storedFx[i]
                vy[i] += storedFy[i]
                vz[i] += storedFz[i]
            }
            vx[i] *= damping; vy[i] *= damping; vz[i] *= damping

            let speedSq = vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]
            maxSpeedSq = max(maxSpeedSq, speedSq)
            totalKineticEnergy += speedSq
            if speedSq > maxSpeed * maxSpeed {
                let scale = maxSpeed / sqrt(speedSq)
                vx[i] *= scale; vy[i] *= scale; vz[i] *= scale
            }

            x[i] += vx[i]; y[i] += vy[i]; z[i] += vz[i]
        }
        alpha = max(alpha * alphaDecay, alphaFloor)
        lastMaxSpeedSq = maxSpeedSq
        framesSinceWake += 1

        // Settle detection — simple threshold like JS version.
        // No scale-adaptive hacks needed since forces are synchronous (no micro-oscillation).
        let settleThreshold: Float = 0.01
        if hasForcesComputed
            && framesSinceWake >= settleGuardFrames
            && maxSpeedSq < settleThreshold {
            settledFrameCount += 1
            if settledFrameCount >= 30 {
                isSettled = true
                syncPositions()
                return
            }
        } else if maxSpeedSq >= settleThreshold {
            settledFrameCount = 0
        }

        syncPositions()

        // Dispatch CPU force computation on background thread.
        // GPU is reserved for rendering — no GPU force compute to avoid contention.
        if !tickInFlight {
            tickInFlight = true
            dispatchedToGPU = false
            let input = ForceComputationInput(
                n: n, x: x, y: y, z: z,
                projectGroup: projectGroup, topicGroup: topicGroup,
                edgeIndices: edgeIndices, topicProjectGroup: topicProjectGroup,
                alpha: alpha, center: center,
                chargeStrength: chargeStrength, crossChargeMultiplier: crossChargeMultiplier,
                sameProjectChargeScale: sameProjectChargeScale, sameTopicChargeScale: sameTopicChargeScale,
                centerStrength: centerStrength,
                springLength: springLength, crossProjectSpringLength: crossProjectSpringLength,
                springStrength: springStrength,
                cohesionStrength: cohesionStrength, centroidRepulsion: centroidRepulsion,
                topicCohesionStrength: topicCohesionStrength, topicCentroidRepulsion: topicCentroidRepulsion
            )
            Task.detached(priority: .userInitiated) { [pendingForces] in
                let result = computeAllForces(input)
                pendingForces.withLock { $0 = ForceResult(fx: result.fx, fy: result.fy, fz: result.fz) }
            }
        }

        let integrateMs = (CFAbsoluteTimeGetCurrent() - integrateStart) * 1000.0
        let totalTickMs = (CFAbsoluteTimeGetCurrent() - forceStart) * 1000.0
        if framesSinceWake % 60 == 1 || totalTickMs > 20 {
            print("[engram:sim] tick n=\(n) method=\(forceMethod) force=\(String(format: "%.1f", forceMs))ms integrate=\(String(format: "%.1f", integrateMs))ms total=\(String(format: "%.1f", totalTickMs))ms maxSpeedSq=\(String(format: "%.4f", maxSpeedSq)) alpha=\(String(format: "%.4f", alpha)) settled=\(isSettled)")
        }
    }

    // MARK: - Async force computation

    private struct SimState: Sendable {
        let n: Int
        let x: [Float], y: [Float], z: [Float]
        let projectGroup: [Int], topicGroup: [Int]
        let edgeIndices: [(Int, Int)]
        let topicProjectGroup: [Int]
        let alpha: Float, center: SIMD3<Float>
        let chargeStrength: Float, crossChargeMultiplier: Float
        let sameProjectChargeScale: Float, sameTopicChargeScale: Float
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

        // Capture state needed for CPU forces (springs, cohesion, center)
        let state = SimState(
            n: n, x: x, y: y, z: z,
            projectGroup: projectGroup, topicGroup: topicGroup,
            edgeIndices: edgeIndices, topicProjectGroup: topicProjectGroup,
            alpha: alpha, center: center,
            chargeStrength: chargeStrength, crossChargeMultiplier: crossChargeMultiplier,
            sameProjectChargeScale: sameProjectChargeScale, sameTopicChargeScale: sameTopicChargeScale,
            centerStrength: centerStrength,
            springLength: springLength, crossProjectSpringLength: crossProjectSpringLength,
            springStrength: springStrength,
            cohesionStrength: cohesionStrength, centroidRepulsion: centroidRepulsion,
            topicCohesionStrength: topicCohesionStrength, topicCentroidRepulsion: topicCentroidRepulsion
        )

        // Try full GPU path (charge + springs + cohesion all on GPU)
        // Skip GPU during cooldown after a timeout to prevent concurrent buffer access.
        // Skip GPU entirely for large graphs — O(n²) charge kernel hangs at 4K+ nodes.
        // CPU Barnes-Hut O(n log n) is fast enough (~5ms for 8K nodes).
        if let metal = metalForceCompute, n >= 16, n <= 4096, metal.isFullSimAvailable, gpuCooldownTicks == 0 {
            dispatchedToGPU = true
            // Rebuild CSR topology buffers when graph structure changed
            if topologyDirtyForGPU {
                metal.rebuildTopologyBuffers(
                    nodeCount: n,
                    edges: edgeIndices,
                    projectGroups: projectGroup,
                    topicGroups: topicGroup
                )
                topologyDirtyForGPU = false
            }

            let positions = (x: Array(x), y: Array(y), z: Array(z))
            let projGroups = Array(projectGroup)
            let topGroups = Array(topicGroup)

            Task.detached(priority: .userInitiated) { [pendingForces] in
                let result = await metal.dispatchFullSimulation(
                    positions: positions,
                    projectGroups: projGroups,
                    topicGroups: topGroups,
                    chargeStrength: state.chargeStrength,
                    crossChargeMultiplier: state.crossChargeMultiplier,
                    sameTopicChargeScale: state.sameTopicChargeScale,
                    sameProjectChargeScale: state.sameProjectChargeScale,
                    springLength: state.springLength,
                    crossProjectSpringLength: state.crossProjectSpringLength,
                    springStrength: state.springStrength,
                    cohesionStrength: state.cohesionStrength,
                    centroidRepulsion: state.centroidRepulsion,
                    topicCohesionStrength: state.topicCohesionStrength,
                    topicCentroidRepulsion: state.topicCentroidRepulsion,
                    centerStrength: state.centerStrength,
                    center: state.center,
                    alpha: state.alpha,
                    damping: 0.78,
                    maxSpeed: 12.0
                )
                pendingForces.withLock {
                    $0 = ForceResult(fx: result.fx, fy: result.fy, fz: result.fz)
                }
            }
            return
        }

        // Fallback: GPU charge + CPU springs/cohesion (for older Metal or small graphs)
        if let metal = metalForceCompute, n >= 16, n <= 4096, gpuCooldownTicks == 0 {
            dispatchedToGPU = true
            let positions = (x: Array(x), y: Array(y), z: Array(z))
            let projGroups = Array(projectGroup)
            let topGroups = Array(topicGroup)

            Task.detached(priority: .userInitiated) { [pendingForces] in
                async let chargeResult = metal.dispatchChargeForces(
                    positions: positions,
                    projectGroups: projGroups,
                    topicGroups: topGroups,
                    chargeStrength: state.chargeStrength,
                    crossChargeMultiplier: state.crossChargeMultiplier,
                    sameTopicChargeScale: state.sameTopicChargeScale,
                    sameProjectChargeScale: state.sameProjectChargeScale
                )
                let cpuForces = ForceSimulation3D.computeCPUOnlyForces(state)
                let charge = await chargeResult
                let nn = min(charge.fx.count, cpuForces.fx.count)
                var cfx = [Float](repeating: 0, count: nn)
                var cfy = [Float](repeating: 0, count: nn)
                var cfz = [Float](repeating: 0, count: nn)
                for i in 0..<nn {
                    cfx[i] = charge.fx[i] + cpuForces.fx[i]
                    cfy[i] = charge.fy[i] + cpuForces.fy[i]
                    cfz[i] = charge.fz[i] + cpuForces.fz[i]
                }
                let result = ForceResult(fx: cfx, fy: cfy, fz: cfz)
                pendingForces.withLock { $0 = result }
            }
            return
        }

        // CPU fallback — n > 4096 (Barnes-Hut O(n log n)) or no Metal
        dispatchedToGPU = false
        Task.detached(priority: .userInitiated) { [pendingForces] in
            let result = Self.computeForces(state)
            pendingForces.withLock { $0 = result }
        }
    }

    /// Compute CPU-only forces (spring attraction, topic/project cohesion, center gravity).
    /// CPU-only forces (springs, cohesion, center) — nonisolated static so it can run off MainActor.
    /// Used by GPU path to combine with GPU-computed charge forces.
    ///
    /// Performance: centroid repulsion uses precomputed per-group index lists to avoid
    /// O(groups² × n) full scans. Without this, 3248 nodes with hundreds of topic groups
    /// causes multi-second stalls (the old code did `for i in 0..<n where topicGroup[i] == g`
    /// inside a pairwise loop).
    private nonisolated static func computeCPUOnlyForces(_ s: SimState) -> ForceResult {
        let n = s.n
        let x = s.x, y = s.y, z = s.z
        let projectGroup = s.projectGroup, topicGroup = s.topicGroup
        var fx = [Float](repeating: 0, count: n)
        var fy = [Float](repeating: 0, count: n)
        var fz = [Float](repeating: 0, count: n)

        let hasProjects = !projectGroup.isEmpty
        let hasTopics = !topicGroup.isEmpty

        // Spring attraction
        for (si, ti) in s.edgeIndices {
            let dx = x[ti] - x[si], dy = y[ti] - y[si], dz = z[ti] - z[si]
            var d = sqrt(dx * dx + dy * dy + dz * dz)
            if d < 1 { d = 1 }
            let cross = hasProjects && projectGroup[si] != projectGroup[ti]
            let rest = cross ? s.crossProjectSpringLength : s.springLength
            let force = s.springStrength * (d - rest)
            let efx = (dx / d) * force, efy = (dy / d) * force, efz = (dz / d) * force
            fx[si] += efx; fy[si] += efy; fz[si] += efz
            fx[ti] -= efx; fy[ti] -= efy; fz[ti] -= efz
        }

        // Topic centroid forces
        let topicN = hasTopics ? ((topicGroup.max() ?? -1) + 1) : 0
        if topicN > 1 {
            var tSumX = [Float](repeating: 0, count: topicN)
            var tSumY = [Float](repeating: 0, count: topicN)
            var tSumZ = [Float](repeating: 0, count: topicN)
            var tCount = [Int](repeating: 0, count: topicN)
            // Build per-group index lists (O(n) once, avoids O(n) scan per pair)
            var topicMembers = [[Int]](repeating: [], count: topicN)
            for i in 0..<n {
                let g = topicGroup[i]
                tSumX[g] += x[i]; tSumY[g] += y[i]; tSumZ[g] += z[i]; tCount[g] += 1
                topicMembers[g].append(i)
            }
            for i in 0..<n {
                let g = topicGroup[i]; let cnt = Float(tCount[g])
                if cnt < 2 { continue }
                let cx = tSumX[g] / cnt, cy = tSumY[g] / cnt, cz = tSumZ[g] / cnt
                fx[i] += (cx - x[i]) * s.topicCohesionStrength
                fy[i] += (cy - y[i]) * s.topicCohesionStrength
                fz[i] += (cz - z[i]) * s.topicCohesionStrength
            }
            let tpg = s.topicProjectGroup
            // Accumulate per-group force vectors, then distribute in one pass
            var topicForceX = [Float](repeating: 0, count: topicN)
            var topicForceY = [Float](repeating: 0, count: topicN)
            var topicForceZ = [Float](repeating: 0, count: topicN)
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
                    let force = s.topicCentroidRepulsion / (tdist * tdist)
                    let tfx = (tdx / tdist) * force, tfy = (tdy / tdist) * force, tfz = (tdz / tdist) * force
                    let f1 = 1.0 / Float(tCount[g1]), f2 = 1.0 / Float(tCount[g2])
                    topicForceX[g1] += tfx * f1; topicForceY[g1] += tfy * f1; topicForceZ[g1] += tfz * f1
                    topicForceX[g2] -= tfx * f2; topicForceY[g2] -= tfy * f2; topicForceZ[g2] -= tfz * f2
                }
            }
            // Distribute accumulated per-group forces to members (O(n) total)
            for g in 0..<topicN where tCount[g] > 0 {
                let tfx = topicForceX[g], tfy = topicForceY[g], tfz = topicForceZ[g]
                if tfx == 0 && tfy == 0 && tfz == 0 { continue }
                for i in topicMembers[g] {
                    fx[i] += tfx; fy[i] += tfy; fz[i] += tfz
                }
            }
        }

        // Project centroid forces (with non-linear cohesion for stray node handling)
        if hasProjects {
            let groupN = (projectGroup.max() ?? -1) + 1
            if groupN > 0 {
                var gSumX = [Float](repeating: 0, count: groupN)
                var gSumY = [Float](repeating: 0, count: groupN)
                var gSumZ = [Float](repeating: 0, count: groupN)
                var gCount = [Int](repeating: 0, count: groupN)
                // Build per-group index lists
                var projMembers = [[Int]](repeating: [], count: groupN)
                for i in 0..<n {
                    let g = projectGroup[i]
                    gSumX[g] += x[i]; gSumY[g] += y[i]; gSumZ[g] += z[i]; gCount[g] += 1
                    projMembers[g].append(i)
                }

                // Compute per-group 75th percentile radius for non-linear cohesion
                var gRadii = [[Float]](repeating: [], count: groupN)
                for i in 0..<n {
                    let g = projectGroup[i]; let cnt = Float(gCount[g])
                    if cnt < 2 { continue }
                    let cx = gSumX[g] / cnt, cy = gSumY[g] / cnt, cz = gSumZ[g] / cnt
                    let dx = x[i] - cx, dy = y[i] - cy, dz = z[i] - cz
                    gRadii[g].append(sqrt(dx * dx + dy * dy + dz * dz))
                }
                var gRefR = [Float](repeating: 30.0, count: groupN)
                for g in 0..<groupN {
                    guard !gRadii[g].isEmpty else { continue }
                    gRadii[g].sort()
                    gRefR[g] = max(30.0, gRadii[g][gRadii[g].count * 3 / 4])
                }

                // Non-linear cohesion: quadratic ramp beyond the cluster's core radius
                let cohStr = s.cohesionStrength
                for i in 0..<n {
                    let g = projectGroup[i]; let cnt = Float(gCount[g])
                    if cnt < 2 { continue }
                    let cx = gSumX[g] / cnt, cy = gSumY[g] / cnt, cz = gSumZ[g] / cnt
                    let dx = cx - x[i], dy = cy - y[i], dz = cz - z[i]
                    let dist = sqrt(dx * dx + dy * dy + dz * dz)
                    let ratio = max(1.0, dist / gRefR[g])
                    let scale = ratio * ratio  // quadratic: 2x beyond edge → 4x force
                    fx[i] += dx * cohStr * scale
                    fy[i] += dy * cohStr * scale
                    fz[i] += dz * cohStr * scale
                }

                // Accumulate per-group centroid repulsion, then distribute
                var projForceX = [Float](repeating: 0, count: groupN)
                var projForceY = [Float](repeating: 0, count: groupN)
                var projForceZ = [Float](repeating: 0, count: groupN)
                for g1 in 0..<groupN {
                    guard gCount[g1] > 0 else { continue }
                    let c1 = SIMD3<Float>(gSumX[g1], gSumY[g1], gSumZ[g1]) / Float(gCount[g1])
                    for g2 in (g1 + 1)..<groupN {
                        guard gCount[g2] > 0 else { continue }
                        let c2 = SIMD3<Float>(gSumX[g2], gSumY[g2], gSumZ[g2]) / Float(gCount[g2])
                        var delta = c1 - c2
                        var pdist = simd_length(delta)
                        if pdist < 1 { pdist = 1; delta = SIMD3(.random(in: -1...1), .random(in: -1...1), .random(in: -1...1)) }
                        let force = s.centroidRepulsion / (pdist * pdist)
                        let fVec = (delta / pdist) * force
                        let f1 = 1.0 / Float(gCount[g1]), f2 = 1.0 / Float(gCount[g2])
                        projForceX[g1] += fVec.x * f1; projForceY[g1] += fVec.y * f1; projForceZ[g1] += fVec.z * f1
                        projForceX[g2] -= fVec.x * f2; projForceY[g2] -= fVec.y * f2; projForceZ[g2] -= fVec.z * f2
                    }
                }
                // Distribute accumulated per-group forces to members (O(n) total)
                for g in 0..<groupN where gCount[g] > 0 {
                    let pfx = projForceX[g], pfy = projForceY[g], pfz = projForceZ[g]
                    if pfx == 0 && pfy == 0 && pfz == 0 { continue }
                    for i in projMembers[g] {
                        fx[i] += pfx; fy[i] += pfy; fz[i] += pfz
                    }
                }
            }
        }

        // Center gravity
        let c = s.center
        for i in 0..<n {
            fx[i] += (c.x - x[i]) * s.alpha * s.centerStrength
            fy[i] += (c.y - y[i]) * s.alpha * s.centerStrength
            fz[i] += (c.z - z[i]) * s.alpha * s.centerStrength
        }

        return ForceResult(fx: fx, fy: fy, fz: fz)
    }

    /// Pure force computation — runs on background thread via Task.detached.
    /// nonisolated: takes SimState by value, no shared mutable state, safe to run in parallel.
    private nonisolated static func computeForces(_ s: SimState) -> ForceResult {
        let n = s.n
        let x = s.x, y = s.y, z = s.z
        let projectGroup = s.projectGroup, topicGroup = s.topicGroup
        let alpha = s.alpha

        let hasProjects = !projectGroup.isEmpty
        let hasTopics = !topicGroup.isEmpty

        var fx = [Float](repeating: 0, count: n)
        var fy = [Float](repeating: 0, count: n)
        var fz = [Float](repeating: 0, count: n)

        // --- Charge repulsion ---
        let chargeBase = s.chargeStrength
        let chargeCross = chargeBase * s.crossChargeMultiplier
        let chargeSameProject = chargeBase * s.sameProjectChargeScale
        let chargeSameTopic = chargeBase * s.sameTopicChargeScale

        // Check spatial extent to decide exact vs Barnes-Hut.
        // BH only helps when nodes are spread beyond the charge cutoff (500 units),
        // so distant subtrees can be pruned. For dense initial placement (all nodes
        // within cutoff), BH degenerates to O(n²) without Newton's third — slower
        // than exact. Use exact O(n²) with Newton's third for small or dense graphs.
        let cutoffSq: Float = 500 * 500
        var useBH = n > 256
        if useBH {
            var minX = x[0], maxX = x[0], minY = y[0], maxY = y[0], minZ = z[0], maxZ = z[0]
            for i in 1..<n {
                if x[i] < minX { minX = x[i] } else if x[i] > maxX { maxX = x[i] }
                if y[i] < minY { minY = y[i] } else if y[i] > maxY { maxY = y[i] }
                if z[i] < minZ { minZ = z[i] } else if z[i] > maxZ { maxZ = z[i] }
            }
            let extent = max(maxX - minX, max(maxY - minY, maxZ - minZ))
            // If all nodes fit within 2× cutoff diameter, most pairs are within cutoff
            // and BH can't prune anything — exact with Newton's third is faster
            if extent < 1000 { useBH = false }
        }

        if !useBH {
            // Exact O(n²) with Newton's third — optimal for small/dense graphs
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
                    } else if hasProjects {
                        charge = chargeSameProject
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
        } else {
            // Spread-out graph: Barnes-Hut O(n log n)
            computeChargeForcesBH(
                x: x, y: y, z: z, n: n,
                projectGroup: projectGroup, topicGroup: topicGroup,
                hasProjects: hasProjects, hasTopics: hasTopics,
                chargeBase: chargeBase, chargeCross: chargeCross,
                chargeSameProject: chargeSameProject, chargeSameTopic: chargeSameTopic,
                fx: &fx, fy: &fy, fz: &fz
            )
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
            var topicMembers = [[Int]](repeating: [], count: topicN)
            for i in 0..<n {
                let g = topicGroup[i]
                tSumX[g] += x[i]; tSumY[g] += y[i]; tSumZ[g] += z[i]; tCount[g] += 1
                topicMembers[g].append(i)
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
            var topicForceX = [Float](repeating: 0, count: topicN)
            var topicForceY = [Float](repeating: 0, count: topicN)
            var topicForceZ = [Float](repeating: 0, count: topicN)
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
                    topicForceX[g1] += tfx * f1; topicForceY[g1] += tfy * f1; topicForceZ[g1] += tfz * f1
                    topicForceX[g2] -= tfx * f2; topicForceY[g2] -= tfy * f2; topicForceZ[g2] -= tfz * f2
                }
            }
            for g in 0..<topicN where tCount[g] > 0 {
                let tfx = topicForceX[g], tfy = topicForceY[g], tfz = topicForceZ[g]
                if tfx == 0 && tfy == 0 && tfz == 0 { continue }
                for i in topicMembers[g] {
                    fx[i] += tfx; fy[i] += tfy; fz[i] += tfz
                }
            }
        }

        // --- Project centroid forces (with non-linear cohesion) ---
        if hasProjects {
            let groupN = (projectGroup.max() ?? -1) + 1
            if groupN > 0 {
                var gSumX = [Float](repeating: 0, count: groupN)
                var gSumY = [Float](repeating: 0, count: groupN)
                var gSumZ = [Float](repeating: 0, count: groupN)
                var gCount = [Int](repeating: 0, count: groupN)
                var projMembers = [[Int]](repeating: [], count: groupN)
                for i in 0..<n {
                    let g = projectGroup[i]
                    gSumX[g] += x[i]; gSumY[g] += y[i]; gSumZ[g] += z[i]; gCount[g] += 1
                    projMembers[g].append(i)
                }

                // Compute per-group 75th percentile radius for non-linear cohesion
                var gRadii = [[Float]](repeating: [], count: groupN)
                for i in 0..<n {
                    let g = projectGroup[i]; let cnt = Float(gCount[g])
                    if cnt < 2 { continue }
                    let cx = gSumX[g] / cnt, cy = gSumY[g] / cnt, cz = gSumZ[g] / cnt
                    let dx = x[i] - cx, dy = y[i] - cy, dz = z[i] - cz
                    gRadii[g].append(sqrt(dx * dx + dy * dy + dz * dz))
                }
                var gRefR = [Float](repeating: 30.0, count: groupN)
                for g in 0..<groupN {
                    guard !gRadii[g].isEmpty else { continue }
                    gRadii[g].sort()
                    gRefR[g] = max(30.0, gRadii[g][gRadii[g].count * 3 / 4])
                }

                // Non-linear cohesion: quadratic ramp beyond the cluster's core radius
                let cohStr = s.cohesionStrength
                for i in 0..<n {
                    let g = projectGroup[i]; let cnt = Float(gCount[g])
                    if cnt < 2 { continue }
                    let cx = gSumX[g] / cnt, cy = gSumY[g] / cnt, cz = gSumZ[g] / cnt
                    let dx = cx - x[i], dy = cy - y[i], dz = cz - z[i]
                    let dist = sqrt(dx * dx + dy * dy + dz * dz)
                    let ratio = max(1.0, dist / gRefR[g])
                    let scale = ratio * ratio  // quadratic: 2x beyond edge → 4x force
                    fx[i] += dx * cohStr * scale
                    fy[i] += dy * cohStr * scale
                    fz[i] += dz * cohStr * scale
                }

                let centRep = s.centroidRepulsion
                var projForceX = [Float](repeating: 0, count: groupN)
                var projForceY = [Float](repeating: 0, count: groupN)
                var projForceZ = [Float](repeating: 0, count: groupN)
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
                        projForceX[g1] += fVec.x * f1; projForceY[g1] += fVec.y * f1; projForceZ[g1] += fVec.z * f1
                        projForceX[g2] -= fVec.x * f2; projForceY[g2] -= fVec.y * f2; projForceZ[g2] -= fVec.z * f2
                    }
                }
                for g in 0..<groupN where gCount[g] > 0 {
                    let pfx = projForceX[g], pfy = projForceY[g], pfz = projForceZ[g]
                    if pfx == 0 && pfy == 0 && pfz == 0 { continue }
                    for i in projMembers[g] {
                        fx[i] += pfx; fy[i] += pfy; fz[i] += pfz
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

    // MARK: - Local wake (targeted force computation for single-node inserts)

    /// Compute project cluster centroid from existing node positions.
    private func projectCentroid(group: Int, jitter: Float) -> SIMD3<Float> {
        var sumX: Float = 0, sumY: Float = 0, sumZ: Float = 0, count: Float = 0
        for i in 0..<ids.count {
            if projectGroup[i] == group {
                sumX += x[i]; sumY += y[i]; sumZ += z[i]; count += 1
            }
        }
        if count > 0 {
            return SIMD3<Float>(
                sumX / count + .random(in: -jitter...jitter),
                sumY / count + .random(in: -jitter...jitter),
                sumZ / count + .random(in: -jitter...jitter)
            )
        }
        return randomSpherePosition()
    }

    /// Random position on a spherical shell around center.
    private func randomSpherePosition() -> SIMD3<Float> {
        let angle = Float.random(in: 0...(2 * .pi))
        let phi = Float.random(in: -.pi/2...(.pi/2))
        let r = Float.random(in: 50...250)
        return SIMD3<Float>(
            center.x + cos(angle) * cos(phi) * r,
            center.y + sin(angle) * cos(phi) * r,
            center.z + sin(phi) * r
        )
    }

    /// Expand local wake neighborhood to include same-project-group nodes, capped.
    private func expandLocalWakeNeighborhood() {
        let maxSize = 100
        guard let newIdx = localWakeNewNodeIndex else { return }
        let newProjGroup = projectGroup[newIdx]
        for i in 0..<ids.count {
            if projectGroup[i] == newProjGroup {
                localWakeIndices.insert(i)
            }
            if localWakeIndices.count >= maxSize { break }
        }
    }

    /// Lightweight per-frame tick for local wake: CPU-only forces on neighborhood.
    private func tickLocalWake() {
        // First tick: expand neighborhood to include project peers
        if !localWakeExpanded {
            expandLocalWakeNeighborhood()
            localWakeExpanded = true
        }

        let localIndices = Array(localWakeIndices)
        let k = localIndices.count
        guard k > 0 else { finishLocalWake(); return }

        // --- O(K²) charge repulsion among local nodes ---
        for ii in 0..<k {
            let i = localIndices[ii]
            guard !pinned[i] else { continue }
            for jj in (ii + 1)..<k {
                let j = localIndices[jj]
                let dx = x[i] - x[j], dy = y[i] - y[j], dz = z[i] - z[j]
                var distSq = dx * dx + dy * dy + dz * dz
                if distSq > 250_000 { continue }  // 500² cutoff
                if distSq < 1 { distSq = 1 }

                let charge: Float
                if projectGroup[i] != projectGroup[j] {
                    charge = chargeStrength * crossChargeMultiplier
                } else if topicGroup[i] == topicGroup[j] {
                    charge = chargeStrength * sameTopicChargeScale
                } else {
                    charge = chargeStrength * sameProjectChargeScale
                }

                let dist = sqrt(distSq)
                let forceMag = charge / distSq
                let ux = dx / dist, uy = dy / dist, uz = dz / dist
                let ffx = ux * forceMag, ffy = uy * forceMag, ffz = uz * forceMag
                vx[i] += ffx; vy[i] += ffy; vz[i] += ffz
                vx[j] -= ffx; vy[j] -= ffy; vz[j] -= ffz
            }
        }

        // --- Spring forces for edges touching the local set ---
        let localSet = localWakeIndices
        for (si, ti) in edgeIndices {
            guard localSet.contains(si) || localSet.contains(ti) else { continue }
            let dx = x[ti] - x[si], dy = y[ti] - y[si], dz = z[ti] - z[si]
            var d = sqrt(dx * dx + dy * dy + dz * dz)
            if d < 1 { d = 1 }
            let cross = projectGroup[si] != projectGroup[ti]
            let rest = cross ? crossProjectSpringLength : springLength
            let force = springStrength * (d - rest)
            let efx = (dx / d) * force, efy = (dy / d) * force, efz = (dz / d) * force
            if localSet.contains(si) { vx[si] += efx; vy[si] += efy; vz[si] += efz }
            if localSet.contains(ti) { vx[ti] -= efx; vy[ti] -= efy; vz[ti] -= efz }
        }

        // --- Cohesion: pull new node(s) toward project centroid ---
        if let newIdx = localWakeNewNodeIndex {
            let pg = projectGroup[newIdx]
            var sumX: Float = 0, sumY: Float = 0, sumZ: Float = 0, count: Float = 0
            for i in 0..<ids.count where projectGroup[i] == pg {
                sumX += x[i]; sumY += y[i]; sumZ += z[i]; count += 1
            }
            if count > 1 {
                let cx = sumX / count, cy = sumY / count, cz = sumZ / count
                // Stronger cohesion for fast convergence (3x normal)
                let str = cohesionStrength * 3.0
                let dx = cx - x[newIdx], dy = cy - y[newIdx], dz = cz - z[newIdx]
                vx[newIdx] += dx * str
                vy[newIdx] += dy * str
                vz[newIdx] += dz * str
            }
        }

        // --- Center gravity for local nodes ---
        for i in localIndices where !pinned[i] {
            vx[i] += (center.x - x[i]) * alphaFloor * centerStrength
            vy[i] += (center.y - y[i]) * alphaFloor * centerStrength
            vz[i] += (center.z - z[i]) * alphaFloor * centerStrength
        }

        // --- Integration: only local nodes ---
        var maxSpeedSq: Float = 0
        for i in localIndices where !pinned[i] {
            vx[i] *= damping; vy[i] *= damping; vz[i] *= damping
            let speedSq = vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]
            maxSpeedSq = max(maxSpeedSq, speedSq)
            if speedSq > maxSpeed * maxSpeed {
                let scale = maxSpeed / sqrt(speedSq)
                vx[i] *= scale; vy[i] *= scale; vz[i] *= scale
            }
            x[i] += vx[i]; y[i] += vy[i]; z[i] += vz[i]
        }

        lastMaxSpeedSq = maxSpeedSq

        // Sync only changed positions (O(K) instead of O(n))
        for i in localIndices {
            positions[ids[i]] = SIMD3(x[i], y[i], z[i])
        }

        // Settle: velocities converged or max frames reached
        localWakeFramesRemaining -= 1
        if maxSpeedSq < 0.01 || localWakeFramesRemaining <= 0 {
            finishLocalWake()
        }
    }

    /// Finish local wake — transition back to settled state.
    private func finishLocalWake() {
        clearLocalWake()
        isSettled = true
        lastMaxSpeedSq = 0
        // Final full sync for consistency
        syncPositions()
    }

    /// Reset all local wake state without changing isSettled.
    private func clearLocalWake() {
        isLocalWake = false
        localWakeIndices.removeAll()
        localWakeNewNodeIndex = nil
        localWakeExpanded = false
        localWakeFramesRemaining = 0
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
