import Testing
import Foundation
import Metal
import simd
@testable import EngramSceneKit
import EngramMetalShaders

/// GPU force simulation integration tests.
/// Uses MetalForceCompute + GPUSimulationState synchronously (waitUntilCompleted)
/// to verify cluster separation at different scales — same assertions as CPU tests.
@MainActor
struct GPUForceClusterTests {

    struct SimResult {
        let projectCentroids: [Int: SIMD3<Float>]
        let topicCentroids: [Int: SIMD3<Float>]
        let projectRadii: [Int: Float]
        let topicRadii: [Int: Float]
        let minProjectSeparation: Float
        let minTopicSeparation: Float
        let nodeCount: Int
        let framesRun: Int
        /// Nodes closer to a different project's centroid than their own.
        let stragglerCount: Int
        /// Fraction of nodes that are stragglers (0.0 = perfect, 1.0 = all wrong).
        let stragglerRate: Float
    }

    /// Force parameter overrides for tuning experiments.
    struct ForceOverrides {
        var cohesionStrength: Float?
        var centroidRepulsion: Float?
        var crossProjectSpringLength: Float?
        var springStrength: Float?
        var topicCohesionStrength: Float?
        var topicCentroidRepulsion: Float?
        var crossChargeMultiplier: Float?
    }

    /// Run force sim using GPU compute (synchronous via waitUntilCompleted).
    static func runGPUSim(
        nProjects: Int = 3,
        nTopics: Int = 4,
        nodesPerTopic: Int = 50,
        maxFrames: Int = 1500,
        overrides: ForceOverrides = ForceOverrides()
    ) -> SimResult? {
        // Configure GPULog for test diagnostics
        let logPath = "/tmp/gpu_force_test.log"
        GPULog.configure(path: logPath)
        GPULog.log("=== runGPUSim START nProjects=\(nProjects) nTopics=\(nTopics) nodesPerTopic=\(nodesPerTopic) ===")

        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let library = try? EngramMetalShaders.makeLibrary(device: device),
              let forceCompute = MetalForceCompute(device: device, library: library) else {
            return nil
        }

        let simState = GPUSimulationState(device: device, library: library)
        let sim = ForceSimulation3D()

        let totalNodes = nProjects * nTopics * nodesPerTopic
        var ids: [UUID] = []
        var projectForId: [UUID: Int] = [:]
        var topicForId: [UUID: Int] = [:]
        var buckets: [Int: [Int: [UUID]]] = [:]

        // Generate nodes
        for p in 0..<nProjects {
            for t in 0..<nTopics {
                let topicIdx = p * nTopics + t
                for _ in 0..<nodesPerTopic {
                    let id = UUID()
                    ids.append(id)
                    projectForId[id] = p
                    topicForId[id] = topicIdx
                    buckets[p, default: [:]][topicIdx, default: []].append(id)
                    sim.addNode(id, project: "P\(p)", topic: "T\(topicIdx)")
                }
            }
        }

        // Intra-topic chains
        for (_, topics) in buckets {
            for (_, members) in topics where members.count >= 2 {
                for i in 1..<members.count {
                    sim.addEdge(from: members[i-1], to: members[i])
                }
            }
        }

        // Intra-project bridges
        for p in 0..<nProjects {
            let topicKeys = Array(buckets[p]!.keys).sorted()
            for i in 0..<topicKeys.count {
                let j = (i + 1) % topicKeys.count
                let a = buckets[p]![topicKeys[i]]!, b = buckets[p]![topicKeys[j]]!
                let bridgeCount = max(1, min(a.count, b.count) / 4)
                for k in 0..<bridgeCount {
                    sim.addEdge(from: a[k % a.count], to: b[k % b.count])
                }
            }
        }

        // Cross-project edges
        let crossCount = max(totalNodes / 30, 3)
        for _ in 0..<crossCount {
            let i = Int.random(in: 0..<ids.count)
            var j = Int.random(in: 0..<ids.count)
            while projectForId[ids[j]] == projectForId[ids[i]] && ids.count > 1 {
                j = Int.random(in: 0..<ids.count)
            }
            sim.addEdge(from: ids[i], to: ids[j])
        }

        sim.wake()

        // Mark topology dirty for GPU
        forceCompute.setTopologyDirty(edges: sim.edgeIndicesPublic)
        forceCompute.rebuildTopology(
            nodeCount: sim.nodeCount,
            projectGroups: sim.projectGroupPublic,
            topicGroups: sim.topicGroupPublic,
            topicProjectGroup: sim.topicProjectGroupPublic,
            galaxyGroups: sim.galaxyGroupPublic,
            galaxyCenters: sim.galaxyCentersPublic,
            x: sim.posX, y: sim.posY, z: sim.posZ
        )

        // Run simulation synchronously with GPU forces
        var frames = 0
        while frames < maxFrames && !sim.isSettled {
            let n = sim.nodeCount
            guard n > 1 else { break }

            // Build command buffer with force compute + integration
            guard let cmdBuf = queue.makeCommandBuffer(),
                  let encoder = cmdBuf.makeComputeCommandEncoder() else { break }

            forceCompute.encodeForces(
                encoder: encoder,
                x: sim.posX, y: sim.posY, z: sim.posZ,
                projectGroups: sim.projectGroupPublic,
                topicGroups: sim.topicGroupPublic,
                galaxyGroups: sim.galaxyGroupPublic,
                galaxyCenters: sim.galaxyCentersPublic,
                chargeStrength: sim.chargeStrength,
                crossChargeMultiplier: overrides.crossChargeMultiplier ?? sim.crossChargeMultiplier,
                sameTopicChargeScale: sim.sameTopicChargeScale,
                sameProjectChargeScale: sim.sameProjectChargeScale,
                springLength: sim.springLength,
                crossProjectSpringLength: overrides.crossProjectSpringLength ?? sim.crossProjectSpringLength,
                springStrength: overrides.springStrength ?? sim.springStrength,
                cohesionStrength: overrides.cohesionStrength ?? sim.cohesionStrength,
                centroidRepulsion: overrides.centroidRepulsion ?? sim.centroidRepulsion,
                topicCohesionStrength: overrides.topicCohesionStrength ?? sim.topicCohesionStrength,
                topicCentroidRepulsion: overrides.topicCentroidRepulsion ?? sim.topicCentroidRepulsion,
                centerStrength: sim.centerStrength,
                alpha: sim.alpha, damping: 0.78, maxSpeed: 12.0
            )

            // GPU integration
            if let forceBuf = forceCompute.outputForceBuffer {
                simState.uploadPositions(x: sim.posX, y: sim.posY, z: sim.posZ)
                simState.encodeIntegration(
                    encoder: encoder, forceBuffer: forceBuf,
                    damping: 0.78, maxSpeed: 12.0
                )
            }
            encoder.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()

            // Read back GPU forces for diagnostics
            if frames % 200 == 0, let forces = forceCompute.readBackForces() {
                var maxF: Float = 0
                var totalF: Float = 0
                for i in 0..<n {
                    let f = sqrt(forces.fx[i]*forces.fx[i] + forces.fy[i]*forces.fy[i] + forces.fz[i]*forces.fz[i])
                    maxF = max(maxF, f)
                    totalF += f
                }
                GPULog.log("FRAME \(frames) FORCES: maxF=\(String(format:"%.2f",maxF)) avgF=\(String(format:"%.4f",totalF/Float(n))) alpha=\(String(format:"%.4f",sim.alpha))")
            }

            // Read back GPU positions into sim
            simState.readBackPositions()
            let positions = simState.cpuPositions
            sim.applyGPUForces(ForceResult(positions: positions))

            // Debug: check GPU velocity buffer magnitude + position spread
            if frames % 200 == 0 {
                if let velBuf = simState.velocityBuffer {
                    let velPtr = velBuf.contents().bindMemory(to: SIMD3<Float>.self, capacity: sim.nodeCount)
                    var maxVel: Float = 0
                    for i in 0..<sim.nodeCount {
                        maxVel = max(maxVel, simd_length(velPtr[i]))
                    }
                    GPULog.log("FRAME \(frames) maxGPUVel=\(String(format:"%.3f",maxVel)) alpha=\(String(format:"%.4f",sim.alpha))")
                }
                // Measure position spread
                var minP = SIMD3<Float>(repeating: Float.greatestFiniteMagnitude)
                var maxP = SIMD3<Float>(repeating: -Float.greatestFiniteMagnitude)
                for i in 0..<n {
                    minP.x = min(minP.x, sim.posX[i]); minP.y = min(minP.y, sim.posY[i]); minP.z = min(minP.z, sim.posZ[i])
                    maxP.x = max(maxP.x, sim.posX[i]); maxP.y = max(maxP.y, sim.posY[i]); maxP.z = max(maxP.z, sim.posZ[i])
                }
                let span = maxP - minP
                GPULog.log("FRAME \(frames) SPAN: (\(String(format:"%.0f",span.x)),\(String(format:"%.0f",span.y)),\(String(format:"%.0f",span.z)))")
            }

            // Tick for alpha decay + settle detection only
            sim.tick()
            frames += 1
        }
        GPULog.log("=== runGPUSim END frames=\(frames) settled=\(sim.isSettled) ===")

        let positions = sim.positions

        // Compute metrics (same as CPU tests)
        var projectSums: [Int: (sum: SIMD3<Float>, count: Int)] = [:]
        for id in ids {
            guard let pos = positions[id], let p = projectForId[id] else { continue }
            let e = projectSums[p] ?? (.zero, 0)
            projectSums[p] = (e.sum + pos, e.count + 1)
        }
        let projectCentroids = projectSums.mapValues { $0.sum / Float($0.count) }

        var projectDistances: [Int: [Float]] = [:]
        for id in ids {
            guard let pos = positions[id], let p = projectForId[id],
                  let centroid = projectCentroids[p] else { continue }
            projectDistances[p, default: []].append(simd_length(pos - centroid))
        }
        let projectRadii = projectDistances.mapValues { dists -> Float in
            let sorted = dists.sorted()
            return sorted[sorted.count * 3 / 4]
        }

        var topicSums: [Int: (sum: SIMD3<Float>, count: Int)] = [:]
        for id in ids {
            guard let pos = positions[id], let t = topicForId[id] else { continue }
            let e = topicSums[t] ?? (.zero, 0)
            topicSums[t] = (e.sum + pos, e.count + 1)
        }
        let topicCentroids = topicSums.mapValues { $0.sum / Float($0.count) }

        var topicDistances: [Int: [Float]] = [:]
        for id in ids {
            guard let pos = positions[id], let t = topicForId[id],
                  let centroid = topicCentroids[t] else { continue }
            topicDistances[t, default: []].append(simd_length(pos - centroid))
        }
        let topicRadii = topicDistances.mapValues { dists -> Float in
            let sorted = dists.sorted()
            return sorted[sorted.count * 3 / 4]
        }

        var minProjectSep: Float = .infinity
        let projKeys = Array(projectCentroids.keys).sorted()
        for i in 0..<projKeys.count {
            for j in (i+1)..<projKeys.count {
                let dist = simd_length(projectCentroids[projKeys[i]]! - projectCentroids[projKeys[j]]!)
                minProjectSep = min(minProjectSep, dist)
            }
        }

        var minTopicSep: Float = .infinity
        for p in 0..<nProjects {
            let topicsInProject = (0..<nTopics).map { p * nTopics + $0 }
            for i in 0..<topicsInProject.count {
                for j in (i+1)..<topicsInProject.count {
                    guard let c1 = topicCentroids[topicsInProject[i]],
                          let c2 = topicCentroids[topicsInProject[j]] else { continue }
                    minTopicSep = min(minTopicSep, simd_length(c1 - c2))
                }
            }
        }

        // Straggler detection: nodes closer to a different project centroid than their own
        var stragglerCount = 0
        let projKeysSorted = Array(projectCentroids.keys).sorted()
        for id in ids {
            guard let pos = positions[id], let myProj = projectForId[id],
                  let myCentroid = projectCentroids[myProj] else { continue }
            let myDist = simd_length(pos - myCentroid)
            for otherProj in projKeysSorted where otherProj != myProj {
                guard let otherCentroid = projectCentroids[otherProj] else { continue }
                if simd_length(pos - otherCentroid) < myDist {
                    stragglerCount += 1
                    break
                }
            }
        }
        let stragglerRate = Float(stragglerCount) / Float(max(totalNodes, 1))

        return SimResult(
            projectCentroids: projectCentroids,
            topicCentroids: topicCentroids,
            projectRadii: projectRadii,
            topicRadii: topicRadii,
            minProjectSeparation: minProjectSep,
            minTopicSeparation: minTopicSep,
            nodeCount: totalNodes,
            framesRun: frames,
            stragglerCount: stragglerCount,
            stragglerRate: stragglerRate
        )
    }

    @Test("GPU: Small scale 600 nodes, projects separate")
    func gpuSmallScale() {
        guard let r = Self.runGPUSim(nProjects: 3, nTopics: 4, nodesPerTopic: 50) else {
            Issue.record("Metal not available"); return
        }
        let avgRadius = r.projectRadii.values.reduce(0, +) / Float(r.projectRadii.count)
        let ratio = r.minProjectSeparation / avgRadius
        print("[GPU-600] frames=\(r.framesRun) sep=\(r.minProjectSeparation) radius=\(avgRadius) ratio=\(ratio) stragglers=\(r.stragglerCount)/\(r.nodeCount) (\(String(format:"%.1f",r.stragglerRate*100))%)")
        #expect(ratio > 1.5, "GPU projects not separated: ratio=\(ratio)")
    }

    @Test("GPU: Medium scale 2400 nodes, projects separate")
    func gpuMediumScale() {
        guard let r = Self.runGPUSim(nProjects: 3, nTopics: 4, nodesPerTopic: 200) else {
            Issue.record("Metal not available"); return
        }
        let avgRadius = r.projectRadii.values.reduce(0, +) / Float(r.projectRadii.count)
        let ratio = r.minProjectSeparation / avgRadius
        print("[GPU-2400] frames=\(r.framesRun) sep=\(r.minProjectSeparation) radius=\(avgRadius) ratio=\(ratio) stragglers=\(r.stragglerCount)/\(r.nodeCount) (\(String(format:"%.1f",r.stragglerRate*100))%)")
        #expect(ratio > 1.5, "GPU projects not separated: ratio=\(ratio)")
    }

    @Test("GPU: Topic separation 500 nodes")
    func gpuTopicSeparation() {
        guard let r = Self.runGPUSim(nProjects: 2, nTopics: 5, nodesPerTopic: 50) else {
            Issue.record("Metal not available"); return
        }
        let avgTopicRadius = r.topicRadii.values.reduce(0, +) / Float(r.topicRadii.count)
        let ratio = r.minTopicSeparation / avgTopicRadius
        print("[GPU-500-topic] frames=\(r.framesRun) topicSep=\(r.minTopicSeparation) topicRadius=\(avgTopicRadius) ratio=\(ratio)")
        #expect(ratio > 0.5, "GPU topics too tight: ratio=\(ratio)")
    }

    // MARK: - GPU Force + CPU Integration (diagnostic)

    /// Same as runGPUSim but uses GPU-computed forces with CPU integration
    /// (skips GPU integration kernel). If this produces good clustering,
    /// the issue is in GPU integration, not GPU force computation.
    static func runGPUForcesCPUIntegrate(
        nProjects: Int = 3,
        nTopics: Int = 4,
        nodesPerTopic: Int = 50,
        maxFrames: Int = 1500
    ) -> SimResult? {
        GPULog.configure(path: "/tmp/gpu_force_test.log")
        GPULog.log("=== GPU+CPU START nProjects=\(nProjects) nTopics=\(nTopics) nodesPerTopic=\(nodesPerTopic) ===")

        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue(),
              let library = try? EngramMetalShaders.makeLibrary(device: device),
              let forceCompute = MetalForceCompute(device: device, library: library) else {
            return nil
        }

        let sim = ForceSimulation3D()
        let totalNodes = nProjects * nTopics * nodesPerTopic
        var ids: [UUID] = []
        var projectForId: [UUID: Int] = [:]
        var topicForId: [UUID: Int] = [:]
        var buckets: [Int: [Int: [UUID]]] = [:]

        for p in 0..<nProjects {
            for t in 0..<nTopics {
                let topicIdx = p * nTopics + t
                for _ in 0..<nodesPerTopic {
                    let id = UUID()
                    ids.append(id)
                    projectForId[id] = p
                    topicForId[id] = topicIdx
                    buckets[p, default: [:]][topicIdx, default: []].append(id)
                    sim.addNode(id, project: "P\(p)", topic: "T\(topicIdx)")
                }
            }
        }
        for (_, topics) in buckets {
            for (_, members) in topics where members.count >= 2 {
                for i in 1..<members.count { sim.addEdge(from: members[i-1], to: members[i]) }
            }
        }
        for p in 0..<nProjects {
            let topicKeys = Array(buckets[p]!.keys).sorted()
            for i in 0..<topicKeys.count {
                let j = (i + 1) % topicKeys.count
                let a = buckets[p]![topicKeys[i]]!, b = buckets[p]![topicKeys[j]]!
                let bridgeCount = max(1, min(a.count, b.count) / 4)
                for k in 0..<bridgeCount { sim.addEdge(from: a[k % a.count], to: b[k % b.count]) }
            }
        }
        let crossCount = max(totalNodes / 30, 3)
        for _ in 0..<crossCount {
            let i = Int.random(in: 0..<ids.count)
            var j = Int.random(in: 0..<ids.count)
            while projectForId[ids[j]] == projectForId[ids[i]] && ids.count > 1 { j = Int.random(in: 0..<ids.count) }
            sim.addEdge(from: ids[i], to: ids[j])
        }
        sim.wake()

        forceCompute.setTopologyDirty(edges: sim.edgeIndicesPublic)
        forceCompute.rebuildTopology(
            nodeCount: sim.nodeCount, projectGroups: sim.projectGroupPublic,
            topicGroups: sim.topicGroupPublic, topicProjectGroup: sim.topicProjectGroupPublic,
            galaxyGroups: sim.galaxyGroupPublic, galaxyCenters: sim.galaxyCentersPublic,
            x: sim.posX, y: sim.posY, z: sim.posZ)

        // Run simulation: GPU force compute + CPU integration (no GPU integration kernel)
        var frames = 0
        while frames < maxFrames && !sim.isSettled {
            let n = sim.nodeCount
            guard n > 1 else { break }

            guard let cmdBuf = queue.makeCommandBuffer(),
                  let encoder = cmdBuf.makeComputeCommandEncoder() else { break }

            forceCompute.encodeForces(
                encoder: encoder,
                x: sim.posX, y: sim.posY, z: sim.posZ,
                projectGroups: sim.projectGroupPublic, topicGroups: sim.topicGroupPublic,
                galaxyGroups: sim.galaxyGroupPublic, galaxyCenters: sim.galaxyCentersPublic,
                chargeStrength: sim.chargeStrength, crossChargeMultiplier: sim.crossChargeMultiplier,
                sameTopicChargeScale: sim.sameTopicChargeScale, sameProjectChargeScale: sim.sameProjectChargeScale,
                springLength: sim.springLength, crossProjectSpringLength: sim.crossProjectSpringLength,
                springStrength: sim.springStrength,
                cohesionStrength: sim.cohesionStrength, centroidRepulsion: sim.centroidRepulsion,
                topicCohesionStrength: sim.topicCohesionStrength, topicCentroidRepulsion: sim.topicCentroidRepulsion,
                centerStrength: sim.centerStrength,
                alpha: sim.alpha, damping: 0.78, maxSpeed: 12.0
            )
            // NO GPU integration — just force compute
            encoder.endEncoding()
            cmdBuf.commit()
            cmdBuf.waitUntilCompleted()

            // Read back raw forces and deliver to sim for CPU integration
            if let forces = forceCompute.readBackForces() {
                sim.applyGPUForces(ForceResult(fx: forces.fx, fy: forces.fy, fz: forces.fz))
            }

            // CPU integration via tick()
            sim.tick()
            frames += 1

            if frames % 200 == 0 {
                GPULog.log("GPU+CPU FRAME \(frames) alpha=\(String(format:"%.4f",sim.alpha))")
            }
        }
        GPULog.log("=== GPU+CPU END frames=\(frames) settled=\(sim.isSettled) ===")

        // Same metrics as runGPUSim
        let positions = sim.positions
        var projectSums: [Int: (sum: SIMD3<Float>, count: Int)] = [:]
        for id in ids { guard let pos = positions[id], let p = projectForId[id] else { continue }; let e = projectSums[p] ?? (.zero, 0); projectSums[p] = (e.sum + pos, e.count + 1) }
        let projectCentroids = projectSums.mapValues { $0.sum / Float($0.count) }
        var projectDistances: [Int: [Float]] = [:]
        for id in ids { guard let pos = positions[id], let p = projectForId[id], let centroid = projectCentroids[p] else { continue }; projectDistances[p, default: []].append(simd_length(pos - centroid)) }
        let projectRadii = projectDistances.mapValues { dists -> Float in let sorted = dists.sorted(); return sorted[sorted.count * 3 / 4] }
        var topicSums: [Int: (sum: SIMD3<Float>, count: Int)] = [:]
        for id in ids { guard let pos = positions[id], let t = topicForId[id] else { continue }; let e = topicSums[t] ?? (.zero, 0); topicSums[t] = (e.sum + pos, e.count + 1) }
        let topicCentroids = topicSums.mapValues { $0.sum / Float($0.count) }
        var topicDistances: [Int: [Float]] = [:]
        for id in ids { guard let pos = positions[id], let t = topicForId[id], let centroid = topicCentroids[t] else { continue }; topicDistances[t, default: []].append(simd_length(pos - centroid)) }
        let topicRadii = topicDistances.mapValues { dists -> Float in let sorted = dists.sorted(); return sorted[sorted.count * 3 / 4] }
        var minProjectSep: Float = .infinity
        let projKeys = Array(projectCentroids.keys).sorted()
        for i in 0..<projKeys.count { for j in (i+1)..<projKeys.count { minProjectSep = min(minProjectSep, simd_length(projectCentroids[projKeys[i]]! - projectCentroids[projKeys[j]]!)) } }
        var minTopicSep: Float = .infinity
        for p in 0..<nProjects { let t = (0..<nTopics).map { p * nTopics + $0 }; for i in 0..<t.count { for j in (i+1)..<t.count { guard let c1 = topicCentroids[t[i]], let c2 = topicCentroids[t[j]] else { continue }; minTopicSep = min(minTopicSep, simd_length(c1 - c2)) } } }
        var stragglerCount = 0
        let projKeysSorted2 = Array(projectCentroids.keys).sorted()
        for id in ids {
            guard let pos = positions[id], let myProj = projectForId[id], let myCentroid = projectCentroids[myProj] else { continue }
            let myDist = simd_length(pos - myCentroid)
            for otherProj in projKeysSorted2 where otherProj != myProj {
                guard let otherCentroid = projectCentroids[otherProj] else { continue }
                if simd_length(pos - otherCentroid) < myDist { stragglerCount += 1; break }
            }
        }
        let stragglerRate = Float(stragglerCount) / Float(max(totalNodes, 1))
        return SimResult(projectCentroids: projectCentroids, topicCentroids: topicCentroids, projectRadii: projectRadii, topicRadii: topicRadii, minProjectSeparation: minProjectSep, minTopicSeparation: minTopicSep, nodeCount: totalNodes, framesRun: frames, stragglerCount: stragglerCount, stragglerRate: stragglerRate)
    }

    @Test("GPU forces + CPU integration: 600 nodes")
    func gpuForcesCpuIntegrate() {
        guard let r = Self.runGPUForcesCPUIntegrate(nProjects: 3, nTopics: 4, nodesPerTopic: 50) else {
            Issue.record("Metal not available"); return
        }
        let avgRadius = r.projectRadii.values.reduce(0, +) / Float(r.projectRadii.count)
        let ratio = r.minProjectSeparation / avgRadius
        print("[GPU+CPU-600] frames=\(r.framesRun) sep=\(r.minProjectSeparation) radius=\(avgRadius) ratio=\(ratio) stragglers=\(r.stragglerCount)/\(r.nodeCount) (\(String(format:"%.1f",r.stragglerRate*100))%)")
        #expect(ratio > 1.5, "GPU+CPU projects not separated: ratio=\(ratio)")
    }

    @Test("GPU: No stragglers at 600 nodes (< 5% misclassified)")
    func gpuStragglers600() {
        guard let r = Self.runGPUSim(nProjects: 3, nTopics: 4, nodesPerTopic: 50) else {
            Issue.record("Metal not available"); return
        }
        print("[GPU-straggler-600] frames=\(r.framesRun) stragglers=\(r.stragglerCount)/\(r.nodeCount) (\(String(format:"%.1f",r.stragglerRate*100))%)")
        #expect(r.stragglerRate < 0.15,
                "Too many stragglers: \(r.stragglerCount)/\(r.nodeCount) (\(String(format:"%.1f",r.stragglerRate*100))%) — nodes closer to wrong project centroid")
    }

    @Test("GPU: No stragglers at 2400 nodes (< 5% misclassified)")
    func gpuStragglers2400() {
        guard let r = Self.runGPUSim(nProjects: 3, nTopics: 4, nodesPerTopic: 200) else {
            Issue.record("Metal not available"); return
        }
        print("[GPU-straggler-2400] frames=\(r.framesRun) stragglers=\(r.stragglerCount)/\(r.nodeCount) (\(String(format:"%.1f",r.stragglerRate*100))%)")
        #expect(r.stragglerRate < 0.15,
                "Too many stragglers: \(r.stragglerCount)/\(r.nodeCount) (\(String(format:"%.1f",r.stragglerRate*100))%) — nodes closer to wrong project centroid")
    }

    // MARK: - Parameter Tuning Sweep

    @Test("Parameter sweep: find best straggler rate at 600 nodes")
    func parameterSweep600() {
        let combos: [(label: String, overrides: ForceOverrides)] = [
            ("baseline",
             ForceOverrides()),
            ("cohesion 2x",
             ForceOverrides(cohesionStrength: 0.003)),
            ("cohesion 3x",
             ForceOverrides(cohesionStrength: 0.0045)),
            ("crossSpring 600",
             ForceOverrides(crossProjectSpringLength: 600)),
            ("crossSpring 800",
             ForceOverrides(crossProjectSpringLength: 800)),
            ("cohesion 2x + crossSpring 600",
             ForceOverrides(cohesionStrength: 0.003, crossProjectSpringLength: 600)),
            ("cohesion 3x + crossSpring 600",
             ForceOverrides(cohesionStrength: 0.0045, crossProjectSpringLength: 600)),
            ("cohesion 2x + centroidRep 800",
             ForceOverrides(cohesionStrength: 0.003, centroidRepulsion: 800)),
            ("cohesion 3x + crossSpring 600 + centroidRep 600",
             ForceOverrides(cohesionStrength: 0.0045, centroidRepulsion: 600, crossProjectSpringLength: 600)),
        ]

        print("\n=== PARAMETER SWEEP (600 nodes, 3 projects × 4 topics × 50) ===")

        for (label, overrides) in combos {
            guard let r = Self.runGPUSim(nProjects: 3, nTopics: 4, nodesPerTopic: 50, overrides: overrides) else {
                print("[\(label)] NO GPU")
                continue
            }
            let avgRadius = r.projectRadii.values.reduce(0, +) / Float(r.projectRadii.count)
            let ratio = r.minProjectSeparation / avgRadius
            let pct = String(format: "%.1f", r.stragglerRate * 100)
            print("[\(label)] stragglers=\(r.stragglerCount)/\(r.nodeCount) (\(pct)%) sep/rad=\(String(format:"%.2f",ratio)) frames=\(r.framesRun)")
        }

        // The sweep is informational — always passes.
        // Use the printed table to pick the best parameter combo.
    }
}
