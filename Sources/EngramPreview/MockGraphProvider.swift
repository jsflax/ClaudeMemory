import Foundation
import simd
import Metal
import EngramRealityKit
import EngramSceneKit
import EngramMetalShaders

/// Mock SceneDataProvider using ForceSimulation3D for realistic graph layout.
@Observable
@MainActor
final class MockGraphProvider: SceneDataProvider {
    private(set) var nodes: [RKNodeSnapshot] = []
    private(set) var edges: [RKEdgeSnapshot] = []
    private(set) var hubs: Set<UUID> = []
    private(set) var projectColorMap: [String: SIMD3<Float>] = [:]
    private(set) var positions: [UUID: SIMD3<Float>] = [:]
    var glowingNodes: [UUID: Float] = [:]
    var newNodeGlows: [UUID: Float] = [:]
    var dyingNodes: Set<UUID> = []
    var selectedNode: UUID?
    var expandedHubs: Set<UUID> = []
    var searchMatchIds: Set<UUID> = []
    var isSearchActive: Bool = false
    var projectCentroids: [String: SIMD3<Float>] = [:]
    private(set) var topologyVersion: UInt64 = 0

    // MARK: - Instrumentation (gated on PREVIEW_INSTRUMENTATION=1)
    private var instrumentationEnabled = false
    private var instrumentLogPath = "/tmp/preview-instrumentation"
    private var frameFile: UnsafeMutablePointer<FILE>?
    private var frameIndex = 0
    private var wasSettled = false
    private var forceMsHistory: [Double] = []
    private var wallMsHistory: [Double] = []

    /// Number of nodes to generate.
    var nodeCount: Int = 200 {
        didSet { regenerateGraph() }
    }

    /// Project definitions with relative weight and associated topics.
    private let projectDefs: [(name: String, weight: Float, topics: [String])] = [
        ("Architecture", 0.30, ["rendering", "scene-graph", "ECS", "shaders", "camera"]),
        ("Debugging",    0.20, ["force-sim", "GPU-crash", "concurrency", "memory-leak"]),
        ("Patterns",     0.18, ["MVVM", "coordinator", "dependency-injection", "protocols"]),
        ("Workflow",     0.18, ["CI-CD", "testing", "code-review", "deploy"]),
        ("Preferences",  0.14, ["editor", "keybindings", "theme", "sync"]),
    ]

    /// Real force simulation for proper graph layout.
    let simulation = ForceSimulation3D()

    /// GPU force engine — nil if Metal unavailable.
    private var forceEngine: ForceEngine?
    private var commandQueue: MTLCommandQueue?

    /// Node ID → index mapping for reading simulation positions.
    private var nodeIds: [UUID] = []

    init() {
        // Set up GPU force engine
        if let device = MTLCreateSystemDefaultDevice(),
           let queue = device.makeCommandQueue(),
           let library = try? EngramMetalShaders.makeLibrary(device: device),
           let forceCompute = MetalForceCompute(device: device, library: library) {
            let simState = GPUSimulationState(device: device, library: library)
            self.forceEngine = ForceEngine(forceCompute: forceCompute, simState: simState)
            self.commandQueue = queue
            print("[preview] GPU force engine initialized")
        } else {
            print("[preview] GPU force engine unavailable, using CPU fallback")
        }

        // Instrumentation setup (env vars)
        if ProcessInfo.processInfo.environment["PREVIEW_INSTRUMENTATION"] == "1" {
            instrumentationEnabled = true
            instrumentLogPath = ProcessInfo.processInfo.environment["PREVIEW_LOG_PATH"] ?? instrumentLogPath
            openFrameCSV()
        }
        if let countStr = ProcessInfo.processInfo.environment["PREVIEW_NODE_COUNT"],
           let count = Int(countStr) {
            nodeCount = count  // didSet doesn't fire in class init
        }

        projectColorMap = [
            "Architecture": SIMD3<Float>(0.15, 0.55, 1.0),   // Vivid blue
            "Debugging":    SIMD3<Float>(1.0, 0.2, 0.25),    // Bright red
            "Patterns":     SIMD3<Float>(0.1, 0.85, 0.35),   // Emerald green
            "Workflow":     SIMD3<Float>(1.0, 0.75, 0.0),    // Amber
            "Preferences":  SIMD3<Float>(0.8, 0.2, 1.0),     // Electric purple
        ]
        regenerateGraph()
    }

    /// Pick a project index weighted by `projectDefs.weight`.
    private func weightedProjectIndex() -> Int {
        let r = Float.random(in: 0..<1)
        var cum: Float = 0
        for (i, def) in projectDefs.enumerated() {
            cum += def.weight
            if r < cum { return i }
        }
        return projectDefs.count - 1
    }

    func tick(dt: Float) {
        let wallStart = instrumentationEnabled ? CFAbsoluteTimeGetCurrent() : 0
        var forceMs: Double = 0

        // Dispatch GPU forces if available
        if let engine = forceEngine, let queue = commandQueue {
            let sim = simulation
            if !sim.isSettled, sim.nodeCount > 1, engine.isFullSimAvailable, !engine.isInFlight {
                sim.useGPUForces = true
                let snapshot = ForceEngine.SimulationSnapshot(
                    nodeCount: sim.nodeCount, isSettled: sim.isSettled,
                    posX: sim.posX, posY: sim.posY, posZ: sim.posZ,
                    projectGroups: sim.projectGroupPublic, topicGroups: sim.topicGroupPublic,
                    topicProjectGroup: sim.topicProjectGroupPublic,
                    galaxyGroups: sim.galaxyGroupPublic, galaxyCenters: sim.galaxyCentersPublic,
                    edgeIndices: sim.edgeIndicesPublic, topologyDirty: sim.topologyDirtyForGPU,
                    chargeStrength: sim.chargeStrength, crossChargeMultiplier: sim.crossChargeMultiplier,
                    sameTopicChargeScale: sim.sameTopicChargeScale, sameProjectChargeScale: sim.sameProjectChargeScale,
                    springLength: sim.springLength, crossProjectSpringLength: sim.crossProjectSpringLength,
                    springStrength: sim.springStrength,
                    crossProjectSpringScale: sim.crossProjectSpringScale,
                    cohesionStrength: sim.cohesionStrength, centroidRepulsion: sim.centroidRepulsion,
                    topicCohesionStrength: sim.topicCohesionStrength, topicCentroidRepulsion: sim.topicCentroidRepulsion,
                    centerStrength: sim.centerStrength, center: sim.center,
                    alpha: sim.alpha, damping: 0.78, maxSpeed: 12.0
                )

                let forceStart = instrumentationEnabled ? CFAbsoluteTimeGetCurrent() : 0
                let result = engine.encodeForcePassSync(queue: queue, snapshot: snapshot)
                if instrumentationEnabled { forceMs = (CFAbsoluteTimeGetCurrent() - forceStart) * 1000 }

                if let result {
                    sim.topologyDirtyForGPU = false
                    sim.applyGPUForces(result)
                }

                // Diagnostic: log separation quality every 60 frames
                if sim.framesSinceWake % 60 == 1 || sim.isSettled {
                    var projSums: [Int: (sum: SIMD3<Float>, count: Int)] = [:]
                    for i in 0..<sim.nodeCount {
                        let pg = sim.projectGroupPublic[i]
                        let pos = SIMD3<Float>(sim.posX[i], sim.posY[i], sim.posZ[i])
                        let e = projSums[pg] ?? (.zero, 0)
                        projSums[pg] = (e.sum + pos, e.count + 1)
                    }
                    let centroids = projSums.mapValues { $0.sum / Float($0.count) }
                    var minSep: Float = .infinity
                    let keys = Array(centroids.keys).sorted()
                    for i in 0..<keys.count {
                        for j in (i+1)..<keys.count {
                            minSep = min(minSep, simd_length(centroids[keys[i]]! - centroids[keys[j]]!))
                        }
                    }
                    print("[preview] frame=\(sim.framesSinceWake) alpha=\(String(format:"%.4f",sim.alpha)) maxSpeedSq=\(String(format:"%.2f",sim.lastMaxSpeedSq)) sep=\(String(format:"%.0f",minSep)) settled=\(sim.isSettled)")
                }
            }
        }

        // Tick the force simulation (integrates whatever forces are available)
        simulation.tick()

        // Read positions back from simulation
        positions = simulation.positions

        // Instrumentation: write per-frame CSV and check settlement
        if instrumentationEnabled {
            let wallMs = (CFAbsoluteTimeGetCurrent() - wallStart) * 1000
            let settled = simulation.isSettled
            forceMsHistory.append(forceMs)
            wallMsHistory.append(wallMs)
            let minSep = computeMinProjSep()
            if let f = frameFile {
                let line = String(format: "%d,%.2f,%.2f,%.4f,%.2f,%.1f,%d,%@\n",
                                  frameIndex, wallMs, forceMs, simulation.alpha,
                                  simulation.lastMaxSpeedSq, minSep, simulation.nodeCount,
                                  settled ? "true" : "false")
                fputs(line, f)
                fflush(f)
            }
            frameIndex += 1
            if settled && !wasSettled {
                wasSettled = true
                writeClusterReport()
                writeSettledSignal()
            }
        }

        // Animate glow timers
        for (id, elapsed) in glowingNodes {
            glowingNodes[id] = elapsed + dt
            if elapsed + dt > 4.5 { glowingNodes.removeValue(forKey: id) }
        }
        for (id, elapsed) in newNodeGlows {
            newNodeGlows[id] = elapsed + dt
            if elapsed + dt > 5.8 { newNodeGlows.removeValue(forKey: id) }
        }

        // Recompute centroids occasionally
        if Int.random(in: 0..<60) == 0 {
            computeCentroids()
        }
    }

    // MARK: - Graph Generation

    func regenerateGraph() {
        var newNodes: [RKNodeSnapshot] = []
        var newEdges: [RKEdgeSnapshot] = []
        var newHubs: Set<UUID> = []
        var ids: [UUID] = []
        var edgePairs: [(UUID, UUID)] = []
        var projectForNode: [UUID: String] = [:]
        var topicForNode: [UUID: String] = [:]

        // Nodes grouped by project → topic for intra-cluster edge wiring.
        var buckets: [String: [String: [UUID]]] = [:]

        // --- Generate nodes with weighted project distribution ---
        for i in 0..<nodeCount {
            let id = UUID()
            let pIdx = weightedProjectIndex()
            let def = projectDefs[pIdx]
            let project = def.name
            let topic = def.topics.randomElement()!
            // Skew importance: hubs get 4-5, most nodes 1-3
            let isHub = i % 15 == 0
            let importance = isHub ? Int.random(in: 4...5) : Int.random(in: 1...3)

            newNodes.append(RKNodeSnapshot(
                id: id, project: project, topic: topic,
                label: "\(topic)_\(i)", importance: importance, isHub: isHub
            ))

            if isHub { newHubs.insert(id) }
            ids.append(id)
            projectForNode[id] = project
            topicForNode[id] = topic
            buckets[project, default: [:]][topic, default: []].append(id)
        }

        // --- Intra-topic edges (tight clusters) ---
        for (_, topics) in buckets {
            for (_, members) in topics where members.count >= 2 {
                // Chain within topic
                for i in 1..<members.count {
                    newEdges.append(RKEdgeSnapshot(
                        id: UUID(), sourceId: members[i - 1], targetId: members[i], relation: "relates_to"
                    ))
                    edgePairs.append((members[i - 1], members[i]))
                }
            }
        }

        // --- Intra-project / cross-topic edges (link topics within a project) ---
        for (_, topics) in buckets {
            let topicKeys = Array(topics.keys)
            guard topicKeys.count >= 2 else { continue }
            for i in 0..<topicKeys.count {
                let j = (i + 1) % topicKeys.count
                let a = topics[topicKeys[i]]!, b = topics[topicKeys[j]]!
                // A few bridges between adjacent topics
                let bridgeCount = max(1, min(a.count, b.count) / 3)
                for k in 0..<bridgeCount {
                    let src = a[k % a.count], tgt = b[k % b.count]
                    newEdges.append(RKEdgeSnapshot(
                        id: UUID(), sourceId: src, targetId: tgt, relation: "part_of"
                    ))
                    edgePairs.append((src, tgt))
                }
            }
        }

        // --- Cross-project edges (sparse, keep clusters separated) ---
        let crossCount = max(nodeCount / 25, 3)
        for _ in 0..<crossCount {
            let i = Int.random(in: 0..<newNodes.count)
            var j = Int.random(in: 0..<newNodes.count)
            // Ensure cross-project
            while newNodes[j].project == newNodes[i].project && newNodes.count > 1 {
                j = Int.random(in: 0..<newNodes.count)
            }
            newEdges.append(RKEdgeSnapshot(
                id: UUID(), sourceId: newNodes[i].id, targetId: newNodes[j].id, relation: "derived_from"
            ))
            edgePairs.append((newNodes[i].id, newNodes[j].id))
        }

        self.nodes = newNodes
        self.edges = newEdges
        self.hubs = newHubs
        self.nodeIds = ids
        self.topologyVersion += 1

        // Feed into force simulation
        simulation.updateGraph(
            nodeIds: Set(ids),
            edges: edgePairs,
            projectForNode: projectForNode,
            topicForNode: topicForNode
        )
        simulation.wake()

        // Reset instrumentation state for new graph
        if instrumentationEnabled {
            frameIndex = 0
            wasSettled = false
            forceMsHistory.removeAll()
            wallMsHistory.removeAll()
            if let f = frameFile { fclose(f) }
            openFrameCSV()
            try? FileManager.default.removeItem(atPath: "\(instrumentLogPath)-settled")
            try? FileManager.default.removeItem(atPath: "\(instrumentLogPath)-final.json")
        }

        // Read initial positions
        positions = simulation.positions
        computeCentroids()

        // Demo glow effects
        if newNodes.count > 5 {
            glowingNodes[newNodes[2].id] = 0
            newNodeGlows[newNodes[4].id] = 0
        }
    }

    /// Insert a new node with arrival glow, wired to a random same-project node.
    func insertNewNode() {
        let pIdx = weightedProjectIndex()
        let def = projectDefs[pIdx]
        let project = def.name
        let topic = def.topics.randomElement()!
        let id = UUID()

        let node = RKNodeSnapshot(
            id: id, project: project, topic: topic,
            label: "\(topic)_new_\(nodes.count)", importance: Int.random(in: 1...3), isHub: false
        )
        nodes.append(node)
        nodeIds.append(id)

        // Wire to a random same-project node
        if let peer = nodes.filter({ $0.project == project && $0.id != id }).randomElement() {
            edges.append(RKEdgeSnapshot(
                id: UUID(), sourceId: peer.id, targetId: id, relation: "relates_to"
            ))
            simulation.addEdge(from: peer.id, to: id)
        }

        simulation.addNode(id, project: project, topic: topic)
        newNodeGlows[id] = 0
        topologyVersion += 1
    }

    private func computeCentroids() {
        var sums: [String: (sum: SIMD3<Float>, count: Int)] = [:]
        for node in nodes {
            guard let pos = positions[node.id] else { continue }
            let entry = sums[node.project] ?? (.zero, 0)
            sums[node.project] = (entry.sum + pos, entry.count + 1)
        }
        projectCentroids = sums.mapValues { $0.sum / Float($0.count) }
    }

    // MARK: - Instrumentation Helpers

    /// Replace NaN/infinity with 0 for JSON safety.
    private func safe(_ v: Double) -> Double { v.isFinite ? v : 0 }
    private func safe(_ v: Float) -> Double { Double(v).isFinite ? Double(v) : 0 }

    private func openFrameCSV() {
        let csvPath = "\(instrumentLogPath)-frames.csv"
        frameFile = fopen(csvPath, "w")
        if let f = frameFile {
            fputs("frame,wall_ms,force_ms,alpha,maxSpeedSq,minProjSep,nodeCount,settled\n", f)
            fflush(f)
        }
    }

    private func computeMinProjSep() -> Float {
        let sim = simulation
        guard sim.nodeCount > 0 else { return 0 }
        var projSums: [Int: (sum: SIMD3<Float>, count: Int)] = [:]
        for i in 0..<sim.nodeCount {
            let pg = sim.projectGroupPublic[i]
            let pos = SIMD3<Float>(sim.posX[i], sim.posY[i], sim.posZ[i])
            let e = projSums[pg] ?? (.zero, 0)
            projSums[pg] = (e.sum + pos, e.count + 1)
        }
        let centroids = projSums.mapValues { $0.sum / Float($0.count) }
        var minSep: Float = .infinity
        let keys = Array(centroids.keys).sorted()
        for i in 0..<keys.count {
            for j in (i+1)..<keys.count {
                minSep = min(minSep, simd_length(centroids[keys[i]]! - centroids[keys[j]]!))
            }
        }
        return minSep == .infinity ? 0 : minSep
    }

    private func writeClusterReport() {
        let sim = simulation
        let simIds = sim.nodeIds

        // Build UUID → node info lookup
        var idToProject: [UUID: String] = [:]
        var idToTopic: [UUID: String] = [:]
        var idToLabel: [UUID: String] = [:]
        for node in nodes {
            idToProject[node.id] = node.project
            idToTopic[node.id] = node.topic
            idToLabel[node.id] = node.label
        }

        // Gather positions grouped by project and topic
        var projectPositions: [String: [SIMD3<Float>]] = [:]
        var topicPositions: [String: [String: [SIMD3<Float>]]] = [:]
        struct NodeInfo { let label: String; let project: String; let pos: SIMD3<Float> }
        var allNodeInfo: [NodeInfo] = []

        for i in 0..<sim.nodeCount {
            guard i < simIds.count else { break }
            let id = simIds[i]
            let pos = SIMD3<Float>(sim.posX[i], sim.posY[i], sim.posZ[i])
            let project = idToProject[id] ?? "Unknown"
            let topic = idToTopic[id] ?? "Unknown"
            let label = idToLabel[id] ?? "?"
            projectPositions[project, default: []].append(pos)
            topicPositions[project, default: [:]][topic, default: []].append(pos)
            allNodeInfo.append(NodeInfo(label: label, project: project, pos: pos))
        }

        // Project centroids and P75 radii
        var projCentroids: [String: SIMD3<Float>] = [:]
        var projRadii: [String: Float] = [:]
        var projCounts: [String: Int] = [:]

        for (project, positions) in projectPositions {
            let centroid = positions.reduce(SIMD3<Float>.zero, +) / Float(positions.count)
            projCentroids[project] = centroid
            projCounts[project] = positions.count
            let dists = positions.map { simd_length($0 - centroid) }.sorted()
            projRadii[project] = dists[dists.count * 3 / 4]
        }

        // Topic centroids and P75 radii per project
        struct TopicCluster { let name: String; let centroid: SIMD3<Float>; let radiusP75: Float; let nodeCount: Int }
        var projectTopicClusters: [String: [TopicCluster]] = [:]

        for (project, topics) in topicPositions {
            for (topic, positions) in topics {
                let centroid = positions.reduce(SIMD3<Float>.zero, +) / Float(positions.count)
                let dists = positions.map { simd_length($0 - centroid) }.sorted()
                let r75 = dists.isEmpty ? Float(0) : dists[dists.count * 3 / 4]
                projectTopicClusters[project, default: []].append(
                    TopicCluster(name: topic, centroid: centroid, radiusP75: r75, nodeCount: positions.count)
                )
            }
        }

        // Project separation
        let projNames = Array(projCentroids.keys).sorted()
        var pairs: [(String, String, Float)] = []
        var minProjSep: Float = .infinity
        var totalProjSep: Float = 0
        var pairCount = 0
        for i in 0..<projNames.count {
            for j in (i+1)..<projNames.count {
                let d = simd_length(projCentroids[projNames[i]]! - projCentroids[projNames[j]]!)
                pairs.append((projNames[i], projNames[j], d))
                minProjSep = min(minProjSep, d)
                totalProjSep += d
                pairCount += 1
            }
        }
        let avgProjSep = pairCount > 0 ? totalProjSep / Float(pairCount) : 0

        // Per-project compactness and separation quality
        // R50 = median distance to centroid, compactness = R50/R75 (tight≈0.65, fragmented<0.4)
        // sepRatio = dist to nearest other project / R75 (>2 = clear gap, <1 = overlapping)
        var projR50: [String: Float] = [:]
        for (project, positions) in projectPositions {
            guard let centroid = projCentroids[project] else { continue }
            let dists = positions.map { simd_length($0 - centroid) }.sorted()
            projR50[project] = dists[dists.count / 2]
        }

        // Topic separation per project + topic overlap rate
        // topicOverlapRate = fraction of nodes closer to a *different* topic centroid than their own
        var topicSepEntries: [[String: Any]] = []
        for project in projNames {
            let topics = projectTopicClusters[project] ?? []
            var minSep: Float = .infinity
            var totalSep: Float = 0
            var count = 0
            for i in 0..<topics.count {
                for j in (i+1)..<topics.count {
                    let d = simd_length(topics[i].centroid - topics[j].centroid)
                    minSep = min(minSep, d)
                    totalSep += d
                    count += 1
                }
            }

            // Topic overlap: for each node in this project, check if it's closer to
            // a different topic's centroid than its own topic's centroid
            var topicOverlapCount = 0
            var topicNodeCount = 0
            let topicCentroidMap = Dictionary(uniqueKeysWithValues: topics.map { ($0.name, $0.centroid) })
            if let projTopics = topicPositions[project] {
                for (topic, positions) in projTopics {
                    guard let myCentroid = topicCentroidMap[topic] else { continue }
                    for pos in positions {
                        topicNodeCount += 1
                        let myDist = simd_length(pos - myCentroid)
                        for (otherTopic, otherCentroid) in topicCentroidMap where otherTopic != topic {
                            if simd_length(pos - otherCentroid) < myDist {
                                topicOverlapCount += 1
                                break
                            }
                        }
                    }
                }
            }

            let r75 = projRadii[project] ?? 1
            let r50 = projR50[project] ?? 0
            // Distance to nearest other project centroid
            var nearestProjDist: Float = .infinity
            if let c = projCentroids[project] {
                for (other, oc) in projCentroids where other != project {
                    nearestProjDist = min(nearestProjDist, simd_length(c - oc))
                }
            }
            topicSepEntries.append([
                "project": project,
                "minSep": safe(count > 0 ? minSep : 0),
                "avgSep": safe(count > 0 ? totalSep / Float(count) : 0),
                "topicCount": topics.count,
                "topicOverlapRate": safe(topicNodeCount > 0 ? Float(topicOverlapCount) / Float(topicNodeCount) : 0),
                "compactness": safe(r50 / r75),
                "sepRatio": safe(nearestProjDist / r75)
            ] as [String: Any])
        }

        // Global sep/radius ratio — the single most important quality metric
        let meanR75 = projRadii.values.reduce(Float(0), +) / Float(max(projRadii.count, 1))
        let sepRadiusRatio = meanR75 > 0 ? (minProjSep == .infinity ? 0 : minProjSep) / meanR75 : 0

        // Straggler detection: nodes closer to a different project's centroid than their own
        var stragglerNodes: [[String: Any]] = []
        for info in allNodeInfo {
            guard let myCentroid = projCentroids[info.project] else { continue }
            let myDist = simd_length(info.pos - myCentroid)
            for (proj, centroid) in projCentroids where proj != info.project {
                if simd_length(info.pos - centroid) < myDist {
                    stragglerNodes.append([
                        "label": info.label,
                        "project": info.project,
                        "nearestWrong": proj
                    ])
                    break
                }
            }
        }

        // Performance percentiles from accumulated frame history
        let sortedForce = forceMsHistory.sorted()
        let sortedWall = wallMsHistory.sorted()
        func pct(_ sorted: [Double], _ p: Double) -> Double {
            guard !sorted.isEmpty else { return 0 }
            return sorted[min(Int(Double(sorted.count) * p), sorted.count - 1)]
        }

        // Build JSON via dictionaries — all floats sanitized to avoid NaN/infinity
        let report: [String: Any] = [
            "nodeCount": sim.nodeCount,
            "edgeCount": edges.count,
            "framesToSettle": frameIndex,
            "projects": projNames.map { proj -> [String: Any] in
                let c = projCentroids[proj]!
                return [
                    "name": proj,
                    "centroid": [safe(c.x), safe(c.y), safe(c.z)],
                    "radiusP75": safe(projRadii[proj] ?? 0),
                    "nodeCount": projCounts[proj] ?? 0,
                    "topics": (projectTopicClusters[proj] ?? []).map { t -> [String: Any] in
                        [
                            "name": t.name,
                            "centroid": [safe(t.centroid.x), safe(t.centroid.y), safe(t.centroid.z)],
                            "radiusP75": safe(t.radiusP75),
                            "nodeCount": t.nodeCount
                        ]
                    }
                ]
            },
            "projectSeparation": [
                "min": safe(minProjSep),
                "avg": safe(avgProjSep),
                "sepRadiusRatio": safe(sepRadiusRatio),
                "pairs": pairs.map { [$0.0, $0.1, safe($0.2)] as [Any] }
            ] as [String: Any],
            "topicSeparation": topicSepEntries,
            "stragglers": [
                "count": stragglerNodes.count,
                "rate": safe(Double(stragglerNodes.count) / Double(max(sim.nodeCount, 1))),
                "nodes": Array(stragglerNodes.prefix(20))
            ] as [String: Any],
            "performance": [
                "forceMs": [
                    "p50": safe(pct(sortedForce, 0.5)),
                    "p95": safe(pct(sortedForce, 0.95)),
                    "max": safe(sortedForce.last ?? 0)
                ] as [String: Any],
                "wallMs": [
                    "p50": safe(pct(sortedWall, 0.5)),
                    "p95": safe(pct(sortedWall, 0.95)),
                    "max": safe(sortedWall.last ?? 0)
                ] as [String: Any]
            ] as [String: Any]
        ]

        if let data = try? JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]),
           let json = String(data: data, encoding: .utf8) {
            try? json.write(toFile: "\(instrumentLogPath)-final.json", atomically: true, encoding: .utf8)
        }
    }

    private func writeSettledSignal() {
        FileManager.default.createFile(atPath: "\(instrumentLogPath)-settled", contents: nil)
    }
}
