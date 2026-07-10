import Lattice
import MCP
import Foundation

// MARK: - Knowledge Graph

extension MemoryTools {

    // MARK: - connect

    func handleConnect(_ args: [String: Value]?) throws -> CallTool.Result {
        let a = try args.decode(ConnectArgs.self)
        let fromGid = a.from.value
        let toGid = a.to.value
        let relation = a.relation

        // Validate relation
        guard validRelations.contains(relation) else {
            throw MCPError.invalidParams("Invalid relation '\(relation)'. Must be one of: \(validRelations.sorted().joined(separator: ", "))")
        }

        // Parse relation enum
        guard let relationEnum = Edge.Relation(rawValue: relation) else {
            throw MCPError.invalidParams("Invalid relation '\(relation)'.")
        }

        // Validate both memories exist
        guard findMemory(id: fromGid) != nil else {
            return CallTool.Result(content: [.text("Memory with id \(fromGid.uuidString) not found.")], isError: true)
        }
        guard findMemory(id: toGid) != nil else {
            return CallTool.Result(content: [.text("Memory with id \(toGid.uuidString) not found.")], isError: true)
        }

        // Check for duplicate edge
        let existing = localLattice.objects(Edge.self)
            .where { $0.sourceGlobalId == fromGid && $0.targetGlobalId == toGid && $0.relation == relationEnum }
        if let edge = existing.first {
            let edgeGid = edge.__globalId?.uuidString ?? "?"
            return CallTool.Result(
                content: [.text("Edge already exists (edge id: \(edgeGid), \(fromGid.uuidString) --[\(relation)]--> \(toGid.uuidString)).")],
                isError: false
            )
        }

        // Create edge (always in localLattice)
        let edge = Edge(sourceGlobalId: fromGid, targetGlobalId: toGid, relation: relationEnum)
        localLattice.add(edge)

        guard let edgeGid = edge.__globalId else {
            throw MCPError.internalError("Failed to persist edge — globalId is nil after add()")
        }

        log("Connected [id:\(fromGid.uuidString)] --[\(relation)]--> [id:\(toGid.uuidString)]")
        return CallTool.Result(
            content: [.text("Connected (edge id: \(edgeGid.uuidString)) [id:\(fromGid.uuidString)] --[\(relation)]--> [id:\(toGid.uuidString)]")],
            isError: false
        )
    }

    // MARK: - disconnect

    func handleDisconnect(_ args: [String: Value]?) throws -> CallTool.Result {
        let a = try args.decode(DisconnectArgs.self)

        // By edge UUID
        if let edgeGid = a.id?.value {
            guard let (_, foundLattice) = findEdge(id: edgeGid) else {
                return CallTool.Result(content: [.text("Edge with id \(edgeGid.uuidString) not found.")], isError: true)
            }
            foundLattice.delete(Edge.self, where: { $0.__globalId == edgeGid })
            log("Disconnected edge [id:\(edgeGid.uuidString)]")
            return CallTool.Result(
                content: [.text("Deleted edge (id: \(edgeGid.uuidString)).")],
                isError: false
            )
        }

        // By from + to (memory UUIDs)
        guard let fromGid = a.from?.value, let toGid = a.to?.value else {
            throw MCPError.invalidParams("Provide 'id' or both 'from' and 'to' to target edges.")
        }

        // Validate memories exist
        guard findMemory(id: fromGid) != nil else {
            return CallTool.Result(content: [.text("Memory with id \(fromGid.uuidString) not found.")], isError: true)
        }
        guard findMemory(id: toGid) != nil else {
            return CallTool.Result(content: [.text("Memory with id \(toGid.uuidString) not found.")], isError: true)
        }

        var query = localLattice.objects(Edge.self)
            .where { $0.sourceGlobalId == fromGid && $0.targetGlobalId == toGid }
        if let relation = a.relation, let relationEnum = Edge.Relation(rawValue: relation) {
            query = query.where { $0.relation == relationEnum }
        }

        let count = query.count
        if count == 0 {
            return CallTool.Result(content: [.text("No edges found from \(fromGid.uuidString) to \(toGid.uuidString).")], isError: false)
        }

        if let relation = a.relation, let relationEnum = Edge.Relation(rawValue: relation) {
            localLattice.delete(Edge.self, where: { $0.sourceGlobalId == fromGid && $0.targetGlobalId == toGid && $0.relation == relationEnum })
        } else {
            localLattice.delete(Edge.self, where: { $0.sourceGlobalId == fromGid && $0.targetGlobalId == toGid })
        }

        log("Disconnected \(count) edge(s) from [id:\(fromGid.uuidString)] to [id:\(toGid.uuidString)]")
        return CallTool.Result(
            content: [.text("Deleted \(count) edge(s) from [id:\(fromGid.uuidString)] to [id:\(toGid.uuidString)].")],
            isError: false
        )
    }

    // MARK: - graph

    func handleGraph(_ args: [String: Value]?) throws -> CallTool.Result {
        let a = try args.decode(GraphArgs.self)
        let memGid = a.id.value
        let depth = max(min(a.depth?.value ?? 1, 3), 0)

        // Validate memory exists
        guard let (rootMem, rootLattice) = findMemory(id: memGid) else {
            return CallTool.Result(content: [.text("Memory with id \(memGid.uuidString) not found.")], isError: true)
        }

        // BFS traversal using globalIds on the same lattice the root was found in
        var visited = Set<UUID>([memGid])
        var frontier = Set<UUID>([memGid])
        var allEdges: [(edge: Edge, depth: Int)] = []

        for d in stride(from: 1, through: depth, by: 1) {
            var nextFrontier = Set<UUID>()
            for nodeGlobalId in frontier {
                // Outgoing edges
                let outgoing = rootLattice.objects(Edge.self).where { $0.sourceGlobalId == nodeGlobalId }
                for edge in outgoing {
                    allEdges.append((edge: edge, depth: d))
                    if !visited.contains(edge.targetGlobalId) {
                        visited.insert(edge.targetGlobalId)
                        nextFrontier.insert(edge.targetGlobalId)
                    }
                }
                // Incoming edges
                let incoming = rootLattice.objects(Edge.self).where { $0.targetGlobalId == nodeGlobalId }
                for edge in incoming {
                    allEdges.append((edge: edge, depth: d))
                    if !visited.contains(edge.sourceGlobalId) {
                        visited.insert(edge.sourceGlobalId)
                        nextFrontier.insert(edge.sourceGlobalId)
                    }
                }
            }
            frontier = nextFrontier
            if frontier.isEmpty { break }
        }

        // Deduplicate edges by globalId
        var seenEdgeGids = Set<UUID>()
        var uniqueEdges: [Edge] = []
        for (edge, _) in allEdges {
            guard let edgeGid = edge.__globalId else { continue }
            if !seenEdgeGids.contains(edgeGid) {
                seenEdgeGids.insert(edgeGid)
                uniqueEdges.append(edge)
            }
        }

        // Format output — look up memories by globalId for display
        var output = "[id:\(memGid.uuidString)] \(rootMem.content)"

        if uniqueEdges.isEmpty {
            output += "\n\nNo connections."
        } else {
            output += "\n\nConnections:"
            for edge in uniqueEdges {
                if edge.sourceGlobalId == memGid {
                    // Outgoing
                    let targetMem = rootLattice.objects(Memory.self).where { $0.__globalId == edge.targetGlobalId }.first
                    let targetContent = targetMem?.content ?? "(deleted)"
                    output += "\n  --[\(edge.relation.rawValue)]--> [id:\(edge.targetGlobalId.uuidString)] \(targetContent.prefix(80))"
                } else if edge.targetGlobalId == memGid {
                    // Incoming
                    let sourceMem = rootLattice.objects(Memory.self).where { $0.__globalId == edge.sourceGlobalId }.first
                    let sourceContent = sourceMem?.content ?? "(deleted)"
                    output += "\n  <--[\(edge.relation.rawValue)]-- [id:\(edge.sourceGlobalId.uuidString)] \(sourceContent.prefix(80))"
                } else {
                    // Edge between two non-root nodes (deeper traversal)
                    let sourceMem = rootLattice.objects(Memory.self).where { $0.__globalId == edge.sourceGlobalId }.first
                    let targetMem = rootLattice.objects(Memory.self).where { $0.__globalId == edge.targetGlobalId }.first
                    let sourceContent = sourceMem?.content ?? "(deleted)"
                    let targetContent = targetMem?.content ?? "(deleted)"
                    output += "\n  [id:\(edge.sourceGlobalId.uuidString)] \(sourceContent.prefix(40))... --[\(edge.relation.rawValue)]--> [id:\(edge.targetGlobalId.uuidString)] \(targetContent.prefix(40))..."
                }
            }
        }

        return CallTool.Result(content: [.text(output)], isError: false)
    }

    // MARK: - Graph Helpers

    /// Delete all edges where sourceGlobalId or targetGlobalId is in the given set. Returns count deleted.
    @discardableResult
    func deleteEdgesForMemories(_ globalIds: [UUID]) -> Int {
        var total = 0
        for gid in globalIds {
            let count = localLattice.count(Edge.self, where: { $0.sourceGlobalId == gid || $0.targetGlobalId == gid })
            if count > 0 {
                localLattice.delete(Edge.self, where: { $0.sourceGlobalId == gid || $0.targetGlobalId == gid })
                total += count
            }
        }
        return total
    }

    /// BFS graph traversal from a set of starting memory globalIds, returning connected memories
    /// annotated with the depth at which they were discovered and the edge that connected them.
    ///
    /// When a `filter` closure is provided, each discovered memory is tested against it.
    /// Memories that fail the filter are excluded from results AND removed from the BFS
    /// frontier — so their neighbors at deeper depths are never reached.
    func traverseGraph(
        from startGlobalIds: Set<UUID>,
        depth: Int,
        excludeGlobalIds: Set<UUID>,
        db: Lattice,
        filter: ((Memory, Edge) -> Bool)? = nil
    ) -> [(memory: Memory, depth: Int, connectingEdge: Edge)] {
        var visited = excludeGlobalIds
        var frontier = startGlobalIds
        var result: [(memory: Memory, depth: Int, connectingEdge: Edge)] = []
        // Map from discovered globalId to the edge that first reached it
        var connectingEdges: [UUID: Edge] = [:]
        let now = Date()

        for d in stride(from: 1, through: depth, by: 1) {
            var nextFrontier = Set<UUID>()
            for nodeGlobalId in frontier {
                // Outgoing edges. materialize(): the query hydrated each edge
                // row already — the 2-3 field reads here plus the formatting
                // reads in the caller become statement-free.
                for edge in db.objects(Edge.self).where({ $0.sourceGlobalId == nodeGlobalId }) {
                    edge.materialize()
                    if !visited.contains(edge.targetGlobalId) {
                        visited.insert(edge.targetGlobalId)
                        nextFrontier.insert(edge.targetGlobalId)
                        connectingEdges[edge.targetGlobalId] = edge
                    }
                }
                // Incoming edges
                for edge in db.objects(Edge.self).where({ $0.targetGlobalId == nodeGlobalId }) {
                    edge.materialize()
                    if !visited.contains(edge.sourceGlobalId) {
                        visited.insert(edge.sourceGlobalId)
                        nextFrontier.insert(edge.sourceGlobalId)
                        connectingEdges[edge.sourceGlobalId] = edge
                    }
                }
            }
            // Fetch memories and apply filter; only memories that pass continue in BFS
            var passingFrontier = Set<UUID>()
            for gid in nextFrontier {
                if let mem = db.objects(Memory.self).where({ $0.__globalId == gid && $0.expiresAt > now }).first,
                   let edge = connectingEdges[gid] {
                    // Hydrated by the fetch — the filter's embedding read and
                    // the caller's formatting reads (content up to 4x per
                    // large memory) serve from the snapshot.
                    mem.materialize()
                    if let filter, !filter(mem, edge) {
                        continue  // excluded from results and frontier
                    }
                    result.append((memory: mem, depth: d, connectingEdge: edge))
                    passingFrontier.insert(gid)
                }
            }
            frontier = passingFrontier
            if frontier.isEmpty { break }
        }

        return result
    }
}
