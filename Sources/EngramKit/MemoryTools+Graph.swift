import Lattice
import MCP
import Foundation

// MARK: - Knowledge Graph

extension MemoryTools {

    // MARK: - connect

    func handleConnect(_ args: [String: Value]?) throws -> CallTool.Result {
        let a = try args.decode(ConnectArgs.self)
        let fromId = Int64(a.from.value)
        let toId = Int64(a.to.value)
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
        guard let fromMem = lattice.objects(Memory.self).where({ $0.primaryKey == fromId }).first else {
            return CallTool.Result(content: [.text("Memory with id \(fromId) not found.")], isError: true)
        }
        guard let toMem = lattice.objects(Memory.self).where({ $0.primaryKey == toId }).first else {
            return CallTool.Result(content: [.text("Memory with id \(toId) not found.")], isError: true)
        }

        guard let fromGlobalId = fromMem.__globalId else {
            throw MCPError.internalError("Memory \(fromId) has no globalId")
        }
        guard let toGlobalId = toMem.__globalId else {
            throw MCPError.internalError("Memory \(toId) has no globalId")
        }

        // Check for duplicate edge
        let existing = lattice.objects(Edge.self)
            .where { $0.sourceGlobalId == fromGlobalId && $0.targetGlobalId == toGlobalId && $0.relation == relationEnum }
        if let edge = existing.first {
            let edgeId = edge.primaryKey.map(String.init) ?? "?"
            return CallTool.Result(
                content: [.text("Edge already exists (edge id: \(edgeId), \(fromId) --[\(relation)]--> \(toId)).")],
                isError: false
            )
        }

        // Create edge — route to same DB as the source memory
        let edgeTargetDB = writeLattice(for: fromMem.project)
        let edge = Edge(sourceGlobalId: fromGlobalId, targetGlobalId: toGlobalId, relation: relationEnum)
        edgeTargetDB.add(edge)

        guard let edgeId = edge.primaryKey else {
            throw MCPError.internalError("Failed to persist edge — primaryKey is nil after add()")
        }

        log("Connected [id:\(fromId)] --[\(relation)]--> [id:\(toId)]")
        return CallTool.Result(
            content: [.text("Connected (edge id: \(edgeId)) [id:\(fromId)] --[\(relation)]--> [id:\(toId)]")],
            isError: false
        )
    }

    // MARK: - disconnect

    func handleDisconnect(_ args: [String: Value]?) throws -> CallTool.Result {
        let a = try args.decode(DisconnectArgs.self)

        // By edge ID
        if let id = a.id?.value {
            let id64 = Int64(id)
            let matches = lattice.objects(Edge.self).where { $0.primaryKey == id64 }
            guard matches.first != nil else {
                return CallTool.Result(content: [.text("Edge with id \(id) not found.")], isError: true)
            }
            lattice.delete(Edge.self, where: { $0.primaryKey == id64 })
            log("Disconnected edge [id:\(id)]")
            return CallTool.Result(
                content: [.text("Deleted edge (id: \(id)).")],
                isError: false
            )
        }

        // By from + to
        guard let from = a.from?.value, let to = a.to?.value else {
            throw MCPError.invalidParams("Provide 'id' or both 'from' and 'to' to target edges.")
        }
        let fromId = Int64(from)
        let toId = Int64(to)

        // Look up memories to get their globalIds
        guard let fromMem = lattice.objects(Memory.self).where({ $0.primaryKey == fromId }).first else {
            return CallTool.Result(content: [.text("Memory with id \(fromId) not found.")], isError: true)
        }
        guard let toMem = lattice.objects(Memory.self).where({ $0.primaryKey == toId }).first else {
            return CallTool.Result(content: [.text("Memory with id \(toId) not found.")], isError: true)
        }

        guard let fromGlobalId = fromMem.__globalId else {
            return CallTool.Result(content: [.text("Memory with id \(fromId) has no globalId.")], isError: true)
        }
        guard let toGlobalId = toMem.__globalId else {
            return CallTool.Result(content: [.text("Memory with id \(toId) has no globalId.")], isError: true)
        }

        var query = lattice.objects(Edge.self)
            .where { $0.sourceGlobalId == fromGlobalId && $0.targetGlobalId == toGlobalId }
        if let relation = a.relation, let relationEnum = Edge.Relation(rawValue: relation) {
            query = query.where { $0.relation == relationEnum }
        }

        let count = query.count
        if count == 0 {
            return CallTool.Result(content: [.text("No edges found from \(fromId) to \(toId).")], isError: false)
        }

        if let relation = a.relation, let relationEnum = Edge.Relation(rawValue: relation) {
            lattice.delete(Edge.self, where: { $0.sourceGlobalId == fromGlobalId && $0.targetGlobalId == toGlobalId && $0.relation == relationEnum })
        } else {
            lattice.delete(Edge.self, where: { $0.sourceGlobalId == fromGlobalId && $0.targetGlobalId == toGlobalId })
        }

        log("Disconnected \(count) edge(s) from [id:\(fromId)] to [id:\(toId)]")
        return CallTool.Result(
            content: [.text("Deleted \(count) edge(s) from [id:\(fromId)] to [id:\(toId)].")],
            isError: false
        )
    }

    // MARK: - graph

    func handleGraph(_ args: [String: Value]?) throws -> CallTool.Result {
        let a = try args.decode(GraphArgs.self)
        let memId = Int64(a.id.value)
        let depth = max(min(a.depth?.value ?? 1, 3), 0)

        // Validate memory exists
        guard let rootMem = lattice.objects(Memory.self).where({ $0.primaryKey == memId }).first else {
            return CallTool.Result(content: [.text("Memory with id \(memId) not found.")], isError: true)
        }

        guard let rootGlobalId = rootMem.__globalId else {
            return CallTool.Result(content: [.text("Memory with id \(memId) has no globalId.")], isError: true)
        }

        // BFS traversal using globalIds
        var visited = Set<UUID>([rootGlobalId])
        var frontier = Set<UUID>([rootGlobalId])
        var allEdges: [(edge: Edge, depth: Int)] = []

        for d in stride(from: 1, through: depth, by: 1) {
            var nextFrontier = Set<UUID>()
            for nodeGlobalId in frontier {
                // Outgoing edges
                let outgoing = lattice.objects(Edge.self).where { $0.sourceGlobalId == nodeGlobalId }
                for edge in outgoing {
                    allEdges.append((edge: edge, depth: d))
                    if !visited.contains(edge.targetGlobalId) {
                        visited.insert(edge.targetGlobalId)
                        nextFrontier.insert(edge.targetGlobalId)
                    }
                }
                // Incoming edges
                let incoming = lattice.objects(Edge.self).where { $0.targetGlobalId == nodeGlobalId }
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

        // Deduplicate edges by primary key
        var seenEdgeIds = Set<Int64>()
        var uniqueEdges: [Edge] = []
        for (edge, _) in allEdges {
            guard let edgeId = edge.primaryKey else { continue }
            if !seenEdgeIds.contains(edgeId) {
                seenEdgeIds.insert(edgeId)
                uniqueEdges.append(edge)
            }
        }

        // Format output — look up memories by globalId for display
        var output = "[id:\(memId)] \(rootMem.content)"

        if uniqueEdges.isEmpty {
            output += "\n\nNo connections."
        } else {
            output += "\n\nConnections:"
            for edge in uniqueEdges {
                if edge.sourceGlobalId == rootGlobalId {
                    // Outgoing
                    let targetMem = lattice.objects(Memory.self).where { $0.__globalId == edge.targetGlobalId }.first
                    let targetContent = targetMem?.content ?? "(deleted)"
                    let targetDisplayId = targetMem?.primaryKey.map(String.init) ?? "?"
                    output += "\n  --[\(edge.relation.rawValue)]--> [id:\(targetDisplayId)] \(targetContent.prefix(80))"
                } else if edge.targetGlobalId == rootGlobalId {
                    // Incoming
                    let sourceMem = lattice.objects(Memory.self).where { $0.__globalId == edge.sourceGlobalId }.first
                    let sourceContent = sourceMem?.content ?? "(deleted)"
                    let sourceDisplayId = sourceMem?.primaryKey.map(String.init) ?? "?"
                    output += "\n  <--[\(edge.relation.rawValue)]-- [id:\(sourceDisplayId)] \(sourceContent.prefix(80))"
                } else {
                    // Edge between two non-root nodes (deeper traversal)
                    let sourceMem = lattice.objects(Memory.self).where { $0.__globalId == edge.sourceGlobalId }.first
                    let targetMem = lattice.objects(Memory.self).where { $0.__globalId == edge.targetGlobalId }.first
                    let sourceContent = sourceMem?.content ?? "(deleted)"
                    let targetContent = targetMem?.content ?? "(deleted)"
                    let sourceDisplayId = sourceMem?.primaryKey.map(String.init) ?? "?"
                    let targetDisplayId = targetMem?.primaryKey.map(String.init) ?? "?"
                    output += "\n  [id:\(sourceDisplayId)] \(sourceContent.prefix(40))... --[\(edge.relation.rawValue)]--> [id:\(targetDisplayId)] \(targetContent.prefix(40))..."
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
            let count = lattice.count(Edge.self, where: { $0.sourceGlobalId == gid || $0.targetGlobalId == gid })
            if count > 0 {
                lattice.delete(Edge.self, where: { $0.sourceGlobalId == gid || $0.targetGlobalId == gid })
                total += count
            }
        }
        return total
    }

    /// BFS graph traversal from a set of starting memory globalIds, returning connected memories
    /// annotated with the depth at which they were discovered.
    func traverseGraph(from startGlobalIds: Set<UUID>, depth: Int, excludeGlobalIds: Set<UUID>) -> [(memory: Memory, depth: Int)] {
        var visited = excludeGlobalIds
        var frontier = startGlobalIds
        var result: [(memory: Memory, depth: Int)] = []
        let now = Date()

        for d in stride(from: 1, through: depth, by: 1) {
            var nextFrontier = Set<UUID>()
            for nodeGlobalId in frontier {
                // Outgoing edges
                for edge in lattice.objects(Edge.self).where({ $0.sourceGlobalId == nodeGlobalId }) {
                    if !visited.contains(edge.targetGlobalId) {
                        visited.insert(edge.targetGlobalId)
                        nextFrontier.insert(edge.targetGlobalId)
                    }
                }
                // Incoming edges
                for edge in lattice.objects(Edge.self).where({ $0.targetGlobalId == nodeGlobalId }) {
                    if !visited.contains(edge.sourceGlobalId) {
                        visited.insert(edge.sourceGlobalId)
                        nextFrontier.insert(edge.sourceGlobalId)
                    }
                }
            }
            // Fetch the memories for the next frontier
            for gid in nextFrontier {
                if let mem = lattice.objects(Memory.self).where({ $0.__globalId == gid && $0.expiresAt > now }).first {
                    result.append((memory: mem, depth: d))
                }
            }
            frontier = nextFrontier
            if frontier.isEmpty { break }
        }

        return result
    }
}
