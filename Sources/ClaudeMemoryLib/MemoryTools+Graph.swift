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

        // Validate both memories exist
        guard lattice.objects(Memory.self).where({ $0.primaryKey == fromId }).first != nil else {
            return CallTool.Result(content: [.text("Memory with id \(fromId) not found.")], isError: true)
        }
        guard lattice.objects(Memory.self).where({ $0.primaryKey == toId }).first != nil else {
            return CallTool.Result(content: [.text("Memory with id \(toId) not found.")], isError: true)
        }

        // Check for duplicate edge
        let existing = lattice.objects(Edge.self)
            .where { $0.sourceId == fromId && $0.targetId == toId && $0.relation == relation }
        if let edge = existing.first {
            let edgeId = edge.primaryKey.map(String.init) ?? "?"
            return CallTool.Result(
                content: [.text("Edge already exists (edge id: \(edgeId), \(fromId) --[\(relation)]--> \(toId)).")],
                isError: false
            )
        }

        // Create edge
        let edge = Edge(sourceId: fromId, targetId: toId, relation: relation)
        lattice.add(edge)

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

        var query = lattice.objects(Edge.self)
            .where { $0.sourceId == fromId && $0.targetId == toId }
        if let relation = a.relation {
            query = query.where { $0.relation == relation }
        }

        let count = query.count
        if count == 0 {
            return CallTool.Result(content: [.text("No edges found from \(fromId) to \(toId).")], isError: false)
        }

        if let relation = a.relation {
            lattice.delete(Edge.self, where: { $0.sourceId == fromId && $0.targetId == toId && $0.relation == relation })
        } else {
            lattice.delete(Edge.self, where: { $0.sourceId == fromId && $0.targetId == toId })
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
        let depth = min(a.depth?.value ?? 1, 3)

        // Validate memory exists
        guard let rootMem = lattice.objects(Memory.self).where({ $0.primaryKey == memId }).first else {
            return CallTool.Result(content: [.text("Memory with id \(memId) not found.")], isError: true)
        }

        // BFS traversal
        var visited = Set<Int64>([memId])
        var frontier = Set<Int64>([memId])
        var allEdges: [(edge: Edge, depth: Int)] = []

        for d in 1...depth {
            var nextFrontier = Set<Int64>()
            for nodeId in frontier {
                // Outgoing edges
                let outgoing = lattice.objects(Edge.self).where { $0.sourceId == nodeId }
                for edge in outgoing {
                    allEdges.append((edge: edge, depth: d))
                    if !visited.contains(edge.targetId) {
                        visited.insert(edge.targetId)
                        nextFrontier.insert(edge.targetId)
                    }
                }
                // Incoming edges
                let incoming = lattice.objects(Edge.self).where { $0.targetId == nodeId }
                for edge in incoming {
                    allEdges.append((edge: edge, depth: d))
                    if !visited.contains(edge.sourceId) {
                        visited.insert(edge.sourceId)
                        nextFrontier.insert(edge.sourceId)
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

        // Format output
        var output = "[id:\(memId)] \(rootMem.content)"

        if uniqueEdges.isEmpty {
            output += "\n\nNo connections."
        } else {
            output += "\n\nConnections:"
            for edge in uniqueEdges {
                if edge.sourceId == memId {
                    // Outgoing
                    let targetContent = lattice.objects(Memory.self)
                        .where { $0.primaryKey == edge.targetId }.first?.content ?? "(deleted)"
                    output += "\n  --[\(edge.relation)]--> [id:\(edge.targetId)] \(targetContent.prefix(80))"
                } else if edge.targetId == memId {
                    // Incoming
                    let sourceContent = lattice.objects(Memory.self)
                        .where { $0.primaryKey == edge.sourceId }.first?.content ?? "(deleted)"
                    output += "\n  <--[\(edge.relation)]-- [id:\(edge.sourceId)] \(sourceContent.prefix(80))"
                } else {
                    // Edge between two non-root nodes (deeper traversal)
                    let sourceContent = lattice.objects(Memory.self)
                        .where { $0.primaryKey == edge.sourceId }.first?.content ?? "(deleted)"
                    let targetContent = lattice.objects(Memory.self)
                        .where { $0.primaryKey == edge.targetId }.first?.content ?? "(deleted)"
                    output += "\n  [id:\(edge.sourceId)] \(sourceContent.prefix(40))... --[\(edge.relation)]--> [id:\(edge.targetId)] \(targetContent.prefix(40))..."
                }
            }
        }

        return CallTool.Result(content: [.text(output)], isError: false)
    }

    // MARK: - Graph Helpers

    /// Delete all edges where sourceId or targetId is in the given set. Returns count deleted.
    @discardableResult
    func deleteEdgesForMemories(_ ids: [Int64]) -> Int {
        var total = 0
        for id in ids {
            let count = lattice.count(Edge.self, where: { $0.sourceId == id || $0.targetId == id })
            if count > 0 {
                lattice.delete(Edge.self, where: { $0.sourceId == id || $0.targetId == id })
                total += count
            }
        }
        return total
    }

    /// BFS graph traversal from a set of starting memory IDs, returning connected memories.
    func traverseGraph(from startIds: Set<Int64>, depth: Int, excludeIds: Set<Int64>) -> [Memory] {
        var visited = excludeIds
        var frontier = startIds
        var result: [Memory] = []
        let now = Date()

        for _ in 1...depth {
            var nextFrontier = Set<Int64>()
            for nodeId in frontier {
                // Outgoing edges
                for edge in lattice.objects(Edge.self).where({ $0.sourceId == nodeId }) {
                    if !visited.contains(edge.targetId) {
                        visited.insert(edge.targetId)
                        nextFrontier.insert(edge.targetId)
                    }
                }
                // Incoming edges
                for edge in lattice.objects(Edge.self).where({ $0.targetId == nodeId }) {
                    if !visited.contains(edge.sourceId) {
                        visited.insert(edge.sourceId)
                        nextFrontier.insert(edge.sourceId)
                    }
                }
            }
            // Fetch the memories for the next frontier
            for id in nextFrontier {
                if let mem = lattice.objects(Memory.self).where({ $0.primaryKey == id && $0.expiresAt > now }).first {
                    result.append(mem)
                }
            }
            frontier = nextFrontier
            if frontier.isEmpty { break }
        }

        return result
    }
}
