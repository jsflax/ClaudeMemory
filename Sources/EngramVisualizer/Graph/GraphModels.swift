import SwiftUI
import EngramKit

// MARK: - File-level helpers

func extractLabel(content: String, topic: String) -> String {
    let maxLen = 30
    if let headerRange = content.range(of: #"^#{1,3}\s+"#, options: .regularExpression) {
        let title = content[headerRange.upperBound...].prefix(while: { $0 != "\n" })
        return truncateLabel(String(title), to: maxLen)
    }
    if content.hasPrefix("**"), let end = content.dropFirst(2).range(of: "**") {
        let title = content[content.index(content.startIndex, offsetBy: 2)..<end.lowerBound]
        return truncateLabel(String(title), to: maxLen)
    }
    if let colon = content.firstIndex(of: ":"),
       content.distance(from: content.startIndex, to: colon) < maxLen {
        return String(content[..<colon])
    }
    let firstLine = String(content.prefix(while: { $0 != "\n" }))
    if firstLine.count <= maxLen { return firstLine }
    if topic != "general" {
        let combined = "\(topic): \(firstLine)"
        if combined.count <= maxLen { return combined }
        // Topic fits — truncate content only
        if topic.count + 2 + 5 <= maxLen {
            return "\(topic): \(truncateLabel(firstLine, to: maxLen - topic.count - 2))"
        }
        // Topic too long — truncate both
        let prefix = truncateLabel(topic, to: 12)
        return "\(prefix): \(truncateLabel(firstLine, to: maxLen - prefix.count - 2))"
    }
    return truncateLabel(firstLine, to: maxLen)
}

func truncateLabel(_ text: String, to maxLen: Int) -> String {
    if text.count <= maxLen { return text }
    return String(text.prefix(maxLen - 1)) + "…"
}

// MARK: - Lightweight data structs (replaces heavy Lattice model objects)

struct NodeData {
    let id: UUID
    let project: String
    let topic: String
    let label: String
    let content: String
    let createdAt: Date
    var lastAccessedAt: Date
    let importance: Int
}

struct EdgeData {
    let id: Int64       // Edge primaryKey (DB-local, for observation identity)
    let sourceId: UUID
    let targetId: UUID
    let relation: String
}

struct DyingNode {
    let id: UUID
    let position: CGPoint
    let project: String
    let isHub: Bool
    let importance: Int
    let startTime: Date
}

/// Shared render data store — written by GraphView (insertNodeBatch, handleNodeChange, etc.),
/// read by Graph3DScene's renderTick. Both run on MainActor, so no race.
/// NOT @Observable — must not trigger SwiftUI re-evaluation.
@MainActor
final class GraphRenderStore {
    var nodes: [NodeData] = []
    var nodeById: [UUID: NodeData] = [:]
    var edges: [EdgeData] = []
    var hubs: Set<UUID> = []
    var colorMap: [String: Color] = [:] {
        didSet { colorMapVersion &+= 1 }
    }
    /// Incremented whenever colorMap changes. Scene checks this to invalidate its SIMD cache.
    private(set) var colorMapVersion: UInt64 = 0

    /// Incremented when nodes/hubs/projects change. Replaces O(n) Set comparison per frame in atlas check.
    private(set) var topologyVersion: UInt64 = 0
    func bumpTopology() { topologyVersion &+= 1 }

    // Internal data — never triggers SwiftUI body re-evaluation.
    var allNodes: [UUID: NodeData] = [:]
    var allEdges: [Int64: EdgeData] = [:]
    var edgesByNode: [UUID: [EdgeData]] = [:]
    var pkToGlobalId: [Int64: UUID] = [:]
    var filteredEdgeIds: Set<Int64> = []
    var visibleNodeIds: Set<UUID> = []
    var edgeCountByNode: [UUID: Int] = [:]

    // Visual effects — written by GraphView's observers/timers, read by MetalSceneManager's renderTick.
    var glowingNodes: [UUID: Date] = [:]
    var newNodeGlows: [UUID: Date] = [:]
    var dyingNodes: [UUID: DyingNode] = [:]

    // Derived display data — recomputed on structural changes.
    var topicGroups: [TopicGroupInfo] = []
    var relationCounts: [(key: String, value: Int)] = []
    var clusterGroups: [[UUID]] = []

    // Search state — pushed from @State bindings (SearchBarView writes via Binding).
    var searchMatchIds: Set<UUID> = []
    var isSearchActive: Bool = false

    // Batched observer coalescing — accumulates inserts (pre-built on background thread)
    // and flushes once per run loop iteration. No Lattice reads on main thread.
    var pendingNodeInserts: [(pk: Int64, node: NodeData)] = []
    var pendingEdgeInserts: [(pk: Int64, edge: EdgeData)] = []
    var pendingNodeFlush: Task<Void, Never>?
    var pendingEdgeFlush: Task<Void, Never>?
}

// MARK: - Perf tracking (writable from Canvas closure)

final class PerfStats: @unchecked Sendable {
    var lastFrameTime: CFAbsoluteTime = 0
    var fps: Double = 0
    var drawMs: Double = 0
    var hullMs: Double = 0
    var edgeMs: Double = 0
    var nodeMs: Double = 0
}
