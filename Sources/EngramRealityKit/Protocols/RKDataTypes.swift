import Foundation
import simd

/// Lightweight node snapshot — mirrors NodeData without depending on EngramKit.
public struct RKNodeSnapshot: Sendable {
    public let id: UUID
    public let project: String
    public let topic: String
    public let label: String
    public let content: String
    public let importance: Int
    public let isHub: Bool
    public let createdAt: Date
    public let lastAccessedAt: Date

    public init(
        id: UUID,
        project: String,
        topic: String,
        label: String,
        content: String = "",
        importance: Int,
        isHub: Bool,
        createdAt: Date = .distantPast,
        lastAccessedAt: Date = .distantPast
    ) {
        self.id = id
        self.project = project
        self.topic = topic
        self.label = label
        self.content = content
        self.importance = importance
        self.isHub = isHub
        self.createdAt = createdAt
        self.lastAccessedAt = lastAccessedAt
    }
}

/// Lightweight edge snapshot — mirrors EdgeData without depending on EngramKit.
public struct RKEdgeSnapshot: Sendable {
    public let id: UUID
    public let sourceId: UUID
    public let targetId: UUID
    public let relation: String

    public init(
        id: UUID,
        sourceId: UUID,
        targetId: UUID,
        relation: String
    ) {
        self.id = id
        self.sourceId = sourceId
        self.targetId = targetId
        self.relation = relation
    }
}

/// Mascot behavioral state, mirroring MascotSystem.MascotState.
public enum MascotBehavior: Equatable, Sendable {
    case idle(IdleSubState)
    case patrol(targetId: UUID, targetPos: SIMD3<Float>)
    case hover(nodeId: UUID, timer: Float)
    case conjure(nodeId: UUID, phase: Float)
    case absorb(nodeId: UUID, phase: Float)
    case react(nodeId: UUID, ReactionKind)
    case chatting

    public enum IdleSubState: Equatable, Sendable { case awake, drowsy, sleeping }
    public enum ReactionKind: Equatable, Sendable { case headTurn, inspect, importanceBoost }
}

/// Joint angles for mascot skeleton animation.
public struct MascotJointAngles: Sendable {
    public var leftArmPitch: Float
    public var rightArmPitch: Float
    public var eyePulse: Float
    public var bottomThruster: Float
    public var bodyBob: Float

    public init(
        leftArmPitch: Float = 0,
        rightArmPitch: Float = 0,
        eyePulse: Float = 0,
        bottomThruster: Float = 0,
        bodyBob: Float = 0
    ) {
        self.leftArmPitch = leftArmPitch
        self.rightArmPitch = rightArmPitch
        self.eyePulse = eyePulse
        self.bottomThruster = bottomThruster
        self.bodyBob = bodyBob
    }
}

/// LOD tier for distance-based rendering.
public enum LODTier: Int, Comparable, Sendable {
    case near = 0    // < 500 units — full detail
    case mid = 1     // 500–2000 — reduced
    case far = 2     // 2000–5000 — point sprite
    case culled = 3  // > 5000 — hidden

    public static func < (lhs: LODTier, rhs: LODTier) -> Bool {
        lhs.rawValue < rhs.rawValue
    }

    public static func from(distance: Float) -> LODTier {
        switch distance {
        case ..<500: return .near
        case ..<2000: return .mid
        case ..<5000: return .far
        default: return .culled
        }
    }
}

/// Set of visible nodes/edges per frame, computed by LODSystem.
public struct VisibleSet: Sendable {
    public var nearNodes: [Int]
    public var midNodes: [Int]
    public var farNodes: [Int]
    public var visibleEdgeIndices: [Int]
    public var visibleLabelIndices: [Int]

    /// Total visible node count across all tiers.
    public var totalNodeCount: Int { nearNodes.count + midNodes.count + farNodes.count }

    public init(
        nearNodes: [Int] = [],
        midNodes: [Int] = [],
        farNodes: [Int] = [],
        visibleEdgeIndices: [Int] = [],
        visibleLabelIndices: [Int] = []
    ) {
        self.nearNodes = nearNodes
        self.midNodes = midNodes
        self.farNodes = farNodes
        self.visibleEdgeIndices = visibleEdgeIndices
        self.visibleLabelIndices = visibleLabelIndices
    }
}
