import Lattice
import Foundation

/// Global key-value state for hooks (maintenance tracking).
@Model
public final class HookState {
    @LatticeEnum
    public enum Key: String, Codable, Sendable {
        case maintenanceLastRunTimestamp = "maintenance.lastRunTimestamp"
        case maintenanceActive = "maintenance.active"
        /// Per-device opt-out: when "false", the advise hook excludes
        /// teammates' group-shared memories from context injection.
        /// Absent/any-other-value = included (beta default is ON).
        case adviseIncludeGroupMemories = "advise.includeGroupMemories"
    }

    /// The state key.
    public var key: Key = .maintenanceLastRunTimestamp

    /// The state value (stored as string, parsed by consumer).
    public var value: String

    /// When this state was last updated.
    public var updatedAt: Date

    public init(key: Key, value: String, updatedAt: Date = Date()) {
        self.key = key
        self.value = value
        self.updatedAt = updatedAt
    }
}

/// Per-session state for hooks. One row per Claude Code session.
@Model
public final class SessionState {
    /// Claude Code session ID.
    @Unique()
    public var sessionId: String

    /// Whether the stop hook has already fired for this session.
    public var stopNudgeSent: Bool = false

    /// Number of tool calls in this session (for throttled learning nudge).
    public var toolCallCount: Int = 0

    /// Tool call count at which the last learning nudge was sent.
    public var learningNudgeLastToolCount: Int = 0

    /// When this state was last updated.
    public var updatedAt: Date

    public init(sessionId: String, updatedAt: Date = Date()) {
        self.sessionId = sessionId
        self.updatedAt = updatedAt
    }
}
