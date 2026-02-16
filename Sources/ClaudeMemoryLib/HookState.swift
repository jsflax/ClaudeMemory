import Lattice
import Foundation

/// Key-value state for hooks (maintenance tracking, session flags, etc.).
@Model
public final class HookState {
    @LatticeEnum
    public enum Key: String, Codable, Sendable {
        case crudOperationCount = "crud.operationCount"
        case maintenanceLastOpCount = "maintenance.lastOpCount"
        case toolCallCount = "session.toolCallCount"
        case learningNudgeLastToolCount = "learning.lastToolCount"
    }

    /// The state key.
    public var key: Key = .crudOperationCount

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
