import Lattice
import Foundation

/// A work-in-progress task checkpoint for session continuity.
///
/// Checkpoints let Claude save task state between sessions and resume
/// where it left off. Unlike memories (which store knowledge), checkpoints
/// track active work: what's planned, what's done, and what context is needed.
///
/// - **Status**: `"active"` (in progress), `"paused"` (shelved), or `"completed"` (done).
/// - **Project scoping**: Same as memories — use the project name to organize tasks.
/// - **Plan/Progress/Context**: Free-form fields for structured task state.
@Model
public final class Checkpoint {
    /// Short title describing the task.
    public var title: String

    /// Task status: "active", "paused", or "completed".
    public var status: String

    /// Which project this task belongs to. Defaults to "global".
    public var project: String

    /// The planned approach or steps for the task.
    public var plan: String

    /// What has been accomplished so far.
    public var progress: String

    /// Free-form context: file paths, decisions, blockers, notes, etc.
    public var context: String

    /// When this task was first created.
    public var createdAt: Date

    /// When this task was last checkpointed (created or updated).
    public var checkpointedAt: Date

    public init(
        title: String,
        status: String = "active",
        project: String = "global",
        plan: String = "",
        progress: String = "",
        context: String = "",
        createdAt: Date = Date(),
        checkpointedAt: Date = Date()
    ) {
        self.title = title
        self.status = status
        self.project = project
        self.plan = plan
        self.progress = progress
        self.context = context
        self.createdAt = createdAt
        self.checkpointedAt = checkpointedAt
    }
}
