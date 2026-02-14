import Lattice
import Foundation

/// A session episode that groups related memories into a narrative sequence.
///
/// Episodes bracket a work session or conversation, allowing memories to be
/// recalled as a cohesive story rather than isolated facts. They support both
/// explicit bracketing (begin/end) and automatic creation based on time gaps.
///
/// - **Status**: `"active"` (in progress) or `"ended"` (closed).
/// - **Project scoping**: Same as memories — use the project name to organize episodes.
/// - **Summary**: Client-provided at end_episode, or empty.
@Model
public final class Episode {
    /// Short title describing the episode.
    var title: String

    /// Client-provided summary of what happened in this episode.
    var summary: String

    /// Which project this episode belongs to. Defaults to "global".
    var project: String

    /// Episode status: "active" or "ended".
    var status: String

    /// When this episode started.
    var startedAt: Date

    /// When this episode ended. `.distantFuture` while active.
    var endedAt: Date

    init(
        title: String,
        summary: String = "",
        project: String = "global",
        status: String = "active",
        startedAt: Date = Date(),
        endedAt: Date = .distantFuture
    ) {
        self.title = title
        self.summary = summary
        self.project = project
        self.status = status
        self.startedAt = startedAt
        self.endedAt = endedAt
    }
}
