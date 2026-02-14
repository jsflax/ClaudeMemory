import Lattice
import MCP
import Foundation

// MARK: - Episodic Memory

extension MemoryTools {

    // MARK: - begin_episode

    func handleBeginEpisode(_ args: [String: Value]?) throws -> CallTool.Result {
        let a = try args.decode(BeginEpisodeArgs.self)
        let project = a.project ?? "global"
        let title = a.title ?? "Session: \(Self.isoDateFormatter.string(from: Date()))"

        // End any active auto-episode first
        endActiveEpisode()

        let episode = Episode(title: title, project: project)
        lattice.add(episode)
        activeEpisodeId = episode.primaryKey!
        isExplicitEpisode = true
        lastMemoryTime = Date()

        log("Created episode [episode:\(episode.primaryKey!)] \(title)")
        return CallTool.Result(
            content: [.text("Created episode (episode:\(episode.primaryKey!), project: \(project)): \(title)")],
            isError: false
        )
    }

    // MARK: - end_episode

    func handleEndEpisode(_ args: [String: Value]?) throws -> CallTool.Result {
        let a = try args.decode(EndEpisodeArgs.self)

        let episode: Episode
        if let id = a.episodeId?.value {
            let id64 = Int64(id)
            guard let found = lattice.objects(Episode.self).where({ $0.primaryKey == id64 }).first else {
                return CallTool.Result(content: [.text("Episode with id \(id) not found.")], isError: true)
            }
            episode = found
        } else {
            guard let activeId = activeEpisodeId,
                  let found = lattice.objects(Episode.self).where({ $0.primaryKey == activeId }).first else {
                throw MCPError.invalidParams("No active episode. Provide 'episode_id' or start one with begin_episode.")
            }
            episode = found
        }

        // Guard against ending an already-ended episode
        if episode.status == "ended" {
            let memoryCount = lattice.count(Memory.self, where: { $0.episodeId == episode.primaryKey! })
            return CallTool.Result(
                content: [.text("Episode \(episode.primaryKey!) (\(episode.title)) is already ended. Duration: \(formatDuration(from: episode.startedAt, to: episode.endedAt)), memories: \(memoryCount)")],
                isError: false
            )
        }

        episode.status = "ended"
        episode.endedAt = Date()
        if let summary = a.summary, !summary.isEmpty {
            episode.summary = summary
        }

        // Clear activeEpisodeId if it matches
        if activeEpisodeId == episode.primaryKey {
            activeEpisodeId = nil
            isExplicitEpisode = false
        }

        let memoryCount = lattice.count(Memory.self, where: { $0.episodeId == episode.primaryKey! })
        let duration = formatDuration(from: episode.startedAt, to: episode.endedAt)

        log("Ended episode [episode:\(episode.primaryKey!)] \(episode.title)")
        return CallTool.Result(
            content: [.text("Ended episode (episode:\(episode.primaryKey!), \(episode.title)). Duration: \(duration), memories: \(memoryCount)\(episode.summary.isEmpty ? "" : ", summary: \(episode.summary)")")],
            isError: false
        )
    }

    // MARK: - recall_episode

    func handleRecallEpisode(_ args: [String: Value]?) throws -> CallTool.Result {
        let a = try args.decode(RecallEpisodeArgs.self)
        let id64 = Int64(a.episodeId.value)
        let limit = a.limit?.value ?? 200

        guard let episode = lattice.objects(Episode.self).where({ $0.primaryKey == id64 }).first else {
            return CallTool.Result(content: [.text("Episode with id \(a.episodeId.value) not found.")], isError: true)
        }

        // Fetch memories in this episode, sorted chronologically, with limit
        let totalCount = lattice.count(Memory.self, where: { $0.episodeId == id64 })
        let memories = lattice.objects(Memory.self)
            .where { $0.episodeId == id64 }
            .sortedBy(.init(\.createdAt, order: .forward))
            .snapshot(limit: Int64(limit))

        let duration: String
        if episode.status == "active" {
            duration = formatDuration(from: episode.startedAt, to: Date())
        } else {
            duration = formatDuration(from: episode.startedAt, to: episode.endedAt)
        }

        var output = "## Episode: \(episode.title)\n"
        output += "**ID**: \(episode.primaryKey!) | **Project**: \(episode.project) | **Status**: \(episode.status) | **Duration**: \(duration)"
        output += "\n**Started**: \(Self.dateFormatter.string(from: episode.startedAt))"
        if episode.status == "ended" {
            output += " | **Ended**: \(Self.dateFormatter.string(from: episode.endedAt))"
        }

        if !episode.summary.isEmpty {
            output += "\n\n### Summary\n\(episode.summary)"
        }

        if memories.isEmpty {
            output += "\n\nNo memories in this episode."
        } else {
            let truncated = totalCount > memories.count
            let countLabel = truncated ? "\(memories.count) of \(totalCount)" : "\(memories.count)"
            output += "\n\n### Memories (\(countLabel))"

            let timeFormatter = DateFormatter()
            timeFormatter.dateFormat = "HH:mm:ss"

            for mem in memories {
                let time = timeFormatter.string(from: mem.createdAt)
                output += "\n[id:\(mem.primaryKey!)] [\(time)] \(mem.content)"
            }

            if truncated {
                output += "\n\n(Showing \(memories.count) of \(totalCount) memories. Use limit parameter to see more.)"
            }
        }

        return CallTool.Result(content: [.text(output)], isError: false)
    }

    // MARK: - list_episodes

    func handleListEpisodes(_ args: [String: Value]?) throws -> CallTool.Result {
        let a = try args.decode(ListEpisodesArgs.self)
        let limit = a.limit?.value ?? 20

        if let status = a.status {
            guard validEpisodeStatuses.contains(status) else {
                throw MCPError.invalidParams("Invalid status '\(status)'. Must be one of: active, ended")
            }
        }

        var results = lattice.objects(Episode.self)

        if let project = a.project {
            results = results.where { $0.project == project }
        }
        if let status = a.status {
            results = results.where { $0.status == status }
        }

        let sorted = results.sortedBy(.init(\.startedAt, order: .reverse))
        let limited = sorted.snapshot(limit: Int64(limit))

        if limited.isEmpty {
            return CallTool.Result(content: [.text("No episodes found.")], isError: false)
        }

        let lines = limited.map { ep in
            let memCount = lattice.count(Memory.self, where: { $0.episodeId == ep.primaryKey! })
            let startDate = Self.dateFormatter.string(from: ep.startedAt)
            let endDate = ep.status == "ended" ? Self.dateFormatter.string(from: ep.endedAt) : "ongoing"
            return "[episode:\(ep.primaryKey!)] [\(ep.status)] \(ep.title) (\(ep.project), \(startDate) – \(endDate), \(memCount) memories)"
        }

        return CallTool.Result(content: [.text(lines.joined(separator: "\n"))], isError: false)
    }

    // MARK: - Episode Helpers

    /// Create an auto-episode for the current session.
    func createAutoEpisode(project: String) -> Int64 {
        let title = "Session: \(Self.isoDateFormatter.string(from: Date()))"
        let episode = Episode(title: title, project: project)
        lattice.add(episode)
        activeEpisodeId = episode.primaryKey!
        isExplicitEpisode = false
        log("Auto-created episode [episode:\(episode.primaryKey!)] \(title)")
        return episode.primaryKey!
    }

    /// End the currently active episode (if any).
    func endActiveEpisode() {
        guard let activeId = activeEpisodeId else { return }
        if let episode = lattice.objects(Episode.self).where({ $0.primaryKey == activeId }).first {
            episode.status = "ended"
            episode.endedAt = Date()
            log("Auto-ended episode [episode:\(activeId)]")
        }
        activeEpisodeId = nil
        isExplicitEpisode = false
    }

    /// Format a duration between two dates as a human-readable string.
    func formatDuration(from start: Date, to end: Date) -> String {
        let seconds = Int(end.timeIntervalSince(start))
        if seconds < 60 { return "\(seconds)s" }
        let minutes = seconds / 60
        if minutes < 60 { return "\(minutes)m" }
        let hours = minutes / 60
        let remainingMinutes = minutes % 60
        return "\(hours)h \(remainingMinutes)m"
    }
}
