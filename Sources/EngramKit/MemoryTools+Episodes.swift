import Lattice
import MCP
import Foundation

// MARK: - Episodic Memory
//
// Episodes are hub memories (topic: "episode") with part_of edges linking
// member memories. This reuses the existing memory + knowledge graph
// infrastructure instead of a separate Episode model.

extension MemoryTools {

    // MARK: - begin_episode

    func handleBeginEpisode(_ args: [String: Value]?) async throws -> CallTool.Result {
        let a = try args.decode(BeginEpisodeArgs.self)
        let project = a.project ?? "global"
        let title = a.title ?? "Session: \(Self.isoDateFormatter.string(from: Date()))"

        // End any active episode first
        endActiveEpisode()

        let floats = try await embedder.embed(text: title)
        let embeddingVec = floats.map { Vector<Float>($0) } ?? Vector<Float>([])
        let episodeTargetDB = writeLattice(for: project)
        let episode = Memory(content: title, topic: "episode", project: project, embedding: embeddingVec)
        episodeTargetDB.add(episode)

        guard let episodeId = episode.primaryKey else {
            throw MCPError.internalError("Failed to persist episode memory — primaryKey is nil after add()")
        }
        activeEpisodeId = episodeId
        lastMemoryTime = Date()

        log("Created episode [id:\(episodeId)] \(title)")
        return CallTool.Result(
            content: [.text("Created episode (id:\(episodeId), project: \(project)): \(title)")],
            isError: false
        )
    }

    // MARK: - end_episode

    func handleEndEpisode(_ args: [String: Value]?) throws -> CallTool.Result {
        let a = try args.decode(EndEpisodeArgs.self)

        let episode: Memory
        if let id = a.episodeId?.value {
            let id64 = Int64(id)
            guard let found = lattice.objects(Memory.self).where({ $0.primaryKey == id64 && $0.topic == "episode" }).first else {
                return CallTool.Result(content: [.text("Episode with id \(id) not found.")], isError: true)
            }
            episode = found
        } else {
            guard let activeId = activeEpisodeId,
                  let found = lattice.objects(Memory.self).where({ $0.primaryKey == activeId }).first else {
                throw MCPError.invalidParams("No active episode. Provide 'episode_id' or start one with begin_episode.")
            }
            episode = found
        }

        guard let epId = episode.primaryKey else {
            throw MCPError.internalError("Episode memory has no primaryKey")
        }

        // Append summary to episode content
        if let summary = a.summary, !summary.isEmpty {
            episode.content += "\n\nSummary: \(summary)"
        }

        // Clear activeEpisodeId if it matches
        if activeEpisodeId == epId {
            activeEpisodeId = nil
        }

        let memoryCount = countEpisodeMembers(epId)
        let duration = episodeDuration(epId, startedAt: episode.createdAt)

        log("Ended episode [id:\(epId)]")
        return CallTool.Result(
            content: [.text("Ended episode (id:\(epId)). Duration: \(duration), memories: \(memoryCount)\(a.summary.map { ", summary: \($0)" } ?? "")")],
            isError: false
        )
    }

    // MARK: - recall_episode

    func handleRecallEpisode(_ args: [String: Value]?) throws -> CallTool.Result {
        let a = try args.decode(RecallEpisodeArgs.self)
        let id64 = Int64(a.episodeId.value)
        let limit = a.limit?.value ?? 200

        guard let episode = lattice.objects(Memory.self).where({ $0.primaryKey == id64 && $0.topic == "episode" }).first else {
            return CallTool.Result(content: [.text("Episode with id \(a.episodeId.value) not found.")], isError: true)
        }

        // Find memories linked via part_of edges to this episode
        guard let episodeGlobalId = episode.__globalId else {
            return CallTool.Result(content: [.text("Episode has no globalId.")], isError: true)
        }
        let edges = lattice.objects(Edge.self)
            .where { $0.targetGlobalId == episodeGlobalId && $0.relation == .partOf }

        // Fetch and sort member memories chronologically
        var members: [Memory] = []
        for edge in edges {
            if let mem = lattice.objects(Memory.self).where({ $0.__globalId == edge.sourceGlobalId && $0.topic != "episode" }).first {
                members.append(mem)
            }
        }
        members.sort { $0.createdAt < $1.createdAt }

        let totalCount = members.count
        let limited = Array(members.prefix(limit))

        let isActive = activeEpisodeId == id64
        let duration = episodeDuration(id64, startedAt: episode.createdAt)

        // Parse title and summary from episode content
        let parts = episode.content.components(separatedBy: "\n\nSummary: ")
        let title = parts[0]
        let summary = parts.count > 1 ? parts[1] : nil

        var output = "## Episode: \(title)\n"
        output += "**ID**: \(id64) | **Project**: \(episode.project) | **Status**: \(isActive ? "active" : "ended") | **Duration**: \(duration)"
        output += "\n**Started**: \(Self.dateFormatter.string(from: episode.createdAt))"

        if let summary {
            output += "\n\n### Summary\n\(summary)"
        }

        if limited.isEmpty {
            output += "\n\nNo memories in this episode."
        } else {
            let truncated = totalCount > limited.count
            let countLabel = truncated ? "\(limited.count) of \(totalCount)" : "\(totalCount)"
            output += "\n\n### Memories (\(countLabel))"

            let timeFormatter = DateFormatter()
            timeFormatter.dateFormat = "HH:mm:ss"

            for mem in limited {
                let time = timeFormatter.string(from: mem.createdAt)
                let memId = mem.primaryKey.map(String.init) ?? "?"
                output += "\n[id:\(memId)] [\(time)] \(mem.content)"
            }

            if truncated {
                output += "\n\n(Showing \(limited.count) of \(totalCount) memories. Use limit parameter to see more.)"
            }
        }

        return CallTool.Result(content: [.text(output)], isError: false)
    }

    // MARK: - list_episodes

    func handleListEpisodes(_ args: [String: Value]?) throws -> CallTool.Result {
        let a = try args.decode(ListEpisodesArgs.self)
        let limit = a.limit?.value ?? 20

        var results = lattice.objects(Memory.self).where { $0.topic == "episode" }

        if let project = a.project {
            results = results.where { $0.project == project }
        }

        let sorted = results.sortedBy(.init(\.createdAt, order: .reverse))
        let limited = sorted.snapshot(limit: Int64(limit))

        if limited.isEmpty {
            return CallTool.Result(content: [.text("No episodes found.")], isError: false)
        }

        let lines = limited.compactMap { ep -> String? in
            guard let epId = ep.primaryKey else { return nil }
            let memCount = countEpisodeMembers(epId)
            let isActive = activeEpisodeId == epId
            let startDate = Self.dateFormatter.string(from: ep.createdAt)

            // Parse title from content (before any summary)
            let title = ep.content.components(separatedBy: "\n\nSummary: ")[0]

            return "[id:\(epId)] [\(isActive ? "active" : "ended")] \(title) (\(ep.project), \(startDate), \(memCount) memories)"
        }

        return CallTool.Result(content: [.text(lines.joined(separator: "\n"))], isError: false)
    }

    // MARK: - Episode Helpers

    /// Clear the active episode state.
    func endActiveEpisode() {
        guard activeEpisodeId != nil else { return }
        log("Cleared active episode [id:\(activeEpisodeId!)]")
        activeEpisodeId = nil
    }

    /// Count memories linked to an episode via part_of edges.
    private func countEpisodeMembers(_ episodeId: Int64) -> Int {
        guard let ep = lattice.objects(Memory.self).where({ $0.primaryKey == episodeId }).first,
              let epGlobalId = ep.__globalId else { return 0 }
        return lattice.count(Edge.self, where: { $0.targetGlobalId == epGlobalId && $0.relation == .partOf })
    }

    /// Compute episode duration from creation to latest member memory (or now if active).
    private func episodeDuration(_ episodeId: Int64, startedAt: Date) -> String {
        if activeEpisodeId == episodeId {
            return formatDuration(from: startedAt, to: Date())
        }
        // Find latest member memory
        guard let ep = lattice.objects(Memory.self).where({ $0.primaryKey == episodeId }).first,
              let epGlobalId = ep.__globalId else {
            return formatDuration(from: startedAt, to: startedAt)
        }
        let edges = lattice.objects(Edge.self).where { $0.targetGlobalId == epGlobalId && $0.relation == .partOf }
        var latest = startedAt
        for edge in edges {
            if let mem = lattice.objects(Memory.self).where({ $0.__globalId == edge.sourceGlobalId }).first {
                if mem.createdAt > latest { latest = mem.createdAt }
            }
        }
        return formatDuration(from: startedAt, to: latest)
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
