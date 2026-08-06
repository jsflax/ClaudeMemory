import ArgumentParser
import EngramMemoryCore
import Foundation
#if canImport(EngramKit)
import EngramKit
import Lattice
#endif

/// UserPromptSubmit hook: recalls relevant memories, nudges learning and maintenance.
struct Advise: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Recall relevant memories, nudge learning and maintenance (UserPromptSubmit hook)"
    )

    func run() async throws {
        // Guard against recursion: maintenance subprocess sets this env var
        if ProcessInfo.processInfo.environment["CLAUDE_MEMORY_MAINTENANCE"] != nil {
            return
        }

        let inputData = readStdin()
        guard !inputData.isEmpty else { return }

        let input: UserPromptSubmitInput
        do {
            input = try JSONDecoder().decode(UserPromptSubmitInput.self, from: inputData)
        } catch {
            hookLog("Failed to parse hook input: \(error)")
            return
        }

        guard let prompt = input.prompt, !prompt.isEmpty else {
            hookLog("Advise: empty or nil prompt, skipping")
            return
        }

        let project = projectName(from: input.cwd)
        let proj = project ?? "unknown"
        let sid = input.sessionId
        sessionLog("Advise: START sessionId=\(sid ?? "nil"), project=\(proj)", sessionId: sid)
        hookLog("Advise: project=\(proj), prompt=\(String(prompt.prefix(80)))...")

        var sections: [String] = []

        if let remote = RemoteConfig.active {
            // Remote backend: the server does distillation, ranking,
            // fencing, and budgeting; gate-less by design (it runs once per
            // agent event). Returns the full "## Relevant memories" block.
            if let block = await RemoteMemory.advise(
                remote, prompt: prompt, project: remote.project ?? project) {
                sessionLog("Advise(remote): block \(block.count) chars", sessionId: sid)
                logRecalledMemories(block, hook: "Advise", sessionId: sid)
                sections.append(block)
            } else {
                sessionLog("Advise(remote): no block", sessionId: sid)
                if let sid, !sid.isEmpty { writeSessionRecallSkip(sessionId: sid) }
            }
            appendNudgesAndEmit(input: input, project: proj, sections: &sections)
            return
        }

        #if canImport(EngramKit)
        // Gate: classify whether this prompt is worth recalling memories for.
        sessionLog("Advise: running recall gate", sessionId: sid)
        let gate = try? RecallGateClassifier()
        let shouldRecall = gate?.shouldRecall(query: prompt) ?? true
        sessionLog("Advise: recall gate = \(shouldRecall ? "recall" : "skip")", sessionId: sid)
        hookLog("Advise: recall gate = \(shouldRecall ? "recall" : "skip")")

        // Recall relevant memories (only if gate says the prompt is topical)
        if shouldRecall {
            sessionLog("Advise: calling initMemoryTools", sessionId: sid)
            currentSessionId = sid
            if let tools = await initMemoryTools(sessionId: sid) {
                // Extract content words to focus the embedding on key concepts.
                // Raw prose averages all concepts into a poor vector neighborhood.
                let contentWords = MemoryTools.extractContentWords(from: prompt)
                let recallQuery = contentWords.isEmpty ? prompt : contentWords.joined(separator: " ")
                sessionLog("Advise: calling directRecall (query: \(String(recallQuery.prefix(60)))...)", sessionId: sid)
                hookLog("Advise: running directRecall (query: \(String(recallQuery.prefix(80)))...)")
                do {
                    if let result = try await tools.directRecall(
                        query: recallQuery,
                        project: project,
                        depth: 1,
                        limit: 5
                    ) {
                        sessionLog("Advise: directRecall returned \(result.count) chars", sessionId: sid)
                        logRecalledMemories(result, hook: "Advise", sessionId: sid)
                        sessionLog("Advise: logRecalledMemories done, recall log written", sessionId: sid)
                        sections.append("## Relevant memories\n\n\(result)")
                    } else {
                        sessionLog("Advise: directRecall returned nil", sessionId: sid)
                        hookLog("Advise: recall returned nil")
                    }
                } catch {
                    sessionLog("Advise: directRecall FAILED: \(error)", sessionId: sid)
                    hookLog("Advise: recall failed: \(error)")
                }
            } else {
                sessionLog("Advise: initMemoryTools returned nil", sessionId: sid)
                hookLog("Advise: failed to initialize memory tools")
            }
        } else {
            hookLog("Advise: skipped recall (conversational filler)")
            // Write skip indicator so statusline shows current state
            if let sid, !sid.isEmpty {
                writeSessionRecallSkip(sessionId: sid)
            }
        }

        // Spawn maintenance subprocess if threshold crossed (fire-and-forget).
        // Local-only: the server owns hygiene for the shared PG store.
        spawnMaintenanceIfNeeded(project: proj, cwd: input.cwd)
        #endif

        appendNudgesAndEmit(input: input, project: proj, sections: &sections)
    }

    /// Shared tail for both backends: the stop-hook handshake, the learning
    /// nudge, and the hook output envelope.
    private func appendNudgesAndEmit(input: UserPromptSubmitInput, project: String,
                                     sections: inout [String]) {
        // Learning nudge — skip if stop hook just fired (session-learner already spawned)
        let consumedStopNudge = updateSessionCounters(sessionId: input.sessionId) { state -> Bool in
            guard state.stopNudgeSent else { return false }
            state.stopNudgeSent = false
            return true
        }
        if consumedStopNudge != true {
            sections.append(learningNudge(project: project))
        }

        let output = HookOutput(
            hookSpecificOutput: HookSpecificOutput(
                hookEventName: "UserPromptSubmit",
                additionalContext: sections.joined(separator: "\n\n")
            )
        )
        try? writeOutput(output)
    }

    #if canImport(EngramKit)
    // MARK: - Maintenance Subprocess

    /// The maintenance system prompt, loaded from agents/memory-maintenance.md or inlined.
    private static let maintenanceSystemPrompt: String = loadAgentSystemPrompt(
        name: "memory-maintenance", bundle: agentPromptBundle,
        fallback: """
        You are a memory maintenance agent. Your job is to analyze the memory database and improve its organization for better recall quality.

        ## Workflow
        1. Run stats() and list_topics() to understand memory distribution.
        2. Run find_clusters() to discover redundant memories.
        3. Consolidate genuinely redundant memories. Keep distinct ones separate.
        4. Use organize() to create subtopics for large topic groups.
        5. Connect new memories to related existing ones.
        6. Verify improvements with stats() and list_topics().

        ## Guidelines
        - Read full memory content before deciding what to do.
        - Only consolidate memories that are truly saying the same thing.
        - Be efficient — don't over-process clean databases.
        """
    )

    private static let maintenanceAllowedTools = "mcp__memory__*,Read,Grep,Glob,Bash"

    private static let maintenanceLogPath = NSHomeDirectory() + "/.claude/memory-maintenance.log"

    /// Check if maintenance threshold is crossed and spawn the subprocess if so.
    /// Uses AuditLog row count since last run — immune to concurrent-process counter clobbering.
    private func spawnMaintenanceIfNeeded(project: String, cwd: String?) {
        guard let lattice = openLattice() else { return }

        let lastRunTimestamp = Double(
            getHookState(key: .maintenanceLastRunTimestamp) ?? "0"
        ) ?? 0
        let since = Date(timeIntervalSince1970: lastRunTimestamp)

        // Don't spawn if maintenance is already running — but with a
        // staleness escape: the flag is cleared by the maintenance agent on
        // completion, so a SIGKILL mid-run used to leave it stuck at "1" and
        // disable maintenance forever. lastRunTimestamp is set alongside the
        // flag, so an "active" older than an hour is a corpse, not a run.
        if getHookState(key: .maintenanceActive) == "1" {
            let flagAge = Date().timeIntervalSince(since)
            if flagAge < 3600 { return }
            setHookState(key: .maintenanceActive, value: "0")
        }

        // Enforce minimum cooldown to prevent maintenance's own writes from re-triggering
        guard Date().timeIntervalSince(since) >= maintenanceCooldownSeconds else { return }

        let delta = lattice.objects(AuditLog.self)
            .where { $0.tableName == "Memory" && $0.timestamp > since }
            .count

        guard delta >= maintenanceThreshold else { return }

        // Update baseline immediately to prevent re-spawning on next advise call
        setHookState(key: .maintenanceLastRunTimestamp, value: String(Date().timeIntervalSince1970))

        // Signal the visualizer that maintenance is active
        setHookState(key: .maintenanceActive, value: "1")

        let prompt = """
        Perform maintenance on the memory database. Focus on project "\(project)".

        Current state: \(delta) Memory writes since last maintenance.

        Follow your system prompt workflow: assess, find redundancy, find communities, take action, connect the graph, verify.
        """

        let hooksBin = NSHomeDirectory() + "/.claude/bin/memory-hooks"

        do {
            try spawnClaudeSubprocess(
                prompt: prompt,
                systemPrompt: Self.maintenanceSystemPrompt,
                allowedTools: Self.maintenanceAllowedTools,
                model: "sonnet",
                envGuard: (key: "CLAUDE_MEMORY_MAINTENANCE", value: "1"),
                logPath: Self.maintenanceLogPath,
                cwd: cwd,
                postCommand: "'\(hooksBin)' clear-maintenance"
            )
            hookLog("Advise: spawned memory-maintenance subprocess (ops delta: \(delta))")
        } catch {
            hookLog("Advise: failed to spawn memory-maintenance: \(error)")
        }
    }
    #endif
}