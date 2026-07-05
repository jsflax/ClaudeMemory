#if canImport(FoundationModels)
import Testing
import Foundation
import FoundationModels
import EngramKit
import Lattice
import MCP

// MARK: - Two-Stage Pipeline: Sampler → Assessment

/// Proves the full pipeline: TranscriptSampler surfaces novel excerpts from a transcript,
/// then the on-device LLM correctly classifies them as worth saving (or not).
///
/// The key question: if a transcript mixes rehashed known content with genuinely novel
/// insights in the SAME topic area, does the pipeline:
/// 1. Surface the novel excerpts (not bury them under rehashes)?
/// 2. Correctly assess the surfaced content as worth saving?

@available(macOS 26.0, *)
@Generable
struct PipelineAssessment {
    @Guide(description: "Whether this transcript excerpt contains a non-obvious insight, debugging discovery, architectural decision, or user preference worth saving to long-term memory")
    var worthSaving: Bool
}

@available(macOS 26.0, *)
private func assess(_ text: String, existingMemories: [String] = []) async throws -> (worthSaving: Bool, ms: Int64) {
    let memoryContext: String
    if existingMemories.isEmpty {
        memoryContext = "No existing memories provided."
    } else {
        memoryContext = """
        The following knowledge is ALREADY stored in long-term memory:
        \(existingMemories.enumerated().map { "  [\($0.offset + 1)] \($0.element)" }.joined(separator: "\n"))

        Do NOT mark content as worth saving if it only restates, paraphrases, or summarizes \
        the above existing memories. Only mark it worth saving if it contains NEW information \
        not already captured.
        """
    }

    let prompt = """
    Does this transcript excerpt contain knowledge worth saving to long-term memory?

    \(memoryContext)

    Answer true if the excerpt contains ANY of these that is NOT already in existing memories:
    - A debugging discovery where the root cause was not immediately obvious
    - An architectural or design decision where alternatives were compared
    - A workaround for a platform bug or limitation
    - A new pattern, gotcha, or non-obvious interaction between technologies
    - The user stating preferences or conventions

    Answer false if it is ONLY:
    - A restatement or paraphrase of existing memories or well-known documentation
    - Fixing typos, renaming variables, formatting code
    - Simple Q&A about versions, file counts, project structure
    - General explanations of common concepts everyone already knows

    Excerpt:
    \(text)
    """

    let session = LanguageModelSession()
    let start = ContinuousClock.now
    let response = try await session.respond(to: prompt, generating: PipelineAssessment.self)
    let elapsed = ContinuousClock.now - start
    let ms = elapsed.components.seconds * 1000 + elapsed.components.attoseconds / 1_000_000_000_000_000
    return (response.content.worthSaving, ms)
}

// MARK: - Pipeline Tests

@available(macOS 26.0, *)
@Test func pipeline_novelInsightSurfacedAndAssessedWorthSaving() async throws {
    guard SystemLanguageModel.default.availability == .available else { return }

    let ctx = try await makeSamplerContext()

    // Seed: known facts about Swift actor reentrancy
    let knownFacts = [
        "Swift actors are reentrant by default — when an actor method hits an await, other calls can execute before the first resumes",
        "Actor reentrancy means mutable state can change across await points, so you must re-check invariants after every await",
        "The nonisolated keyword lets you opt methods out of actor isolation when they only read immutable state",
    ]
    for fact in knownFacts {
        _ = try await ctx.tools.handle(CallTool.Parameters(
            name: "remember",
            arguments: ["content": .string(fact), "force": .bool(true)]
        ))
    }

    // Transcript: rehash + novel debugging discovery in same topic
    let jsonl = makeFakeTranscript([
        ("user", "why is my actor method running twice?"),
        ("assistant", padTo150("Swift actors are reentrant — when a method suspends at an await point, other messages can interleave. This means you need to re-validate state after each suspension point because another call may have mutated it.")),
        ("user", "no I get that, but the Combine sink is running outside the actor"),
        ("assistant", padTo150("Found the bug. When a Swift actor subscribes to a Combine publisher via sink, the sink closure executes on the publisher's scheduler, NOT on the actor's executor. This completely bypasses actor isolation — the closure can read and write actor state without serialization. The fix is to wrap the closure body in Task { await self.handle(value) } to re-enter the actor context. This is a common gotcha when mixing Combine with Swift concurrency.")),
        ("assistant", padTo150("The deeper issue is that Combine predates Swift concurrency. Publisher.sink returns an AnyCancellable whose closure has no concept of actor isolation. When you write self.data = newValue inside the sink, the compiler doesn't warn you because Combine's closure isn't checked for sendability in the same way async contexts are. Swift 6 strict concurrency checking catches some of these but not all sink patterns.")),
    ])
    let transcriptPath = writeTempFile(jsonl)
    defer { try? FileManager.default.removeItem(atPath: transcriptPath) }

    // Stage 1: Sample
    let excerpts = await TranscriptSampler.sample(
        transcriptPath: transcriptPath,
        lattice: ctx.lattice,
        embedder: ctx.embedder,
        maxExcerpts: 3,
        maxCharsPerExcerpt: 500,
        maxTotalChars: 2000
    )

    #expect(!excerpts.isEmpty, "Sampler should return excerpts")

    // The novel Combine insight should rank higher than the reentrancy rehash
    let rehash = excerpts.first { $0.text.contains("re-validate state") }
    let novel = excerpts.first { $0.text.contains("publisher's scheduler") || $0.text.contains("Combine") }

    print("=== Stage 1: Sampler ===")
    for (i, e) in excerpts.enumerated() {
        let label = e.text.prefix(60).replacingOccurrences(of: "\n", with: " ")
        print("  [\(i+1)] dist=\(String(format: "%.3f", e.minDistance)) \(label)...")
    }

    #expect(novel != nil, "Combine insight should be in sampled excerpts")
    if let rehash, let novel {
        #expect(novel.minDistance > rehash.minDistance,
                "Novel Combine insight (\(novel.minDistance)) should rank higher than rehash (\(rehash.minDistance))")
    }

    // Stage 2: Assess the top excerpt(s) with on-device LLM, providing existing memories
    let existingMemories = knownFacts
    let formatted = TranscriptSampler.formatForAssessment(excerpts)
    let (worthSaving, ms) = try await assess(formatted, existingMemories: existingMemories)

    print("=== Stage 2: Assessment ===")
    print("  worthSaving: \(worthSaving), latency: \(ms)ms")

    #expect(worthSaving, "Pipeline should assess the Combine+actor isolation discovery as worth saving")
}

@available(macOS 26.0, *)
@Test func pipeline_rehashOnlyTranscriptNotWorthSaving() async throws {
    guard SystemLanguageModel.default.availability == .available else { return }

    let ctx = try await makeSamplerContext()

    // Seed: known facts
    let knownFacts = [
        "Swift actors are reentrant by default — when an actor method hits an await, other calls can execute before the first resumes",
        "Actor reentrancy means mutable state can change across await points, so you must re-check invariants after every await",
    ]
    for fact in knownFacts {
        _ = try await ctx.tools.handle(CallTool.Parameters(
            name: "remember",
            arguments: ["content": .string(fact), "force": .bool(true)]
        ))
    }

    // Transcript: ONLY paraphrases of known content, no new insights
    let jsonl = makeFakeTranscript([
        ("user", "remind me how actor reentrancy works in Swift"),
        ("assistant", padTo150("In Swift, actors are reentrant by default. When an actor method suspends at an await point, other calls to that actor can run before the first call resumes. Mutable state may change between suspension points.")),
        ("assistant", padTo150("Because of reentrancy, mutable state can change across await points in an actor. You must always re-check your invariants after every await, since another message may have been processed while you were suspended.")),
    ])
    let transcriptPath = writeTempFile(jsonl)
    defer { try? FileManager.default.removeItem(atPath: transcriptPath) }

    // Stage 1: Sample
    let excerpts = await TranscriptSampler.sample(
        transcriptPath: transcriptPath,
        lattice: ctx.lattice,
        embedder: ctx.embedder,
        maxExcerpts: 3,
        maxCharsPerExcerpt: 500,
        maxTotalChars: 2000
    )

    print("=== Stage 1: Sampler (rehash-only) ===")
    for (i, e) in excerpts.enumerated() {
        let label = e.text.prefix(60).replacingOccurrences(of: "\n", with: " ")
        print("  [\(i+1)] dist=\(String(format: "%.3f", e.minDistance)) \(label)...")
    }

    // The novelty threshold (0.28) should filter out all excerpts —
    // pure paraphrases of known content should not reach Stage 2.
    // This IS the pipeline working correctly: Stage 1 (embedding distance)
    // handles dedup, Stage 2 (LLM) only sees genuinely novel content.
    print("  Excerpt count after novelty filter: \(excerpts.count)")

    if excerpts.isEmpty {
        print("=== Stage 1 correctly filtered all rehash content ===")
        // Success — the sampler's novelty gate prevented rehash from reaching assessment
    } else {
        // If any excerpts survived the novelty filter, they should still be close
        for excerpt in excerpts {
            print("  Survived filter — distance: \(excerpt.minDistance), text: \(excerpt.text.prefix(60))...")
        }
        // Even if something slips through the novelty gate, the LLM should reject it
        let existingMemories = knownFacts
        let formatted = TranscriptSampler.formatForAssessment(excerpts)
        let (worthSaving, ms) = try await assess(formatted, existingMemories: existingMemories)
        print("=== Stage 2: Assessment (rehash-only) ===")
        print("  worthSaving: \(worthSaving), latency: \(ms)ms")
        #expect(!worthSaving, "If rehash content passes novelty gate, LLM should still reject it")
    }
}

@available(macOS 26.0, *)
@Test func pipeline_mixedTranscriptSurfacesOnlyNovelContent() async throws {
    guard SystemLanguageModel.default.availability == .available else { return }

    let ctx = try await makeSamplerContext()

    // Seed: extensive knowledge about Elasticsearch
    let knownFacts = [
        "Elasticsearch client should be a module-level singleton to avoid TLS handshake overhead on every request",
        "Creating a new ES client per request costs 2-3 seconds due to TLS handshake, connection pool init, and node discovery",
        "Elasticsearch index mappings should be defined upfront for production use to avoid dynamic mapping surprises",
    ]
    for fact in knownFacts {
        _ = try await ctx.tools.handle(CallTool.Parameters(
            name: "remember",
            arguments: ["content": .string(fact), "force": .bool(true)]
        ))
    }

    // Transcript: ES rehash + novel discovery about scroll API pitfall
    let jsonl = makeFakeTranscript([
        ("user", "search is slow again"),
        ("assistant", padTo150("Checking the search endpoint. The Elasticsearch client should be a singleton — creating a new client per request causes a 2-3 second TLS handshake overhead. Let me verify it's configured correctly.")),
        ("user", "no the client is fine, it's the pagination that's slow for large result sets"),
        ("assistant", padTo150("Found it. The search is using from+size pagination which gets exponentially slower past 10,000 results because Elasticsearch must fetch and discard all preceding documents. For deep pagination, the scroll API is better — but there's a critical gotcha: each scroll context holds an in-memory segment snapshot. If you open 100 concurrent scroll contexts with a 5-minute timeout, you're pinning 100 segment snapshots in memory, preventing segment merges. This caused our OOM last month. The fix is to use search_after with a point-in-time (PIT) instead — it's stateless on the server side and handles concurrent deep pagination without the memory pressure.")),
        ("assistant", padTo150("Switched from scroll to search_after+PIT. Response time for page 500 went from 12 seconds to 180ms. The key insight is that scroll was designed for bulk data export, not user-facing pagination — the API name is misleading.")),
    ])
    let transcriptPath = writeTempFile(jsonl)
    defer { try? FileManager.default.removeItem(atPath: transcriptPath) }

    // Stage 1: Sample — novel content should rank highest
    let excerpts = await TranscriptSampler.sample(
        transcriptPath: transcriptPath,
        lattice: ctx.lattice,
        embedder: ctx.embedder,
        maxExcerpts: 3,
        maxCharsPerExcerpt: 500,
        maxTotalChars: 2000
    )

    print("=== Stage 1: Sampler (mixed ES transcript) ===")
    for (i, e) in excerpts.enumerated() {
        let label = e.text.prefix(60).replacingOccurrences(of: "\n", with: " ")
        print("  [\(i+1)] dist=\(String(format: "%.3f", e.minDistance)) \(label)...")
    }

    // The scroll/PIT excerpts should rank above the singleton rehash
    let rehash = excerpts.first { $0.text.contains("singleton") || $0.text.contains("TLS handshake") }
    let novelScroll = excerpts.first { $0.text.contains("scroll") || $0.text.contains("search_after") || $0.text.contains("PIT") }

    #expect(novelScroll != nil, "Scroll/PIT discovery should be in excerpts")
    if let rehash, let novelScroll {
        #expect(novelScroll.minDistance > rehash.minDistance,
                "Scroll/PIT insight (\(novelScroll.minDistance)) should rank above singleton rehash (\(rehash.minDistance))")
    }

    // Stage 2: Assess with existing memories context
    let existingMemories = knownFacts
    let formatted = TranscriptSampler.formatForAssessment(excerpts)
    let (worthSaving, ms) = try await assess(formatted, existingMemories: existingMemories)

    print("=== Stage 2: Assessment (mixed ES) ===")
    print("  worthSaving: \(worthSaving), latency: \(ms)ms")

    #expect(worthSaving, "Pipeline should assess scroll→search_after discovery as worth saving")
}

// MARK: - Helpers (reuse from TranscriptSamplerTests)

private func makeFakeTranscript(_ entries: [(role: String, text: String?)]) -> String {
    entries.map { role, text in
        if role == "user" {
            return makeJsonlLine(type: "user", message: ["role": "user", "content": text ?? ""])
        } else {
            if let text {
                return makeJsonlLine(type: "assistant", content: [["type": "text", "text": text]])
            } else {
                return makeJsonlLine(type: "assistant", content: [
                    ["type": "tool_use", "id": "toolu_\(UUID().uuidString.prefix(8))", "name": "Read", "input": ["file_path": "/tmp/foo"]],
                ])
            }
        }
    }.joined(separator: "\n")
}

private func makeJsonlLine(type: String, content: [[String: Any]]) -> String {
    let obj: [String: Any] = [
        "type": type,
        "message": ["role": type, "content": content] as [String: Any],
        "uuid": UUID().uuidString,
        "timestamp": ISO8601DateFormatter().string(from: Date()),
    ]
    guard let data = try? JSONSerialization.data(withJSONObject: obj),
          let str = String(data: data, encoding: .utf8) else { return "{}" }
    return str
}

private func makeJsonlLine(type: String, message: [String: Any]) -> String {
    let obj: [String: Any] = [
        "type": type,
        "message": message,
        "uuid": UUID().uuidString,
        "timestamp": ISO8601DateFormatter().string(from: Date()),
    ]
    guard let data = try? JSONSerialization.data(withJSONObject: obj),
          let str = String(data: data, encoding: .utf8) else { return "{}" }
    return str
}

private func writeTempFile(_ content: String) -> String {
    let path = FileManager.default.temporaryDirectory
        .appending(path: "test-transcript-\(UUID().uuidString).jsonl").path
    FileManager.default.createFile(atPath: path, contents: Data(content.utf8))
    return path
}

private func padTo150(_ s: String) -> String {
    if s.count >= 150 { return s }
    return s + String(repeating: " ", count: 150 - s.count) + "."
}

#endif
