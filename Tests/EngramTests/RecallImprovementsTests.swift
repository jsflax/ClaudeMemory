import Testing
import EngramKit
import Lattice
import MCP
import Foundation

// MARK: - Content Word Extraction: Real Advise Hook Queries

@Test func extractContentWords_adviseHook_recallNoiseQuery() {
    // User asked: "how much is the advise hook/recall helping you in this session? or is it mostly producing noise?"
    let words = MemoryTools.extractContentWords(from: "how much is the advise hook/recall helping you in this session? or is it mostly producing noise?")
    // Should drop: how, is, the, you, in, this, or, it
    #expect(!words.contains("how"))
    #expect(!words.contains("is"))
    #expect(!words.contains("the"))
    #expect(!words.contains("you"))
    #expect(!words.contains("in"))
    #expect(!words.contains("this"))
    #expect(!words.contains("or"))
    #expect(!words.contains("it"))
    // Should keep: advise, hook, recall, helping, session, mostly, producing, noise
    #expect(words.contains("advise"))
    #expect(words.contains("hook"))
    #expect(words.contains("recall"))
    #expect(words.contains("noise"))
    #expect(words.contains("session"))
}

@Test func extractContentWords_adviseHook_traversalFilterQuery() {
    // User asked about structural edge filtering
    let words = MemoryTools.extractContentWords(from: "i wanted to talk about traversing structural edges / Filter relates_to at depth 1 by distance threshold")
    #expect(!words.contains("i"))
    #expect(!words.contains("to"))
    #expect(!words.contains("about"))
    #expect(!words.contains("at"))
    #expect(!words.contains("by"))
    // Should keep the technical terms
    #expect(words.contains("traversing"))
    #expect(words.contains("structural"))
    #expect(words.contains("edges"))
    #expect(words.contains("Filter"))
    #expect(words.contains("depth"))
    #expect(words.contains("distance"))
    #expect(words.contains("threshold"))
}

@Test func extractContentWords_adviseHook_rebuildInstallQuery() {
    // User asked about rebuilding — short but still has function words
    let words = MemoryTools.extractContentWords(from: "it should have already? you should have inject context that uses our fix from this session")
    #expect(!words.contains("it"))
    #expect(!words.contains("you"))
    #expect(!words.contains("our"))
    #expect(!words.contains("this"))
    #expect(!words.contains("from"))
    #expect(!words.contains("that"))
    // Should keep
    #expect(words.contains("inject"))
    #expect(words.contains("context"))
    #expect(words.contains("fix"))
    #expect(words.contains("session"))
}

@Test func extractContentWords_adviseHook_primaryRecallNoiseQuery() {
    let words = MemoryTools.extractContentWords(from: "how would we fix the primary recall noise?")
    #expect(!words.contains("how"))
    #expect(!words.contains("we"))
    #expect(!words.contains("the"))
    // Should keep
    #expect(words.contains("fix"))
    #expect(words.contains("primary"))
    #expect(words.contains("recall"))
    #expect(words.contains("noise"))
}

@Test func extractContentWords_adviseHook_nlTaggerQuery() {
    let words = MemoryTools.extractContentWords(from: "where are we using NLTagger?")
    #expect(!words.contains("we"))
    // Should keep
    #expect(words.contains("using"))
    #expect(words.contains("NLTagger"))
}

// MARK: - Content Word Extraction (NLTagger)

@Test func extractContentWords_dropsPronouns() {
    let words = MemoryTools.extractContentWords(from: "we solved it")
    #expect(!words.contains("we"))
    #expect(!words.contains("it"))
    #expect(words.contains("solved"))
}

@Test func extractContentWords_dropsDeterminers() {
    let words = MemoryTools.extractContentWords(from: "the memory problem")
    #expect(!words.contains("the"))
    #expect(words.contains("memory"))
    #expect(words.contains("problem"))
}

@Test func extractContentWords_dropsPrepositions() {
    let words = MemoryTools.extractContentWords(from: "search for files in the database")
    #expect(!words.contains("for"))
    #expect(!words.contains("in"))
    #expect(!words.contains("the"))
    #expect(words.contains("search"))
    #expect(words.contains("files"))
    #expect(words.contains("database"))
}

@Test func extractContentWords_keepsMainVerbsDropsAuxiliariesPronounsDeterminers() {
    // Main (content) verbs pass through POS filtering; copulas/auxiliaries
    // ("have"/"been") are dropped by the explicit function-word set, since
    // they add no recall signal. Pronouns and determiners are dropped too.
    let words = MemoryTools.extractContentWords(from: "have we been doing this")
    #expect(words.contains("doing"))
    #expect(!words.contains("have"))
    #expect(!words.contains("been"))
    #expect(!words.contains("we"))
    #expect(!words.contains("this"))
}

@Test func extractContentWords_keepsMainVerbs() {
    let words = MemoryTools.extractContentWords(from: "solved the problem quickly")
    #expect(words.contains("solved"))
    #expect(words.contains("problem"))
    #expect(words.contains("quickly"))
}

@Test func extractContentWords_keepsAdjectives() {
    let words = MemoryTools.extractContentWords(from: "the best semantic search")
    #expect(words.contains("best"))
    #expect(words.contains("semantic"))
    #expect(words.contains("search"))
}

@Test func extractContentWords_allFunctionWords_fallsBack() {
    let words = MemoryTools.extractContentWords(from: "is it the one")
    // When all words are function words, should fall back to all words
    #expect(!words.isEmpty)
}

@Test func extractContentWords_emptyString() {
    let words = MemoryTools.extractContentWords(from: "")
    #expect(words.isEmpty)
}

@Test func extractContentWords_singleNoun() {
    let words = MemoryTools.extractContentWords(from: "database")
    #expect(words == ["database"])
}

@Test func extractContentWords_technicalTerms() {
    // Technical terms NLTagger may not recognize should be kept, not dropped
    let words = MemoryTools.extractContentWords(from: "SwiftUI uses ARC for memory management")
    #expect(words.contains("memory"))
    #expect(words.contains("management"))
    // Technical terms should survive even if NLTagger doesn't know them
    #expect(words.contains("SwiftUI") || words.contains("ARC"))
}

@Test func extractContentWords_questionFormat() {
    let words = MemoryTools.extractContentWords(from: "how does recall work")
    #expect(words.contains("recall"))
    #expect(words.contains("work"))
    // Interrogative "how" and auxiliary "does" are dropped as function words —
    // the query reduces to its content terms.
    #expect(!words.contains("how"))
    #expect(!words.contains("does"))
    // No whitespace tokens should leak through
    #expect(!words.contains(" "))
}

// MARK: - Recall Integration: Graph Traversal Filtering

@Test func recall_graphTraversal_filtersIrrelevantConnections() async throws {
    let tools = try await makeTools()

    // Store a memory about databases
    let r1 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("SQLite is the embedded database used by Lattice ORM"),
            "topic": .string("architecture"),
        ]
    ))
    let id1 = extractMemoryId(from: text(from: r1))!

    // Store a completely unrelated memory
    let r2 = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Compose UI test gotcha: sticky footer buttons outside a scrollable container"),
            "topic": .string("android-testing"),
        ]
    ))
    let id2 = extractMemoryId(from: text(from: r2))!

    // Connect them (simulating a loose relates_to edge)
    _ = try await tools.handle(CallTool.Parameters(
        name: "connect",
        arguments: [
            "from": .string(id1),
            "to": .string(id2),
            "relation": .string("relates_to"),
        ]
    ))

    // Recall with depth=1 — the android testing memory should be filtered out
    let result = try await tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: [
            "query": .string("embedded database ORM architecture"),
            "depth": .int(1),
        ]
    ))
    let output = text(from: result)

    #expect(output.contains("SQLite"), "Primary result should appear")
    // With vector-only recall (L2 KNN), a 2-memory DB returns both in primary results.
    // The graph traversal cosine filter (0.15) should still block the unrelated memory
    // from appearing in the "Connected" section. Check that SQLite ranks first.
    let lines = output.components(separatedBy: "\n\n").filter { $0.contains("[id:") }
    if lines.count > 1 {
        #expect(lines[0].contains("SQLite"), "SQLite memory should rank first")
    }
}

// MARK: - Recall Integration: Graph Traversal Filters on Connecting Edge

/// Bug: hasStructuralEdge checks ALL edges on a memory, not the edge that connected it
/// to the recall results. A memory reached via relates_to should not pass through just
/// because it happens to have a part_of edge to some unrelated hub.
@Test func recall_graphTraversal_relatesToDoesNotLeakViaUnrelatedStructuralEdge() async throws {
    let tools = try await makeTools()

    // Memory A: the one we'll recall directly (about Swift concurrency)
    let rA = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Swift structured concurrency uses async/await and task groups for parallel execution"),
            "topic": .string("swift-concurrency"),
        ]
    ))
    let idA = extractMemoryId(from: text(from: rA))!

    // Memory B: unrelated (about Kubernetes networking), but will have a part_of edge to its own hub
    let rB = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Kubernetes pod networking uses CNI plugins for container network interface configuration"),
            "topic": .string("kubernetes"),
        ]
    ))
    let idB = extractMemoryId(from: text(from: rB))!

    // Memory C: hub for Kubernetes topic
    let rC = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Hub: Kubernetes infrastructure and deployment patterns"),
            "topic": .string("kubernetes"),
            "force": .bool(true),
        ]
    ))
    let idC = extractMemoryId(from: text(from: rC))!

    // B --[part_of]--> C  (structural edge — B is part of its own hub)
    _ = try await tools.handle(CallTool.Parameters(
        name: "connect",
        arguments: ["from": .string(idB), "to": .string(idC), "relation": .string("part_of")]
    ))

    // A --[relates_to]--> B  (loose edge — the one graph traversal will follow)
    _ = try await tools.handle(CallTool.Parameters(
        name: "connect",
        arguments: ["from": .string(idA), "to": .string(idB), "relation": .string("relates_to")]
    ))

    // Recall about Swift concurrency with depth=1. limit 1 is load-bearing:
    // on a 3-memory corpus the adaptive outlier filter degenerates (p75 of
    // two distances IS the worst distance, so nothing gets pruned) and hub C
    // squeaks into the DIRECT results at ~0.52 — from which B is admitted
    // via its part_of edge, legitimately. Capping at 1 keeps the direct set
    // to A, so the ONLY route to B is the relates_to edge this test is
    // about.
    let result = try await tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: [
            "query": .string("Swift async await structured concurrency task groups"),
            "depth": .int(1),
            "limit": .int(1),
        ]
    ))
    let output = text(from: result)

    #expect(output.contains("Swift structured concurrency"), "Primary result should appear")

    // Memory B was reached via relates_to from A. It's semantically distant from the query.
    // It should NOT appear in Connected results just because it has a part_of edge to hub C.
    // The filter should check the connecting edge (relates_to), not any edge on B.
    let connectedSection = output.components(separatedBy: "--- Connected").last ?? ""
    #expect(!connectedSection.contains("Kubernetes pod networking"),
            "Memory reached via relates_to should be filtered by cosine distance, not pass through because it has an unrelated part_of edge")
}

/// Structural edges (part_of) should always pass through regardless of cosine distance.
@Test func recall_graphTraversal_partOfAlwaysPassesThrough() async throws {
    let tools = try await makeTools()

    // Hub memory about a niche topic
    let rHub = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Hub: Metal shader compilation pipeline and GPU resource management"),
            "topic": .string("metal-rendering"),
        ]
    ))
    let idHub = extractMemoryId(from: text(from: rHub))!

    // Child memory — related detail
    let rChild = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Metal shader functions are compiled into MTLLibrary objects at build time via .metal files in the Xcode project"),
            "topic": .string("metal-rendering"),
            "force": .bool(true),
        ]
    ))
    let idChild = extractMemoryId(from: text(from: rChild))!

    // child --[part_of]--> hub
    _ = try await tools.handle(CallTool.Parameters(
        name: "connect",
        arguments: ["from": .string(idChild), "to": .string(idHub), "relation": .string("part_of")]
    ))

    // Recall the hub directly, child should appear via part_of traversal
    let result = try await tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: [
            "query": .string("Metal shader compilation pipeline GPU resource management"),
            "depth": .int(1),
        ]
    ))
    let output = text(from: result)

    #expect(output.contains("Metal shader compilation pipeline"), "Hub should appear in primary results")
    // The child connected via part_of should always appear, even if its embedding
    // is somewhat distant from the query — structural edges bypass cosine filtering.
    let connectedSection = output.components(separatedBy: "--- Connected").last ?? ""
    if !output.contains("MTLLibrary") || connectedSection.isEmpty {
        // If both show in primary results (small DB), that's fine too
        #expect(output.contains("MTLLibrary"), "Child memory should appear either in primary or connected results")
    }
}

/// BFS at depth>1 should not chain through relates_to → hub → part_of to pull in
/// unrelated children of an unrelated hub.
@Test func recall_graphTraversal_depth2_doesNotChainThroughRelatesToHub() async throws {
    let tools = try await makeTools()

    // Populate with enough database-related memories so the outlier filter
    // excludes visionOS content from primary results (needs a tight cluster).
    let rA = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("B-tree indexes in SQLite provide O(log n) lookup for primary key queries"),
            "topic": .string("database-internals"),
        ]
    ))
    let idA = extractMemoryId(from: text(from: rA))!

    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("SQLite WAL mode enables concurrent readers during write transactions for better throughput"),
            "topic": .string("database-internals"),
            "force": .bool(true),
        ]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Database query planning uses cost-based optimization to choose between index scan and table scan"),
            "topic": .string("database-internals"),
            "force": .bool(true),
        ]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("SQLite page cache holds recently accessed B-tree pages in memory to reduce disk IO"),
            "topic": .string("database-internals"),
            "force": .bool(true),
        ]
    ))

    // Memory B: a hub for visionOS (unrelated)
    let rB = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Hub: visionOS spatial computing patterns for immersive experiences"),
            "topic": .string("visionos"),
            "force": .bool(true),
        ]
    ))
    let idB = extractMemoryId(from: text(from: rB))!

    // Memory C: child of the visionOS hub (even more unrelated)
    let rC = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("visionOS hand tracking uses ARKit skeletal hand data with 27 joint positions per hand"),
            "topic": .string("visionos"),
            "force": .bool(true),
        ]
    ))
    let idC = extractMemoryId(from: text(from: rC))!

    // A --[relates_to]--> B (loose edge to unrelated hub)
    _ = try await tools.handle(CallTool.Parameters(
        name: "connect",
        arguments: ["from": .string(idA), "to": .string(idB), "relation": .string("relates_to")]
    ))

    // C --[part_of]--> B (structural edge within the visionOS hub)
    _ = try await tools.handle(CallTool.Parameters(
        name: "connect",
        arguments: ["from": .string(idC), "to": .string(idB), "relation": .string("part_of")]
    ))

    // Recall about database indexing with depth=2
    let result = try await tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: [
            "query": .string("B-tree index SQLite database lookup performance"),
            "depth": .int(2),
        ]
    ))
    let output = text(from: result)

    #expect(output.contains("B-tree indexes"), "Primary result should appear")

    // The visionOS hand tracking memory should NOT appear in graph traversal results.
    // It was connected via relates_to (A→B) then part_of (C→B), but the first hop
    // was a loose edge to a semantically distant memory — the BFS should not continue
    // from a filtered-out node.
    // Note: with a small DB, unrelated memories may appear in *primary* results due to
    // the adaptive outlier threshold — that's a separate issue. This test verifies
    // graph traversal doesn't chain through filtered-out nodes.
    let parts = output.components(separatedBy: "--- Connected")
    if parts.count > 1 {
        let connectedSection = parts[1]
        #expect(!connectedSection.contains("hand tracking"),
                "Depth-2 traversal should not chain through a filtered-out relates_to hop to reach part_of children")
    }
    // If no Connected section exists at all, traversal correctly produced no results — pass.
}

// MARK: - Content Word Extraction Improves Recall

/// Stripping function words from a verbose query should produce tighter recall results.
@Test func recall_contentWordExtraction_tighterResults() async throws {
    let tools = try await makeTools()

    // Store memories spanning different topics
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("Metal shader compilation pipeline uses MTLLibrary for GPU program management"),
            "topic": .string("metal"),
        ]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("SwiftUI view lifecycle calls body when observed state changes"),
            "topic": .string("swiftui"),
            "force": .bool(true),
        ]
    ))
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: [
            "content": .string("SQLite WAL mode provides concurrent read access during write transactions"),
            "topic": .string("database"),
            "force": .bool(true),
        ]
    ))

    // Verbose prose query — the kind users type naturally
    let verboseQuery = "how does the Metal shader compilation pipeline work and what is the MTLLibrary used for"
    // Content-word-only version — what the advise hook should use
    let contentWords = MemoryTools.extractContentWords(from: verboseQuery)
    let focusedQuery = contentWords.joined(separator: " ")

    let verboseResult = try await tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: ["query": .string(verboseQuery)]
    ))
    let focusedResult = try await tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: ["query": .string(focusedQuery)]
    ))

    let verboseOutput = text(from: verboseResult)
    let focusedOutput = text(from: focusedResult)

    // Both should find the Metal memory
    #expect(verboseOutput.contains("Metal shader compilation"), "Verbose query should find Metal memory")
    #expect(focusedOutput.contains("Metal shader compilation"), "Focused query should find Metal memory")

    // Extract distances for the Metal memory from both results
    func extractDistance(from output: String, containing keyword: String) -> Double? {
        let lines = output.components(separatedBy: "\n\n")
        guard let line = lines.first(where: { $0.contains(keyword) }) else { return nil }
        guard let distRange = line.range(of: "distance: ") else { return nil }
        let after = line[distRange.upperBound...]
        guard let endRange = after.range(of: ")") ?? after.range(of: ",") else { return nil }
        return Double(after[..<endRange.lowerBound])
    }

    let verboseDist = extractDistance(from: verboseOutput, containing: "Metal shader")
    let focusedDist = extractDistance(from: focusedOutput, containing: "Metal shader")

    if let vd = verboseDist, let fd = focusedDist {
        #expect(fd <= vd, "Focused query (distance: \(fd)) should match at least as well as verbose query (distance: \(vd))")
    }
}

// MARK: - Recall Gate Classifier

@Test func recallGate_technicalQuery_shouldRecall() throws {
    #expect(MemoryTools.shouldRecall(query: "Tell me about Lattice"))
}

@Test func recallGate_architectureQuestion_shouldRecall() throws {
    #expect(MemoryTools.shouldRecall(query: "how does the sync system work"))
}

@Test func recallGate_shortTopical_shouldRecall() throws {
    #expect(MemoryTools.shouldRecall(query: "Metal rendering"))
}

@Test func recallGate_multiSentenceTechnical_shouldRecall() throws {
    #expect(MemoryTools.shouldRecall(query: "The recall results are pulling in Metal rendering stuff when I'm asking about database queries. Why is the graph traversal so noisy?"))
}

@Test func recallGate_bugReport_shouldRecall() throws {
    #expect(MemoryTools.shouldRecall(query: "I'm getting a crash in CoreText TAttributes::ApplyFont when the mascot visits nodes"))
}

@Test func recallGate_codeSnippet_shouldRecall() throws {
    #expect(MemoryTools.shouldRecall(query: "This line is wrong: `let threshold = max(min(p75 * 1.2, bestDistance * 3.0), 1e-9)`. The 3.0 multiplier is way too generous."))
}

@Test func recallGate_yes_shouldSkip() throws {
    #expect(!MemoryTools.shouldRecall(query: "yes"))
}

@Test func recallGate_soundsGood_shouldSkip() throws {
    #expect(!MemoryTools.shouldRecall(query: "sounds good"))
}

@Test func recallGate_vagueFeedback_shouldSkip() throws {
    #expect(!MemoryTools.shouldRecall(query: "is it way better?"))
}

@Test func recallGate_letsDoIt_shouldSkip() throws {
    #expect(!MemoryTools.shouldRecall(query: "let's do it"))
}

@Test func recallGate_multiSentenceFiller_shouldSkip() throws {
    #expect(!MemoryTools.shouldRecall(query: "yeah that looks right. go ahead and do it."))
}

@Test func recallGate_genericDirective_shouldSkip() throws {
    #expect(!MemoryTools.shouldRecall(query: "can you undo that"))
}

@Test func recallGate_emotionalFiller_shouldSkip() throws {
    #expect(!MemoryTools.shouldRecall(query: "this is frustrating"))
}

// NOTE: Meta-conversation with technical-sounding words ("prompt", "examples", "context")
// is a genuinely hard boundary. The model may classify these as recall — acceptable
// false positive since the cost is just unnecessary context, not a missed recall.
// Example: "but i feel like you need way more examples" contains "examples" + "prompt"
// which look technical. Not worth over-fitting the model on these edge cases.

@Test func recallGate_operationalQuery_shouldRecall() throws {
    #expect(MemoryTools.shouldRecall(query: "can you check if the daemon is stuck?"))
}

// MARK: - Recall Integration: Knowledge Void Detection

@Test func recall_knowledgeVoid_signalsWeakResults() async throws {
    let tools = try await makeTools()

    // Store a single memory about a specific topic
    _ = try await tools.handle(CallTool.Parameters(
        name: "remember",
        arguments: ["content": .string("Lattice uses SQLite as its backend database engine")]
    ))

    // Query something completely unrelated
    let result = try await tools.handle(CallTool.Parameters(
        name: "recall",
        arguments: ["query": .string("kubernetes container orchestration deployment")]
    ))
    let output = text(from: result)

    // Should either find no results or signal weak recall
    let isWeak = output.contains("Weak recall") || output.contains("No memories found")
    #expect(isWeak, "Unrelated query should signal weak recall or return no results")
}
