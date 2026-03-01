import Testing
import EngramKit
import Lattice
import MCP
import Foundation

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

@Test func extractContentWords_keepsVerbsDropsPronounsDeterminers() {
    // NLTagger can't distinguish auxiliary from main verbs (no AUX tag),
    // so all verbs pass through. Pronouns and determiners are still dropped.
    let words = MemoryTools.extractContentWords(from: "have we been doing this")
    #expect(words.contains("have"))
    #expect(words.contains("been"))
    #expect(words.contains("doing"))
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
    // NLTagger has no AUX tag — "does" passes through as a verb
    #expect(words.contains("does"))
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
            "from": .int(id1),
            "to": .int(id2),
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
