import Testing
import EngramKit
import Lattice
import MCP
import Foundation

private func milliseconds(_ elapsed: Duration) -> Int64 {
    elapsed.components.seconds * 1000 + elapsed.components.attoseconds / 1_000_000_000_000_000
}

@Test func embeddingModelLoadTime() async throws {
    let start = ContinuousClock.now
    let embedder = EmbeddingService()
    await embedder.load()
    let elapsed = ContinuousClock.now - start

    let isLoaded = await embedder.isLoaded
    #expect(isLoaded, "Model should have loaded successfully")

    print("Embedding model load time: \(milliseconds(elapsed))ms")
}

@Test func embeddingLatencyShortQuery() async throws {
    let embedder = EmbeddingService()
    await embedder.load()

    let query = "sync teams orgs 3d graph galaxy"
    let start = ContinuousClock.now
    let result = try await embedder.embed(text: query)
    let elapsed = ContinuousClock.now - start

    #expect(result != nil, "Embedding should succeed")
    let ms = milliseconds(elapsed)
    print("Short query embedding (\(query.count) chars): \(ms)ms")
    #expect(ms < 5000, "Short query embedding should complete in under 5s")
}

@Test func embeddingLatencyLongPrompt() async throws {
    let embedder = EmbeddingService()
    await embedder.load()

    let prompt = """
    once we have sync fully working , for single user, teams, and orgs, we will want to display \
    those graphs. this is a pretty big change. we have all the pieces i believe- but still \
    requires us to really expand the 3d world. i imagine it currently as each graph being its \
    own "galaxy" of sorts. personal/user/members of teams graphs will sit on the same x plane. \
    the synthesized / hive mind team graph will sit on top of the individual team graphs, org \
    graph sits on top of that, etc etc
    """
    let start = ContinuousClock.now
    let result = try await embedder.embed(text: prompt)
    let elapsed = ContinuousClock.now - start

    #expect(result != nil, "Embedding should succeed")
    let ms = milliseconds(elapsed)
    print("Long prompt embedding (\(prompt.count) chars): \(ms)ms")
    #expect(ms < 5000, "Long prompt embedding should complete in under 5s")
}

@Test func contentWordStrippingEmbeddingLatency() async throws {
    let embedder = EmbeddingService()
    await embedder.load()

    let prompt = """
    once we have sync fully working , for single user, teams, and orgs, we will want to display \
    those graphs. this is a pretty big change. we have all the pieces i believe- but still \
    requires us to really expand the 3d world. i imagine it currently as each graph being its \
    own "galaxy" of sorts. personal/user/members of teams graphs will sit on the same x plane. \
    the synthesized / hive mind team graph will sit on top of the individual team graphs, org \
    graph sits on top of that, etc etc
    """

    let contentWords = MemoryTools.extractContentWords(from: prompt)
    let stripped = contentWords.joined(separator: " ")
    print("Original: \(prompt.count) chars -> Stripped: \(stripped.count) chars (\(contentWords.count) words)")
    print("Stripped text: \(stripped)")

    // Time raw prompt embedding
    let startRaw = ContinuousClock.now
    let rawResult = try await embedder.embed(text: prompt)
    let rawMs = milliseconds(ContinuousClock.now - startRaw)

    // Time stripped embedding
    let startStripped = ContinuousClock.now
    let strippedResult = try await embedder.embed(text: stripped)
    let strippedMs = milliseconds(ContinuousClock.now - startStripped)

    #expect(rawResult != nil)
    #expect(strippedResult != nil)
    print("Raw prompt embedding: \(rawMs)ms")
    print("Stripped embedding: \(strippedMs)ms")
    #expect(rawMs < 5000, "Raw embedding should complete in under 5s")
    #expect(strippedMs < 5000, "Stripped embedding should complete in under 5s")
}

/// KeyBERT-style keyword extraction: embed each content word, rank by cosine
/// similarity to full query embedding, apply MMR diversification, take top N.
/// Proves the approach selects semantically relevant keywords and that a
/// reduced FTS term set is fast against the real database.
@Test func keyBERTKeywordExtraction() async throws {
    let embedder = EmbeddingService()
    await embedder.load()

    let prompt = """
    once we have sync fully working , for single user, teams, and orgs, we will want to display \
    those graphs. this is a pretty big change. we have all the pieces i believe- but still \
    requires us to really expand the 3d world. i imagine it currently as each graph being its \
    own "galaxy" of sorts. personal/user/members of teams graphs will sit on the same x plane. \
    the synthesized / hive mind team graph will sit on top of the individual team graphs, org \
    graph sits on top of that, etc etc
    """

    // Step 1: extract content words, strip punctuation/single-char tokens, deduplicate
    let contentWords = MemoryTools.extractContentWords(from: prompt)
    let cleaned = contentWords.filter { word in
        word.count > 1 && word.unicodeScalars.contains(where: CharacterSet.letters.contains)
    }
    let uniqueWords = Array(Set(cleaned))
    print("Content words: \(contentWords.count) raw → \(cleaned.count) cleaned → \(uniqueWords.count) unique")
    print("Unique words: \(uniqueWords.sorted())")

    // Step 2: embed the full query
    let startTotal = ContinuousClock.now
    guard let queryEmbedding = try await embedder.embed(text: prompt) else {
        Issue.record("Failed to embed full query")
        return
    }

    // Step 3: embed each unique word
    var wordEmbeddings: [(word: String, embedding: [Float])] = []
    for word in uniqueWords {
        guard let emb = try await embedder.embed(text: word) else { continue }
        wordEmbeddings.append((word: word, embedding: emb))
    }

    // Step 4a: raw cosine similarity ranking
    var rawScores: [(word: String, similarity: Float)] = wordEmbeddings.map { we in
        (word: we.word, similarity: cosineSimilarity(queryEmbedding, we.embedding))
    }.sorted { $0.similarity > $1.similarity }

    print("\nRaw cosine ranking (top 20):")
    for (i, ws) in rawScores.prefix(20).enumerated() {
        print("  \(i + 1). \(ws.word) → \(String(format: "%.4f", ws.similarity))")
    }

    // Step 4b: MMR (Maximal Marginal Relevance) — balances relevance with diversity
    // This is what real KeyBERT uses to avoid selecting redundant keywords.
    // MMR score = lambda * cos(word, query) - (1 - lambda) * max(cos(word, already_selected))
    let lambda: Float = 0.5
    let topN = 15
    var selected: [(word: String, embedding: [Float], score: Float)] = []
    var remaining = wordEmbeddings

    for _ in 0..<min(topN, remaining.count) {
        var bestIdx = 0
        var bestScore: Float = -Float.infinity
        for (i, candidate) in remaining.enumerated() {
            let relevance = cosineSimilarity(queryEmbedding, candidate.embedding)
            let maxRedundancy: Float
            if selected.isEmpty {
                maxRedundancy = 0
            } else {
                maxRedundancy = selected.map { cosineSimilarity(candidate.embedding, $0.embedding) }.max() ?? 0
            }
            let mmrScore = lambda * relevance - (1 - lambda) * maxRedundancy
            if mmrScore > bestScore {
                bestScore = mmrScore
                bestIdx = i
            }
        }
        let chosen = remaining.remove(at: bestIdx)
        selected.append((word: chosen.word, embedding: chosen.embedding, score: bestScore))
    }

    let extractionMs = milliseconds(ContinuousClock.now - startTotal)

    let mmrKeywords = selected.map(\.word)
    print("\nMMR keywords (lambda=\(lambda)): \(mmrKeywords)")
    print("MMR scores:")
    for (i, s) in selected.enumerated() {
        print("  \(i + 1). \(s.word) → \(String(format: "%.4f", s.score))")
    }
    print("Total KeyBERT extraction time: \(extractionMs)ms")
    #expect(extractionMs < 5000, "KeyBERT extraction should complete in under 5s")

    // Step 5: verify against real DB at multiple term counts
    let dbPath = NSHomeDirectory() + "/.claude/memory.sqlite"
    guard FileManager.default.fileExists(atPath: dbPath) else {
        print("No real DB — keyword extraction test passed, skipping recall comparison")
        return
    }

    let lattice = try Lattice(
        Memory.self, Edge.self, Checkpoint.self, HookState.self,
        configuration: .init(fileURL: URL(fileURLWithPath: dbPath), migration: engramMigrations)
    )
    let tools = MemoryTools(localRef: lattice.sendableReference, syncedRef: nil, embedder: embedder)
    let memoryCount = lattice.objects(Memory.self).count
    print("\nDatabase: \(memoryCount) memories")

    // Test multiple term counts to find the sweet spot
    for termCount in [5, 8, 10, 15] {
        let terms = Array(mmrKeywords.prefix(termCount))
        let query = terms.joined(separator: " ")
        let start = ContinuousClock.now
        let result = try await tools.directRecall(query: query, project: "Engram", depth: 0, limit: 5)
        let ms = milliseconds(ContinuousClock.now - start)
        print("directRecall \(termCount) terms depth=0: \(ms)ms (\(result?.count ?? 0) chars) — \(terms)")
    }

    // Baseline: all terms
    let startAll = ContinuousClock.now
    let resultAll = try await tools.directRecall(query: prompt, project: "Engram", depth: 0, limit: 5)
    let allMs = milliseconds(ContinuousClock.now - startAll)
    print("directRecall ALL \(contentWords.count) terms depth=0: \(allMs)ms (\(resultAll?.count ?? 0) chars)")

    // The 8-term query should be under 5s (hook timeout)
    let hookTerms = Array(mmrKeywords.prefix(8))
    let hookQuery = hookTerms.joined(separator: " ")
    let startHook = ContinuousClock.now
    let resultHook = try await tools.directRecall(query: hookQuery, project: "Engram", depth: 1, limit: 5)
    let hookMs = milliseconds(ContinuousClock.now - startHook)
    print("\ndirectRecall 8 terms depth=1 (hook scenario): \(hookMs)ms (\(resultHook?.count ?? 0) chars)")

    if let resultHook {
        print("\n--- KeyBERT recall results (8 terms, depth=1) ---")
        print(resultHook.prefix(2000))
    }

    #expect(hookMs < 5000, "KeyBERT directRecall (8 terms, depth=1) should complete in under 5s")
}

/// Cosine similarity between two Float vectors.
private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
    guard a.count == b.count, !a.isEmpty else { return 0 }
    var dot: Float = 0
    var magA: Float = 0
    var magB: Float = 0
    for i in 0..<a.count {
        dot += a[i] * b[i]
        magA += a[i] * a[i]
        magB += b[i] * b[i]
    }
    let denom = sqrt(magA) * sqrt(magB)
    return denom > 0 ? dot / denom : 0
}

/// Option 3: Vector-first (L2 KNN index) → FTS5 re-rank in application code.
/// Bypasses the FTS5→vector chain that forces brute-force cosine.
@Test func vectorFirstWithFTSRerank() async throws {
    let dbPath = NSHomeDirectory() + "/.claude/memory.sqlite"
    guard FileManager.default.fileExists(atPath: dbPath) else {
        print("Skipping: no real database at \(dbPath)")
        return
    }

    let lattice = try Lattice(
        Memory.self, Edge.self, Checkpoint.self, HookState.self,
        configuration: .init(fileURL: URL(fileURLWithPath: dbPath), migration: engramMigrations)
    )
    let embedder = EmbeddingService()
    await embedder.load()

    let memoryCount = lattice.objects(Memory.self).count
    print("Database: \(memoryCount) memories")

    let prompt = """
    once we have sync fully working , for single user, teams, and orgs, we will want to display \
    those graphs. this is a pretty big change. we have all the pieces i believe- but still \
    requires us to really expand the 3d world. i imagine it currently as each graph being its \
    own "galaxy" of sorts. personal/user/members of teams graphs will sit on the same x plane. \
    the synthesized / hive mind team graph will sit on top of the individual team graphs, org \
    graph sits on top of that, etc etc
    """

    guard let queryEmbedding = try await embedder.embed(text: prompt) else {
        Issue.record("Failed to embed query")
        return
    }
    let embedding = Vector<Float>(queryEmbedding)

    // Step 1: Check if embeddings are normalized (L2 ≡ cosine for unit vectors)
    let magnitude = sqrt(queryEmbedding.reduce(0) { $0 + $1 * $1 })
    print("Query embedding magnitude: \(magnitude) (should be ~1.0 for normalized)")

    // Step 2: Vector-only L2 KNN — uses native MATCH index, no FTS5
    let startL2 = ContinuousClock.now
    let l2Results = lattice.objects(Memory.self)
        .nearest(to: embedding, on: \.embedding, limit: 20, distance: .l2)
    let l2Memories = Array(l2Results)
    let l2Ms = milliseconds(ContinuousClock.now - startL2)
    print("\nL2 KNN (20 candidates): \(l2Ms)ms")
    for (i, match) in l2Memories.prefix(10).enumerated() {
        let m = match.object
        let dist = match.distance
        print("  \(i + 1). [id:\(m.__globalId)] d=\(String(format: "%.4f", dist)) [\(m.project ?? "?")/\(m.topic ?? "?")] \(String(m.content.prefix(80)))")
    }

    // Step 3: Vector-only cosine for comparison (brute-force, no index)
    let startCos = ContinuousClock.now
    let cosResults = lattice.objects(Memory.self)
        .nearest(to: embedding, on: \.embedding, limit: 20, distance: .cosine)
    let cosMemories = Array(cosResults)
    let cosMs = milliseconds(ContinuousClock.now - startCos)
    print("\nCosine brute-force (20 candidates): \(cosMs)ms")
    for (i, match) in cosMemories.prefix(10).enumerated() {
        let m = match.object
        let dist = match.distance
        print("  \(i + 1). [id:\(m.__globalId)] d=\(String(format: "%.4f", dist)) [\(m.project ?? "?")/\(m.topic ?? "?")] \(String(m.content.prefix(80)))")
    }

    // Step 4: FTS5 re-rank on L2 candidates (application-side term overlap scoring)
    let contentWords = Set(MemoryTools.extractContentWords(from: prompt))
    let reranked: [(match: _NearestMatch<Memory>, ftsScore: Double)] = l2Memories.map { match in
        let words = Set(match.object.content.lowercased().split(separator: " ").map(String.init))
        let overlap = Double(contentWords.intersection(words).count) / Double(max(contentWords.count, 1))
        return (match: match, ftsScore: overlap)
    }
    .sorted { a, b in
        // Combined score: lower distance is better, higher overlap is better
        let scoreA = a.match.distance - a.ftsScore * 0.1
        let scoreB = b.match.distance - b.ftsScore * 0.1
        return scoreA < scoreB
    }

    print("\nL2 + FTS re-ranked (top 5):")
    for (i, item) in reranked.prefix(5).enumerated() {
        let m = item.match.object
        print("  \(i + 1). [id:\(m.__globalId)] d=\(String(format: "%.4f", item.match.distance)) fts=\(String(format: "%.2f", item.ftsScore)) [\(m.project ?? "?")/\(m.topic ?? "?")] \(String(m.content.prefix(80)))")
    }

    // Step 5: Compare with current approach (FTS5 + cosine, slow)
    let tools = MemoryTools(localRef: lattice.sendableReference, syncedRef: nil, embedder: embedder)
    let startOld = ContinuousClock.now
    let resultOld = try await tools.directRecall(query: prompt, project: "Engram", depth: 0, limit: 5)
    let oldMs = milliseconds(ContinuousClock.now - startOld)
    print("\nCurrent FTS5+cosine depth=0: \(oldMs)ms (\(resultOld?.count ?? 0) chars)")

    // Do the L2 and cosine results overlap?
    let l2Ids = Set(l2Memories.prefix(10).map { $0.object.__globalId })
    let cosIds = Set(cosMemories.prefix(10).map { $0.object.__globalId })
    let overlap = l2Ids.intersection(cosIds)
    print("\nTop-10 overlap (L2 vs cosine): \(overlap.count)/10 shared IDs")

    print("\nSummary:")
    print("  L2 KNN (indexed):    \(l2Ms)ms")
    print("  Cosine (brute-force): \(cosMs)ms")
    print("  Current FTS+cosine:  \(oldMs)ms")
    #expect(l2Ms < 1000, "L2 KNN should complete in under 1s")
}

/// End-to-end directRecall latency against the real database.
/// This is what the UserPromptSubmit hook actually runs.
@Test func directRecallLatency() async throws {
    let dbPath = NSHomeDirectory() + "/.claude/memory.sqlite"
    guard FileManager.default.fileExists(atPath: dbPath) else {
        print("Skipping: no real database at \(dbPath)")
        return
    }

    let lattice = try Lattice(
        Memory.self, Edge.self, Checkpoint.self, HookState.self,
        configuration: .init(fileURL: URL(fileURLWithPath: dbPath), migration: engramMigrations)
    )
    let embedder = EmbeddingService()
    await embedder.load()
    let tools = MemoryTools(localRef: lattice.sendableReference, syncedRef: nil, embedder: embedder)

    let memoryCount = lattice.objects(Memory.self).count
    let edgeCount = lattice.objects(Edge.self).count
    print("Database: \(memoryCount) memories, \(edgeCount) edges")

    let prompt = """
    once we have sync fully working , for single user, teams, and orgs, we will want to display \
    those graphs. this is a pretty big change. we have all the pieces i believe- but still \
    requires us to really expand the 3d world. i imagine it currently as each graph being its \
    own "galaxy" of sorts. personal/user/members of teams graphs will sit on the same x plane. \
    the synthesized / hive mind team graph will sit on top of the individual team graphs, org \
    graph sits on top of that, etc etc
    """

    // Depth 0 (no graph traversal)
    let startD0 = ContinuousClock.now
    let resultD0 = try await tools.directRecall(query: prompt, project: "Engram", depth: 0, limit: 5)
    let d0Ms = milliseconds(ContinuousClock.now - startD0)
    print("directRecall depth=0: \(d0Ms)ms (\(resultD0?.count ?? 0) chars)")

    // Depth 1 (graph traversal — what the hook uses)
    let startD1 = ContinuousClock.now
    let resultD1 = try await tools.directRecall(query: prompt, project: "Engram", depth: 1, limit: 5)
    let d1Ms = milliseconds(ContinuousClock.now - startD1)
    print("directRecall depth=1: \(d1Ms)ms (\(resultD1?.count ?? 0) chars)")

    print("Graph traversal overhead: \(d1Ms - d0Ms)ms")
    #expect(d1Ms < 5000, "directRecall depth=1 should complete in under 5s (hook timeout)")
}
