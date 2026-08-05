import Foundation
import Testing
import EngramKit
import EngramMemoryCore

/// Golden parity between the NLTagger content-word extractor (Apple, the
/// recall/advise behavior shipping today) and the portable stopword
/// extractor (Linux, what the agents service runs).
///
/// The contract is CORE-TERM parity, not token-for-token equality: both
/// implementations must retain the same load-bearing terms on
/// representative queries, so a recall query distills to the same search
/// regardless of platform. NLTagger may keep an extra marginal token the
/// list-based impl drops (or vice versa) — that's tolerated; losing a core
/// term is not.
@Suite("Content-word extractor parity (NLTagger vs stopword list)")
struct ContentWordParityTests {

    static let goldens: [(query: String, coreTerms: [String])] = [
        ("how do I configure the sync daemon for the group relay",
         ["configure", "sync", "daemon", "group", "relay"]),
        ("what is the deployment pipeline for the ios app",
         ["deployment", "pipeline", "ios", "app"]),
        ("why does sqlite3_column_name return NULL under memory pressure",
         ["sqlite3_column_name", "return", "NULL", "memory", "pressure"]),
        ("remember that the stripe webhook needs the group id in metadata",
         ["remember", "stripe", "webhook", "needs", "group", "metadata"]),
        ("fix the race condition in the token refresh middleware",
         ["fix", "race", "condition", "token", "refresh", "middleware"]),
        ("agents should recall their previous debugging sessions",
         ["agents", "recall", "previous", "debugging", "sessions"]),
    ]

    @Test func portableExtractorKeepsCoreTerms() {
        let portable = StopwordContentWordExtractor()
        for golden in Self.goldens {
            let words = Set(portable.extractContentWords(from: golden.query)
                .map { $0.lowercased() })
            for term in golden.coreTerms {
                #expect(words.contains(term.lowercased()),
                        "portable extractor lost core term '\(term)' for: \(golden.query)")
            }
        }
    }

    @Test func nlTaggerExtractorKeepsCoreTerms() {
        for golden in Self.goldens {
            let words = Set(MemoryTools.extractContentWords(from: golden.query)
                .map { $0.lowercased() })
            for term in golden.coreTerms {
                #expect(words.contains(term.lowercased()),
                        "NLTagger extractor lost core term '\(term)' for: \(golden.query)")
            }
        }
    }

    @Test func bothDropTheSameFunctionWords() {
        let portable = StopwordContentWordExtractor()
        // NOTE: modals ("should", "can", "must") are deliberately absent —
        // NLTagger tags them as verbs and KEEPS them, and the portable list
        // mirrors that. Parity means agreeing with the shipped behavior.
        let functionWords = ["the", "for", "does", "under", "their", "what", "how"]
        for golden in Self.goldens {
            let a = Set(portable.extractContentWords(from: golden.query).map { $0.lowercased() })
            let b = Set(MemoryTools.extractContentWords(from: golden.query).map { $0.lowercased() })
            for fw in functionWords {
                #expect(!a.contains(fw), "portable kept function word '\(fw)'")
                #expect(!b.contains(fw), "NLTagger kept function word '\(fw)'")
            }
        }
    }
}
