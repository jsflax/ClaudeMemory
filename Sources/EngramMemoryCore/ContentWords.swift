import Foundation

/// Distills a query to its content words for FTS + embedding focus.
/// Two conformances: NLTagger POS filtering (EngramKit, Apple-only) and the
/// portable stopword extractor below. The contract-suite golden test pins
/// both to the same core terms on representative queries.
public protocol ContentWordExtracting: Sendable {
    func extractContentWords(from query: String) -> [String]
}

/// Portable content-word extraction via a static closed-class stopword list.
///
/// The NLTagger implementation drops the POS classes determiner, pronoun,
/// preposition, conjunction, and particle — all CLOSED word classes in
/// English, so a static list approximates the tagger faithfully without
/// NaturalLanguage. The explicit copula/auxiliary/interrogative drops are
/// copied verbatim from the Apple implementation (NLTagger keeps those as
/// verbs/adverbs, so both impls special-case them identically).
///
/// Same fallback contract as the Apple impl: if nothing survives, return
/// all whitespace-split words.
public struct StopwordContentWordExtractor: ContentWordExtracting {

    public init() {}

    /// Copied verbatim from the NLTagger implementation's dropWords —
    /// function words NLTagger's lexical class KEEPS but which carry no
    /// recall signal.
    static let dropWords: Set<String> = [
        "is", "are", "was", "were", "be", "been", "being", "am",  // copulas
        "have", "has", "had", "do", "does", "did",                // auxiliaries
        "how", "what", "when", "where", "why", "who", "which",    // interrogatives
    ]

    /// The closed-class function words the NLTagger impl drops by POS tag.
    /// Determiners, pronouns, prepositions, conjunctions, particles.
    static let closedClassWords: Set<String> = [
        // Determiners
        "a", "an", "the", "this", "that", "these", "those", "each", "every",
        "either", "neither", "some", "any", "no", "another", "such", "both",
        "all", "few", "several", "many", "much", "more", "most", "less",
        "least", "own", "other", "enough",
        // Pronouns
        "i", "me", "my", "mine", "myself", "we", "us", "our", "ours",
        "ourselves", "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself", "she", "her", "hers", "herself",
        "it", "its", "itself", "they", "them", "their", "theirs",
        "themselves", "one", "ones", "oneself", "someone", "somebody",
        "something", "anyone", "anybody", "anything", "everyone",
        "everybody", "everything", "nobody", "nothing", "whoever",
        "whatever", "whichever", "whom", "whose",
        // Prepositions
        "in", "on", "at", "by", "for", "with", "about", "against",
        "between", "into", "through", "during", "before", "after",
        "above", "below", "to", "from", "up", "down", "of", "off",
        "over", "under", "again", "further", "then", "once", "onto",
        "upon", "within", "without", "along", "across", "behind",
        "beyond", "near", "among", "around", "amid", "beside", "besides",
        "despite", "except", "inside", "outside", "past", "per", "since",
        "till", "until", "toward", "towards", "underneath", "unlike",
        "via", "vs", "versus",
        // Conjunctions
        "and", "but", "or", "nor", "so", "yet", "although", "because",
        "if", "unless", "while", "whereas", "though", "whether", "as",
        // Particles
        "not", "nt", "too", "also", "just", "only", "even", "than",
    ]

    public func extractContentWords(from query: String) -> [String] {
        guard !query.isEmpty else { return [] }
        let words = query.components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
        let content = words.filter { word in
            let normalized = word.lowercased()
                .trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
            guard !normalized.isEmpty else { return false }
            return !Self.closedClassWords.contains(normalized)
                && !Self.dropWords.contains(normalized)
        }
        // Same fallback as the NLTagger impl: an all-function-word query
        // still needs SOMETHING to search on.
        return content.isEmpty ? words : content
    }
}
