import Foundation

/// The recall ranking + conflict-detection constants and pure functions,
/// shared by every `MemoryService` conformance so backends cannot drift.
///
/// Extracted verbatim from `MemoryTools+Core` (the lattice implementation) —
/// the Postgres conformance fetches raw KNN candidates and re-ranks with
/// THESE functions, never with SQL-side approximations. Any tuning (including
/// the increment-12 feedback actuator) happens here, once, for both backends.
public enum RecallRanking {

    /// Same-project match boost (strong), applied to L2 distance
    /// (lower = better, so boosts multiply the distance DOWN).
    public static let sameProjectBoost = 0.7
    /// Global-scope match boost (moderate).
    public static let globalScopeBoost = 0.85
    /// No boost.
    public static let noBoost = 1.0

    /// Log-scaled access-frequency boost, capped at 15%.
    public static func frequencyBoost(accessCount: Int) -> Double {
        1.0 - min(log2(1.0 + Double(accessCount)) * 0.04, 0.15)
    }

    /// Importance boost: up to 20% at importance 5. Importance 0 = unset.
    public static func importanceBoost(importance: Int) -> Double {
        importance > 0 ? 1.0 - Double(importance - 1) * 0.05 : 1.0
    }

    /// Recency boost: up to 10%, exponentially decaying over ~30 days since
    /// last access.
    public static func recencyBoost(daysSinceAccess: Double) -> Double {
        1.0 - 0.1 * exp(-daysSinceAccess / 30.0)
    }

    /// Staleness penalty: +20% distance for rows never accessed and older
    /// than 14 days.
    public static func stalenessPenalty(accessCount: Int, daysSinceCreation: Double) -> Double {
        (accessCount == 0 && daysSinceCreation > 14.0) ? 1.2 : 1.0
    }

    public enum ProjectScope: Sendable {
        /// Row's project equals the query's project filter.
        case sameProject
        /// Row is global-scoped while a project filter is active.
        case global
        /// Row belongs to a different project, or no filter is active.
        case other
    }

    public static func projectBoost(_ scope: ProjectScope) -> Double {
        switch scope {
        case .sameProject: return sameProjectBoost
        case .global: return globalScopeBoost
        case .other: return noBoost
        }
    }

    /// The full boosted distance — the ranking key (ascending).
    public static func boostedDistance(
        l2Distance: Double,
        scope: ProjectScope,
        accessCount: Int,
        importance: Int,
        daysSinceAccess: Double,
        daysSinceCreation: Double
    ) -> Double {
        l2Distance
            * projectBoost(scope)
            * frequencyBoost(accessCount: accessCount)
            * importanceBoost(importance: importance)
            * recencyBoost(daysSinceAccess: daysSinceAccess)
            * stalenessPenalty(accessCount: accessCount, daysSinceCreation: daysSinceCreation)
    }
}

/// Duplicate/conflict detection thresholds (remember-time), shared verbatim.
public enum ConflictThresholds {
    /// L2 distance below which two memories are conflict candidates.
    /// L2 equivalents of cosine 0.12 (same project) / 0.05 (cross-scope)
    /// over normalized vectors.
    public static func l2Threshold(sameProject: Bool) -> Double {
        sameProject ? 0.49 : 0.316
    }

    /// A conflict candidate must ALSO share terms: Jaccard ≥ 0.4.
    public static let jaccardThreshold = 0.4

    /// Auto-connect band: [conflict threshold, 0.632) — L2 of cosine 0.20.
    /// Related-but-distinct memories get `relates_to` edges, top 3.
    public static let autoConnectUpperBound = 0.632
}

/// Term-overlap similarity — tokenization must stay IDENTICAL across
/// backends (lowercased, split on non-alphanumerics, tokens ≥ 3 chars).
public func jaccardSimilarity(_ a: String, _ b: String) -> Double {
    func tokenize(_ s: String) -> Set<String> {
        Set(
            s.lowercased()
                .components(separatedBy: CharacterSet.alphanumerics.inverted)
                .filter { $0.count >= 3 }
        )
    }
    let setA = tokenize(a)
    let setB = tokenize(b)
    guard !setA.isEmpty || !setB.isEmpty else { return 0.0 }
    return Double(setA.intersection(setB).count) / Double(setA.union(setB).count)
}
