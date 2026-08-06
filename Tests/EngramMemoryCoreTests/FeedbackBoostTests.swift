import EngramMemoryCore
import Testing

// MARK: - Feedback actuator (increment 12)

@Suite("Feedback boost")
struct FeedbackBoostTests {

    @Test func zeroFeedbackIsIdentity() {
        #expect(RecallRanking.feedbackBoost(positive: 0, negative: 0) == 1.0)
    }

    @Test func positiveFeedbackImprovesRankAndCaps() {
        let one = RecallRanking.feedbackBoost(positive: 1, negative: 0)
        let ten = RecallRanking.feedbackBoost(positive: 10, negative: 0)
        let thousand = RecallRanking.feedbackBoost(positive: 1000, negative: 0)
        #expect(one < 1.0)
        #expect(ten < one)
        #expect(thousand >= 0.85, "positive cap is 15%")
        #expect(thousand <= ten)
    }

    @Test func negativeFeedbackDemotesAndCaps() {
        let one = RecallRanking.feedbackBoost(positive: 0, negative: 1)
        let thousand = RecallRanking.feedbackBoost(positive: 0, negative: 1000)
        #expect(one > 1.0)
        #expect(thousand <= 1.25 + 1e-9, "negative cap is 25%")
    }

    @Test func feedbackFoldsIntoBoostedDistance() {
        let base = RecallRanking.boostedDistance(
            l2Distance: 1.0, scope: .other, accessCount: 0, importance: 0,
            daysSinceAccess: 0, daysSinceCreation: 0)
        let promoted = RecallRanking.boostedDistance(
            l2Distance: 1.0, scope: .other, accessCount: 0, importance: 0,
            daysSinceAccess: 0, daysSinceCreation: 0,
            feedbackPositive: 8, feedbackNegative: 0)
        let demoted = RecallRanking.boostedDistance(
            l2Distance: 1.0, scope: .other, accessCount: 0, importance: 0,
            daysSinceAccess: 0, daysSinceCreation: 0,
            feedbackPositive: 0, feedbackNegative: 8)
        #expect(promoted < base)
        #expect(demoted > base)
    }
}
