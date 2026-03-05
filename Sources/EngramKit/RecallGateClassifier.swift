@preconcurrency import CoreML
import Foundation

/// Classifies user prompts as "recall" (worth searching memories) or "skip" (conversational filler).
/// Uses a CoreML text classifier trained on labeled examples via CreateML transfer learning.
public final class RecallGateClassifier: @unchecked Sendable {
    private let model: MLModel

    public init() throws {
        guard let url = Bundle.module.url(forResource: "RecallGateClassifier", withExtension: "mlmodelc") else {
            throw RecallGateError.modelNotFound
        }
        self.model = try MLModel(contentsOf: url)
    }

    /// Returns true if the query is worth recalling memories for.
    public func shouldRecall(query: String) -> Bool {
        guard !query.isEmpty else { return false }
        guard let input = try? MLDictionaryFeatureProvider(dictionary: ["text": query as NSString]) else {
            return true // fail open
        }
        guard let output = try? model.prediction(from: input) else {
            return true // fail open
        }
        guard let label = output.featureValue(for: "label")?.stringValue else {
            return true // fail open
        }
        return label == "recall"
    }

    enum RecallGateError: Error {
        case modelNotFound
    }
}

// MARK: - Convenience on MemoryTools

extension MemoryTools {
    /// Convenience: classify whether a query should trigger recall.
    /// Uses the bundled RecallGateClassifier CoreML model.
    public static func shouldRecall(query: String) -> Bool {
        guard let gate = try? RecallGateClassifier() else { return true }
        return gate.shouldRecall(query: query)
    }
}
