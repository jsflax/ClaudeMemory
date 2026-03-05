import SwiftUI
import EngramKit
import Lattice

#if canImport(FoundationModels)
import FoundationModels
#endif

/// Manages on-device LLM chat sessions for the mascot companion.
///
/// Uses a RAG-style architecture: we gather context (semantic recall, recent nodes,
/// current selection) and bake it into the system instructions. The on-device model
/// just has a conversation — no tool schemas, no constrained decoding.
/// Session compaction triggers at ~70% estimated context usage to prevent blowout.
@available(macOS 26.0, *)
@Observable @MainActor
final class MascotChatEngine {

    struct ChatMessage: Identifiable {
        let id = UUID()
        let role: Role
        let text: String
        enum Role { case user, assistant }
    }

    private(set) var messages: [ChatMessage] = []
    private(set) var isThinking = false
    private(set) var isModelAvailable = false

    // MARK: - Dependencies (set via setup())

    private var memoryTools: MemoryTools?
    private var renderStore: GraphRenderStore?
    private var selectedMemoryIdProvider: (() -> UUID?)?

    // MARK: - Session state

    private var session: LanguageModelSession?
    private var sessionCharCount = 0

    /// Estimated token capacity of the on-device model's context window.
    private let estimatedTokenCapacity = 4096
    /// Trigger compaction when estimated usage exceeds this fraction.
    private let compactionThreshold = 0.70

    // MARK: - Setup

    func setup(
        lattice: Lattice,
        renderStore: GraphRenderStore,
        selectedMemoryId: @escaping () -> UUID?
    ) async {
        guard memoryTools == nil else { return }

        guard SystemLanguageModel.default.availability == .available else {
            isModelAvailable = false
            messages.append(ChatMessage(
                role: .assistant,
                text: "On-device model not available on this hardware."
            ))
            return
        }
        isModelAvailable = true

        self.renderStore = renderStore
        self.selectedMemoryIdProvider = selectedMemoryId

        let embedder = EmbeddingService()
        if await !embedder.isLoaded { await embedder.load() }
        self.memoryTools = MemoryTools(localRef: lattice.sendableReference, syncedRef: nil, embedder: embedder)
    }

    // MARK: - Send

    func send(_ text: String) async {
        messages.append(ChatMessage(role: .user, text: text))
        isThinking = true
        defer { isThinking = false }

        guard memoryTools != nil else {
            messages.append(ChatMessage(role: .assistant, text: "Still initializing..."))
            return
        }

        do {
            // Lazy session creation on first message
            if session == nil {
                await createSession(query: text)
            } else if shouldCompact {
                await compactSession(query: text)
            }

            sessionCharCount += text.count
            let response = try await session!.respond(to: text)
            let content = response.content
            sessionCharCount += content.count
            messages.append(ChatMessage(role: .assistant, text: content))
        } catch {
            messages.append(ChatMessage(
                role: .assistant,
                text: "Error: \(error.localizedDescription)"
            ))
        }
    }

    func reset() {
        messages.removeAll()
        session = nil
        sessionCharCount = 0
        memoryTools = nil
        renderStore = nil
        selectedMemoryIdProvider = nil
    }

    // MARK: - Session management

    private func createSession(query: String) async {
        let instructions = await buildSystemInstructions(query: query)
        session = LanguageModelSession(instructions: instructions)
        sessionCharCount = instructions.count
    }

    private func compactSession(query: String) async {
        guard let currentSession = session else { return }

        // Ask the current session to summarize the conversation so far
        let summary: String
        do {
            let summaryResponse = try await currentSession.respond(
                to: "Summarize our conversation so far in 2-3 sentences. Be concise."
            )
            summary = summaryResponse.content
        } catch {
            summary = ""
        }

        // Build fresh instructions with refreshed RAG context + conversation summary
        let ragInstructions = await buildSystemInstructions(query: query)
        let compactedInstructions: String
        if summary.isEmpty {
            compactedInstructions = ragInstructions
        } else {
            compactedInstructions = """
                \(ragInstructions)

                Conversation so far: \(summary)
                """
        }
        session = LanguageModelSession(instructions: compactedInstructions)
        sessionCharCount = compactedInstructions.count
    }

    private var shouldCompact: Bool {
        // Rough estimate: ~4 chars per token
        let estimatedTokens = sessionCharCount / 4
        let threshold = Int(Double(estimatedTokenCapacity) * compactionThreshold)
        return estimatedTokens >= threshold
    }

    // MARK: - RAG context gathering

    private func buildSystemInstructions(query: String) async -> String {
        var sections: [String] = []

        // Persona
        sections.append("""
            You are an Engram memory companion — a small robot living inside the user's \
            memory graph. Be helpful, concise, and friendly. Keep responses to 1-3 sentences.
            """)

        // 1. Semantic recall (~750 chars)
        if let mt = memoryTools {
            do {
                if let recallResult = try await mt.directRecall(
                    query: query, project: nil, depth: 1, limit: 5
                ) {
                    let truncated = String(recallResult.prefix(750))
                    sections.append("Relevant memories:\n\(truncated)")
                }
            } catch {
                // Recall failure is non-fatal — proceed without it
            }
        }

        // 2. Recent nodes (~400 chars)
        if let store = renderStore {
            let recent = store.recentNodes.prefix(8)
            if !recent.isEmpty {
                let items = recent.map { "- [\($0.project)/\($0.topic)] \($0.label)" }
                    .joined(separator: "\n")
                sections.append("Recent memories:\n\(items)")
            }
        }

        // 3. Recent activity with content snippets (~300 chars)
        if let store = renderStore {
            let active = store.recentNodes.prefix(5)
            if !active.isEmpty {
                let items = active.map {
                    let snippet = String($0.content.prefix(60))
                    return "- \($0.label): \(snippet)"
                }.joined(separator: "\n")
                sections.append("Recent activity:\n\(items)")
            }
        }

        // 4. Current selection (~200 chars)
        if let provider = selectedMemoryIdProvider,
           let selectedId = provider(),
           let store = renderStore,
           let node = store.nodeById[selectedId] {
            let snippet = String(node.content.prefix(200))
            sections.append("Currently selected: [\(node.project)/\(node.topic)] \(node.label)\n\(snippet)")
        }

        return sections.joined(separator: "\n\n")
    }
}
