import SwiftUI
import EngramKit
import Lattice

#if canImport(FoundationModels)
import FoundationModels
import EngramFoundationModels
#endif

/// Manages on-device LLM chat sessions for the mascot companion.
/// Uses Apple's FoundationModels framework with Engram memory tools (read-only).
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
    private var session: LanguageModelSession?
    private var memoryTools: MemoryTools?

    func setup(lattice: Lattice) async {
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

        let embedder = EmbeddingService()
        if await !embedder.isLoaded { await embedder.load() }
        let mt = MemoryTools(ref: lattice.sendableReference, embedder: embedder)
        self.memoryTools = mt
        self.session = LanguageModelSession(
            tools: EngramTools.readOnly(memoryTools: mt),
            instructions: """
                You are an Engram memory companion — a small robot that lives inside the \
                user's memory graph. You have access to a persistent memory system. Use \
                the recall tool to search for stored memories when the user asks about \
                them. Be helpful, concise, and friendly. Keep responses short (1-3 sentences).
                """
        )
    }

    func send(_ text: String) async {
        messages.append(ChatMessage(role: .user, text: text))
        isThinking = true
        defer { isThinking = false }
        guard let session else {
            messages.append(ChatMessage(role: .assistant, text: "Still initializing..."))
            return
        }
        do {
            let response = try await session.respond(to: text)
            messages.append(ChatMessage(role: .assistant, text: response.content))
        } catch {
            messages.append(ChatMessage(role: .assistant, text: "Error: \(error.localizedDescription)"))
        }
    }

    func reset() {
        messages.removeAll()
        session = nil
        memoryTools = nil
    }
}
