import SwiftUI
import EngramKit
import Lattice

#if canImport(FoundationModels)
import FoundationModels
import EngramFoundationModels
#endif

/// Manages on-device LLM chat sessions for the mascot companion.
///
/// Uses a two-stage pipeline to work around context window limitations:
/// - **Stage 1 (Retrieval)**: Session with read-only tools produces a `MemoryContext`
///   via constrained decoding. Tool call transcripts stay in this session's context.
/// - **Stage 2 (Response)**: Session with NO tools receives the user question + structured
///   context. Clean context window → better response quality.
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

    /// Read-only tools for retrieval sessions (created once, reused).
    private var readOnlyTools: [any Tool] = []
    /// Stage 2: persistent response session with no tools (keeps chat continuity).
    private var responseSession: LanguageModelSession?
    private var memoryTools: MemoryTools?

    private static let retrievalInstructions = """
        You are a memory retrieval assistant. Use the recall tool to search for \
        relevant memories based on the user's question. You may call recall multiple \
        times with different queries if needed. Return all relevant findings as \
        structured data. Do NOT generate a user-facing response.
        """

    private static let responseInstructions = """
        You are an Engram memory companion — a small robot that lives inside the \
        user's memory graph. You receive the user's question along with relevant \
        memories retrieved from their knowledge base. Be helpful, concise, and \
        friendly. Keep responses short (1-3 sentences). If no relevant memories \
        were found, say so honestly.
        """

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

        // Cache tools for per-message retrieval sessions
        self.readOnlyTools = EngramTools.readOnly(memoryTools: mt)

        // Stage 2: persistent response session (no tools, clean context)
        self.responseSession = LanguageModelSession(
            instructions: Self.responseInstructions
        )
    }

    func send(_ text: String) async {
        messages.append(ChatMessage(role: .user, text: text))
        isThinking = true
        defer { isThinking = false }

        guard let responseSession, !readOnlyTools.isEmpty else {
            messages.append(ChatMessage(role: .assistant, text: "Still initializing..."))
            return
        }

        do {
            // Stage 1: fresh retrieval session per message — no accumulated
            // tool call transcripts from prior turns eating context window
            let retrievalSession = LanguageModelSession(
                tools: readOnlyTools,
                instructions: Self.retrievalInstructions
            )
            let retrieval = try await retrievalSession.respond(
                to: text,
                generating: MemoryContext.self
            )
            let ctx = retrieval.content

            // Stage 2: generate response with clean context
            let contextBlock: String
            if ctx.memories.isEmpty {
                contextBlock = "No relevant memories found."
            } else {
                let items = ctx.memories.map {
                    "- [\($0.project)/\($0.topic)] \($0.content)"
                }.joined(separator: "\n")
                contextBlock = """
                    Relevant memories:
                    \(items)

                    Summary: \(ctx.summary)
                    """
            }

            let prompt = """
                User question: \(text)

                \(contextBlock)
                """
            let response = try await responseSession.respond(to: prompt)
            messages.append(ChatMessage(role: .assistant, text: response.content))
        } catch {
            messages.append(ChatMessage(
                role: .assistant,
                text: "Error: \(error.localizedDescription)"
            ))
        }
    }

    func reset() {
        messages.removeAll()
        readOnlyTools = []
        responseSession = nil
        memoryTools = nil
    }
}
