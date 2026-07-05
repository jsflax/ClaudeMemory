import SwiftUI

#if canImport(FoundationModels)
import FoundationModels
#endif

/// Chat panel overlay for the mascot companion.
/// Holographic/cyan theme matching the mascot's visual style.
@available(macOS 26.0, *)
struct MascotChatPanel: View {
    @Bindable var engine: MascotChatEngine
    var onClose: () -> Void
    @State private var inputText = ""
    @FocusState private var inputFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Image(systemName: "brain.head.profile")
                    .foregroundStyle(.cyan)
                Text("Engram")
                    .font(.system(size: 14, weight: .semibold, design: .monospaced))
                    .foregroundStyle(.cyan)
                Spacer()
                Button(action: onClose) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 16))
                }
                .buttonStyle(.plain)
                .foregroundStyle(.white.opacity(0.5))
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)

            Rectangle()
                .fill(.cyan.opacity(0.2))
                .frame(height: 1)

            // Messages
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 8) {
                        ForEach(engine.messages) { msg in
                            chatBubble(msg)
                                .id(msg.id)
                        }
                        if engine.isThinking {
                            thinkingIndicator
                        }
                    }
                    .padding(12)
                }
                .onChange(of: engine.messages.count) {
                    withAnimation(.easeOut(duration: 0.2)) {
                        proxy.scrollTo(engine.messages.last?.id, anchor: .bottom)
                    }
                }
            }

            Rectangle()
                .fill(.cyan.opacity(0.2))
                .frame(height: 1)

            // Input
            HStack(spacing: 8) {
                TextField("Ask about your memories...", text: $inputText)
                    .textFieldStyle(.plain)
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.9))
                    .focused($inputFocused)
                    .onSubmit { sendMessage() }

                Button(action: sendMessage) {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.system(size: 18))
                        .foregroundStyle(inputText.trimmingCharacters(in: .whitespaces).isEmpty || engine.isThinking
                            ? .white.opacity(0.2)
                            : .cyan)
                }
                .buttonStyle(.plain)
                .disabled(inputText.trimmingCharacters(in: .whitespaces).isEmpty || engine.isThinking)
            }
            .padding(12)
        }
        .frame(width: 320, height: 400)
        .background(.black.opacity(0.75))
        .background(.ultraThinMaterial.opacity(0.3))
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(.cyan.opacity(0.3), lineWidth: 1)
        )
        .shadow(color: .cyan.opacity(0.1), radius: 20)
        .onAppear { inputFocused = true }
    }

    @ViewBuilder
    private func chatBubble(_ message: MascotChatEngine.ChatMessage) -> some View {
        let isUser = message.role == .user
        HStack {
            if isUser { Spacer(minLength: 40) }
            Text(message.text)
                .font(.system(size: 12, design: .monospaced))
                .foregroundStyle(isUser ? .white.opacity(0.9) : .cyan.opacity(0.9))
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(
                    RoundedRectangle(cornerRadius: 8)
                        .fill(isUser ? .white.opacity(0.1) : .cyan.opacity(0.08))
                )
            if !isUser { Spacer(minLength: 40) }
        }
    }

    private var thinkingIndicator: some View {
        HStack(spacing: 4) {
            ForEach(0..<3, id: \.self) { i in
                Circle()
                    .fill(.cyan.opacity(0.6))
                    .frame(width: 6, height: 6)
                    .scaleEffect(thinkingScale(index: i))
                    .animation(
                        .easeInOut(duration: 0.5)
                            .repeatForever()
                            .delay(Double(i) * 0.15),
                        value: engine.isThinking
                    )
            }
        }
        .padding(.vertical, 8)
        .padding(.horizontal, 10)
    }

    private func thinkingScale(index: Int) -> CGFloat {
        engine.isThinking ? 1.2 : 0.6
    }

    private func sendMessage() {
        let text = inputText.trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty else { return }
        inputText = ""
        Task { await engine.send(text) }
    }
}
