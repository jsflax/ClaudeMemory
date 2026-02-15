import SwiftUI
import Lattice
import ClaudeMemoryLib

/// Detail panel showing a selected memory's content with typewriter effect.
/// Uses @Bindable for live observation — property changes propagate immediately.
struct MemoryDetailPanel: View {
    @Bindable var memory: Memory
    let connectedCount: Int
    let onClose: () -> Void

    @State private var typingProgress: Int = 0
    private var content: String { memory.content }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            HStack {
                Circle()
                    .fill(GraphView.projectColor(for: memory.project))
                    .frame(width: 10, height: 10)
                Text(memory.project)
                    .font(.system(size: 11, weight: .semibold, design: .monospaced))
                    .foregroundStyle(GraphView.projectColor(for: memory.project))
                Text("·")
                    .foregroundStyle(.white.opacity(0.3))
                Text(memory.topic)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.5))
                Spacer()
                Button(action: onClose) {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.white.opacity(0.4))
                        .font(.system(size: 16))
                }
                .buttonStyle(.plain)
            }
            .padding(.bottom, 12)

            // Content (typewriter effect)
            ScrollView {
                Text(String(content.prefix(typingProgress)))
                    .font(.system(size: 13, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.9))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .contentShape(Rectangle())
            }

            Spacer(minLength: 12)

            // Footer
            VStack(alignment: .leading, spacing: 4) {
                if connectedCount > 0 {
                    Divider().overlay(Color.white.opacity(0.1))
                    Text("\(connectedCount) connection\(connectedCount == 1 ? "" : "s")")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.4))
                }
                HStack {
                    if memory.importance > 0 {
                        Text("importance: \(memory.importance)")
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(.white.opacity(0.4))
                    }
                    Spacer()
                    Text(memory.createdAt, style: .date)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.3))
                }
            }
        }
        .padding(20)
        .background {
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(red: 0.08, green: 0.1, blue: 0.14))
                .overlay {
                    RoundedRectangle(cornerRadius: 16)
                        .strokeBorder(
                            GraphView.projectColor(for: memory.project).opacity(0.3),
                            lineWidth: 1
                        )
                }
                .shadow(color: .black.opacity(0.5), radius: 30, y: 10)
        }
        .task(id: memory.primaryKey) {
            typingProgress = 0
            while !Task.isCancelled && typingProgress < content.count {
                try? await Task.sleep(for: .milliseconds(10))
                typingProgress += 2
            }
            typingProgress = content.count
        }
    }
}
