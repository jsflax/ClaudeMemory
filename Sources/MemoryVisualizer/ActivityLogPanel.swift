import SwiftUI
import ClaudeMemoryLib

struct ActivityLogPanel: View {
    let memories: [Memory]
    let colorMap: [String: Color]
    let onSelect: (Int64) -> Void

    @State private var knownIds: Set<Int64> = []
    @State private var glowingIds: Set<Int64> = []

    fileprivate static let relativeFormatter: RelativeDateTimeFormatter = {
        let f = RelativeDateTimeFormatter()
        f.unitsStyle = .abbreviated
        return f
    }()

    var body: some View {
        VStack(alignment: .trailing, spacing: 0) {
            HStack(spacing: 6) {
                Image(systemName: "clock.arrow.circlepath")
                    .font(.system(size: 10))
                Text("Activity")
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
            }
            .foregroundStyle(.white.opacity(0.5))
            .padding(.horizontal, 10)
            .padding(.vertical, 6)

            Divider().overlay(Color.white.opacity(0.15))

            ScrollView(.vertical, showsIndicators: true) {
                LazyVStack(alignment: .trailing, spacing: 2) {
                    let recent = memories.suffix(50).reversed()
                    ForEach(Array(recent), id: \.primaryKey) { memory in
                        let isGlowing = memory.primaryKey.map { glowingIds.contains($0) } ?? false
                        ActivityRow(
                            memory: memory,
                            colorMap: colorMap,
                            isGlowing: isGlowing,
                            onSelect: onSelect
                        )
                        .transition(.move(edge: .top).combined(with: .opacity))
                    }
                }
                .padding(.vertical, 4)
                .animation(.easeInOut(duration: 0.3), value: memories.count)
            }
        }
        .frame(width: 220)
        .frame(maxHeight: 300)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 10))
        .foregroundStyle(.white.opacity(0.7))
        .onAppear {
            knownIds = Set(memories.compactMap(\.primaryKey))
        }
        .onChange(of: memories.count) { _, _ in
            let currentIds = Set(memories.compactMap(\.primaryKey))
            let newIds = currentIds.subtracting(knownIds)
            knownIds = currentIds
            guard !newIds.isEmpty else { return }
            withAnimation(.easeInOut(duration: 0.3)) {
                glowingIds.formUnion(newIds)
            }
            Task { @MainActor in
                try? await Task.sleep(for: .seconds(1))
                withAnimation(.easeOut(duration: 0.6)) {
                    glowingIds.subtract(newIds)
                }
            }
        }
    }
}

private struct ActivityRow: View {
    let memory: Memory
    let colorMap: [String: Color]
    let isGlowing: Bool
    let onSelect: (Int64) -> Void

    var body: some View {
        Button {
            if let pk = memory.primaryKey {
                onSelect(pk)
            }
        } label: {
            HStack(spacing: 6) {
                Spacer(minLength: 0)
                Text(relativeTime)
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.35))
                    .frame(minWidth: 32, alignment: .trailing)
                Text(label)
                    .font(.system(size: 10, design: .monospaced))
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .foregroundStyle(isGlowing ? .cyan : .white.opacity(0.7))
                Circle()
                    .fill(GraphView.projectColor(for: memory.project, in: colorMap))
                    .frame(width: 6, height: 6)
                    .shadow(color: isGlowing ? .cyan.opacity(0.8) : .clear, radius: 4)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 3)
            .background(
                RoundedRectangle(cornerRadius: 4)
                    .fill(.cyan.opacity(isGlowing ? 0.1 : 0))
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .animation(.easeOut(duration: 0.6), value: isGlowing)
    }

    private var label: String {
        GraphView.extractLabel(content: memory.content, topic: memory.topic)
    }

    private var relativeTime: String {
        ActivityLogPanel.relativeFormatter.localizedString(for: memory.createdAt, relativeTo: Date())
    }
}
