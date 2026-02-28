import SwiftUI
import Lattice
import EngramKit

struct MenuBarActivityFeed: View {
    @LatticeQuery(sort: \Memory.createdAt, order: .reverse)
    private var memories: TableResults<Memory>

    @State private var projectColors: [String: Color] = [:]

    var body: some View {
        TimelineView(.periodic(from: .now, by: 1)) { context in
            VStack(spacing: 0) {
                // Header
                HStack(spacing: 8) {
                    Image(systemName: "brain.head.profile")
                        .font(.system(size: 14))
                    Text("Engram")
                        .font(.system(size: 13, weight: .semibold, design: .monospaced))
                    Spacer()
                    Text("\(memories.count)")
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 10)

                Divider()

                // Open Engram button
                Button {
                    NSApp.activate(ignoringOtherApps: true)
                    NSApp.windows.first { $0.title == "Engram" }?.makeKeyAndOrderFront(nil)
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "macwindow")
                            .font(.system(size: 11))
                        Text("Open Engram")
                            .font(.system(size: 12, weight: .medium))
                        Spacer()
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .background(.quaternary.opacity(0.5))

                Divider()

                // Recent memories
                ScrollView(.vertical, showsIndicators: true) {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        let recent = Array(memories.prefix(20))
                        ForEach(recent, id: \.primaryKey) { memory in
                            MenuBarMemoryRow(
                                memory: memory,
                                color: projectColors[memory.project] ?? .gray,
                                now: context.date
                            )
                        }
                    }
                    .padding(.vertical, 4)
                }
            }
            .frame(width: 300, height: 400)
        }
        .onChange(of: memories.count, initial: true) {
            rebuildColorMap()
        }
    }

    private func rebuildColorMap() {
        var projects = Set<String>()
        for m in memories { projects.insert(m.project) }
        let sorted = projects.sorted()
        var map: [String: Color] = [:]
        for (i, p) in sorted.enumerated() {
            map[p] = GraphView.goldenAngleColor(at: i)
        }
        projectColors = map
    }
}

private struct MenuBarMemoryRow: View {
    let memory: Memory
    let color: Color
    let now: Date

    private static let relativeFormatter: RelativeDateTimeFormatter = {
        let f = RelativeDateTimeFormatter()
        f.unitsStyle = .abbreviated
        return f
    }()

    var body: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(color)
                .frame(width: 6, height: 6)
            Text(extractLabel(content: memory.content, topic: memory.topic))
                .font(.system(size: 11, design: .monospaced))
                .lineLimit(1)
                .truncationMode(.tail)
            Spacer(minLength: 0)
            Text(Self.relativeFormatter.localizedString(for: memory.createdAt, relativeTo: now))
                .font(.system(size: 9, design: .monospaced))
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 4)
    }
}
