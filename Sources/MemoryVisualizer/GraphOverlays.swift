import SwiftUI
import ClaudeMemoryLib

// MARK: - Stats Overlay

struct StatsOverlay: View {
    let nodes: [NodeInfo]
    let edgeData: [EdgeInfo]
    let totalMemories: Int
    let hiddenProjects: Set<String>
    let hiddenRelations: Set<String>
    let projects: [String]
    let allRelationCounts: [(key: String, value: Int)]
    let toggleProject: (String) -> Void
    let toggleRelation: (String) -> Void
    let colorMap: [String: Color]

    private var dbFileSize: String {
        let dbPath = ProcessInfo.processInfo.environment["CLAUDE_MEMORY_DB"]
            ?? NSHomeDirectory() + "/.claude/memory.sqlite"
        let fm = FileManager.default
        var total: Int64 = 0
        for path in [dbPath, dbPath + "-wal", dbPath + "-shm"] {
            if let attrs = try? fm.attributesOfItem(atPath: path),
               let size = attrs[.size] as? Int64 {
                total += size
            }
        }
        guard total > 0 else { return "—" }
        if total < 1024 { return "\(total) B" }
        let kb = Double(total) / 1024
        if kb < 1024 { return String(format: "%.1f KB", kb) }
        let mb = kb / 1024
        return String(format: "%.1f MB", mb)
    }

    var body: some View {
        VStack(alignment: .trailing, spacing: 6) {
            let visible = nodes.count
            if visible < totalMemories {
                Text("\(visible)/\(totalMemories) memories")
                    .font(.system(size: 13, weight: .medium, design: .monospaced))
            } else {
                Text("\(totalMemories) memories")
                    .font(.system(size: 13, weight: .medium, design: .monospaced))
            }
            Text("\(edgeData.count) edges")
                .font(.system(size: 13, weight: .medium, design: .monospaced))
            Text("db: \(dbFileSize)")
                .font(.system(size: 11, design: .monospaced))

            Divider().frame(width: 100).overlay(Color.white.opacity(0.2))

            // Project filters
            ForEach(projects, id: \.self) { project in
                Button {
                    toggleProject(project)
                } label: {
                    HStack(spacing: 6) {
                        Circle()
                            .fill(GraphView.projectColor(for: project, in: colorMap))
                            .frame(width: 8, height: 8)
                            .opacity(hiddenProjects.contains(project) ? 0.3 : 1.0)
                        Text(project)
                            .font(.system(size: 11, design: .monospaced))
                            .strikethrough(hiddenProjects.contains(project))
                    }
                }
                .buttonStyle(.plain)
            }

            // Edge type filters (use unfiltered counts so hidden types remain visible)
            if !allRelationCounts.isEmpty {
                Divider().frame(width: 100).overlay(Color.white.opacity(0.2))

                ForEach(allRelationCounts, id: \.key) { relation, count in
                    Button {
                        toggleRelation(relation)
                    } label: {
                        HStack(spacing: 6) {
                            Circle()
                                .fill(GraphCanvas.relationColors[relation] ?? .white)
                                .frame(width: 8, height: 8)
                                .opacity(hiddenRelations.contains(relation) ? 0.3 : 1.0)
                            Text("\(relation.replacingOccurrences(of: "_", with: " ")) (\(count))")
                                .font(.system(size: 11, design: .monospaced))
                                .strikethrough(hiddenRelations.contains(relation))
                        }
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        .foregroundStyle(.white.opacity(0.7))
        .padding(12)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 10))
    }
}

// MARK: - Minimap

struct MinimapView: View {
    let simulation: ForceSimulation
    let nodes: [NodeInfo]
    let viewport: ViewportState
    let viewportSize: CGSize
    let colorMap: [String: Color]

    @State private var frameCount: UInt64 = 0
    private let timer = Timer.publish(every: 1.0 / 60.0, on: .main, in: .common).autoconnect()
    private let minimapSize = CGSize(width: 160, height: 120)

    var body: some View {
        Canvas { context, size in
            let _ = frameCount
            let positions = simulation.positions
            guard !positions.isEmpty else { return }

            let bounds = computeBounds(positions: positions)
            let mapScale = min(size.width / bounds.width, size.height / bounds.height)

            // Draw nodes as dots
            for node in nodes {
                guard let pos = positions[node.id] else { continue }
                let mx = (pos.x - bounds.minX) * mapScale
                let my = (pos.y - bounds.minY) * mapScale
                let color = GraphView.projectColor(for: node.project, in: colorMap)
                let dotSize: CGFloat = node.isHub ? 4 : 2.5
                let rect = CGRect(x: mx - dotSize / 2, y: my - dotSize / 2, width: dotSize, height: dotSize)
                context.fill(Circle().path(in: rect), with: .color(color.opacity(0.8)))
            }

            // Draw viewport rectangle
            let vpWorldX = -viewport.offset.x / viewport.scale
            let vpWorldY = -viewport.offset.y / viewport.scale
            let vpWorldW = viewportSize.width / viewport.scale
            let vpWorldH = viewportSize.height / viewport.scale

            let vpRect = CGRect(
                x: (vpWorldX - bounds.minX) * mapScale,
                y: (vpWorldY - bounds.minY) * mapScale,
                width: vpWorldW * mapScale,
                height: vpWorldH * mapScale
            )
            context.stroke(
                Rectangle().path(in: vpRect),
                with: .color(.white.opacity(0.5)),
                lineWidth: 1
            )
        }
        .frame(width: minimapSize.width, height: minimapSize.height)
        .background(Color(red: 0.051, green: 0.067, blue: 0.09).opacity(0.85))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .strokeBorder(.white.opacity(0.15), lineWidth: 1)
        )
        .onReceive(timer) { _ in frameCount &+= 1 }
        .onTapGesture { location in
            let positions = simulation.positions
            guard !positions.isEmpty else { return }

            let bounds = computeBounds(positions: positions)
            let mapScale = min(minimapSize.width / bounds.width, minimapSize.height / bounds.height)

            let worldX = location.x / mapScale + bounds.minX
            let worldY = location.y / mapScale + bounds.minY

            withAnimation(.easeInOut(duration: 0.3)) {
                viewport.offset = CGPoint(
                    x: viewportSize.width / 2 - worldX * viewport.scale,
                    y: viewportSize.height / 2 - worldY * viewport.scale
                )
            }
        }
    }

    private func computeBounds(positions: [Int64: CGPoint]) -> (minX: CGFloat, minY: CGFloat, width: CGFloat, height: CGFloat) {
        var minX: CGFloat = .greatestFiniteMagnitude
        var minY: CGFloat = .greatestFiniteMagnitude
        var maxX: CGFloat = -.greatestFiniteMagnitude
        var maxY: CGFloat = -.greatestFiniteMagnitude
        for (_, pos) in positions {
            minX = min(minX, pos.x); minY = min(minY, pos.y)
            maxX = max(maxX, pos.x); maxY = max(maxY, pos.y)
        }
        let padding: CGFloat = 50
        return (minX - padding, minY - padding, max(maxX - minX + padding * 2, 1), max(maxY - minY + padding * 2, 1))
    }
}

// MARK: - Time Slider

struct TimeSliderBar: View {
    let earliestDate: Date
    let latestDate: Date
    @Binding var sliderDate: Date?
    @Binding var isPlaying: Bool

    @State private var sliderValue: Double = 1.0

    private var timeRange: TimeInterval {
        max(latestDate.timeIntervalSince(earliestDate), 1)
    }

    var body: some View {
        HStack(spacing: 12) {
            Button {
                if isPlaying {
                    isPlaying = false
                } else {
                    if sliderValue >= 1.0 { sliderValue = 0.0 }
                    isPlaying = true
                }
            } label: {
                Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                    .font(.system(size: 12))
                    .foregroundStyle(.white.opacity(0.7))
            }
            .buttonStyle(.plain)

            Text(dateLabel)
                .font(.system(size: 10, weight: .medium, design: .monospaced))
                .foregroundStyle(.white.opacity(0.5))
                .frame(width: 80, alignment: .leading)

            Slider(value: $sliderValue, in: 0...1)
                .tint(.cyan.opacity(0.6))
                .onChange(of: sliderValue) { _, newVal in
                    if newVal >= 1.0 {
                        sliderDate = nil
                        isPlaying = false
                    } else {
                        let interval = newVal * timeRange
                        sliderDate = earliestDate.addingTimeInterval(interval)
                    }
                }

            Button {
                sliderValue = 1.0
                sliderDate = nil
                isPlaying = false
            } label: {
                Text("All")
                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.5))
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(Color(red: 0.051, green: 0.067, blue: 0.09).opacity(0.9))
        .task(id: isPlaying) {
            guard isPlaying else { return }
            while !Task.isCancelled && isPlaying && sliderValue < 1.0 {
                try? await Task.sleep(for: .milliseconds(50))
                sliderValue = min(1.0, sliderValue + 0.005)
            }
            if sliderValue >= 1.0 {
                isPlaying = false
                sliderDate = nil
            }
        }
    }

    private var dateLabel: String {
        if let date = sliderDate {
            let formatter = DateFormatter()
            formatter.dateStyle = .short
            return formatter.string(from: date)
        }
        return "All time"
    }
}
