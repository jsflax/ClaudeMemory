import SwiftUI

/// A log file source displayed in the logs tab.
struct LogSource: Identifiable {
    let id: String
    let label: String
    let path: String
    let icon: String
    let color: Color
}

/// Parsed log entry from a log file.
struct LogEntry: Identifiable {
    let id: Int
    let timestamp: Date?
    let source: String
    let message: String
    let raw: String
}

@MainActor @Observable
final class LogsStore {
    var entries: [LogEntry] = []
    var selectedSource: String? = nil
    var searchText = ""
    var autoScroll = true

    private var fileWatcher: DispatchSourceFileSystemObject?

    static let sources: [LogSource] = [
        LogSource(
            id: "memory",
            label: "MCP",
            path: NSHomeDirectory() + "/.claude/memory.log",
            icon: "server.rack",
            color: .green
        ),
        LogSource(
            id: "hooks",
            label: "Hooks",
            path: NSHomeDirectory() + "/.claude/hooks.log",
            icon: "arrow.triangle.branch",
            color: .cyan
        ),
        LogSource(
            id: "session-learner",
            label: "Learner",
            path: NSHomeDirectory() + "/.claude/session-learner.log",
            icon: "brain",
            color: .purple
        ),
        LogSource(
            id: "maintenance",
            label: "Maintenance",
            path: NSHomeDirectory() + "/.claude/memory-maintenance.log",
            icon: "wrench.and.screwdriver",
            color: .orange
        ),
    ]

    var filteredEntries: [LogEntry] {
        var result = entries
        if let source = selectedSource {
            result = result.filter { $0.source == source }
        }
        if !searchText.isEmpty {
            let query = searchText.lowercased()
            result = result.filter { $0.raw.lowercased().contains(query) }
        }
        return result
    }

    func loadLogs() {
        var all: [LogEntry] = []
        var counter = 0

        for source in Self.sources {
            guard let data = FileManager.default.contents(atPath: source.path),
                  let content = String(data: data, encoding: .utf8) else { continue }

            let lines = content.components(separatedBy: .newlines)
            let recent = lines.suffix(500)
            for line in recent {
                let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty else { continue }
                let parsed = Self.parseLine(trimmed, source: source.id)
                all.append(LogEntry(id: counter, timestamp: parsed.timestamp, source: source.id, message: parsed.message, raw: trimmed))
                counter += 1
            }
        }

        all.sort { a, b in
            guard let ta = a.timestamp else { return false }
            guard let tb = b.timestamp else { return true }
            return ta < tb
        }

        entries = all.enumerated().map { i, e in
            LogEntry(id: i, timestamp: e.timestamp, source: e.source, message: e.message, raw: e.raw)
        }
    }

    func startWatching() {
        guard fileWatcher == nil else { return }
        fileWatcher = Self.makeWatcher { [weak self] in
            self?.loadLogs()
        }
    }

    /// Creates the DispatchSource outside of @MainActor context so that
    /// the event/cancel handler closures don't inherit MainActor isolation.
    /// GCD runs these on the utility queue — an inherited @MainActor assertion
    /// would crash at runtime (swift_task_isCurrentExecutor).
    nonisolated private static func makeWatcher(
        onChange: @escaping @MainActor @Sendable () -> Void
    ) -> DispatchSourceFileSystemObject? {
        let dirPath = NSHomeDirectory() + "/.claude"
        let fd = open(dirPath, O_EVTONLY)
        guard fd >= 0 else { return nil }

        let source = DispatchSource.makeFileSystemObjectSource(
            fileDescriptor: fd,
            eventMask: .write,
            queue: .global(qos: .utility)
        )
        source.setEventHandler {
            Task { @MainActor in
                onChange()
            }
        }
        source.setCancelHandler {
            close(fd)
        }
        source.resume()
        return source
    }

    func stopWatching() {
        fileWatcher?.cancel()
        fileWatcher = nil
    }

    // MARK: - Parsing

    private static func parseLine(_ line: String, source: String) -> (timestamp: Date?, message: String) {
        let isoFormatter = ISO8601DateFormatter()

        if source == "memory" {
            let mcpPattern = /^\[claude-memory\]\s+(\d{4}-\d{2}-\d{2}T[\d:]+Z)\s+(.*)/
            if let match = line.wholeMatch(of: mcpPattern) {
                let ts = isoFormatter.date(from: String(match.1))
                return (ts, String(match.2))
            }
        }

        if source == "hooks" {
            let isoPattern = /^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s+\[memory-hooks\]\s*(.*)/
            if let match = line.wholeMatch(of: isoPattern) {
                let ts = isoFormatter.date(from: String(match.1))
                return (ts, String(match.2))
            }
        }

        let startPattern = /^=+\s*started at '([^']+)'\s*=+$/
        if let match = line.wholeMatch(of: startPattern) {
            let ts = isoFormatter.date(from: String(match.1))
            return (ts, "--- Session started ---")
        }

        let genericPattern = /^(\d{4}-\d{2}-\d{2}T[\d:]+Z?)\s+(.*)/
        if let match = line.wholeMatch(of: genericPattern) {
            let ts = isoFormatter.date(from: String(match.1))
            return (ts, String(match.2))
        }

        return (nil, line)
    }
}

/// Content view for the Logs sidebar tab.
struct LogsContentView: View {
    @State private var store = LogsStore()

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            sourceFilter

            // Search
            HStack(spacing: 6) {
                Image(systemName: "magnifyingglass")
                    .font(.system(size: 10))
                    .foregroundStyle(.white.opacity(0.3))
                TextField("Filter logs…", text: $store.searchText)
                    .textFieldStyle(.plain)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.8))
                if !store.searchText.isEmpty {
                    Button { store.searchText = "" } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 10))
                            .foregroundStyle(.white.opacity(0.3))
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(Color(red: 0.08, green: 0.1, blue: 0.14))
                    .overlay(
                        RoundedRectangle(cornerRadius: 6)
                            .strokeBorder(.white.opacity(0.1), lineWidth: 1)
                    )
            )

            // Log entries
            ScrollViewReader { proxy in
                ScrollView(.vertical, showsIndicators: true) {
                    LazyVStack(alignment: .leading, spacing: 1) {
                        let filtered = store.filteredEntries
                        if filtered.isEmpty {
                            emptyState
                        } else {
                            ForEach(filtered) { entry in
                                logRow(entry)
                                    .id(entry.id)
                            }
                        }
                    }
                    .padding(.vertical, 4)
                }
                .onChange(of: store.entries.count) { _, _ in
                    if store.autoScroll, let last = store.filteredEntries.last {
                        withAnimation(.easeOut(duration: 0.2)) {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    }
                }
            }

            // Bottom toolbar
            HStack(spacing: 8) {
                Button {
                    store.autoScroll.toggle()
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: store.autoScroll ? "arrow.down.to.line" : "arrow.down.to.line.compact")
                            .font(.system(size: 9))
                        Text(store.autoScroll ? "Auto-scroll" : "Paused")
                            .font(.system(size: 9, design: .monospaced))
                    }
                    .foregroundStyle(store.autoScroll ? .cyan.opacity(0.8) : .white.opacity(0.4))
                }
                .buttonStyle(.plain)

                Spacer()

                Text("\(store.filteredEntries.count) lines")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.3))

                Button {
                    store.loadLogs()
                } label: {
                    Image(systemName: "arrow.clockwise")
                        .font(.system(size: 10))
                        .foregroundStyle(.white.opacity(0.4))
                }
                .buttonStyle(.plain)
                .help("Reload logs")
            }
        }
        .onAppear {
            store.loadLogs()
            store.startWatching()
        }
        .onDisappear {
            store.stopWatching()
        }
    }

    // MARK: - Source Filter

    private var sourceFilter: some View {
        HStack(spacing: 4) {
            filterChip(label: "All", id: nil)
            ForEach(LogsStore.sources) { source in
                filterChip(label: source.label, id: source.id, icon: source.icon, color: source.color)
            }
        }
        .fixedSize(horizontal: false, vertical: true)
    }

    private func filterChip(label: String, id: String?, icon: String? = nil, color: Color = .white) -> some View {
        let selected = store.selectedSource == id
        return Button {
            store.selectedSource = id
        } label: {
            HStack(spacing: 3) {
                if let icon {
                    Image(systemName: icon)
                        .font(.system(size: 8))
                }
                Text(label)
                    .font(.system(size: 9, weight: selected ? .semibold : .regular, design: .monospaced))
            }
            .foregroundStyle(selected ? color.opacity(0.9) : .white.opacity(0.4))
            .padding(.horizontal, 6)
            .padding(.vertical, 4)
            .background(
                RoundedRectangle(cornerRadius: 4)
                    .fill(selected ? color.opacity(0.15) : .clear)
            )
        }
        .buttonStyle(.plain)
    }

    // MARK: - Log Row

    private func logRow(_ entry: LogEntry) -> some View {
        let source = LogsStore.sources.first { $0.id == entry.source }
        let color = source?.color ?? .white
        return HStack(alignment: .top, spacing: 6) {
            if let ts = entry.timestamp {
                Text(Self.timeFormatter.string(from: ts))
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.25))
                    .frame(width: 42, alignment: .trailing)
            }
            Circle()
                .fill(color)
                .frame(width: 4, height: 4)
                .padding(.top, 4)
            Text(entry.message)
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(.white.opacity(0.7))
                .lineLimit(3)
                .textSelection(.enabled)
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "doc.text")
                .font(.system(size: 20))
                .foregroundStyle(.white.opacity(0.2))
            Text("No log entries")
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(.white.opacity(0.3))
            Text("Logs appear when hooks or\nbackground tasks run.")
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(.white.opacity(0.2))
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 40)
    }

    // MARK: - Formatters

    private static let timeFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss"
        return f
    }()
}
