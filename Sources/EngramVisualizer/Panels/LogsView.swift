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

/// Content view for the Logs sidebar tab.
struct LogsContentView: View {
    @State private var entries: [LogEntry] = []
    @State private var selectedSource: String? = nil
    @State private var searchText = ""
    @State private var autoScroll = true
    @State private var isLoading = false
    @State private var fileWatcher: DispatchSourceFileSystemObject?

    private static let sources: [LogSource] = [
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

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Source filter chips
            sourceFilter

            // Search
            HStack(spacing: 6) {
                Image(systemName: "magnifyingglass")
                    .font(.system(size: 10))
                    .foregroundStyle(.white.opacity(0.3))
                TextField("Filter logs…", text: $searchText)
                    .textFieldStyle(.plain)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.8))
                if !searchText.isEmpty {
                    Button { searchText = "" } label: {
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
                        let filtered = filteredEntries
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
                .onChange(of: entries.count) { _, _ in
                    if autoScroll, let last = filteredEntries.last {
                        withAnimation(.easeOut(duration: 0.2)) {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    }
                }
            }

            // Bottom toolbar
            HStack(spacing: 8) {
                Button {
                    autoScroll.toggle()
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: autoScroll ? "arrow.down.to.line" : "arrow.down.to.line.compact")
                            .font(.system(size: 9))
                        Text(autoScroll ? "Auto-scroll" : "Paused")
                            .font(.system(size: 9, design: .monospaced))
                    }
                    .foregroundStyle(autoScroll ? .cyan.opacity(0.8) : .white.opacity(0.4))
                }
                .buttonStyle(.plain)

                Spacer()

                Text("\(filteredEntries.count) lines")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.3))

                Button {
                    loadLogs()
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
            loadLogs()
            startWatching()
        }
        .onDisappear {
            stopWatching()
        }
    }

    // MARK: - Source Filter

    private var sourceFilter: some View {
        HStack(spacing: 4) {
            filterChip(label: "All", id: nil)
            ForEach(Self.sources) { source in
                filterChip(label: source.label, id: source.id, icon: source.icon, color: source.color)
            }
        }
        .fixedSize(horizontal: false, vertical: true)
    }

    private func filterChip(label: String, id: String?, icon: String? = nil, color: Color = .white) -> some View {
        let selected = selectedSource == id
        return Button {
            selectedSource = id
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
        let source = Self.sources.first { $0.id == entry.source }
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

    // MARK: - Filtering

    private var filteredEntries: [LogEntry] {
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

    // MARK: - Loading

    private func loadLogs() {
        isLoading = true
        var all: [LogEntry] = []
        var counter = 0

        for source in Self.sources {
            guard let data = FileManager.default.contents(atPath: source.path),
                  let content = String(data: data, encoding: .utf8) else { continue }

            let lines = content.components(separatedBy: .newlines)
            // Take last 500 lines per file to avoid memory issues
            let recent = lines.suffix(500)
            for line in recent {
                let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty else { continue }
                let parsed = parseLine(trimmed, source: source.id)
                all.append(LogEntry(id: counter, timestamp: parsed.timestamp, source: source.id, message: parsed.message, raw: trimmed))
                counter += 1
            }
        }

        // Sort by timestamp (entries without timestamps go at the end of their file group)
        all.sort { a, b in
            guard let ta = a.timestamp else { return false }
            guard let tb = b.timestamp else { return true }
            return ta < tb
        }

        // Re-assign sequential IDs after sorting
        entries = all.enumerated().map { i, e in
            LogEntry(id: i, timestamp: e.timestamp, source: e.source, message: e.message, raw: e.raw)
        }
        isLoading = false
    }

    // MARK: - Parsing

    private func parseLine(_ line: String, source: String) -> (timestamp: Date?, message: String) {
        // memory.log format: "[claude-memory] 2026-03-07T16:24:52Z message"
        if source == "memory" {
            let mcpPattern = /^\[claude-memory\]\s+(\d{4}-\d{2}-\d{2}T[\d:]+Z)\s+(.*)/
            if let match = line.wholeMatch(of: mcpPattern) {
                let ts = ISO8601DateFormatter().date(from: String(match.1))
                return (ts, String(match.2))
            }
        }

        // hooks.log format: "2026-03-07T16:24:52Z [memory-hooks] message"
        if source == "hooks" {
            let isoPattern = /^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s+\[memory-hooks\]\s*(.*)/
            if let match = line.wholeMatch(of: isoPattern) {
                let ts = ISO8601DateFormatter().date(from: String(match.1))
                return (ts, String(match.2))
            }
        }

        // session-learner / maintenance: "===== started at '2026-03-07...' ====="
        let startPattern = /^=+\s*started at '([^']+)'\s*=+$/
        if let match = line.wholeMatch(of: startPattern) {
            let ts = ISO8601DateFormatter().date(from: String(match.1))
            return (ts, "--- Session started ---")
        }

        // Try to extract a leading ISO timestamp from any line
        let genericPattern = /^(\d{4}-\d{2}-\d{2}T[\d:]+Z?)\s+(.*)/
        if let match = line.wholeMatch(of: genericPattern) {
            let ts = ISO8601DateFormatter().date(from: String(match.1))
            return (ts, String(match.2))
        }

        return (nil, line)
    }

    // MARK: - File Watching

    private func startWatching() {
        // Watch the directory for changes to any log file
        let dirPath = NSHomeDirectory() + "/.claude"
        let fd = open(dirPath, O_EVTONLY)
        guard fd >= 0 else { return }

        let source = DispatchSource.makeFileSystemObjectSource(
            fileDescriptor: fd,
            eventMask: .write,
            queue: .global(qos: .utility)
        )
        source.setEventHandler { [self] in
            DispatchQueue.main.async {
                self.loadLogs()
            }
        }
        source.setCancelHandler {
            close(fd)
        }
        source.resume()
        fileWatcher = source
    }

    private func stopWatching() {
        fileWatcher?.cancel()
        fileWatcher = nil
    }

    // MARK: - Formatters

    private static let timeFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss"
        return f
    }()
}
