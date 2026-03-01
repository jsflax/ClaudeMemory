import EngramModels
import Lattice
import SwiftUI
import UserNotifications

// MARK: - Sidebar (hover-to-peek, click-to-pin)

struct SidebarView: View {
    enum Tab: String, CaseIterable {
        case visualizer = "Graph"
        case settings = "Settings"
        case account = "Account"
    }

    @Environment(VisualizerConfig.self) private var config
    @Environment(\.lattice) private var lattice
    @Environment(SyncManager.self) var syncManager
    @State private var selectedTab: Tab = .visualizer
    @State private var isCompacting = false

    // Visualizer tab
    let projects: [String]
    let colorMap: [String: Color]
    let allRelationCounts: [(key: String, value: Int)]
    let visibleMemoryCount: Int
    let visibleEdgeCount: Int
    let totalMemories: Int
    let projectionState: ProjectionState
    let toggleProject: (String) -> Void
    let toggleRelation: (String) -> Void
    let switchLayoutMode: (LayoutMode) -> Void
    let switchDimensionMode: (DimensionMode) -> Void
    let driveToProject: ((String) -> Void)?

    // Account tab
    @Bindable var accountService: AccountService
    @LatticeQuery<SyncConfig>(sort: \SyncConfig.project) var syncConfigs
    @State var email = ""
    @State var password = ""
    @State var isRegistering = false

    var body: some View {
        VStack(spacing: 0) {
            Spacer().frame(height: 28) // clear traffic light buttons
            tabBar
            Divider().overlay(Color.white.opacity(0.08))

            ScrollView(.vertical, showsIndicators: false) {
                Group {
                    switch selectedTab {
                    case .visualizer: visualizerContent
                    case .settings: settingsContent
                    case .account: accountContent
                    }
                }
                .padding(16)
            }
        }
        .frame(width: 260)
        .frame(maxHeight: .infinity)
        .background(Color(red: 0.055, green: 0.07, blue: 0.095).opacity(0.98))
        .background(.ultraThinMaterial)
        .clipShape(
            UnevenRoundedRectangle(
                topLeadingRadius: 0, bottomLeadingRadius: 0,
                bottomTrailingRadius: 10, topTrailingRadius: 10
            )
        )
        .overlay(
            UnevenRoundedRectangle(
                topLeadingRadius: 0, bottomLeadingRadius: 0,
                bottomTrailingRadius: 10, topTrailingRadius: 10
            )
            .strokeBorder(.white.opacity(0.08), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.5), radius: 20, x: 5)
    }

    // MARK: - Tab Bar

    private var tabBar: some View {
        HStack(spacing: 2) {
            ForEach(Tab.allCases, id: \.self) { tab in
                Button {
                    selectedTab = tab
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: tabIcon(tab))
                            .font(.system(size: 10))
                        Text(tab.rawValue)
                            .font(.system(size: 11, weight: selectedTab == tab ? .semibold : .regular, design: .monospaced))
                    }
                    .foregroundStyle(.white.opacity(selectedTab == tab ? 0.9 : 0.4))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 7)
                    .background(
                        RoundedRectangle(cornerRadius: 5)
                            .fill(selectedTab == tab ? .white.opacity(0.1) : .clear)
                    )
                }
                .buttonStyle(.plain)
            }
        }
        .padding(6)
    }

    private func tabIcon(_ tab: Tab) -> String {
        switch tab {
        case .visualizer: "cube.transparent"
        case .settings: "gearshape"
        case .account: "person.circle"
        }
    }

    // MARK: - Section Helper

    func section(_ title: String, @ViewBuilder content: () -> some View) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title.uppercased())
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .foregroundStyle(.white.opacity(0.3))
                .tracking(1.2)
            content()
        }
    }

    // MARK: - Visualizer Tab

    @ViewBuilder
    private var visualizerContent: some View {
        VStack(alignment: .leading, spacing: 24) {
            section("Stats") {
                VStack(alignment: .leading, spacing: 6) {
                    if visibleMemoryCount < totalMemories {
                        statRow("Memories", value: "\(visibleMemoryCount)/\(totalMemories)")
                    } else {
                        statRow("Memories", value: "\(totalMemories)")
                    }
                    statRow("Edges", value: "\(visibleEdgeCount)")
                    statRow("Database", value: dbFileSize)
                }
            }

            section("Layout") {
                VStack(alignment: .leading, spacing: 10) {
                    layoutModePicker
                    dimensionModePicker

                    if config.layoutMode == .embedding && config.dimensionMode == .twoD && projectionState == .ready {
                        toggleRow("Knowledge Voids", icon: "sparkles", isOn: config.showVoids) {
                            config.showVoids.toggle()
                        }
                    }
                }
            }

            section("Projects") {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(projects, id: \.self) { project in
                        projectRow(project)
                    }
                }
            }

            if !allRelationCounts.isEmpty {
                section("Relations") {
                    VStack(alignment: .leading, spacing: 2) {
                        ForEach(allRelationCounts, id: \.key) { relation, count in
                            relationRow(relation, count: count)
                        }
                    }
                }
            }
        }
    }

    // MARK: - Layout Controls

    private var layoutModePicker: some View {
        HStack(spacing: 0) {
            ForEach(LayoutMode.allCases, id: \.rawValue) { mode in
                Button { switchLayoutMode(mode) } label: {
                    HStack(spacing: 4) {
                        if mode == .embedding, case .computing(let progress) = projectionState {
                            ZStack {
                                Circle().stroke(.white.opacity(0.15), lineWidth: 1.5)
                                Circle()
                                    .trim(from: 0, to: progress)
                                    .stroke(.cyan.opacity(0.8), style: StrokeStyle(lineWidth: 1.5, lineCap: .round))
                                    .rotationEffect(.degrees(-90))
                            }
                            .frame(width: 10, height: 10)
                        }
                        Text(mode.rawValue)
                            .font(.system(size: 11, weight: config.layoutMode == mode ? .semibold : .regular, design: .monospaced))
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 6)
                    .background(
                        RoundedRectangle(cornerRadius: 5)
                            .fill(config.layoutMode == mode ? .white.opacity(0.12) : .clear)
                    )
                }
                .buttonStyle(.plain)
                .foregroundStyle(.white.opacity(config.layoutMode == mode ? 0.9 : 0.4))
            }
        }
        .padding(2)
        .background(
            RoundedRectangle(cornerRadius: 7)
                .fill(Color(red: 0.08, green: 0.1, blue: 0.14))
                .overlay(
                    RoundedRectangle(cornerRadius: 7)
                        .strokeBorder(.white.opacity(0.1), lineWidth: 1)
                )
        )
    }

    private var dimensionModePicker: some View {
        HStack(spacing: 0) {
            ForEach(DimensionMode.allCases, id: \.rawValue) { mode in
                Button { switchDimensionMode(mode) } label: {
                    HStack(spacing: 4) {
                        Image(systemName: mode == .twoD ? "square" : "cube")
                            .font(.system(size: 10))
                        Text(mode.rawValue)
                            .font(.system(size: 11, weight: config.dimensionMode == mode ? .semibold : .regular, design: .monospaced))
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 6)
                    .background(
                        RoundedRectangle(cornerRadius: 5)
                            .fill(config.dimensionMode == mode ? .white.opacity(0.12) : .clear)
                    )
                }
                .buttonStyle(.plain)
                .foregroundStyle(.white.opacity(config.dimensionMode == mode ? 0.9 : 0.4))
            }
        }
        .padding(2)
        .background(
            RoundedRectangle(cornerRadius: 7)
                .fill(Color(red: 0.08, green: 0.1, blue: 0.14))
                .overlay(
                    RoundedRectangle(cornerRadius: 7)
                        .strokeBorder(.white.opacity(0.1), lineWidth: 1)
                )
        )
    }

    // MARK: - Project Row

    private func projectRow(_ project: String) -> some View {
        let hidden = config.hiddenProjects.contains(project)
        return Button { toggleProject(project) } label: {
            HStack(spacing: 8) {
                Circle()
                    .fill(GraphView.projectColor(for: project, in: colorMap))
                    .frame(width: 8, height: 8)
                    .opacity(hidden ? 0.3 : 1.0)

                Text(project)
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(.white.opacity(hidden ? 0.3 : 0.7))
                    .strikethrough(hidden)
                    .lineLimit(1)

                Spacer()

                if let driveToProject {
                    Button { driveToProject(project) } label: {
                        Image(systemName: "scope")
                            .font(.system(size: 10))
                            .foregroundStyle(.white.opacity(hidden ? 0.15 : 0.4))
                    }
                    .buttonStyle(.plain)
                    .help("Fly to \(project)")
                }

                Image(systemName: hidden ? "eye.slash" : "eye")
                    .font(.system(size: 11))
                    .foregroundStyle(.white.opacity(hidden ? 0.3 : 0.5))
            }
            .padding(.vertical, 5)
            .padding(.horizontal, 8)
            .background(
                RoundedRectangle(cornerRadius: 5)
                    .fill(.white.opacity(hidden ? 0 : 0.03))
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    // MARK: - Relation Row

    private func relationRow(_ relation: String, count: Int) -> some View {
        let hidden = config.hiddenRelations.contains(relation)
        return Button { toggleRelation(relation) } label: {
            HStack(spacing: 8) {
                Circle()
                    .fill(GraphCanvas.relationColors[relation] ?? .white)
                    .frame(width: 8, height: 8)
                    .opacity(hidden ? 0.3 : 1.0)

                Text(relation.replacingOccurrences(of: "_", with: " "))
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(.white.opacity(hidden ? 0.3 : 0.7))
                    .strikethrough(hidden)
                    .lineLimit(1)

                Spacer()

                Text("\(count)")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.3))

                Image(systemName: hidden ? "eye.slash" : "eye")
                    .font(.system(size: 11))
                    .foregroundStyle(.white.opacity(hidden ? 0.3 : 0.5))
            }
            .padding(.vertical, 5)
            .padding(.horizontal, 8)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    // MARK: - Toggle Row

    private func toggleRow(_ label: String, icon: String, isOn: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 8) {
                Image(systemName: icon)
                    .font(.system(size: 11))
                    .frame(width: 16)
                Text(label)
                    .font(.system(size: 12, design: .monospaced))
                Spacer()
                togglePill(isOn: isOn)
            }
            .foregroundStyle(.white.opacity(0.7))
        }
        .buttonStyle(.plain)
    }

    func togglePill(isOn: Bool) -> some View {
        RoundedRectangle(cornerRadius: 8)
            .fill(isOn ? Color.cyan.opacity(0.5) : .white.opacity(0.1))
            .frame(width: 32, height: 18)
            .overlay(
                Circle()
                    .fill(.white)
                    .frame(width: 14, height: 14)
                    .offset(x: isOn ? 7 : -7)
            )
            .animation(.easeInOut(duration: 0.15), value: isOn)
    }

    // MARK: - Stat Row

    private func statRow(_ label: String, value: String) -> some View {
        HStack {
            Text(label)
                .font(.system(size: 12, design: .monospaced))
                .foregroundStyle(.white.opacity(0.5))
            Spacer()
            Text(value)
                .font(.system(size: 12, weight: .medium, design: .monospaced))
                .foregroundStyle(.white.opacity(0.7))
        }
    }

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

    // MARK: - Settings Tab

    @ViewBuilder
    private var settingsContent: some View {
        VStack(alignment: .leading, spacing: 24) {
            section("Notifications") {
                toggleRow(
                    "Push Notifications",
                    icon: config.notificationsEnabled ? "bell.fill" : "bell.slash.fill",
                    isOn: config.notificationsEnabled
                ) {
                    if config.notificationsEnabled {
                        config.notificationsEnabled = false
                    } else {
                        Task {
                            let center = UNUserNotificationCenter.current()
                            let granted = try? await center.requestAuthorization(options: [.alert, .sound, .badge])
                            if granted == true {
                                config.notificationsEnabled = true
                            }
                        }
                    }
                }
            }

            section("Panels") {
                toggleRow(
                    "Activity Log",
                    icon: config.showActivityLog ? "list.bullet.rectangle" : "list.bullet.rectangle",
                    isOn: config.showActivityLog
                ) {
                    config.showActivityLog.toggle()
                }
                toggleRow(
                    "Detail Panel",
                    icon: config.showDetailPanel ? "sidebar.trailing" : "sidebar.trailing",
                    isOn: config.showDetailPanel
                ) {
                    config.showDetailPanel.toggle()
                }
            }

            section("Audio") {
                toggleRow(
                    "Sound Effects",
                    icon: config.soundEnabled ? "speaker.wave.2.fill" : "speaker.slash.fill",
                    isOn: config.soundEnabled
                ) {
                    config.soundEnabled.toggle()
                }
            }

            section("Storage") {
                VStack(alignment: .leading, spacing: 10) {
                    statRow("Database", value: dbFileSize)

                    Button {
                        guard !isCompacting else { return }
                        isCompacting = true
                        Task {
                            lattice.compactHistory()
                            lattice.vacuum()
                            lattice.checkpoint()
                            isCompacting = false
                        }
                    } label: {
                        HStack(spacing: 6) {
                            if isCompacting {
                                ProgressView()
                                    .controlSize(.small)
                                    .scaleEffect(0.7)
                            } else {
                                Image(systemName: "arrow.triangle.2.circlepath")
                                    .font(.system(size: 11))
                            }
                            Text(isCompacting ? "Compacting…" : "Compact Database")
                                .font(.system(size: 12, design: .monospaced))
                        }
                        .foregroundStyle(.white.opacity(isCompacting ? 0.4 : 0.7))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                        .background(
                            RoundedRectangle(cornerRadius: 6)
                                .fill(.white.opacity(0.06))
                        )
                    }
                    .buttonStyle(.plain)
                    .disabled(isCompacting)
                }
            }

            section("About") {
                VStack(alignment: .leading, spacing: 6) {
                    let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "dev"
                    let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "?"
                    statRow("Version", value: "v\(version)")
                    statRow("Build", value: build)
                }
            }
        }
    }
}
