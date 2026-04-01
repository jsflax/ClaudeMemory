import SwiftUI
import EngramRealityKit

/// Overlay controls for the preview app.
struct PreviewControls: View {
    @Bindable var provider: MockGraphProvider
    let camera: CameraController
    let scene: EngramRealityScene

    @State private var nodeCountSlider: Double = 200
    @State private var soundEnabled: Bool = true

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Engram Preview")
                .font(.headline)

            Divider()

            // Node count
            VStack(alignment: .leading) {
                Text("Nodes: \(Int(nodeCountSlider))")
                    .font(.caption)
                Slider(value: $nodeCountSlider, in: 10...20_000, step: 10) { editing in
                    if !editing {
                        provider.nodeCount = Int(nodeCountSlider)
                    }
                }
            }

            // Stats
            VStack(alignment: .leading, spacing: 4) {
                Text("Nodes: \(provider.nodes.count)")
                    .font(.caption.monospaced())
                Text("Edges: \(provider.edges.count)")
                    .font(.caption.monospaced())
                Text("Hubs: \(provider.hubs.count)")
                    .font(.caption.monospaced())
                Text("Topology v\(provider.topologyVersion)")
                    .font(.caption.monospaced())
            }

            // Galaxies
            VStack(alignment: .leading, spacing: 4) {
                Text("Galaxies")
                    .font(.caption.bold())
                ForEach(provider.galaxyAssignment.sorted(by: { $0.key < $1.key }), id: \.key) { project, galaxy in
                    HStack(spacing: 6) {
                        Circle()
                            .fill(Color(
                                red: Double(provider.projectColorMap[project]?.x ?? 0.5),
                                green: Double(provider.projectColorMap[project]?.y ?? 0.5),
                                blue: Double(provider.projectColorMap[project]?.z ?? 0.5)
                            ))
                            .frame(width: 8, height: 8)
                        Text(project)
                            .font(.caption2)
                        Spacer()
                        Text(galaxy)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            Divider()

            // Audio
            Toggle("Spatial Audio", isOn: $soundEnabled)
                .onChange(of: soundEnabled) { _, enabled in
                    scene.spatialAudioSystem.isEnabled = enabled
                }

            Divider()

            // Effects toggles
            Button("Trigger Glow") {
                if let node = provider.nodes.randomElement() {
                    provider.glowingNodes[node.id] = 0
                }
            }
            .buttonStyle(.bordered)

            Button("Trigger Arrival") {
                provider.insertNewNode()
            }
            .buttonStyle(.bordered)

            Button("Migrate Project") {
                provider.migrateRandomProject()
            }
            .buttonStyle(.bordered)

            Button("Toggle Search") {
                provider.isSearchActive.toggle()
                if provider.isSearchActive {
                    // Match ~20% of nodes
                    let matchCount = max(1, provider.nodes.count / 5)
                    provider.searchMatchIds = Set(provider.nodes.prefix(matchCount).map(\.id))
                } else {
                    provider.searchMatchIds.removeAll()
                }
            }
            .buttonStyle(.bordered)

            Button("Regenerate") {
                provider.regenerateGraph()
            }
            .buttonStyle(.bordered)
        }
        .frame(width: 200)
    }
}
