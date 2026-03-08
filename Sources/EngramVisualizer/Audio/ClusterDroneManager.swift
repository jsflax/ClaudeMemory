import AVFoundation
import simd
import SwiftUI

/// Per-project spatialized drones positioned at cluster centroids.
/// Each project gets a unique pitch derived from its color hue.
@MainActor
final class ClusterDroneManager {

    private let environment: AVAudioEnvironmentNode

    private struct DroneVoice {
        let player: AVAudioPlayerNode
        let buffer: AVAudioPCMBuffer
        var isPlaying: Bool = false
    }

    /// Active drones keyed by project name.
    private var drones: [String: DroneVoice] = [:]
    /// Pre-generated buffers keyed by pentatonic frequency.
    private var bufferCache: [Float: AVAudioPCMBuffer] = [:]
    /// Projects seen on the last update — used to detect additions/removals.
    private var lastProjects: Set<String> = []

    private let maxDrones = 10

    init(environment: AVAudioEnvironmentNode) {
        self.environment = environment
    }

    func setup() {
        // Pre-generate a buffer for each pentatonic note
        for freq in AudioSynthesis.pentatonicFrequencies {
            if let buf = AudioSynthesis.sineLoop(frequency: freq, duration: 2.0, amplitude: 0.25) {
                bufferCache[freq] = buf
            }
        }
    }

    /// Update drone positions and lifecycle from nebula groups.
    func update(groups: [NebulaFogSystem.NebulaGroup], colorMap: [String: Color], scaleFactor: Float) {
        let currentProjects = Set(groups.map(\.project))

        // Remove drones for projects that disappeared
        for project in lastProjects.subtracting(currentProjects) {
            if let voice = drones.removeValue(forKey: project) {
                voice.player.stop()
                environment.engine?.detach(voice.player)
            }
        }

        // Add/update drones for current projects
        for group in groups {
            let project = group.project
            let audioPos = group.centroid * scaleFactor

            if let voice = drones[project] {
                // Update position
                voice.player.position = AVAudio3DPoint(x: audioPos.x, y: audioPos.y, z: audioPos.z)
            } else if drones.count < maxDrones {
                // Create new drone
                let hue = hueForProject(project, colorMap: colorMap)
                let freq = AudioSynthesis.pentatonicFrequency(forHue: hue)
                guard let buffer = bufferCache[freq] else { continue }

                let player = AVAudioPlayerNode()
                player.renderingAlgorithm = .HRTFHQ
                player.position = AVAudio3DPoint(x: audioPos.x, y: audioPos.y, z: audioPos.z)
                player.volume = 0.4
                player.reverbBlend = 0.5

                guard let eng = environment.engine else { continue }
                eng.attach(player)
                eng.connect(player, to: environment, format: AudioSynthesis.monoFormat)

                player.scheduleBuffer(buffer, at: nil, options: .loops)
                player.play()

                drones[project] = DroneVoice(player: player, buffer: buffer, isPlaying: true)
            }
        }

        lastProjects = currentProjects
    }

    func stopAll() {
        for (_, voice) in drones {
            voice.player.stop()
        }
    }

    // MARK: - Helpers

    private func hueForProject(_ project: String, colorMap: [String: Color]) -> Float {
        guard let color = colorMap[project] else {
            // Deterministic fallback from project name hash
            return Float((project.hashValue & Int.max) % 100) / 100.0
        }
        let nsColor = NSColor(color).usingColorSpace(.sRGB) ?? NSColor(color)
        return Float(nsColor.hueComponent)
    }
}
