import AVFoundation
import simd

/// Oneshot event sounds and mascot thruster hum management.
final class EventSoundPlayer: @unchecked Sendable {

    private let environment: AVAudioEnvironmentNode

    // Oneshot player pool (4 voices for overlapping events)
    private var oneshotPlayers: [AVAudioPlayerNode] = []
    private var oneshotIndex: Int = 0
    private let oneshotPoolSize = 4

    // Mascot thruster voices: project → player
    private var thrusterPlayers: [String: AVAudioPlayerNode] = [:]

    // Pre-generated buffers
    private var selectionBuffer: AVAudioPCMBuffer?
    private var deselectionBuffer: AVAudioPCMBuffer?
    private var expansionBuffer: AVAudioPCMBuffer?
    private var collapseBuffer: AVAudioPCMBuffer?
    private var teleportBuffer: AVAudioPCMBuffer?
    private var arrivalBuffer: AVAudioPCMBuffer?
    private var holoOpenBuffer: AVAudioPCMBuffer?
    private var holoTickBuffer: AVAudioPCMBuffer?
    private var thrusterBuffer: AVAudioPCMBuffer?

    init(environment: AVAudioEnvironmentNode) {
        self.environment = environment
    }

    func setup() {
        guard let eng = environment.engine else { return }

        // Create oneshot pool
        for _ in 0..<oneshotPoolSize {
            let player = AVAudioPlayerNode()
            player.renderingAlgorithm = .HRTFHQ
            player.reverbBlend = 0.4
            eng.attach(player)
            eng.connect(player, to: environment, format: AudioSynthesis.monoFormat)
            oneshotPlayers.append(player)
        }

        // Pre-generate all event buffers
        selectionBuffer = AudioSynthesis.click(frequency: 1200, duration: 0.15, amplitude: 0.5)
        deselectionBuffer = AudioSynthesis.click(frequency: 800, duration: 0.1, amplitude: 0.3)
        expansionBuffer = AudioSynthesis.bloom(duration: 0.8, amplitude: 0.3)
        collapseBuffer = AudioSynthesis.descendingTone(startFreq: 400, endFreq: 150, duration: 0.5, amplitude: 0.2)
        teleportBuffer = AudioSynthesis.risingChirp(startFreq: 150, endFreq: 1200, duration: 0.4, amplitude: 0.35)
        arrivalBuffer = AudioSynthesis.risingChirp(startFreq: 400, endFreq: 1000, duration: 0.3, amplitude: 0.25)
        holoOpenBuffer = AudioSynthesis.crtBuzz(duration: 0.3, amplitude: 0.2)
        holoTickBuffer = AudioSynthesis.typewriterTick(frequency: 3000, amplitude: 0.1)
        thrusterBuffer = AudioSynthesis.thrusterHum(frequency: 80, duration: 2.0, amplitude: 0.2)
    }

    // MARK: - Event Triggers

    func playSelection(at position: SIMD3<Float>) {
        playOneshot(selectionBuffer, at: position, volume: 0.6)
    }

    func playDeselection() {
        // Non-spatialized deselection
        playOneshot(deselectionBuffer, at: nil, volume: 0.3)
    }

    func playExpansion(at position: SIMD3<Float>) {
        playOneshot(expansionBuffer, at: position, volume: 0.5)
    }

    func playCollapse(at position: SIMD3<Float>) {
        playOneshot(collapseBuffer, at: position, volume: 0.4)
    }

    func playTeleport() {
        // Non-spatialized — teleport is a camera event
        playOneshot(teleportBuffer, at: nil, volume: 0.5)
    }

    func playArrival(at position: SIMD3<Float>) {
        playOneshot(arrivalBuffer, at: position, volume: 0.4)
    }

    func playHoloOpen(at position: SIMD3<Float>) {
        playOneshot(holoOpenBuffer, at: position, volume: 0.35)
    }

    func playHoloTick(at position: SIMD3<Float>) {
        playOneshot(holoTickBuffer, at: position, volume: 0.2)
    }

    // MARK: - Mascot Thruster

    func updateMascotThruster(projectId: String, position: SIMD3<Float>, volume: Float) {
        let project = projectId
        let pos = position

        if let player = thrusterPlayers[project] {
            player.position = AVAudio3DPoint(x: pos.x, y: pos.y, z: pos.z)
            player.volume = volume
        } else {
            // Create new thruster voice
            guard let buffer = thrusterBuffer,
                  let eng = environment.engine else { return }

            let player = AVAudioPlayerNode()
            player.renderingAlgorithm = .HRTFHQ
            player.position = AVAudio3DPoint(x: pos.x, y: pos.y, z: pos.z)
            player.volume = 0.1
            player.reverbBlend = 0.2

            eng.attach(player)
            eng.connect(player, to: environment, format: AudioSynthesis.monoFormat)
            player.scheduleBuffer(buffer, at: nil, options: .loops)
            player.play()

            thrusterPlayers[project] = player
        }
    }

    /// Remove thruster for a project whose mascot was removed.
    func removeMascotThruster(project: String) {
        if let player = thrusterPlayers.removeValue(forKey: project) {
            player.stop()
            environment.engine?.detach(player)
        }
    }

    // MARK: - Internal

    private func playOneshot(_ buffer: AVAudioPCMBuffer?, at position: SIMD3<Float>?, volume: Float) {
        guard let buffer else { return }
        let player = oneshotPlayers[oneshotIndex]
        oneshotIndex = (oneshotIndex + 1) % oneshotPoolSize

        player.stop()

        if let pos = position {
            player.position = AVAudio3DPoint(x: pos.x, y: pos.y, z: pos.z)
        } else {
            player.position = AVAudio3DPoint(x: 0, y: 0, z: 0)
        }

        player.volume = volume
        player.scheduleBuffer(buffer, at: nil, options: [])
        player.play()
    }
}
