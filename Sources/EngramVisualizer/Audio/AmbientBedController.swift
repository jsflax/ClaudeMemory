import AVFoundation

/// Non-spatialized ambient pad that represents the void between thoughts.
/// Volume/timbre modulated by camera velocity and simulation state.
@MainActor
final class AmbientBedController {

    private let engine: AVAudioEngine
    private let player = AVAudioPlayerNode()
    private let eq = AVAudioUnitEQ(numberOfBands: 1)
    private var padBuffer: AVAudioPCMBuffer?
    private var isPlaying = false

    /// Base volume for the ambient bed.
    private let baseVolume: Float = 0.15

    init(engine: AVAudioEngine) {
        self.engine = engine
    }

    func setup() {
        engine.attach(player)
        engine.attach(eq)

        // High-shelf filter for velocity modulation
        let band = eq.bands[0]
        band.filterType = .highShelf
        band.frequency = 2000
        band.gain = -12 // start quiet — velocity opens it up
        band.bypass = false

        // Player → EQ → main mixer (non-spatialized, bypasses environment node)
        engine.connect(player, to: eq, format: AudioSynthesis.monoFormat)
        engine.connect(eq, to: engine.mainMixerNode, format: AudioSynthesis.monoFormat)

        // Pre-generate pad buffer
        padBuffer = AudioSynthesis.ambientPad(fundamental: 65.0, duration: 4.0, amplitude: 0.3)
    }

    func start() {
        guard let buffer = padBuffer, !isPlaying else { return }
        player.volume = baseVolume
        player.scheduleBuffer(buffer, at: nil, options: .loops)
        player.play()
        isPlaying = true
    }

    func stop() {
        player.stop()
        isPlaying = false
    }

    /// Called each frame from SpatialAudioEngine.tick().
    func update(velocity: Float, simAlpha: Float) {
        guard isPlaying else { return }

        // Velocity → high-shelf gain: still = -12dB (muffled), sprinting = +3dB (bright)
        let normalizedVelocity = min(velocity / 2.0, 1.0) // 2.0 render-space units/sec = full speed
        let shelfGain = -12.0 + normalizedVelocity * 15.0  // -12 to +3
        eq.bands[0].gain = shelfGain

        // Simulation alpha → volume tension: high alpha (unsettled) = louder, settled = quieter
        let tensionBoost: Float = 1.0 + simAlpha * 0.3 // up to 30% louder when graph is active
        player.volume = baseVolume * (0.6 + normalizedVelocity * 0.4) * tensionBoost
    }
}
