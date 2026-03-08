import AVFoundation
import simd

/// Programmatic audio buffer synthesis — no asset files needed.
/// All buffers are mono, 44100 Hz, Float32.
@MainActor
enum AudioSynthesis {

    static let sampleRate: Double = 44100
    static let monoFormat = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!

    // MARK: - Pentatonic Scale (all combinations consonant)

    /// Maps hue (0..1) to a pentatonic pitch in Hz.
    /// C3=130.81, D3=146.83, E3=164.81, G3=196.00, A3=220.00
    static let pentatonicFrequencies: [Float] = [130.81, 146.83, 164.81, 196.00, 220.00]

    static func pentatonicFrequency(forHue hue: Float) -> Float {
        let index = Int((hue * 5).truncatingRemainder(dividingBy: 5))
        return pentatonicFrequencies[max(0, min(4, index))]
    }

    // MARK: - Buffer Generators

    /// Looping sine pad with slow amplitude modulation. Duration in seconds.
    static func sineLoop(frequency: Float, duration: Float = 2.0, amplitude: Float = 0.3) -> AVAudioPCMBuffer? {
        let frameCount = AVAudioFrameCount(Double(duration) * sampleRate)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: monoFormat, frameCapacity: frameCount) else { return nil }
        buffer.frameLength = frameCount
        guard let data = buffer.floatChannelData?[0] else { return nil }

        let omega = 2.0 * Float.pi * frequency / Float(sampleRate)
        let lfoRate = 2.0 * Float.pi * 0.1 / Float(sampleRate) // 0.1 Hz tremolo

        for i in 0..<Int(frameCount) {
            let fi = Float(i)
            let lfo = 0.7 + 0.3 * sin(fi * lfoRate)
            data[i] = amplitude * lfo * sin(fi * omega)
        }

        return buffer
    }

    /// Additive pad with multiple partials and slow evolution. Good for ambient bed.
    static func ambientPad(fundamental: Float = 65.0, duration: Float = 4.0, amplitude: Float = 0.15) -> AVAudioPCMBuffer? {
        let frameCount = AVAudioFrameCount(Double(duration) * sampleRate)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: monoFormat, frameCapacity: frameCount) else { return nil }
        buffer.frameLength = frameCount
        guard let data = buffer.floatChannelData?[0] else { return nil }

        let partials: [(harmonic: Float, amp: Float, lfoRate: Float)] = [
            (1.0, 0.4, 0.05),   // fundamental
            (2.0, 0.25, 0.07),  // octave
            (3.0, 0.12, 0.11),  // fifth above octave
            (5.0, 0.06, 0.13),  // two octaves + major third
        ]

        for i in 0..<Int(frameCount) {
            let fi = Float(i)
            var sample: Float = 0
            for p in partials {
                let omega = 2.0 * Float.pi * fundamental * p.harmonic / Float(sampleRate)
                let lfo = 0.6 + 0.4 * sin(fi * 2.0 * Float.pi * p.lfoRate / Float(sampleRate))
                sample += p.amp * lfo * sin(fi * omega)
            }
            data[i] = amplitude * sample
        }

        return buffer
    }

    /// Short sine ping with ADSR envelope. Good for proximity tones.
    static func sinePing(frequency: Float, duration: Float = 0.6, amplitude: Float = 0.4,
                         attack: Float = 0.02, decay: Float = 0.1, sustain: Float = 0.3, release: Float = 0.3) -> AVAudioPCMBuffer? {
        let frameCount = AVAudioFrameCount(Double(duration) * sampleRate)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: monoFormat, frameCapacity: frameCount) else { return nil }
        buffer.frameLength = frameCount
        guard let data = buffer.floatChannelData?[0] else { return nil }

        let omega = 2.0 * Float.pi * frequency / Float(sampleRate)
        let sr = Float(sampleRate)
        let attackSamples = attack * sr
        let decaySamples = decay * sr
        let releaseSamples = release * sr
        let sustainEnd = Float(frameCount) - releaseSamples

        for i in 0..<Int(frameCount) {
            let fi = Float(i)
            let env: Float
            if fi < attackSamples {
                env = fi / attackSamples
            } else if fi < attackSamples + decaySamples {
                let t = (fi - attackSamples) / decaySamples
                env = 1.0 - t * (1.0 - sustain)
            } else if fi < sustainEnd {
                env = sustain
            } else {
                let t = (fi - sustainEnd) / releaseSamples
                env = sustain * (1.0 - t)
            }
            data[i] = amplitude * env * sin(fi * omega)
        }

        return buffer
    }

    /// Rising pitch sweep (portamento). Good for node arrival.
    static func risingChirp(startFreq: Float = 200, endFreq: Float = 800, duration: Float = 0.5, amplitude: Float = 0.3) -> AVAudioPCMBuffer? {
        let frameCount = AVAudioFrameCount(Double(duration) * sampleRate)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: monoFormat, frameCapacity: frameCount) else { return nil }
        buffer.frameLength = frameCount
        guard let data = buffer.floatChannelData?[0] else { return nil }

        let sr = Float(sampleRate)
        var phase: Float = 0

        for i in 0..<Int(frameCount) {
            let t = Float(i) / Float(frameCount)
            let freq = startFreq * pow(endFreq / startFreq, t)
            let env = sin(t * Float.pi) // bell envelope
            phase += 2.0 * Float.pi * freq / sr
            data[i] = amplitude * env * sin(phase)
        }

        return buffer
    }

    /// Descending tone. Good for node death / power-down.
    static func descendingTone(startFreq: Float = 600, endFreq: Float = 100, duration: Float = 0.8, amplitude: Float = 0.25) -> AVAudioPCMBuffer? {
        let frameCount = AVAudioFrameCount(Double(duration) * sampleRate)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: monoFormat, frameCapacity: frameCount) else { return nil }
        buffer.frameLength = frameCount
        guard let data = buffer.floatChannelData?[0] else { return nil }

        let sr = Float(sampleRate)
        var phase: Float = 0

        for i in 0..<Int(frameCount) {
            let t = Float(i) / Float(frameCount)
            let freq = startFreq * pow(endFreq / startFreq, t)
            let env = 1.0 - t * t // quadratic fadeout
            phase += 2.0 * Float.pi * freq / sr
            data[i] = amplitude * env * sin(phase)
        }

        return buffer
    }

    /// Resonant click with short decay. Good for selection.
    static func click(frequency: Float = 1200, duration: Float = 0.15, amplitude: Float = 0.5) -> AVAudioPCMBuffer? {
        let frameCount = AVAudioFrameCount(Double(duration) * sampleRate)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: monoFormat, frameCapacity: frameCount) else { return nil }
        buffer.frameLength = frameCount
        guard let data = buffer.floatChannelData?[0] else { return nil }

        let omega = 2.0 * Float.pi * frequency / Float(sampleRate)

        for i in 0..<Int(frameCount) {
            let t = Float(i) / Float(frameCount)
            let env = exp(-t * 20.0) // fast exponential decay
            data[i] = amplitude * env * sin(Float(i) * omega)
        }

        return buffer
    }

    /// CRT power-on buzz. Good for holo screen open.
    static func crtBuzz(duration: Float = 0.3, amplitude: Float = 0.2) -> AVAudioPCMBuffer? {
        let frameCount = AVAudioFrameCount(Double(duration) * sampleRate)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: monoFormat, frameCapacity: frameCount) else { return nil }
        buffer.frameLength = frameCount
        guard let data = buffer.floatChannelData?[0] else { return nil }

        let sr = Float(sampleRate)
        let baseFreq: Float = 120.0 // power line hum

        for i in 0..<Int(frameCount) {
            let fi = Float(i)
            let t = fi / Float(frameCount)
            let env = min(1.0, t * 10.0) * (1.0 - t * t) // fast attack, slow fade
            // Square-ish wave with harmonics for that CRT character
            var sample = sin(fi * 2.0 * Float.pi * baseFreq / sr)
            sample += 0.3 * sin(fi * 2.0 * Float.pi * baseFreq * 3.0 / sr) // 3rd harmonic
            sample += 0.15 * sin(fi * 2.0 * Float.pi * baseFreq * 5.0 / sr) // 5th harmonic
            // High-frequency whine component
            sample += 0.08 * sin(fi * 2.0 * Float.pi * 12000.0 / sr) * (1.0 - t)
            data[i] = amplitude * env * sample
        }

        return buffer
    }

    /// Typewriter tick. Good for holo screen line reveal.
    static func typewriterTick(frequency: Float = 3000, amplitude: Float = 0.15) -> AVAudioPCMBuffer? {
        let frameCount = AVAudioFrameCount(0.03 * sampleRate) // 30ms
        guard let buffer = AVAudioPCMBuffer(pcmFormat: monoFormat, frameCapacity: frameCount) else { return nil }
        buffer.frameLength = frameCount
        guard let data = buffer.floatChannelData?[0] else { return nil }

        let omega = 2.0 * Float.pi * frequency / Float(sampleRate)

        for i in 0..<Int(frameCount) {
            let t = Float(i) / Float(frameCount)
            let env = exp(-t * 40.0) // very fast decay
            // Mix of noise-like and tonal for mechanical character
            let noise = Float.random(in: -1...1)
            data[i] = amplitude * env * (0.6 * sin(Float(i) * omega) + 0.4 * noise)
        }

        return buffer
    }

    /// Thruster hum — low looping buzz for mascot engines.
    static func thrusterHum(frequency: Float = 80, duration: Float = 2.0, amplitude: Float = 0.2) -> AVAudioPCMBuffer? {
        let frameCount = AVAudioFrameCount(Double(duration) * sampleRate)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: monoFormat, frameCapacity: frameCount) else { return nil }
        buffer.frameLength = frameCount
        guard let data = buffer.floatChannelData?[0] else { return nil }

        let sr = Float(sampleRate)

        for i in 0..<Int(frameCount) {
            let fi = Float(i)
            let lfo = 0.8 + 0.2 * sin(fi * 2.0 * Float.pi * 4.0 / sr) // 4 Hz flutter
            var sample = sin(fi * 2.0 * Float.pi * frequency / sr)
            sample += 0.5 * sin(fi * 2.0 * Float.pi * frequency * 2.0 / sr)  // 2nd harmonic
            sample += 0.2 * sin(fi * 2.0 * Float.pi * frequency * 3.5 / sr)  // non-integer harmonic
            data[i] = amplitude * lfo * sample
        }

        return buffer
    }

    /// Bloom/expansion sound — rising harmonic cluster.
    static func bloom(duration: Float = 0.8, amplitude: Float = 0.3) -> AVAudioPCMBuffer? {
        let frameCount = AVAudioFrameCount(Double(duration) * sampleRate)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: monoFormat, frameCapacity: frameCount) else { return nil }
        buffer.frameLength = frameCount
        guard let data = buffer.floatChannelData?[0] else { return nil }

        let sr = Float(sampleRate)
        let baseFreq: Float = 220.0

        for i in 0..<Int(frameCount) {
            let fi = Float(i)
            let t = fi / Float(frameCount)
            let env = sin(t * Float.pi) * (1.0 - t * 0.3) // bell with slight decay
            // Rising cluster: each partial sweeps up at different rates
            var sample: Float = 0
            for h in 1...5 {
                let hf = Float(h)
                let freqSweep = baseFreq * hf * (1.0 + t * 0.5 * hf / 5.0)
                sample += (1.0 / hf) * sin(fi * 2.0 * Float.pi * freqSweep / sr)
            }
            data[i] = amplitude * env * sample / 2.0
        }

        return buffer
    }
}
