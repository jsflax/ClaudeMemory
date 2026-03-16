import Metal
import CEngramSceneTypes
import simd

/// Triple-buffered uniform management for the Metal render loop.
/// Rotates uniform buffers per frame. No semaphore — CAMetalLayer's
/// drawable pool (3 drawables) provides natural backpressure.
@MainActor
public final class MetalBufferPool {
    public let maxInflightFrames = 3

    private let device: MTLDevice
    private var frameUniformBuffers: [MTLBuffer] = []
    private var lightingUniformBuffers: [MTLBuffer] = []
    private var frameIndex: Int = 0

    public init(device: MTLDevice) {
        self.device = device

        for _ in 0..<maxInflightFrames {
            let fuBuf = device.makeBuffer(
                length: MemoryLayout<FrameUniforms>.stride,
                options: .storageModeShared
            )!
            frameUniformBuffers.append(fuBuf)

            let luBuf = device.makeBuffer(
                length: MemoryLayout<LightingUniforms>.stride,
                options: .storageModeShared
            )!
            lightingUniformBuffers.append(luBuf)
        }
    }

    /// Advance to the next frame slot. Called at the start of each draw.
    public func advanceFrame() {
        frameIndex = (frameIndex + 1) % maxInflightFrames
    }

    /// Current frame's uniform buffer for FrameUniforms.
    public var currentFrameUniformBuffer: MTLBuffer {
        frameUniformBuffers[frameIndex]
    }

    /// Current frame's uniform buffer for LightingUniforms.
    public var currentLightingUniformBuffer: MTLBuffer {
        lightingUniformBuffers[frameIndex]
    }

    /// Write FrameUniforms into the current frame's buffer.
    public func updateFrameUniforms(_ uniforms: FrameUniforms) {
        let buf = currentFrameUniformBuffer
        buf.contents().storeBytes(of: uniforms, as: FrameUniforms.self)
    }

    /// Write LightingUniforms into the current frame's buffer.
    public func updateLightingUniforms(_ uniforms: LightingUniforms) {
        let buf = currentLightingUniformBuffer
        buf.contents().storeBytes(of: uniforms, as: LightingUniforms.self)
    }
}
