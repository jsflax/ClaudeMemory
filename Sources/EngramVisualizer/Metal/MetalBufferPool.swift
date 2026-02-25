import Metal
import simd

/// Triple-buffered frame management for the Metal render loop.
/// Rotates uniform buffers per frame; semaphore prevents CPU from
/// getting more than `maxInflightFrames` ahead of the GPU.
@MainActor
final class MetalBufferPool {
    let maxInflightFrames = 3
    let frameSemaphore: DispatchSemaphore

    private let device: MTLDevice
    private var frameUniformBuffers: [MTLBuffer] = []
    private var lightingUniformBuffers: [MTLBuffer] = []
    private var frameIndex: Int = 0

    init(device: MTLDevice) {
        self.device = device
        self.frameSemaphore = DispatchSemaphore(value: maxInflightFrames)

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

    /// Call at the start of each frame. Blocks if GPU is 3 frames behind.
    func waitForNextFrame() {
        frameSemaphore.wait()
        frameIndex = (frameIndex + 1) % maxInflightFrames
    }

    /// Signal that the GPU has finished a frame (call from command buffer completion handler).
    nonisolated func signalFrameComplete() {
        frameSemaphore.signal()
    }

    /// Current frame's uniform buffer for FrameUniforms.
    var currentFrameUniformBuffer: MTLBuffer {
        frameUniformBuffers[frameIndex]
    }

    /// Current frame's uniform buffer for LightingUniforms.
    var currentLightingUniformBuffer: MTLBuffer {
        lightingUniformBuffers[frameIndex]
    }

    /// Write FrameUniforms into the current frame's buffer.
    func updateFrameUniforms(_ uniforms: FrameUniforms) {
        let buf = currentFrameUniformBuffer
        buf.contents().storeBytes(of: uniforms, as: FrameUniforms.self)
    }

    /// Write LightingUniforms into the current frame's buffer.
    func updateLightingUniforms(_ uniforms: LightingUniforms) {
        let buf = currentLightingUniformBuffer
        buf.contents().storeBytes(of: uniforms, as: LightingUniforms.self)
    }
}
