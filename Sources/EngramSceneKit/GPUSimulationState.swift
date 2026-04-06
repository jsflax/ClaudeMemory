import Metal
import CEngramSceneTypes
import simd

/// Owns GPU-resident simulation buffers (positions, velocities).
/// Handles capacity management, CPU→GPU upload, GPU integration, and CPU readback.
public final class GPUSimulationState {
    private let device: MTLDevice
    public let integratePipeline: MTLComputePipelineState?

    // GPU-resident buffers (storageModeShared for CPU readback)
    public private(set) var positionBuffer: MTLBuffer?    // float3[N]
    public private(set) var velocityBuffer: MTLBuffer?    // float3[N]
    private(set) var capacity: Int = 0
    public var nodeCount: Int = 0

    // Readback
    public private(set) var cpuPositions: [SIMD3<Float>] = []

    // Integration params buffer (reused each frame)
    private var integrateParamsBuffer: MTLBuffer?

    public init(device: MTLDevice, library: MTLLibrary) {
        self.device = device
        if let fn = library.makeFunction(name: "integrate_positions") {
            self.integratePipeline = try? device.makeComputePipelineState(function: fn)
        } else {
            self.integratePipeline = nil
        }
        self.integrateParamsBuffer = device.makeBuffer(
            length: MemoryLayout<IntegrateParams>.stride, options: .storageModeShared)
    }

    // MARK: - Capacity Management

    public func ensureCapacity(_ n: Int) {
        guard n > capacity else { return }
        let cap = max(n * 2, 512)
        let stride3 = MemoryLayout<SIMD3<Float>>.stride
        positionBuffer = device.makeBuffer(length: cap * stride3, options: .storageModeShared)
        velocityBuffer = device.makeBuffer(length: cap * stride3, options: .storageModeShared)
        // Zero velocity buffer on resize
        if let vb = velocityBuffer {
            memset(vb.contents(), 0, cap * stride3)
        }
        capacity = cap
    }

    // MARK: - Upload from CPU

    /// Upload positions from SoA arrays (ForceSimulation3D layout).
    public func uploadPositions(x: [Float], y: [Float], z: [Float]) {
        let n = x.count
        ensureCapacity(n)
        let topologyChanged = n != nodeCount
        nodeCount = n
        guard let pb = positionBuffer else { return }
        let ptr = pb.contents().bindMemory(to: SIMD3<Float>.self, capacity: n)
        for i in 0..<n {
            ptr[i] = SIMD3<Float>(x[i], y[i], z[i])
        }
        // Only zero velocities on topology change (node count changed).
        if topologyChanged, let vb = velocityBuffer {
            memset(vb.contents(), 0, n * MemoryLayout<SIMD3<Float>>.stride)
        }
    }

    // MARK: - GPU Integration

    /// Encode integrate_positions kernel using an external force buffer.
    /// Matches JS: vel = (vel + force) * damping; clamp; pos += vel
    /// Alpha is NOT used in integration (it's only in center gravity in the spring kernel).
    public func encodeIntegration(encoder: MTLComputeCommandEncoder, forceBuffer: MTLBuffer,
                                  damping: Float, maxSpeed: Float) {
        guard let pipeline = integratePipeline,
              let posBuf = positionBuffer,
              let velBuf = velocityBuffer,
              let paramsBuf = integrateParamsBuffer,
              nodeCount > 0 else { return }

        var params = IntegrateParams(
            nodeCount: UInt32(nodeCount),
            damping: damping,
            maxSpeed: maxSpeed,
            _pad: 0
        )
        paramsBuf.contents().copyMemory(from: &params, byteCount: MemoryLayout<IntegrateParams>.stride)

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(posBuf, offset: 0, index: 0)
        encoder.setBuffer(velBuf, offset: 0, index: 1)
        encoder.setBuffer(forceBuffer, offset: 0, index: 2)
        encoder.setBuffer(paramsBuf, offset: 0, index: 3)

        let tgSize = min(Int(pipeline.maxTotalThreadsPerThreadgroup), 256)
        encoder.dispatchThreads(
            MTLSize(width: nodeCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        encoder.memoryBarrier(scope: .buffers)
    }

    // MARK: - CPU Readback

    /// Force a readback (for initial frame or after topology change).
    @discardableResult
    public func readBackPositions() -> Bool {
        guard let pb = positionBuffer, nodeCount > 0 else { return false }
        let ptr = pb.contents().bindMemory(to: SIMD3<Float>.self, capacity: nodeCount)
        if cpuPositions.count != nodeCount {
            cpuPositions = [SIMD3<Float>](repeating: .zero, count: nodeCount)
        }
        cpuPositions.withUnsafeMutableBufferPointer { dst in
            dst.baseAddress!.update(from: ptr, count: nodeCount)
        }
        return true
    }
}
