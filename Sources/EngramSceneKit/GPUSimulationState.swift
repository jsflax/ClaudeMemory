import Metal
import CEngramSceneTypes
import simd

/// Owns GPU-resident simulation buffers (positions, velocities, forces).
/// Handles capacity management, CPU→GPU upload, GPU integration, and CPU readback.
public final class GPUSimulationState {
    private let device: MTLDevice
    public let integratePipeline: MTLComputePipelineState?

    // GPU-resident buffers (storageModeShared for CPU readback)
    public private(set) var positionBuffer: MTLBuffer?    // float3[N]
    public private(set) var velocityBuffer: MTLBuffer?    // float3[N]
    public private(set) var forceBuffer: MTLBuffer?       // float3[N] — zeroed each frame by force compute
    private(set) var capacity: Int = 0
    public var nodeCount: Int = 0

    // Alpha decay (CPU-side, passed as uniform)
    public var alpha: Float = 1.0
    public let alphaDecay: Float = 0.995
    public let alphaFloor: Float = 0.01
    private var settledFrameCount: Int = 0
    private let settleThreshold: Int = 30
    public private(set) var isSettled: Bool = false

    // Readback (every N frames, 1-3 frame latency)
    public private(set) var cpuPositions: [SIMD3<Float>] = []
    public var readbackInterval: Int = 3
    private var framesSinceReadback: Int = 0

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
        forceBuffer = device.makeBuffer(length: cap * stride3, options: .storageModeShared)
        // Zero velocity buffer on resize
        if let vb = velocityBuffer {
            memset(vb.contents(), 0, cap * stride3)
        }
        capacity = cap
    }

    // MARK: - Upload from CPU (initial positions or topology change)

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
        // Per-frame uploads preserve GPU-computed velocities for momentum.
        if topologyChanged, let vb = velocityBuffer {
            memset(vb.contents(), 0, n * MemoryLayout<SIMD3<Float>>.stride)
        }
    }

    // MARK: - GPU Integration

    /// Encode integrate_positions kernel into the compute encoder.
    /// Called after force compute, before node packing.
    public func encodeIntegration(encoder: MTLComputeCommandEncoder, damping: Float, maxSpeed: Float, alpha: Float) {
        guard let pipeline = integratePipeline,
              let posBuf = positionBuffer,
              let velBuf = velocityBuffer,
              let forceBuf = forceBuffer,
              let paramsBuf = integrateParamsBuffer,
              nodeCount > 0 else { return }

        var params = IntegrateParams(
            nodeCount: UInt32(nodeCount),
            damping: damping,
            maxSpeed: maxSpeed,
            alpha: alpha
        )
        paramsBuf.contents().copyMemory(from: &params, byteCount: MemoryLayout<IntegrateParams>.stride)

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(posBuf, offset: 0, index: 0)
        encoder.setBuffer(velBuf, offset: 0, index: 1)
        encoder.setBuffer(forceBuf, offset: 0, index: 2)
        encoder.setBuffer(paramsBuf, offset: 0, index: 3)

        let tgSize = min(Int(pipeline.maxTotalThreadsPerThreadgroup), 256)
        encoder.dispatchThreads(
            MTLSize(width: nodeCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        encoder.memoryBarrier(scope: .buffers)
    }

    /// Encode integration using an external force buffer (e.g. from MetalForceCompute).
    /// Same as encodeIntegration but uses a caller-provided force buffer instead of self.forceBuffer.
    public func encodeIntegration(encoder: MTLComputeCommandEncoder, forceBuffer: MTLBuffer,
                                  damping: Float, maxSpeed: Float, alpha: Float) {
        guard let pipeline = integratePipeline,
              let posBuf = positionBuffer,
              let velBuf = velocityBuffer,
              let paramsBuf = integrateParamsBuffer,
              nodeCount > 0 else { return }

        var params = IntegrateParams(
            nodeCount: UInt32(nodeCount),
            damping: damping,
            maxSpeed: maxSpeed,
            alpha: alpha
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

    // MARK: - Alpha Decay + Settlement (Phase 4)

    public func decayAlpha() {
        guard !isSettled else { return }
        alpha *= alphaDecay
        if alpha < alphaFloor {
            alpha = alphaFloor
            settledFrameCount += 1
            if settledFrameCount >= settleThreshold {
                isSettled = true
                GPULog.log("GPU SIM SETTLED after \(settleThreshold) frames at alpha floor")
            }
        } else {
            settledFrameCount = 0
        }
    }

    /// Wake the simulation (topology change, new nodes, etc.)
    public func wake() {
        alpha = 1.0
        settledFrameCount = 0
        isSettled = false
    }

    // MARK: - CPU Readback

    /// Read back positions to CPU. Call after command buffer completes.
    /// Returns true if readback was performed this frame.
    @discardableResult
    public func readBackIfNeeded() -> Bool {
        framesSinceReadback += 1
        guard framesSinceReadback >= readbackInterval else { return false }
        framesSinceReadback = 0
        return readBackPositions()
    }

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
