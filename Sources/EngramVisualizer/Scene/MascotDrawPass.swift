import Metal
import EngramSceneKit

/// Bridges MascotFleet rendering into FrameOrchestrator's ExtraDrawPass protocol.
/// Holds mascot-specific pipelines and galaxy registry reference.
@MainActor
final class MascotDrawPass: ExtraDrawPass {

    weak var galaxyRegistry: GalaxyRegistry?
    weak var cameraController: CameraController?
    var showMascots: Bool = true

    // Mascot render pipelines (built from MetalPipelineBuilder in app target)
    var mascotPipeline: MTLRenderPipelineState?
    var arcaneCirclePipeline: MTLRenderPipelineState?
    var nodeRingPipeline: MTLRenderPipelineState?
    var holoScreenPipeline: MTLRenderPipelineState?
    var conjureScanPipeline: MTLRenderPipelineState?
    var conjureOrbPipeline: MTLRenderPipelineState?
    var flowPipeline: MTLRenderPipelineState?
    var opaqueDepthState: MTLDepthStencilState?

    func encodeCompute(encoder: MTLComputeCommandEncoder, dt: Float, camera: CameraState) {
        guard showMascots, let registry = galaxyRegistry, let cam = cameraController else { return }
        for galaxy in registry.galaxies.values {
            galaxy.mascotFleet?.dispatchMatrixCompute(encoder: encoder)
            galaxy.mascotFleet?.dispatchParticleCompute(
                encoder: encoder, dt: dt, camera: cam
            )
        }
    }

    func encodeOpaque(encoder: MTLRenderCommandEncoder, frameUniform: MTLBuffer, lightUniform: MTLBuffer) {
        guard showMascots,
              let mascotPSO = mascotPipeline,
              let depthState = opaqueDepthState,
              let registry = galaxyRegistry else { return }
        for galaxy in registry.galaxies.values {
            galaxy.mascotFleet?.drawAll(
                encoder: encoder,
                frameUniformBuf: frameUniform,
                lightUniformBuf: lightUniform,
                pipeline: mascotPSO,
                depthState: depthState
            )
        }
    }

    func encodeTransparent(encoder: MTLRenderCommandEncoder, frameUniform: MTLBuffer) {
        guard showMascots, let registry = galaxyRegistry else { return }

        // Thruster particles
        if let flowPSO = flowPipeline {
            for galaxy in registry.galaxies.values {
                galaxy.mascotFleet?.drawAllThrusterParticles(
                    encoder: encoder, frameUniformBuf: frameUniform, pipeline: flowPSO)
            }
        }
        // Arcane circles
        if let arcanePSO = arcaneCirclePipeline {
            for galaxy in registry.galaxies.values {
                galaxy.mascotFleet?.drawAllArcaneCircles(
                    encoder: encoder, frameUniformBuf: frameUniform, pipeline: arcanePSO)
            }
        }
        // Scan rings
        if let scanPSO = conjureScanPipeline {
            for galaxy in registry.galaxies.values {
                galaxy.mascotFleet?.drawAllScanRings(
                    encoder: encoder, frameUniformBuf: frameUniform, pipeline: scanPSO)
            }
        }
        // Conjure orbs
        if let orbPSO = conjureOrbPipeline {
            for galaxy in registry.galaxies.values {
                galaxy.mascotFleet?.drawAllConjureOrbs(
                    encoder: encoder, frameUniformBuf: frameUniform, pipeline: orbPSO)
            }
        }
        // Node rings
        if let ringPSO = nodeRingPipeline {
            for galaxy in registry.galaxies.values {
                galaxy.mascotFleet?.drawAllNodeRings(
                    encoder: encoder, frameUniformBuf: frameUniform, pipeline: ringPSO)
            }
        }
        // Holo screens
        if let holoPSO = holoScreenPipeline {
            for galaxy in registry.galaxies.values {
                galaxy.mascotFleet?.drawAllHoloScreens(
                    encoder: encoder, frameUniformBuf: frameUniform, pipeline: holoPSO)
            }
        }
    }
}
