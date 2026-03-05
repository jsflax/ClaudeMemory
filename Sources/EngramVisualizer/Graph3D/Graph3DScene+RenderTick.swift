import SwiftUI
import RealityKit
import GameController
import simd
import os

private let frameLog = Logger(subsystem: "com.claudememory.visualizer", category: "3DFrameTiming")

extension Graph3DScene {

    #if ENGRAM_INSTRUMENTATION
    func setupProfiling() {
        let path = "/tmp/frame-timing.csv"
        FileManager.default.createFile(atPath: path, contents: nil)
        profilingFileHandle = FileHandle(forWritingAtPath: path)
        let header = "frame,dt_ms,work_ms,expand_ms,nodes_ms,edges_ms,neb_ms,node_count,edge_count,labels_ms,flow_ms,idle,entities,reason\n"
        profilingFileHandle?.write(header.data(using: .utf8)!)
        profilingReady = true
    }

    func flushProfiling() {
        guard profilingReady, !profilingLines.isEmpty else { return }
        let data = profilingLines.joined().data(using: .utf8)!
        profilingFileHandle?.write(data)
        profilingLines.removeAll(keepingCapacity: true)
    }

    func setupLabelDiag() {
        let path = "/tmp/label-diag.csv"
        FileManager.default.createFile(atPath: path, contents: nil)
        labelDiagFileHandle = FileHandle(forWritingAtPath: path)
        let header = "frame,positions,atlasRects,missingRects,instances,capacity,atlasRegen,depthRange,minDepth,maxDepth,camX,camY,camZ,partsReplaced,projLabels\n"
        labelDiagFileHandle?.write(header.data(using: .utf8)!)
        labelDiagReady = true
    }

    func flushLabelDiag() {
        guard labelDiagReady, !labelDiagLines.isEmpty else { return }
        let data = labelDiagLines.joined().data(using: .utf8)!
        labelDiagFileHandle?.write(data)
        labelDiagLines.removeAll(keepingCapacity: true)
    }
    #endif

    // MARK: - Render Tick

    func renderTick() {
        let now = CFAbsoluteTimeGetCurrent()
        let dtSec = lastRenderTime > 0 ? Float(now - lastRenderTime) : Float(1.0 / 60.0)
        let dt = Double(dtSec) * 1000.0

        // Always poll inputs + update camera (cheap, needed for responsiveness)
        let preInputSelection = renderSelectedNode
        pollKeyboard(dt: dtSec)
        pollGamepad(dt: dtSec, selectedNode: &renderSelectedNode,
                    positions: renderPositions, viewSize: renderViewSize)
        updateCamera(dt: dtSec)

        // Detect selection change from gamepad/keyboard → notify parent
        if renderSelectedNode != preInputSelection {
            selectionCallback?(renderSelectedNode)
        }

        // --- Single-loop: tick simulation + projection directly (no closures crossing SwiftUI boundary) ---
        var didUpdatePositions = false
        if let sim = simulation3D, let proj = embeddingProjection {
            sim.tick()
            proj.tickAnimation3D()

            // Compute positions (same logic as GraphView.positions3D, but reads objects directly)
            let newPositions: [UUID: SIMD3<Float>]?
            if sim.isSettled && proj.is3DAnimationSettled {
                newPositions = nil  // settled — no position update needed
            } else if renderLayoutMode == .embedding {
                let tsne3D = proj.projectedPositions3D
                if tsne3D.isEmpty {
                    newPositions = sim.positions
                } else if transitionProgress >= 1.0 {
                    newPositions = tsne3D
                } else {
                    // Lerp between force snapshot and t-SNE
                    var blended: [UUID: SIMD3<Float>] = [:]
                    let allIds = Set(forcePositionSnapshot3D.keys).union(tsne3D.keys)
                    for id in allIds {
                        let forcePos = forcePositionSnapshot3D[id] ?? sim.positions[id] ?? .zero
                        let tsnePos = tsne3D[id] ?? forcePos
                        blended[id] = forcePos + (tsnePos - forcePos) * Float(transitionProgress)
                    }
                    newPositions = blended
                }
            } else {
                newPositions = sim.positions
            }
            if let positions = newPositions {
                renderPositions = positions
                didUpdatePositions = true

                // Camera centering (first 3 seconds after positions appear)
                if cameraStartTime == nil { cameraStartTime = Date() }
                let elapsed = Date().timeIntervalSince(cameraStartTime!)
                if (!hasCenteredCamera || elapsed < 3.0) && !isDragging {
                    centerTickCount += 1
                    if centerTickCount % 6 == 0 {
                        centerOnGraph(positions: renderPositions)
                    }
                    if elapsed >= 3.0 { hasCenteredCamera = true }
                }
            }
        }

        // Consume pending hub toggles — pin/unpin children in simulation directly
        if !pendingHubToggles.isEmpty, let sim = simulation3D {
            for toggle in pendingHubToggles {
                let children = renderEdges.filter { $0.relation == "part_of" && $0.targetId == toggle.hubId }.map(\.sourceId)
                for childId in children {
                    if toggle.expanding { sim.pin(childId) } else { sim.unpin(childId) }
                }
            }
            pendingHubToggles.removeAll()
        }

        // --- Idle detection ---
        // Camera still lerping?
        let cameraMoving = abs(targetAzimuth - azimuth) > 0.0001
            || abs(targetElevation - elevation) > 0.0001
            || simd_length(targetCameraPos - cameraTarget) > 0.01
        let hasInput = !heldKeys.isEmpty || GCController.current?.extendedGamepad != nil
            && (abs(GCController.current?.extendedGamepad?.leftThumbstick.xAxis.value ?? 0) > 0.1
             || abs(GCController.current?.extendedGamepad?.leftThumbstick.yAxis.value ?? 0) > 0.1
             || abs(GCController.current?.extendedGamepad?.rightThumbstick.xAxis.value ?? 0) > 0.1
             || abs(GCController.current?.extendedGamepad?.rightThumbstick.yAxis.value ?? 0) > 0.1)
        let hasExpansions = !expandedHubs.isEmpty
        let hasParticles = !flowParticles.isEmpty
        let hasGlowing = !renderGlowingNodes.isEmpty || !renderNewNodes.isEmpty || !renderDyingNodes.isEmpty
        let selectionChanged = renderSelectedNode != renderLastSelectedNode
        let searchChanged = renderIsSearchActive != renderLastSearchActive
            || renderSearchMatchIds != renderLastSearchMatchIds

        // Position change detection: use the authoritative flag from the simulation
        // update path. The old hash-based approach was unreliable — Dictionary iteration
        // order is non-deterministic, so the 3-value hash could collide even when
        // positions changed, causing skipped buffer writes and stale-buffer flicker.
        let positionsChanged = didUpdatePositions

        // When positions change, set dirty counter to flush all RealityKit internal
        // LowLevelMesh buffers (double/triple-buffered). This ensures the GPU never
        // renders from a stale or zero-initialized buffer.
        if positionsChanged || selectionChanged || hasGlowing || searchChanged || hasExpansions {
            meshDirtyFrames = 10
        }

        // Scene-level changes that require entity updates (nodes, edges, nebulae).
        // meshDirtyFrames > 0 means we still need to write buffers to flush internal
        // double/triple-buffering, even if nothing changed THIS frame.
        let sceneNeedsUpdate = meshDirtyFrames > 0
        // Camera-only activity (orbit, keyboard) doesn't need entity work —
        // GPU shaders compute fog/lighting from camera position autonomously.
        let isActive = sceneNeedsUpdate || cameraMoving || hasInput || hasParticles

        if !isActive {
            idleFrameCount += 1
            // When idle: skip all heavy work. Only wake every 30th frame
            // for a cheap maintenance pass (nebula breathing, etc.)
            if idleFrameCount > 10 && idleFrameCount % 30 != 0 {
                lastRenderTime = now
                renderFrameCount &+= 1
                return
            }
        } else {
            idleFrameCount = 0
        }

        animationTime += dtSec

        if renderFrameCount == 10 {
            frameLog.error("[DIAG] SKIP_NEBULAE=\(Self.DIAG_SKIP_NEBULAE) SKIP_POINT_LIGHTS=\(Self.DIAG_SKIP_POINT_LIGHTS) SKIP_GEOM_MOD=\(Self.DIAG_SKIP_GEOM_MOD) SKIP_NODE_BATCH=\(Self.DIAG_SKIP_NODE_BATCH) SKIP_EDGE_BATCH=\(Self.DIAG_SKIP_EDGE_BATCH) SKIP_LABEL_BATCH=\(Self.DIAG_SKIP_LABEL_BATCH)")
        }

        let t0 = CFAbsoluteTimeGetCurrent()
        var msExpand = 0.0, msNodes = 0.0, msEdges = 0.0, msNeb = 0.0, msFlow = 0.0

        // Write node/edge/nebula data when scene state changed.
        // Node batch uses LowLevelMesh.replace(bufferIndex:using:) for GPU compute —
        // properly synchronized with RealityKit's triple-buffering, no keep-alive needed.
        if sceneNeedsUpdate {
            // Hub expansion animation (before node/edge updates so positions are current)
            updateExpansions(dt: dtSec)
            let tExpand = CFAbsoluteTimeGetCurrent()
            msExpand = (tExpand - t0) * 1000

            // Batched nodes: Metal compute stamps sphere instances → single LowLevelMesh.
            if !Self.DIAG_SKIP_NODE_BATCH {
                updateNodeBatch(
                    positions: renderPositions, nodes: renderNodes, hubs: renderHubs,
                    colorMap: renderColorMap, selectedNode: renderSelectedNode,
                    glowingNodes: renderGlowingNodes, newNodes: renderNewNodes
                )
            }
            let tNodes = CFAbsoluteTimeGetCurrent()
            msNodes = (tNodes - tExpand) * 1000

            // Dying nodes: individual PBR entities (max ~5, temporary)
            if !renderDyingNodes.isEmpty || !dyingNodeEntities.isEmpty {
                updateDyingNodes(positions: renderPositions, dyingNodes: renderDyingNodes)
            }

            // Batched edge mesh: rebuild when positions, selection, or search changes.
            // meshDirtyFrames flushes all triple-buffers after state changes.
            if !Self.DIAG_SKIP_EDGE_BATCH,
               positionsChanged || selectionChanged || searchChanged || hasExpansions || meshDirtyFrames > 0 {
                updateEdgeBatch(
                    positions: renderPositions, edges: renderEdges,
                    nodes: renderNodes, hubs: renderHubs,
                    layoutMode: renderLayoutMode,
                    colorMap: renderColorMap, selectedNode: renderSelectedNode
                )
                renderLastSelectedNode = renderSelectedNode
                renderLastSearchActive = renderIsSearchActive
                renderLastSearchMatchIds = renderSearchMatchIds
            }
            let tEdges = CFAbsoluteTimeGetCurrent()
            msEdges = (tEdges - tNodes) * 1000

            // Edge flow particles — only when selection exists
            if renderSelectedNode != nil || !flowParticles.isEmpty {
                updateFlowParticles(dt: dtSec)
            }
            let tFlow = CFAbsoluteTimeGetCurrent()
            msFlow = (tFlow - tEdges) * 1000

            // Nebulae
            if renderFrameCount > 180 {
                if renderFrameCount % 30 == 0 {
                    updateNebulae()
                } else if positionsChanged && renderFrameCount % 4 == 0 {
                    updateNebulaBreathing()
                }
            }
            msNeb = (CFAbsoluteTimeGetCurrent() - tFlow) * 1000

            // Decrement dirty counter — ensures we write buffers for N consecutive
            // frames to flush all RealityKit internal double/triple-buffers.
            meshDirtyFrames = max(0, meshDirtyFrames - 1)
        }

        // GPU billboard labels — update when scene changes or camera moves (for LOD/opacity).
        // Billboarding itself is handled by the geometry modifier on GPU.
        // Must update every frame (no throttle) to avoid stale data in triple-buffered mesh.
        if !Self.DIAG_SKIP_LABEL_BATCH,
           sceneNeedsUpdate || cameraMoving {
            updateLabelBatch(positions: renderPositions, nodes: renderNodes,
                             hubs: renderHubs, selectedNode: renderSelectedNode)
        }

        let elapsed = (CFAbsoluteTimeGetCurrent() - now) * 1000.0

        // Count total entities in scene (used for both logging and CSV)
        func countEntities(_ e: Entity) -> Int {
            1 + e.children.reduce(0) { $0 + countEntities($1) }
        }
        let totalEntities = countEntities(rootEntity)

        renderLogCounter &+= 1
        if renderLogCounter % 60 == 0 || dt > 25 || elapsed > 10 {
            frameLog.error("[3D-render] dt=\(dt, format: .fixed(precision: 1))ms work=\(elapsed, format: .fixed(precision: 2))ms expand=\(msExpand, format: .fixed(precision: 2)) nodes=\(msNodes, format: .fixed(precision: 2)) edges=\(msEdges, format: .fixed(precision: 2)) flow=\(msFlow, format: .fixed(precision: 2)) neb=\(msNeb, format: .fixed(precision: 2)) | n=\(self.renderPositions.count) e=\(self.renderEdges.count) entities=\(totalEntities) nebEmitters=\(self.nebulaEmitters.count) ptLights=\(self.pointLightEntities.count) frame=\(self.renderFrameCount) idle=\(self.idleFrameCount)")
        }

        #if ENGRAM_INSTRUMENTATION
        // Write every frame to CSV for profiling analysis
        if profilingReady {
            let idleFlag = isActive ? 0 : 1
            // Build reason flags: P=posChanged S=selChanged E=expansions G=glowing X=searchChanged C=cameraMoving K=keyInput
            var reason = ""
            if positionsChanged { reason += "P" }
            if selectionChanged { reason += "S" }
            if hasExpansions { reason += "E" }
            if hasGlowing { reason += "G" }
            if searchChanged { reason += "X" }
            if cameraMoving { reason += "C" }
            if hasInput { reason += "K" }
            if reason.isEmpty { reason = "-" }
            profilingLines.append("\(renderFrameCount),\(String(format: "%.2f", dt)),\(String(format: "%.2f", elapsed)),\(String(format: "%.2f", msExpand)),\(String(format: "%.2f", msNodes)),\(String(format: "%.2f", msEdges)),\(String(format: "%.2f", msNeb)),\(renderPositions.count),\(renderEdges.count),\(String(format: "%.2f", lastCanvasLabelMs)),\(String(format: "%.2f", msFlow)),\(idleFlag),\(totalEntities),\(reason)\n")
            if renderLogCounter % 60 == 0 { flushProfiling() }
        }
        #endif

        // Report camera state directly (writes to Camera3DState @Observable —
        // only MinimapView re-renders, GraphView body is NOT re-evaluated).
        // Only write when values differ to avoid triggering SwiftUI observation every frame.
        if let camState = camera3DState {
            if azimuth != camState.azimuth { camState.azimuth = azimuth }
            if cameraPosition != camState.position { camState.position = cameraPosition }
            if cameraTarget != camState.target { camState.target = cameraTarget }
            if didUpdatePositions { camState.positions = renderPositions }
        }

        lastRenderTime = now
        renderFrameCount &+= 1

        // DIAGNOSTIC: main queue saturation probe — measures how much main-thread work
        // is queued between renderTick calls. If this delta is large (~100ms), something
        // on the main queue (SwiftUI body evals, Canvas draws, etc.) is starving SceneEvents.Update.
        let probePostTime = CFAbsoluteTimeGetCurrent()
        let probeFrame = renderFrameCount
        DispatchQueue.main.async {
            let probeDelta = (CFAbsoluteTimeGetCurrent() - probePostTime) * 1000.0
            if probeDelta > 5.0 {
                frameLog.error("[PROBE] frame=\(probeFrame) mainQ-delay=\(probeDelta, format: .fixed(precision: 1))ms")
            }
        }
    }
}
