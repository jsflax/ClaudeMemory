import Metal
import CoreText
import AppKit
import simd
import os
import EngramSceneKit

private let atlasLog = Logger(subsystem: "io.engram.app", category: "LabelAtlas")

/// Generates a packed label atlas MTLTexture from CoreText-rendered labels.
/// Owns all atlas-related state (rects, sizing, regen bookkeeping).
@MainActor
final class LabelAtlasGenerator {

    let device: MTLDevice

    // Atlas rect lookups — keyed by node UUID / project name / galaxy name / topic string
    var labelAtlasRects: [UUID: (u0: Float, v0: Float, u1: Float, v1: Float)] = [:]
    var projectLabelAtlasRects: [String: (u0: Float, v0: Float, u1: Float, v1: Float)] = [:]
    var galaxyLabelAtlasRects: [String: (u0: Float, v0: Float, u1: Float, v1: Float)] = [:]
    var topicLabelAtlasRects: [String: (u0: Float, v0: Float, u1: Float, v1: Float)] = [:]

    // Which IDs/projects/galaxies are currently in the atlas
    var labelAtlasNodeIds: Set<UUID> = []
    var labelAtlasHubIds: Set<UUID> = []
    var labelAtlasProjects: Set<String> = []
    var labelAtlasGalaxyNames: [String] = []

    // Sizing & scheduling
    var lastAtlasTopicGroupCount: Int = 0
    var labelAtlasAspectCorrection: Float = 1.0
    var labelAtlasAllocW: Int = 0
    var labelAtlasAllocH: Int = 0
    var labelAtlasRegenFrame: UInt64 = 0
    var lastAtlasTopologyVersion: UInt64 = 0
    var isAtlasGenerating = false
    var pendingAtlasRegen = false

    init(device: MTLDevice) {
        self.device = device
    }

    // MARK: - Types

    struct AtlasLabelEntry: Sendable {
        let id: UUID; let label: String; let isHub: Bool
    }

    struct AtlasResult: @unchecked Sendable {
        let texture: MTLTexture
        let nodeRects: [UUID: (u0: Float, v0: Float, u1: Float, v1: Float)]
        let projectRects: [String: (u0: Float, v0: Float, u1: Float, v1: Float)]
        let galaxyRects: [String: (u0: Float, v0: Float, u1: Float, v1: Float)]
        let topicRects: [String: (u0: Float, v0: Float, u1: Float, v1: Float)]
        let aspectCorrection: Float
        let allocW: Int; let allocH: Int
    }

    // MARK: - Public API

    /// Synchronous atlas generation (used for the first atlas only).
    func generateLabelAtlas(nodes: [NodeData], hubs: Set<UUID>, projects: Set<String>,
                            galaxyNames: [String], topicLabels: [String],
                            bufferManager: InstanceBufferManager) {
        let entries = nodes.map { AtlasLabelEntry(id: $0.id, label: $0.label, isHub: hubs.contains($0.id)) }
        let monoMedium: CTFont = NSFont.monospacedSystemFont(ofSize: 28, weight: .medium) as CTFont
        let monoBold: CTFont = NSFont.monospacedSystemFont(ofSize: 28, weight: .bold) as CTFont
        let projCTFont: CTFont = NSFont.systemFont(ofSize: 40, weight: .bold) as CTFont
        guard let result = Self.renderLabelAtlas(
            entries: entries, sortedProjects: projects.sorted(), galaxyNames: galaxyNames,
            topicLabels: topicLabels,
            device: device,
            monoMedium: monoMedium, monoBold: monoBold, projFont: projCTFont
        ) else { return }
        applyAtlasResult(result, nodeIds: Set(nodes.map(\.id)), hubs: hubs, projects: projects,
                          galaxyNames: galaxyNames, bufferManager: bufferManager)
        atlasLog.info("[labelAtlas] \(result.nodeRects.count) node + \(result.projectRects.count) project + \(result.galaxyRects.count) galaxy labels, \(result.allocW/2)x\(result.allocH/2)")
    }

    /// Dispatch label atlas regeneration to a background thread. Old atlas stays visible until complete.
    func dispatchAtlasRegen(nodes: [NodeData], hubs: Set<UUID>, projects: Set<String>,
                            galaxyNames: [String], topicLabels: [String],
                            bufferManager: InstanceBufferManager,
                            renderFrameCount: UInt64) {
        isAtlasGenerating = true
        // Minimal main-thread prep: only sorts + font creation (~0.3ms).
        // O(n) entries mapping and Set creation are deferred to the background thread.
        let sortedProjects = projects.sorted()
        let device = device
        let capturedNodes = nodes
        let capturedHubs = hubs
        let capturedProjects = projects

        // Create fonts on the main thread (NSFont is safe here), then pass CTFont refs
        // to the background. CTFont is immutable and thread-safe; this avoids NSFont
        // shared cache lock contention from the background thread.
        let monoMedium: CTFont = NSFont.monospacedSystemFont(ofSize: 28, weight: .medium) as CTFont
        let monoBold: CTFont = NSFont.monospacedSystemFont(ofSize: 28, weight: .bold) as CTFont
        let projCTFont: CTFont = NSFont.systemFont(ofSize: 40, weight: .bold) as CTFont

        #if ENGRAM_INSTRUMENTATION
        let dispatchFrame = renderFrameCount
        #endif

        Task.detached(priority: .userInitiated) { [weak self] in
            #if ENGRAM_INSTRUMENTATION
            let bgStart = CFAbsoluteTimeGetCurrent()
            #endif
            // O(n) work done on background thread to reduce insert-frame jitter
            let entries = capturedNodes.map { AtlasLabelEntry(id: $0.id, label: $0.label, isHub: capturedHubs.contains($0.id)) }
            let capturedNodeIds = Set(capturedNodes.map(\.id))

            let result = Self.renderLabelAtlas(
                entries: entries, sortedProjects: sortedProjects, galaxyNames: galaxyNames,
                topicLabels: topicLabels,
                device: device,
                monoMedium: monoMedium, monoBold: monoBold, projFont: projCTFont
            )
            #if ENGRAM_INSTRUMENTATION
            let bgMs = (CFAbsoluteTimeGetCurrent() - bgStart) * 1000.0
            #endif
            await MainActor.run { [weak self] in
                #if ENGRAM_INSTRUMENTATION
                let applyStart = CFAbsoluteTimeGetCurrent()
                #endif
                guard let self else { return }
                if let result {
                    self.applyAtlasResult(result, nodeIds: capturedNodeIds, hubs: capturedHubs,
                                           projects: capturedProjects, galaxyNames: galaxyNames,
                                           bufferManager: bufferManager)
                }
                self.isAtlasGenerating = false
                #if ENGRAM_INSTRUMENTATION
                let applyMs = (CFAbsoluteTimeGetCurrent() - applyStart) * 1000.0
                let msg = "dispatched=frame\(dispatchFrame) bg=\(String(format: "%.1f", bgMs))ms apply=\(String(format: "%.1f", applyMs))ms currentFrame=\(renderFrameCount)\n"
                if let data = msg.data(using: .utf8) {
                    let url = URL(fileURLWithPath: "/tmp/atlas-timing.log")
                    if let fh = try? FileHandle(forWritingTo: url) {
                        fh.seekToEndOfFile(); fh.write(data); fh.closeFile()
                    } else { try? data.write(to: url) }
                }
                #endif
            }
        }
    }

    // MARK: - Internal

    private func applyAtlasResult(_ result: AtlasResult, nodeIds: Set<UUID>, hubs: Set<UUID>,
                                    projects: Set<String>, galaxyNames: [String],
                                    bufferManager: InstanceBufferManager) {
        bufferManager.labelAtlasTexture = result.texture
        labelAtlasRects = result.nodeRects
        projectLabelAtlasRects = result.projectRects
        galaxyLabelAtlasRects = result.galaxyRects
        topicLabelAtlasRects = result.topicRects
        labelAtlasAspectCorrection = result.aspectCorrection
        labelAtlasAllocW = result.allocW
        labelAtlasAllocH = result.allocH
        labelAtlasNodeIds = nodeIds
        labelAtlasHubIds = hubs
        labelAtlasProjects = projects
        labelAtlasGalaxyNames = galaxyNames
    }

    // MARK: - Static Rendering

    /// Pure rendering work — can run on any thread.
    /// Uses CoreText (CTLine) + CGContext directly — no AppKit (NSGraphicsContext/NSString.draw)
    /// to avoid internal AppKit lock contention with the main thread.
    /// Fonts are pre-created on the main thread and passed in (CTFont is immutable, thread-safe).
    nonisolated static func renderLabelAtlas(
        entries: [AtlasLabelEntry], sortedProjects: [String], galaxyNames: [String],
        topicLabels: [String],
        device: MTLDevice,
        monoMedium: CTFont, monoBold: CTFont, projFont: CTFont
    ) -> AtlasResult? {
        let atlasW = 4096
        let padding: CGFloat = 4

        // --- Pure CoreText measurement + line creation (no AppKit calls) ---
        let white = CGColor(red: 1, green: 1, blue: 1, alpha: 1)

        func makeLine(_ text: String, font: CTFont) -> (line: CTLine, width: CGFloat, ascent: CGFloat, descent: CGFloat, leading: CGFloat) {
            let attrs: [CFString: Any] = [
                kCTFontAttributeName: font,
                kCTForegroundColorAttributeName: white
            ]
            let attrStr = CFAttributedStringCreate(nil, text as CFString, attrs as CFDictionary)!
            let line = CTLineCreateWithAttributedString(attrStr)
            var ascent: CGFloat = 0, descent: CGFloat = 0, leading: CGFloat = 0
            let width = CGFloat(CTLineGetTypographicBounds(line, &ascent, &descent, &leading))
            return (line, width, ascent, descent, leading)
        }

        struct LabelSize { let width: CGFloat; let height: CGFloat }
        struct LabelInfo { let line: CTLine; let size: LabelSize; let ascent: CGFloat }

        // Measure node labels
        var nodeInfos: [LabelInfo] = []
        nodeInfos.reserveCapacity(entries.count)
        for entry in entries {
            let font = entry.isHub ? monoBold : monoMedium
            let m = makeLine(entry.label, font: font)
            let w = ceil(m.width) + padding * 2
            let h = ceil(m.ascent + m.descent + m.leading) + padding
            nodeInfos.append(LabelInfo(line: m.line, size: LabelSize(width: w, height: h), ascent: m.ascent))
        }

        // Measure project labels
        var projInfos: [LabelInfo] = []
        for project in sortedProjects {
            let m = makeLine(project, font: projFont)
            let w = ceil(m.width) + padding * 2
            let h = ceil(m.ascent + m.descent + m.leading) + padding
            projInfos.append(LabelInfo(line: m.line, size: LabelSize(width: w, height: h), ascent: m.ascent))
        }

        // Measure galaxy labels (larger than project labels)
        var galaxyInfos: [LabelInfo] = []
        for name in galaxyNames {
            let m = makeLine(name.uppercased(), font: projFont)
            let w = ceil(m.width) + padding * 2
            let h = ceil(m.ascent + m.descent + m.leading) + padding
            galaxyInfos.append(LabelInfo(line: m.line, size: LabelSize(width: w, height: h), ascent: m.ascent))
        }

        // Measure topic labels (force-directed topic group labels)
        let topicFont: CTFont = projFont
        var topicInfos: [LabelInfo] = []
        for label in topicLabels {
            let m = makeLine(label, font: topicFont)
            let w = ceil(m.width) + padding * 2
            let h = ceil(m.ascent + m.descent + m.leading) + padding
            topicInfos.append(LabelInfo(line: m.line, size: LabelSize(width: w, height: h), ascent: m.ascent))
        }

        // Simulate packing
        var cursorX: CGFloat = 2
        var cursorY: CGFloat = 0
        var rowHeight: CGFloat = 0
        for info in nodeInfos {
            let s = info.size
            if cursorX + s.width > CGFloat(atlasW) { cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0 }
            cursorX += s.width + padding; rowHeight = max(rowHeight, s.height)
        }
        cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0
        for info in projInfos {
            let s = info.size
            if cursorX + s.width > CGFloat(atlasW) { cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0 }
            cursorX += s.width + padding; rowHeight = max(rowHeight, s.height)
        }
        cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0
        for info in galaxyInfos {
            let s = info.size
            if cursorX + s.width > CGFloat(atlasW) { cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0 }
            cursorX += s.width + padding; rowHeight = max(rowHeight, s.height)
        }
        cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0
        for info in topicInfos {
            let s = info.size
            if cursorX + s.width > CGFloat(atlasW) { cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0 }
            cursorX += s.width + padding; rowHeight = max(rowHeight, s.height)
        }
        let neededH = Int(ceil(cursorY + rowHeight + padding))

        // Fall back to 1x scale if 2x would exceed Metal's max texture dimension (16384).
        // Labels that still don't fit are gracefully skipped (break guards in drawing loops).
        let maxDim = 16384
        let scale = (max(512, neededH + 32) * 2 <= maxDim) ? 2 : 1
        let allocW = atlasW * scale
        let allocH = min(max(512, neededH + 32) * scale, maxDim)

        let uvAtlasW = allocW / scale
        let uvAtlasH = allocH / scale
        let aspectCorrection = Float(uvAtlasW) / (2.0 * Float(uvAtlasH))

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil, width: allocW, height: allocH,
            bitsPerComponent: 8, bytesPerRow: allocW * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }

        ctx.translateBy(x: 0, y: CGFloat(allocH))
        ctx.scaleBy(x: CGFloat(scale), y: CGFloat(-scale))

        // Draw node labels using CTLineDraw (no AppKit locks)
        var rects: [UUID: (u0: Float, v0: Float, u1: Float, v1: Float)] = [:]
        cursorX = 2; cursorY = 0; rowHeight = 0

        for (i, entry) in entries.enumerated() {
            let info = nodeInfos[i]
            let s = info.size
            if cursorX + s.width > CGFloat(atlasW) { cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0 }
            if cursorY + s.height > CGFloat(uvAtlasH) { break }

            // CTLineDraw renders glyphs upward from baseline, but the CGContext has
            // a negative y-scale (flipped). Locally unflip at the draw point so text
            // renders right-side up, then restore.
            ctx.saveGState()
            ctx.translateBy(x: cursorX + padding, y: cursorY + padding * 0.5 + info.ascent)
            ctx.scaleBy(x: 1, y: -1)
            ctx.textPosition = .zero
            CTLineDraw(info.line, ctx)
            ctx.restoreGState()

            rects[entry.id] = (
                Float(cursorX) / Float(uvAtlasW), Float(cursorY) / Float(uvAtlasH),
                Float(cursorX + s.width) / Float(uvAtlasW), Float(cursorY + s.height) / Float(uvAtlasH)
            )
            cursorX += s.width + padding; rowHeight = max(rowHeight, s.height)
        }

        // Draw project labels using CTLineDraw (no AppKit locks)
        var projRects: [String: (u0: Float, v0: Float, u1: Float, v1: Float)] = [:]
        cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0

        for (i, project) in sortedProjects.enumerated() {
            let info = projInfos[i]
            let s = info.size
            if cursorX + s.width > CGFloat(atlasW) { cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0 }
            if cursorY + s.height > CGFloat(uvAtlasH) { break }

            ctx.saveGState()
            ctx.translateBy(x: cursorX + padding, y: cursorY + padding * 0.5 + info.ascent)
            ctx.scaleBy(x: 1, y: -1)
            ctx.textPosition = .zero
            CTLineDraw(info.line, ctx)
            ctx.restoreGState()

            projRects[project] = (
                Float(cursorX) / Float(uvAtlasW), Float(cursorY) / Float(uvAtlasH),
                Float(cursorX + s.width) / Float(uvAtlasW), Float(cursorY + s.height) / Float(uvAtlasH)
            )
            cursorX += s.width + padding; rowHeight = max(rowHeight, s.height)
        }

        // Draw galaxy labels using CTLineDraw
        var galaxyRects: [String: (u0: Float, v0: Float, u1: Float, v1: Float)] = [:]
        cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0

        for (i, name) in galaxyNames.enumerated() {
            let info = galaxyInfos[i]
            let s = info.size
            if cursorX + s.width > CGFloat(atlasW) { cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0 }
            if cursorY + s.height > CGFloat(uvAtlasH) { break }

            ctx.saveGState()
            ctx.translateBy(x: cursorX + padding, y: cursorY + padding * 0.5 + info.ascent)
            ctx.scaleBy(x: 1, y: -1)
            ctx.textPosition = .zero
            CTLineDraw(info.line, ctx)
            ctx.restoreGState()

            galaxyRects[name] = (
                Float(cursorX) / Float(uvAtlasW), Float(cursorY) / Float(uvAtlasH),
                Float(cursorX + s.width) / Float(uvAtlasW), Float(cursorY + s.height) / Float(uvAtlasH)
            )
            cursorX += s.width + padding; rowHeight = max(rowHeight, s.height)
        }

        // Draw topic labels using CTLineDraw
        var topicRects: [String: (u0: Float, v0: Float, u1: Float, v1: Float)] = [:]
        cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0

        for (i, label) in topicLabels.enumerated() {
            let info = topicInfos[i]
            let s = info.size
            if cursorX + s.width > CGFloat(atlasW) { cursorX = 2; cursorY += rowHeight + padding; rowHeight = 0 }
            if cursorY + s.height > CGFloat(uvAtlasH) { break }

            ctx.saveGState()
            ctx.translateBy(x: cursorX + padding, y: cursorY + padding * 0.5 + info.ascent)
            ctx.scaleBy(x: 1, y: -1)
            ctx.textPosition = .zero
            CTLineDraw(info.line, ctx)
            ctx.restoreGState()

            topicRects[label] = (
                Float(cursorX) / Float(uvAtlasW), Float(cursorY) / Float(uvAtlasH),
                Float(cursorX + s.width) / Float(uvAtlasW), Float(cursorY + s.height) / Float(uvAtlasH)
            )
            cursorX += s.width + padding; rowHeight = max(rowHeight, s.height)
        }

        // Create MTLTexture from CGContext
        guard let pixelData = ctx.data else { return nil }
        let bytesPerRow = allocW * 4

        // Validate dimensions before Metal texture creation (max 16384 on Apple Silicon)
        guard allocW > 0, allocH > 0, allocW <= 16384, allocH <= 16384 else {
            atlasLog.error("[labelAtlas] invalid texture dimensions: \(allocW)x\(allocH)")
            return nil
        }

        let texDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: allocW,
            height: allocH,
            mipmapped: false
        )
        texDesc.usage = [.shaderRead]
        texDesc.storageMode = .shared

        guard let texture = device.makeTexture(descriptor: texDesc) else { return nil }
        texture.replace(
            region: MTLRegion(origin: MTLOrigin(), size: MTLSize(width: allocW, height: allocH, depth: 1)),
            mipmapLevel: 0,
            withBytes: pixelData,
            bytesPerRow: bytesPerRow
        )

        return AtlasResult(
            texture: texture, nodeRects: rects, projectRects: projRects,
            galaxyRects: galaxyRects, topicRects: topicRects,
            aspectCorrection: aspectCorrection, allocW: allocW, allocH: allocH
        )
    }
}
