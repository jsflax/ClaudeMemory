import Foundation

/// Atlas rect packing types and greedy row-packing algorithm.

/// A packed rectangle in the atlas.
public struct PackedRect: Sendable {
    public let x: Int
    public let y: Int
    public let width: Int
    public let height: Int

    public init(x: Int, y: Int, width: Int, height: Int) {
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    }
}

/// Greedy row-based rectangle packing.
///
/// Places rects left-to-right, top-to-bottom. Simple and fast.
/// Input rects should be sorted by height descending for best results.
public func packRects(
    sizes: [(width: Int, height: Int)],
    atlasWidth: Int,
    atlasHeight: Int
) -> [PackedRect?] {
    var result: [PackedRect?] = Array(repeating: nil, count: sizes.count)
    var curX = 0
    var curY = 0
    var rowHeight = 0

    for i in 0..<sizes.count {
        let w = sizes[i].width
        let h = sizes[i].height

        // Wrap to next row if needed
        if curX + w > atlasWidth {
            curX = 0
            curY += rowHeight
            rowHeight = 0
        }

        // Check vertical fit
        if curY + h > atlasHeight {
            continue  // Doesn't fit — skip
        }

        result[i] = PackedRect(x: curX, y: curY, width: w, height: h)
        curX += w
        rowHeight = max(rowHeight, h)
    }

    return result
}
