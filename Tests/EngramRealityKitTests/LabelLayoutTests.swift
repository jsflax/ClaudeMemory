import Testing
import Foundation
@testable import EngramRealityKit

@Suite("Label Layout Tests")
struct LabelLayoutTests {

    @Test("Pack single rect")
    func packSingleRect() {
        let sizes = [(width: 100, height: 50)]
        let result = packRects(sizes: sizes, atlasWidth: 512, atlasHeight: 512)
        #expect(result.count == 1)
        #expect(result[0] != nil)
        #expect(result[0]!.x == 0)
        #expect(result[0]!.y == 0)
        #expect(result[0]!.width == 100)
        #expect(result[0]!.height == 50)
    }

    @Test("Pack multiple rects in single row")
    func packSingleRow() {
        let sizes = [
            (width: 100, height: 50),
            (width: 150, height: 50),
            (width: 200, height: 50),
        ]
        let result = packRects(sizes: sizes, atlasWidth: 512, atlasHeight: 512)
        #expect(result.allSatisfy { $0 != nil })
        // All should be on the same row (total width = 450 < 512)
        #expect(result[0]!.y == 0)
        #expect(result[1]!.y == 0)
        #expect(result[2]!.y == 0)
        // Sequential x positions
        #expect(result[1]!.x == 100)
        #expect(result[2]!.x == 250)
    }

    @Test("Pack wraps to next row")
    func packWrapsRow() {
        let sizes = [
            (width: 300, height: 50),
            (width: 300, height: 60),
        ]
        let result = packRects(sizes: sizes, atlasWidth: 512, atlasHeight: 512)
        #expect(result.allSatisfy { $0 != nil })
        // Second rect should wrap (300 + 300 > 512)
        #expect(result[0]!.y == 0)
        #expect(result[1]!.y == 50) // row height of first rect
    }

    @Test("Pack skips rects that don't fit vertically")
    func packSkipsOverflow() {
        let sizes = [
            (width: 300, height: 400),
            (width: 300, height: 400), // Won't fit: wraps to row 2, 400 + 400 > 512
        ]
        let result = packRects(sizes: sizes, atlasWidth: 512, atlasHeight: 512)
        #expect(result[0] != nil)
        #expect(result[1] == nil) // second rect at y=400, needs 400 more → 800 > 512
    }

    @Test("Pack empty input")
    func packEmpty() {
        let result = packRects(sizes: [], atlasWidth: 512, atlasHeight: 512)
        #expect(result.isEmpty)
    }

    @Test("Pack many small rects")
    func packManySmall() {
        let sizes = (0..<100).map { _ in (width: 40, height: 20) }
        let result = packRects(sizes: sizes, atlasWidth: 512, atlasHeight: 512)
        // 512 / 40 = 12 per row, 100 / 12 = ~9 rows, 9 * 20 = 180 < 512
        let packed = result.compactMap { $0 }
        #expect(packed.count == 100)
    }
}
