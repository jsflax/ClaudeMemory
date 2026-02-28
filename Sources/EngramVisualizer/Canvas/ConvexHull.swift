import Foundation

/// Andrew's monotone chain convex hull algorithm — O(n log n).
/// Returns vertices in counter-clockwise order.
enum ConvexHull {
    static func compute(points: [CGPoint]) -> [CGPoint] {
        guard points.count >= 3 else { return points }

        let sorted = points.sorted { $0.x < $1.x || ($0.x == $1.x && $0.y < $1.y) }
        var lower: [CGPoint] = []
        for p in sorted {
            while lower.count >= 2 && cross(lower[lower.count - 2], lower[lower.count - 1], p) <= 0 {
                lower.removeLast()
            }
            lower.append(p)
        }

        var upper: [CGPoint] = []
        for p in sorted.reversed() {
            while upper.count >= 2 && cross(upper[upper.count - 2], upper[upper.count - 1], p) <= 0 {
                upper.removeLast()
            }
            upper.append(p)
        }

        // Remove last point of each half because it's repeated
        lower.removeLast()
        upper.removeLast()
        return lower + upper
    }

    /// Expand hull outward by `amount` pixels (approximate via centroid offset).
    static func expand(_ hull: [CGPoint], by amount: CGFloat) -> [CGPoint] {
        guard hull.count >= 3 else { return hull }
        let cx = hull.map(\.x).reduce(0, +) / CGFloat(hull.count)
        let cy = hull.map(\.y).reduce(0, +) / CGFloat(hull.count)
        return hull.map { p in
            let dx = p.x - cx, dy = p.y - cy
            let dist = max(hypot(dx, dy), 1)
            return CGPoint(x: p.x + dx / dist * amount, y: p.y + dy / dist * amount)
        }
    }

    private static func cross(_ o: CGPoint, _ a: CGPoint, _ b: CGPoint) -> CGFloat {
        (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
    }
}
