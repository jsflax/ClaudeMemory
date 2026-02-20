import Foundation
import Accelerate

/// Pure t-SNE computation kernel. Runs on `@ForceSimulatorActor` (background). Fully `Sendable`.
///
/// Algorithm:
/// 1. Pairwise cosine distance matrix — O(n²) using vDSP for 384-dim dot products
/// 2. Perplexity calibration — binary search for sigma per point to match target perplexity
/// 3. Symmetrize P matrix — P_ij = (P(j|i) + P(i|j)) / 2n
/// 4. Gradient descent — 1000 iterations with early exaggeration (first 250 iters, 12x),
///    momentum (0.5 early, 0.8 late), learning rate 200. Student-t distribution in low-dim space.
struct TSNEKernel: Sendable {
    struct Input: Sendable {
        let embeddings: [[Float]]  // n x dim
        let ids: [Int64]
        let perplexity: Double     // default 30
        let maxIterations: Int     // default 1000
        let initialPositions: [(x: Double, y: Double)]?  // optional init from force layout
    }

    struct Output: Sendable {
        let positions: [(id: Int64, x: Double, y: Double)]
    }

    @ForceSimulatorActor
    static func compute(
        _ input: Input,
        progress: @Sendable (Double) -> Void,
        onPositions: @Sendable ([(id: Int64, x: Double, y: Double)]) -> Void = { _ in }
    ) -> Output {
        let n = input.embeddings.count
        guard n >= 2 else {
            return Output(positions: input.ids.enumerated().map { (i, id) in
                (id: id, x: Double(i) * 50, y: 0)
            })
        }

        // --- Step 1: Pairwise cosine distance matrix ---
        let dim = input.embeddings[0].count
        guard dim > 0 else {
            return Output(positions: input.ids.map { (id: $0, x: 0, y: 0) })
        }

        // Precompute norms
        var norms = [Float](repeating: 0, count: n)
        for i in 0..<n {
            var sumSq: Float = 0
            vDSP_dotpr(input.embeddings[i], 1, input.embeddings[i], 1, &sumSq, vDSP_Length(dim))
            norms[i] = sqrt(sumSq)
        }

        // Distance matrix (cosine distance = 1 - cosine similarity)
        var dist = [Double](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in (i + 1)..<n {
                var dot: Float = 0
                vDSP_dotpr(input.embeddings[i], 1, input.embeddings[j], 1, &dot, vDSP_Length(dim))
                let denom = norms[i] * norms[j]
                let cosDist: Double = denom > 0 ? Double(1 - dot / denom) : 1.0
                dist[i * n + j] = max(cosDist, 0)
                dist[j * n + i] = max(cosDist, 0)
            }
        }

        progress(0.1)

        // --- Step 2: Perplexity calibration (binary search for sigma per point) ---
        let targetEntropy = log(min(input.perplexity, Double(n - 1) / 3.0))
        var P = [Double](repeating: 0, count: n * n)  // conditional probabilities P(j|i)

        for i in 0..<n {
            var lo: Double = 1e-10
            var hi: Double = 1e4
            var sigma: Double = 1.0

            for _ in 0..<50 {  // binary search iterations
                sigma = (lo + hi) / 2
                let twoSigmaSq = 2.0 * sigma * sigma

                // Compute P(j|i) = exp(-d²/2σ²) / Σ_k exp(-d²_ik/2σ²)
                var sumExp: Double = 0
                for j in 0..<n where j != i {
                    let d = dist[i * n + j]
                    let val = exp(-d * d / twoSigmaSq)
                    P[i * n + j] = val
                    sumExp += val
                }
                if sumExp < 1e-300 { sumExp = 1e-300 }
                // Normalize and compute entropy
                var entropy: Double = 0
                for j in 0..<n where j != i {
                    P[i * n + j] /= sumExp
                    if P[i * n + j] > 1e-300 {
                        entropy -= P[i * n + j] * log(P[i * n + j])
                    }
                }

                if entropy > targetEntropy {
                    hi = sigma
                } else {
                    lo = sigma
                }
            }
        }

        progress(0.2)

        // --- Step 3: Symmetrize P matrix ---
        let twoN = Double(2 * n)
        for i in 0..<n {
            for j in (i + 1)..<n {
                let sym = (P[i * n + j] + P[j * n + i]) / twoN
                P[i * n + j] = max(sym, 1e-12)
                P[j * n + i] = max(sym, 1e-12)
            }
        }

        // --- Step 4: Gradient descent with early exaggeration ---
        var Y = [Double](repeating: 0, count: n * 2)  // 2D positions, interleaved [x0,y0,x1,y1,...]
        // Initialize from force layout positions (for smooth visual transition) or random
        if let initPos = input.initialPositions, initPos.count == n {
            var meanX: Double = 0, meanY: Double = 0
            for p in initPos { meanX += p.x; meanY += p.y }
            meanX /= Double(n); meanY /= Double(n)
            var maxRange: Double = 1e-10
            for p in initPos {
                maxRange = max(maxRange, Swift.abs(p.x - meanX))
                maxRange = max(maxRange, Swift.abs(p.y - meanY))
            }
            let scale = 0.01 / maxRange
            for i in 0..<n {
                Y[i * 2] = (initPos[i].x - meanX) * scale
                Y[i * 2 + 1] = (initPos[i].y - meanY) * scale
            }
        } else {
            for i in 0..<(n * 2) {
                Y[i] = Double.random(in: -0.01...0.01)
            }
        }

        var gains = [Double](repeating: 1, count: n * 2)
        var velocities = [Double](repeating: 0, count: n * 2)

        let earlyExaggerationIters = min(250, input.maxIterations / 4)
        let earlyExaggeration: Double = 12.0
        let learningRate: Double = 200.0

        let lastProgress = 0.2
        let progressRange = 0.8  // 0.2 to 1.0

        for iter in 0..<input.maxIterations {
            let momentum: Double = iter < earlyExaggerationIters ? 0.5 : 0.8
            let exaggeration: Double = iter < earlyExaggerationIters ? earlyExaggeration : 1.0

            // Compute Q distribution (Student-t with 1 DOF)
            var Q = [Double](repeating: 0, count: n * n)
            var sumQ: Double = 0
            for i in 0..<n {
                for j in (i + 1)..<n {
                    let dy0 = Y[i * 2] - Y[j * 2]
                    let dy1 = Y[i * 2 + 1] - Y[j * 2 + 1]
                    let qij = 1.0 / (1.0 + dy0 * dy0 + dy1 * dy1)
                    Q[i * n + j] = qij
                    Q[j * n + i] = qij
                    sumQ += 2 * qij
                }
            }
            if sumQ < 1e-300 { sumQ = 1e-300 }

            // Compute gradients
            var gradients = [Double](repeating: 0, count: n * 2)
            for i in 0..<n {
                var gx: Double = 0
                var gy: Double = 0
                for j in 0..<n where j != i {
                    let pij = P[i * n + j] * exaggeration
                    let qij = Q[i * n + j] / sumQ
                    let mult = 4.0 * (pij - qij) * Q[i * n + j]
                    gx += mult * (Y[i * 2] - Y[j * 2])
                    gy += mult * (Y[i * 2 + 1] - Y[j * 2 + 1])
                }
                gradients[i * 2] = gx
                gradients[i * 2 + 1] = gy
            }

            // Update with adaptive gains and momentum
            for i in 0..<(n * 2) {
                let sameSign = (gradients[i] > 0) == (velocities[i] > 0)
                gains[i] = sameSign ? max(gains[i] * 0.8, 0.01) : gains[i] + 0.2
                velocities[i] = momentum * velocities[i] - learningRate * gains[i] * gradients[i]
                Y[i] += velocities[i]
            }

            // Center the solution
            var meanX: Double = 0, meanY: Double = 0
            for i in 0..<n {
                meanX += Y[i * 2]
                meanY += Y[i * 2 + 1]
            }
            meanX /= Double(n)
            meanY /= Double(n)
            for i in 0..<n {
                Y[i * 2] -= meanX
                Y[i * 2 + 1] -= meanY
            }

            // Progress callback at ~10% intervals
            let iterProgress = lastProgress + progressRange * Double(iter + 1) / Double(input.maxIterations)
            if (iter + 1) % max(1, input.maxIterations / 10) == 0 {
                progress(iterProgress)
            }

            // Emit intermediate positions for progressive animation.
            // During early exaggeration: emit every 25 iters (positions are volatile but
            // frame-level lerp in EmbeddingProjection dampens jumps). After: every 10 iters.
            let emitInterval = iter < earlyExaggerationIters ? 25 : 10
            if (iter + 1) % emitInterval == 0 {
                var interPositions: [(id: Int64, x: Double, y: Double)] = []
                interPositions.reserveCapacity(n)
                for i in 0..<n {
                    interPositions.append((id: input.ids[i], x: Y[i * 2], y: Y[i * 2 + 1]))
                }
                onPositions(interPositions)
            }
        }

        progress(1.0)

        // Build output
        var positions: [(id: Int64, x: Double, y: Double)] = []
        positions.reserveCapacity(n)
        for i in 0..<n {
            positions.append((id: input.ids[i], x: Y[i * 2], y: Y[i * 2 + 1]))
        }
        return Output(positions: positions)
    }
}
