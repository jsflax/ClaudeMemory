// Pure visual state functions — no @MainActor, no Metal imports.
// All marked @inlinable for cross-module inlining (EngramSceneKit → EngramVisualizer).

/// Recall glow intensity: cubic fade-in (1.0s), hold (1.5s), quadratic fade-out (2.0s).
@inlinable
public func recallGlowIntensity(elapsed: Float) -> Float {
    let fadeIn: Float = 1.0, hold: Float = 1.5, fadeOut: Float = 2.0
    if elapsed < 0 { return 0 }
    if elapsed < fadeIn {
        let t = elapsed / fadeIn
        return t * t * t
    } else if elapsed < fadeIn + hold {
        return 1.0
    } else if elapsed < fadeIn + hold + fadeOut {
        let t = 1.0 - (elapsed - fadeIn - hold) / fadeOut
        return t * t
    }
    return 0
}

/// Arrival glow intensity: cubic fade-in (0.8s), hold (2.0s), quadratic fade-out (3.0s).
@inlinable
public func arrivalGlowIntensity(elapsed: Float) -> Float {
    let fadeIn: Float = 0.8, hold: Float = 2.0, fadeOut: Float = 3.0
    if elapsed < 0 { return 0 }
    if elapsed < fadeIn {
        let t = elapsed / fadeIn
        return t * t * t
    } else if elapsed < fadeIn + hold {
        return 1.0
    } else if elapsed < fadeIn + hold + fadeOut {
        let t = 1.0 - (elapsed - fadeIn - hold) / fadeOut
        return t * t
    }
    return 0
}

/// Birth intensity: linear 0→1 over 1.5s, clamped.
@inlinable
public func birthIntensity(elapsed: Float) -> Float {
    min(max(elapsed / 1.5, 0), 1.0)
}

/// Encode visual state into a single float for GPU consumption.
/// Format: stateType + (searchDimmed ? 10.0 : 0.0) + intensity * 0.01
@inlinable
public func packNodeState(stateType: Float, searchDimmed: Bool, intensity: Float) -> Float {
    stateType + (searchDimmed ? 10.0 : 0.0) + intensity * 0.01
}
