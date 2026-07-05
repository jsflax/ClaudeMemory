import Foundation
import simd

/// Abstracts input for platform-agnostic interaction.
///
/// On macOS: keyboard/gamepad/mouse.
/// On visionOS: eye tracking, hand gestures, pinch.
@MainActor
public protocol InputProvider: AnyObject {
    /// Currently selected node (tap/click).
    var selectedNode: UUID? { get set }
    /// Currently expanded hubs.
    var expandedHubs: Set<UUID> { get }
    /// Nodes matching the current search query.
    var searchMatchIds: Set<UUID> { get }
    /// Whether search mode is active.
    var isSearchActive: Bool { get }
    /// Hit test at a 3D world-space point, returning the nearest node ID.
    func hitTest(at point: SIMD3<Float>) -> UUID?
}
