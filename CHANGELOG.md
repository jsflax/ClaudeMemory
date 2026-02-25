# Changelog

All notable changes to Engram are documented in this file.

> Formerly "ClaudeMemory" — renamed in v0.12.0 to be tool-agnostic.

## [0.12.2] - 2026-02-25

### Fixed
- **Sparkle auto-update broken** — `sparkle:version` (build number) was hardcoded to `1` for every release, so Sparkle never offered updates. Build number now auto-increments from the previous appcast entry.

## [0.12.1] - 2026-02-25

### Changed
- **Raw Metal renderer** — replaced RealityKit with a custom MTKView pipeline, eliminating 67.5% main thread blocking in `re::DrawingManager::commitFrameInternal()` and 13.5% VFX lock contention from `ParticleEmitterComponent`
- **GPU procedural nebulae** — fBm noise billboards rendered entirely in Metal fragment shaders, replacing RealityKit particle emitters
- **Blinn-Phong lighting** — sphere impostor nodes use analytical normal reconstruction with Blinn-Phong shading tuned for dark-background aesthetic
- **Opaque rendering** — fog baked into `base_color` in shaders instead of `.transparent` blending, avoiding GPU depth sorting of 55K+ triangles
- **Search spotlight** — suppressed recall glow on non-matching nodes for cleaner visual contrast
- **EngramModels library extraction** — core model types (Memory, Edge, Checkpoint, HookState) moved into a separate SPM target; EngramKit re-exports via `@_exported import EngramModels`
- **Account UI** — Apple and Google Sign-In via `AuthenticationServices` and `GoogleSignIn-iOS`, posting identity tokens to cloud sync backend
- **Streaming graph load** — batched node loading off main actor for faster initial render
- **GraphRenderStore** — bypass SwiftUI observation for 3D render data, reducing unnecessary view recomputation
- **Project labels in 3D view** — project name labels rendered in the 3D scene with tighter clustering
- Lattice dependency switched from local path to remote URL (`0.4.0`)

### Fixed
- Node flicker during force simulation convergence
- FTS5 full-text search query handling
- GPU compute label batch sizing and edge buffer synchronization
- Drive-to-project camera animation

## [0.12.0] - 2026-02-22

### Changed
- **Rebrand to Engram** — product name, Xcode project, bundle ID (`io.engram.app`), CI/CD artifacts, and documentation all renamed. CLI tool names (`memory` / `memory-hooks`) and MCP server name (`memory`) are unchanged.
- **Fire-and-forget memory maintenance** — maintenance agent now spawns as a detached `claude` CLI subprocess (same pattern as session-learner) instead of injecting a nudge into conversation context. Saves tokens and turns in the main conversation.
- Extracted shared `spawnClaudeSubprocess()` utility used by both session-learner and maintenance spawners
- Maintenance nudge removed from all hook handlers (SessionStart, PreToolUse, PostToolUseFailure, PreCompact) — only the Advise hook spawns maintenance now
- Subprocess allowed tools use wildcard `mcp__memory__*` instead of enumerating each tool

## [0.11.0] - 2026-02-21

### Added
- **Sparkle auto-update** — in-app "Check for Updates..." menu item with Ed25519-signed appcast feed
- **DMG distribution** — notarized DMG with drag-to-Applications install, built in CI
- **CLI sync on launch** — app bundles MCP server, hooks, and agents; auto-installs to `~/.claude/bin/` when app version is newer
- **3D hub expansion** — tap a hub node to orbit its children in a Fibonacci sphere arrangement
- **3D edge flow particles** — animated particles traveling along edges of selected/expanded nodes
- **3D search spotlight** — matching nodes glow cyan, non-matches dim to 12% opacity

### Changed
- Release CI pipeline overhauled: xcodebuild archive, Developer ID signing, DMG creation, notarization, Sparkle appcast generation
- `appcast.xml` committed to repo root and auto-updated by CI on each release
- `install.sh` now mentions DMG download as an alternative

## [0.10.0] - 2026-02-21

### Added
- **3D graph visualization** — full RealityKit-based 3D view with orbit camera, fog, nebula particle clusters, and depth-sorted canvas labels with shadow outlines
- **Gamepad support** — FPS-style controls: L-stick move, R-stick look, triggers rise/descend, A select, B deselect, X/Y teleport prev/next project, bumpers cycle nodes
- **Keyboard teleport** — T/R keys teleport to next/previous project (equivalent to gamepad Y/X)
- **Teleport to hub node** — camera jumps to the project's hub memory (target of `part_of` edges) with tight orbit radius, falling back to centroid
- **Live 3D updates** — new memories and edges appear incrementally without app restart via `ForceSimulation3D.addNode()`/`addEdge()`
- **Progressive t-SNE animation** — nodes smoothly drift from force layout to semantic positions as t-SNE converges, with lerp-based display/target separation
- **ScreenCaptureKit export** — Export as PNG captures Metal/RealityKit content correctly (replaces obsoleted `CGWindowListCreateImage`)
- **UI test suite** — Xcode UI test target with frame timing profiler and teleport proximity verification
- **Xcode project** — `Engram.xcodeproj` for building/testing outside SPM
- `set_project` parameter on `update` tool for moving memories between project scopes

### Changed
- Session-learner rewritten as fire-and-forget `claude` CLI subprocess (eliminates Conductor UI collision)
- Session-learner always spawns when transcript is present (removed productive-tool-call threshold)
- Uninstall script now cleans up all installed components (hooks, agents, binary)
- `.mcp.json` added to `.gitignore`

### Improved
- **3D render performance** — opacity caching skips redundant ECS mutations, edge distance culling hides off-screen edges, position-change detection gates edge geometry updates, label cap at 80 nearest, shadow filter replaces 5-draw stroke (62% work time reduction)
- **3D drag sync** — `isDragging` flag skips camera lerp during drag so labels and entities move in lockstep
- **Activity panel selection** — two-way sync via `lastSyncedSelection` prevents Timer from stomping binding changes

### Fixed
- Teleport double-scaling bug: camera target was scaled by `scaleFactor` twice (once in teleport, again in `cameraTransform`), placing orbit center near origin instead of the target node
- Teleport label stomping: rapid teleports no longer cancel each other's dismiss timers (counter-based task ID)
- Session-learner infinite recursion via `CLAUDE_MEMORY_LEARNER` env var guard
- Conductor UI collision from session-learner nudge ordering

## [0.9.1] - 2026-02-20

### Changed
- Session-learner always spawns when transcript is present (removed `productiveCount` threshold gate)
- Improved session-learner prompt: framing changed to "capture what was learned"; explicit skip instruction for trivial sessions
- Added `--output-format text` to CLI invocation for human-readable session-learner logs

## [0.9.0] - 2026-02-20

### Added
- **Fire-and-forget session learning** — Stop hook spawns detached `claude` CLI subprocess, eliminating Conductor UI collision
- **Progressive t-SNE animation** — nodes drift from force layout to semantic positions as t-SNE converges
- `set_project` parameter on `update` tool for moving memories between project scopes

### Changed
- Uninstall script cleans up all installed components
- `.mcp.json` added to `.gitignore`

## [0.8.2] - 2026-02-16

### Added
- Node lifecycle animations: arrival glow, death fade, and snap-back physics on drag release

## [0.8.1] - 2026-02-16

### Changed
- Rebalanced force simulation physics for better node spacing
- Added `summary` parameter to `organize` tool for custom hub descriptions
- Added test step to release CI workflow

## [0.8.0] - 2026-02-16

### Added
- **Stop hook** (`on-stop`) — blocks session end when significant code changes detected, nudges session-learner to capture insights
- **Per-session state model** (`SessionState`) — replaces global HookState keys for tool call tracking and learning nudge throttling; cleaned up on session end
- `EmbeddingService.similarity()` method for direct text-to-text similarity comparison
- Visualizer: floating PiP panel with drag, resize, and corner-snapping
- Visualizer: `@globalActor`-based force simulation for off-MainActor O(n^2) computation
- Visualizer: synchronous local repulsion on drag for immediate visual feedback

### Changed
- Recall output label changed from "relevance:" to "distance:" for clarity
- Lattice dependency updated from local path to remote `0.3.1`
- Hooks: learning nudge skipped when stop hook already fired (prevents double-nudge)
- Hooks: `openLattice()` simplified — single function replaces `openLattice(at:)` and `openLatticeReadOnly()`
- Visualizer: ActivityLogPanel refactored to use `NodeData` instead of `Memory` directly
- install.sh: auto-approve memory MCP tools

### Fixed
- Graph traversal crash when depth=0 (now uses `stride` instead of range)
- `FlexibleIntArray` parsing of bracket-wrapped string arrays
- ForceSimulation `tickInFlight` starvation — `isActive` now also checks `maxSpeed`
- Throttled learning nudge counter reset properly scoped to session instead of global state

## [0.7.1] - 2026-02-15

### Added
- Visualizer: activity log panel — persistent scrollable list of recent memories with animated slide-in, cyan glow on new entries, and click-to-navigate

### Changed
- Visualizer: replace transient toast notifications with the persistent activity log
- Visualizer: raise normal label zoom threshold from 1.4x to 1.8x for cleaner overview
- README: use `swift run -c release` for visualizer launch commands

### Improved
- Visualizer: extract search bar into isolated view to prevent expensive recomputation on each keystroke
- Visualizer: move cluster computation to reactive state for snappier search typing
- Hooks: simplified hook handlers, removed unused Analyze/TranscriptParser modules

## [0.7.0] - 2026-02-15

### Added
- **`organize` tool** — batch re-topic memories with automatic hub creation and `part_of` edge linking
- **Cross-project hub linking** — `remember` auto-links memories to other projects' hubs when content mentions them by name (word-boundary, case-insensitive)
- **Hooks binary** (`memory-hooks`) — `advise` hook injects recalled memories as context before each message; `analyze` hook nudges session learning after substantive interactions
- **Custom agents** — `session-learner` and `memory-maintenance` agent definitions for background work
- Install script improvements: better update detection, hooks binary installation, agents directory

### Changed
- Episodes refactored to hub memories (topic: "episode") linked via `part_of` edges — no separate Episode model
- Label propagation uses synchronous updates for deterministic community detection across bridge edges

### Improved
- Visualizer: smaller node label fonts, material background on stats overlay, bold hub labels, golden-angle colors, trackpad gestures, better clustering layout

### Fixed
- Replace force-unwraps with safe optional handling across MCP tool handlers
- Improve MCP error reporting with descriptive messages instead of crashes
- Label propagation bridge-edge test now passes (synchronous updates prevent label cascading)

## [0.6.0] - 2026-02-15

### Added
- Visualizer: search, filtering, clustering visualization, and performance fixes
- Pre-compile CoreML embedding model at build time for fast MCP startup
- Lazy-load embeddings on first use instead of at initialization

## [0.5.2] - 2026-02-14

### Changed
- Update Lattice dependency to versioned 0.1.0

## [0.5.1] - 2026-02-14

### Fixed
- Install script now updates existing memory instructions instead of skipping them

## [0.5.0] - 2026-02-14

### Added
- Jaccard term-overlap similarity to conflict detection (reduces false positives from topically similar but distinct memories)
- Visualizer macOS app for exploring the knowledge graph
- Demo GIF in README

## [0.4.4] - 2026-02-14

### Added
- `parent_id` parameter on `remember` for hierarchical memories (auto-creates `part_of` edges)
- Atomic memory nudges in system instructions (one concept per memory guidance)
- Smarter graph display with hub/detail hierarchy

## [0.4.3] - 2026-02-14

### Fixed
- Add consolidation guardrails to prevent over-consolidation of distinct memories

## [0.4.2] - 2026-02-14

### Fixed
- Parse `ids` parameter correctly when MCP client sends comma-separated string instead of array (affects `merge` and `consolidate`)

## [0.4.1] - 2026-02-14

### Added
- `summarized_by` relation type for the `connect` tool
- README updated with all 20 tools, Tier 2 features, and hybrid search documentation

## [0.4.0] - 2026-02-14

### Added
- **Tier 2 features**: temporal queries (`timeline`), task continuity (`checkpoint`/`resume`/`list_tasks`), episodic memory (`begin_episode`/`end_episode`/`recall_episode`/`list_episodes`), and clustering (`find_clusters`/`consolidate`/`detect_communities`)

## [0.3.0] - 2026-02-14

### Added
- Knowledge graph with directed edges (`connect`, `disconnect`, `graph` tools)
- Relation types: `relates_to`, `contradicts`, `supersedes`, `derived_from`, `part_of`
- Fine-grained `update` tool with `append`, `prepend`, `find`+`replace`, and metadata-only modes
- Conflict detection on `remember` using embedding distance thresholds
- Reinforcement and importance scoring in recall ranking

## [0.1.0] - 2026-02-13

### Added
- Initial release — local MCP memory server for Claude Code
- Core memory tools: `remember`, `recall`, `forget`, `update`, `merge`, `list_topics`, `stats`
- Hybrid search combining semantic embeddings (CoreML) and full-text search (FTS5)
- SQLite-backed persistent storage with per-project scoping
- Pre-built universal binary releases for macOS
- One-liner install script (`scripts/install.sh`)
- CI/CD with GitHub Actions on macOS 26 / Swift 6.2
- Homebrew tap workflow

[0.12.1]: https://github.com/jsflax/Engram/compare/v0.12.0...v0.12.1
[0.12.0]: https://github.com/jsflax/Engram/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/jsflax/Engram/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/jsflax/Engram/compare/v0.9.1...v0.10.0
[0.9.1]: https://github.com/jsflax/Engram/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/jsflax/Engram/compare/v0.8.2...v0.9.0
[0.8.2]: https://github.com/jsflax/Engram/compare/v0.8.1...v0.8.2
[0.8.1]: https://github.com/jsflax/Engram/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/jsflax/Engram/compare/v0.7.3...v0.8.0
[0.7.1]: https://github.com/jsflax/Engram/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/jsflax/Engram/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/jsflax/Engram/compare/v0.5.2...v0.6.0
[0.5.2]: https://github.com/jsflax/Engram/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/jsflax/Engram/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/jsflax/Engram/compare/v0.4.4...v0.5.0
[0.4.4]: https://github.com/jsflax/Engram/compare/v0.4.3...v0.4.4
[0.4.3]: https://github.com/jsflax/Engram/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/jsflax/Engram/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/jsflax/Engram/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/jsflax/Engram/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/jsflax/Engram/compare/v0.1.0...v0.3.0
[0.1.0]: https://github.com/jsflax/Engram/releases/tag/v0.1.0
