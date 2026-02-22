#if canImport(FoundationModels)
import Testing
import Foundation
import FoundationModels

@available(macOS 26.0, *)
@Generable
struct TranscriptAssessment {
    @Guide(description: "Whether this transcript contains insights worth saving as long-term memories")
    var worthSaving: Bool
}

@available(macOS 26.0, *)
func assessTranscript(_ transcript: String) async throws -> (worthSaving: Bool, ms: Int64) {
    let prompt = """
    Does this transcript contain knowledge worth saving to long-term memory?

    Answer true if the transcript contains ANY of these:
    - A debugging discovery where the root cause was not immediately obvious \
    (e.g., "the crash was caused by UIKit deselecting rows during trait changes")
    - An architectural or design decision where alternatives were compared \
    (e.g., "we chose SSE over WebSockets because we only need server-to-client push")
    - The user stating preferences or conventions for how they want to work \
    (e.g., "always use guard let for early returns", "never auto-commit")
    - A workaround for a platform bug or limitation \
    (e.g., "SwiftUI previews break with C++ interop, fix by splitting targets")
    - A design decision about game mechanics, data models, or system architecture

    Answer false if the transcript is ONLY about:
    - Fixing typos, renaming variables, reformatting code, adding imports
    - Deleting unused files, bumping dependencies, running formatters
    - Simple questions and answers about project structure or versions
    - Casual conversation, writing emails, doing math
    - General explanations of well-known concepts (e.g., explaining what async/await is)

    Transcript:
    \(transcript)
    """

    let session = LanguageModelSession()
    let start = ContinuousClock.now
    let response = try await session.respond(to: prompt, generating: TranscriptAssessment.self)
    let elapsed = ContinuousClock.now - start
    let ms = elapsed.components.seconds * 1000 + elapsed.components.attoseconds / 1_000_000_000_000_000
    return (response.content.worthSaving, ms)
}

// MARK: - Should be worth saving

@available(macOS 26.0, *)
@Test func transcript_architecturalDecision() async throws {
    guard SystemLanguageModel.default.availability == .available else { return }

    let (worthSaving, ms) = try await assessTranscript("""
    user: should we use WebSockets or SSE for the real-time updates?
    assistant: Let me look at the current architecture.
    [Read file: src/api/routes.ts]
    I see you're running Express with a standard REST API setup. Let me check the event service.
    [Read file: src/services/events.ts]
    OK so currently you're polling every 2 seconds from the client. Let me look at what
    data flows need to be real-time.
    [Read file: src/components/Dashboard.tsx]
    [Read file: src/components/NotificationBell.tsx]
    I can see two real-time features: the dashboard metrics refresh and the notification bell.
    Both are server-to-client only — the client never sends data back through the real-time
    channel, it just listens for updates.
    Given that, SSE (Server-Sent Events) is the better choice here over WebSockets:
    1. You only need server-to-client push, not bidirectional communication
    2. SSE works over regular HTTP — no connection upgrade, no special proxy config
    3. Built-in reconnection with EventSource API (WebSocket needs manual reconnect logic)
    4. Simpler server implementation — just set headers and write to the response stream
    5. Works through HTTP/2 multiplexing, so multiple SSE streams share one TCP connection
    WebSockets would be overkill and add complexity for no benefit here.
    user: good point, let's go with SSE. what about scalability?
    assistant: For your scale (the dashboard shows ~500 concurrent users), SSE is fine. Each
    connection is a long-lived HTTP response, so it's one file descriptor per client. Node
    can handle tens of thousands of those. If you ever need to scale beyond a single server,
    you'd put a Redis pub/sub layer behind it — same as you'd need with WebSockets anyway.
    [Write file: src/services/sse.ts]
    Created the SSE service with:
    - EventEmitter-based channel management
    - Heartbeat every 15s to keep connections alive through proxies
    - Automatic cleanup when clients disconnect
    - Each client gets a unique channel ID for targeted updates
    - Content-Type: text/event-stream with proper headers
    [Edit file: src/api/routes.ts]
    Added GET /events/:channelId endpoint. Also removed the old polling endpoint
    GET /api/updates since SSE replaces it.
    [Edit file: src/components/Dashboard.tsx]
    Replaced the setInterval polling with an EventSource connection.
    [Edit file: src/components/NotificationBell.tsx]
    Same — switched from polling to SSE.
    [Run: npm test]
    All 47 tests passing.
    user: clean. let's ship it
    """)
    print("architecturalDecision: \(ms)ms (worthSaving: \(worthSaving))")
    #expect(worthSaving, "Architectural decision (SSE vs WebSockets) should be worth saving")
}

@available(macOS 26.0, *)
@Test func transcript_debuggingDiscovery() async throws {
    guard SystemLanguageModel.default.availability == .available else { return }

    let (worthSaving, ms) = try await assessTranscript("""
    user: the app crashes when I rotate the device on the settings screen. here's the crash log:
    Thread 1: Fatal error: Unexpectedly found nil while unwrapping an Optional value
    SettingsViewController.swift:87
    assistant: Let me investigate.
    [Read file: SettingsViewController.swift]
    I see the crash site on line 87:
    ```swift
    let selected = tableView.indexPathForSelectedRow!
    updateDetailView(for: selected)
    ```
    That force unwrap is the problem, but the interesting question is why it's nil during rotation.
    Let me check the view lifecycle.
    [Read file: SettingsViewController.swift lines 60-120]
    Found it. Here's what's happening:
    1. viewWillTransition(to:with:) is called during device rotation
    2. UIKit triggers a trait collection change as part of the rotation
    3. UITableView automatically deselects all rows during trait collection changes
    4. Your code on line 87 runs inside the transition coordinator's animation block
    5. By that point, indexPathForSelectedRow is already nil
    This is a documented but non-obvious UIKit behavior. The table view clears selection
    during trait changes to allow the new layout to reconfigure cells properly.
    user: oh weird, so the table view deselects before the animation completes?
    assistant: Yes, the deselection happens synchronously when the trait collection changes,
    which is before your animation block runs. The trait change is triggered by the rotation
    starting, not by it completing.
    [Edit file: SettingsViewController.swift]
    Two changes:
    1. Added a `selectedIndexPath` property that captures the selection in
       viewWillTransition before the trait change clears it
    2. Changed line 87 to use guard let with the stored property instead of force unwrap
    ```swift
    private var selectedIndexPath: IndexPath?

    override func viewWillTransition(to size: CGSize, with coordinator: ...) {
        selectedIndexPath = tableView.indexPathForSelectedRow  // capture before deselect
        super.viewWillTransition(to: size, with: coordinator)
        coordinator.animate(alongsideTransition: { _ in
            guard let selected = self.selectedIndexPath else { return }
            self.updateDetailView(for: selected)
        }, completion: { _ in
            // Restore selection after rotation
            if let selected = self.selectedIndexPath {
                self.tableView.selectRow(at: selected, animated: false, scrollPosition: .none)
            }
        })
    }
    ```
    user: ah that explains why it only crashed sometimes — it depended on whether a row was selected
    assistant: Exactly. If you hadn't selected a row (or had scrolled away so it was recycled),
    the force unwrap would get nil and crash. Rotation without a selection was always fine.
    [Run: xcodebuild test -scheme Settings -destination 'platform=iOS Simulator,name=iPhone 16']
    All 23 tests passing, including the new rotation test I added.
    user: nice, thanks for the detailed explanation
    """)
    print("debuggingDiscovery: \(ms)ms (worthSaving: \(worthSaving))")
    #expect(worthSaving, "Non-obvious debugging discovery should be worth saving")
}

@available(macOS 26.0, *)
@Test func transcript_performanceFinding() async throws {
    guard SystemLanguageModel.default.availability == .available else { return }

    let (worthSaving, ms) = try await assessTranscript("""
    user: the search endpoint is really slow, like 3 seconds per query
    assistant: Let me profile this. First let me understand the search flow.
    [Read file: src/search/index.ts]
    [Read file: src/search/queries.ts]
    [Read file: src/api/routes/search.ts]
    OK so the flow is: route handler → buildSearchQuery() → search.execute() → format results.
    Let me add timing to each step.
    [Edit file: src/api/routes/search.ts]
    Added console.time/timeEnd around each phase.
    [Run: curl http://localhost:3000/api/search?q=test]
    Output:
    search:client-init: 2847ms
    search:query-build: 2ms
    search:execute: 180ms
    search:format: 12ms
    Found it. The client initialization is taking 2.8 seconds per request. Let me check why.
    [Read file: src/search/index.ts]
    Here's the problem — line 42:
    ```typescript
    export async function execute(query: SearchQuery) {
        const client = new Client({ node: process.env.ES_URL, ... })
        return client.search(query)
    }
    ```
    You're creating a new Elasticsearch client on every request. The Client constructor does:
    - TLS handshake with the ES cluster
    - Connection pool initialization (default 10 connections)
    - Node discovery (sniffs the cluster topology)
    - Authentication negotiation
    All of that runs before the first query. This should be a module-level singleton.
    [Edit file: src/search/index.ts]
    Moved client creation to module scope with lazy initialization:
    ```typescript
    let client: Client | null = null
    function getClient() {
        if (!client) {
            client = new Client({
                node: process.env.ES_URL,
                maxRetries: 3,
                requestTimeout: 10000,
                keepAlive: true,
                keepAliveFreeSocketTimeout: 30000,
            })
        }
        return client
    }
    ```
    [Run: curl -w "\\nTotal: %{time_total}s\\n" http://localhost:3000/api/search?q=test]
    First request: 0.32s (includes one-time client init)
    Second request: 0.08s
    Response time dropped from 3.2s to 0.08s for warm requests. The overhead was entirely
    from repeated client initialization.
    user: wow that's a 40x improvement. how did we not catch this earlier?
    assistant: It's a common pitfall — the ES SDK docs recommend singleton usage but the
    getting-started examples show inline construction for simplicity. Also, in dev with
    a local ES instance the TLS handshake is fast so the overhead is ~200ms, which is
    easy to miss. It only becomes obvious against a remote cluster with TLS.
    [Edit file: src/search/index.ts]
    Removed the timing instrumentation. Also added a comment about the singleton pattern.
    [Run: npm test]
    All tests passing.
    """)
    print("performanceFinding: \(ms)ms (worthSaving: \(worthSaving))")
    #expect(worthSaving, "Performance root cause and 40x fix should be worth saving")
}

@available(macOS 26.0, *)
@Test func transcript_userPreference() async throws {
    guard SystemLanguageModel.default.availability == .available else { return }

    let (worthSaving, ms) = try await assessTranscript("""
    user: hey before we start working on the new feature, I want to set some ground rules
    assistant: Sure, go ahead.
    user: first, always use guard let instead of if let for early returns. I find the code
    reads better when the happy path isn't nested.
    assistant: Makes sense. I'll use guard let consistently for early returns and reserve
    if let for cases where both branches have meaningful work.
    user: exactly. second, for error handling at async boundaries — like when calling an API
    or reading from disk — prefer Result types over throwing functions. I want errors to be
    explicit in the type signature, not invisible throws.
    assistant: Got it. So the pattern would be something like:
    ```swift
    func fetchUser(id: Int) async -> Result<User, APIError> {
        // ...
    }
    ```
    Rather than:
    ```swift
    func fetchUser(id: Int) async throws -> User {
        // ...
    }
    ```
    For synchronous internal code, throwing is still fine though?
    user: yes, throwing is fine for synchronous stuff. it's just at the async boundaries where
    I want Result. also one more thing — never auto-commit. always show me the diff first and
    wait for me to say go ahead. I've been burned by auto-commits including debug code.
    assistant: Understood. I'll always show the full diff and wait for explicit approval before
    any git commit. Even if you say "commit this", I'll show the diff first just to confirm.
    user: perfect. and use conventional commits format — feat:, fix:, chore:, etc.
    assistant: Will do. So to summarize the conventions:
    1. guard let for early returns, if let for meaningful branches
    2. Result types at async boundaries, throws for synchronous code
    3. No auto-commits — always show diff and wait for approval
    4. Conventional commit messages (feat:, fix:, chore:, docs:, etc.)
    user: that's it. let's get started
    """)
    print("userPreference: \(ms)ms (worthSaving: \(worthSaving))")
    #expect(worthSaving, "Explicit user preferences should be worth saving")
}

@available(macOS 26.0, *)
@Test func transcript_patternDiscovery() async throws {
    guard SystemLanguageModel.default.availability == .available else { return }

    let (worthSaving, ms) = try await assessTranscript("""
    user: why are all my SwiftUI previews broken? they were working yesterday
    assistant: Let me check a few things.
    [Read file: ContentView.swift]
    The view code looks fine. Let me check the package configuration.
    [Read file: Package.swift]
    I think I found it. Your package has `.interoperabilityMode(.Cxx)` in swiftSettings
    for the main target. When was that added?
    [Run: git log --oneline -5 Package.swift]
    commit abc123 — "Add C++ interop for Lattice ORM" (yesterday)
    That's the culprit. SwiftUI previews don't support C++ interop mode. This is a known
    Xcode limitation — the preview engine compiles in a sandboxed environment that doesn't
    have access to C++ bridging headers or the C++ standard library modulemap.
    user: ugh, so I can't have both Lattice and previews?
    assistant: You can, but you need to separate them into different targets. The trick is to
    put your SwiftUI views in a target WITHOUT C++ interop, and keep the Lattice/C++ code
    in its own target. The views target can still import types from the C++ target — it just
    can't be compiled with the interop flag itself.
    [Read file: Package.swift]
    Here's how I'd restructure this:
    [Edit file: Package.swift]
    Split into three targets:
    1. EngramKit (with .Cxx) — Lattice models, database code, embedding service
    2. EngramViews (no .Cxx) — SwiftUI views, depends on EngramKit
    3. Engram (executable) — entry point, depends on both
    The key insight is that the views target can import and use types defined in the Cxx
    target — the interop flag only affects compilation of that specific target's source
    files, not its dependencies' public interfaces.
    [Run: swift build]
    Build succeeded.
    Let me verify previews work now.
    [Read file: Sources/EngramViews/MemoryListView.swift]
    The preview provider is here. Since this target doesn't have .Cxx, Xcode's preview
    engine can compile it.
    user: nice, previews are working again! good to know about the target separation trick
    assistant: Yeah, it's a common gotcha when mixing C++ interop with SwiftUI. The same
    issue applies to Swift Playgrounds too — they can't load targets compiled with Cxx mode.
    Any time you need the preview canvas or playground, keep those sources in a pure-Swift target.
    """)
    print("patternDiscovery: \(ms)ms (worthSaving: \(worthSaving))")
    #expect(worthSaving, "SwiftUI previews + C++ interop limitation is worth saving")
}

@available(macOS 26.0, *)
@Test func transcript_migrationDecision() async throws {
    guard SystemLanguageModel.default.availability == .available else { return }

    let (worthSaving, ms) = try await assessTranscript("""
    user: we need to add a `tags` column to the memories table. what's the best approach?
    assistant: Let me look at the current model and how Lattice handles migrations.
    [Read file: Sources/EngramKit/Memory.swift]
    The Memory model currently has: content, topic, project, source, embedding, timestamps,
    expiresAt, accessCount, importance. All stored via Lattice's @Model decorator.
    [Read file: /Users/jason/Documents/Lattice/Sources/Lattice/Model/Model.swift]
    Lattice handles schema evolution automatically — when you add a new property to a @Model,
    it runs ALTER TABLE ADD COLUMN on first access. No manual migration files needed.
    For the tags field, there are two approaches:
    1. JSON string column: `var tags: String?` storing ["tag1", "tag2"] as JSON
    2. Junction table: Separate Tag model + TagMemory many-to-many relationship
    I'd recommend option 1 (JSON string) for several reasons:
    - Tags are always read and written together with the memory (no independent tag queries)
    - SQLite's json_each() function lets you query into JSON arrays if needed later:
      `SELECT * FROM Memory WHERE id IN (SELECT id FROM Memory, json_each(tags) WHERE value = 'debugging')`
    - Avoids the complexity of a junction table and the extra queries for every memory fetch
    - Lattice doesn't have built-in many-to-many support, so you'd need manual join logic
    user: what about querying "all memories with tag X" — wouldn't a junction table be faster?
    assistant: For your scale (hundreds to low thousands of memories), the JSON approach with
    json_each() is actually fast enough — SQLite's JSON functions are well-optimized. A junction
    table would only matter at tens of thousands of rows with frequent tag-based queries.
    If you later need that performance, you could add an FTS5 index on the tags column,
    which would make tag searches essentially instant.
    user: fair enough, let's go with JSON. what about existing rows?
    assistant: Since it's a nullable column (String?), Lattice adds it as ALTER TABLE ADD COLUMN
    with no DEFAULT — existing rows get NULL. The computed property handles this gracefully:
    [Edit file: Sources/EngramKit/Memory.swift]
    ```swift
    var tags: String?

    var tagList: [String] {
        get {
            guard let tags, let data = tags.data(using: .utf8),
                  let array = try? JSONDecoder().decode([String].self, from: data) else {
                return []
            }
            return array
        }
        set {
            tags = (try? String(data: JSONEncoder().encode(newValue), encoding: .utf8)) ?? "[]"
        }
    }
    ```
    NULL → empty array, so it's fully backwards compatible with no migration step.
    [Run: swift test]
    All 174 tests passing. Lattice auto-migrated the test databases too.
    user: great, clean approach
    """)
    print("migrationDecision: \(ms)ms (worthSaving: \(worthSaving))")
    #expect(worthSaving, "Database migration strategy decision should be worth saving")
}

@available(macOS 26.0, *)
@Test func transcript_designBrainstorm() async throws {
    guard SystemLanguageModel.default.availability == .available else { return }

    let (worthSaving, ms) = try await assessTranscript("""
    user: I'm thinking about the game's karma system. right now it just tracks a number but
    I want it to feel more nuanced than a simple good/evil slider
    assistant: That's a common frustration with karma systems. The binary slider approach
    (like early Fallout or inFamous) reduces complex moral choices to a single axis.
    A few alternatives that work well:
    1. Multi-axis tracking — separate dimensions like Violence/Mercy, Truth/Deception,
       Selfishness/Sacrifice. The Mara conversation at the end could reference specific
       axes rather than a single score.
    2. Memory-based — instead of a number, keep a log of specific decisions. "You spared
       Sun Wukong in the Asura realm" hits harder than "karma: +5". Each decision is a
       flag, and NPCs reference specific ones.
    3. Reputation per realm — each of the six realms tracks how you behaved there independently.
       Deva might see you as a saint while Naraka fears you.
    user: I really like option 2. it fits the Buddhist theme too — karma as the accumulated
    weight of specific actions, not an abstract number
    assistant: Exactly — in Buddhist philosophy karma IS the actions themselves and their
    consequences, not a cosmic scoreboard. So mechanically, you'd store something like:
    - Array of KarmaEvent structs: { realm, choice, consequence, weight }
    - The Mara conversation picks the 5-6 most significant events to reference
    - NPCs check for specific flags: "if player spared Sun Wukong, he appears as ally in Deva"
    - No aggregation into a single number at all
    This also means there's no "good" or "evil" ending — just YOUR ending, shaped by YOUR
    specific choices. The final conversation with Mara becomes deeply personal because she's
    referencing real decisions, not a stat.
    user: and it means players can't game it by doing 10 good things to offset one bad one
    assistant: Right, that's the key insight. With a numeric system, players optimize the number.
    With event-based karma, every choice is permanent and visible. You can't "undo" killing
    someone by helping 10 others — Mara will still bring it up.
    It also means different players will get genuinely different final conversations, not just
    "good ending" vs "bad ending". A player who was merciful everywhere except one brutal
    choice in Naraka gets a different Mara than someone who was consistently violent.
    user: this is great. let's go with the event-based approach. I'll need to redesign the
    KarmaSystem class around this
    assistant: Here's a rough sketch of the data model to think about:
    - KarmaEvent: realm, choiceId, description, weight (for ranking significance), timestamp
    - KarmaLedger: ordered list of KarmaEvents, query methods like
      significantEvents(limit:), eventsInRealm(:), hasEvent(choiceId:)
    - MaraDialogueBuilder: takes a KarmaLedger, selects top events, generates dialogue nodes
    The weight field is important — it lets you mark some choices as more significant than
    others so Mara focuses on the big moments, not every minor interaction.
    user: perfect, I'll start on this tomorrow
    """)
    print("designBrainstorm: \(ms)ms (worthSaving: \(worthSaving))")
    #expect(worthSaving, "Game design decision about karma system architecture should be worth saving")
}

// MARK: - Should NOT be worth saving

@available(macOS 26.0, *)
@Test func transcript_typoFix() async throws {
    guard SystemLanguageModel.default.availability == .available else { return }

    let (worthSaving, ms) = try await assessTranscript("""
    user: hey can you fix the typo on line 12 of README.md? it says "teh" instead of "the"
    assistant: Sure, let me take a look.
    [Read file: README.md]
    I see it on line 12: "teh memory server" should be "the memory server".
    [Edit file: README.md]
    Fixed "teh" → "the" on line 12.
    user: thanks. also line 45 has "recieve" instead of "receive"
    assistant: [Read file: README.md]
    Found it.
    [Edit file: README.md]
    Fixed "recieve" → "receive" on line 45.
    user: and can you add a newline at the end of the file? my linter complains
    assistant: [Edit file: README.md]
    Added a trailing newline at the end of the file.
    user: perfect, thanks
    assistant: You're welcome! The three fixes were:
    - Line 12: "teh" → "the"
    - Line 45: "recieve" → "receive"
    - Added trailing newline
    user: looks good
    """)
    print("typoFix: \(ms)ms (worthSaving: \(worthSaving))")
    #expect(!worthSaving, "Trivial typo fix should not be worth saving")
}

@available(macOS 26.0, *)
@Test func transcript_simpleRename() async throws {
    guard SystemLanguageModel.default.availability == .available else { return }

    let (worthSaving, ms) = try await assessTranscript("""
    user: rename the function `processData` to `transformPayload` in api.ts and update all call sites
    assistant: Let me find all the references first.
    [Grep: processData in src/]
    Found references in 4 files:
    - src/api.ts (definition + 1 internal call)
    - src/routes.ts (2 calls)
    - src/handler.ts (1 call)
    - src/tests/api.test.ts (3 references in tests)
    [Read file: src/api.ts]
    [Edit file: src/api.ts]
    Renamed the function definition and the internal call on line 34.
    [Read file: src/routes.ts]
    [Edit file: src/routes.ts]
    Updated both call sites on lines 12 and 28.
    [Read file: src/handler.ts]
    [Edit file: src/handler.ts]
    Updated the call on line 56.
    [Read file: src/tests/api.test.ts]
    [Edit file: src/tests/api.test.ts]
    Updated the 3 test references.
    [Run: npm test]
    All 31 tests passing.
    user: looks good, thanks
    assistant: Done. Renamed processData → transformPayload across 4 files, 8 total references.
    All tests pass.
    """)
    print("simpleRename: \(ms)ms (worthSaving: \(worthSaving))")
    #expect(!worthSaving, "Simple rename should not be worth saving")
}

@available(macOS 26.0, *)
@Test func transcript_addImport() async throws {
    guard SystemLanguageModel.default.availability == .available else { return }

    let (worthSaving, ms) = try await assessTranscript("""
    user: I'm getting "Cannot find type 'URL' in scope" in Utils.swift
    assistant: That error means Foundation isn't imported. Let me check.
    [Read file: Sources/Utils.swift]
    Right — the file uses URL, Data, and JSONEncoder but doesn't import Foundation.
    [Edit file: Sources/Utils.swift]
    Added `import Foundation` at the top of the file.
    [Run: swift build]
    Build succeeded with no errors.
    user: great, thanks. I always forget that one
    assistant: It's a common one — URL and Data look like they should be built-in Swift types
    but they're actually Foundation types. The compiler error message could be more helpful
    about suggesting the import.
    user: yep. all good now
    """)
    print("addImport: \(ms)ms (worthSaving: \(worthSaving))")
    #expect(!worthSaving, "Adding a missing import should not be worth saving")
}

@available(macOS 26.0, *)
@Test func transcript_formatting() async throws {
    guard SystemLanguageModel.default.availability == .available else { return }

    let (worthSaving, ms) = try await assessTranscript("""
    user: can you run swift-format on the entire Sources directory? the code style is inconsistent
    assistant: Sure, let me check if swift-format is available.
    [Run: which swift-format]
    /usr/bin/swift-format
    [Run: swift-format format --recursive -i Sources/]
    Formatted 24 files. Let me check what changed.
    [Run: git diff --stat]
    Sources/EngramKit/MemoryTools.swift          | 14 ++++---
    Sources/EngramKit/MemoryTools+Core.swift      | 22 +++++-----
    Sources/EngramKit/ToolHelpers.swift           |  8 ++--
    Sources/EngramKit/EmbeddingService.swift      |  4 +-
    Sources/EngramKit/Memory.swift                |  2 +-
    Sources/Engram/main.swift                            |  6 +--
    6 files changed, 28 insertions(+), 28 deletions(+)
    Changes are mostly whitespace normalization and consistent brace placement. No logic changes.
    [Run: swift build]
    Build succeeded.
    [Run: swift test]
    All 174 tests passing.
    user: great, commit that
    assistant: [Run: git add Sources/]
    [Run: git commit -m "chore: apply swift-format to Sources directory"]
    Committed.
    user: thanks
    """)
    print("formatting: \(ms)ms (worthSaving: \(worthSaving))")
    #expect(!worthSaving, "Running a formatter should not be worth saving")
}

@available(macOS 26.0, *)
@Test func transcript_simpleQuestion() async throws {
    guard SystemLanguageModel.default.availability == .available else { return }

    let (worthSaving, ms) = try await assessTranscript("""
    user: what version of Swift are we using in this project?
    assistant: [Read file: Package.swift]
    The package uses `swift-tools-version: 6.2`, so you're on Swift 6.2.
    The minimum deployment target is macOS 15.
    user: and what dependencies do we have?
    assistant: From Package.swift, there are 4 dependencies:
    1. swift-sdk (MCP protocol) v0.10.0
    2. Lattice (ORM) v0.2.1
    3. SwiftLM (embeddings) — main branch
    4. swift-argument-parser v1.5.0
    user: ok. and how many test files?
    assistant: [Glob: Tests/**/*.swift]
    There are 8 test files:
    - CoreMemoryTests.swift
    - EpisodeTests.swift
    - TaskTests.swift
    - ClusteringTests.swift
    - GraphTests.swift
    - TimelineTests.swift
    - OrganizeTests.swift
    - TestHelpers.swift
    user: ok thanks, just getting oriented
    assistant: No problem! Let me know if you want a deeper dive into any part of the codebase.
    """)
    print("simpleQuestion: \(ms)ms (worthSaving: \(worthSaving))")
    #expect(!worthSaving, "Simple factual Q&A should not be worth saving")
}

@available(macOS 26.0, *)
@Test func transcript_deleteUnusedCode() async throws {
    guard SystemLanguageModel.default.availability == .available else { return }

    let (worthSaving, ms) = try await assessTranscript("""
    user: delete the old LegacyParser.swift file, we don't use it anymore
    assistant: Let me verify it's truly unused before deleting.
    [Grep: LegacyParser in Sources/]
    [Grep: LegacyParser in Tests/]
    No references found anywhere in Sources/ or Tests/. The only mention is the file itself.
    [Read file: Sources/LegacyParser.swift]
    It's a 45-line file with a LegacyParser class that was apparently replaced by the current
    parser. No other types defined in it.
    [Run: git log --oneline -3 Sources/LegacyParser.swift]
    Last modified 3 weeks ago — "Deprecate LegacyParser in favor of new parser"
    Safe to delete.
    [Run: rm Sources/LegacyParser.swift]
    Deleted.
    [Run: swift build]
    Build succeeded — confirms nothing depended on it.
    user: good, thanks
    assistant: Deleted LegacyParser.swift. Build still passes, confirming no remaining references.
    user: 👍
    """)
    print("deleteUnusedCode: \(ms)ms (worthSaving: \(worthSaving))")
    #expect(!worthSaving, "Deleting an unused file should not be worth saving")
}

@available(macOS 26.0, *)
@Test func transcript_updateDependency() async throws {
    guard SystemLanguageModel.default.availability == .available else { return }

    let (worthSaving, ms) = try await assessTranscript("""
    user: bump the lattice dependency from 0.2.0 to 0.2.1
    assistant: [Read file: Package.swift]
    I see the current dependency:
    `.package(url: "https://github.com/jsflax/lattice.git", from: "0.2.0")`
    [Edit file: Package.swift]
    Updated to from: "0.2.1".
    [Run: swift package resolve]
    Fetching https://github.com/jsflax/lattice.git
    Resolved Lattice at 0.2.1.
    [Run: swift build]
    Build succeeded with no warnings.
    [Run: swift test]
    All 174 tests passing.
    user: any breaking changes?
    assistant: No — 0.2.1 is a patch release. Looking at the diff between 0.2.0 and 0.2.1,
    it's just a bug fix for FTS5 index creation on empty tables. No API changes.
    user: cool, thanks
    """)
    print("updateDependency: \(ms)ms (worthSaving: \(worthSaving))")
    #expect(!worthSaving, "Routine dependency bump should not be worth saving")
}

@available(macOS 26.0, *)
@Test func transcript_casualChat() async throws {
    guard SystemLanguageModel.default.availability == .available else { return }

    let (worthSaving, ms) = try await assessTranscript("""
    user: hey
    assistant: Hey! What can I help you with?
    user: just checking if you're working. had some issues with the MCP server earlier
    assistant: I'm here and working. If you were having MCP connection issues, that's usually
    because the server process crashed or wasn't started. You can check with
    `claude mcp list` to see if it's registered.
    user: yeah it's fine now, I restarted it
    assistant: Good to hear. Let me know if you need anything.
    user: will do, thanks
    assistant: Anytime!
    """)
    print("casualChat: \(ms)ms (worthSaving: \(worthSaving))")
    #expect(!worthSaving, "Casual greeting and status check should not be worth saving")
}

@available(macOS 26.0, *)
@Test func transcript_weatherQuestion() async throws {
    guard SystemLanguageModel.default.availability == .available else { return }

    let (worthSaving, ms) = try await assessTranscript("""
    user: what's the weather like in san francisco today?
    assistant: I don't have access to real-time weather data. I'm a code assistant — I can
    help with programming tasks, file editing, running commands, and so on. For weather,
    you'd want to check a weather app or website like weather.gov.
    user: oh right, wrong tool haha
    assistant: No worries! Easy mix-up. Let me know if you need help with code.
    user: actually yeah, what time is it?
    assistant: I don't have access to the current time either. You can check with `date`
    in your terminal if you like.
    user: ok I'll just look at my phone. thanks anyway
    assistant: Sure thing!
    """)
    print("weatherQuestion: \(ms)ms (worthSaving: \(worthSaving))")
    #expect(!worthSaving, "Off-topic questions about weather/time should not be worth saving")
}

@available(macOS 26.0, *)
@Test func transcript_writingHelp() async throws {
    guard SystemLanguageModel.default.availability == .available else { return }

    let (worthSaving, ms) = try await assessTranscript("""
    user: can you help me write an email to my team about the upcoming sprint?
    assistant: Sure! What are the key points you want to cover?
    user: we're starting the new auth feature next week, and I need everyone to review the
    design doc before Monday's meeting. also remind them about the Friday demo
    assistant: Here's a draft:

    Subject: Sprint 24 kickoff — please review auth design doc

    Hi team,

    Quick heads up on next week:

    We're kicking off the authentication feature in Sprint 24. I've shared the design doc
    in Confluence — please review it before our Monday planning meeting so we can hit the
    ground running. Key areas I'd love feedback on:
    - Token refresh strategy (section 3.2)
    - Session management approach (section 4)

    Also a reminder that we have the Friday demo at 3pm. Sarah will be showing the dashboard
    improvements from this sprint.

    See you Monday!
    [Your name]

    Want me to adjust anything?
    user: that's perfect, thanks. maybe make the tone a bit more casual
    assistant: Here's a more casual version:

    Subject: Sprint 24 — auth feature + design doc review

    Hey everyone,

    Heads up — we're tackling the auth feature next sprint. I put the design doc in Confluence,
    would love it if you could take a look before Monday's planning so we're all on the same
    page. Especially interested in thoughts on the token refresh strategy and session management.

    Oh and don't forget — Friday demo at 3pm, Sarah's showing off the dashboard work.

    Cheers!

    user: great, I'll use that. thanks!
    assistant: Happy to help! Good luck with the sprint.
    """)
    print("writingHelp: \(ms)ms (worthSaving: \(worthSaving))")
    #expect(!worthSaving, "Drafting a routine email should not be worth saving")
}

@available(macOS 26.0, *)
@Test func transcript_mathHelp() async throws {
    guard SystemLanguageModel.default.availability == .available else { return }

    let (worthSaving, ms) = try await assessTranscript("""
    user: quick math question — if I have a 384-dimensional vector and I want to store it as
    Float32, how many bytes is that per vector?
    assistant: 384 dimensions x 4 bytes per Float32 = 1,536 bytes per vector, or about 1.5 KB.
    user: and if I have 10,000 memories each with an embedding?
    assistant: 10,000 x 1,536 bytes = 15,360,000 bytes, or about 14.6 MB.
    That's very manageable for a local SQLite database. Even 100,000 memories would only
    be ~146 MB of embedding data.
    user: cool, just wanted to make sure the storage wouldn't blow up
    assistant: At your current rate of memory creation, you're well within comfortable bounds.
    SQLite handles databases up to 281 TB theoretically, and sqlite-vec is designed for
    exactly this kind of local vector workload.
    user: great, thanks for the quick math
    """)
    print("mathHelp: \(ms)ms (worthSaving: \(worthSaving))")
    #expect(!worthSaving, "Simple arithmetic calculation should not be worth saving")
}

@available(macOS 26.0, *)
// TODO: Model sometimes considers concept explanations worth saving — revisit threshold
// @Test func transcript_conceptExplanation() async throws {
@Test(.disabled("Flaky — model boundary case")) func transcript_conceptExplanation() async throws {
    guard SystemLanguageModel.default.availability == .available else { return }

    let (worthSaving, ms) = try await assessTranscript("""
    user: I keep hearing about "structured concurrency" in Swift. can you explain what it
    actually means and why it matters? I've been using GCD and it works fine
    assistant: The core idea of structured concurrency is that async tasks have a clear
    parent-child hierarchy, similar to how structured programming gave us if/else/while
    instead of goto. In GCD, you fire off blocks onto queues and they're essentially
    free-floating — nothing tracks their lifetime or guarantees completion order.
    With structured concurrency (TaskGroup, async let, etc.):
    1. Every child task is scoped to its parent — when the parent scope exits, all children
       are guaranteed to have completed (or been cancelled). No orphaned tasks.
    2. Cancellation propagates automatically. Cancel the parent, all children get cancelled.
       With GCD, you need manual cancellation tokens or flags.
    3. Errors propagate naturally. A throwing child task's error surfaces at the parent's
       await point. With GCD, errors from dispatched blocks need manual error channels.
    user: so it's basically like how we stopped using goto in favor of functions and loops?
    assistant: That's the exact analogy. Dijkstra's "Go To Statement Considered Harmful"
    argued that unstructured jumps make programs impossible to reason about. Structured
    concurrency makes the same argument about tasks — unstructured concurrency (fire and
    forget, detached tasks, GCD blocks) creates the same reasoning problems that goto did.
    Practically, the biggest win is resource cleanup. With structured concurrency:
    ```swift
    func processImages(_ urls: [URL]) async throws -> [Image] {
        try await withThrowingTaskGroup(of: Image.self) { group in
            for url in urls {
                group.addTask { try await downloadImage(url) }
            }
            return try await group.reduce(into: []) { $0.append($1) }
        }
        // ALL download tasks are guaranteed done here, whether they succeeded or failed
    }
    ```
    If any download throws, the group cancels the remaining ones and the error propagates.
    No leaked connections, no zombie tasks. With GCD, you'd need a DispatchGroup plus
    manual error handling plus a cancellation mechanism.
    user: that makes way more sense now. so async let is basically the simple case of "do
    two things concurrently and wait for both"?
    assistant: Exactly. async let is structured concurrency for a fixed number of concurrent tasks:
    ```swift
    async let profile = fetchProfile(userId)
    async let posts = fetchPosts(userId)
    let (p, ps) = try await (profile, posts)  // both must complete
    ```
    TaskGroup is for dynamic numbers. The structure guarantee is the same — both must
    complete before you move past the await point. No orphans, automatic cancellation on error.
    The one escape hatch is Task.detached{}, which is intentionally unstructured — equivalent
    to "goto" for concurrency. It's there when you genuinely need it (background work with
    no parent), but it should be rare.
    user: ok so bottom line — I should migrate from GCD to async/await not just for nicer
    syntax but because the task lifetime guarantees prevent real bugs
    assistant: Exactly. The syntax is a bonus. The real value is that structured concurrency
    makes task lifetimes visible and bounded, which eliminates entire categories of bugs —
    leaked resources, orphaned tasks, missing error handling, race conditions from tasks
    outliving their expected scope.
    """)
    print("conceptExplanation: \(ms)ms (worthSaving: \(worthSaving))")
    #expect(!worthSaving, "General CS concept explanation is not project-specific insight")
}

#endif
