import EngramMemoryCore
import Foundation

/// The advise hook's LAST line of defense (Aug 2026 incident, H1.6).
///
/// Budgets (`HookBudget.race`) stop WAITING on a stuck operation, but they
/// cannot stop the operation itself: a C++ frame parked in an
/// uninterruptible lock wait ignores cooperative cancellation, each
/// abandoned race leaves one more cooperative-pool thread blocked, and a
/// starved pool means the racing TIMER task never runs either — at which
/// point every in-process budget is dead and the hook rides, silent and
/// outputless, to the harness's 60s kill. This watchdog is the one
/// mechanism that holds in that world: a detached Foundation `Thread` (its
/// own OS thread — no Task machinery, no cooperative pool, nothing shared
/// with the wedged path but this class's lock), armed BEFORE the first
/// lattice/DB touch. At its deadline it writes whatever sections the main
/// path has accumulated so far plus a one-line degradation notice — in the
/// hook's normal output envelope, so the additionalContext contract stays
/// valid — flushes, and exits.
///
/// `exit(0)`, never `exit(1)`: a degraded-but-valid answer must not read
/// as a hook FAILURE to the harness.
final class HookWatchdog: @unchecked Sendable {
    private let eventName: String
    private let lock = NSLock()
    private var sections: [String] = []
    private var claimed = false

    init(eventName: String) {
        self.eventName = eventName
    }

    /// Thread-safe accumulator: the main path appends sections as they
    /// complete, so a watchdog fire ships everything finished by then.
    func append(_ section: String) {
        lock.lock()
        sections.append(section)
        lock.unlock()
    }

    /// Arm the watchdog. `Thread.sleep(until:)` on a dedicated thread is
    /// immune to every failure mode this class exists for; if the main
    /// path completes first, `completeNormally()` exits the process and
    /// takes this thread with it.
    func arm(firesAt deadline: Date, notice: String) {
        let thread = Thread { [self] in
            Thread.sleep(until: deadline)
            guard claim() else { return }
            hookLog("\(eventName) WATCHDOG fired — main path still wedged at its deadline; "
                + "emitting accumulated context and exiting 0")
            emitAndExit(extra: [notice])
        }
        thread.name = "engram-hook-watchdog"
        thread.stackSize = 512 * 1024
        thread.start()
    }

    /// Normal completion: take the one-shot claim, emit everything, exit.
    /// Losing the claim means the watchdog fired microseconds ago and is
    /// mid-write on its own thread — park; its `exit(0)` takes the process.
    func completeNormally() -> Never {
        if claim() {
            emitAndExit(extra: [])
        }
        while true { Thread.sleep(forTimeInterval: 60) }
    }

    /// One-shot gate between normal completion and the watchdog — exactly
    /// one of them writes the envelope.
    private func claim() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if claimed { return false }
        claimed = true
        return true
    }

    private func emitAndExit(extra: [String]) -> Never {
        lock.lock()
        let all = sections + extra
        lock.unlock()
        let output = HookOutput(
            hookSpecificOutput: HookSpecificOutput(
                hookEventName: eventName,
                additionalContext: all.joined(separator: "\n\n")
            )
        )
        try? writeOutput(output)
        // stdout is an unbuffered fd write (already flushed); hard exit —
        // see appendNudgesAndEmit for why a one-shot hook never pays
        // teardown costs.
        Foundation.exit(0)
    }
}
