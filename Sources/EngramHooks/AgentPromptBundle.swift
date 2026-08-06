import Foundation
#if canImport(EngramKit)
import EngramKit
#endif

/// The resource bundle carrying the packaged agent prompts
/// (session-learner.md etc.) — EngramKit's on the local backend, none on
/// Linux (sandboxes rely on the inline fallbacks).
var agentPromptBundle: Bundle? {
    #if canImport(EngramKit)
    return engramKitResourceBundle
    #else
    return nil
    #endif
}
