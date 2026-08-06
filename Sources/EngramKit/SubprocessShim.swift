import Foundation

/// EngramKit resource bundle handle for callers OUTSIDE the module (the
/// hooks pass it to loadAgentSystemPrompt for the bundled agent prompts —
/// Bundle.module is only visible inside the defining target).
public let engramKitResourceBundle: Bundle = Bundle.module
