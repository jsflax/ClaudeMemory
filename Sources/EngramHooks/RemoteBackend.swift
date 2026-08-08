import EngramMemoryCore
import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

// The REMOTE hook backend: the same seven hooks, but memory lives in
// engram-server (agents mode) instead of a local lattice. Selected by env —
// a sandbox boots with ENGRAM_URL/ENGRAM_TOKEN injected by the
// orchestrator; a dev Mac without them keeps the local backend untouched.
//
// Every call is fail-open: hooks must never break a prompt. Timeouts are
// short (advise sits on the prompt path); errors log and return nil.

struct RemoteConfig {
    let baseURL: URL
    let token: String
    let project: String?

    /// Remote iff BOTH credentials are present (matches the .mcp.json
    /// injection gate — entry, env, and hooks stay in lockstep).
    static let active: RemoteConfig? = {
        let env = ProcessInfo.processInfo.environment
        guard let raw = env["ENGRAM_URL"], let url = URL(string: raw),
              let token = env["ENGRAM_TOKEN"], !token.isEmpty else { return nil }
        return RemoteConfig(baseURL: url, token: token, project: env["ENGRAM_PROJECT"])
    }()
}

enum RemoteMemory {

    /// POST /advise — returns the injection-ready "## Relevant memories"
    /// block (fenced/ranked/budgeted server-side), or nil.
    static func advise(_ config: RemoteConfig, prompt: String,
                       project: String?, timeout: TimeInterval = 5) async -> String? {
        struct Body: Encodable { let prompt: String; let project: String? }
        struct Reply: Decodable { let block: String? }
        let reply: Reply? = await post(config, path: "/advise",
                                       body: Body(prompt: prompt, project: project),
                                       timeout: timeout)
        return reply?.block
    }

    /// Rendered recall text via the MCP mount (for on-start/pre-tool, which
    /// wrap it under their own section headers). Returns nil on "No
    /// memories found." so callers skip the section entirely.
    static func recallRendered(_ config: RemoteConfig, query: String,
                               project: String?, timeout: TimeInterval = 5) async -> String? {
        var arguments: [String: Any] = ["query": query, "depth": 1, "limit": 5]
        if let project { arguments["project"] = project }
        let rpc: [String: Any] = [
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": ["name": "recall", "arguments": arguments],
        ]
        guard let data = await postRaw(config, path: "/mcp", json: rpc, timeout: timeout),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let result = obj["result"] as? [String: Any],
              let content = result["content"] as? [[String: Any]],
              let text = content.first?["text"] as? String,
              !text.isEmpty, text != "No memories found." else { return nil }
        return text
    }

    struct NoveltyVerdict: Decodable {
        let index: Int
        let distance: Double
        let novel: Bool
    }

    /// POST /sample-gate — server embeds each candidate (TEI) and KNNs the
    /// agent's graph; returns per-candidate novelty. nil = endpoint
    /// unreachable (callers fail OPEN and spawn the learner).
    static func sampleGate(_ config: RemoteConfig, texts: [String],
                           timeout: TimeInterval = 30) async -> [NoveltyVerdict]? {
        struct Body: Encodable { let texts: [String] }
        struct Reply: Decodable { let results: [NoveltyVerdict] }
        let reply: Reply? = await post(config, path: "/sample-gate",
                                       body: Body(texts: texts), timeout: timeout)
        return reply?.results
    }

    // MARK: - Transport

    private static func post<B: Encodable, R: Decodable>(
        _ config: RemoteConfig, path: String, body: B, timeout: TimeInterval
    ) async -> R? {
        guard let payload = try? JSONEncoder().encode(body) else { return nil }
        guard let data = await send(config, path: path, payload: payload, timeout: timeout) else {
            return nil
        }
        return try? JSONDecoder().decode(R.self, from: data)
    }

    private static func postRaw(_ config: RemoteConfig, path: String,
                                json: [String: Any], timeout: TimeInterval) async -> Data? {
        guard let payload = try? JSONSerialization.data(withJSONObject: json) else { return nil }
        return await send(config, path: path, payload: payload, timeout: timeout)
    }

    private static func send(_ config: RemoteConfig, path: String,
                             payload: Data, timeout: TimeInterval) async -> Data? {
        var request = URLRequest(url: config.baseURL.appendingPathComponent(path))
        request.httpMethod = "POST"
        request.httpBody = payload
        request.timeoutInterval = timeout
        request.setValue("Bearer \(config.token)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue("2025-03-26", forHTTPHeaderField: "MCP-Protocol-Version")
        // Analytics attribution: the hook binary shares the agent's token, so
        // without this its ops are indistinguishable from the agent's own.
        request.setValue("memory-hooks", forHTTPHeaderField: "X-Engram-Client")

        // Continuation-based (not the async overloads): identical behavior
        // on Darwin and corelibs-foundation.
        return await withCheckedContinuation { continuation in
            let task = URLSession.shared.dataTask(with: request) { data, response, error in
                if let error {
                    hookLog("remote \(path) failed: \(error)")
                    continuation.resume(returning: nil)
                    return
                }
                guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                    let code = (response as? HTTPURLResponse)?.statusCode ?? -1
                    hookLog("remote \(path) status \(code)")
                    continuation.resume(returning: nil)
                    return
                }
                continuation.resume(returning: data)
            }
            task.resume()
        }
    }
}

extension RemoteConfig {
    /// Write (or refresh) an MCP config for spawned learner subprocesses and
    /// return its path. Passed via `--mcp-config` + `--strict-mcp-config` so
    /// the learner's tooling does NOT depend on a `.mcp.json` existing in its
    /// cwd (bare-mode sandbox agents have none — the orchestrator passes the
    /// main agent its config as a CLI flag the learner never inherits).
    ///
    /// The `X-Engram-Client: session-learner` header is the analytics
    /// attribution: learner ops become distinguishable rows in `memory_ops`,
    /// so "did the learner run?" is a dashboard query, not a sandbox exec.
    func writeLearnerMcpConfig() -> String? {
        let dir = NSHomeDirectory() + "/.claude"
        let path = dir + "/learner-mcp.json"
        let config: [String: Any] = [
            "mcpServers": [
                "engram": [
                    "type": "http",
                    "url": baseURL.appendingPathComponent("mcp").absoluteString,
                    "headers": [
                        "Authorization": "Bearer \(token)",
                        "X-Engram-Client": "session-learner",
                    ],
                ]
            ]
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: config) else { return nil }
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        guard FileManager.default.createFile(
            atPath: path, contents: data,
            attributes: [.posixPermissions: 0o600]) else { return nil }
        return path
    }

    /// CLI flags for a learner spawn: explicit MCP config when it can be
    /// written, empty (fall back to cwd discovery) when it can't.
    var learnerMcpArgs: String {
        guard let path = writeLearnerMcpConfig() else { return "" }
        return "--mcp-config '\(path)' --strict-mcp-config"
    }
}
