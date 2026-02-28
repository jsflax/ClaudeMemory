import FoundationModels

/// Structured output from the retrieval stage of the two-stage chat pipeline.
/// The on-device model calls memory tools and returns results in this typed format
/// via constrained decoding (`respond(to:generating: MemoryContext.self)`).
@available(macOS 26.0, iOS 26.0, *)
@Generable
public struct MemoryContext {
    @Guide(description: "Relevant memories found, each as a short summary")
    public var memories: [MemorySnippet]

    @Guide(description: "Brief summary of what was found, for the response model")
    public var summary: String
}

/// A single memory item extracted during the retrieval stage.
@available(macOS 26.0, iOS 26.0, *)
@Generable
public struct MemorySnippet {
    @Guide(description: "The memory content")
    public var content: String

    @Guide(description: "Project scope")
    public var project: String

    @Guide(description: "Topic")
    public var topic: String
}
