import ArgumentParser

@main
struct MemoryHooks: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "memory-hooks",
        abstract: "Claude Code hooks for automatic memory recall and learning",
        subcommands: [
            Advise.self,
            OnStart.self,
            OnEnd.self,
            OnFailure.self,
            PreTool.self,
            PreCompact.self,
        ]
    )
}
