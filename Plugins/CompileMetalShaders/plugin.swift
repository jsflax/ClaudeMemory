import PackagePlugin
import Foundation

@main
struct CompileMetalShaders: BuildToolPlugin {
    func createBuildCommands(context: PluginContext, target: Target) throws -> [Command] {
        guard let sourceTarget = target as? SourceModuleTarget else { return [] }

        let metalFiles = sourceTarget.sourceFiles.filter { $0.path.extension == "metal" }
        guard !metalFiles.isEmpty else { return [] }

        let headerFiles = sourceTarget.sourceFiles.filter { $0.path.extension == "h" }

        let outputDir = context.pluginWorkDirectory
        var airFiles: [Path] = []
        var commands: [Command] = []

        for file in metalFiles {
            let airPath = outputDir.appending("\(file.path.stem).air")
            airFiles.append(airPath)

            var arguments = ["-c", file.path.string, "-o", airPath.string]
            if let firstHeader = headerFiles.first {
                arguments += ["-I", firstHeader.path.removingLastComponent().string]
            }

            commands.append(.buildCommand(
                displayName: "Compile Metal: \(file.path.lastComponent)",
                executable: .init("/usr/bin/xcrun"),
                arguments: ["metal"] + arguments,
                inputFiles: [file.path] + headerFiles.map(\.path),
                outputFiles: [airPath]
            ))
        }

        let metallibPath = outputDir.appending("default.metallib")
        commands.append(.buildCommand(
            displayName: "Link metallib",
            executable: .init("/usr/bin/xcrun"),
            arguments: ["metallib"] + airFiles.map(\.string) + ["-o", metallibPath.string],
            inputFiles: airFiles,
            outputFiles: [metallibPath]
        ))

        return commands
    }
}
