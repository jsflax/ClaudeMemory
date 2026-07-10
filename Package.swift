// swift-tools-version: 6.2

import PackageDescription
import Foundation

let bridgingHeader = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .appendingPathComponent("Sources/EngramVisualizer/EngramVisualizer-Bridging-Header.h")
    .path

var engramMetalShaders: PackageDescription.Target {
    #if SWIFT_PACKAGE_TEST
    .target(
        name: "EngramMetalShaders",
        resources: [.process("Shaders")],
        plugins: [.plugin(name: "CompileMetalShaders")]
    )
    #else
    .target(
        name: "EngramMetalShaders",
        resources: [.process("Shaders")]
    )
    #endif
}

let package = Package(
    name: "Engram",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "EngramKit", targets: ["EngramKit"]),
        .library(name: "EngramModels", targets: ["EngramModels"]),
        .library(name: "EngramFoundationModels", targets: ["EngramFoundationModels"]),
        .library(name: "EngramSceneKit", targets: ["EngramSceneKit"]),
        .library(name: "EngramMetalShaders", targets: ["EngramMetalShaders"]),
        .library(name: "CEngramSceneTypes", targets: ["CEngramSceneTypes"]),
        .library(name: "EngramRealityKit", targets: ["EngramRealityKit"]),
    ],
    dependencies: [
        .package(url: "https://github.com/modelcontextprotocol/swift-sdk.git", from: "0.12.0"),
        .package(url: "https://github.com/jsflax/lattice.git", from: "0.10.7"),
        .package(url: "https://github.com/jsflax/SwiftLM.git", branch: "main"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.5.0"),
        .package(url: "https://github.com/sparkle-project/Sparkle.git", from: "2.0.0"),
        .package(url: "https://github.com/google/GoogleSignIn-iOS.git", from: "7.0.0"),
    ],
    targets: [
        // C target exposing SharedTypes.h (GPU struct definitions) to Swift modules.
        // Keep this header in sync with Sources/EngramVisualizer/Metal/SharedTypes.h.
        .target(
            name: "CEngramSceneTypes"
        ),
        // Pure-logic library for scene frame building — no Metal, no @MainActor.
        .target(
            name: "EngramSceneKit",
            dependencies: ["CEngramSceneTypes"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
            ],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("Accelerate"),
            ]
        ),
        // Metal shader library. Plugin compiles .metal → metallib at build time.
        engramMetalShaders,
        .target(
            name: "EngramModels",
            dependencies: [
                .product(name: "Lattice", package: "lattice"),
            ],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
            ]
        ),
        .target(
            name: "EngramKit",
            dependencies: [
                "EngramModels",
                .product(name: "MCP", package: "swift-sdk"),
                .product(name: "Lattice", package: "lattice"),
                .product(name: "SwiftLM", package: "SwiftLM"),
            ],
            resources: [
                .copy("Resources/paraphrase-MiniLM-L6-v2_Embedding.mlmodelc"),
                .copy("Resources/paraphrase-MiniLM-L6-v2_tokenizer"),
                .copy("Resources/RecallGateClassifier.mlmodelc"),
                .copy("Resources/session-learner.md"),
                .copy("Resources/memory-maintenance.md"),
                .copy("Resources/sync-reconciliation.md"),
            ],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
            ],
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("NaturalLanguage"),
            ]
        ),
        .executableTarget(
            name: "Engram",
            dependencies: [
                "EngramKit",
                .product(name: "MCP", package: "swift-sdk"),
                .product(name: "Lattice", package: "lattice"),
            ],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
            ]
        ),
        .executableTarget(
            name: "EngramVisualizer",
            dependencies: [
                "EngramKit",
                "EngramFoundationModels",
                "EngramSceneKit",
                "EngramMetalShaders",
                "CEngramSceneTypes",
                "EngramRealityKit",
                .product(name: "Lattice", package: "lattice"),
                .product(name: "Sparkle", package: "Sparkle"),
                .product(name: "GoogleSignIn", package: "GoogleSignIn-iOS"),
                .product(name: "GoogleSignInSwift", package: "GoogleSignIn-iOS"),
            ],
            exclude: [
                "Info.plist",
            ],
            resources: [
                .copy("Resources/under_construction.png"),
                .copy("Resources/mascot_mesh.bin"),
                .copy("Resources/mascot_basecolor.jpg"),
                .copy("Resources/mascot_metalrough.png"),
            ],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(["-import-objc-header", bridgingHeader]),
            ],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("MetalKit"),
                .linkedFramework("RealityKit"),
            ]
        ),
        .executableTarget(
            name: "EngramHooks",
            dependencies: [
                "EngramKit",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Lattice", package: "lattice"),
            ],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
            ]
        ),
        .executableTarget(
            name: "EngramDaemon",
            dependencies: [
                "EngramKit",
                .product(name: "Lattice", package: "lattice"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
            ]
        ),
        .target(
            name: "EngramFoundationModels",
            dependencies: [
                "EngramKit",
                .product(name: "MCP", package: "swift-sdk"),
            ],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
            ]
        ),
        .target(
            name: "EngramRealityKit",
            dependencies: ["EngramSceneKit", "CEngramSceneTypes"],
            resources: [.process("Shaders"), .copy("Resources/mascot.usdz")],
            swiftSettings: [.interoperabilityMode(.Cxx)],
            linkerSettings: [
                .linkedFramework("RealityKit"),
                .linkedFramework("Metal"),
            ]
        ),
        .executableTarget(
            name: "EngramPreview",
            dependencies: [
                "EngramRealityKit", "EngramSceneKit", "EngramMetalShaders",
                "EngramKit",
                .product(name: "Lattice", package: "lattice"),
            ],
            resources: [.copy("Resources/mascot.usdz")],
            swiftSettings: [.interoperabilityMode(.Cxx)],
            linkerSettings: [.linkedFramework("RealityKit")]
        ),
        .testTarget(
            name: "EngramRealityKitTests",
            dependencies: ["EngramRealityKit"],
            swiftSettings: [.interoperabilityMode(.Cxx)]
        ),
        .testTarget(
            name: "EngramTests",
            dependencies: [
                "EngramKit",
                .product(name: "Lattice", package: "lattice"),
            ],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
            ]
        ),
        .testTarget(
            name: "EngramSceneKitTests",
            dependencies: ["EngramSceneKit", "EngramMetalShaders"],
            swiftSettings: [.interoperabilityMode(.Cxx)]
        ),
        .testTarget(
            name: "EngramVisualizerTests",
            dependencies: ["EngramSceneKit", "EngramMetalShaders"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
            ]
        ),
        .testTarget(
            name: "EngramFoundationModelsTests",
            dependencies: [
                "EngramFoundationModels",
                "EngramKit",
                .product(name: "Lattice", package: "lattice"),
            ],
            resources: [
                .copy("Resources/engram_memory.fmadapter"),
            ],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
            ]
        ),
        .plugin(
            name: "CompileMetalShaders",
            capability: .buildTool()
        ),
    ]
)
