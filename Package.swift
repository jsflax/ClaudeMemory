// swift-tools-version: 6.2

import PackageDescription
import Foundation

let bridgingHeader = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .appendingPathComponent("Sources/EngramVisualizer/EngramVisualizer-Bridging-Header.h")
    .path

let package = Package(
    name: "Engram",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "EngramKit", targets: ["EngramKit"]),
        .library(name: "EngramModels", targets: ["EngramModels"]),
    ],
    dependencies: [
        .package(url: "https://github.com/modelcontextprotocol/swift-sdk.git", from: "0.10.0"),
        .package(url: "https://github.com/jsflax/lattice", from: "0.4.1"),
        .package(url: "https://github.com/jsflax/SwiftLM.git", branch: "main"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.5.0"),
        .package(url: "https://github.com/sparkle-project/Sparkle.git", from: "2.0.0"),
        .package(url: "https://github.com/google/GoogleSignIn-iOS.git", from: "7.0.0"),
    ],
    targets: [
        .target(
            name: "EngramModels",
            dependencies: [
                .product(name: "Lattice", package: "Lattice"),
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
                .product(name: "Lattice", package: "Lattice"),
                .product(name: "SwiftLM", package: "SwiftLM"),
            ],
            resources: [
                .copy("Resources/paraphrase-MiniLM-L6-v2_Embedding.mlmodelc"),
                .copy("Resources/paraphrase-MiniLM-L6-v2_tokenizer"),
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
                .product(name: "Lattice", package: "Lattice"),
            ],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
            ]
        ),
        .executableTarget(
            name: "EngramVisualizer",
            dependencies: [
                "EngramKit",
                .product(name: "Lattice", package: "Lattice"),
                .product(name: "Sparkle", package: "Sparkle"),
                .product(name: "GoogleSignIn", package: "GoogleSignIn-iOS"),
                .product(name: "GoogleSignInSwift", package: "GoogleSignIn-iOS"),
            ],
            resources: [
                .copy("Resources/under_construction.png"),
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
                .product(name: "Lattice", package: "Lattice"),
            ],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
            ]
        ),
        .testTarget(
            name: "EngramTests",
            dependencies: [
                "EngramKit",
                .product(name: "Lattice", package: "Lattice"),
            ],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
            ]
        ),
    ]
)
