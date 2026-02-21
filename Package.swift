// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "ClaudeMemory",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "ClaudeMemoryLib", targets: ["ClaudeMemoryLib"]),
    ],
    dependencies: [
        .package(url: "https://github.com/modelcontextprotocol/swift-sdk.git", from: "0.10.0"),
        .package(url: "https://github.com/jsflax/lattice.git", from: "0.3.2"),
        .package(url: "https://github.com/jsflax/SwiftLM.git", branch: "main"),
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.5.0"),
    ],
    targets: [
        .target(
            name: "ClaudeMemoryLib",
            dependencies: [
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
            ]
        ),
        .executableTarget(
            name: "ClaudeMemory",
            dependencies: [
                "ClaudeMemoryLib",
                .product(name: "MCP", package: "swift-sdk"),
                .product(name: "Lattice", package: "Lattice"),
            ],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
            ]
        ),
        .executableTarget(
            name: "MemoryVisualizer",
            dependencies: [
                "ClaudeMemoryLib",
                .product(name: "Lattice", package: "Lattice"),
            ],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
            ],
            linkerSettings: [
                .linkedFramework("RealityKit"),
            ]
        ),
        .executableTarget(
            name: "ClaudeMemoryHooks",
            dependencies: [
                "ClaudeMemoryLib",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Lattice", package: "Lattice"),
            ],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
            ]
        ),
        .testTarget(
            name: "ClaudeMemoryTests",
            dependencies: [
                "ClaudeMemoryLib",
                .product(name: "Lattice", package: "Lattice"),
            ],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
            ]
        ),
    ]
)
