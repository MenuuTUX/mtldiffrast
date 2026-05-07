// swift-tools-version:5.9

import PackageDescription

let package = Package(
    name: "MTLDiffRast",
    platforms: [
        .macOS(.v12),
        .iOS(.v15),
        .tvOS(.v15),
    ],
    products: [
        .library(
            name: "MTLDiffRast",
            targets: ["MTLDiffRast"]
        ),
    ],
    targets: [
        .target(
            name: "MTLDiffRast",
            dependencies: [],
            path: "Sources/MTLDiffRast",
            resources: [
                .process("Resources"),
            ],
            swiftSettings: [
                .enableUpcomingFeature("StrictConcurrency"),
            ]
        ),
        .testTarget(
            name: "MTLDiffRastTests",
            dependencies: ["MTLDiffRast"],
            path: "Tests/MTLDiffRastTests"
        ),
    ],
    swiftLanguageVersions: [.v5]
)
