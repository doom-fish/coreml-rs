// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "CoreMLBridge",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "CoreMLBridge",
            type: .static,
            targets: ["CoreMLBridge"]
        )
    ],
    targets: [
        .target(
            name: "CoreMLBridge",
            path: "Sources/CoreMLBridge",
            publicHeadersPath: "include"
        )
    ]
)
