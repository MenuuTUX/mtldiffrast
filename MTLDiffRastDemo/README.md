# MTLDiffRast Demo App

A macOS demo application showcasing the Swift package implementation of MTLDiffRast. Rendering is presented through MetalKit's `MTKView`, so animated scenes are visible to Xcode's Metal/FPS diagnostics and static scenes do not burn CPU while idle.

## Requirements

- macOS 14.0 or later
- Xcode 15.0 or later
- Metal-compatible GPU

## Getting Started

1. Open the project:
   ```bash
   open MTLDiffRastDemo.xcodeproj
   ```

2. Build and run (⌘+R)

## Features Demonstrated

### 1. Basic Rasterization
Renders a single solid-color triangle through the package rasterizer.

### 2. Gradient Fill
Shows smooth color gradients across triangle surfaces using barycentric interpolation.

### 3. Barycentric Interpolation
Visualizes how barycentric coordinates work for attribute interpolation.

### 4. Antialiasing
Demonstrates edge smoothing techniques for triangle rendering.

### 5. Multiple Triangles
Renders multiple overlapping triangles with package depth testing.

### 6. Depth Testing
Shows proper depth buffering with three overlapping triangles at different depths.

### 7. Original Sample Demos
The official sample ideas are available directly in the sidebar:

- `Sample: Triangle`
- `Sample: Cube`
- `Sample: Earth`
- `Sample: Pose`
- `Sample: EnvPhong`

### 8. Performance Benchmark
Measure rasterization performance with configurable triangle counts (1-1000).

## Project Structure

```
MTLDiffRastDemo/
├── MTLDiffRastDemo.xcodeproj    # Xcode project file
└── MTLDiffRastDemo/
    ├── AppDelegate.swift        # App entry point
    ├── ContentView.swift        # Main UI layout
    ├── DemoViewModel.swift      # Demo logic and triangle generation
    ├── MetalView.swift          # MetalKit presentation surface for package output
    ├── MTLDiffRast.swift        # Demo adapter around the package API
    ├── OriginalSamples.swift    # Xcode-app versions of the original samples
    ├── Assets.xcassets/         # App assets
    └── MTLDiffRastDemo.entitlements
```

## Usage Tips

- **Sidebar**: Select different demo features from the left sidebar
- **Wireframe Toggle**: Enable to see triangle outlines
- **Benchmark**: Adjust triangle count and run performance tests

## Architecture

The app follows MVVM pattern:
- **Model**: `Vertex`, `RasterizationResult` structs
- **View**: `ContentView`, `MetalView` SwiftUI views
- **ViewModel**: `DemoViewModel` manages state and demo logic

## Troubleshooting

If the app fails to launch:
1. Ensure you're on macOS 14.0+
2. Check that your Mac has Metal support
3. Try cleaning the build folder (⇧⌘+K)
4. Delete derived data and rebuild

## License

Same as the main MTLDiffRast project.
