# MTLDiffRast

MTLDiffRast is a Swift Package that implements differentiable rasterization primitives on Apple GPUs with Metal compute shaders. It mirrors the practical API surface of `mtldiffrast-python`/`nvdiffrast` where that maps cleanly to a small pure-Swift package.

## Features

<<<<<<< HEAD
- **Pure Swift API** - Public surface is Swift only; no Objective-C bridging required. Compute kernels are written in Metal Shading Language and shipped as `.metal` source files (compiled at runtime).
- **No external dependencies** - Just `Metal`, `simd`, `Foundation`.
- **Metal Accelerated** - GPU compute shaders for forward, backward, interpolation, and antialiasing.
- **Differentiable** - Backward pass produces gradients of the depth output w.r.t. clip-space `z` and `w` (silhouette/coverage gradients are not yet implemented — see *Status* below).
- **Modern API** - Swift-native interface with proper error handling.
=======
- Forward triangle rasterization in clip space
- Barycentric output and barycentric screen-space derivatives
- Rasterization backward pass for vertex-position gradients
- Attribute interpolation and interpolation backward pass
- Optional attribute derivatives from raster barycentric derivatives
- Texture sampling with nearest/linear filtering and wrap/clamp/zero boundaries
- Texture backward pass for texel gradients and linear-filter UV gradients
- Silhouette antialiasing for color or coverage buffers
- GPU display-texture output for low-overhead MetalKit viewers
- macOS demo app backed by the package implementation
- Xcode demo app with Swift remakes of the nvdiffrast sample demos
>>>>>>> 077888c (important)

## Requirements

- macOS 12.0+ / iOS 15.0+ / tvOS 15.0+
- A Metal-capable Apple GPU; Apple Silicon is the primary target
- Xcode 15.0+
- Swift 5.9+

## Installation

Add the package to `Package.swift`:

```swift
dependencies: [
<<<<<<< HEAD
    .package(url: "https://github.com/MenuuTUX/mtldiffrast.git", from: "0.1.0")
=======
    .package(url: "https://github.com/yourusername/mtldiffrast-swift.git", from: "1.0.0")
>>>>>>> 077888c (important)
]
```

Then add `MTLDiffRast` as a dependency of your target.

## Quick Start

```swift
import MTLDiffRast
import simd

let rasterizer = try Rasterizer()

let positions: [SIMD4<Float>] = [
    SIMD4<Float>(0.0, 0.7, 0.5, 1.0),
    SIMD4<Float>(-0.7, -0.6, 0.5, 1.0),
    SIMD4<Float>(0.7, -0.6, 0.5, 1.0)
]

let triangles: [SIMD3<Int32>] = [
    SIMD3<Int32>(0, 1, 2)
]

let rast = try rasterizer.rasterize(
    positions: positions,
    triangles: triangles,
    width: 512,
    height: 512
)

let colors: [Float] = [
    1, 0, 0,
    0, 1, 0,
    0, 0, 1
]

let interp = try rasterizer.interpolate(
    attributes: colors,
    triangles: triangles,
    rasterOutput: rast,
    numAttributes: 3,
    computeDerivatives: true
)

let antialiased = try rasterizer.antialias(
    color: interp.attributes,
    channels: 3,
    rasterOutput: rast,
    positions: positions,
    triangles: triangles
)
```

## Gradients

`rasterizeBackward` accepts upstream gradients for the raster output. Pass either `pixelCount` scalars to drive the `u` barycentric component only, or `pixelCount * 4` floats laid out as `(du, dv, dz, dtri)` per pixel. Like the Python reference kernel, the position-gradient path uses `u`, `v`, and optional barycentric-derivative gradients; `z` and triangle-ID gradients are ignored. If your loss depends on `RasterOutput.baryDerivatives`, pass `gradBaryDerivatives` as `pixelCount * 4` floats.

```swift
let dy = Array(repeating: Float(1), count: rast.pixelCount * 4)

let gradients = try rasterizer.rasterizeBackward(
    positions: positions,
    triangles: triangles,
    forwardOutput: rast,
    gradOutput: dy,
    vertexCount: positions.count
)

print(gradients.positionGradients)
```

Texture sampling also has a backward pass:

```swift
let texGrad = try rasterizer.textureBackward(
    texture: texture,
    texWidth: texWidth,
    texHeight: texHeight,
    channels: channels,
    uv: uv,
    gradOutput: textureDy,
    outWidth: width,
    outHeight: height,
    filterMode: .linear,
    boundaryMode: .wrap
)
```

## Coordinate Conventions

Input positions are clip-space `(x, y, z, w)`. The kernels evaluate pixel centers in normalized device coordinates after perspective divide. Output arrays are row-major with `y = 0` at the bottom of clip space, matching the Python reference kernels. If you display the buffer in a top-left-origin UI, flip rows during presentation.

Triangles with non-positive signed area after projection are culled as back-facing or degenerate.

## Current Scope

This package intentionally focuses on the core primitives needed for the Swift/Metal port:

<<<<<<< HEAD
#### `RasterOutput`
- `width`, `height` - Output dimensions
- `triangleIds` - Triangle ID per pixel (-1 if no triangle)
- `depthBuffer` - Interpolated NDC depth per pixel
- `barycentrics` - Screen-space barycentric coordinates per pixel, packed `[b0, b1, b2, ...]` (length `pixelCount * 3`)
- `pixelCount` - Total number of pixels
=======
- no CUDA/PyTorch dependency or autograd bridge
- no batched tensor API
- no mipmapped/cube texture sampling
- no antialias backward pass or topology-hash cache
- integer triangle indices are not differentiable, so `triangleGradients` is always `nil`
>>>>>>> 077888c (important)

The tests include CPU-reference parity checks, finite-difference texture-gradient checks, barycentric derivative checks, and demo build coverage.

## Demo App Samples

The original sample demos are available from the `MTLDiffRastDemo` Xcode app sidebar:

- `Sample: Triangle`
- `Sample: Cube`
- `Sample: Earth`
- `Sample: Pose`
- `Sample: EnvPhong`

Open `MTLDiffRastDemo/MTLDiffRastDemo.xcodeproj`, build the `MTLDiffRastDemo` scheme, and select the sample from the sidebar. The demo presents frames through `MTKView`, so animated samples show up in Xcode's Metal/FPS diagnostics while static samples idle after drawing.

## Development

Run the package tests:

```bash
swift test
```

Build the demo:

<<<<<<< HEAD
## Status

This is an early release (0.1.x). What works today:

- Forward rasterization with proper depth test
- Per-pixel barycentric coordinates exposed in `RasterOutput`
- GPU attribute interpolation
- Edge-aware depth-only antialiasing
- Backward pass for the depth output: gradients flow into clip-space `z` and `w` of each vertex via atomic float accumulation

Not yet implemented:

- Silhouette / coverage gradients for `x` and `y` (the analogue of the AA-aware backward pass in *nvdiffrast*)
- Texture sampling kernels (the `TextureOutput` struct is reserved for a future release)

API may change before 1.0.

## Examples

See the `Tests/MTLDiffRastTests` directory for usage examples and test cases.
=======
```bash
xcodebuild -project MTLDiffRastDemo/MTLDiffRastDemo.xcodeproj -scheme MTLDiffRastDemo -configuration Debug build
```
>>>>>>> 077888c (important)

## License

MIT License. See `LICENSE` for details.
