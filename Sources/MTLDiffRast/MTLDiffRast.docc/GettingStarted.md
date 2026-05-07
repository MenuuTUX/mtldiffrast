# Getting Started with MTLDiffRast

Render a differentiable triangle in five steps.

## Overview

MTLDiffRast provides Metal-accelerated differentiable rasterization primitives
for Swift.  This article walks through the most common workflow: rasterize a
mesh, interpolate vertex attributes, apply silhouette antialiasing, then read
back gradients for vertex positions.

## Step 1 — Create a Rasterizer

```swift
import MTLDiffRast
import simd

let rast = try Rasterizer()
```

``Rasterizer/init()`` loads and compiles the Metal compute kernels from the
bundle.  Create a single instance and reuse it across frames.

## Step 2 — Define Geometry

Positions are clip-space `(x, y, z, w)`.  Triangles use zero-based vertex
indices in counter-clockwise winding order.

```swift
let positions: [SIMD4<Float>] = [
    SIMD4<Float>( 0.0,  0.7, 0.5, 1.0),   // top
    SIMD4<Float>(-0.7, -0.6, 0.5, 1.0),   // bottom-left
    SIMD4<Float>( 0.7, -0.6, 0.5, 1.0),   // bottom-right
]
let triangles: [SIMD3<Int32>] = [SIMD3<Int32>(0, 1, 2)]
```

## Step 3 — Rasterize

```swift
let output = try rast.rasterize(
    positions: positions,
    triangles: triangles,
    width: 512, height: 512
)
// output.triangleIds  — [Int32], -1 for background pixels
// output.barycentrics — [SIMD2<Float>], (u, v) per pixel
// output.depthBuffer  — [Float], z/w per pixel
```

## Step 4 — Interpolate Vertex Attributes

```swift
// RGB colours per vertex (row-major, 3 floats each)
let vertexColors: [Float] = [
    1, 0, 0,   // red   at top
    0, 1, 0,   // green at bottom-left
    0, 0, 1,   // blue  at bottom-right
]

let interp = try rast.interpolate(
    attributes: vertexColors,
    triangles: triangles,
    rasterOutput: output,
    numAttributes: 3,
    computeDerivatives: false
)
// interp.attributes — [Float], layout [pixelCount × 3]
```

## Step 5 — Antialias

```swift
let aa = try rast.antialias(
    color: interp.attributes,
    channels: 3,
    rasterOutput: output,
    positions: positions,
    triangles: triangles
)
// aa.colors — [Float], silhouette edges blended for gradient flow
```

## Backward Pass (Gradients)

Pass an upstream loss gradient to recover vertex-position gradients:

```swift
// Constant upstream gradient of 1 for every pixel / channel
let dy = [Float](repeating: 1, count: output.pixelCount * 4)

let grads = try rast.rasterizeBackward(
    positions: positions,
    triangles: triangles,
    forwardOutput: output,
    gradOutput: dy,
    vertexCount: positions.count
)
// grads.positionGradients — [Float], layout [vertexCount × 4]
```

## Coordinate Conventions

| Convention | Value |
|------------|-------|
| Origin | Bottom-left `(0, 0)` |
| Y axis | Up (matching OpenGL / nvdiffrast) |
| Depth | Largest `z/w` wins (reversed-Z NDC) |
| Winding | CCW front-facing |

## Display Shortcut

For real-time preview without CPU readback, use
``Rasterizer/rasterizeColorTexture(positions:triangles:colors:width:height:antialias:)``
to get a `MTLTexture` in a single GPU pass:

```swift
let tex = try rast.rasterizeColorTexture(
    positions: positions,
    triangles: triangles,
    colors: vertexColors,
    width: 512, height: 512,
    antialias: true
)
```
