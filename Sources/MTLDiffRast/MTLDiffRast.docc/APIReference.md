# API Reference

This reference describes the public Swift API exposed by `MTLDiffRast`.

## Rasterizer

```swift
public init() throws
public init(device: MTLDevice) throws
```

Creates a rasterizer using the default Metal device or a supplied device. Initialization compiles the Metal compute shaders used by the package.

### rasterize

```swift
public func rasterize(
    positions: [SIMD4<Float>],
    triangles: [SIMD3<Int32>],
    width: Int,
    height: Int
) throws -> RasterOutput
```

Rasterizes front-facing triangles in clip space. Positions are `(x, y, z, w)` and are perspective-divided in the kernel. The output is row-major with `y = 0` at the bottom of clip space.

### rasterizeBackward

```swift
public func rasterizeBackward(
    positions: [SIMD4<Float>],
    triangles: [SIMD3<Int32>],
    forwardOutput: RasterOutput,
    gradOutput: [Float],
    vertexCount: Int,
    gradBaryDerivatives: [Float]? = nil
) throws -> RasterGradientOutput
```

Computes gradients with respect to vertex positions. `gradOutput` may be either `pixelCount` floats, which are treated as gradients for the `u` barycentric component, or `pixelCount * 4` floats laid out as `(du, dv, dz, dtri)` per pixel. Like the Python reference kernel, this path uses `u`, `v`, and optional barycentric-derivative gradients; `z` and triangle-ID gradients are ignored. `gradBaryDerivatives`, when supplied, must be `pixelCount * 4` floats matching `(du/dx, du/dy, dv/dx, dv/dy)`.

`triangleGradients` is always `nil`; triangle indices are integer topology, not differentiable values.

### interpolate

```swift
public func interpolate(
    attributes: [Float],
    triangles: [SIMD3<Int32>],
    rasterOutput: RasterOutput,
    numAttributes: Int,
    computeDerivatives: Bool = false
) throws -> InterpolateOutput
```

Interpolates per-vertex attributes over covered pixels using the barycentrics stored in `RasterOutput`. `attributes` is laid out as `[vertex * numAttributes + attribute]`.

When `computeDerivatives` is `true`, `InterpolateOutput.attributeDerivatives` is populated with `[pixel * numAttributes * 2 + attribute * 2 + axis]`, where axis `0` is x and axis `1` is y.

### interpolateBackward

```swift
public func interpolateBackward(
    attributes: [Float],
    triangles: [SIMD3<Int32>],
    rasterOutput: RasterOutput,
    gradOutput: [Float],
    numAttributes: Int
) throws -> (gradAttributes: [Float], gradRast: [Float])
```

Backpropagates through attribute interpolation. `gradOutput` is `[pixelCount * numAttributes]`. The returned `gradAttributes` is `[vertexCount * numAttributes]`; `gradRast` is `[pixelCount * 4]`.

### antialias

```swift
public func antialias(
    color: [Float],
    channels: Int,
    rasterOutput: RasterOutput,
    positions: [SIMD4<Float>],
    triangles: [SIMD3<Int32>]
) throws -> AntialiasOutput

public func antialias(
    rasterOutput: RasterOutput,
    positions: [SIMD4<Float>],
    triangles: [SIMD3<Int32>]
) throws -> AntialiasOutput

public func antialias(rasterOutput: RasterOutput) throws -> AntialiasOutput
```

Applies silhouette-edge antialiasing to a color buffer. The two convenience overloads synthesize a one-channel coverage image; the single-argument overload is a compatibility fallback that does not have access to triangle geometry.

### texture

```swift
public func texture(
    texture: [Float],
    texWidth: Int,
    texHeight: Int,
    channels: Int,
    uv: [SIMD2<Float>],
    outWidth: Int,
    outHeight: Int,
    filterMode: TextureFilterMode = .linear,
    boundaryMode: TextureBoundaryMode = .wrap
) throws -> TextureOutput
```

Samples a 2D texture at per-pixel UV coordinates. `texture` is `[texHeight * texWidth * channels]`; `uv` is `[outHeight * outWidth]`.

### textureBackward

```swift
public func textureBackward(
    texture: [Float],
    texWidth: Int,
    texHeight: Int,
    channels: Int,
    uv: [SIMD2<Float>],
    gradOutput: [Float],
    outWidth: Int,
    outHeight: Int,
    filterMode: TextureFilterMode = .linear,
    boundaryMode: TextureBoundaryMode = .wrap
) throws -> TextureGradientOutput
```

Backpropagates through nearest or linear texture sampling. Nearest filtering accumulates texel gradients and returns zero UV gradients. Linear filtering returns both texel gradients and UV gradients.

## Output Types

### RasterOutput

```swift
public struct RasterOutput {
    public let width: Int
    public let height: Int
    public let triangleIds: [Int32]
    public let depthBuffer: [Float]
    public let barycentrics: [SIMD2<Float>]
    public let baryDerivatives: [SIMD4<Float>]
    public var pixelCount: Int { get }
}
```

`triangleIds` is `-1` for uncovered pixels. `barycentrics` stores `(u, v)`; the third barycentric is `1 - u - v`. `baryDerivatives` stores `(du/dx, du/dy, dv/dx, dv/dy)`.

### InterpolateOutput

```swift
public struct InterpolateOutput {
    public let pixelCount: Int
    public let numAttributes: Int
    public let attributes: [Float]
    public let barycentricCoords: [Float]
    public let attributeDerivatives: [Float]?
}
```

`barycentricCoords` stores `(b0, b1, b2)` per pixel.

### TextureOutput

```swift
public struct TextureOutput {
    public let width: Int
    public let height: Int
    public let channels: Int
    public let samples: [Float]
    public var pixelCount: Int { get }
}
```

### TextureGradientOutput

```swift
public struct TextureGradientOutput {
    public let textureGradients: [Float]
    public let uvGradients: [SIMD2<Float>]
}
```

### AntialiasOutput

```swift
public struct AntialiasOutput {
    public let width: Int
    public let height: Int
    public let channels: Int
    public let colors: [Float]
    public var pixelCount: Int { get }
}
```

### RasterGradientOutput

```swift
public struct RasterGradientOutput {
    public let positionGradients: [Float]
    public let triangleGradients: [Float]?
}
```

## Texture Modes

```swift
public enum TextureFilterMode: Int32 {
    case nearest = 0
    case linear = 1
}

public enum TextureBoundaryMode: Int32 {
    case wrap = 0
    case clamp = 1
    case zero = 2
}
```

## Utilities

```swift
public func isMetalAvailable() -> Bool
public func isAppleSilicon() -> Bool
public func getMetalDeviceInfo() -> MetalDeviceInfo?
public func createRasterizer() throws -> Rasterizer
public let version: String
public let description: String
```

## Errors

Most public methods throw `RasterizerError`:

```swift
public enum RasterizerError: LocalizedError {
    case metalUnavailable
    case deviceNotFound
    case pipelineCreationFailed(String)
    case invalidTriangleCount(Int)
    case invalidVertexCount(Int)
    case invalidResolution(width: Int, height: Int)
    case bufferCreationFailed(String)
    case encodingFailed(String)
    case commandExecutionFailed(String)
    case unsupportedFeature(String)
}
```
