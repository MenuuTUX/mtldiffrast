# Architecture Overview

MTLDiffRast is organized as a small Swift Package around one `Rasterizer` class and one Metal shader source file.

## Layers

```text
Swift caller
  |
  v
Rasterizer public API
  |
  v
Metal buffer packing and command encoding
  |
  v
Resources/MetalShaders.metal compute kernels
  |
  v
Apple GPU
```

## Rasterizer

`Rasterizer` owns:

- `MTLDevice`
- `MTLCommandQueue`
- compiled `MTLLibrary`
- cached `MTLComputePipelineState` objects
- a serial dispatch queue for command encoding

Swift arrays are copied into shared Metal buffers for each call. Results are copied back into Swift output structs after command completion.

## Kernels

| Kernel | Purpose | Grid |
| --- | --- | --- |
| `rasterizeKernel` | Forward triangle rasterization | `width x height` |
| `rasterizeBackwardKernel` | Vertex-position gradients from raster output gradients | `width x height` |
| `interpolateKernel` | Per-vertex attribute interpolation | `width x height` |
| `interpolateBackwardKernel` | Attribute and raster-output gradients | `width x height` |
| `antialiasKernel` | Silhouette color/coverage antialiasing | `width x height` |
| `textureKernel` | Nearest/linear texture sampling | `outWidth x outHeight` |
| `textureBackwardKernel` | Texture and UV gradients for sampling | `outWidth x outHeight` |

## Raster Output Layout

The Metal rasterizer writes one `float4` per pixel:

```text
(u, v, zOverW, triIdPlusOne)
```

A second `float4` buffer stores barycentric derivatives:

```text
(du/dx, du/dy, dv/dx, dv/dy)
```

Swift exposes those buffers as `RasterOutput.triangleIds`, `depthBuffer`, `barycentrics`, and `baryDerivatives`. Uncovered pixels have `triangleIds[pixel] == -1`.

Output arrays are row-major with `y = 0` at the bottom of clip space.

## Depth and Facing

The forward rasterizer evaluates projected signed area and skips triangles with non-positive area. Among covered front-facing triangles, the largest `z / w` value wins the depth test.

## Backward Passes

`rasterizeBackward` follows the Python reference math for barycentric and barycentric-derivative gradients, accumulating into vertex positions with atomic float adds.

`interpolateBackward` returns:

- `gradAttributes`, laid out as `[vertex * numAttributes + attribute]`
- `gradRast`, laid out as `[pixel * 4 + component]`

`textureBackward` accumulates texel gradients for nearest and linear filtering. UV gradients are zero for nearest filtering and populated for linear filtering.

## Demo

The macOS demo is intentionally a package client. Its local `DiffRasterizer` adapter converts simple demo vertices into package inputs, then displays the returned color buffer in SwiftUI.
