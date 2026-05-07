# ``MTLDiffRast``

Metal-accelerated differentiable rasterization for Apple Silicon.

## Overview

MTLDiffRast is a pure-Swift package that implements the five core primitives
of differentiable rendering — rasterize, interpolate, antialias, texture sample,
and their backward passes — using Metal compute shaders.

It mirrors the practical API surface of
[nvdiffrast](https://github.com/NVlabs/nvdiffrast) /
[mtldiffrast-python](https://github.com/your-org/mtldiffrast-python), adapted
for Apple GPU targets with no CUDA or PyTorch dependency.

### Key design goals

- **Pure Swift** — no C++ bridging, no Objective-C, no external dependencies.
- **Zero-copy on Apple Silicon** — all Metal buffers use `storageModeShared`;
  GPU results are accessible from Swift without any `blit` or staging copy.
- **Single command buffer per call** — the GPU display path chains rasterize,
  interpolate, antialias, and pack into one command buffer, minimising
  CPU–GPU synchronisation.
- **Runtime shader compilation** — the `.metal` source is bundled as a
  resource; SPM copies it at build time and the library compiles it on first
  use.  Pre-compiled `.metallib` is preferred when available.

## Topics

### Guides & Tutorials

- <doc:GettingStarted>
- <doc:Examples>
- <doc:Architecture>
- <doc:MetalShaders>
- <doc:Performance>

### Troubleshooting & Support

- <doc:FAQ>
- <doc:Troubleshooting>
- <doc:APIReference>
- <doc:Contributing>

### Forward Primitives

- ``Rasterizer/rasterize(positions:triangles:width:height:)``
- ``Rasterizer/interpolate(attributes:triangles:rasterOutput:numAttributes:computeDerivatives:)``
- ``Rasterizer/antialias(color:channels:rasterOutput:positions:triangles:)``
- ``Rasterizer/texture(texture:texWidth:texHeight:channels:uv:outWidth:outHeight:filterMode:boundaryMode:)``
- ``Rasterizer/rasterizeColorTexture(positions:triangles:colors:width:height:antialias:)``

### Backward Primitives

- ``Rasterizer/rasterizeBackward(positions:triangles:forwardOutput:gradOutput:vertexCount:gradBaryDerivatives:)``
- ``Rasterizer/interpolateBackward(attributes:triangles:rasterOutput:gradOutput:numAttributes:)``
- ``Rasterizer/textureBackward(texture:texWidth:texHeight:channels:uv:gradOutput:outWidth:outHeight:filterMode:boundaryMode:)``

### Output Types

- ``RasterOutput``
- ``InterpolateOutput``
- ``AntialiasOutput``
- ``TextureOutput``
- ``RasterGradientOutput``
- ``TextureGradientOutput``

### Configuration

- ``TextureFilterMode``
- ``TextureBoundaryMode``

### Errors

- ``RasterizerError``

### Device Utilities

- ``getMetalDeviceInfo()``
- ``MetalDeviceInfo``
