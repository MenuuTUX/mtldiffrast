# Frequently Asked Questions (FAQ)

Common questions about MTLDiffRast with answers and solutions.

## Table of Contents

1. [General Questions](#general-questions)
2. [Installation & Setup](#installation--setup)
3. [Usage & API](#usage--api)
4. [Performance](#performance)
5. [Troubleshooting](#troubleshooting)
6. [Development](#development)

---

## General Questions

### What is MTLDiffRast?

MTLDiffRast is a Swift differentiable rasterization package designed for Apple Silicon (M-series chips). It uses Metal compute shaders to provide GPU-accelerated triangle rasterization with support for forward and backward passes, making it suitable for optimization workflows that require gradient computation.

### Why was MTLDiffRast created?

Traditional rasterization libraries often rely on C++ or Python bindings. MTLDiffRast provides:
- Swift-first API for seamless integration with Swift projects
- Native Metal acceleration optimized for Apple Silicon
- Differentiable operations for ML/optimization workflows
- Zero external dependencies beyond Apple's frameworks

### What platforms are supported?

| Platform | Minimum Version | Notes |
|----------|-----------------|-------|
| macOS | 12.0+ | Apple Silicon only |
| iOS | 15.0+ | A12 Bionic or later recommended |
| tvOS | 15.0+ | Apple TV 4K (2nd gen+) |

**Note:** Apple Silicon is the tested and optimized target. Intel Macs are not part of the supported test matrix.

### Is MTLDiffRast open source?

Yes, MTLDiffRast is released under the MIT License. You can view, modify, and distribute the source code freely.

### How does it compare to other rasterizers?

| Feature | MTLDiffRast | PyTorch3D | Nvdiffrast |
|---------|-------------|-----------|------------|
| Language | Pure Swift | Python/C++ | Python/C++/CUDA |
| Platform | Apple Silicon | Cross-platform | NVIDIA GPUs |
| Differentiable | Yes | Yes | Yes |
| Dependencies | None | PyTorch | PyTorch, CUDA |
| Metal Support | Native | No | No |

---

## Installation & Setup

### How do I install MTLDiffRast?

The recommended method is Swift Package Manager:

```swift
dependencies: [
    .package(url: "https://github.com/yourusername/mtldiffrast.git", from: "1.0.0")
]
```

See <doc:GettingStarted> for detailed instructions.

### Can I use MTLDiffRast in an iOS app?

Yes! MTLDiffRast supports iOS 15.0+. However, note:
- Requires A12 Bionic chip or later for best performance
- Test thoroughly on actual devices (simulator has limited Metal support)
- Consider battery impact for intensive rasterization

### Do I need Xcode to use this library?

For development, yes. Xcode 15.0+ is required. For runtime, you only need the compiled binary and Metal framework (included with macOS/iOS).

### Can I use MTLDiffRast with SwiftUI?

Absolutely! Here's a basic example:

```swift
import SwiftUI
import MTLDiffRast

struct RasterizerView: View {
    @State private var output: RasterOutput?
    
    var body: some View {
        Canvas { context, size in
            if let output = output {
                // Render using Core Graphics based on raster output
            }
        }
        .onAppear {
            Task {
                output = try? await performRasterization()
            }
        }
    }
}
```

---

## Usage & API

### What coordinate system does MTLDiffRast use?

MTLDiffRast uses **clip space coordinates**:
- X: -1 (left) to 1 (right)
- Y: -1 (bottom) to 1 (top)
- Z: 0 (near) to 1 (far)
- W: Perspective divide component

Vertices are transformed to screen space internally:
```swift
screenX = (clipX / clipW + 1) * 0.5 * width
bufferY = (clipY / clipW + 1) * 0.5 * height
```

Output arrays are row-major with `y = 0` at the bottom of clip space. Flip rows when presenting into top-left-origin UI frameworks.

### How do I render multiple triangles?

Simply add more triangles to your array:

```swift
let triangles: [SIMD3<Int32>] = [
    SIMD3<Int32>(0, 1, 2),  // Triangle 1
    SIMD3<Int32>(2, 3, 0),  // Triangle 2
    SIMD3<Int32>(4, 5, 6),  // Triangle 3
    // ... more triangles
]
```

Ensure vertex indices reference valid positions in your position array.

### Can I render textured meshes?

Yes. Use `interpolate` to get UV coordinates, then call the package texture sampler:

```swift
// Interpolate UVs
let uvOutput = try rasterizer.interpolate(
    attributes: uvs,
    triangles: triangles,
    rasterOutput: rasterOutput,
    numAttributes: 2
)

let uv = (0..<rasterOutput.pixelCount).map { pixel in
    SIMD2<Float>(
        uvOutput.attributes[pixel * 2 + 0],
        uvOutput.attributes[pixel * 2 + 1]
    )
}

let sampled = try rasterizer.texture(
    texture: textureData,
    texWidth: texWidth,
    texHeight: texHeight,
    channels: channels,
    uv: uv,
    outWidth: rasterOutput.width,
    outHeight: rasterOutput.height
)
```

### How do I handle depth testing?

Depth testing is built into the rasterization:

```swift
let output = try rasterizer.rasterize(...)

// output.depthBuffer contains per-pixel depth values
// Larger z/w values win the built-in depth test
// Triangles are automatically depth-sorted during rasterization
```

### Can I customize the depth range?

The default depth range is [0, 1]. To change it, apply a transformation to your Z coordinates before passing them to the rasterizer:

```swift
// Transform from [-1, 1] to [0, 1]
let transformedZ = (originalZ + 1) * 0.5
```

### How do I implement transparency?

MTLDiffRast doesn't directly support transparency in the rasterizer. Implement it in post-processing:

```swift
// 1. Rasterize with depth
let output = try rasterizer.rasterize(...)

// 2. Interpolate alpha channel
let alphaOutput = try rasterizer.interpolate(
    attributes: alphas,
    triangles: triangles,
    rasterOutput: output,
    numAttributes: 1
)

// 3. Alpha composite in Swift
for pixelIdx in 0..<output.pixelCount {
    let alpha = alphaOutput.attributes[pixelIdx]
    finalColor = alpha * foregroundColor + (1 - alpha) * backgroundColor
}
```

---

## Performance

### What resolution should I use?

Depends on your use case:

| Use Case | Recommended Resolution | Reason |
|----------|----------------------|--------|
| ML Training | 128×128 to 256×256 | Faster iteration |
| Preview/Debug | 256×256 to 512×512 | Good balance |
| Final Render | 1024×1024+ | Quality |

Remember: doubling resolution quadruples computation time.

### How many triangles can I render?

Performance varies by device:

| Device | Comfortable Limit | Maximum |
|--------|------------------|---------|
| M1/M2 Base | 5,000 | ~20,000 |
| M1/M2 Pro | 20,000 | ~100,000 |
| M1/M2 Max | 50,000 | ~200,000 |

For larger scenes, implement level-of-detail (LOD) and culling.

### Why is my rasterization slow?

Common causes:

1. **Too many triangles**: Reduce count or implement LOD
2. **High resolution**: Lower resolution for training
3. **Frequent re-initialization**: Reuse `Rasterizer` instances
4. **CPU-GPU transfers**: Minimize data copying
5. **Synchronization**: Avoid waiting for completion every frame

See <doc:Performance> for optimization strategies.

### Does MTLDiffRast support multi-threading?

Yes, but with caveats:
- The `Rasterizer` class is thread-safe for concurrent calls
- Operations are queued internally
- For best performance, batch work rather than parallelizing small tasks

---

## Troubleshooting

### "Metal is not available on this device"

**Cause:** Running on unsupported hardware or simulator.

**Solution:**
- Ensure you're running on Apple Silicon (M1/M2/M3)
- Run on actual device, not simulator
- Check `isMetalAvailable()` before initializing

### "Could not load shader library"

**Cause:** Resources not properly bundled.

**Solution:**
1. Verify `Resources` folder is included in target
2. Check `MetalShaders.metal` is in Build Phases → Copy Bundle Resources
3. Clean build folder (Cmd+Shift+K) and rebuild

### "Invalid triangle count"

**Cause:** Empty or invalid triangle array.

**Solution:**
```swift
// Ensure triangles array is not empty
guard !triangles.isEmpty else {
    print("Error: Need at least one triangle")
    return
}
```

### My triangle isn't visible

**Possible causes:**

1. **Wrong winding order**: Try reversing vertex order
   ```swift
   // Change from clockwise to counter-clockwise
   SIMD3<Int32>(0, 2, 1)  // instead of (0, 1, 2)
   ```

2. **Outside view frustum**: Check vertex coordinates are in [-1, 1]

3. **Backface culling**: Both sides should be visible by default

4. **Depth conflict**: Adjust Z coordinates

### Gradients are all zeros

**Cause:** No pixels covered by triangles, or disconnected computation graph.

**Solution:**
- Verify triangles cover some pixels
- Check `gradOutput` has non-zero values
- Ensure forward pass completed successfully

---

## Development

### How do I contribute?

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

See <doc:Contributing> for details.

### How do I report bugs?

Open an issue on GitHub with:
- Minimal reproducible example
- Expected vs actual behavior
- System information (macOS version, chip type)
- MTLDiffRast version

### Can I add custom shaders?

Yes! Extend the Metal shader file:

1. Add kernel function to `MetalShaders.metal`
2. Create pipeline state in `Rasterizer.swift`
3. Add public API method

Example:
```metal
kernel void myCustomKernel(...) {
    // Your implementation
}
```

### Is there a changelog?

See `RELEASES.md` for version history and changes.

---

## Still Need Help?

If your question isn't answered here:

1. Check the <doc:APIReference>
2. Review <doc:Examples>
3. Read <doc:Troubleshooting>
4. Open a GitHub Discussion
5. Contact maintainers
