# Metal Shaders Guide

This document provides a detailed explanation of the Metal compute shaders used in MTLDiffRast for GPU-accelerated rasterization.

## Overview

MTLDiffRast uses four main compute shader kernels written in Metal shading language:

1. **rasterizeKernel** - Forward pass triangle rasterization
2. **rasterizeBackwardKernel** - Backward pass gradient computation
3. **interpolateKernel** - Per-pixel attribute interpolation
4. **antialiasKernel** - Edge smoothing filter

All shaders are located in `Sources/MTLDiffRast/Resources/MetalShaders.metal`.

---

## 1. Rasterize Kernel

The forward pass kernel that determines which triangle covers each pixel.

### Function Signature

```metal
kernel void rasterizeKernel(
    device const float4* positions [[buffer(0)]],
    device const int3* triangles [[buffer(1)]],
    device RasterOutput* outputBuffer [[buffer(2)]],
    constant int& triangleCount [[buffer(3)]],
    constant int& width [[buffer(4)]],
    constant int& height [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
)
```

### Buffer Bindings

| Buffer Index | Type | Purpose |
|--------------|------|---------|
| 0 | `float4[]` | Vertex positions in clip space |
| 1 | `int3[]` | Triangle vertex indices |
| 2 | `RasterOutput[]` | Output buffer (triangle ID + depth) |
| 3 | `int` | Number of triangles |
| 4 | `int` | Output width |
| 5 | `int` | Output height |

### Algorithm

```
For each pixel (x, y) in parallel:
    1. Initialize bestTriangleId = -1, bestDepth = MAX_FLOAT
    2. For each triangle t:
        a. Fetch vertices v0, v1, v2
        b. Apply perspective division: ndc = xyz / w
        c. Transform to screen space:
           screen.x = (ndc.x + 1) * 0.5 * width
           screen.y = (1 - ndc.y) * 0.5 * height
        d. Compute edge functions:
           e0 = edge(v0, v1, pixel)
           e1 = edge(v1, v2, pixel)
           e2 = edge(v2, v0, pixel)
        e. If all edges have same sign (pixel inside triangle):
           - Compute barycentric coordinates (u, v, w)
           - Interpolate depth: z = u*z0 + v*z1 + w*z2
           - If z < bestDepth (depth test passes):
             * Update bestDepth = z
             * Update bestTriangleId = t
    3. Write (bestTriangleId, bestDepth) to output
```

### Edge Function

The edge function determines which side of an edge a point lies on:

```metal
float edgeFunction(float2 a, float2 b, float2 p) {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}
```

This is equivalent to the 2D cross product and represents the signed area of the parallelogram formed by vectors (b-a) and (p-a).

### Barycentric Coordinates

```metal
float3 computeBarycentric(float2 v0, float2 v1, float2 v2, float2 p) {
    float2 v0v1 = v1 - v0;
    float2 v0v2 = v2 - v0;
    float2 v0p = p - v0;
    
    float d00 = dot(v0v1, v0v1);
    float d01 = dot(v0v1, v0v2);
    float d11 = dot(v0v2, v0v2);
    float d20 = dot(v0p, v0v1);
    float d21 = dot(v0p, v0v2);
    
    float denom = d00 * d11 - d01 * d01;
    if (abs(denom) < EPSILON) {
        return float3(0.0f, 0.0f, 0.0f);
    }
    
    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    float u = 1.0f - v - w;
    
    return float3(u, v, w);
}
```

### Thread Configuration

```swift
let threadsPerGrid = MTLSize(width: width, height: height, depth: 1)
let threadGroupSize = MTLSize(width: min(16, width), height: min(16, height), depth: 1)
```

Each thread processes one pixel independently.

---

## 2. Rasterize Backward Kernel

Computes gradients with respect to vertex positions for differentiable rendering.

### Function Signature

```metal
kernel void rasterizeBackwardKernel(
    device const float4* positions [[buffer(0)]],
    device const int3* triangles [[buffer(1)]],
    device const RasterOutput* forwardOutput [[buffer(2)]],
    device const float* gradOutput [[buffer(3)]],
    device float* gradPositions [[buffer(4)]],
    constant int& triangleCount [[buffer(5)]],
    constant int& vertexCount [[buffer(6)]],
    constant int& width [[buffer(7)]],
    constant int& height [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
)
```

### Buffer Bindings

| Buffer Index | Type | Purpose |
|--------------|------|---------|
| 0 | `float4[]` | Original vertex positions |
| 1 | `int3[]` | Triangle indices |
| 2 | `RasterOutput[]` | Forward pass output |
| 3 | `float[]` | Downstream gradients (∂L/∂output) |
| 4 | `float[]` | Output position gradients (∂L/∂position) |
| 5 | `int` | Triangle count |
| 6 | `int` | Vertex count |
| 7 | `int` | Width |
| 8 | `int` | Height |

### Gradient Computation

For each pixel covered by a triangle:

```
Given:
    - Pixel position (x, y)
    - Covering triangle t with vertices v0, v1, v2
    - Downstream gradient g = gradOutput[pixel]
    - Barycentric coordinates (u, v, w)

Compute:
    ∂L/∂v0 = g * ∂(barycentric_interp)/∂v0
    ∂L/∂v1 = g * ∂(barycentric_interp)/∂v1
    ∂L/∂v2 = g * ∂(barycentric_interp)/∂v2

Where barycentric_interp = u*attr0 + v*attr1 + w*attr2
```

### Chain Rule Application

The gradient flows through the rasterization operation:

```
∂L/∂position = ∂L/∂output × ∂output/∂position

For depth interpolation:
    depth = u*z0 + v*z1 + w*z2
    
∂depth/∂v0 = (∂u/∂v0)*z0 + u*(∂z0/∂v0) + ...
```

### Atomic Operations Note

The current implementation includes a threadgroup barrier but may need atomic operations for proper gradient accumulation when multiple pixels affect the same vertex:

```metal
// Current simplified version
threadgroup_barrier(mem_flags::mem_threadgroup);

// For production use, consider:
atomic_fetch_add_explicit(&gradPositions[vertexIdx], gradient, memory_order_relaxed);
```

---

## 3. Interpolate Kernel

Interpolates per-vertex attributes across rasterized triangles.

### Function Signature

```metal
kernel void interpolateKernel(
    device const float* attributes [[buffer(0)]],
    device const int3* triangles [[buffer(1)]],
    device const RasterOutput* rasterOutput [[buffer(2)]],
    device float* outputAttributes [[buffer(3)]],
    device float* barycentricCoords [[buffer(4)]],
    constant int& triangleCount [[buffer(5)]],
    constant int& numAttributes [[buffer(6)]],
    constant int& width [[buffer(7)]],
    constant int& height [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
)
```

### Buffer Bindings

| Buffer Index | Type | Purpose |
|--------------|------|---------|
| 0 | `float[]` | Per-vertex attributes |
| 1 | `int3[]` | Triangle indices |
| 2 | `RasterOutput[]` | Rasterization output |
| 3 | `float[]` | Output interpolated attributes |
| 4 | `float[]` | Output barycentric coordinates |
| 5 | `int` | Triangle count |
| 6 | `int` | Attributes per vertex |
| 7 | `int` | Width |
| 8 | `int` | Height |

### Interpolation Formula

For each pixel covered by triangle t:

```metal
// Get triangle vertex indices
int3 tri = triangles[triangleId];

// Compute barycentric coordinates (u, v, w)
float3 bary = computeBarycentric(screen0, screen1, screen2, pixelPos);

// Interpolate each attribute
for (int a = 0; a < numAttributes; a++) {
    float attr0 = attributes[tri.x * numAttributes + a];
    float attr1 = attributes[tri.y * numAttributes + a];
    float attr2 = attributes[tri.z * numAttributes + a];
    
    outputAttributes[pixelIndex * numAttributes + a] = 
        bary.x * attr0 + bary.y * attr1 + bary.z * attr2;
}
```

### Use Cases

| Attribute Type | numAttributes | Example |
|----------------|---------------|---------|
| Color (RGB) | 3 | Vertex colors |
| Color (RGBA) | 4 | Colors with alpha |
| UV Coordinates | 2 | Texture mapping |
| Normals | 3 | Surface normals |
| Custom | N | Any per-vertex data |

---

## 4. Antialias Kernel

Applies a spatial filter to reduce aliasing artifacts at triangle edges.

### Function Signature

```metal
kernel void antialiasKernel(
    device const RasterOutput* inputBuffer [[buffer(0)]],
    device RasterOutput* outputBuffer [[buffer(1)]],
    constant int& width [[buffer(2)]],
    constant int& height [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
)
```

### Algorithm

3×3 box filter averaging:

```metal
int sumTriangles = 0;
float sumDepth = 0.0f;
int count = 0;

// Sample 3x3 neighborhood
for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
        int nx = x + dx;
        int ny = y + dy;
        
        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            int nIndex = ny * width + nx;
            if (inputBuffer[nIndex].triangleId >= 0) {
                sumTriangles += inputBuffer[nIndex].triangleId;
                sumDepth += inputBuffer[nIndex].depth;
                count++;
            }
        }
    }
}

if (count > 0) {
    outputBuffer[pixelIndex].triangleId = sumTriangles / count;
    outputBuffer[pixelIndex].depth = sumDepth / count;
}
```

### Filter Characteristics

- **Kernel Size:** 3×3 pixels
- **Filter Type:** Box filter (uniform weights)
- **Boundary Handling:** Clamps to valid pixels
- **Performance:** 9 samples per pixel

### Alternative Filters

For better quality, consider implementing:

```metal
// Gaussian weights for smoother filtering
constant float gaussianWeights[9] = {
    1.0/16, 2.0/16, 1.0/16,
    2.0/16, 4.0/16, 2.0/16,
    1.0/16, 2.0/16, 1.0/16
};
```

---

## Constants and Definitions

### Global Constants

```metal
constant int MAX_TRIANGLES = 1048576;  // 2^20 triangles max
constant float EPSILON = 1e-8f;         // Numerical stability threshold
```

### Structures

```metal
struct Vertex {
    float4 position [[attribute(0)]]
};

struct Triangle {
    int3 indices [[attribute(0)]]
};

struct RasterOutput {
    int triangleId;
    float depth;
};
```

---

## Performance Optimizations

### Current Implementation

1. **Parallel Processing:** One thread per pixel
2. **Shared Memory:** `.storageModeShared` for CPU-GPU communication
3. **Early Exit:** Skip pixels outside triangle bounds quickly

### Potential Improvements

1. **Tile-Based Rendering:** Process tiles to improve cache coherence
2. **Hierarchical Z-Buffer:** Early depth testing optimization
3. **Triangle Sorting:** Sort triangles by depth for better early-out
4. **SIMD Optimization:** Leverage Metal SIMD groups
5. **Register Blocking:** Reduce register pressure for better occupancy

### Memory Access Patterns

Optimal access patterns for coalesced memory transactions:

```
✓ Sequential pixel access within threadgroup
✗ Random triangle access (could be improved with sorting)
✓ Contiguous attribute arrays
```

---

## Debugging Tips

### Common Issues

1. **Incorrect Triangle Winding:**
   - Ensure consistent clockwise or counter-clockwise winding
   - Check edge function sign convention

2. **Perspective Division Errors:**
   - Verify w component is non-zero
   - Handle w < 0 cases (behind camera)

3. **Depth Precision:**
   - Consider using reversed Z-buffer for better precision
   - Use appropriate near/far plane distances

### Validation

```metal
// Add debug output to verify intermediate values
if (gid.x == 0 && gid.y == 0) {
    printf("Pixel (0,0): triangle=%d, depth=%f\n", 
           outputBuffer[0].triangleId, 
           outputBuffer[0].depth);
}
```

---

## See Also

- <doc:Architecture> - System design
- <doc:APIReference> - Swift API documentation
- <doc:Performance> - Optimization strategies
