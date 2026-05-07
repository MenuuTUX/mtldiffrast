# Performance Best Practices

This guide covers performance optimization strategies for MTLDiffRast to help you get the most out of Apple Silicon GPUs.

## Understanding Performance Characteristics

### Computational Complexity

| Operation | Time Complexity | Bottleneck |
|-----------|-----------------|------------|
| Forward Pass | O(pixels × triangles) | Triangle iteration per pixel |
| Backward Pass | O(pixels) | Gradient accumulation |
| Interpolation | O(pixels × attributes) | Attribute fetch & interpolation |
| Antialiasing | O(pixels × 9) | Neighborhood sampling |

### Memory Bandwidth

Apple Silicon unified memory architecture provides:
- **M1/M2**: ~100 GB/s (base) to ~400 GB/s (Max/Ultra)
- **M3**: ~150 GB/s (base) to ~400 GB/s (Max)

Optimize memory access patterns to maximize bandwidth utilization.

---

## Optimization Strategies

### 1. Batch Processing

Process multiple frames or objects together to amortize overhead:

```swift
// ❌ Inefficient: Multiple small rasterizations
for frame in frames {
    let output = try rasterizer.rasterize(
        positions: frame.positions,
        triangles: frame.triangles,
        width: 512, height: 512
    )
}

// ✅ Efficient: Batch if possible
let batchedPositions = concatenateAllPositions(frames)
let batchedTriangles = concatenateAllTriangles(frames)
let output = try rasterizer.rasterize(
    positions: batchedPositions,
    triangles: batchedTriangles,
    width: 512, height: 512
)
```

### 2. Resolution Selection

Choose appropriate output resolution for your use case:

```swift
// For ML training, lower resolution often suffices
let trainingResolution = (256, 256)  // 4× faster than 512×512

// For final rendering, use target resolution
let renderResolution = (1024, 1024)
```

**Performance Impact:**

| Resolution | Relative Time | Memory |
|------------|---------------|--------|
| 256×256 | 1.0× | 256 KB |
| 512×512 | 4.0× | 1 MB |
| 1024×1024 | 16.0× | 4 MB |
| 2048×2048 | 64.0× | 16 MB |

### 3. Triangle Count Management

Reduce triangle count where possible:

```swift
// Use Level of Detail (LOD)
func selectLOD(distance: Float) -> [SIMD3<Int32>] {
    switch distance {
    case 0..<10: return highDetailMesh
    case 10..<50: return mediumDetailMesh
    default: return lowDetailMesh
    }
}

// Mesh simplification for distant objects
let simplified = simplifyMesh(original, targetCount: 1000)
```

### 4. Reuse Rasterizer Instance

Creating a `Rasterizer` has overhead. Reuse instances:

```swift
// ❌ Inefficient: Create new instance each time
func render() throws {
    let rasterizer = try Rasterizer()
    return try rasterizer.rasterize(...)
}

// ✅ Efficient: Singleton or long-lived instance
class Renderer {
    private let rasterizer: Rasterizer
    
    init() throws {
        self.rasterizer = try Rasterizer()
    }
    
    func render() throws -> RasterOutput {
        return try rasterizer.rasterize(...)
    }
}
```

### 5. Minimize CPU-GPU Transfers

Keep data on GPU when possible:

```swift
// ❌ Inefficient: Frequent buffer transfers
for i in 0..<100 {
    let output = try rasterizer.rasterize(positions: updatedPositions, ...)
    let result = processOnCPU(output)
    updatedPositions = computeNewPositions(result)
}

// ✅ Better: Batch operations, minimize round-trips
let outputs = try (0..<100).map { i in
    try rasterizer.rasterize(positions: positionBatches[i], ...)
}
let results = processBatchOnCPU(outputs)
```

### 6. Thread Group Optimization

Adjust thread group size based on resolution:

```swift
// Default works well for most cases
let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)

// For very small resolutions, reduce thread group
if width < 16 || height < 16 {
    threadGroupSize = MTLSize(width: min(8, width), height: min(8, height), depth: 1)
}
```

### 7. Memory Alignment

Align data to optimal boundaries:

```swift
// Ensure vertex count is multiple of 4 for SIMD efficiency
let alignedVertexCount = (vertexCount + 3) & ~3
var positions = [SIMD4<Float>](repeating: .zero, count: alignedVertexCount)
```

---

## Profiling and Benchmarking

### Using Instruments

1. Open Xcode → Product → Profile
2. Select "Metal System Trace" template
3. Record your application
4. Analyze GPU utilization and bottlenecks

### Manual Timing

```swift
import os.log

let logger = OSLog(subsystem: "com.yourapp", category: "MTLDiffRast")

func timedRasterize() throws -> RasterOutput {
    let start = CFAbsoluteTimeGetCurrent()
    
    let output = try rasterizer.rasterize(...)
    
    let elapsed = CFAbsoluteTimeGetCurrent() - start
    os_log("Rasterization took %{public}.3f ms", log: logger, type: .info, elapsed * 1000)
    
    return output
}
```

### Benchmark Template

```swift
func benchmark() throws {
    let rasterizer = try Rasterizer()
    let iterations = 100
    
    // Warm up
    _ = try rasterizer.rasterize(positions: positions, triangles: triangles, 
                                  width: 512, height: 512)
    
    // Benchmark
    let start = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iterations {
        _ = try rasterizer.rasterize(positions: positions, triangles: triangles,
                                      width: 512, height: 512)
    }
    let elapsed = CFAbsoluteTimeGetCurrent() - start
    
    print("Average: \((elapsed / Double(iterations)) * 1000) ms per frame")
    print("Throughput: \(Double(iterations) / elapsed) fps")
}
```

---

## Common Performance Issues

### Issue 1: Too Many Triangles

**Symptom:** Rasterization takes >10ms per frame

**Solution:**
```swift
// Implement frustum culling
func cullTriangles(positions: [SIMD4<Float>], triangles: [SIMD3<Int32>]) 
    -> [SIMD3<Int32>] {
    return triangles.filter { tri in
        let v0 = positions[Int(tri.x)]
        let v1 = positions[Int(tri.y)]
        let v2 = positions[Int(tri.z)]
        return isVisibleInFrustum(v0, v1, v2)
    }
}
```

### Issue 2: Excessive Memory Allocation

**Symptom:** Memory usage grows over time

**Solution:**
```swift
// Pre-allocate buffers when possible
class OptimizedRenderer {
    private var positionBuffer: MTLBuffer?
    private var triangleBuffer: MTLBuffer?
    
    func prepareBuffers(capacity: Int) {
        positionBuffer = device.makeBuffer(
            length: capacity * MemoryLayout<SIMD4<Float>>.stride,
            options: .storageModeShared
        )
    }
}
```

### Issue 3: Synchronization Overhead

**Symptom:** CPU waits for GPU completion

**Solution:**
```swift
// Use async command submission when possible
func renderAsync(completion: @escaping (RasterOutput) -> Void) {
    queue.async { [weak self] in
        guard let self = self else { return }
        do {
            let output = try self.rasterizer.rasterize(...)
            DispatchQueue.main.async {
                completion(output)
            }
        } catch {
            print("Error: \(error)")
        }
    }
}
```

---

## Device-Specific Optimizations

### M1/M2 Base Chips

- 8-core GPU (7-core on some M1)
- Optimal thread group: 16×16
- Avoid excessive triangle counts (>10K)

### M1/M2 Pro/Max/Ultra

- Up to 38-core GPU
- Higher memory bandwidth
- Can handle larger batches and resolutions

### M3 Series

- Hardware ray tracing support (future optimization)
- Dynamic caching
- Improved shader compilation

---

## Best Practices Checklist

- [ ] Reuse `Rasterizer` instances across frames
- [ ] Choose appropriate resolution for task
- [ ] Implement LOD for distant objects
- [ ] Batch operations when possible
- [ ] Profile with Instruments regularly
- [ ] Monitor memory usage
- [ ] Use frustum/occlusion culling
- [ ] Align data to SIMD boundaries
- [ ] Minimize CPU-GPU synchronization
- [ ] Consider using compute shaders for post-processing

---

## Performance Targets

| Device Class | Target FPS (512×512) | Max Triangles |
|--------------|----------------------|---------------|
| M1/M2 Base | 60+ | 5,000 |
| M1/M2 Pro | 120+ | 20,000 |
| M1/M2 Max | 120+ | 50,000 |
| M3 Series | 120+ | 50,000+ |

*Note: Actual performance depends on scene complexity and optimization level.*

---

## See Also

- <doc:Architecture> - Understanding internals
- <doc:MetalShaders> - Shader optimization
- <doc:Examples> - Efficient usage patterns
