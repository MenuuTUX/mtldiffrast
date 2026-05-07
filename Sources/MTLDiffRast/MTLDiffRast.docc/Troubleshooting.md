# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with MTLDiffRast.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Runtime Errors](#runtime-errors)
3. [Rendering Problems](#rendering-problems)
4. [Performance Issues](#performance-issues)
5. [Gradient Computation Issues](#gradient-computation-issues)
6. [Build Errors](#build-errors)

---

## Installation Issues

### Package Resolution Fails

**Error:** `error: Dependencies could not be resolved`

**Solutions:**

1. Clean package cache:
   ```bash
   rm -rf .build
   rm -rf ~/Library/Caches/org.swift.swiftpm/repositories
   ```

2. Update package dependencies:
   ```bash
   swift package update
   ```

3. Check network connectivity and GitHub access

4. Verify Swift version:
   ```bash
   swift --version  # Should be 5.9+
   ```

### Module Not Found

**Error:** `no such module 'MTLDiffRast'`

**Solutions:**

1. Ensure MTLDiffRast is added to target dependencies in Package.swift:
   ```swift
   .executableTarget(
       name: "YourApp",
       dependencies: [
           .product(name: "MTLDiffRast", package: "mtldiffrast")
       ]
   )
   ```

2. Clean build folder:
   ```bash
   swift package clean
   ```

3. Rebuild project in Xcode (Cmd+Shift+K, then Cmd+B)

---

## Runtime Errors

### Metal Unavailable

**Error:** `RasterizerError.metalUnavailable`

**Diagnosis:**
```swift
print("Metal available: \(isMetalAvailable())")
print("Apple Silicon: \(isAppleSilicon())")
print("Device info: \(getMetalDeviceInfo()?.name ?? "none")")
```

**Solutions:**

1. **Running on Intel Mac**: MTLDiffRast requires Apple Silicon
2. **Running in simulator**: Use physical device
3. **Metal framework missing**: Ensure Metal.framework is linked

### Shader Library Not Found

**Error:** `Could not load shader library`

**Diagnosis:**
```swift
do {
    let rasterizer = try Rasterizer()
} catch {
    print("Error: \(error)")
    // Check Bundle.module resources
    print("Bundle resources: \(Bundle.module.resources)")
}
```

**Solutions:**

1. Verify `Resources` folder exists in source directory
2. Check `MetalShaders.metal` is present
3. In Xcode:
   - Select project → Target → Build Phases
   - Ensure `Resources` folder is in "Copy Bundle Resources"
4. Clean and rebuild

### Buffer Creation Failed

**Error:** `Failed to create Metal buffer`

**Causes:**
- Insufficient memory
- Invalid buffer size
- Device lost

**Solutions:**

1. Check available memory:
   ```swift
   if let info = getMetalDeviceInfo() {
       print("Max threads per group: \(info.maxThreadsPerThreadgroup)")
   }
   ```

2. Reduce resolution or triangle count

3. Close other GPU-intensive applications

4. Restart application to reset Metal device

---

## Rendering Problems

### Nothing Renders (Black Screen)

**Checklist:**

1. **Vertex coordinates**: Ensure vertices are in clip space [-1, 1]
   ```swift
   // Correct
   SIMD4<Float>(0.0, 0.5, 0.5, 1.0)
   
   // Wrong - outside clip space
   SIMD4<Float>(2.0, 2.0, 2.0, 1.0)
   ```

2. **Triangle winding**: Try reversing vertex order
   ```swift
   // If (0, 1, 2) doesn't work, try (0, 2, 1)
   SIMD3<Int32>(0, 2, 1)
   ```

3. **Z coordinate**: Ensure Z is in valid range
   ```swift
   // Typical Z range
   SIMD4<Float>(x, y, 0.5, 1.0)  // Between near (0) and far (1)
   ```

4. **Check coverage**:
   ```swift
   let coveredPixels = output.triangleIds.filter { $0 >= 0 }.count
   print("Covered: \(coveredPixels) pixels")
   ```

### Triangle Appears Distorted

**Possible causes:**

1. **Perspective division issue**: Check W component
   ```swift
   // W should typically be 1.0 for orthographic
   SIMD4<Float>(x, y, z, 1.0)
   ```

2. **Incorrect screen transform**: Verify width/height ratio
   ```swift
   // Account for aspect ratio
   let aspect = Float(width) / Float(height)
   ```

3. **Vertex index errors**: Ensure indices reference valid vertices
   ```swift
   // If you have 4 vertices, max index is 3
   SIMD3<Int32>(0, 1, 3)  // Valid
   SIMD3<Int32>(0, 1, 4)  // Invalid!
   ```

### Flickering or Z-Fighting

**Cause:** Multiple triangles at similar depth values

**Solutions:**

1. Increase depth separation between objects

2. Use polygon offset:
   ```swift
   // Add small offset to Z based on triangle ID
   positions[i].z += Float(triangleId) * 0.0001
   ```

3. Sort triangles by depth before rendering

4. Implement depth bias in post-processing

### Jagged Edges (Aliasing)

**Solution:** Apply antialiasing

```swift
let output = try rasterizer.rasterize(...)
let aaOutput = try rasterizer.antialias(rasterOutput: output)
```

For better quality, render at higher resolution and downsample:

```swift
// Render at 2x resolution
let highRes = try rasterizer.rasterize(
    positions: positions,
    triangles: triangles,
    width: 1024,
    height: 1024
)

// Downsample to final resolution
let finalImage = downsample(highRes, to: 512)
```

---

## Performance Issues

### Slow Rasterization

**Diagnostic steps:**

1. Profile with Instruments:
   ```bash
   xcrun instruments -template "Metal System Trace" ./YourApp
   ```

2. Check triangle count:
   ```swift
   print("Triangles: \(triangles.count)")
   print("Pixels: \(width * height)")
   print("Complexity: \(triangles.count * width * height) operations")
   ```

3. Measure execution time:
   ```swift
   let start = CFAbsoluteTimeGetCurrent()
   let output = try rasterizer.rasterize(...)
   let elapsed = CFAbsoluteTimeGetCurrent() - start
   print("Time: \(elapsed * 1000) ms")
   ```

**Solutions:**

- Reduce resolution (quadratic speedup)
- Reduce triangle count
- Implement LOD (Level of Detail)
- Use frustum culling
- Reuse Rasterizer instance

### Memory Leaks

**Symptoms:**
- Memory usage grows over time
- App crashes with memory pressure

**Diagnosis:**
```swift
// Check for retained references
deinit {
    print("Rasterizer deallocated")
}
```

**Solutions:**

1. Ensure proper cleanup:
   ```swift
   class Renderer {
       private var rasterizer: Rasterizer?
       
       func cleanup() {
           rasterizer = nil
       }
   }
   ```

2. Use weak references where appropriate

3. Profile with Xcode Memory Graph

### High CPU Usage

**Cause:** Excessive CPU-GPU synchronization

**Solutions:**

1. Avoid `waitUntilCompleted()` in tight loops:
   ```swift
   // Bad: Waits for every frame
   for frame in frames {
       let output = try rasterizer.rasterize(...)
       commandBuffer.waitUntilCompleted()
   }
   
   // Better: Batch and wait once
   let outputs = try frames.map { frame in
       try rasterizer.rasterize(...)
   }
   commandBuffer.waitUntilCompleted()
   ```

2. Use async patterns:
   ```swift
   func renderAsync() async throws -> RasterOutput {
       return try await withCheckedThrowingContinuation { continuation in
           queue.async {
               do {
                   let output = try self.rasterizer.rasterize(...)
                   continuation.resume(returning: output)
               } catch {
                   continuation.resume(throwing: error)
               }
           }
       }
   }
   ```

---

## Gradient Computation Issues

### Gradients Are Zero

**Possible causes:**

1. No pixels covered by triangles
2. gradOutput is all zeros
3. Disconnected computation graph

**Debugging:**

```swift
// Check forward pass first
let output = try rasterizer.rasterize(...)
let covered = output.triangleIds.filter { $0 >= 0 }.count
print("Covered pixels: \(covered)")

// Check gradOutput
let nonZeroGrads = gradOutput.filter { $0 != 0 }.count
print("Non-zero gradients: \(nonZeroGrads)")

// Compute backward
let gradients = try rasterizer.rasterizeBackward(...)

// Check result
let nonZeroPositionGrads = gradients.positionGradients.filter { $0 != 0 }.count
print("Non-zero position gradients: \(nonZeroPositionGrads)")
```

### Gradients Explode (NaN or Inf)

**Cause:** Numerical instability

**Solutions:**

1. Clip gradient values:
   ```swift
   let clippedGradients = gradients.positionGradients.map {
       max(-1000, min(1000, $0))
   }
   ```

2. Use gradient clipping in optimizer:
   ```swift
   let gradNorm = sqrt(gradients.positionGradients.reduce(0) { $0 + $1 * $1 })
   if gradNorm > maxGradNorm {
       let scale = maxGradNorm / gradNorm
       // Scale gradients
   }
   ```

3. Check for degenerate triangles (zero area)

4. Add epsilon to divisions:
   ```metal
   // In shader
   float denom = d00 * d11 - d01 * d01 + EPSILON;
   ```

---

## Build Errors

### Swift Version Mismatch

**Error:** `built with newer version of Swift`

**Solution:** Update Xcode and Swift:
```bash
xcode-select --install
```

Or specify Swift version in Package.swift:
```swift
// swift-tools-version:5.9
```

### Architecture Mismatch

**Error:** `building for iOS Simulator, but linking in object file built for iOS`

**Solutions:**

1. Clean build folder completely:
   ```bash
   rm -rf .build
   rm -rf ~/Library/Developer/Xcode/DerivedData
   ```

2. In Xcode: Build → Clean Build Folder (Cmd+Shift+K)

3. Ensure building for correct architecture:
   ```bash
   arch -arm64 swift build  # Force ARM64
   ```

### Resource Bundling Issues

**Error:** `unable to lookup resource`

**Solutions:**

1. Verify Resources folder structure:
   ```
   Sources/
     MTLDiffRast/
       Resources/
         MetalShaders.metal
   ```

2. Check Package.swift:
   ```swift
   .target(
       name: "MTLDiffRast",
       resources: [
           .process("Resources")
       ]
   )
   ```

3. In Xcode, ensure Resources folder has yellow folder icon (group)

---

## Getting More Help

If these solutions don't resolve your issue:

1. **Check logs**: Enable verbose logging
   ```swift
   os_log("Debug info: %{public}@", log: OSLog.default, type: .debug, someInfo)
   ```

2. **Create minimal reproduction**: Isolate the issue in a small test case

3. **Gather system info**:
   ```swift
   print("macOS: \(ProcessInfo.processInfo.operatingSystemVersion)")
   print("Chip: \(isAppleSilicon() ? "Apple Silicon" : "Intel")")
   print("Metal: \(getMetalDeviceInfo()?.name ?? "unavailable")")
   ```

4. **Search existing issues**: Check GitHub Issues for similar problems

5. **Open new issue**: Include reproduction steps and system info
