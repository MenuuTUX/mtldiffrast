# Examples and Tutorials

Practical examples demonstrating MTLDiffRast usage patterns, from basic rasterization to advanced differentiable rendering.

## Table of Contents

1. [Basic Rasterization](#basic-rasterization)
2. [Attribute Interpolation](#attribute-interpolation)
3. [Gradient Computation](#gradient-computation)
4. [Multiple Triangles](#multiple-triangles)
5. [Animated Mesh](#animated-mesh)
6. [ML Training Loop](#ml-training-loop)
7. [Texture Mapping](#texture-mapping)

---

## Basic Rasterization

Render a single triangle to a depth buffer.

```swift
import MTLDiffRast
import simd

func renderSingleTriangle() throws {
    // Initialize rasterizer
    let rasterizer = try Rasterizer()
    
    // Define vertices in clip space (-1 to 1)
    let positions: [SIMD4<Float>] = [
        SIMD4<Float>(0.0, 0.5, 0.5, 1.0),   // Top
        SIMD4<Float>(-0.5, -0.5, 0.5, 1.0), // Bottom left
        SIMD4<Float>(0.5, -0.5, 0.5, 1.0)   // Bottom right
    ]
    
    // Define one triangle using vertex indices
    let triangles: [SIMD3<Int32>] = [
        SIMD3<Int32>(0, 1, 2)
    ]
    
    // Rasterize at 512x512 resolution
    let output = try rasterizer.rasterize(
        positions: positions,
        triangles: triangles,
        width: 512,
        height: 512
    )
    
    // Analyze results
    let coveredPixels = output.triangleIds.filter { $0 >= 0 }.count
    print("Covered pixels: \(coveredPixels) / \(output.pixelCount)")
    print("Coverage: \(Double(coveredPixels) / Double(output.pixelCount) * 100)%")
}
```

---

## Attribute Interpolation

Interpolate colors across a triangle (Gouraud shading).

```swift
func interpolateColors() throws {
    let rasterizer = try Rasterizer()
    
    // Triangle vertices
    let positions: [SIMD4<Float>] = [
        SIMD4<Float>(0.0, 0.5, 0.5, 1.0),
        SIMD4<Float>(-0.5, -0.5, 0.5, 1.0),
        SIMD4<Float>(0.5, -0.5, 0.5, 1.0)
    ]
    
    let triangles: [SIMD3<Int32>] = [SIMD3<Int32>(0, 1, 2)]
    
    // Per-vertex RGB colors
    let colors: [Float] = [
        1.0, 0.0, 0.0,  // Vertex 0: Red
        0.0, 1.0, 0.0,  // Vertex 1: Green
        0.0, 0.0, 1.0   // Vertex 2: Blue
    ]
    
    // Forward pass
    let rasterOutput = try rasterizer.rasterize(
        positions: positions,
        triangles: triangles,
        width: 512,
        height: 512
    )
    
    // Interpolate colors
    let interpOutput = try rasterizer.interpolate(
        attributes: colors,
        triangles: triangles,
        rasterOutput: rasterOutput,
        numAttributes: 3  // RGB
    )
    
    // Access interpolated colors for each pixel
    for pixelIdx in 0..<interpOutput.pixelCount {
        let r = interpOutput.attributes[pixelIdx * 3 + 0]
        let g = interpOutput.attributes[pixelIdx * 3 + 1]
        let b = interpOutput.attributes[pixelIdx * 3 + 2]
        
        if rasterOutput.triangleIds[pixelIdx] >= 0 {
            print("Pixel \(pixelIdx): RGB(\(r), \(g), \(b))")
        }
    }
}
```

---

## Gradient Computation

Compute gradients for optimization (differentiable rendering).

```swift
func computeGradients() throws {
    let rasterizer = try Rasterizer()
    
    // Initial triangle
    var positions: [SIMD4<Float>] = [
        SIMD4<Float>(0.0, 0.5, 0.5, 1.0),
        SIMD4<Float>(-0.5, -0.5, 0.5, 1.0),
        SIMD4<Float>(0.5, -0.5, 0.5, 1.0)
    ]
    
    let triangles: [SIMD3<Int32>] = [SIMD3<Int32>(0, 1, 2)]
    
    // Forward pass
    let output = try rasterizer.rasterize(
        positions: positions,
        triangles: triangles,
        width: 256,
        height: 256
    )
    
    // Define a simple differentiable loss on the first barycentric component.
    let loss = output.barycentrics.reduce(Float(0)) { sum, bary in
        sum + bary.x
    }
    print("Loss: \(loss)")

    // d(loss)/d(u) = 1 for each pixel. A scalar-per-pixel gradient drives
    // the u component in rasterizeBackward.
    let gradOutput = Array(repeating: Float(1), count: output.pixelCount)
    
    // Backward pass
    let gradients = try rasterizer.rasterizeBackward(
        positions: positions,
        triangles: triangles,
        forwardOutput: output,
        gradOutput: gradOutput,
        vertexCount: positions.count
    )
    
    // Access gradients
    print("Position gradients:")
    for i in 0..<positions.count {
        let gx = gradients.positionGradients[i * 4 + 0]
        let gy = gradients.positionGradients[i * 4 + 1]
        let gz = gradients.positionGradients[i * 4 + 2]
        let gw = gradients.positionGradients[i * 4 + 3]
        print("  Vertex \(i): (\(gx), \(gy), \(gz), \(gw))")
    }
    
    // Simple gradient descent update
    let learningRate: Float = 0.01
    for i in 0..<positions.count {
        positions[i].x -= learningRate * gradients.positionGradients[i * 4 + 0]
        positions[i].y -= learningRate * gradients.positionGradients[i * 4 + 1]
    }
}
```

---

## Multiple Triangles

Render a mesh with multiple triangles.

```swift
func renderMesh() throws {
    let rasterizer = try Rasterizer()
    
    // Define a simple quad (two triangles)
    let positions: [SIMD4<Float>] = [
        SIMD4<Float>(-0.5, 0.5, 0.5, 1.0),   // 0: Top left
        SIMD4<Float>(0.5, 0.5, 0.5, 1.0),    // 1: Top right
        SIMD4<Float>(-0.5, -0.5, 0.5, 1.0),  // 2: Bottom left
        SIMD4<Float>(0.5, -0.5, 0.5, 1.0)    // 3: Bottom right
    ]
    
    // Two triangles forming a quad
    let triangles: [SIMD3<Int32>] = [
        SIMD3<Int32>(0, 1, 2),  // Top-left triangle
        SIMD3<Int32>(1, 3, 2)   // Bottom-right triangle
    ]
    
    let output = try rasterizer.rasterize(
        positions: positions,
        triangles: triangles,
        width: 512,
        height: 512
    )
    
    // Count unique triangles visible
    let visibleTriangles = Set(output.triangleIds.filter { $0 >= 0 })
    print("Visible triangles: \(visibleTriangles.count)")
}
```

---

## Animated Mesh

Animate vertex positions over time.

```swift
class AnimatedRenderer {
    private let rasterizer: Rasterizer
    private var basePositions: [SIMD4<Float>]
    private let triangles: [SIMD3<Int32>]
    private var time: Float = 0
    
    init() throws {
        self.rasterizer = try Rasterizer()
        
        // Base triangle
        self.basePositions = [
            SIMD4<Float>(0.0, 0.5, 0.5, 1.0),
            SIMD4<Float>(-0.5, -0.5, 0.5, 1.0),
            SIMD4<Float>(0.5, -0.5, 0.5, 1.0)
        ]
        
        self.triangles = [SIMD3<Int32>(0, 1, 2)]
    }
    
    func animate(deltaTime: Float) throws -> RasterOutput {
        time += deltaTime
        
        // Apply sine wave animation to vertices
        var animatedPositions = basePositions
        for i in 0..<animatedPositions.count {
            animatedPositions[i].y += sin(time + Float(i)) * 0.1
        }
        
        return try rasterizer.rasterize(
            positions: animatedPositions,
            triangles: triangles,
            width: 512,
            height: 512
        )
    }
    
    func renderFrame(duration: Float, frameRate: Int) throws {
        let frameTime = 1.0 / Float(frameRate)
        var currentTime: Float = 0
        
        while currentTime < duration {
            let output = try animate(deltaTime: frameTime)
            
            // Process frame (e.g., display or save)
            processFrame(output)
            
            currentTime += frameTime
        }
    }
    
    private func processFrame(_ output: RasterOutput) {
        // Your frame processing logic here
        print("Frame rendered: \(output.triangleIds.filter { $0 >= 0 }.count) pixels covered")
    }
}

// Usage
let renderer = try AnimatedRenderer()
try renderer.renderFrame(duration: 5.0, frameRate: 60)
```

---

## ML Training Loop

Example training loop for machine learning applications.

```swift
class DifferentiableRenderer {
    private let rasterizer: Rasterizer
    private let learningRate: Float
    
    init(learningRate: Float = 0.01) throws {
        self.rasterizer = try Rasterizer()
        self.learningRate = learningRate
    }
    
    func trainStep(
        positions: inout [SIMD4<Float>],
        triangles: [SIMD3<Int32>],
        targetImage: [Float]
    ) throws -> Float {
        // Forward pass
        let output = try rasterizer.rasterize(
            positions: positions,
            triangles: triangles,
            width: 256,
            height: 256
        )
        
        // Compute loss (MSE against target)
        var loss: Float = 0
        var gradOutput: [Float] = []
        
        for i in 0..<output.pixelCount {
            let predicted = output.triangleIds[i] >= 0 ? 1.0 : 0.0
            let target = targetImage[i]
            let diff = predicted - target
            loss += diff * diff
            gradOutput.append(2 * diff)
        }
        
        loss /= Float(output.pixelCount)
        
        // Backward pass
        let gradients = try rasterizer.rasterizeBackward(
            positions: positions,
            triangles: triangles,
            forwardOutput: output,
            gradOutput: gradOutput,
            vertexCount: positions.count
        )
        
        // Update positions
        for i in 0..<positions.count {
            positions[i].x -= learningRate * gradients.positionGradients[i * 4 + 0]
            positions[i].y -= learningRate * gradients.positionGradients[i * 4 + 1]
            positions[i].z -= learningRate * gradients.positionGradients[i * 4 + 2]
            positions[i].w -= learningRate * gradients.positionGradients[i * 4 + 3]
        }
        
        return loss
    }
    
    func train(
        initialPositions: inout [SIMD4<Float>],
        triangles: [SIMD3<Int32>],
        targetImage: [Float],
        epochs: Int
    ) throws -> [Float] {
        var losses: [Float] = []
        
        for epoch in 0..<epochs {
            let loss = try trainStep(
                positions: &initialPositions,
                triangles: triangles,
                targetImage: targetImage
            )
            losses.append(loss)
            
            if epoch % 10 == 0 {
                print("Epoch \(epoch): Loss = \(loss)")
            }
        }
        
        return losses
    }
}

// Usage example
func runTraining() throws {
    let trainer = try DifferentiableRenderer(learningRate: 0.001)
    
    var positions: [SIMD4<Float>] = [
        SIMD4<Float>(0.0, 0.5, 0.5, 1.0),
        SIMD4<Float>(-0.5, -0.5, 0.5, 1.0),
        SIMD4<Float>(0.5, -0.5, 0.5, 1.0)
    ]
    
    let triangles: [SIMD3<Int32>] = [SIMD3<Int32>(0, 1, 2)]
    
    // Create target image (simple circle pattern)
    let targetImage: [Float] = (0..<256*256).map { i in
        let x = Float(i % 256) - 128
        let y = Float(i / 256) - 128
        return sqrt(x*x + y*y) < 64 ? 1.0 : 0.0
    }
    
    let losses = try trainer.train(
        initialPositions: &positions,
        triangles: triangles,
        targetImage: targetImage,
        epochs: 100
    )
}
```

---

## Texture Mapping

Sample textures using interpolated UV coordinates.

```swift
func textureMapping() throws {
    let rasterizer = try Rasterizer()
    
    // Quad vertices
    let positions: [SIMD4<Float>] = [
        SIMD4<Float>(-0.5, 0.5, 0.5, 1.0),
        SIMD4<Float>(0.5, 0.5, 0.5, 1.0),
        SIMD4<Float>(-0.5, -0.5, 0.5, 1.0),
        SIMD4<Float>(0.5, -0.5, 0.5, 1.0)
    ]
    
    let triangles: [SIMD3<Int32>] = [
        SIMD3<Int32>(0, 1, 2),
        SIMD3<Int32>(1, 3, 2)
    ]
    
    // UV coordinates per vertex
    let uvs: [Float] = [
        0.0, 0.0,  // Vertex 0
        1.0, 0.0,  // Vertex 1
        0.0, 1.0,  // Vertex 2
        1.0, 1.0   // Vertex 3
    ]
    
    // Rasterize
    let rasterOutput = try rasterizer.rasterize(
        positions: positions,
        triangles: triangles,
        width: 512,
        height: 512
    )
    
    // Interpolate UVs
    let interpOutput = try rasterizer.interpolate(
        attributes: uvs,
        triangles: triangles,
        rasterOutput: rasterOutput,
        numAttributes: 2
    )
    
    let uv = (0..<rasterOutput.pixelCount).map { pixel in
        SIMD2<Float>(
            interpOutput.attributes[pixel * 2 + 0],
            interpOutput.attributes[pixel * 2 + 1]
        )
    }

    let checkerTexture: [Float] = [
        1, 1, 1,   0, 0, 0,
        0, 0, 0,   1, 1, 1
    ]

    let textured = try rasterizer.texture(
        texture: checkerTexture,
        texWidth: 2,
        texHeight: 2,
        channels: 3,
        uv: uv,
        outWidth: rasterOutput.width,
        outHeight: rasterOutput.height,
        filterMode: .nearest,
        boundaryMode: .wrap
    )

    print("Sampled \(textured.samples.count) texture values")
}
```

---

## Antialiasing Example

Apply antialiasing to reduce jagged edges.

```swift
func renderWithAntialiasing() throws {
    let rasterizer = try Rasterizer()
    
    let positions: [SIMD4<Float>] = [
        SIMD4<Float>(0.0, 0.5, 0.5, 1.0),
        SIMD4<Float>(-0.5, -0.5, 0.5, 1.0),
        SIMD4<Float>(0.5, -0.5, 0.5, 1.0)
    ]
    
    let triangles: [SIMD3<Int32>] = [SIMD3<Int32>(0, 1, 2)]
    
    // Standard rasterization
    let output = try rasterizer.rasterize(
        positions: positions,
        triangles: triangles,
        width: 512,
        height: 512
    )
    
    // Apply antialiasing
    let aaOutput = try rasterizer.antialias(rasterOutput: output)
    
    // Compare edge quality
    print("Original edges: \(countEdgePixels(output))")
    print("Antialiased edges: \(countEdgePixels(aaOutput))")
}

func countEdgePixels(_ output: RasterOutput) -> Int {
    var edgeCount = 0
    for y in 0..<output.height {
        for x in 0..<output.width {
            let idx = y * output.width + x
            let currentId = output.triangleIds[idx]
            
            // Check neighbors
            if x > 0 && output.triangleIds[idx - 1] != currentId {
                edgeCount += 1
            }
            if y > 0 && output.triangleIds[idx - output.width] != currentId {
                edgeCount += 1
            }
        }
    }
    return edgeCount
}
```

---

## See Also

- <doc:GettingStarted> - Installation and setup
- <doc:APIReference> - Complete API documentation
- <doc:Performance> - Optimization tips
