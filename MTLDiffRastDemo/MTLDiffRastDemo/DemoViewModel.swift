//
//  DemoViewModel.swift
//  MTLDiffRastDemo
//
//  ViewModel managing all demo features
//

import Metal
import SwiftUI
import MTLDiffRast

/// Available demo features
enum DemoFeature: String, CaseIterable, Identifiable {
    case basicRasterization = "Basic Rasterization"
    case gradientFill = "Gradient Fill"
    case barycentricInterpolation = "Barycentric Interpolation"
    case antialiasing = "Antialiasing"
    case multipleTriangles = "Multiple Triangles"
    case depthTesting = "Depth Testing"
    case sampleTriangle = "Sample: Triangle"
    case sampleCube = "Sample: Cube"
    case sampleEarth = "Sample: Earth"
    case samplePose = "Sample: Pose"
    case sampleEnvPhong = "Sample: EnvPhong"
    case performanceBenchmark = "Performance Benchmark"
    
    var id: String { rawValue }

    var group: DemoFeatureGroup {
        switch self {
        case .basicRasterization, .gradientFill, .barycentricInterpolation, .antialiasing, .multipleTriangles, .depthTesting:
            return .primitives
        case .sampleTriangle, .sampleCube, .sampleEarth, .samplePose, .sampleEnvPhong:
            return .samples
        case .performanceBenchmark:
            return .diagnostics
        }
    }

    var symbolName: String {
        switch self {
        case .basicRasterization:
            return "triangle"
        case .gradientFill:
            return "paintpalette"
        case .barycentricInterpolation:
            return "point.3.connected.trianglepath.dotted"
        case .antialiasing:
            return "sparkles"
        case .multipleTriangles:
            return "square.3.layers.3d"
        case .depthTesting:
            return "square.stack.3d.down.forward"
        case .sampleTriangle:
            return "triangle.fill"
        case .sampleCube:
            return "cube"
        case .sampleEarth:
            return "globe"
        case .samplePose:
            return "viewfinder"
        case .sampleEnvPhong:
            return "sun.max"
        case .performanceBenchmark:
            return "gauge"
        }
    }
    
    var description: String {
        switch self {
        case .basicRasterization:
            return "Render a single triangle with solid color"
        case .gradientFill:
            return "Triangle with smooth color gradients"
        case .barycentricInterpolation:
            return "Visualize barycentric coordinates"
        case .antialiasing:
            return "Compare aliased vs anti-aliased rendering"
        case .multipleTriangles:
            return "Render multiple overlapping triangles"
        case .depthTesting:
            return "Test depth buffering with overlapping geometry"
        case .sampleTriangle:
            return "Original colored triangle sample"
        case .sampleCube:
            return "Original vertex-colored cube sample"
        case .sampleEarth:
            return "Original textured Earth sphere sample"
        case .samplePose:
            return "Gradient-based pose fitting sample"
        case .sampleEnvPhong:
            return "Original environment Phong sample"
        case .performanceBenchmark:
            return "Measure rasterization performance"
        }
    }

    var isOriginalSample: Bool {
        switch self {
        case .sampleTriangle, .sampleCube, .sampleEarth, .samplePose, .sampleEnvPhong:
            return true
        default:
            return false
        }
    }

    var isAnimated: Bool {
        switch self {
        case .basicRasterization, .antialiasing, .multipleTriangles:
            return true
        default:
            return false
        }
    }

    var preferredAspectRatio: Double {
        switch self {
        case .samplePose, .sampleEnvPhong:
            return 16.0 / 9.0
        default:
            return 1.0
        }
    }

    var supportsWireframe: Bool {
        !isOriginalSample && self != .performanceBenchmark
    }
}

enum DemoFeatureGroup: String, CaseIterable, Identifiable {
    case primitives = "Primitives"
    case samples = "Original Samples"
    case diagnostics = "Diagnostics"

    var id: String { rawValue }

    var features: [DemoFeature] {
        DemoFeature.allCases.filter { $0.group == self }
    }
}

/// Main view model for the demo application
class DemoViewModel: ObservableObject {
    @Published var selectedFeature: DemoFeature = .basicRasterization
    @Published var isRendering: Bool = false
    @Published var renderTime: Double = 0.0
    @Published var triangleCount: Int = 1
    @Published var renderWidth: Int = 0
    @Published var renderHeight: Int = 0
    @Published var showWireframe: Bool = false
    @Published var displayedFPS: Double = 0
    let deviceName: String
    var animationOffset: Double = 0.0

    private var rasterizer: DiffRasterizer?

    init() {
        rasterizer = DiffRasterizer()
        deviceName = getMetalDeviceInfo()?.name ?? "Metal GPU"
    }

    func selectFeature(_ feature: DemoFeature) {
        selectedFeature = feature
        displayedFPS = 0
        if !feature.supportsWireframe {
            showWireframe = false
        }
    }

    func updateRenderSize(width: Int, height: Int) {
        guard width != renderWidth || height != renderHeight else { return }
        renderWidth = width
        renderHeight = height
    }

    func advanceAnimation(deltaTime: Double) {
        guard selectedFeature.isAnimated else { return }
        animationOffset += deltaTime * 1.2
        if animationOffset > Double.pi * 2 {
            animationOffset.formTruncatingRemainder(dividingBy: Double.pi * 2)
        }
    }

    func updateFPS(_ fps: Double) {
        guard abs(displayedFPS - fps) >= 0.5 else { return }
        displayedFPS = fps
    }
    
    /// Generate triangles for the selected demo
    func generateTriangles() -> [(Vertex, Vertex, Vertex)] {
        var triangles: [(Vertex, Vertex, Vertex)] = []
        
        switch selectedFeature {
        case .basicRasterization:
            triangles.append(createBasicTriangle())
            
        case .gradientFill:
            triangles.append(createGradientTriangle())
            
        case .barycentricInterpolation:
            triangles.append(createBarycentricTriangle())
            
        case .antialiasing:
            triangles.append(createAATriangle())
            
        case .multipleTriangles:
            triangles = createMultipleTriangles()
            
        case .depthTesting:
            triangles = createDepthTestTriangles()

        case .sampleTriangle, .sampleCube, .sampleEarth, .samplePose, .sampleEnvPhong:
            triangles = []
            
        case .performanceBenchmark:
            triangles = createBenchmarkTriangles(count: triangleCount)
        }
        
        return triangles
    }

    func renderCurrentFrame(
        triangles: [(Vertex, Vertex, Vertex)]? = nil,
        width: Int,
        height: Int
    ) -> RasterizationResult? {
        guard let rasterizer = rasterizer else { return nil }

        if selectedFeature.isOriginalSample {
            return rasterizer.renderOriginalSample(
                selectedFeature,
                width: width,
                height: height
            )
        } else {
            let triangles = triangles ?? generateTriangles()
            let useAntialias = selectedFeature == .antialiasing
            return rasterizer.rasterizeTriangles(
                triangles,
                width: width,
                height: height,
                antialias: useAntialias
            )
        }
    }

    func renderCurrentTexture(width: Int, height: Int) -> MTLTexture? {
        guard let rasterizer else { return nil }
        guard !selectedFeature.isOriginalSample else {
            return nil
        }

        let triangles = generateTriangles()
        let useAntialias = selectedFeature == .antialiasing
        return rasterizer.rasterizeTrianglesToTexture(
            triangles,
            width: width,
            height: height,
            antialias: useAntialias
        )
    }
    
    private func createBasicTriangle() -> (Vertex, Vertex, Vertex) {
        let offset = Float(sin(animationOffset) * 0.16)
        let color = SIMD3<Float>(0.95, 0.22, 0.12)
        return (
            Vertex(position: SIMD2<Float>(0.0 + offset, 0.62), color: color, depth: 0.5),
            Vertex(position: SIMD2<Float>(-0.62 + offset, -0.54), color: color, depth: 0.5),
            Vertex(position: SIMD2<Float>(0.62 + offset, -0.54), color: color, depth: 0.5)
        )
    }
    
    private func createGradientTriangle() -> (Vertex, Vertex, Vertex) {
        return (
            Vertex(position: SIMD2<Float>(0.0, 0.7), color: SIMD3<Float>(1, 0.5, 0), depth: 0.3),
            Vertex(position: SIMD2<Float>(-0.7, -0.7), color: SIMD3<Float>(0, 1, 0.5), depth: 0.3),
            Vertex(position: SIMD2<Float>(0.7, -0.7), color: SIMD3<Float>(0.5, 0, 1), depth: 0.3)
        )
    }
    
    private func createBarycentricTriangle() -> (Vertex, Vertex, Vertex) {
        return (
            Vertex(position: SIMD2<Float>(0.0, 0.6), color: SIMD3<Float>(1, 0, 0), depth: 0.4),
            Vertex(position: SIMD2<Float>(-0.6, -0.6), color: SIMD3<Float>(0, 1, 0), depth: 0.4),
            Vertex(position: SIMD2<Float>(0.6, -0.6), color: SIMD3<Float>(0, 0, 1), depth: 0.4)
        )
    }
    
    private func createAATriangle() -> (Vertex, Vertex, Vertex) {
        let offset = Float(cos(animationOffset) * 0.2)
        return (
            Vertex(position: SIMD2<Float>(offset, 0.5), color: SIMD3<Float>(1, 1, 1), depth: 0.5),
            Vertex(position: SIMD2<Float>(-0.5 + offset, -0.5), color: SIMD3<Float>(0.5, 0.5, 0.5), depth: 0.5),
            Vertex(position: SIMD2<Float>(0.5 + offset, -0.5), color: SIMD3<Float>(0.8, 0.8, 0.8), depth: 0.5)
        )
    }
    
    private func createMultipleTriangles() -> [(Vertex, Vertex, Vertex)] {
        var triangles: [(Vertex, Vertex, Vertex)] = []
        let count = 5
        for i in 0..<count {
            let angle = Float(i) / Float(count) * Float.pi * 2 + Float(animationOffset)
            let radius: Float = 0.4
            let centerX = cos(angle) * radius
            let centerY = sin(angle) * radius
            let hue = Float(i) / Float(count)
            let c0 = palette(hue: hue)
            let c1 = palette(hue: hue + 0.22)
            let c2 = palette(hue: hue + 0.44)
            
            triangles.append((
                Vertex(position: SIMD2<Float>(centerX, centerY + 0.3), color: c0, depth: Float(i) * 0.1),
                Vertex(position: SIMD2<Float>(centerX - 0.25, centerY - 0.2), color: c1, depth: Float(i) * 0.1),
                Vertex(position: SIMD2<Float>(centerX + 0.25, centerY - 0.2), color: c2, depth: Float(i) * 0.1)
            ))
        }
        return triangles
    }
    
    private func createDepthTestTriangles() -> [(Vertex, Vertex, Vertex)] {
        return [
            // Back triangle (red)
            (
                Vertex(position: SIMD2<Float>(0.0, 0.5), color: SIMD3<Float>(1, 0, 0), depth: 0.8),
                Vertex(position: SIMD2<Float>(-0.6, -0.5), color: SIMD3<Float>(1, 0, 0), depth: 0.8),
                Vertex(position: SIMD2<Float>(0.6, -0.5), color: SIMD3<Float>(1, 0, 0), depth: 0.8)
            ),
            // Middle triangle (green)
            (
                Vertex(position: SIMD2<Float>(-0.3, 0.5), color: SIMD3<Float>(0, 1, 0), depth: 0.5),
                Vertex(position: SIMD2<Float>(-0.9, -0.5), color: SIMD3<Float>(0, 1, 0), depth: 0.5),
                Vertex(position: SIMD2<Float>(0.3, -0.5), color: SIMD3<Float>(0, 1, 0), depth: 0.5)
            ),
            // Front triangle (blue)
            (
                Vertex(position: SIMD2<Float>(0.3, 0.5), color: SIMD3<Float>(0, 0, 1), depth: 0.2),
                Vertex(position: SIMD2<Float>(-0.3, -0.5), color: SIMD3<Float>(0, 0, 1), depth: 0.2),
                Vertex(position: SIMD2<Float>(0.9, -0.5), color: SIMD3<Float>(0, 0, 1), depth: 0.2)
            )
        ]
    }
    
    private func createBenchmarkTriangles(count: Int) -> [(Vertex, Vertex, Vertex)] {
        var triangles: [(Vertex, Vertex, Vertex)] = []
        for i in 0..<count {
            let angle = Float(i) / Float(count) * Float.pi * 2
            let radius = 0.3 + 0.3 * pseudoRandom(i, seed: 17)
            let centerX = cos(angle) * radius
            let centerY = sin(angle) * radius
            let rotation = pseudoRandom(i, seed: 31) * Float.pi * 2
            let c0 = SIMD3<Float>(
                0.3 + 0.7 * pseudoRandom(i, seed: 101),
                0.3 + 0.7 * pseudoRandom(i, seed: 103),
                0.3 + 0.7 * pseudoRandom(i, seed: 107)
            )
            let c1 = SIMD3<Float>(
                0.3 + 0.7 * pseudoRandom(i, seed: 109),
                0.3 + 0.7 * pseudoRandom(i, seed: 113),
                0.3 + 0.7 * pseudoRandom(i, seed: 127)
            )
            let c2 = SIMD3<Float>(
                0.3 + 0.7 * pseudoRandom(i, seed: 131),
                0.3 + 0.7 * pseudoRandom(i, seed: 137),
                0.3 + 0.7 * pseudoRandom(i, seed: 139)
            )
            let d0 = pseudoRandom(i, seed: 149)
            let d1 = pseudoRandom(i, seed: 151)
            let d2 = pseudoRandom(i, seed: 157)
            
            triangles.append((
                Vertex(position: SIMD2<Float>(centerX + cos(rotation) * 0.1, centerY + sin(rotation) * 0.1), 
                       color: c0,
                       depth: d0),
                Vertex(position: SIMD2<Float>(centerX + cos(rotation + 2.094) * 0.1, centerY + sin(rotation + 2.094) * 0.1), 
                       color: c1,
                       depth: d1),
                Vertex(position: SIMD2<Float>(centerX + cos(rotation + 4.188) * 0.1, centerY + sin(rotation + 4.188) * 0.1), 
                       color: c2,
                       depth: d2)
            ))
        }
        return triangles
    }
    
    /// Run performance benchmark
    func runBenchmark() {
        guard let rasterizer = rasterizer else { return }
        
        let triangles = createBenchmarkTriangles(count: triangleCount)

        renderTime = rasterizer.benchmarkTextureRender(
            triangles,
            width: max(renderWidth, 1),
            height: max(renderHeight, 1),
            iterations: 16
        ) ?? 0
    }

    private func palette(hue: Float) -> SIMD3<Float> {
        let t = hue - floor(hue)
        let r = 0.5 + 0.5 * cos((t + 0.00) * Float.pi * 2)
        let g = 0.5 + 0.5 * cos((t + 0.33) * Float.pi * 2)
        let b = 0.5 + 0.5 * cos((t + 0.66) * Float.pi * 2)
        return SIMD3<Float>(r, g, b)
    }

    private func pseudoRandom(_ value: Int, seed: Int) -> Float {
        let x = sin(Float(value * 97 + seed * 131)) * 43_758.547
        return x - floor(x)
    }
}
