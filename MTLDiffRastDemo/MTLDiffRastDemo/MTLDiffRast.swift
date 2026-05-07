//
//  MTLDiffRast.swift
//  MTLDiffRastDemo
//
//  Demo adapter around the MTLDiffRast package.
//

import Foundation
import Metal
import MTLDiffRast
import simd

/// Triangle vertex with position and attributes
public struct Vertex {
    public var position: SIMD2<Float>
    public var color: SIMD3<Float>
    public var depth: Float
    
    public init(position: SIMD2<Float>, color: SIMD3<Float>, depth: Float = 0.0) {
        self.position = position
        self.color = color
        self.depth = depth
    }
}

/// Rasterization result containing pixel buffer and depth buffer
public struct RasterizationResult {
    public var pixels: [SIMD4<Float>]
    public var depthBuffer: [Float]
    public var width: Int
    public var height: Int
    
    public init(width: Int, height: Int) {
        self.width = width
        self.height = height
        self.pixels = Array(repeating: SIMD4<Float>(0, 0, 0, 0), count: width * height)
        self.depthBuffer = Array(repeating: Float.infinity, count: width * height)
    }
}

struct SampleLightingMaterial {
    var tint: SIMD3<Float>
    var diffuse: Float
    var specular: Float
    var shininess: Float
}

private struct SampleLightingParams {
    var pixelCount: UInt32
}

private struct SampleLightingMaterialParams {
    var tint: SIMD3<Float>
    var diffuse: Float
    var specular: Float
    var shininess: Float
}

private final class SampleLightingRenderer {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let shadedMeshPipeline: MTLComputePipelineState
    private let texturedSpherePipeline: MTLComputePipelineState
    private let phongPipeline: MTLComputePipelineState

    init(device: MTLDevice) throws {
        self.device = device
        guard let commandQueue = device.makeCommandQueue() else {
            throw RasterizerError.bufferCreationFailed("sampleLightingCommandQueue")
        }
        self.commandQueue = commandQueue

        let library = try device.makeLibrary(source: Self.shaderSource, options: nil)
        shadedMeshPipeline = try device.makeComputePipelineState(
            function: library.makeFunction(name: "shadeColoredMeshKernel").unwrap("shadeColoredMeshKernel")
        )
        texturedSpherePipeline = try device.makeComputePipelineState(
            function: library.makeFunction(name: "shadeTexturedSphereKernel").unwrap("shadeTexturedSphereKernel")
        )
        phongPipeline = try device.makeComputePipelineState(
            function: library.makeFunction(name: "shadePhongKernel").unwrap("shadePhongKernel")
        )
    }

    func shadeColoredMesh(attributes: [Float], mask: [Int32]) throws -> [Float] {
        try shade(
            pipeline: shadedMeshPipeline,
            floatInputs: [attributes],
            intInputs: [mask]
        )
    }

    func shadeTexturedSphere(samples: [Float], normals: [Float], mask: [Int32]) throws -> [Float] {
        try shade(
            pipeline: texturedSpherePipeline,
            floatInputs: [samples, normals],
            intInputs: [mask]
        )
    }

    func shadePhong(
        environment: [Float],
        normals: [Float],
        mask: [Int32],
        material: SampleLightingMaterial
    ) throws -> [Float] {
        let pixelCount = mask.count
        let params = SampleLightingParams(pixelCount: UInt32(pixelCount))
        let materialParams = SampleLightingMaterialParams(
            tint: material.tint,
            diffuse: material.diffuse,
            specular: material.specular,
            shininess: material.shininess
        )

        let environmentBuffer = try makeFloatBuffer(environment, label: "phongEnvironment")
        let normalsBuffer = try makeFloatBuffer(normals, label: "phongNormals")
        let maskBuffer = try makeIntBuffer(mask, label: "phongMask")
        let outputBuffer = try makeOutputBuffer(floatCount: pixelCount * 3, label: "phongOutput")
        let paramsBuffer = try makeParameterBuffer(params, label: "phongParams")
        let materialBuffer = try makeParameterBuffer(materialParams, label: "phongMaterial")

        try run(
            pipeline: phongPipeline,
            buffers: [
                environmentBuffer,
                normalsBuffer,
                maskBuffer,
                outputBuffer,
                paramsBuffer,
                materialBuffer
            ],
            pixelCount: pixelCount
        )

        return copyFloats(from: outputBuffer, count: pixelCount * 3)
    }

    private func shade(
        pipeline: MTLComputePipelineState,
        floatInputs: [[Float]],
        intInputs: [[Int32]]
    ) throws -> [Float] {
        let pixelCount = intInputs.first?.count ?? 0
        let params = SampleLightingParams(pixelCount: UInt32(pixelCount))

        let floatBuffers = try floatInputs.enumerated().map { index, values in
            try makeFloatBuffer(values, label: "lightingFloatInput\(index)")
        }
        let intBuffers = try intInputs.enumerated().map { index, values in
            try makeIntBuffer(values, label: "lightingIntInput\(index)")
        }
        let outputBuffer = try makeOutputBuffer(floatCount: pixelCount * 3, label: "lightingOutput")
        let paramsBuffer = try makeParameterBuffer(params, label: "lightingParams")

        try run(
            pipeline: pipeline,
            buffers: floatBuffers + intBuffers + [outputBuffer, paramsBuffer],
            pixelCount: pixelCount
        )

        return copyFloats(from: outputBuffer, count: pixelCount * 3)
    }

    private func run(
        pipeline: MTLComputePipelineState,
        buffers: [MTLBuffer],
        pixelCount: Int
    ) throws {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw RasterizerError.encodingFailed("sampleLightingCommandBuffer")
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw RasterizerError.encodingFailed("sampleLightingEncoder")
        }

        encoder.setComputePipelineState(pipeline)
        for (index, buffer) in buffers.enumerated() {
            encoder.setBuffer(buffer, offset: 0, index: index)
        }

        let width = max(1, pipeline.threadExecutionWidth)
        let threadsPerThreadgroup = MTLSize(width: width, height: 1, depth: 1)
        let threadsPerGrid = MTLSize(width: max(pixelCount, 1), height: 1, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw RasterizerError.commandExecutionFailed(error.localizedDescription)
        }
    }

    private func makeFloatBuffer(_ values: [Float], label: String) throws -> MTLBuffer {
        try makeBuffer(values, label: label)
    }

    private func makeIntBuffer(_ values: [Int32], label: String) throws -> MTLBuffer {
        try makeBuffer(values, label: label)
    }

    private func makeOutputBuffer(floatCount: Int, label: String) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(
            length: max(floatCount * MemoryLayout<Float>.stride, MemoryLayout<Float>.stride),
            options: .storageModeShared
        ) else {
            throw RasterizerError.bufferCreationFailed(label)
        }
        memset(buffer.contents(), 0, max(floatCount * MemoryLayout<Float>.stride, MemoryLayout<Float>.stride))
        return buffer
    }

    private func makeParameterBuffer<T>(_ value: T, label: String) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: MemoryLayout<T>.stride, options: .storageModeShared) else {
            throw RasterizerError.bufferCreationFailed(label)
        }
        var mutableValue = value
        withUnsafeBytes(of: &mutableValue) { bytes in
            guard let baseAddress = bytes.baseAddress else { return }
            memcpy(buffer.contents(), baseAddress, bytes.count)
        }
        return buffer
    }

    private func makeBuffer<T>(_ values: [T], label: String) throws -> MTLBuffer {
        let byteCount = max(values.count * MemoryLayout<T>.stride, MemoryLayout<T>.stride)
        guard let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw RasterizerError.bufferCreationFailed(label)
        }
        if !values.isEmpty {
            values.withUnsafeBytes { bytes in
                guard let baseAddress = bytes.baseAddress else { return }
                memcpy(buffer.contents(), baseAddress, bytes.count)
            }
        } else {
            memset(buffer.contents(), 0, byteCount)
        }
        return buffer
    }

    private func copyFloats(from buffer: MTLBuffer, count: Int) -> [Float] {
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct SampleLightingParams {
        uint pixelCount;
    };

    struct SampleLightingMaterialParams {
        float3 tint;
        float diffuse;
        float specular;
        float shininess;
    };

    inline float3 clampColor(float3 color) {
        return clamp(color, 0.0f, 1.0f);
    }

    kernel void shadeColoredMeshKernel(
        device const float* attributes [[buffer(0)]],
        device const int* mask [[buffer(1)]],
        device float* output [[buffer(2)]],
        constant SampleLightingParams& params [[buffer(3)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id >= params.pixelCount) {
            return;
        }

        uint outputBase = id * 3;
        if (mask[id] < 0) {
            output[outputBase + 0] = 0.0f;
            output[outputBase + 1] = 0.0f;
            output[outputBase + 2] = 0.0f;
            return;
        }

        uint attributeBase = id * 6;
        float3 baseColor = float3(
            attributes[attributeBase + 0],
            attributes[attributeBase + 1],
            attributes[attributeBase + 2]
        );
        float3 normal = normalize(float3(
            attributes[attributeBase + 3],
            attributes[attributeBase + 4],
            attributes[attributeBase + 5]
        ));
        float3 light = normalize(float3(-0.35f, 0.55f, 0.76f));
        float3 view = float3(0.0f, 0.0f, 1.0f);
        float diffuse = max(dot(normal, light), 0.0f);
        float rim = pow(max(1.0f - max(dot(normal, view), 0.0f), 0.0f), 2.0f);
        float3 lit = baseColor * (0.36f + 0.72f * diffuse) + float3(0.18f, 0.22f, 0.28f) * rim;
        lit = clampColor(lit);

        output[outputBase + 0] = lit.x;
        output[outputBase + 1] = lit.y;
        output[outputBase + 2] = lit.z;
    }

    kernel void shadeTexturedSphereKernel(
        device const float* samples [[buffer(0)]],
        device const float* normals [[buffer(1)]],
        device const int* mask [[buffer(2)]],
        device float* output [[buffer(3)]],
        constant SampleLightingParams& params [[buffer(4)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id >= params.pixelCount) {
            return;
        }

        uint outputBase = id * 3;
        if (mask[id] < 0) {
            output[outputBase + 0] = 0.0f;
            output[outputBase + 1] = 0.0f;
            output[outputBase + 2] = 0.0f;
            return;
        }

        uint base = id * 3;
        float3 normal = normalize(float3(normals[base + 0], normals[base + 1], normals[base + 2]));
        float3 sampleColor = float3(samples[base + 0], samples[base + 1], samples[base + 2]);
        float3 light = normalize(float3(0.45f, 0.65f, 0.85f));
        float3 view = float3(0.0f, 0.0f, 1.0f);
        float diffuse = max(dot(normal, light), 0.0f);
        float rim = pow(max(1.0f - max(dot(normal, view), 0.0f), 0.0f), 2.2f);
        float shade = 0.30f + 0.76f * diffuse;
        float3 atmosphere = float3(0.12f, 0.28f, 0.45f) * rim;
        float3 color = clampColor(sampleColor * shade + atmosphere);

        output[outputBase + 0] = color.x;
        output[outputBase + 1] = color.y;
        output[outputBase + 2] = color.z;
    }

    kernel void shadePhongKernel(
        device const float* environment [[buffer(0)]],
        device const float* normals [[buffer(1)]],
        device const int* mask [[buffer(2)]],
        device float* output [[buffer(3)]],
        constant SampleLightingParams& params [[buffer(4)]],
        constant SampleLightingMaterialParams& material [[buffer(5)]],
        uint id [[thread_position_in_grid]]
    ) {
        if (id >= params.pixelCount) {
            return;
        }

        uint outputBase = id * 3;
        if (mask[id] < 0) {
            output[outputBase + 0] = 0.0f;
            output[outputBase + 1] = 0.0f;
            output[outputBase + 2] = 0.0f;
            return;
        }

        uint base = id * 3;
        float3 normal = normalize(float3(normals[base + 0], normals[base + 1], normals[base + 2]));
        float3 envColor = float3(
            environment[base + 0] * material.tint.x,
            environment[base + 1] * material.tint.y,
            environment[base + 2] * material.tint.z
        );
        float3 light = normalize(float3(-0.30f, 0.55f, 0.78f));
        float3 view = float3(0.0f, 0.0f, 1.0f);
        float3 halfVector = normalize(light + view);
        float diffuse = max(dot(normal, light), 0.0f);
        float specular = pow(max(dot(normal, halfVector), 0.0f), material.shininess);
        float fresnel = pow(max(1.0f - max(dot(normal, view), 0.0f), 0.0f), 3.0f);
        float3 baseColor = float3(0.62f, 0.66f, 0.72f);
        float3 color =
            0.30f * envColor +
            material.diffuse * diffuse * baseColor * material.tint +
            float3(material.specular * specular) +
            float3(0.18f, 0.26f, 0.36f) * fresnel;
        color = clampColor(color);

        output[outputBase + 0] = color.x;
        output[outputBase + 1] = color.y;
        output[outputBase + 2] = color.z;
    }
    """
}

private extension Optional where Wrapped == MTLFunction {
    func unwrap(_ label: String) throws -> MTLFunction {
        guard let self else {
            throw RasterizerError.pipelineCreationFailed(label)
        }
        return self
    }
}

/// Differentiable rasterizer using Metal
public class DiffRasterizer {
    let rasterizer: MTLDiffRast.Rasterizer
    private let lightingRenderer: SampleLightingRenderer
    
    public init?(device: MTLDevice? = nil) {
        guard let device = device ?? MTLCreateSystemDefaultDevice() else {
            return nil
        }
        do {
            self.rasterizer = try MTLDiffRast.Rasterizer(device: device)
            self.lightingRenderer = try SampleLightingRenderer(device: device)
        } catch {
            return nil
        }
    }
    
    /// Rasterize a single triangle
    public func rasterizeTriangle(_ triangle: (Vertex, Vertex, Vertex), 
                                   width: Int, height: Int,
                                   antialias: Bool = false) -> RasterizationResult {
        return rasterizeTriangles([triangle], width: width, height: height, antialias: antialias)
    }
    
    /// Rasterize multiple triangles with depth testing
    public func rasterizeTriangles(_ triangles: [(Vertex, Vertex, Vertex)], 
                                    width: Int, height: Int,
                                    antialias: Bool = false) -> RasterizationResult {
        var result = RasterizationResult(width: width, height: height)
        guard width > 0, height > 0, !triangles.isEmpty else {
            return result
        }

        var positions: [SIMD4<Float>] = []
        var indices: [SIMD3<Int32>] = []
        var colors: [Float] = []

        for triangle in triangles {
            let base = Int32(positions.count)
            for vertex in [triangle.0, triangle.1, triangle.2] {
                positions.append(
                    SIMD4<Float>(
                        vertex.position.x,
                        vertex.position.y,
                        1.0 - vertex.depth,
                        1.0
                    )
                )
                colors.append(contentsOf: [vertex.color.x, vertex.color.y, vertex.color.z])
            }
            indices.append(SIMD3<Int32>(base, base + 1, base + 2))
        }

        do {
            let rasterOutput = try rasterizer.rasterize(
                positions: positions,
                triangles: indices,
                width: width,
                height: height
            )

            let interpolated = try rasterizer.interpolate(
                attributes: colors,
                triangles: indices,
                rasterOutput: rasterOutput,
                numAttributes: 3
            )

            let outputColors: [Float]
            if antialias {
                outputColors = try rasterizer.antialias(
                    color: interpolated.attributes,
                    channels: 3,
                    rasterOutput: rasterOutput,
                    positions: positions,
                    triangles: indices
                ).colors
            } else {
                outputColors = interpolated.attributes
            }

            for y in 0..<height {
                for x in 0..<width {
                    let sourceIndex = y * width + x
                    let r = outputColors[sourceIndex * 3 + 0]
                    let g = outputColors[sourceIndex * 3 + 1]
                    let b = outputColors[sourceIndex * 3 + 2]
                    let covered = rasterOutput.triangleIds[sourceIndex] >= 0
                    let edgeCoverage = min(max(max(r, max(g, b)), 0), 1)
                    guard covered || (antialias && edgeCoverage > 0.001) else {
                        continue
                    }

                    let destinationIndex = (height - 1 - y) * width + x
                    result.pixels[destinationIndex] = SIMD4<Float>(r, g, b, 1.0)
                    result.depthBuffer[destinationIndex] = 1.0 - rasterOutput.depthBuffer[sourceIndex]
                }
            }
        } catch {
            return result
        }

        return result
    }

    /// Rasterize directly to a GPU display texture, avoiding CPU frame readback.
    public func rasterizeTrianglesToTexture(
        _ triangles: [(Vertex, Vertex, Vertex)],
        width: Int,
        height: Int,
        antialias: Bool = false
    ) -> MTLTexture? {
        guard width > 0, height > 0, !triangles.isEmpty else {
            return nil
        }

        var positions: [SIMD4<Float>] = []
        var indices: [SIMD3<Int32>] = []
        var colors: [Float] = []

        for triangle in triangles {
            let base = Int32(positions.count)
            for vertex in [triangle.0, triangle.1, triangle.2] {
                positions.append(
                    SIMD4<Float>(
                        vertex.position.x,
                        vertex.position.y,
                        1.0 - vertex.depth,
                        1.0
                    )
                )
                colors.append(contentsOf: [vertex.color.x, vertex.color.y, vertex.color.z])
            }
            indices.append(SIMD3<Int32>(base, base + 1, base + 2))
        }

        do {
            return try rasterizer.rasterizeColorTexture(
                positions: positions,
                triangles: indices,
                colors: colors,
                width: width,
                height: height,
                antialias: antialias
            )
        } catch {
            return nil
        }
    }

    public func benchmarkTextureRender(
        _ triangles: [(Vertex, Vertex, Vertex)],
        width: Int,
        height: Int,
        iterations: Int = 12
    ) -> Double? {
        guard !triangles.isEmpty, width > 0, height > 0 else {
            return nil
        }

        let runCount = max(iterations, 1)
        let start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<runCount {
            _ = rasterizeTrianglesToTexture(
                triangles,
                width: width,
                height: height,
                antialias: false
            )
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        return elapsed * 1000 / Double(runCount)
    }

    func shadeColoredMeshPixels(attributes: [Float], mask: [Int32]) throws -> [Float] {
        try lightingRenderer.shadeColoredMesh(attributes: attributes, mask: mask)
    }

    func shadeTexturedSpherePixels(samples: [Float], normals: [Float], mask: [Int32]) throws -> [Float] {
        try lightingRenderer.shadeTexturedSphere(samples: samples, normals: normals, mask: mask)
    }

    func shadePhongPixels(
        environment: [Float],
        normals: [Float],
        mask: [Int32],
        material: SampleLightingMaterial
    ) throws -> [Float] {
        try lightingRenderer.shadePhong(
            environment: environment,
            normals: normals,
            mask: mask,
            material: material
        )
    }
    
    /// Apply gradient fill to a triangle
    public func applyGradient(_ triangle: (Vertex, Vertex, Vertex), 
                              type: GradientType = .linear,
                              width: Int, height: Int) -> RasterizationResult {
        return rasterizeTriangle(triangle, width: width, height: height)
    }
    
    public enum GradientType {
        case linear, radial, angular
    }
}
