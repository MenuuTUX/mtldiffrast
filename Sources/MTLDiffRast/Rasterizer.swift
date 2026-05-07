//
//  Rasterizer.swift
//  MTLDiffRast
//
//  Pure-Swift differentiable rasterizer over Metal compute kernels.
//  Mirrors the operations exposed by mtldiffrast-python:
//    rasterize / rasterizeBackward / interpolate / interpolateBackward /
//    antialias / texture.
//

import Foundation
import Metal
import simd

// ---------------------------------------------------------------------------
// Param structs — must match the kernel-side layouts in MetalShaders.metal.
// All fields are 4-byte aligned, so SIMD-style packing is safe.
// ---------------------------------------------------------------------------

private struct RasterizeParams {
    var numTriangles: Int32
    var numVertices:  Int32
    var width:        Int32
    var height:       Int32
    var xs:           Float
    var xo:           Float
    var ys:           Float
    var yo:           Float
}

private struct RasterizeBackwardParams {
    var numTriangles: Int32
    var numVertices:  Int32
    var width:        Int32
    var height:       Int32
    var xs:           Float
    var xo:           Float
    var ys:           Float
    var yo:           Float
    var enableDB:     Int32
}

private struct InterpolateParams {
    var numTriangles: Int32
    var numVertices:  Int32
    var numAttr:      Int32
    var width:        Int32
    var height:       Int32
}

private struct AntialiasParams {
    var numTriangles: Int32
    var numVertices:  Int32
    var width:        Int32
    var height:       Int32
    var channels:     Int32
    var xh:           Float
    var yh:           Float
}

private struct TextureParams {
    var filterMode:   Int32
    var boundaryMode: Int32
    var channels:     Int32
    var imgWidth:     Int32
    var imgHeight:    Int32
    var texWidth:     Int32
    var texHeight:    Int32
}

private struct PackColorTextureParams {
    var width:    Int32
    var height:   Int32
    var channels: Int32
    var unused:   Int32 = 0
}

private struct CopyParams {
    var count: Int32
}

// ---------------------------------------------------------------------------
// Rasterizer.
// ---------------------------------------------------------------------------

/// A Metal-accelerated differentiable rasterizer.
///
/// `Rasterizer` exposes forward and backward passes for the five core
/// primitives of differentiable rendering:
///
/// | Primitive | Forward | Backward |
/// |-----------|---------|----------|
/// | Rasterize | ``rasterize(positions:triangles:width:height:)`` | ``rasterizeBackward(positions:triangles:forwardOutput:gradOutput:vertexCount:gradBaryDerivatives:)`` |
/// | Interpolate | ``interpolate(attributes:triangles:rasterOutput:numAttributes:computeDerivatives:)`` | ``interpolateBackward(attributes:triangles:rasterOutput:gradOutput:numAttributes:)`` |
/// | Antialias | ``antialias(color:channels:rasterOutput:positions:triangles:)`` | *(no backward needed; corrections are differentiable by construction)* |
/// | Texture | ``texture(texture:texWidth:texHeight:channels:uv:outWidth:outHeight:filterMode:boundaryMode:)`` | ``textureBackward(texture:texWidth:texHeight:channels:uv:gradOutput:outWidth:outHeight:filterMode:boundaryMode:)`` |
/// | Display | ``rasterizeColorTexture(positions:triangles:colors:width:height:antialias:)`` | *(display path; not part of the gradient graph)* |
///
/// ## Coordinate Conventions
///
/// - Input positions are clip-space `(x, y, z, w)`.
/// - Pixel `(0, 0)` is the **bottom-left** corner (y-up, matching OpenGL /
///   nvdiffrast convention).
/// - Output arrays are row-major with `y = 0` at the bottom row.
/// - Triangles whose projected signed area is ≤ 0 are culled (back-face or
///   degenerate).
/// - Depth comparison retains the **largest** `z/w` (closest to the camera
///   in standard NDC, where near = +1 for reversed-Z / far = −1).
///
/// ## Triangle ID Encoding
///
/// Triangle IDs ≤ 16 777 216 are stored as `float(id + 1)`.  IDs above that
/// threshold use a bit-pattern encoding: `as_type<float>(0x4A800000 + id + 1)`.
/// Both representations are exact for up to ≈ 2³¹ triangles.
///
/// ## Thread Safety
///
/// `Rasterizer` is **not** thread-safe.  All calls must originate from a
/// single thread.  Use separate `Rasterizer` instances for concurrent use.
///
/// ## Example
///
/// ```swift
/// let rast = try Rasterizer()
///
/// let positions: [SIMD4<Float>] = [
///     .init( 0.0,  0.7, 0.5, 1),
///     .init(-0.7, -0.6, 0.5, 1),
///     .init( 0.7, -0.6, 0.5, 1),
/// ]
/// let triangles: [SIMD3<Int32>] = [.init(0, 1, 2)]
///
/// let out  = try rast.rasterize(positions: positions, triangles: triangles,
///                               width: 512, height: 512)
/// let attr = try rast.interpolate(attributes: [1,0,0, 0,1,0, 0,0,1],
///                                 triangles: triangles,
///                                 rasterOutput: out, numAttributes: 3)
/// let aa   = try rast.antialias(color: attr.attributes, channels: 3,
///                               rasterOutput: out,
///                               positions: positions, triangles: triangles)
/// ```
public final class Rasterizer {

    // MARK: - Properties

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let library: MTLLibrary
    private var pipelineStates: [String: MTLComputePipelineState] = [:]

    // MARK: - Init

    /// Creates a `Rasterizer` backed by the system default Metal device.
    ///
    /// On Apple Silicon the default device is the integrated GPU, which
    /// shares memory with the CPU and allows zero-copy buffer access.
    ///
    /// - Throws: ``RasterizerError/metalUnavailable`` when no GPU is present;
    ///   ``RasterizerError/pipelineCreationFailed(_:)`` when a kernel cannot
    ///   be compiled or linked.
    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw RasterizerError.metalUnavailable
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw RasterizerError.deviceNotFound
        }
        self.commandQueue = queue

        self.library = try Self.loadLibrary(device: device)
        try createPipelineStates()
    }

    /// Creates a `Rasterizer` backed by a specific Metal device.
    ///
    /// Use this initialiser when you need to target a particular GPU in a
    /// multi-GPU system, or when you already hold a `MTLDevice` reference
    /// for other Metal work and want to share it.
    ///
    /// - Parameter device: The Metal device to use for all compute work.
    /// - Throws: ``RasterizerError/deviceNotFound`` when the command queue
    ///   cannot be created; ``RasterizerError/pipelineCreationFailed(_:)``
    ///   when a kernel cannot be compiled or linked.
    public init(device: MTLDevice) throws {
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw RasterizerError.deviceNotFound
        }
        self.commandQueue = queue

        self.library = try Self.loadLibrary(device: device)
        try createPipelineStates()
    }

    private static func loadLibrary(device: MTLDevice) throws -> MTLLibrary {
        // Prefer a precompiled default library if one exists in the bundle.
        if let lib = try? device.makeDefaultLibrary(bundle: Bundle.module) {
            return lib
        }

        // Otherwise compile the .metal source from the bundle's resources.
        guard let url = Bundle.module.url(forResource: "MetalShaders", withExtension: "metal") else {
            throw RasterizerError.pipelineCreationFailed(
                "MetalShaders.metal not found in bundle"
            )
        }
        do {
            let source = try String(contentsOf: url, encoding: .utf8)
            return try device.makeLibrary(source: source, options: nil)
        } catch {
            throw RasterizerError.pipelineCreationFailed(
                "Could not compile shader library: \(error.localizedDescription)"
            )
        }
    }

    private func createPipelineStates() throws {
        let kernelNames = [
            "rasterizeKernel",
            "rasterizeBackwardKernel",
            "interpolateKernel",
            "interpolateBackwardKernel",
            "packColorTextureKernel",
            "antialiasKernel",
            "textureKernel",
            "textureBackwardKernel",
            "copyFloatBufferKernel",
        ]
        for name in kernelNames {
            guard let function = library.makeFunction(name: name) else {
                throw RasterizerError.pipelineCreationFailed("Kernel '\(name)' not found")
            }
            do {
                pipelineStates[name] = try device.makeComputePipelineState(function: function)
            } catch {
                throw RasterizerError.pipelineCreationFailed(error.localizedDescription)
            }
        }
    }

    private func pipeline(_ name: String) throws -> MTLComputePipelineState {
        guard let p = pipelineStates[name] else {
            throw RasterizerError.pipelineCreationFailed("\(name) pipeline missing")
        }
        return p
    }

    // MARK: - Buffer helpers

    private func makeBuffer<T>(_ values: [T], label: String) throws -> MTLBuffer {
        let length = max(values.count * MemoryLayout<T>.stride, 16)
        guard let buf = device.makeBuffer(length: length, options: .storageModeShared) else {
            throw RasterizerError.bufferCreationFailed(label)
        }
        if !values.isEmpty {
            values.withUnsafeBytes { src in
                if let base = src.baseAddress {
                    memcpy(buf.contents(), base, values.count * MemoryLayout<T>.stride)
                }
            }
        }
        return buf
    }

    private func makeZeroBuffer(byteCount: Int, label: String) throws -> MTLBuffer {
        guard let buf = device.makeBuffer(length: max(byteCount, 16), options: .storageModeShared) else {
            throw RasterizerError.bufferCreationFailed(label)
        }
        memset(buf.contents(), 0, max(byteCount, 16))
        return buf
    }

    private func makeParamBuffer<T>(_ value: T, label: String) throws -> MTLBuffer {
        let length = MemoryLayout<T>.stride
        guard let buf = device.makeBuffer(length: length, options: .storageModeShared) else {
            throw RasterizerError.bufferCreationFailed(label)
        }
        _ = withUnsafePointer(to: value) { ptr in
            memcpy(buf.contents(), ptr, length)
        }
        return buf
    }

    private func decodeTriangleID(_ encoded: Float) -> Int32 {
        if encoded <= 0 {
            return -1
        }
        if encoded <= 16_777_216 {
            return Int32(encoded) - 1
        }
        return Int32(bitPattern: encoded.bitPattern) - Int32(0x4a800000) - 1
    }

    private func encodeTriangleID(_ triangleID: Int32) -> Float {
        let x = triangleID + 1
        if x <= 0 {
            return 0
        }
        if x <= 0x01000000 {
            return Float(x)
        }
        return Float(bitPattern: UInt32(bitPattern: 0x4a800000 + x))
    }

    // dispatch2D — non-uniform dispatch; Metal handles grid/threadgroup rounding.
    // A fixed 16×16 threadgroup gives good occupancy on all Apple GPUs without
    // the extra branches that a min(16, width) guard would add for typical
    // image sizes.  The in-kernel bounds checks remain as a safety net.
    private func dispatch2D(
        encoder: MTLComputeCommandEncoder,
        pipeline: MTLComputePipelineState,
        width: Int,
        height: Int
    ) {
        encoder.setComputePipelineState(pipeline)
        let threadGroup     = MTLSize(width: 16, height: 16, depth: 1)
        let threadsPerGrid  = MTLSize(width: width, height: height, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadGroup)
    }

    private func runCommand(
        _ body: (MTLCommandBuffer, MTLComputeCommandEncoder) throws -> Void
    ) throws {
        guard let cmd = commandQueue.makeCommandBuffer() else {
            throw RasterizerError.encodingFailed("Could not create command buffer")
        }
        guard let enc = cmd.makeComputeCommandEncoder() else {
            throw RasterizerError.encodingFailed("Could not create compute encoder")
        }
        try body(cmd, enc)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        if let err = cmd.error {
            throw RasterizerError.commandExecutionFailed(err.localizedDescription)
        }
    }

    // MARK: - Public API: forward rasterize

    /// Rasterizes a triangle mesh in clip space.
    ///
    /// Each thread covers one output pixel.  For every pixel the kernel
    /// iterates over all triangles, evaluates clip-space edge functions, and
    /// keeps the nearest front-facing (CCW) triangle.  The result contains
    /// barycentric coordinates, depth, and the triangle index for every covered
    /// pixel.
    ///
    /// - Parameters:
    ///   - positions: Clip-space vertex positions `(x, y, z, w)`.  Must contain
    ///     at least 3 elements.
    ///   - triangles: Per-triangle vertex-index triples (CCW winding).
    ///   - width:     Output image width in pixels.  Must be > 0.
    ///   - height:    Output image height in pixels.  Must be > 0.
    ///
    /// - Returns: A ``RasterOutput`` with per-pixel barycentric coordinates,
    ///   depth, triangle IDs, and screen-space barycentric derivatives.
    ///
    /// - Throws: ``RasterizerError/invalidResolution(width:height:)`` for a
    ///   zero-dimension image; ``RasterizerError/invalidTriangleCount(_:)`` for
    ///   an empty triangle list; ``RasterizerError/invalidVertexCount(_:)`` for
    ///   fewer than three vertices.
    ///
    /// - Complexity: O(*T* × *W* × *H*) GPU threads, where *T* is the triangle
    ///   count and *W* × *H* is the pixel count.
    public func rasterize(
        positions: [SIMD4<Float>],
        triangles: [SIMD3<Int32>],
        width: Int,
        height: Int
    ) throws -> RasterOutput {
        guard width > 0, height > 0 else {
            throw RasterizerError.invalidResolution(width: width, height: height)
        }
        guard !triangles.isEmpty else {
            throw RasterizerError.invalidTriangleCount(0)
        }
        guard positions.count >= 3 else {
            throw RasterizerError.invalidVertexCount(positions.count)
        }

        let pixelCount = width * height
        let numTriangles = Int32(triangles.count)
        let numVertices = Int32(positions.count)

        let positionBuffer = try makeBuffer(positions, label: "positions")
        let triangleBuffer = try makeBuffer(triangles, label: "triangles")
        let rastBuffer = try makeZeroBuffer(
            byteCount: pixelCount * MemoryLayout<SIMD4<Float>>.stride,
            label: "rastOut"
        )
        let rastDBBuffer = try makeZeroBuffer(
            byteCount: pixelCount * MemoryLayout<SIMD4<Float>>.stride,
            label: "rastDB"
        )

        let params = RasterizeParams(
            numTriangles: numTriangles,
            numVertices: numVertices,
            width: Int32(width),
            height: Int32(height),
            xs: 2.0 / Float(width),
            xo: -1.0 + 1.0 / Float(width),
            ys: 2.0 / Float(height),
            yo: -1.0 + 1.0 / Float(height)
        )
        let paramsBuffer = try makeParamBuffer(params, label: "rasterizeParams")

        try runCommand { _, enc in
            let pipeline = try self.pipeline("rasterizeKernel")
            enc.setBuffer(positionBuffer, offset: 0, index: 0)
            enc.setBuffer(triangleBuffer, offset: 0, index: 1)
            enc.setBuffer(paramsBuffer,   offset: 0, index: 2)
            enc.setBuffer(rastBuffer,     offset: 0, index: 3)
            enc.setBuffer(rastDBBuffer,   offset: 0, index: 4)
            self.dispatch2D(encoder: enc, pipeline: pipeline, width: width, height: height)
        }

        // Unpack the (u, v, z, triId+1) buffer.
        // bindMemory gives a typed pointer into the shared buffer — no extra
        // copy is needed.  baryDerivatives is a direct UnsafeBufferPointer
        // copy; triangleIds/depth/barycentrics need per-element decode/extract.
        let rastPtr   = rastBuffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: pixelCount)
        let rastDBPtr = rastDBBuffer.contents().bindMemory(to: SIMD4<Float>.self, capacity: pixelCount)

        // Direct O(n) copy of the derivative buffer — no intermediate allocation.
        let baryDerivatives = Array(UnsafeBufferPointer(start: rastDBPtr, count: pixelCount))

        var triangleIds  = [Int32](repeating: -1, count: pixelCount)
        var depthBuffer  = [Float](repeating:  0, count: pixelCount)
        var barycentrics = [SIMD2<Float>](repeating: .zero, count: pixelCount)

        for i in 0..<pixelCount {
            let r = rastPtr[i]
            triangleIds[i]  = decodeTriangleID(r.w)
            depthBuffer[i]  = r.z
            barycentrics[i] = SIMD2<Float>(r.x, r.y)
        }

        return RasterOutput(
            width: width,
            height: height,
            triangleIds: triangleIds,
            depthBuffer: depthBuffer,
            barycentrics: barycentrics,
            baryDerivatives: baryDerivatives
        )
    }

    // MARK: - Public API: backward rasterize

    /// Computes vertex-position gradients for the rasterize backward pass.
    ///
    /// Implements the Laine et al. 2020 position-gradient kernel.  The
    /// gradient flows through the `u` and `v` barycentric components; `z`
    /// and triangle-ID outputs are treated as non-differentiable.
    ///
    /// - Parameters:
    ///   - positions:          The same clip-space positions used in the
    ///     corresponding ``rasterize(positions:triangles:width:height:)`` call.
    ///   - triangles:          The same triangle index list.
    ///   - forwardOutput:      The ``RasterOutput`` from the forward pass.
    ///   - gradOutput:         Upstream gradient for the rasterizer output.
    ///     Accepts either `pixelCount` scalars (driving only `du`) or
    ///     `pixelCount × 4` floats laid out as `(du, dv, dz, dtri)` per pixel.
    ///   - vertexCount:        Total number of vertices.  Must equal
    ///     `positions.count`.
    ///   - gradBaryDerivatives: Optional `pixelCount × 4` gradient tensor for
    ///     the screen-space barycentric derivatives `(du/dx, du/dy, dv/dx,
    ///     dv/dy)`.  Pass `nil` (default) to disable the DB chain.
    ///
    /// - Returns: A ``RasterGradientOutput`` whose `positionGradients` array
    ///   has layout `[vertexCount × 4]`.
    ///
    /// - Throws: ``RasterizerError/invalidVertexCount(_:)`` when `vertexCount`
    ///   doesn't match `positions.count`; ``RasterizerError/encodingFailed(_:)``
    ///   when `gradOutput` has an unexpected length.
    public func rasterizeBackward(
        positions: [SIMD4<Float>],
        triangles: [SIMD3<Int32>],
        forwardOutput: RasterOutput,
        gradOutput: [Float],
        vertexCount: Int,
        gradBaryDerivatives: [Float]? = nil
    ) throws -> RasterGradientOutput {
        let pixelCount = forwardOutput.pixelCount
        let width = forwardOutput.width
        let height = forwardOutput.height

        guard vertexCount > 0 else {
            throw RasterizerError.invalidVertexCount(vertexCount)
        }
        guard vertexCount == positions.count else {
            throw RasterizerError.invalidVertexCount(vertexCount)
        }

        // Repack the forward rast buffer into the kernel's float4 layout.
        var rastPacked = [SIMD4<Float>](repeating: SIMD4<Float>(0, 0, 0, 0), count: pixelCount)
        for i in 0..<pixelCount {
            let triPlusOne = encodeTriangleID(forwardOutput.triangleIds[i])
            let bary = forwardOutput.barycentrics[i]
            rastPacked[i] = SIMD4<Float>(bary.x, bary.y, forwardOutput.depthBuffer[i], triPlusOne)
        }

        // Expand grad to [H*W*4]. Accept either flat scalars-per-pixel (only
        // .x is set) or fully-shaped [H*W*4] grads.
        var gradFloat4 = [SIMD4<Float>](repeating: SIMD4<Float>(0, 0, 0, 0), count: pixelCount)
        if gradOutput.count == pixelCount * 4 {
            for i in 0..<pixelCount {
                gradFloat4[i] = SIMD4<Float>(
                    gradOutput[i * 4 + 0],
                    gradOutput[i * 4 + 1],
                    gradOutput[i * 4 + 2],
                    gradOutput[i * 4 + 3]
                )
            }
        } else if gradOutput.count == pixelCount {
            for i in 0..<pixelCount {
                gradFloat4[i] = SIMD4<Float>(gradOutput[i], 0, 0, 0)
            }
        } else {
            throw RasterizerError.encodingFailed(
                "gradOutput length \(gradOutput.count) doesn't match pixel count \(pixelCount)"
            )
        }

        var gradDB = [SIMD4<Float>](repeating: SIMD4<Float>(0, 0, 0, 0), count: pixelCount)
        let enableDB: Int32
        if let gradBaryDerivatives {
            guard gradBaryDerivatives.count == pixelCount * 4 else {
                throw RasterizerError.encodingFailed(
                    "gradBaryDerivatives must have length pixelCount*4"
                )
            }
            for i in 0..<pixelCount {
                gradDB[i] = SIMD4<Float>(
                    gradBaryDerivatives[i * 4 + 0],
                    gradBaryDerivatives[i * 4 + 1],
                    gradBaryDerivatives[i * 4 + 2],
                    gradBaryDerivatives[i * 4 + 3]
                )
            }
            enableDB = 1
        } else {
            enableDB = 0
        }

        let positionBuffer  = try makeBuffer(positions, label: "positions")
        let triangleBuffer  = try makeBuffer(triangles, label: "triangles")
        let rastBuffer      = try makeBuffer(rastPacked, label: "rastOut")
        let dyBuffer        = try makeBuffer(gradFloat4, label: "dy")
        let ddbBuffer       = try makeBuffer(gradDB, label: "ddb")
        let gradPosBuffer   = try makeZeroBuffer(
            byteCount: vertexCount * 4 * MemoryLayout<Float>.stride,
            label: "gradPos"
        )

        let params = RasterizeBackwardParams(
            numTriangles: Int32(triangles.count),
            numVertices:  Int32(vertexCount),
            width:        Int32(width),
            height:       Int32(height),
            xs: 2.0 / Float(width),
            xo: -1.0 + 1.0 / Float(width),
            ys: 2.0 / Float(height),
            yo: -1.0 + 1.0 / Float(height),
            enableDB: enableDB
        )
        let paramsBuffer = try makeParamBuffer(params, label: "rasterizeParams")

        try runCommand { _, enc in
            let pipeline = try self.pipeline("rasterizeBackwardKernel")
            enc.setBuffer(positionBuffer, offset: 0, index: 0)
            enc.setBuffer(triangleBuffer, offset: 0, index: 1)
            enc.setBuffer(rastBuffer,     offset: 0, index: 2)
            enc.setBuffer(dyBuffer,       offset: 0, index: 3)
            enc.setBuffer(gradPosBuffer,  offset: 0, index: 4)
            enc.setBuffer(ddbBuffer,      offset: 0, index: 5)
            enc.setBuffer(paramsBuffer,   offset: 0, index: 6)
            self.dispatch2D(encoder: enc, pipeline: pipeline, width: width, height: height)
        }

        let gradPtr       = gradPosBuffer.contents().bindMemory(to: Float.self, capacity: vertexCount * 4)
        let gradPositions = Array(UnsafeBufferPointer(start: gradPtr, count: vertexCount * 4))
        return RasterGradientOutput(positionGradients: gradPositions)
    }

    // MARK: - Public API: interpolate

    /// Interpolates per-vertex attributes across the rasterized image.
    ///
    /// For each covered pixel, the kernel weights the three vertex attribute
    /// vectors of the covering triangle by the pixel's perspective-correct
    /// barycentric coordinates.
    ///
    /// - Parameters:
    ///   - attributes:          Per-vertex attribute data, layout
    ///     `[vertexCount × numAttributes]`.
    ///   - triangles:           Triangle index list.
    ///   - rasterOutput:        Forward rasterizer output.
    ///   - numAttributes:       Number of scalar attributes per vertex.
    ///   - computeDerivatives:  When `true`, the returned
    ///     ``InterpolateOutput/attributeDerivatives`` array contains
    ///     screen-space `(dx, dy)` derivatives for every pixel and attribute,
    ///     computed from ``RasterOutput/baryDerivatives``.  Defaults to `false`.
    ///
    /// - Returns: An ``InterpolateOutput`` containing per-pixel attributes,
    ///   full barycentric coordinates, and optional screen-space derivatives.
    ///
    /// - Throws: ``RasterizerError/encodingFailed(_:)`` for mismatched array
    ///   sizes; ``RasterizerError/invalidVertexCount(_:)`` when the attribute
    ///   array is not evenly divisible by `numAttributes`.
    public func interpolate(
        attributes: [Float],
        triangles: [SIMD3<Int32>],
        rasterOutput: RasterOutput,
        numAttributes: Int,
        computeDerivatives: Bool = false
    ) throws -> InterpolateOutput {
        let width = rasterOutput.width
        let height = rasterOutput.height
        let pixelCount = rasterOutput.pixelCount

        guard numAttributes > 0 else {
            throw RasterizerError.encodingFailed("numAttributes must be > 0")
        }
        let vertexCount = attributes.count / numAttributes
        guard vertexCount * numAttributes == attributes.count, vertexCount > 0 else {
            throw RasterizerError.invalidVertexCount(vertexCount)
        }

        // Pack the rast buffer the same way as in the forward kernel.
        var rastPacked = [SIMD4<Float>](repeating: SIMD4<Float>(0, 0, 0, 0), count: pixelCount)
        for i in 0..<pixelCount {
            let triPlusOne = rasterOutput.triangleIds[i] < 0
                ? Float(0)
                : encodeTriangleID(rasterOutput.triangleIds[i])
            let bary = rasterOutput.barycentrics[i]
            rastPacked[i] = SIMD4<Float>(bary.x, bary.y, rasterOutput.depthBuffer[i], triPlusOne)
        }

        let triangleBuffer  = try makeBuffer(triangles, label: "triangles")
        let attrBuffer      = try makeBuffer(attributes, label: "attributes")
        let rastBuffer      = try makeBuffer(rastPacked, label: "rastOut")
        let outputBuffer    = try makeZeroBuffer(
            byteCount: pixelCount * numAttributes * MemoryLayout<Float>.stride,
            label: "interpOut"
        )
        let baryBuffer      = try makeZeroBuffer(
            byteCount: pixelCount * 3 * MemoryLayout<Float>.stride,
            label: "interpBary"
        )

        let params = InterpolateParams(
            numTriangles: Int32(triangles.count),
            numVertices:  Int32(vertexCount),
            numAttr:      Int32(numAttributes),
            width:        Int32(width),
            height:       Int32(height)
        )
        let paramsBuffer = try makeParamBuffer(params, label: "interpolateParams")

        try runCommand { _, enc in
            let pipeline = try self.pipeline("interpolateKernel")
            enc.setBuffer(triangleBuffer, offset: 0, index: 0)
            enc.setBuffer(attrBuffer,     offset: 0, index: 1)
            enc.setBuffer(rastBuffer,     offset: 0, index: 2)
            enc.setBuffer(paramsBuffer,   offset: 0, index: 3)
            enc.setBuffer(outputBuffer,   offset: 0, index: 4)
            enc.setBuffer(baryBuffer,     offset: 0, index: 5)
            self.dispatch2D(encoder: enc, pipeline: pipeline, width: width, height: height)
        }

        let outPtr = outputBuffer.contents().bindMemory(
            to: Float.self,
            capacity: pixelCount * numAttributes
        )
        let baryPtr = baryBuffer.contents().bindMemory(
            to: Float.self,
            capacity: pixelCount * 3
        )

        let attrs = Array(UnsafeBufferPointer(start: outPtr,  count: pixelCount * numAttributes))
        let bary  = Array(UnsafeBufferPointer(start: baryPtr, count: pixelCount * 3))

        let attributeDerivatives: [Float]?
        if computeDerivatives {
            var derivs = [Float](repeating: 0, count: pixelCount * numAttributes * 2)
            for p in 0..<pixelCount {
                let triID = rasterOutput.triangleIds[p]
                if triID < 0 || Int(triID) >= triangles.count {
                    continue
                }
                let tri = triangles[Int(triID)]
                let i0 = Int(tri.x), i1 = Int(tri.y), i2 = Int(tri.z)
                if i0 < 0 || i0 >= vertexCount || i1 < 0 || i1 >= vertexCount || i2 < 0 || i2 >= vertexCount {
                    continue
                }
                let db = rasterOutput.baryDerivatives[p]
                for a in 0..<numAttributes {
                    let attr0 = attributes[i0 * numAttributes + a]
                    let attr1 = attributes[i1 * numAttributes + a]
                    let attr2 = attributes[i2 * numAttributes + a]
                    let dx = db.x * (attr0 - attr2) + db.z * (attr1 - attr2)
                    let dy = db.y * (attr0 - attr2) + db.w * (attr1 - attr2)
                    let base = (p * numAttributes + a) * 2
                    derivs[base + 0] = dx
                    derivs[base + 1] = dy
                }
            }
            attributeDerivatives = derivs
        } else {
            attributeDerivatives = nil
        }

        return InterpolateOutput(
            pixelCount: pixelCount,
            numAttributes: numAttributes,
            attributes: attrs,
            barycentricCoords: bary,
            attributeDerivatives: attributeDerivatives
        )
    }

    /// Computes attribute and rasterizer gradients for the interpolate backward pass.
    ///
    /// The kernel scatters per-pixel upstream gradients back to the three
    /// vertex-attribute slots of each covering triangle using atomic float adds.
    ///
    /// - Parameters:
    ///   - attributes:   Per-vertex attribute data, layout
    ///     `[vertexCount × numAttributes]`.
    ///   - triangles:    Triangle index list.
    ///   - rasterOutput: Forward rasterizer output.
    ///   - gradOutput:   Upstream gradient for the interpolated output, layout
    ///     `[pixelCount × numAttributes]`.
    ///   - numAttributes: Number of scalar attributes per vertex.
    ///
    /// - Returns: A tuple `(gradAttributes, gradRast)`:
    ///   - `gradAttributes` has shape `[vertexCount × numAttributes]`.
    ///   - `gradRast` has shape `[pixelCount × 4]` and contains gradients for
    ///     the `(u, v, z, triId)` raster buffer.
    ///
    /// - Throws: ``RasterizerError/encodingFailed(_:)`` for mismatched array sizes.
    public func interpolateBackward(
        attributes: [Float],
        triangles: [SIMD3<Int32>],
        rasterOutput: RasterOutput,
        gradOutput: [Float],          // [H * W * A]
        numAttributes: Int
    ) throws -> (gradAttributes: [Float], gradRast: [Float]) {
        let width = rasterOutput.width
        let height = rasterOutput.height
        let pixelCount = rasterOutput.pixelCount
        guard numAttributes > 0 else {
            throw RasterizerError.encodingFailed("numAttributes must be > 0")
        }
        let vertexCount = attributes.count / numAttributes
        guard vertexCount * numAttributes == attributes.count, vertexCount > 0 else {
            throw RasterizerError.invalidVertexCount(vertexCount)
        }
        guard gradOutput.count == pixelCount * numAttributes else {
            throw RasterizerError.encodingFailed(
                "gradOutput must have length pixelCount*numAttributes"
            )
        }

        var rastPacked = [SIMD4<Float>](repeating: SIMD4<Float>(0, 0, 0, 0), count: pixelCount)
        for i in 0..<pixelCount {
            let triPlusOne = rasterOutput.triangleIds[i] < 0
                ? Float(0)
                : encodeTriangleID(rasterOutput.triangleIds[i])
            let bary = rasterOutput.barycentrics[i]
            rastPacked[i] = SIMD4<Float>(bary.x, bary.y, rasterOutput.depthBuffer[i], triPlusOne)
        }

        let triangleBuffer = try makeBuffer(triangles, label: "triangles")
        let attrBuffer     = try makeBuffer(attributes, label: "attributes")
        let rastBuffer     = try makeBuffer(rastPacked, label: "rastOut")
        let dyBuffer       = try makeBuffer(gradOutput, label: "dy")
        let gradAttrBuffer = try makeZeroBuffer(
            byteCount: vertexCount * numAttributes * MemoryLayout<Float>.stride,
            label: "gradAttr"
        )
        let gradRastBuffer = try makeZeroBuffer(
            byteCount: pixelCount * MemoryLayout<SIMD4<Float>>.stride,
            label: "gradRast"
        )

        let params = InterpolateParams(
            numTriangles: Int32(triangles.count),
            numVertices:  Int32(vertexCount),
            numAttr:      Int32(numAttributes),
            width:        Int32(width),
            height:       Int32(height)
        )
        let paramsBuffer = try makeParamBuffer(params, label: "interpolateParams")

        try runCommand { _, enc in
            let pipeline = try self.pipeline("interpolateBackwardKernel")
            enc.setBuffer(triangleBuffer, offset: 0, index: 0)
            enc.setBuffer(attrBuffer,     offset: 0, index: 1)
            enc.setBuffer(rastBuffer,     offset: 0, index: 2)
            enc.setBuffer(dyBuffer,       offset: 0, index: 3)
            enc.setBuffer(gradAttrBuffer, offset: 0, index: 4)
            enc.setBuffer(gradRastBuffer, offset: 0, index: 5)
            enc.setBuffer(paramsBuffer,   offset: 0, index: 6)
            self.dispatch2D(encoder: enc, pipeline: pipeline, width: width, height: height)
        }

        let gAttrPtr = gradAttrBuffer.contents().bindMemory(to: Float.self, capacity: vertexCount * numAttributes)
        let gRastPtr = gradRastBuffer.contents().bindMemory(to: Float.self, capacity: pixelCount * 4)
        let gradAttr = Array(UnsafeBufferPointer(start: gAttrPtr, count: vertexCount * numAttributes))
        let gradRast = Array(UnsafeBufferPointer(start: gRastPtr, count: pixelCount * 4))

        return (gradAttr, gradRast)
    }

    // MARK: - Public API: GPU display texture

    /// Rasterizes a colour mesh directly into a BGRA `MTLTexture` for display.
    ///
    /// This path chains rasterize → interpolate → (optional antialias) →
    /// pack into a BGRA8 texture in a *single* command buffer, avoiding any
    /// CPU readback.  It is designed for real-time `MTKView` rendering;
    /// it is **not** part of the differentiable gradient graph.
    ///
    /// - Parameters:
    ///   - positions:  Clip-space vertex positions `(x, y, z, w)`.
    ///   - triangles:  Triangle index list.
    ///   - colors:     Per-vertex RGB colours, layout `[vertexCount × 3]`.
    ///   - width:      Texture width in pixels.
    ///   - height:     Texture height in pixels.
    ///   - antialias:  When `true`, silhouette antialiasing is applied before
    ///     packing.  Defaults to `false`.
    ///
    /// - Returns: A `.bgra8Unorm`, `.private` `MTLTexture` of size
    ///   `width × height`.
    ///
    /// - Throws: ``RasterizerError`` variants for invalid inputs or GPU
    ///   resource failures.
    public func rasterizeColorTexture(
        positions: [SIMD4<Float>],
        triangles: [SIMD3<Int32>],
        colors: [Float],
        width: Int,
        height: Int,
        antialias: Bool = false
    ) throws -> MTLTexture {
        guard width > 0, height > 0 else {
            throw RasterizerError.invalidResolution(width: width, height: height)
        }
        guard !triangles.isEmpty else {
            throw RasterizerError.invalidTriangleCount(0)
        }
        guard positions.count >= 3 else {
            throw RasterizerError.invalidVertexCount(positions.count)
        }
        let channels = 3
        guard colors.count == positions.count * channels else {
            throw RasterizerError.encodingFailed("colors must have length vertexCount*3")
        }

        let pixelCount = width * height
        let positionBuffer = try makeBuffer(positions, label: "positions")
        let triangleBuffer = try makeBuffer(triangles, label: "triangles")
        let colorBuffer = try makeBuffer(colors, label: "colors")
        let rastBuffer = try makeZeroBuffer(
            byteCount: pixelCount * MemoryLayout<SIMD4<Float>>.stride,
            label: "rastOut"
        )
        let rastDBBuffer = try makeZeroBuffer(
            byteCount: pixelCount * MemoryLayout<SIMD4<Float>>.stride,
            label: "rastDB"
        )
        let shadedBuffer = try makeZeroBuffer(
            byteCount: pixelCount * channels * MemoryLayout<Float>.stride,
            label: "shadedColor"
        )
        let baryBuffer = try makeZeroBuffer(
            byteCount: pixelCount * 3 * MemoryLayout<Float>.stride,
            label: "interpBary"
        )
        let aaBuffer = try makeZeroBuffer(
            byteCount: pixelCount * channels * MemoryLayout<Float>.stride,
            label: "aaColor"
        )

        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        textureDescriptor.usage = [.shaderRead, .shaderWrite]
        textureDescriptor.storageMode = .private
        guard let texture = device.makeTexture(descriptor: textureDescriptor) else {
            throw RasterizerError.bufferCreationFailed("displayTexture")
        }

        let rasterParams = RasterizeParams(
            numTriangles: Int32(triangles.count),
            numVertices:  Int32(positions.count),
            width:        Int32(width),
            height:       Int32(height),
            xs: 2.0 / Float(width),
            xo: -1.0 + 1.0 / Float(width),
            ys: 2.0 / Float(height),
            yo: -1.0 + 1.0 / Float(height)
        )
        let rasterParamsBuffer = try makeParamBuffer(rasterParams, label: "rasterizeParams")

        let interpolateParams = InterpolateParams(
            numTriangles: Int32(triangles.count),
            numVertices:  Int32(positions.count),
            numAttr:      Int32(channels),
            width:        Int32(width),
            height:       Int32(height)
        )
        let interpolateParamsBuffer = try makeParamBuffer(
            interpolateParams,
            label: "interpolateParams"
        )

        let antialiasParams = AntialiasParams(
            numTriangles: Int32(triangles.count),
            numVertices:  Int32(positions.count),
            width:        Int32(width),
            height:       Int32(height),
            channels:     Int32(channels),
            xh: Float(width) * 0.5,
            yh: Float(height) * 0.5
        )
        let antialiasParamsBuffer = try makeParamBuffer(antialiasParams, label: "antialiasParams")

        let packParams = PackColorTextureParams(
            width: Int32(width),
            height: Int32(height),
            channels: Int32(channels)
        )
        let packParamsBuffer = try makeParamBuffer(packParams, label: "packColorParams")

        try runCommand { _, enc in
            let rasterPipeline = try self.pipeline("rasterizeKernel")
            enc.setBuffer(positionBuffer,      offset: 0, index: 0)
            enc.setBuffer(triangleBuffer,      offset: 0, index: 1)
            enc.setBuffer(rasterParamsBuffer,  offset: 0, index: 2)
            enc.setBuffer(rastBuffer,          offset: 0, index: 3)
            enc.setBuffer(rastDBBuffer,        offset: 0, index: 4)
            self.dispatch2D(encoder: enc, pipeline: rasterPipeline, width: width, height: height)

            let interpolatePipeline = try self.pipeline("interpolateKernel")
            enc.setBuffer(triangleBuffer,            offset: 0, index: 0)
            enc.setBuffer(colorBuffer,               offset: 0, index: 1)
            enc.setBuffer(rastBuffer,                offset: 0, index: 2)
            enc.setBuffer(interpolateParamsBuffer,   offset: 0, index: 3)
            enc.setBuffer(shadedBuffer,              offset: 0, index: 4)
            enc.setBuffer(baryBuffer,                offset: 0, index: 5)
            self.dispatch2D(encoder: enc, pipeline: interpolatePipeline, width: width, height: height)

            let outputColorBuffer: MTLBuffer
            if antialias {
                // Seed aaBuffer with shadedBuffer first (the AA kernel only
                // applies atomic-add corrections; the base colour pass-through
                // must happen before it, so the corrections compose correctly).
                let copyPipeline = try self.pipeline("copyFloatBufferKernel")
                let copyCount = Int32(width * height * channels)
                let copyParams = CopyParams(count: copyCount)
                let copyParamsBuffer = try self.makeParamBuffer(copyParams, label: "copyParams")
                enc.setComputePipelineState(copyPipeline)
                enc.setBuffer(shadedBuffer,    offset: 0, index: 0)
                enc.setBuffer(aaBuffer,        offset: 0, index: 1)
                enc.setBuffer(copyParamsBuffer, offset: 0, index: 2)
                let total = width * height * channels
                let tg = min(copyPipeline.maxTotalThreadsPerThreadgroup, 256)
                enc.dispatchThreads(
                    MTLSize(width: total, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)
                )

                let aaPipeline = try self.pipeline("antialiasKernel")
                enc.setBuffer(shadedBuffer,           offset: 0, index: 0)
                enc.setBuffer(rastBuffer,             offset: 0, index: 1)
                enc.setBuffer(triangleBuffer,         offset: 0, index: 2)
                enc.setBuffer(positionBuffer,         offset: 0, index: 3)
                enc.setBuffer(aaBuffer,               offset: 0, index: 4)
                enc.setBuffer(antialiasParamsBuffer,  offset: 0, index: 5)
                self.dispatch2D(encoder: enc, pipeline: aaPipeline, width: width, height: height)
                outputColorBuffer = aaBuffer
            } else {
                outputColorBuffer = shadedBuffer
            }

            let packPipeline = try self.pipeline("packColorTextureKernel")
            enc.setBuffer(outputColorBuffer, offset: 0, index: 0)
            enc.setBuffer(rastBuffer,        offset: 0, index: 1)
            enc.setBuffer(packParamsBuffer,  offset: 0, index: 2)
            enc.setTexture(texture, index: 0)
            self.dispatch2D(encoder: enc, pipeline: packPipeline, width: width, height: height)
        }

        return texture
    }

    // MARK: - Public API: antialias

    /// Applies silhouette-edge antialiasing to a colour image.
    ///
    /// The antialias pass detects silhouette edges — pixels where one side
    /// covers a triangle and the other does not — and blends the foreground
    /// and background colours by the sub-pixel edge crossing fraction.  This
    /// makes the loss differentiable with respect to vertex positions along
    /// occlusion boundaries.
    ///
    /// The output buffer is pre-filled with `color` (CPU-side) before the
    /// kernel runs.  The Metal kernel applies only the silhouette *corrections*
    /// via atomic float adds, avoiding the scatter-write race that would occur
    /// if the initialisation happened inside the kernel itself.
    ///
    /// Shared internal edges (two triangles sharing an edge) are skipped — only
    /// true silhouette edges produce corrections.
    ///
    /// - Parameters:
    ///   - color:        Input colour image, layout `[H × W × channels]`.
    ///   - channels:     Number of colour channels.  Must be ≥ 1.
    ///   - rasterOutput: Forward rasterizer output from the same frame.
    ///   - positions:    Clip-space vertex positions used to rasterize.
    ///   - triangles:    Triangle index list used to rasterize.
    ///
    /// - Returns: An ``AntialiasOutput`` whose `colors` array is in `[0, 1]`
    ///   for a `[0, 1]` input.
    ///
    /// - Throws: ``RasterizerError/encodingFailed(_:)`` for mismatched array
    ///   sizes.
    public func antialias(
        color: [Float],
        channels: Int,
        rasterOutput: RasterOutput,
        positions: [SIMD4<Float>],
        triangles: [SIMD3<Int32>]
    ) throws -> AntialiasOutput {
        let width = rasterOutput.width
        let height = rasterOutput.height
        let pixelCount = rasterOutput.pixelCount
        guard channels > 0 else {
            throw RasterizerError.encodingFailed("channels must be > 0")
        }
        guard color.count == pixelCount * channels else {
            throw RasterizerError.encodingFailed(
                "color length must be width * height * channels"
            )
        }

        var rastPacked = [SIMD4<Float>](repeating: SIMD4<Float>(0, 0, 0, 0), count: pixelCount)
        for i in 0..<pixelCount {
            let triPlusOne = rasterOutput.triangleIds[i] < 0
                ? Float(0)
                : encodeTriangleID(rasterOutput.triangleIds[i])
            let bary = rasterOutput.barycentrics[i]
            rastPacked[i] = SIMD4<Float>(bary.x, bary.y, rasterOutput.depthBuffer[i], triPlusOne)
        }

        let colorBuffer = try makeBuffer(color, label: "color")
        let rastBuffer  = try makeBuffer(rastPacked, label: "rastOut")
        let triBuffer   = try makeBuffer(triangles, label: "triangles")
        let posBuffer   = try makeBuffer(positions, label: "positions")
        // Seed the output buffer with `color` directly so the kernel only has
        // to apply silhouette corrections (atomic adds). Doing the seed inside
        // the kernel races with neighbour atomic adds.
        let outBuffer   = try makeBuffer(color, label: "aaOut")

        let params = AntialiasParams(
            numTriangles: Int32(triangles.count),
            numVertices:  Int32(positions.count),
            width:        Int32(width),
            height:       Int32(height),
            channels:     Int32(channels),
            xh: Float(width) * 0.5,
            yh: Float(height) * 0.5
        )
        let paramsBuffer = try makeParamBuffer(params, label: "antialiasParams")

        try runCommand { _, enc in
            let pipeline = try self.pipeline("antialiasKernel")
            enc.setBuffer(colorBuffer, offset: 0, index: 0)
            enc.setBuffer(rastBuffer,  offset: 0, index: 1)
            enc.setBuffer(triBuffer,   offset: 0, index: 2)
            enc.setBuffer(posBuffer,   offset: 0, index: 3)
            enc.setBuffer(outBuffer,   offset: 0, index: 4)
            enc.setBuffer(paramsBuffer, offset: 0, index: 5)
            self.dispatch2D(encoder: enc, pipeline: pipeline, width: width, height: height)
        }

        let outPtr = outBuffer.contents().bindMemory(to: Float.self, capacity: pixelCount * channels)
        let colors = Array(UnsafeBufferPointer(start: outPtr, count: pixelCount * channels))
        return AntialiasOutput(width: width, height: height, channels: channels, colors: colors)
    }

    /// Convenience overload that antialiases a binary coverage mask.
    ///
    /// Synthesises a 1-channel `covered / not-covered` image from the triangle
    /// IDs in `rasterOutput` and antialiases it.  Useful when you don't yet
    /// have an interpolated colour image but want differentiable silhouettes.
    ///
    /// - Parameters:
    ///   - rasterOutput: Forward rasterizer output.
    ///   - positions:    Clip-space vertex positions used to rasterize.
    ///   - triangles:    Triangle index list used to rasterize.
    ///
    /// - Returns: An ``AntialiasOutput`` with `channels == 1`.
    public func antialias(
        rasterOutput: RasterOutput,
        positions: [SIMD4<Float>],
        triangles: [SIMD3<Int32>]
    ) throws -> AntialiasOutput {
        var coverage = [Float](repeating: 0, count: rasterOutput.pixelCount)
        for i in 0..<rasterOutput.pixelCount {
            coverage[i] = rasterOutput.triangleIds[i] >= 0 ? 1.0 : 0.0
        }
        return try antialias(
            color: coverage,
            channels: 1,
            rasterOutput: rasterOutput,
            positions: positions,
            triangles: triangles
        )
    }

    /// Backward-compatible single-argument form that applies a CPU box-filter.
    ///
    /// This overload does not require vertex positions or triangle data.
    /// Instead it applies a 3×3 box filter over the binary coverage mask to
    /// produce a soft silhouette.  The result is **not** differentiable with
    /// respect to vertex positions; prefer
    /// ``antialias(color:channels:rasterOutput:positions:triangles:)`` for
    /// gradient-aware use.
    ///
    /// Kept for backward compatibility with callers that only hold a
    /// ``RasterOutput``.
    ///
    /// - Parameter rasterOutput: Forward rasterizer output.
    /// - Returns: An ``AntialiasOutput`` with `channels == 1`.
    public func antialias(rasterOutput: RasterOutput) throws -> AntialiasOutput {
        let width = rasterOutput.width
        let height = rasterOutput.height
        let pixelCount = rasterOutput.pixelCount
        var coverage = [Float](repeating: 0, count: pixelCount)
        for y in 0..<height {
            for x in 0..<width {
                var sum: Float = 0
                var n: Float = 0
                for dy in -1...1 {
                    for dx in -1...1 {
                        let nx = x + dx, ny = y + dy
                        if nx < 0 || nx >= width || ny < 0 || ny >= height { continue }
                        let idx = ny * width + nx
                        sum += rasterOutput.triangleIds[idx] >= 0 ? 1.0 : 0.0
                        n += 1
                    }
                }
                coverage[y * width + x] = n > 0 ? sum / n : 0
            }
        }
        return AntialiasOutput(width: width, height: height, channels: 1, colors: coverage)
    }

    // MARK: - Public API: texture sampling

    /// Samples a 2-D texture at per-pixel UV coordinates.
    ///
    /// - Parameters:
    ///   - texture:      Texel data, layout `[texHeight × texWidth × channels]`.
    ///   - texWidth:     Texture width in texels.
    ///   - texHeight:    Texture height in texels.
    ///   - channels:     Number of channels per texel.
    ///   - uv:           Per-output-pixel UV coordinates, layout
    ///     `[outHeight × outWidth]`.
    ///   - outWidth:     Output image width in pixels.
    ///   - outHeight:    Output image height in pixels.
    ///   - filterMode:   ``TextureFilterMode/nearest`` or
    ///     ``TextureFilterMode/linear``.  Defaults to `.linear`.
    ///   - boundaryMode: ``TextureBoundaryMode/wrap``,
    ///     ``TextureBoundaryMode/clamp``, or ``TextureBoundaryMode/zero``.
    ///     Defaults to `.wrap`.
    ///
    /// - Returns: A ``TextureOutput`` containing sampled values of shape
    ///   `[outHeight × outWidth × channels]`.
    ///
    /// - Throws: ``RasterizerError/encodingFailed(_:)`` for size mismatches;
    ///   ``RasterizerError/invalidResolution(width:height:)`` for zero
    ///   dimensions.
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
    ) throws -> TextureOutput {
        guard texWidth > 0, texHeight > 0, channels > 0 else {
            throw RasterizerError.invalidResolution(width: texWidth, height: texHeight)
        }
        guard outWidth > 0, outHeight > 0 else {
            throw RasterizerError.invalidResolution(width: outWidth, height: outHeight)
        }
        guard texture.count == texWidth * texHeight * channels else {
            throw RasterizerError.encodingFailed("texture array size mismatch")
        }
        guard uv.count == outWidth * outHeight else {
            throw RasterizerError.encodingFailed("uv array size mismatch")
        }

        let texBuffer = try makeBuffer(texture, label: "texture")
        let uvBuffer  = try makeBuffer(uv, label: "uv")
        let outBuffer = try makeZeroBuffer(
            byteCount: outWidth * outHeight * channels * MemoryLayout<Float>.stride,
            label: "texOut"
        )

        let params = TextureParams(
            filterMode:   filterMode.rawValue,
            boundaryMode: boundaryMode.rawValue,
            channels:     Int32(channels),
            imgWidth:     Int32(outWidth),
            imgHeight:    Int32(outHeight),
            texWidth:     Int32(texWidth),
            texHeight:    Int32(texHeight)
        )
        let paramsBuffer = try makeParamBuffer(params, label: "textureParams")

        try runCommand { _, enc in
            let pipeline = try self.pipeline("textureKernel")
            enc.setBuffer(texBuffer,    offset: 0, index: 0)
            enc.setBuffer(uvBuffer,     offset: 0, index: 1)
            enc.setBuffer(paramsBuffer, offset: 0, index: 2)
            enc.setBuffer(outBuffer,    offset: 0, index: 3)
            self.dispatch2D(encoder: enc, pipeline: pipeline, width: outWidth, height: outHeight)
        }

        let outPtr  = outBuffer.contents().bindMemory(to: Float.self, capacity: outWidth * outHeight * channels)
        let samples = Array(UnsafeBufferPointer(start: outPtr, count: outWidth * outHeight * channels))
        return TextureOutput(width: outWidth, height: outHeight, channels: channels, samples: samples)
    }

    /// Computes gradients for the texture sampling backward pass.
    ///
    /// - Parameters:
    ///   - texture:      The same texel data passed to the forward call,
    ///     layout `[texHeight × texWidth × channels]`.
    ///   - texWidth:     Texture width in texels.
    ///   - texHeight:    Texture height in texels.
    ///   - channels:     Number of channels per texel.
    ///   - uv:           The same UV coordinates passed to the forward call.
    ///   - gradOutput:   Upstream gradient for the sampled output, layout
    ///     `[outHeight × outWidth × channels]`.
    ///   - outWidth:     Output image width.
    ///   - outHeight:    Output image height.
    ///   - filterMode:   Must match the forward call.  Defaults to `.linear`.
    ///   - boundaryMode: Must match the forward call.  Defaults to `.wrap`.
    ///
    /// - Returns: A ``TextureGradientOutput`` containing gradients w.r.t.
    ///   texture texels (all filter modes) and UV coordinates (linear only;
    ///   nearest always returns zero UV gradients).
    ///
    /// - Throws: ``RasterizerError/encodingFailed(_:)`` for size mismatches.
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
    ) throws -> TextureGradientOutput {
        guard texWidth > 0, texHeight > 0, channels > 0 else {
            throw RasterizerError.invalidResolution(width: texWidth, height: texHeight)
        }
        guard outWidth > 0, outHeight > 0 else {
            throw RasterizerError.invalidResolution(width: outWidth, height: outHeight)
        }
        guard texture.count == texWidth * texHeight * channels else {
            throw RasterizerError.encodingFailed("texture array size mismatch")
        }
        guard uv.count == outWidth * outHeight else {
            throw RasterizerError.encodingFailed("uv array size mismatch")
        }
        guard gradOutput.count == outWidth * outHeight * channels else {
            throw RasterizerError.encodingFailed("gradOutput array size mismatch")
        }

        let texBuffer = try makeBuffer(texture, label: "texture")
        let uvBuffer  = try makeBuffer(uv, label: "uv")
        let dyBuffer  = try makeBuffer(gradOutput, label: "textureDy")
        let gradTexBuffer = try makeZeroBuffer(
            byteCount: texWidth * texHeight * channels * MemoryLayout<Float>.stride,
            label: "gradTexture"
        )
        let gradUVBuffer = try makeZeroBuffer(
            byteCount: outWidth * outHeight * MemoryLayout<SIMD2<Float>>.stride,
            label: "gradUV"
        )

        let params = TextureParams(
            filterMode:   filterMode.rawValue,
            boundaryMode: boundaryMode.rawValue,
            channels:     Int32(channels),
            imgWidth:     Int32(outWidth),
            imgHeight:    Int32(outHeight),
            texWidth:     Int32(texWidth),
            texHeight:    Int32(texHeight)
        )
        let paramsBuffer = try makeParamBuffer(params, label: "textureParams")

        try runCommand { _, enc in
            let pipeline = try self.pipeline("textureBackwardKernel")
            enc.setBuffer(texBuffer,      offset: 0, index: 0)
            enc.setBuffer(uvBuffer,       offset: 0, index: 1)
            enc.setBuffer(dyBuffer,       offset: 0, index: 2)
            enc.setBuffer(paramsBuffer,   offset: 0, index: 3)
            enc.setBuffer(gradTexBuffer,  offset: 0, index: 4)
            enc.setBuffer(gradUVBuffer,   offset: 0, index: 5)
            self.dispatch2D(encoder: enc, pipeline: pipeline, width: outWidth, height: outHeight)
        }

        let gradTexPtr = gradTexBuffer.contents().bindMemory(
            to: Float.self,
            capacity: texWidth * texHeight * channels
        )
        let gradUVPtr = gradUVBuffer.contents().bindMemory(
            to: SIMD2<Float>.self,
            capacity: outWidth * outHeight
        )

        let gradTexture = Array(UnsafeBufferPointer(start: gradTexPtr, count: texWidth * texHeight * channels))
        let gradUV      = Array(UnsafeBufferPointer(start: gradUVPtr,  count: outWidth * outHeight))
        return TextureGradientOutput(textureGradients: gradTexture, uvGradients: gradUV)
    }
}
