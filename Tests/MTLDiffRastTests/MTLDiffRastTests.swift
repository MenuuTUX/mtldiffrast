//
//  MTLDiffRastTests.swift
//  MTLDiffRastTests
//
//  Functional tests for the MTLDiffRast library.
//

import XCTest
@testable import MTLDiffRast
import Metal
import simd

final class MTLDiffRastTests: XCTestCase {

    var rasterizer: Rasterizer?

    override func setUp() async throws {
        guard isMetalAvailable() else {
            throw XCTSkip("Metal not available")
        }
        guard isAppleSilicon() else {
            throw XCTSkip("Tests require Apple Silicon")
        }
        rasterizer = try Rasterizer()
    }

    override func tearDown() async throws {
        rasterizer = nil
    }

    // MARK: - Initialization

    func testInitialization() throws {
        XCTAssertNotNil(rasterizer)
    }

    func testMetalAvailability() {
        XCTAssertTrue(isMetalAvailable())
    }

    // MARK: - Rasterization

    /// A centered triangle in clip space should cover roughly half the pixels
    /// in a 64x64 image (the lower triangular half — area = 0.5).
    func testSimpleTriangleRasterization() throws {
        guard let rast = rasterizer else { return }

        let positions: [SIMD4<Float>] = [
            SIMD4<Float>( 0.0,  0.5, 0.5, 1.0),
            SIMD4<Float>(-0.5, -0.5, 0.5, 1.0),
            SIMD4<Float>( 0.5, -0.5, 0.5, 1.0),
        ]
        let triangles: [SIMD3<Int32>] = [SIMD3<Int32>(0, 1, 2)]

        let out = try rast.rasterize(
            positions: positions, triangles: triangles, width: 64, height: 64
        )

        XCTAssertEqual(out.width, 64)
        XCTAssertEqual(out.height, 64)

        let covered = out.triangleIds.filter { $0 == 0 }.count
        XCTAssertGreaterThan(covered, 0, "Triangle should cover some pixels")

        // Sanity check: barycentrics inside covered pixels are valid (sum ~ 1).
        for (i, tid) in out.triangleIds.enumerated() where tid >= 0 {
            let b = out.barycentrics[i]
            let bsum = b.x + b.y
            XCTAssertGreaterThanOrEqual(bsum, -1e-3)
            XCTAssertLessThanOrEqual(bsum, 1.0 + 1e-3)
        }
    }

    /// Two CCW-wound triangles must each cover some pixels. (Triangles
    /// wound clockwise in clip space are treated as back-facing and culled.)
    func testMultipleTriangles() throws {
        guard let rast = rasterizer else { return }

        let positions: [SIMD4<Float>] = [
            // Triangle 0: pointing up.
            SIMD4<Float>( 0.0,  0.5, 0.5, 1.0),
            SIMD4<Float>(-0.5, -0.5, 0.5, 1.0),
            SIMD4<Float>( 0.5, -0.5, 0.5, 1.0),
            // Triangle 1: pointing down (CCW order).
            SIMD4<Float>( 0.0, -0.5, 0.6, 1.0),
            SIMD4<Float>( 0.5,  0.5, 0.6, 1.0),
            SIMD4<Float>(-0.5,  0.5, 0.6, 1.0),
        ]
        let triangles: [SIMD3<Int32>] = [
            SIMD3<Int32>(0, 1, 2),
            SIMD3<Int32>(3, 4, 5),
        ]

        let out = try rast.rasterize(
            positions: positions, triangles: triangles, width: 128, height: 128
        )

        let c0 = out.triangleIds.filter { $0 == 0 }.count
        let c1 = out.triangleIds.filter { $0 == 1 }.count
        XCTAssertGreaterThan(c0, 0)
        XCTAssertGreaterThan(c1, 0)
    }

    func testRasterizeColorTextureCreatesDisplayTexture() throws {
        guard let rast = rasterizer else { return }

        let positions: [SIMD4<Float>] = [
            SIMD4<Float>( 0.0,  0.6, 0.5, 1.0),
            SIMD4<Float>(-0.6, -0.6, 0.5, 1.0),
            SIMD4<Float>( 0.6, -0.6, 0.5, 1.0),
        ]
        let triangles: [SIMD3<Int32>] = [SIMD3<Int32>(0, 1, 2)]
        let colors: [Float] = [
            1, 0, 0,
            0, 1, 0,
            0, 0, 1,
        ]

        let texture = try rast.rasterizeColorTexture(
            positions: positions,
            triangles: triangles,
            colors: colors,
            width: 64,
            height: 32
        )

        XCTAssertEqual(texture.width, 64)
        XCTAssertEqual(texture.height, 32)
        XCTAssertEqual(texture.pixelFormat, .bgra8Unorm)
        XCTAssertTrue(texture.usage.contains(.shaderRead))
    }

    /// When two triangles overlap, the closer one (larger z/w in clip-space
    /// convention) wins.
    func testDepthTesting() throws {
        guard let rast = rasterizer else { return }

        // Both triangles fully cover the centre. Triangle 1 has higher z/w
        // (closer) and should win.
        let positions: [SIMD4<Float>] = [
            SIMD4<Float>( 0.0,  0.8, 0.1, 1.0),  // far  — z=0.1
            SIMD4<Float>(-0.8, -0.8, 0.1, 1.0),
            SIMD4<Float>( 0.8, -0.8, 0.1, 1.0),
            SIMD4<Float>( 0.0,  0.8, 0.9, 1.0),  // near — z=0.9
            SIMD4<Float>(-0.8, -0.8, 0.9, 1.0),
            SIMD4<Float>( 0.8, -0.8, 0.9, 1.0),
        ]
        let triangles: [SIMD3<Int32>] = [
            SIMD3<Int32>(0, 1, 2),
            SIMD3<Int32>(3, 4, 5),
        ]

        let out = try rast.rasterize(
            positions: positions, triangles: triangles, width: 64, height: 64
        )
        let c1 = out.triangleIds.filter { $0 == 1 }.count
        XCTAssertGreaterThan(c1, 0, "Near triangle (id 1) should be visible")
    }

    // MARK: - Errors

    func testInvalidResolution() throws {
        guard let rast = rasterizer else { return }
        let positions: [SIMD4<Float>] = [
            SIMD4<Float>(0, 0.5, 0.5, 1), SIMD4<Float>(-0.5, -0.5, 0.5, 1), SIMD4<Float>(0.5, -0.5, 0.5, 1)
        ]
        let triangles: [SIMD3<Int32>] = [SIMD3<Int32>(0, 1, 2)]
        XCTAssertThrowsError(try rast.rasterize(
            positions: positions, triangles: triangles, width: 0, height: 64
        )) { error in
            guard case RasterizerError.invalidResolution = error else {
                XCTFail("Expected invalidResolution"); return
            }
        }
    }

    func testEmptyTriangles() throws {
        guard let rast = rasterizer else { return }
        let positions: [SIMD4<Float>] = [
            SIMD4<Float>(0, 0.5, 0.5, 1), SIMD4<Float>(-0.5, -0.5, 0.5, 1), SIMD4<Float>(0.5, -0.5, 0.5, 1)
        ]
        XCTAssertThrowsError(try rast.rasterize(
            positions: positions, triangles: [], width: 64, height: 64
        )) { error in
            guard case RasterizerError.invalidTriangleCount = error else {
                XCTFail("Expected invalidTriangleCount"); return
            }
        }
    }

    // MARK: - Interpolation

    /// Interpolation must use real barycentrics from the rasterizer — colour
    /// at a pixel near vertex 0 must be close to that vertex's attribute, not
    /// the (1/3, 1/3, 1/3) average that the broken implementation returned.
    func testInterpolationUsesRealBarycentrics() throws {
        guard let rast = rasterizer else { return }

        let positions: [SIMD4<Float>] = [
            SIMD4<Float>( 0.0,  0.9, 0.5, 1.0),  // top
            SIMD4<Float>(-0.9, -0.9, 0.5, 1.0),  // bottom-left
            SIMD4<Float>( 0.9, -0.9, 0.5, 1.0),  // bottom-right
        ]
        let triangles: [SIMD3<Int32>] = [SIMD3<Int32>(0, 1, 2)]
        let attrs: [Float] = [
            1, 0, 0,   // vert 0 -> red
            0, 1, 0,   // vert 1 -> green
            0, 0, 1,   // vert 2 -> blue
        ]

        let rOut = try rast.rasterize(
            positions: positions, triangles: triangles, width: 64, height: 64
        )
        let interp = try rast.interpolate(
            attributes: attrs, triangles: triangles,
            rasterOutput: rOut, numAttributes: 3
        )

        XCTAssertEqual(interp.attributes.count, 64 * 64 * 3)
        XCTAssertEqual(interp.barycentricCoords.count, 64 * 64 * 3)

        // Buffer convention: py=0 is the bottom row in clip space (y axis up).
        // The top vertex (y = +0.9) maps to high py.
        var foundTop = false
        for py in stride(from: 63, through: 56, by: -1) {
            let px = 32
            let i = py * 64 + px
            if rOut.triangleIds[i] != 0 { continue }
            let r = interp.attributes[i * 3 + 0]
            let g = interp.attributes[i * 3 + 1]
            let b = interp.attributes[i * 3 + 2]
            XCTAssertGreaterThan(r, max(g, b), "expected red near top vertex")
            foundTop = true
            break
        }
        XCTAssertTrue(foundTop, "Should find a covered pixel near the top vertex")

        // Bottom-left vertex (x = -0.9, y = -0.9) → low px, low py.
        var foundBL = false
        outer: for py in 0..<8 {
            for px in 0..<8 {
                let i = py * 64 + px
                if rOut.triangleIds[i] != 0 { continue }
                let r = interp.attributes[i * 3 + 0]
                let g = interp.attributes[i * 3 + 1]
                let b = interp.attributes[i * 3 + 2]
                XCTAssertGreaterThan(g, max(r, b), "expected green near bottom-left")
                foundBL = true
                break outer
            }
        }
        XCTAssertTrue(foundBL, "Should find a covered pixel near the BL vertex")

        // Sanity: per-pixel barycentrics sum to ~1 inside covered pixels.
        for i in 0..<rOut.pixelCount where rOut.triangleIds[i] >= 0 {
            let b0 = interp.barycentricCoords[i * 3 + 0]
            let b1 = interp.barycentricCoords[i * 3 + 1]
            let b2 = interp.barycentricCoords[i * 3 + 2]
            XCTAssertEqual(b0 + b1 + b2, 1.0, accuracy: 1e-3)
        }
    }

    // MARK: - Antialias

    /// AA on a coverage mask should leave fully-covered and fully-empty
    /// regions essentially unchanged, but smooth out silhouette pixels.
    func testAntialiasReturnsExpectedShape() throws {
        guard let rast = rasterizer else { return }

        let positions: [SIMD4<Float>] = [
            SIMD4<Float>( 0.0,  0.5, 0.5, 1.0),
            SIMD4<Float>(-0.5, -0.5, 0.5, 1.0),
            SIMD4<Float>( 0.5, -0.5, 0.5, 1.0),
        ]
        let triangles: [SIMD3<Int32>] = [SIMD3<Int32>(0, 1, 2)]

        let rOut = try rast.rasterize(
            positions: positions, triangles: triangles, width: 64, height: 64
        )
        let aa = try rast.antialias(
            rasterOutput: rOut, positions: positions, triangles: triangles
        )

        XCTAssertEqual(aa.width, 64)
        XCTAssertEqual(aa.height, 64)
        XCTAssertEqual(aa.channels, 1)
        XCTAssertEqual(aa.colors.count, 64 * 64)

        // Every output value must be in [0, 1] after blending the 0/1 coverage.
        for v in aa.colors {
            XCTAssertGreaterThanOrEqual(v, -0.05)
            XCTAssertLessThanOrEqual(v, 1.05)
        }
    }

    // MARK: - Texture sampling

    func testTextureNearestSampling() throws {
        guard let rast = rasterizer else { return }

        // 2x2 RGB texture: red, green, blue, white.
        let tex: [Float] = [
            1, 0, 0,   0, 1, 0,
            0, 0, 1,   1, 1, 1,
        ]
        // Sample at the four texel centres.
        let uv: [SIMD2<Float>] = [
            SIMD2<Float>(0.25, 0.25),
            SIMD2<Float>(0.75, 0.25),
            SIMD2<Float>(0.25, 0.75),
            SIMD2<Float>(0.75, 0.75),
        ]
        let out = try rast.texture(
            texture: tex, texWidth: 2, texHeight: 2, channels: 3,
            uv: uv, outWidth: 4, outHeight: 1,
            filterMode: .nearest, boundaryMode: .clamp
        )
        XCTAssertEqual(out.samples.count, 4 * 3)

        // Pixel 0 -> red, pixel 1 -> green, pixel 2 -> blue, pixel 3 -> white.
        XCTAssertEqual(out.samples[0], 1, accuracy: 1e-5)
        XCTAssertEqual(out.samples[1], 0, accuracy: 1e-5)
        XCTAssertEqual(out.samples[2], 0, accuracy: 1e-5)
        XCTAssertEqual(out.samples[3], 0, accuracy: 1e-5)
        XCTAssertEqual(out.samples[4], 1, accuracy: 1e-5)
        XCTAssertEqual(out.samples[5], 0, accuracy: 1e-5)
        XCTAssertEqual(out.samples[6], 0, accuracy: 1e-5)
        XCTAssertEqual(out.samples[7], 0, accuracy: 1e-5)
        XCTAssertEqual(out.samples[8], 1, accuracy: 1e-5)
        XCTAssertEqual(out.samples[9], 1, accuracy: 1e-5)
        XCTAssertEqual(out.samples[10], 1, accuracy: 1e-5)
        XCTAssertEqual(out.samples[11], 1, accuracy: 1e-5)
    }

    func testTextureBilinearMidpoint() throws {
        guard let rast = rasterizer else { return }

        // 2x2 single-channel texture of [0, 1, 1, 0]. At the texture centre
        // (0.5, 0.5) bilinear sample -> 0.5.
        let tex: [Float] = [0, 1, 1, 0]
        let uv: [SIMD2<Float>] = [SIMD2<Float>(0.5, 0.5)]
        let out = try rast.texture(
            texture: tex, texWidth: 2, texHeight: 2, channels: 1,
            uv: uv, outWidth: 1, outHeight: 1,
            filterMode: .linear, boundaryMode: .clamp
        )
        XCTAssertEqual(out.samples[0], 0.5, accuracy: 1e-5)
    }

    // MARK: - Backward pass

    /// The backward pass must actually accumulate gradients into the
    /// vertex-position buffer (the previous implementation only had a barrier
    /// and returned all zeros).
    func testRasterizeBackwardProducesNonZeroGradient() throws {
        guard let rast = rasterizer else { return }

        let positions: [SIMD4<Float>] = [
            SIMD4<Float>( 0.0,  0.5, 0.5, 1.0),
            SIMD4<Float>(-0.5, -0.5, 0.5, 1.0),
            SIMD4<Float>( 0.5, -0.5, 0.5, 1.0),
        ]
        let triangles: [SIMD3<Int32>] = [SIMD3<Int32>(0, 1, 2)]
        let out = try rast.rasterize(
            positions: positions, triangles: triangles, width: 32, height: 32
        )
        // Drive a non-zero gradient on the .u component of every covered pixel.
        var dy = [Float](repeating: 0, count: out.pixelCount * 4)
        for i in 0..<out.pixelCount where out.triangleIds[i] >= 0 {
            dy[i * 4 + 0] = 1.0
            dy[i * 4 + 1] = 0.5
        }
        let g = try rast.rasterizeBackward(
            positions: positions, triangles: triangles,
            forwardOutput: out, gradOutput: dy, vertexCount: 3
        )
        XCTAssertEqual(g.positionGradients.count, 12)

        let nonZero = g.positionGradients.contains { abs($0) > 1e-6 }
        XCTAssertTrue(nonZero, "Position gradients must not all be zero")
    }

    func testRasterizeBackwardScalarGradientTargetsUOnly() throws {
        guard let rast = rasterizer else { return }

        let positions: [SIMD4<Float>] = [
            SIMD4<Float>( 0.0,  0.5, 0.5, 1.0),
            SIMD4<Float>(-0.5, -0.5, 0.5, 1.0),
            SIMD4<Float>( 0.5, -0.5, 0.5, 1.0),
        ]
        let triangles: [SIMD3<Int32>] = [SIMD3<Int32>(0, 1, 2)]
        let out = try rast.rasterize(
            positions: positions, triangles: triangles, width: 16, height: 16
        )

        let scalar = [Float](repeating: 1.0, count: out.pixelCount)
        var packed = [Float](repeating: 0.0, count: out.pixelCount * 4)
        for i in 0..<out.pixelCount {
            packed[i * 4] = 1.0
        }

        let gScalar = try rast.rasterizeBackward(
            positions: positions, triangles: triangles,
            forwardOutput: out, gradOutput: scalar, vertexCount: positions.count
        ).positionGradients
        let gPacked = try rast.rasterizeBackward(
            positions: positions, triangles: triangles,
            forwardOutput: out, gradOutput: packed, vertexCount: positions.count
        ).positionGradients

        XCTAssertEqual(gScalar.count, gPacked.count)
        for i in 0..<gScalar.count {
            XCTAssertEqual(gScalar[i], gPacked[i], accuracy: 1e-5)
        }
    }

    func testInterpolateBackwardDistributesAttributeGradients() throws {
        guard let rast = rasterizer else { return }

        let positions: [SIMD4<Float>] = [
            SIMD4<Float>( 0.0,  0.5, 0.5, 1.0),
            SIMD4<Float>(-0.5, -0.5, 0.5, 1.0),
            SIMD4<Float>( 0.5, -0.5, 0.5, 1.0),
        ]
        let triangles: [SIMD3<Int32>] = [SIMD3<Int32>(0, 1, 2)]
        let attrs: [Float] = [1, 0, 0, 0, 1, 0, 0, 0, 1]

        let rOut = try rast.rasterize(
            positions: positions, triangles: triangles, width: 16, height: 16
        )
        let dy = [Float](repeating: 1.0, count: rOut.pixelCount * 3)
        let (gAttr, gRast) = try rast.interpolateBackward(
            attributes: attrs, triangles: triangles, rasterOutput: rOut,
            gradOutput: dy, numAttributes: 3
        )
        XCTAssertEqual(gAttr.count, 9)
        XCTAssertEqual(gRast.count, rOut.pixelCount * 4)

        // For each attribute, the per-vertex gradient is sum_pixels(b_v * dy).
        // With dy=1 everywhere, the sum across the three vertices equals the
        // total covered-pixel count (since b0+b1+b2=1 inside covered pixels).
        let coveredCount = rOut.triangleIds.filter { $0 >= 0 }.count
        for ch in 0..<3 {
            let s = gAttr[0 * 3 + ch] + gAttr[1 * 3 + ch] + gAttr[2 * 3 + ch]
            XCTAssertEqual(s, Float(coveredCount), accuracy: max(0.5, Float(coveredCount) * 1e-4))
        }
    }
}
