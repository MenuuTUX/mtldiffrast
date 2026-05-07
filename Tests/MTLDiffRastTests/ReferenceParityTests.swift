//
//  ReferenceParityTests.swift
//  MTLDiffRastTests
//
//  CPU reference and finite-difference checks for the Metal kernels.
//

import XCTest
@testable import MTLDiffRast
import simd

final class ReferenceParityTests: XCTestCase {

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

    func testRasterizeMatchesCPUReference() throws {
        guard let rast = rasterizer else { return }
        let positions: [SIMD4<Float>] = [
            SIMD4<Float>(-0.75, -0.65, 0.25, 1.0),
            SIMD4<Float>( 0.15, -0.55, 0.25, 1.0),
            SIMD4<Float>(-0.25,  0.75, 0.25, 1.0),
            SIMD4<Float>( 0.20, -0.70, 0.80, 1.0),
            SIMD4<Float>( 0.85, -0.35, 0.80, 1.0),
            SIMD4<Float>( 0.55,  0.70, 0.80, 1.0),
        ]
        let triangles: [SIMD3<Int32>] = [
            SIMD3<Int32>(0, 1, 2),
            SIMD3<Int32>(3, 4, 5),
        ]

        let gpu = try rast.rasterize(
            positions: positions, triangles: triangles, width: 32, height: 24
        )
        let cpu = Self.cpuRasterize(
            positions: positions, triangles: triangles, width: 32, height: 24
        )

        var idMismatches = 0
        for i in 0..<gpu.pixelCount where gpu.triangleIds[i] != cpu.triangleIds[i] {
            idMismatches += 1
        }
        XCTAssertLessThanOrEqual(idMismatches, 2, "Only boundary pixels may differ")

        for i in 0..<gpu.pixelCount where gpu.triangleIds[i] >= 0 && gpu.triangleIds[i] == cpu.triangleIds[i] {
            XCTAssertEqual(gpu.barycentrics[i].x, cpu.barycentrics[i].x, accuracy: 2e-5)
            XCTAssertEqual(gpu.barycentrics[i].y, cpu.barycentrics[i].y, accuracy: 2e-5)
            XCTAssertEqual(gpu.depthBuffer[i], cpu.depthBuffer[i], accuracy: 2e-5)
        }
    }

    func testInterpolateDerivativesMatchNeighborDifferences() throws {
        guard let rast = rasterizer else { return }
        let positions: [SIMD4<Float>] = [
            SIMD4<Float>( 0.0,  0.9, 0.5, 1.0),
            SIMD4<Float>(-0.9, -0.9, 0.5, 1.0),
            SIMD4<Float>( 0.9, -0.9, 0.5, 1.0),
        ]
        let triangles: [SIMD3<Int32>] = [SIMD3<Int32>(0, 1, 2)]
        let attrs: [Float] = [2, 5, 11]
        let out = try rast.rasterize(
            positions: positions, triangles: triangles, width: 64, height: 64
        )
        let interp = try rast.interpolate(
            attributes: attrs, triangles: triangles, rasterOutput: out,
            numAttributes: 1, computeDerivatives: true
        )
        guard let derivs = interp.attributeDerivatives else {
            XCTFail("Expected attribute derivatives")
            return
        }

        let px = 32
        let py = 28
        let i = py * 64 + px
        XCTAssertEqual(out.triangleIds[i], 0)
        XCTAssertEqual(out.triangleIds[i + 1], 0)
        XCTAssertEqual(out.triangleIds[i + 64], 0)

        let dxFinite = interp.attributes[i + 1] - interp.attributes[i]
        let dyFinite = interp.attributes[i + 64] - interp.attributes[i]
        XCTAssertEqual(derivs[i * 2 + 0], dxFinite, accuracy: 2e-4)
        XCTAssertEqual(derivs[i * 2 + 1], dyFinite, accuracy: 2e-4)
    }

    func testRasterizeBackwardIncludesBaryDerivativeGradients() throws {
        guard let rast = rasterizer else { return }
        let positions: [SIMD4<Float>] = [
            SIMD4<Float>( 0.0,  0.7, 0.5, 1.0),
            SIMD4<Float>(-0.7, -0.7, 0.5, 1.0),
            SIMD4<Float>( 0.7, -0.7, 0.5, 1.0),
        ]
        let triangles: [SIMD3<Int32>] = [SIMD3<Int32>(0, 1, 2)]
        let out = try rast.rasterize(
            positions: positions, triangles: triangles, width: 24, height: 24
        )
        let dy = [Float](repeating: 0, count: out.pixelCount * 4)
        var ddb = [Float](repeating: 0, count: out.pixelCount * 4)
        for i in 0..<out.pixelCount where out.triangleIds[i] >= 0 {
            ddb[i * 4 + 0] = 0.25
            ddb[i * 4 + 1] = -0.5
            ddb[i * 4 + 2] = 0.75
            ddb[i * 4 + 3] = 0.125
        }
        let withoutDB = try rast.rasterizeBackward(
            positions: positions, triangles: triangles,
            forwardOutput: out, gradOutput: dy, vertexCount: positions.count
        ).positionGradients
        let withDB = try rast.rasterizeBackward(
            positions: positions, triangles: triangles,
            forwardOutput: out, gradOutput: dy, vertexCount: positions.count,
            gradBaryDerivatives: ddb
        ).positionGradients

        XCTAssertTrue(withoutDB.allSatisfy { abs($0) < 1e-6 })
        XCTAssertTrue(withDB.contains { abs($0) > 1e-5 })
    }

    func testTextureBackwardLinearMatchesFiniteDifferences() throws {
        guard let rast = rasterizer else { return }
        let texture: [Float] = [0, 2, 1, 5]
        let uv = [SIMD2<Float>(0.5, 0.5)]
        let dy: [Float] = [1]

        let grad = try rast.textureBackward(
            texture: texture, texWidth: 2, texHeight: 2, channels: 1,
            uv: uv, gradOutput: dy, outWidth: 1, outHeight: 1,
            filterMode: .linear, boundaryMode: .clamp
        )

        for g in grad.textureGradients {
            XCTAssertEqual(g, 0.25, accuracy: 1e-5)
        }
        XCTAssertEqual(grad.uvGradients[0].x, 6.0, accuracy: 1e-5)
        XCTAssertEqual(grad.uvGradients[0].y, 4.0, accuracy: 1e-5)

        let eps: Float = 1e-3
        let fxPlus = try sampleTextureScalar(rast, texture, SIMD2<Float>(0.5 + eps, 0.5))
        let fxMinus = try sampleTextureScalar(rast, texture, SIMD2<Float>(0.5 - eps, 0.5))
        let fyPlus = try sampleTextureScalar(rast, texture, SIMD2<Float>(0.5, 0.5 + eps))
        let fyMinus = try sampleTextureScalar(rast, texture, SIMD2<Float>(0.5, 0.5 - eps))
        XCTAssertEqual((fxPlus - fxMinus) / (2 * eps), grad.uvGradients[0].x, accuracy: 2e-3)
        XCTAssertEqual((fyPlus - fyMinus) / (2 * eps), grad.uvGradients[0].y, accuracy: 2e-3)
    }

    func testTextureBackwardNearestAccumulation() throws {
        guard let rast = rasterizer else { return }
        let texture = [Float](repeating: 0, count: 2 * 2 * 2)
        let uv = [
            SIMD2<Float>(0.25, 0.25),
            SIMD2<Float>(0.26, 0.24),
            SIMD2<Float>(0.75, 0.75),
        ]
        let dy: [Float] = [1, 2, 3, 4, 5, 6]

        let grad = try rast.textureBackward(
            texture: texture, texWidth: 2, texHeight: 2, channels: 2,
            uv: uv, gradOutput: dy, outWidth: 3, outHeight: 1,
            filterMode: .nearest, boundaryMode: .clamp
        )

        XCTAssertEqual(grad.textureGradients[0], 4, accuracy: 1e-5)
        XCTAssertEqual(grad.textureGradients[1], 6, accuracy: 1e-5)
        XCTAssertEqual(grad.textureGradients[6], 5, accuracy: 1e-5)
        XCTAssertEqual(grad.textureGradients[7], 6, accuracy: 1e-5)
        XCTAssertTrue(grad.uvGradients.allSatisfy { $0 == SIMD2<Float>(0, 0) })
    }

    func testAntialiasDoesNotBlendSharedInternalEdges() throws {
        guard let rast = rasterizer else { return }
        let positions: [SIMD4<Float>] = [
            SIMD4<Float>(-0.8, -0.8, 0.5, 1.0),
            SIMD4<Float>( 0.8, -0.8, 0.5, 1.0),
            SIMD4<Float>( 0.8,  0.8, 0.5, 1.0),
            SIMD4<Float>(-0.8,  0.8, 0.5, 1.0),
        ]
        let triangles: [SIMD3<Int32>] = [
            SIMD3<Int32>(0, 1, 2),
            SIMD3<Int32>(0, 2, 3),
        ]
        let attrs: [Float] = [
            1, 0, 0,
            1, 0, 0,
            0, 0, 1,
            0, 0, 1,
        ]

        let out = try rast.rasterize(
            positions: positions, triangles: triangles, width: 64, height: 64
        )
        let interp = try rast.interpolate(
            attributes: attrs, triangles: triangles, rasterOutput: out, numAttributes: 3
        )
        let aa = try rast.antialias(
            color: interp.attributes, channels: 3,
            rasterOutput: out, positions: positions, triangles: triangles
        )

        var checked = 0
        for y in 8..<56 {
            for x in 8..<56 {
                let i = y * 64 + x
                guard out.triangleIds[i] >= 0 else { continue }
                let right = i + 1
                let down = i + 64
                if out.triangleIds[right] >= 0, out.triangleIds[right] != out.triangleIds[i] {
                    assertColorUnchanged(aa.colors, interp.attributes, at: i, channels: 3)
                    assertColorUnchanged(aa.colors, interp.attributes, at: right, channels: 3)
                    checked += 1
                }
                if out.triangleIds[down] >= 0, out.triangleIds[down] != out.triangleIds[i] {
                    assertColorUnchanged(aa.colors, interp.attributes, at: i, channels: 3)
                    assertColorUnchanged(aa.colors, interp.attributes, at: down, channels: 3)
                    checked += 1
                }
            }
        }
        XCTAssertGreaterThan(checked, 0, "Expected to find covered internal shared-edge pixels")
    }

    private func sampleTextureScalar(_ rast: Rasterizer, _ texture: [Float], _ uv: SIMD2<Float>) throws -> Float {
        let out = try rast.texture(
            texture: texture, texWidth: 2, texHeight: 2, channels: 1,
            uv: [uv], outWidth: 1, outHeight: 1,
            filterMode: .linear, boundaryMode: .clamp
        )
        return out.samples[0]
    }

    private func assertColorUnchanged(
        _ actual: [Float],
        _ expected: [Float],
        at pixel: Int,
        channels: Int,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        for c in 0..<channels {
            XCTAssertEqual(
                actual[pixel * channels + c],
                expected[pixel * channels + c],
                accuracy: 1e-5,
                file: file,
                line: line
            )
        }
    }

    private static func cpuRasterize(
        positions: [SIMD4<Float>],
        triangles: [SIMD3<Int32>],
        width: Int,
        height: Int
    ) -> RasterOutput {
        let pixelCount = width * height
        var triangleIds = [Int32](repeating: -1, count: pixelCount)
        var depth = [Float](repeating: 0, count: pixelCount)
        var bary = [SIMD2<Float>](repeating: SIMD2<Float>(0, 0), count: pixelCount)
        var db = [SIMD4<Float>](repeating: SIMD4<Float>(0, 0, 0, 0), count: pixelCount)
        let xs = 2.0 / Float(width)
        let xo = -1.0 + 1.0 / Float(width)
        let ys = 2.0 / Float(height)
        let yo = -1.0 + 1.0 / Float(height)

        for py in 0..<height {
            for px in 0..<width {
                let pidx = px + width * py
                let fx = xs * Float(px) + xo
                let fy = ys * Float(py) + yo
                var bestZ: Float = -2
                var bestTri: Int32 = -1
                var bestB = SIMD2<Float>(0, 0)
                var bestDB = SIMD4<Float>(0, 0, 0, 0)

                for (tid, tri) in triangles.enumerated() {
                    let vi0 = Int(tri.x), vi1 = Int(tri.y), vi2 = Int(tri.z)
                    if vi0 < 0 || vi0 >= positions.count || vi1 < 0 || vi1 >= positions.count || vi2 < 0 || vi2 >= positions.count {
                        continue
                    }
                    let p0 = positions[vi0], p1 = positions[vi1], p2 = positions[vi2]
                    if p0.w == 0 || p1.w == 0 || p2.w == 0 { continue }

                    let sx0 = p0.x / p0.w, sy0 = p0.y / p0.w
                    let sx1 = p1.x / p1.w, sy1 = p1.y / p1.w
                    let sx2 = p2.x / p2.w, sy2 = p2.y / p2.w
                    if fx < min(sx0, min(sx1, sx2)) || fx > max(sx0, max(sx1, sx2)) ||
                        fy < min(sy0, min(sy1, sy2)) || fy > max(sy0, max(sy1, sy2)) {
                        continue
                    }

                    let p0x = p0.x - fx * p0.w
                    let p0y = p0.y - fy * p0.w
                    let p1x = p1.x - fx * p1.w
                    let p1y = p1.y - fy * p1.w
                    let p2x = p2.x - fx * p2.w
                    let p2y = p2.y - fy * p2.w
                    let a0 = p1x * p2y - p1y * p2x
                    let a1 = p2x * p0y - p2y * p0x
                    let a2 = p0x * p1y - p0y * p1x
                    let at = a0 + a1 + a2
                    if at <= 0 { continue }
                    let iw = 1.0 / at
                    let u = a0 * iw
                    let v = a1 * iw
                    let z = p0.z * a0 + p1.z * a1 + p2.z * a2
                    let w = p0.w * a0 + p1.w * a1 + p2.w * a2
                    let zw = z / w
                    if u >= 0, v >= 0, u + v <= 1, zw >= bestZ {
                        let dfxdx = xs * iw
                        let dfydy = ys * iw
                        let da0dx = p2.y * p1.w - p1.y * p2.w
                        let da0dy = p1.x * p2.w - p2.x * p1.w
                        let da1dx = p0.y * p2.w - p2.y * p0.w
                        let da1dy = p2.x * p0.w - p0.x * p2.w
                        let da2dx = p1.y * p0.w - p0.y * p1.w
                        let da2dy = p0.x * p1.w - p1.x * p0.w
                        let datdx = da0dx + da1dx + da2dx
                        let datdy = da0dy + da1dy + da2dy
                        bestDB = SIMD4<Float>(
                            dfxdx * (u * datdx - da0dx),
                            dfydy * (u * datdy - da0dy),
                            dfxdx * (v * datdx - da1dx),
                            dfydy * (v * datdy - da1dy)
                        )
                        bestZ = zw
                        var bu = min(max(u, 0), 1)
                        var bv = min(max(v, 0), 1)
                        let bs = 1.0 / max(bu + bv, 1.0)
                        bu *= bs
                        bv *= bs
                        bestB = SIMD2<Float>(bu, bv)
                        bestTri = Int32(tid)
                    }
                }

                if bestTri >= 0 {
                    triangleIds[pidx] = bestTri
                    depth[pidx] = min(max(bestZ, -1), 1)
                    bary[pidx] = bestB
                    db[pidx] = bestDB
                }
            }
        }

        return RasterOutput(
            width: width,
            height: height,
            triangleIds: triangleIds,
            depthBuffer: depth,
            barycentrics: bary,
            baryDerivatives: db
        )
    }
}
