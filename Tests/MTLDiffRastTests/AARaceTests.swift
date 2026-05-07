//
//  AARaceTests.swift
//
//  Verifies the antialias pass produces in-range, in-between blends along
//  silhouette edges. The previous kernel did its base-colour pass-through
//  inside the same kernel as silhouette corrections, which is racy on the
//  scatter writes; the kernel now writes only via atomic-add and the caller
//  pre-fills the output with the source colour. This test confirms the AA
//  pass produces some grey pixels along a strong silhouette.
//

import XCTest
@testable import MTLDiffRast
import simd

final class AARaceTests: XCTestCase {

    func testAntialiasProducesInRangeGreyPixels() throws {
        guard isMetalAvailable(), isAppleSilicon() else { throw XCTSkip("no GPU") }
        let rast = try Rasterizer()

        let positions: [SIMD4<Float>] = [
            SIMD4<Float>( 0.13,  0.71, 0.5, 1.0),
            SIMD4<Float>(-0.81, -0.43, 0.5, 1.0),
            SIMD4<Float>( 0.79, -0.55, 0.5, 1.0),
        ]
        let triangles: [SIMD3<Int32>] = [SIMD3<Int32>(0, 1, 2)]
        let W = 256, H = 256
        let r = try rast.rasterize(positions: positions, triangles: triangles, width: W, height: H)

        var color = [Float](repeating: 0, count: r.pixelCount * 3)
        for i in 0..<r.pixelCount where r.triangleIds[i] >= 0 {
            color[i * 3 + 0] = 1
            color[i * 3 + 1] = 1
            color[i * 3 + 2] = 1
        }
        let aa = try rast.antialias(
            color: color, channels: 3,
            rasterOutput: r, positions: positions, triangles: triangles
        )

        var greyCount = 0
        for v in aa.colors {
            // Output must always stay in [0, 1] given a [0, 1] input.
            XCTAssertGreaterThanOrEqual(v, -1e-4)
            XCTAssertLessThanOrEqual(v, 1.0 + 1e-4)
            if v > 0.001 && v < 0.999 { greyCount += 1 }
        }
        XCTAssertGreaterThan(
            greyCount, 50,
            "AA must blend some silhouette pixels (got \(greyCount))"
        )
    }
}
