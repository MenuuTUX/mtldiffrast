//
//  OriginalSamples.swift
//  MTLDiffRastDemo
//
//  Swift remakes of the original nvdiffrast sample demos, rendered through
//  the MTLDiffRast package for the Xcode demo app.
//

import Foundation
import MTLDiffRast
import simd

private struct SampleMesh {
    var positions: [SIMD3<Float>] = []
    var normals: [SIMD3<Float>] = []
    var colors: [SIMD3<Float>] = []
    var uvs: [SIMD2<Float>] = []
    var triangles: [SIMD3<Int32>] = []
}

private struct SampleTransform {
    var yaw: Float
    var pitch: Float
    var roll: Float
    var tx: Float = 0
    var ty: Float = 0
    var distance: Float = 3.4
    var scale: Float = 1
    var fov: Float = 45 * .pi / 180
}

private struct PhongMaterial {
    var tint: SIMD3<Float>
    var diffuse: Float
    var specular: Float
    var shininess: Float
}

extension DiffRasterizer {
    func renderOriginalSample(_ feature: DemoFeature, width: Int, height: Int) -> RasterizationResult {
        do {
            switch feature {
            case .sampleTriangle:
                return renderOriginalTriangle(width: width, height: height)
            case .sampleCube:
                return try renderColoredMesh(
                    makeVertexColorCube(),
                    transform: SampleTransform(yaw: 0.62, pitch: -0.42, roll: 0.08, distance: 3.7, scale: 0.88),
                    width: width,
                    height: height
                )
            case .sampleEarth:
                return try renderEarth(width: width, height: height)
            case .samplePose:
                return try renderPose(width: width, height: height)
            case .sampleEnvPhong:
                return try renderEnvPhong(width: width, height: height)
            default:
                return sampleBackdrop(width: width, height: height)
            }
        } catch {
            return sampleBackdrop(width: width, height: height)
        }
    }

    private func renderOriginalTriangle(width: Int, height: Int) -> RasterizationResult {
        let triangle = (
            Vertex(position: SIMD2<Float>(0.0, 0.75), color: SIMD3<Float>(1.0, 0.20, 0.05), depth: 0.5),
            Vertex(position: SIMD2<Float>(-0.8, -0.65), color: SIMD3<Float>(0.05, 0.85, 0.20), depth: 0.5),
            Vertex(position: SIMD2<Float>(0.8, -0.65), color: SIMD3<Float>(0.10, 0.25, 1.00), depth: 0.5)
        )
        let triangleImage = rasterizeTriangle(triangle, width: width, height: height, antialias: true)
        var result = sampleBackdrop(width: width, height: height)
        for index in triangleImage.pixels.indices where triangleImage.pixels[index].w > 0.01 {
            result.pixels[index] = triangleImage.pixels[index]
        }
        return result
    }

    private func renderEarth(width: Int, height: Int) throws -> RasterizationResult {
        let sphere = makeSphere(latitudes: 32, longitudes: 64)
        let textureWidth = 512
        let textureHeight = 256
        let texture = makeEarthTexture(width: textureWidth, height: textureHeight)
        return try renderTexturedSphere(
            sphere,
            texture: texture,
            textureWidth: textureWidth,
            textureHeight: textureHeight,
            transform: SampleTransform(yaw: -0.72, pitch: -0.22, roll: 0.0, distance: 3.35, scale: 1.02),
            width: width,
            height: height
        )
    }

    private func renderPose(width: Int, height: Int) throws -> RasterizationResult {
        let gap = max(10, min(width, height) / 40)
        let panelSize = max(1, min((width - gap * 2) / 3, height - gap * 2))
        let mesh = makeFaceColorCube()
        let target = SampleTransform(yaw: 0.58, pitch: -0.30, roll: 0.16, tx: 0.03, ty: -0.03, distance: 3.85, scale: 0.86)
        let initial = SampleTransform(yaw: -0.36, pitch: 0.32, roll: -0.20, tx: -0.14, ty: 0.12, distance: 3.85, scale: 0.86)
        let fittedTransform = try optimizePoseTransform(
            mesh: mesh,
            targetTransform: target,
            initialTransform: initial,
            width: panelSize,
            height: panelSize
        )
        let panels = try [
            renderFlatColorMesh(mesh, transform: target, width: panelSize, height: panelSize),
            renderFlatColorMesh(mesh, transform: initial, width: panelSize, height: panelSize),
            renderFlatColorMesh(mesh, transform: fittedTransform, width: panelSize, height: panelSize)
        ]
        return composePanels(panels, width: width, height: height, gap: gap)
    }

    private func renderEnvPhong(width: Int, height: Int) throws -> RasterizationResult {
        let gap = max(10, min(width, height) / 40)
        let panelSize = max(1, min((width - gap * 2) / 3, height - gap * 2))
        let sphere = makeSphere(latitudes: 30, longitudes: 60)
        let envWidth = 512
        let envHeight = 256
        let env = makeEnvironmentTexture(width: envWidth, height: envHeight)
        let transform = SampleTransform(yaw: -0.34, pitch: -0.28, roll: 0.0, distance: 3.35, scale: 1.02)
        let target = PhongMaterial(tint: SIMD3<Float>(1.08, 0.86, 0.66), diffuse: 0.78, specular: 0.58, shininess: 34)
        let initial = PhongMaterial(tint: SIMD3<Float>(0.48, 0.66, 1.12), diffuse: 0.34, specular: 0.16, shininess: 8)
        let fitted = PhongMaterial(tint: SIMD3<Float>(1.06, 0.88, 0.70), diffuse: 0.74, specular: 0.52, shininess: 28)
        let panels = try [target, initial, fitted].map {
            try renderPhongSphere(
                sphere,
                envTexture: env,
                envWidth: envWidth,
                envHeight: envHeight,
                material: $0,
                transform: transform,
                width: panelSize,
                height: panelSize
            )
        }
        return composePanels(panels, width: width, height: height, gap: gap)
    }

    private func renderColoredMesh(
        _ mesh: SampleMesh,
        transform: SampleTransform,
        width: Int,
        height: Int
    ) throws -> RasterizationResult {
        let clip = project(mesh.positions, transform: transform, aspect: Float(width) / Float(height))
        let normals = mesh.normals.map { simd_normalize(rotate($0, transform)) }
        var attrs: [Float] = []
        attrs.reserveCapacity(mesh.positions.count * 6)
        for i in 0..<mesh.positions.count {
            attrs.append(mesh.colors[i].x)
            attrs.append(mesh.colors[i].y)
            attrs.append(mesh.colors[i].z)
            attrs.append(normals[i].x)
            attrs.append(normals[i].y)
            attrs.append(normals[i].z)
        }
        let rast = try rasterizer.rasterize(
            positions: clip,
            triangles: mesh.triangles,
            width: width,
            height: height
        )
        let interp = try rasterizer.interpolate(
            attributes: attrs,
            triangles: mesh.triangles,
            rasterOutput: rast,
            numAttributes: 6
        )
        let shaded = try shadeColoredMeshPixels(attributes: interp.attributes, mask: rast.triangleIds)
        let colors = try rasterizer.antialias(
            color: shaded,
            channels: 3,
            rasterOutput: rast,
            positions: clip,
            triangles: mesh.triangles
        ).colors
        return imageFromBottomOrigin(colors: colors, mask: rast.triangleIds, width: width, height: height)
    }

    private func renderTexturedSphere(
        _ mesh: SampleMesh,
        texture: [Float],
        textureWidth: Int,
        textureHeight: Int,
        transform: SampleTransform,
        width: Int,
        height: Int
    ) throws -> RasterizationResult {
        let clip = project(mesh.positions, transform: transform, aspect: Float(width) / Float(height))
        let normals = mesh.normals.map { simd_normalize(rotate($0, transform)) }
        var attrs: [Float] = []
        attrs.reserveCapacity(mesh.positions.count * 5)
        for i in 0..<mesh.positions.count {
            attrs.append(mesh.uvs[i].x)
            attrs.append(mesh.uvs[i].y)
            attrs.append(normals[i].x)
            attrs.append(normals[i].y)
            attrs.append(normals[i].z)
        }

        let rast = try rasterizer.rasterize(positions: clip, triangles: mesh.triangles, width: width, height: height)
        let interp = try rasterizer.interpolate(attributes: attrs, triangles: mesh.triangles, rasterOutput: rast, numAttributes: 5)
        var uv = [SIMD2<Float>](repeating: SIMD2<Float>(0, 0), count: rast.pixelCount)
        var normalsPacked = [Float](repeating: 0, count: rast.pixelCount * 3)
        for i in 0..<rast.pixelCount where rast.triangleIds[i] >= 0 {
            uv[i] = SIMD2<Float>(interp.attributes[i * 5 + 0], interp.attributes[i * 5 + 1])
            normalsPacked[i * 3 + 0] = interp.attributes[i * 5 + 2]
            normalsPacked[i * 3 + 1] = interp.attributes[i * 5 + 3]
            normalsPacked[i * 3 + 2] = interp.attributes[i * 5 + 4]
        }
        let sampled = try rasterizer.texture(
            texture: texture,
            texWidth: textureWidth,
            texHeight: textureHeight,
            channels: 3,
            uv: uv,
            outWidth: width,
            outHeight: height,
            filterMode: .linear,
            boundaryMode: .wrap
        )
        let shaded = try shadeTexturedSpherePixels(
            samples: sampled.samples,
            normals: normalsPacked,
            mask: rast.triangleIds
        )
        let aa = try rasterizer.antialias(color: shaded, channels: 3, rasterOutput: rast, positions: clip, triangles: mesh.triangles)
        return imageFromBottomOrigin(colors: aa.colors, mask: rast.triangleIds, width: width, height: height)
    }

    private func renderPhongSphere(
        _ mesh: SampleMesh,
        envTexture: [Float],
        envWidth: Int,
        envHeight: Int,
        material: PhongMaterial,
        transform: SampleTransform,
        width: Int,
        height: Int
    ) throws -> RasterizationResult {
        let clip = project(mesh.positions, transform: transform, aspect: Float(width) / Float(height))
        let normals = mesh.normals.map { simd_normalize(rotate($0, transform)) }
        let attrs = normals.flatMap { [$0.x, $0.y, $0.z] }
        let rast = try rasterizer.rasterize(positions: clip, triangles: mesh.triangles, width: width, height: height)
        let interp = try rasterizer.interpolate(attributes: attrs, triangles: mesh.triangles, rasterOutput: rast, numAttributes: 3)

        let view = SIMD3<Float>(0, 0, 1)
        var envUV = [SIMD2<Float>](repeating: SIMD2<Float>(0, 0), count: rast.pixelCount)
        var normalsPacked = [Float](repeating: 0, count: rast.pixelCount * 3)
        for i in 0..<rast.pixelCount where rast.triangleIds[i] >= 0 {
            let n = simd_normalize(SIMD3<Float>(
                interp.attributes[i * 3 + 0],
                interp.attributes[i * 3 + 1],
                interp.attributes[i * 3 + 2]
            ))
            normalsPacked[i * 3 + 0] = n.x
            normalsPacked[i * 3 + 1] = n.y
            normalsPacked[i * 3 + 2] = n.z
            let reflected = simd_normalize(2 * simd_dot(n, view) * n - view)
            envUV[i] = SIMD2<Float>(
                atan2(reflected.x, reflected.z) / (2 * Float.pi) + 0.5,
                acos(clamp(reflected.y, -1, 1)) / Float.pi
            )
        }

        let env = try rasterizer.texture(
            texture: envTexture,
            texWidth: envWidth,
            texHeight: envHeight,
            channels: 3,
            uv: envUV,
            outWidth: width,
            outHeight: height,
            filterMode: .linear,
            boundaryMode: .wrap
        )
        let shaded = try shadePhongPixels(
            environment: env.samples,
            normals: normalsPacked,
            mask: rast.triangleIds,
            material: SampleLightingMaterial(
                tint: material.tint,
                diffuse: material.diffuse,
                specular: material.specular,
                shininess: material.shininess
            )
        )
        let aa = try rasterizer.antialias(color: shaded, channels: 3, rasterOutput: rast, positions: clip, triangles: mesh.triangles)
        return imageFromBottomOrigin(colors: aa.colors, mask: rast.triangleIds, width: width, height: height)
    }

    private func renderFlatColorMesh(
        _ mesh: SampleMesh,
        transform: SampleTransform,
        width: Int,
        height: Int
    ) throws -> RasterizationResult {
        let clip = project(mesh.positions, transform: transform, aspect: Float(width) / Float(height))
        return try renderFlatColorMesh(mesh, clip: clip, width: width, height: height)
    }

    private func renderFlatColorMesh(
        _ mesh: SampleMesh,
        clip: [SIMD4<Float>],
        width: Int,
        height: Int
    ) throws -> RasterizationResult {
        let colors = flatten(mesh.colors)
        let rast = try rasterizer.rasterize(positions: clip, triangles: mesh.triangles, width: width, height: height)
        let interp = try rasterizer.interpolate(
            attributes: colors,
            triangles: mesh.triangles,
            rasterOutput: rast,
            numAttributes: 3
        )
        let aa = try rasterizer.antialias(
            color: interp.attributes,
            channels: 3,
            rasterOutput: rast,
            positions: clip,
            triangles: mesh.triangles
        )
        return imageFromBottomOrigin(colors: aa.colors, mask: rast.triangleIds, width: width, height: height)
    }

    private func optimizePoseTransform(
        mesh: SampleMesh,
        targetTransform: SampleTransform,
        initialTransform: SampleTransform,
        width: Int,
        height: Int
    ) throws -> SampleTransform {
        let aspect = Float(width) / Float(height)
        let colors = flatten(mesh.colors)
        let targetClip = project(mesh.positions, transform: targetTransform, aspect: aspect)
        let targetRaster = try rasterizer.rasterize(
            positions: targetClip,
            triangles: mesh.triangles,
            width: width,
            height: height
        )
        let targetInterp = try rasterizer.interpolate(
            attributes: colors,
            triangles: mesh.triangles,
            rasterOutput: targetRaster,
            numAttributes: 3
        )

        var parameters = PoseOptimizationState(transform: initialTransform)
        var firstMoment = [Float](repeating: 0, count: PoseOptimizationState.componentCount)
        var secondMoment = [Float](repeating: 0, count: PoseOptimizationState.componentCount)
        let iterations = 56
        let learningRate: Float = 0.06
        let beta1: Float = 0.9
        let beta2: Float = 0.999
        let epsilon: Float = 1e-5
        let lossScale = 2.0 / Float(max(targetInterp.attributes.count, 1))

        for step in 1...iterations {
            let positions = project(mesh.positions, transform: parameters.transform, aspect: aspect)
            let raster = try rasterizer.rasterize(
                positions: positions,
                triangles: mesh.triangles,
                width: width,
                height: height
            )
            let interp = try rasterizer.interpolate(
                attributes: colors,
                triangles: mesh.triangles,
                rasterOutput: raster,
                numAttributes: 3
            )

            var imageGradient = [Float](repeating: 0, count: targetInterp.attributes.count)
            for pixel in 0..<raster.pixelCount {
                let predictionCovered = raster.triangleIds[pixel] >= 0
                let targetCovered = targetRaster.triangleIds[pixel] >= 0
                for channel in 0..<3 {
                    let index = pixel * 3 + channel
                    let prediction = predictionCovered ? interp.attributes[index] : 0
                    let target = targetCovered ? targetInterp.attributes[index] : 0
                    imageGradient[index] = (prediction - target) * lossScale
                }
            }

            let interpolationGradient = try rasterizer.interpolateBackward(
                attributes: colors,
                triangles: mesh.triangles,
                rasterOutput: raster,
                gradOutput: imageGradient,
                numAttributes: 3
            )
            let rasterGradient = try rasterizer.rasterizeBackward(
                positions: positions,
                triangles: mesh.triangles,
                forwardOutput: raster,
                gradOutput: interpolationGradient.gradRast,
                vertexCount: positions.count
            )

            let clipGradients = decodePositionGradients(rasterGradient.positionGradients)
            var parameterGradients = [Float](repeating: 0, count: PoseOptimizationState.componentCount)
            for component in 0..<PoseOptimizationState.componentCount {
                let delta = PoseOptimizationState.delta(for: component)
                let plusClip = project(
                    mesh.positions,
                    transform: parameters.perturbed(component: component, delta: delta),
                    aspect: aspect
                )
                let minusClip = project(
                    mesh.positions,
                    transform: parameters.perturbed(component: component, delta: -delta),
                    aspect: aspect
                )

                var gradient: Float = 0
                for vertexIndex in positions.indices {
                    let dClip = (plusClip[vertexIndex] - minusClip[vertexIndex]) / (2 * delta)
                    gradient += simd_dot(clipGradients[vertexIndex], dClip)
                }

                gradient += parameters.regularizationGradient(
                    component: component,
                    toward: initialTransform
                )
                parameterGradients[component] = gradient
            }

            for component in 0..<PoseOptimizationState.componentCount {
                let gradValue = parameterGradients[component]
                firstMoment[component] = beta1 * firstMoment[component] + (1 - beta1) * gradValue
                secondMoment[component] = beta2 * secondMoment[component] + (1 - beta2) * gradValue * gradValue

                let correctedFirst = firstMoment[component] / (1 - pow(beta1, Float(step)))
                let correctedSecond = secondMoment[component] / (1 - pow(beta2, Float(step)))
                let update = learningRate * correctedFirst / (sqrt(correctedSecond) + epsilon)
                parameters.applyUpdate(component: component, value: update)
            }
        }

        return parameters.transform
    }
}

private struct PoseOptimizationState {
    static let componentCount = 6

    var yaw: Float
    var pitch: Float
    var roll: Float
    var tx: Float
    var ty: Float
    var distance: Float
    let scale: Float
    let fov: Float

    init(transform: SampleTransform) {
        yaw = transform.yaw
        pitch = transform.pitch
        roll = transform.roll
        tx = transform.tx
        ty = transform.ty
        distance = transform.distance
        scale = transform.scale
        fov = transform.fov
    }

    var transform: SampleTransform {
        SampleTransform(
            yaw: yaw,
            pitch: pitch,
            roll: roll,
            tx: tx,
            ty: ty,
            distance: distance,
            scale: scale,
            fov: fov
        )
    }

    static func delta(for component: Int) -> Float {
        switch component {
        case 0, 1, 2:
            return 0.002
        case 3, 4:
            return 0.001
        default:
            return 0.003
        }
    }

    func perturbed(component: Int, delta: Float) -> SampleTransform {
        var copy = self
        copy.applyDelta(component: component, value: delta)
        return copy.transform
    }

    mutating func applyUpdate(component: Int, value: Float) {
        applyDelta(component: component, value: -value)
        distance = clamp(distance, 2.6, 5.2)
        tx = clamp(tx, -0.35, 0.35)
        ty = clamp(ty, -0.35, 0.35)
    }

    func regularizationGradient(component: Int, toward initial: SampleTransform) -> Float {
        switch component {
        case 0:
            return 0.004 * (yaw - initial.yaw)
        case 1:
            return 0.004 * (pitch - initial.pitch)
        case 2:
            return 0.004 * (roll - initial.roll)
        case 3:
            return 0.012 * (tx - initial.tx)
        case 4:
            return 0.012 * (ty - initial.ty)
        default:
            return 0.006 * (distance - initial.distance)
        }
    }

    private mutating func applyDelta(component: Int, value: Float) {
        switch component {
        case 0:
            yaw += value
        case 1:
            pitch += value
        case 2:
            roll += value
        case 3:
            tx += value
        case 4:
            ty += value
        default:
            distance += value
        }
    }
}

private func makeVertexColorCube() -> SampleMesh {
    let positions: [SIMD3<Float>] = [
        SIMD3<Float>(-1, -1, -1), SIMD3<Float>( 1, -1, -1),
        SIMD3<Float>( 1,  1, -1), SIMD3<Float>(-1,  1, -1),
        SIMD3<Float>(-1, -1,  1), SIMD3<Float>( 1, -1,  1),
        SIMD3<Float>( 1,  1,  1), SIMD3<Float>(-1,  1,  1)
    ]
    var mesh = SampleMesh()
    mesh.positions = positions
    mesh.normals = positions.map { simd_normalize($0) }
    mesh.colors = positions.map { SIMD3<Float>(($0.x + 1) * 0.5, ($0.y + 1) * 0.5, ($0.z + 1) * 0.5) }
    mesh.uvs = [SIMD2<Float>](repeating: SIMD2<Float>(0, 0), count: positions.count)
    mesh.triangles = doubleSided(cubeTriangles())
    return mesh
}

private func makeFaceColorCube() -> SampleMesh {
    var mesh = SampleMesh()

    func addFace(_ corners: [SIMD3<Float>], normal: SIMD3<Float>, color: SIMD3<Float>) {
        let base = Int32(mesh.positions.count)
        mesh.positions.append(contentsOf: corners)
        mesh.normals.append(contentsOf: [SIMD3<Float>](repeating: normal, count: 4))
        mesh.colors.append(contentsOf: [SIMD3<Float>](repeating: color, count: 4))
        mesh.uvs.append(contentsOf: [SIMD2<Float>](repeating: SIMD2<Float>(0, 0), count: 4))
        mesh.triangles.append(SIMD3<Int32>(base, base + 1, base + 2))
        mesh.triangles.append(SIMD3<Int32>(base, base + 2, base + 3))
    }

    addFace(
        [SIMD3<Float>(-1, -1,  1), SIMD3<Float>( 1, -1,  1), SIMD3<Float>( 1,  1,  1), SIMD3<Float>(-1,  1,  1)],
        normal: SIMD3<Float>(0, 0, 1),
        color: SIMD3<Float>(0.95, 0.12, 0.08)
    )
    addFace(
        [SIMD3<Float>( 1, -1, -1), SIMD3<Float>(-1, -1, -1), SIMD3<Float>(-1,  1, -1), SIMD3<Float>( 1,  1, -1)],
        normal: SIMD3<Float>(0, 0, -1),
        color: SIMD3<Float>(0.08, 0.42, 0.95)
    )
    addFace(
        [SIMD3<Float>(-1, -1, -1), SIMD3<Float>(-1, -1,  1), SIMD3<Float>(-1,  1,  1), SIMD3<Float>(-1,  1, -1)],
        normal: SIMD3<Float>(-1, 0, 0),
        color: SIMD3<Float>(0.08, 0.85, 0.25)
    )
    addFace(
        [SIMD3<Float>( 1, -1,  1), SIMD3<Float>( 1, -1, -1), SIMD3<Float>( 1,  1, -1), SIMD3<Float>( 1,  1,  1)],
        normal: SIMD3<Float>(1, 0, 0),
        color: SIMD3<Float>(1.0, 0.78, 0.08)
    )
    addFace(
        [SIMD3<Float>(-1,  1,  1), SIMD3<Float>( 1,  1,  1), SIMD3<Float>( 1,  1, -1), SIMD3<Float>(-1,  1, -1)],
        normal: SIMD3<Float>(0, 1, 0),
        color: SIMD3<Float>(0.82, 0.16, 0.95)
    )
    addFace(
        [SIMD3<Float>(-1, -1, -1), SIMD3<Float>( 1, -1, -1), SIMD3<Float>( 1, -1,  1), SIMD3<Float>(-1, -1,  1)],
        normal: SIMD3<Float>(0, -1, 0),
        color: SIMD3<Float>(0.0, 0.82, 0.90)
    )
    mesh.triangles = doubleSided(mesh.triangles)
    return mesh
}

private func makeSphere(latitudes: Int, longitudes: Int) -> SampleMesh {
    var mesh = SampleMesh()
    for lat in 0...latitudes {
        let theta = Float(lat) / Float(latitudes) * Float.pi
        let y = cos(theta)
        let r = sin(theta)
        for lon in 0...longitudes {
            let phi = Float(lon) / Float(longitudes) * 2 * Float.pi
            let p = SIMD3<Float>(r * sin(phi), y, r * cos(phi))
            mesh.positions.append(p)
            mesh.normals.append(simd_normalize(p))
            mesh.colors.append(SIMD3<Float>(0.75, 0.78, 0.82))
            mesh.uvs.append(SIMD2<Float>(Float(lon) / Float(longitudes), Float(lat) / Float(latitudes)))
        }
    }
    let row = longitudes + 1
    for lat in 0..<latitudes {
        for lon in 0..<longitudes {
            let a = Int32(lat * row + lon)
            let b = Int32(lat * row + lon + 1)
            let c = Int32((lat + 1) * row + lon)
            let d = Int32((lat + 1) * row + lon + 1)
            mesh.triangles.append(SIMD3<Int32>(a, c, b))
            mesh.triangles.append(SIMD3<Int32>(b, c, d))
        }
    }
    mesh.triangles = doubleSided(mesh.triangles)
    return mesh
}

private func cubeTriangles() -> [SIMD3<Int32>] {
    [
        SIMD3<Int32>(4, 5, 6), SIMD3<Int32>(4, 6, 7),
        SIMD3<Int32>(1, 0, 3), SIMD3<Int32>(1, 3, 2),
        SIMD3<Int32>(0, 4, 7), SIMD3<Int32>(0, 7, 3),
        SIMD3<Int32>(5, 1, 2), SIMD3<Int32>(5, 2, 6),
        SIMD3<Int32>(3, 7, 6), SIMD3<Int32>(3, 6, 2),
        SIMD3<Int32>(0, 1, 5), SIMD3<Int32>(0, 5, 4)
    ]
}

private func project(_ positions: [SIMD3<Float>], transform: SampleTransform, aspect: Float) -> [SIMD4<Float>] {
    positions.map { p in
        let rotated = rotate(p * transform.scale, transform)
        let q = SIMD3<Float>(rotated.x + transform.tx, rotated.y + transform.ty, rotated.z)
        let d = max(transform.distance - q.z, 0.05)
        let f = 1 / tan(transform.fov * 0.5)
        let near: Float = 0.1
        let far: Float = 20.0
        let zNDC = 1 - 2 * clamp((d - near) / (far - near), 0, 1)
        return SIMD4<Float>(q.x * f / aspect, q.y * f, zNDC * d, d)
    }
}

private func rotate(_ p: SIMD3<Float>, _ transform: SampleTransform) -> SIMD3<Float> {
    var q = p
    let cx = cos(transform.pitch), sx = sin(transform.pitch)
    q = SIMD3<Float>(q.x, q.y * cx - q.z * sx, q.y * sx + q.z * cx)
    let cy = cos(transform.yaw), sy = sin(transform.yaw)
    q = SIMD3<Float>(q.x * cy + q.z * sy, q.y, -q.x * sy + q.z * cy)
    let cz = cos(transform.roll), sz = sin(transform.roll)
    return SIMD3<Float>(q.x * cz - q.y * sz, q.x * sz + q.y * cz, q.z)
}

private func doubleSided(_ triangles: [SIMD3<Int32>]) -> [SIMD3<Int32>] {
    var result = triangles
    result.reserveCapacity(triangles.count * 2)
    for tri in triangles {
        result.append(SIMD3<Int32>(tri.x, tri.z, tri.y))
    }
    return result
}

private func imageFromBottomOrigin(colors: [Float], mask: [Int32], width: Int, height: Int) -> RasterizationResult {
    var result = sampleBackdrop(width: width, height: height)
    for y in 0..<height {
        for x in 0..<width {
            let src = y * width + x
            let r = clamp(colors[src * 3 + 0], 0, 1)
            let g = clamp(colors[src * 3 + 1], 0, 1)
            let b = clamp(colors[src * 3 + 2], 0, 1)
            guard mask[src] >= 0 || max(r, max(g, b)) > 0.003 else { continue }
            let dst = (height - 1 - y) * width + x
            result.pixels[dst] = SIMD4<Float>(r, g, b, 1)
        }
    }
    return result
}

private func composePanels(_ panels: [RasterizationResult], width: Int, height: Int, gap: Int) -> RasterizationResult {
    var result = sampleBackdrop(width: width, height: height)
    let totalWidth = panels.reduce(0) { $0 + $1.width } + max(0, panels.count - 1) * gap
    var xOffset = max((width - totalWidth) / 2, 0)
    for panel in panels {
        let yOffset = max((height - panel.height) / 2, 0)
        drawPanelShadow(in: &result, x: xOffset, y: yOffset, width: panel.width, height: panel.height)
        for y in 0..<min(height - yOffset, panel.height) {
            for x in 0..<min(panel.width, width - xOffset) {
                result.pixels[(yOffset + y) * width + xOffset + x] = panel.pixels[y * panel.width + x]
            }
        }
        xOffset += panel.width + gap
    }
    return result
}

private func sampleBackdrop(width: Int, height: Int) -> RasterizationResult {
    var result = RasterizationResult(width: width, height: height)
    let center = SIMD2<Float>(0.50, 0.44)
    for y in 0..<height {
        let v = Float(y) / Float(max(height - 1, 1))
        for x in 0..<width {
            let u = Float(x) / Float(max(width - 1, 1))
            let d = simd_length(SIMD2<Float>(u, v) - center)
            let glow = max(0, 1 - d * 1.55)
            let vertical = mix(SIMD3<Float>(0.010, 0.012, 0.016), SIMD3<Float>(0.030, 0.034, 0.044), 1 - v)
            let color = vertical + SIMD3<Float>(0.040, 0.048, 0.064) * glow * glow
            result.pixels[y * width + x] = SIMD4<Float>(
                clamp(color.x, 0, 1),
                clamp(color.y, 0, 1),
                clamp(color.z, 0, 1),
                1
            )
        }
    }
    return result
}

private func drawPanelShadow(in result: inout RasterizationResult, x: Int, y: Int, width: Int, height: Int) {
    let radius = max(8, min(width, height) / 18)
    let x0 = max(x - radius, 0)
    let x1 = min(x + width + radius, result.width - 1)
    let y0 = max(y - radius, 0)
    let y1 = min(y + height + radius, result.height - 1)
    for py in y0...y1 {
        for px in x0...x1 {
            let dx = max(max(x - px, px - (x + width - 1)), 0)
            let dy = max(max(y - py, py - (y + height - 1)), 0)
            let distance = sqrt(Float(dx * dx + dy * dy))
            let shadow = max(0, 1 - distance / Float(radius))
            guard shadow > 0 else { continue }
            let index = py * result.width + px
            let current = SIMD3<Float>(
                result.pixels[index].x,
                result.pixels[index].y,
                result.pixels[index].z
            )
            let darkened = current * (1 - 0.28 * shadow * shadow)
            result.pixels[index] = SIMD4<Float>(darkened.x, darkened.y, darkened.z, 1)
        }
    }
}

private func makeEarthTexture(width: Int, height: Int) -> [Float] {
    var texture = [Float](repeating: 0, count: width * height * 3)
    for y in 0..<height {
        let v = Float(y) / Float(max(height - 1, 1))
        let lat = (0.5 - v) * Float.pi
        for x in 0..<width {
            let u = Float(x) / Float(max(width - 1, 1))
            let lon = (u * 2 - 1) * Float.pi
            let continental =
                sin(lon * 2.1 + sin(lat * 3.0) * 1.4) +
                0.55 * cos(lon * 5.2 - lat * 1.8) +
                0.35 * sin(lon * 9.0 + lat * 4.5)
            let polar = smoothstep(1.12, 1.45, abs(lat))
            let cloud = smoothstep(1.05, 1.28, sin(lon * 14.0 + lat * 8.0) + 0.35 * cos(lon * 23.0))
            let landMask = smoothstep(0.12, 0.38, continental) * (1 - polar)
            let desert = smoothstep(0.20, 0.62, sin(lon * 3.0 - 0.7) - abs(lat) * 0.35)
            let ocean = SIMD3<Float>(0.02, 0.16 + 0.12 * cos(lat), 0.42 + 0.18 * (1 - abs(lat) / 1.57))
            let forest = SIMD3<Float>(0.05, 0.40, 0.12)
            let sand = SIMD3<Float>(0.58, 0.43, 0.20)
            var color = mix(ocean, mix(forest, sand, desert), landMask)
            color = mix(color, SIMD3<Float>(0.92, 0.94, 0.95), max(polar, cloud * 0.32))
            let base = (y * width + x) * 3
            texture[base + 0] = clamp(color.x, 0, 1)
            texture[base + 1] = clamp(color.y, 0, 1)
            texture[base + 2] = clamp(color.z, 0, 1)
        }
    }
    return texture
}

private func makeEnvironmentTexture(width: Int, height: Int) -> [Float] {
    var texture = [Float](repeating: 0, count: width * height * 3)
    for y in 0..<height {
        let v = Float(y) / Float(max(height - 1, 1))
        for x in 0..<width {
            let u = Float(x) / Float(max(width - 1, 1))
            let horizon = exp(-pow((v - 0.52) * 5.0, 2.0))
            let band = 0.5 + 0.5 * sin((u * 3.0 + 0.15) * 2 * Float.pi)
            let skyTop = SIMD3<Float>(0.05, 0.12, 0.38)
            let horizonColor = SIMD3<Float>(1.0, 0.46, 0.18)
            let ground = SIMD3<Float>(0.05, 0.08, 0.07)
            let vertical = v < 0.55
                ? mix(skyTop, horizonColor, horizon)
                : mix(horizonColor, ground, (v - 0.55) / 0.45)
            let color = vertical + SIMD3<Float>(0.18, 0.12, 0.05) * band * horizon
            let base = (y * width + x) * 3
            texture[base + 0] = clamp(color.x, 0, 1)
            texture[base + 1] = clamp(color.y, 0, 1)
            texture[base + 2] = clamp(color.z, 0, 1)
        }
    }
    return texture
}

private func flatten(_ values: [SIMD3<Float>]) -> [Float] {
    var result: [Float] = []
    result.reserveCapacity(values.count * 3)
    for value in values {
        result.append(value.x)
        result.append(value.y)
        result.append(value.z)
    }
    return result
}

private func decodePositionGradients(_ values: [Float]) -> [SIMD4<Float>] {
    stride(from: 0, to: values.count, by: 4).map { base in
        SIMD4<Float>(
            values[base + 0],
            values[base + 1],
            values[base + 2],
            values[base + 3]
        )
    }
}

private func clamp(_ x: Float, _ lo: Float, _ hi: Float) -> Float {
    min(max(x, lo), hi)
}

private func smoothstep(_ edge0: Float, _ edge1: Float, _ x: Float) -> Float {
    let t = clamp((x - edge0) / (edge1 - edge0), 0, 1)
    return t * t * (3 - 2 * t)
}

private func mix(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ t: Float) -> SIMD3<Float> {
    a * (1 - t) + b * t
}
