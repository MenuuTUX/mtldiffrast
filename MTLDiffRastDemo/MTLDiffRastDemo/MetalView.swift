//
//  MetalView.swift
//  MTLDiffRastDemo
//
//  MetalKit-backed presentation view for the rasterizer demo.
//

import MetalKit
import QuartzCore
import SwiftUI
import simd

private struct ViewportUniforms {
    var scale: SIMD2<Float>
}

struct MetalView: NSViewRepresentable {
    @ObservedObject var viewModel: DemoViewModel

    func makeCoordinator() -> Coordinator {
        Coordinator(viewModel: viewModel)
    }

    func makeNSView(context: Context) -> MTKView {
        let view = MTKView()
        guard let device = MTLCreateSystemDefaultDevice() else {
            return view
        }

        view.device = device
        view.delegate = context.coordinator
        view.colorPixelFormat = .bgra8Unorm
        view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        view.framebufferOnly = true
        view.autoResizeDrawable = true
        view.enableSetNeedsDisplay = false
        view.isPaused = false
        view.presentsWithTransaction = false
        view.preferredFramesPerSecond = context.coordinator.preferredFPS(for: view)
        context.coordinator.configure(device: device, view: view)
        return view
    }

    func updateNSView(_ view: MTKView, context: Context) {
        context.coordinator.viewModel = viewModel
        context.coordinator.updateLoopMode(for: view)

        let resolution = context.coordinator.renderResolution(for: view)
        context.coordinator.publishRenderSize(resolution)

        if !viewModel.selectedFeature.isAnimated {
            view.setNeedsDisplay(view.bounds)
        }
    }

    final class Coordinator: NSObject, MTKViewDelegate {
        var viewModel: DemoViewModel

        private var device: MTLDevice?
        private var commandQueue: MTLCommandQueue?
        private var pipelineState: MTLRenderPipelineState?
        private var wireframePipelineState: MTLRenderPipelineState?
        private var displayTexture: MTLTexture?
        private var uploadTexture: MTLTexture?
        private var pixelBytes: [UInt8] = []
        private var lastDrawTime = CACurrentMediaTime()
        private var fpsWindowStart = CACurrentMediaTime()
        private var framesInWindow = 0

        init(viewModel: DemoViewModel) {
            self.viewModel = viewModel
        }

        func configure(device: MTLDevice, view: MTKView) {
            self.device = device
            commandQueue = device.makeCommandQueue()

            do {
                let library = try device.makeLibrary(source: Self.shaderSource, options: nil)
                let descriptor = MTLRenderPipelineDescriptor()
                descriptor.vertexFunction = library.makeFunction(name: "fullscreenVertex")
                descriptor.fragmentFunction = library.makeFunction(name: "fullscreenFragment")
                descriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat
                pipelineState = try device.makeRenderPipelineState(descriptor: descriptor)

                let wireframeDescriptor = MTLRenderPipelineDescriptor()
                wireframeDescriptor.vertexFunction = library.makeFunction(name: "wireframeVertex")
                wireframeDescriptor.fragmentFunction = library.makeFunction(name: "wireframeFragment")
                wireframeDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat
                wireframePipelineState = try device.makeRenderPipelineState(descriptor: wireframeDescriptor)
            } catch {
                pipelineState = nil
                wireframePipelineState = nil
            }

            updateLoopMode(for: view)
        }

        func preferredFPS(for view: MTKView) -> Int {
            let screenFPS = view.window?.screen?.maximumFramesPerSecond
                ?? NSScreen.main?.maximumFramesPerSecond
                ?? 60
            return min(max(screenFPS, 60), 120)
        }

        func updateLoopMode(for view: MTKView) {
            view.preferredFramesPerSecond = preferredFPS(for: view)
            let shouldAnimate = viewModel.selectedFeature.isAnimated
            view.isPaused = !shouldAnimate
            view.enableSetNeedsDisplay = !shouldAnimate
        }

        func publishRenderSize(_ resolution: (width: Int, height: Int)) {
            publish {
                self.viewModel.updateRenderSize(width: resolution.width, height: resolution.height)
            }
        }

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
            let resolution = renderResolution(width: Int(size.width), height: Int(size.height))
            publishRenderSize(resolution)
            if !viewModel.selectedFeature.isAnimated {
                view.setNeedsDisplay(view.bounds)
            }
        }

        func draw(in view: MTKView) {
            autoreleasepool {
                drawFrame(in: view)
            }
        }

        private func drawFrame(in view: MTKView) {
            guard
                let commandQueue,
                let pipelineState,
                let drawable = view.currentDrawable,
                let renderPassDescriptor = view.currentRenderPassDescriptor,
                view.drawableSize.width >= 1,
                view.drawableSize.height >= 1
            else {
                return
            }

            let now = CACurrentMediaTime()
            let deltaTime = min(max(now - lastDrawTime, 0), 1.0 / 15.0)
            lastDrawTime = now
            viewModel.advanceAnimation(deltaTime: deltaTime)

            let resolution = renderResolution(for: view)
            if viewModel.renderWidth != resolution.width || viewModel.renderHeight != resolution.height {
                publishRenderSize(resolution)
            }

            if let texture = viewModel.renderCurrentTexture(width: resolution.width, height: resolution.height) {
                displayTexture = texture
            } else if let result = viewModel.renderCurrentFrame(width: resolution.width, height: resolution.height) {
                updateTexture(from: result)
            }

            guard let texture = displayTexture, let commandBuffer = commandQueue.makeCommandBuffer() else {
                return
            }

            let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)
            var uniforms = ViewportUniforms(
                scale: fittedScale(texture: texture, drawableSize: view.drawableSize)
            )
            encoder?.setRenderPipelineState(pipelineState)
            encoder?.setVertexBytes(&uniforms, length: MemoryLayout<ViewportUniforms>.stride, index: 0)
            encoder?.setFragmentTexture(texture, index: 0)
            encoder?.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            drawWireframeIfNeeded(encoder: encoder, uniforms: uniforms)
            encoder?.endEncoding()

            commandBuffer.present(drawable)
            commandBuffer.commit()

            updateFPS(now: now)
        }

        func renderResolution(for view: MTKView) -> (width: Int, height: Int) {
            renderResolution(width: Int(view.drawableSize.width), height: Int(view.drawableSize.height))
        }

        func renderResolution(width: Int, height: Int) -> (width: Int, height: Int) {
            let width = max(width, 1)
            let height = max(height, 1)
            let aspect = max(viewModel.selectedFeature.preferredAspectRatio, 0.1)
            let drawableAspect = Double(width) / Double(height)

            let fittedWidth: Int
            let fittedHeight: Int
            if drawableAspect > aspect {
                fittedHeight = height
                fittedWidth = max(Int(Double(height) * aspect), 1)
            } else {
                fittedWidth = width
                fittedHeight = max(Int(Double(width) / aspect), 1)
            }

            let maxDimension: Int
            if viewModel.selectedFeature.isOriginalSample {
                maxDimension = 640
            } else if viewModel.selectedFeature.isAnimated {
                maxDimension = 1280
            } else {
                maxDimension = 1600
            }

            let longest = max(fittedWidth, fittedHeight)
            guard longest > maxDimension else {
                return (fittedWidth, fittedHeight)
            }

            let scale = Double(maxDimension) / Double(longest)
            return (
                max(Int(Double(fittedWidth) * scale), 1),
                max(Int(Double(fittedHeight) * scale), 1)
            )
        }

        private func fittedScale(texture: MTLTexture, drawableSize: CGSize) -> SIMD2<Float> {
            let drawableWidth = max(Double(drawableSize.width), 1)
            let drawableHeight = max(Double(drawableSize.height), 1)
            let drawableAspect = drawableWidth / drawableHeight
            let textureAspect = Double(texture.width) / Double(max(texture.height, 1))

            if drawableAspect > textureAspect {
                return SIMD2<Float>(Float(textureAspect / drawableAspect), 1)
            } else {
                return SIMD2<Float>(1, Float(drawableAspect / textureAspect))
            }
        }

        private func updateTexture(from result: RasterizationResult) {
            guard let device else { return }

            let requiredByteCount = result.width * result.height * 4
            if pixelBytes.count != requiredByteCount {
                pixelBytes = Array(repeating: 0, count: requiredByteCount)
            }

            for index in 0..<result.pixels.count {
                let pixel = result.pixels[index]
                let byteIndex = index * 4
                pixelBytes[byteIndex + 0] = byte(pixel.z)
                pixelBytes[byteIndex + 1] = byte(pixel.y)
                pixelBytes[byteIndex + 2] = byte(pixel.x)
                pixelBytes[byteIndex + 3] = byte(pixel.w)
            }

            if uploadTexture?.width != result.width || uploadTexture?.height != result.height {
                let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                    pixelFormat: .bgra8Unorm,
                    width: result.width,
                    height: result.height,
                    mipmapped: false
                )
                descriptor.usage = [.shaderRead]
                descriptor.storageMode = .shared
                uploadTexture = device.makeTexture(descriptor: descriptor)
            }

            let region = MTLRegionMake2D(0, 0, result.width, result.height)
            pixelBytes.withUnsafeBytes { bytes in
                guard let baseAddress = bytes.baseAddress else { return }
                uploadTexture?.replace(
                    region: region,
                    mipmapLevel: 0,
                    withBytes: baseAddress,
                    bytesPerRow: result.width * 4
                )
            }
            displayTexture = uploadTexture
        }

        private func byte(_ value: Float) -> UInt8 {
            UInt8((min(max(value, 0), 1) * 255).rounded())
        }

        private func drawWireframeIfNeeded(
            encoder: MTLRenderCommandEncoder?,
            uniforms: ViewportUniforms
        ) {
            guard
                viewModel.showWireframe,
                viewModel.selectedFeature.supportsWireframe,
                let encoder,
                let wireframePipelineState
            else {
                return
            }

            let triangles = viewModel.generateTriangles()
            guard !triangles.isEmpty else { return }

            var vertices: [SIMD2<Float>] = []
            vertices.reserveCapacity(triangles.count * 6)
            for triangle in triangles {
                vertices.append(triangle.0.position)
                vertices.append(triangle.1.position)
                vertices.append(triangle.1.position)
                vertices.append(triangle.2.position)
                vertices.append(triangle.2.position)
                vertices.append(triangle.0.position)
            }

            var uniforms = uniforms
            vertices.withUnsafeBytes { vertexBytes in
                guard let baseAddress = vertexBytes.baseAddress else { return }
                encoder.setRenderPipelineState(wireframePipelineState)
                encoder.setVertexBytes(baseAddress, length: vertexBytes.count, index: 0)
                encoder.setVertexBytes(&uniforms, length: MemoryLayout<ViewportUniforms>.stride, index: 1)
                encoder.drawPrimitives(type: .line, vertexStart: 0, vertexCount: vertices.count)
            }
        }

        private func updateFPS(now: CFTimeInterval) {
            framesInWindow += 1
            let elapsed = now - fpsWindowStart
            guard elapsed >= 1.0 else { return }

            let fps = Double(framesInWindow) / elapsed
            framesInWindow = 0
            fpsWindowStart = now
            publish {
                self.viewModel.updateFPS(fps)
            }
        }

        private func publish(_ update: @escaping () -> Void) {
            DispatchQueue.main.async(execute: update)
        }

        private static let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;

        struct VertexOut {
            float4 position [[position]];
            float2 uv;
        };

        struct ViewportUniforms {
            float2 scale;
        };

        vertex VertexOut fullscreenVertex(
            uint vertexID [[vertex_id]],
            constant ViewportUniforms& uniforms [[buffer(0)]]
        ) {
            const float2 positions[3] = {
                float2(-1.0, -1.0),
                float2( 3.0, -1.0),
                float2(-1.0,  3.0)
            };
            const float2 uvs[3] = {
                float2(0.0, 1.0),
                float2(2.0, 1.0),
                float2(0.0, -1.0)
            };

            VertexOut out;
            out.position = float4(positions[vertexID] * uniforms.scale, 0.0, 1.0);
            out.uv = uvs[vertexID];
            return out;
        }

        fragment float4 fullscreenFragment(
            VertexOut in [[stage_in]],
            texture2d<float> sourceTexture [[texture(0)]]
        ) {
            constexpr sampler textureSampler(
                coord::normalized,
                address::clamp_to_edge,
                filter::linear
            );
            return sourceTexture.sample(textureSampler, in.uv);
        }

        vertex VertexOut wireframeVertex(
            uint vertexID [[vertex_id]],
            device const float2* positions [[buffer(0)]],
            constant ViewportUniforms& uniforms [[buffer(1)]]
        ) {
            VertexOut out;
            out.position = float4(positions[vertexID] * uniforms.scale, 0.0, 1.0);
            out.uv = float2(0.0);
            return out;
        }

        fragment float4 wireframeFragment() {
            return float4(1.0, 1.0, 1.0, 1.0);
        }
        """
    }
}
