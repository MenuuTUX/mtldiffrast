//
//  RasterizerOutput.swift
//  MTLDiffRast
//
//  Value-type outputs returned by Rasterizer's public API.
//
//  All arrays are row-major with pixel (0, 0) at the bottom-left, matching
//  the OpenGL / nvdiffrast clip-space convention.
//

import Foundation
import simd

// MARK: - Rasterize

/// Output from a forward rasterization pass.
///
/// Each pixel stores the two leading barycentric coordinates `(u, v)` of the
/// nearest front-facing triangle, its perspective-correct depth `z/w`, and the
/// zero-based triangle index (`-1` for background pixels).
///
/// The Metal kernel writes one `float4` per pixel laid out as
/// `(u, v, z/w, encodedTriId)`.  This struct decodes that buffer into
/// typed Swift arrays that downstream operations can consume directly.
///
/// Pass this value to ``Rasterizer/interpolate(attributes:triangles:rasterOutput:numAttributes:computeDerivatives:)``
/// and ``Rasterizer/antialias(color:channels:rasterOutput:positions:triangles:)``
/// without any manual unpacking.
public struct RasterOutput {

    /// Width of the rasterized image in pixels.
    public let width: Int

    /// Height of the rasterized image in pixels.
    public let height: Int

    /// Zero-based triangle index per pixel, or `-1` for background pixels.
    ///
    /// Layout: `[height × width]`, row-major, `y = 0` is the bottom row.
    public let triangleIds: [Int32]

    /// Perspective-correct depth `z/w` per pixel.
    ///
    /// Larger values are closer to the camera (standard NDC depth).
    /// Background pixels have depth `0`.
    ///
    /// Layout: `[height × width]`, row-major.
    public let depthBuffer: [Float]

    /// Leading two barycentric coordinates `(u, v)` per pixel.
    ///
    /// The third coordinate is `w = 1 − u − v`.  Background pixels have
    /// `(0, 0)`.
    ///
    /// Layout: `[height × width]`, row-major.
    public let barycentrics: [SIMD2<Float>]

    /// Screen-space derivatives of the barycentric coordinates.
    ///
    /// Each element is `(du/dx, du/dy, dv/dx, dv/dy)` for the corresponding
    /// pixel.  Pass this to
    /// ``Rasterizer/interpolate(attributes:triangles:rasterOutput:numAttributes:computeDerivatives:)``
    /// with `computeDerivatives: true` to obtain attribute screen-space
    /// derivatives, or supply them to ``Rasterizer/rasterizeBackward(positions:triangles:forwardOutput:gradOutput:vertexCount:gradBaryDerivatives:)``
    /// for second-order gradient flow.
    ///
    /// Layout: `[height × width]`, row-major.
    public let baryDerivatives: [SIMD4<Float>]

    /// Total number of pixels (`width × height`).
    public var pixelCount: Int { width * height }

    /// Creates a ``RasterOutput`` from decoded kernel output.
    ///
    /// You typically receive this value from
    /// ``Rasterizer/rasterize(positions:triangles:width:height:)``
    /// rather than constructing it directly.
    public init(
        width: Int,
        height: Int,
        triangleIds: [Int32],
        depthBuffer: [Float],
        barycentrics: [SIMD2<Float>],
        baryDerivatives: [SIMD4<Float>]
    ) {
        self.width           = width
        self.height          = height
        self.triangleIds     = triangleIds
        self.depthBuffer     = depthBuffer
        self.barycentrics    = barycentrics
        self.baryDerivatives = baryDerivatives
    }
}

// MARK: - Interpolate

/// Output from an attribute interpolation pass.
///
/// Each pixel's attributes are the barycentric-weighted combination of the
/// three vertex attributes of the covering triangle.  The optional
/// `attributeDerivatives` array contains screen-space `(dx, dy)` derivatives
/// for every attribute channel, which are needed to compute texture MIP levels
/// or higher-order losses.
///
/// - SeeAlso: ``Rasterizer/interpolate(attributes:triangles:rasterOutput:numAttributes:computeDerivatives:)``
public struct InterpolateOutput {

    /// Number of pixels in the output image (`width × height`).
    public let pixelCount: Int

    /// Number of attribute channels per vertex / pixel.
    public let numAttributes: Int

    /// Interpolated per-pixel attributes.
    ///
    /// Layout: `[pixelCount × numAttributes]`, inner axis is the attribute index.
    public let attributes: [Float]

    /// Full barycentric coordinates `(b0, b1, b2)` per pixel, where
    /// `b2 = 1 − b0 − b1`.
    ///
    /// Layout: `[pixelCount × 3]`.
    public let barycentricCoords: [Float]

    /// Optional screen-space attribute derivatives.
    ///
    /// `nil` unless `computeDerivatives: true` was passed to the interpolate
    /// call.  When present, layout is `[pixelCount × numAttributes × 2]`
    /// with the innermost axis being `[dx, dy]` for each attribute.
    public let attributeDerivatives: [Float]?

    /// Creates an ``InterpolateOutput``.
    ///
    /// You typically receive this value from
    /// ``Rasterizer/interpolate(attributes:triangles:rasterOutput:numAttributes:computeDerivatives:)``
    /// rather than constructing it directly.
    public init(
        pixelCount: Int,
        numAttributes: Int,
        attributes: [Float],
        barycentricCoords: [Float],
        attributeDerivatives: [Float]? = nil
    ) {
        self.pixelCount          = pixelCount
        self.numAttributes       = numAttributes
        self.attributes          = attributes
        self.barycentricCoords   = barycentricCoords
        self.attributeDerivatives = attributeDerivatives
    }
}

// MARK: - Texture

/// Output from a texture sampling pass.
///
/// - SeeAlso: ``Rasterizer/texture(texture:texWidth:texHeight:channels:uv:outWidth:outHeight:filterMode:boundaryMode:)``
public struct TextureOutput {

    /// Width of the sampled output image.
    public let width: Int

    /// Height of the sampled output image.
    public let height: Int

    /// Number of channels in the sampled texture.
    public let channels: Int

    /// Sampled texture values.
    ///
    /// Layout: `[height × width × channels]`, row-major, channel-last.
    public let samples: [Float]

    /// Total number of output pixels (`width × height`).
    public var pixelCount: Int { width * height }

    /// Creates a ``TextureOutput``.
    public init(width: Int, height: Int, channels: Int, samples: [Float]) {
        self.width    = width
        self.height   = height
        self.channels = channels
        self.samples  = samples
    }
}

/// Gradient output from a texture-sampling backward pass.
///
/// - SeeAlso: ``Rasterizer/textureBackward(texture:texWidth:texHeight:channels:uv:gradOutput:outWidth:outHeight:filterMode:boundaryMode:)``
public struct TextureGradientOutput {

    /// Gradient with respect to the texture texels.
    ///
    /// Layout: `[texHeight × texWidth × channels]`, channel-last.
    /// For nearest-filter sampling this always matches the forward scatter
    /// from a single texel; for bilinear sampling it is the sum of four
    /// weighted contributions.
    public let textureGradients: [Float]

    /// Gradient with respect to the per-pixel UV coordinates.
    ///
    /// For nearest-filter sampling all elements are zero because the floor
    /// operation is non-differentiable.
    ///
    /// Layout: `[outHeight × outWidth]`.
    public let uvGradients: [SIMD2<Float>]

    /// Creates a ``TextureGradientOutput``.
    public init(textureGradients: [Float], uvGradients: [SIMD2<Float>]) {
        self.textureGradients = textureGradients
        self.uvGradients      = uvGradients
    }
}

// MARK: - Antialias

/// Output from a silhouette antialiasing pass.
///
/// The color buffer contains the source image with sub-pixel silhouette
/// corrections applied via atomic scatter.  The values remain in the range
/// `[0, 1]` for a `[0, 1]` input.
///
/// - SeeAlso: ``Rasterizer/antialias(color:channels:rasterOutput:positions:triangles:)``
public struct AntialiasOutput {

    /// Width of the antialiased image.
    public let width: Int

    /// Height of the antialiased image.
    public let height: Int

    /// Number of colour channels (matches the input colour buffer).
    public let channels: Int

    /// Antialiased colour values.
    ///
    /// Layout: `[height × width × channels]`, row-major, channel-last.
    public let colors: [Float]

    /// Total number of output pixels (`width × height`).
    public var pixelCount: Int { width * height }

    /// Creates an ``AntialiasOutput``.
    public init(width: Int, height: Int, channels: Int, colors: [Float]) {
        self.width    = width
        self.height   = height
        self.channels = channels
        self.colors   = colors
    }
}

// MARK: - Gradient outputs

/// Gradient outputs from the rasterize backward pass.
///
/// - SeeAlso: ``Rasterizer/rasterizeBackward(positions:triangles:forwardOutput:gradOutput:vertexCount:gradBaryDerivatives:)``
public struct RasterGradientOutput {

    /// Gradient with respect to each vertex position.
    ///
    /// Layout: `[vertexCount × 4]` (`x`, `y`, `z`, `w` per vertex).
    public let positionGradients: [Float]

    /// Gradient with respect to triangle indices.
    ///
    /// Always `nil` — integer triangle indices are not differentiable.
    public let triangleGradients: [Float]?

    /// Creates a ``RasterGradientOutput``.
    public init(positionGradients: [Float], triangleGradients: [Float]? = nil) {
        self.positionGradients  = positionGradients
        self.triangleGradients  = triangleGradients
    }
}

// MARK: - Enumerations

/// Texture sampling filter mode used by
/// ``Rasterizer/texture(texture:texWidth:texHeight:channels:uv:outWidth:outHeight:filterMode:boundaryMode:)``.
public enum TextureFilterMode: Int32 {

    /// Nearest-neighbour sampling: uses the texel whose centre is closest
    /// to the sample UV.  Fast; UV gradients are always zero.
    case nearest = 0

    /// Bilinear sampling: blends the four surrounding texels weighted by
    /// fractional UV.  Produces smooth results and non-zero UV gradients in
    /// the backward pass.
    case linear  = 1
}

/// Texture boundary mode applied when UVs fall outside `[0, 1]`.
///
/// Used by
/// ``Rasterizer/texture(texture:texWidth:texHeight:channels:uv:outWidth:outHeight:filterMode:boundaryMode:)``.
public enum TextureBoundaryMode: Int32 {

    /// UVs wrap modulo the texture dimensions (tiling).
    case wrap  = 0

    /// UVs are clamped to the valid `[0, 1]` range (border replication).
    case clamp = 1

    /// Texels outside `[0, 1]` contribute zero (black border).
    case zero  = 2
}
