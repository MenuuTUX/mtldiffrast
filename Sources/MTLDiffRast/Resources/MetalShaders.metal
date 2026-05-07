//
//  MetalShaders.metal
//  MTLDiffRast
//
//  Metal compute shaders for differentiable rasterization.
//  Ported from mtldiffrast-python (Laine et al. 2020, "Modular Primitives
//  for High-Performance Differentiable Rendering").
//

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Section 1: Shared shader types and helpers.
//
// This block defines the parameter structs that the Swift wrapper uploads for
// each kernel invocation, plus tiny utilities shared by multiple passes.
// The rest of the file is split into the forward/backward raster, attribute
// interpolation, display packing, antialiasing, and texture-sampling stages.
// ============================================================================

// Shared params struct.
// imgWidth/imgHeight: output resolution
// xs, xo, ys, yo: pixel-center -> clip-space transform
// xs = 2/W, xo = -1 + 1/W, ys = 2/H, yo = -1 + 1/H
// ============================================================================

struct RasterizeParams {
    int   numTriangles;
    int   numVertices;
    int   width;
    int   height;
    float xs, xo, ys, yo;
};

struct RasterizeBackwardParams {
    int   numTriangles;
    int   numVertices;
    int   width;
    int   height;
    float xs, xo, ys, yo;
    int   enableDB;
};

struct InterpolateParams {
    int numTriangles;
    int numVertices;
    int numAttr;
    int width;
    int height;
};

struct AntialiasParams {
    int   numTriangles;
    int   numVertices;
    int   width;
    int   height;
    int   channels;
    float xh, yh;     // 0.5 * width, 0.5 * height
};

struct TextureParams {
    int   filterMode;     // 0 = nearest, 1 = linear
    int   boundaryMode;   // 0 = wrap, 1 = clamp, 2 = zero
    int   channels;
    int   imgWidth;
    int   imgHeight;
    int   texWidth;
    int   texHeight;
};

struct PackColorTextureParams {
    int width;
    int height;
    int channels;
    int unused;
};

inline float triidx_to_float(int x) {
    if (x <= 0x01000000) return float(x);
    return as_type<float>(0x4a800000 + x);
}

inline int float_to_triidx(float x) {
    if (x <= 16777216.0f) return int(x);
    return as_type<int>(x) - 0x4a800000;
}

// ============================================================================
// Section 2: Rasterization.
//
// These kernels convert clip-space triangles into a per-pixel raster buffer.
// The forward pass writes barycentrics, depth, triangle ID, and barycentric
// derivatives. The backward pass consumes upstream gradients and accumulates
// gradients with respect to vertex clip positions.
// ============================================================================

// Forward rasterization (compute kernel).
// Each thread = one pixel; iterates over all triangles, picks the closest
// front-facing covered triangle, writes (u, v, z, triId+1) and bary derivs.
// ============================================================================

kernel void rasterizeKernel(
    device const float4*           positions   [[buffer(0)]],
    device const int3*             triangles   [[buffer(1)]],
    constant RasterizeParams&      params      [[buffer(2)]],
    device float4*                 rastOut     [[buffer(3)]],   // [H*W] (u, v, z, triId+1)
    device float4*                 rastDB      [[buffer(4)]],   // [H*W] (du/dx, du/dy, dv/dx, dv/dy)
    uint2 gid [[thread_position_in_grid]]
) {
    int px = int(gid.x);
    int py = int(gid.y);
    if (px >= params.width || py >= params.height) return;

    int pidx = px + params.width * py;
    float fx = params.xs * float(px) + params.xo;
    float fy = params.ys * float(py) + params.yo;

    // best-so-far tracking. We track the largest z/w (closest to camera in
    // standard NDC convention used by nvdiffrast).
    float bestZ = -2.0f;
    float bestU = 0.0f, bestV = 0.0f;
    int   bestTri = -1;
    float bestDudx = 0.0f, bestDudy = 0.0f, bestDvdx = 0.0f, bestDvdy = 0.0f;

    for (int t = 0; t < params.numTriangles; t++) {
        int3 tri = triangles[t];
        int vi0 = tri.x;
        int vi1 = tri.y;
        int vi2 = tri.z;

        if (vi0 < 0 || vi0 >= params.numVertices ||
            vi1 < 0 || vi1 >= params.numVertices ||
            vi2 < 0 || vi2 >= params.numVertices) continue;

        float4 p0 = positions[vi0];
        float4 p1 = positions[vi1];
        float4 p2 = positions[vi2];

        if (p0.w == 0.0f || p1.w == 0.0f || p2.w == 0.0f) continue;

        // Cheap clip-space bounding-box rejection before evaluating edges.
        float sx0 = p0.x / p0.w, sy0 = p0.y / p0.w;
        float sx1 = p1.x / p1.w, sy1 = p1.y / p1.w;
        float sx2 = p2.x / p2.w, sy2 = p2.y / p2.w;
        float minX = min(sx0, min(sx1, sx2));
        float maxX = max(sx0, max(sx1, sx2));
        float minY = min(sy0, min(sy1, sy2));
        float maxY = max(sy0, max(sy1, sy2));
        if (fx < minX || fx > maxX || fy < minY || fy > maxY) continue;

        // Edge functions evaluated at the pixel centre in clip space.
        float p0x = p0.x - fx * p0.w;
        float p0y = p0.y - fy * p0.w;
        float p1x = p1.x - fx * p1.w;
        float p1y = p1.y - fy * p1.w;
        float p2x = p2.x - fx * p2.w;
        float p2y = p2.y - fy * p2.w;

        float a0 = p1x * p2y - p1y * p2x;
        float a1 = p2x * p0y - p2y * p0x;
        float a2 = p0x * p1y - p0y * p1x;

        float at = a0 + a1 + a2;
        if (at <= 0.0f) continue;          // back-face / degenerate

        float iw = 1.0f / at;
        float u  = a0 * iw;
        float v  = a1 * iw;

        // Perspective-correct depth z/w (matches Python convention).
        float z = p0.z * a0 + p1.z * a1 + p2.z * a2;
        float w = p0.w * a0 + p1.w * a1 + p2.w * a2;
        float zw = z / w;

        if (u >= 0.0f && v >= 0.0f && (u + v) <= 1.0f && zw >= bestZ) {
            bestZ = zw;

            float dfxdx = params.xs * iw;
            float dfydy = params.ys * iw;
            float da0dx = p2.y * p1.w - p1.y * p2.w;
            float da0dy = p1.x * p2.w - p2.x * p1.w;
            float da1dx = p0.y * p2.w - p2.y * p0.w;
            float da1dy = p2.x * p0.w - p0.x * p2.w;
            float da2dx = p1.y * p0.w - p0.y * p1.w;
            float da2dy = p0.x * p1.w - p1.x * p0.w;
            float datdx = da0dx + da1dx + da2dx;
            float datdy = da0dy + da1dy + da2dy;
            bestDudx = dfxdx * (u * datdx - da0dx);
            bestDudy = dfydy * (u * datdy - da0dy);
            bestDvdx = dfxdx * (v * datdx - da1dx);
            bestDvdy = dfydy * (v * datdy - da1dy);

            // Clamp barycentrics for storage.
            bestU = saturate(u);
            bestV = saturate(v);
            float bs = 1.0f / max(bestU + bestV, 1.0f);
            bestU *= bs;
            bestV *= bs;
            bestTri = t;
        }
    }

    if (bestTri >= 0) {
        // Triangle ID stored as float, with +1 offset so 0 = "no triangle".
        rastOut[pidx] = float4(bestU, bestV, clamp(bestZ, -1.0f, 1.0f), triidx_to_float(bestTri + 1));
        rastDB[pidx]  = float4(bestDudx, bestDudy, bestDvdx, bestDvdy);
    } else {
        rastOut[pidx] = float4(0.0f);
        rastDB[pidx]  = float4(0.0f);
    }
}

// ============================================================================
// Atomic float-add helper (Metal 3.0 fallback via CAS loop).
// ============================================================================

inline void atomicAddFloat(device atomic_uint* addr, float value) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    while (true) {
        float current = as_type<float>(expected);
        float desired = current + value;
        uint  desired_bits = as_type<uint>(desired);
        if (atomic_compare_exchange_weak_explicit(addr, &expected, desired_bits,
                                                   memory_order_relaxed,
                                                   memory_order_relaxed))
            break;
    }
}

// ============================================================================
// Backward rasterization. Computes d(loss)/d(positions) given d(loss)/d(rast).
// dy is the upstream gradient on (u, v, z, triId) per pixel. The reference
// rasterizer gradient uses u/v and optional rastDB gradients; z and triId
// gradients are ignored.
// ============================================================================

kernel void rasterizeBackwardKernel(
    device const float4*           positions   [[buffer(0)]],
    device const int3*             triangles   [[buffer(1)]],
    device const float4*           rastOut     [[buffer(2)]],   // [H*W]
    device const float4*           dy          [[buffer(3)]],   // [H*W] grads on (u, v, z, t)
    device float*                  gradPos     [[buffer(4)]],   // [V * 4] atomic accum
    device const float4*           ddb         [[buffer(5)]],   // [H*W] grads on rastDB
    constant RasterizeBackwardParams& params   [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int px = int(gid.x);
    int py = int(gid.y);
    if (px >= params.width || py >= params.height) return;

    int pidx = px + params.width * py;

    float4 r = rastOut[pidx];
    int triIdx = float_to_triidx(r.w) - 1;
    if (triIdx < 0 || triIdx >= params.numTriangles) return;

    float4 dyv = dy[pidx];
    float dy_x = dyv.x;
    float dy_y = dyv.y;
    float4 ddb_v = params.enableDB != 0 ? ddb[pidx] : float4(0.0f);

    // Quick zero-gradient check (sign bit ignored).
    int grad_all = as_type<int>(dy_x) | as_type<int>(dy_y);
    int grad_all_ddb = params.enableDB != 0
        ? (as_type<int>(ddb_v.x) | as_type<int>(ddb_v.y) | as_type<int>(ddb_v.z) | as_type<int>(ddb_v.w))
        : 0;
    if (((grad_all | grad_all_ddb) << 1) == 0) return;

    int3 tri = triangles[triIdx];
    int vi0 = tri.x, vi1 = tri.y, vi2 = tri.z;
    if (vi0 < 0 || vi0 >= params.numVertices ||
        vi1 < 0 || vi1 >= params.numVertices ||
        vi2 < 0 || vi2 >= params.numVertices) return;

    float4 p0 = positions[vi0];
    float4 p1 = positions[vi1];
    float4 p2 = positions[vi2];

    float fx = params.xs * float(px) + params.xo;
    float fy = params.ys * float(py) + params.yo;

    float p0x = p0.x - fx * p0.w;
    float p0y = p0.y - fy * p0.w;
    float p1x = p1.x - fx * p1.w;
    float p1y = p1.y - fy * p1.w;
    float p2x = p2.x - fx * p2.w;
    float p2y = p2.y - fy * p2.w;

    float a0 = p1x * p2y - p1y * p2x;
    float a1 = p2x * p0y - p2y * p0x;
    float a2 = p0x * p1y - p0y * p1x;
    float at = a0 + a1 + a2;
    float ep = copysign(1e-6f, at);
    float iw = 1.0f / (at + ep);

    float b0 = a0 * iw;
    float b1 = a1 * iw;

    float gb0 = dy_x * iw;
    float gb1 = dy_y * iw;
    float gbb = gb0 * b0 + gb1 * b1;
    float gp0x = gbb * (p2y - p1y) - gb1 * p2y;
    float gp1x = gbb * (p0y - p2y) + gb0 * p2y;
    float gp2x = gbb * (p1y - p0y) - gb0 * p1y + gb1 * p0y;
    float gp0y = gbb * (p1x - p2x) + gb1 * p2x;
    float gp1y = gbb * (p2x - p0x) - gb0 * p2x;
    float gp2y = gbb * (p0x - p1x) + gb0 * p1x - gb1 * p0x;
    float gp0w = -fx * gp0x - fy * gp0y;
    float gp1w = -fx * gp1x - fy * gp1y;
    float gp2w = -fx * gp2x - fy * gp2y;

    if (params.enableDB != 0 && ((grad_all_ddb) << 1) != 0) {
        float dfxdX = params.xs * iw;
        float dfydY = params.ys * iw;
        ddb_v.x *= dfxdX;
        ddb_v.y *= dfydY;
        ddb_v.z *= dfxdX;
        ddb_v.w *= dfydY;

        float da0dX = p1.y * p2.w - p2.y * p1.w;
        float da1dX = p2.y * p0.w - p0.y * p2.w;
        float da2dX = p0.y * p1.w - p1.y * p0.w;
        float da0dY = p2.x * p1.w - p1.x * p2.w;
        float da1dY = p0.x * p2.w - p2.x * p0.w;
        float da2dY = p1.x * p0.w - p0.x * p1.w;
        float datdX = da0dX + da1dX + da2dX;
        float datdY = da0dY + da1dY + da2dY;

        float x01 = p0.x - p1.x;
        float x12 = p1.x - p2.x;
        float x20 = p2.x - p0.x;
        float y01 = p0.y - p1.y;
        float y12 = p1.y - p2.y;
        float y20 = p2.y - p0.y;
        float w01 = p0.w - p1.w;
        float w12 = p1.w - p2.w;
        float w20 = p2.w - p0.w;

        float a0p1 = fy * p2.x - fx * p2.y;
        float a0p2 = fx * p1.y - fy * p1.x;
        float a1p0 = fx * p2.y - fy * p2.x;
        float a1p2 = fy * p0.x - fx * p0.y;

        float wdudX = 2.0f * b0 * datdX - da0dX;
        float wdudY = 2.0f * b0 * datdY - da0dY;
        float wdvdX = 2.0f * b1 * datdX - da1dX;
        float wdvdY = 2.0f * b1 * datdY - da1dY;

        float c0  = iw * (ddb_v.x * wdudX + ddb_v.y * wdudY + ddb_v.z * wdvdX + ddb_v.w * wdvdY);
        float cx  = c0 * fx - ddb_v.x * b0 - ddb_v.z * b1;
        float cy  = c0 * fy - ddb_v.y * b0 - ddb_v.w * b1;
        float cxy = iw * (ddb_v.x * datdX + ddb_v.y * datdY);
        float czw = iw * (ddb_v.z * datdX + ddb_v.w * datdY);

        gp0x += c0 * y12 - cy * w12              + czw * p2y                                  + ddb_v.w * p2.w;
        gp1x += c0 * y20 - cy * w20 - cxy * p2y                     - ddb_v.y * p2.w;
        gp2x += c0 * y01 - cy * w01 + cxy * p1y  - czw * p0y        + ddb_v.y * p1.w - ddb_v.w * p0.w;
        gp0y += cx * w12 - c0 * x12              - czw * p2x                         - ddb_v.z * p2.w;
        gp1y += cx * w20 - c0 * x20 + cxy * p2x                + ddb_v.x * p2.w;
        gp2y += cx * w01 - c0 * x01 - cxy * p1x  + czw * p0x  - ddb_v.x * p1.w + ddb_v.z * p0.w;
        gp0w += cy * x12 - cx * y12              - czw * a1p0                         + ddb_v.z * p2.y - ddb_v.w * p2.x;
        gp1w += cy * x20 - cx * y20 - cxy * a0p1              - ddb_v.x * p2.y + ddb_v.y * p2.x;
        gp2w += cy * x01 - cx * y01 - cxy * a0p2 - czw * a1p2 + ddb_v.x * p1.y - ddb_v.y * p1.x - ddb_v.z * p0.y + ddb_v.w * p0.x;
    }

    // Atomically accumulate into gradPos (laid out as [V*4] float).
    device atomic_uint* g0 = (device atomic_uint*)(gradPos + 4 * vi0);
    device atomic_uint* g1 = (device atomic_uint*)(gradPos + 4 * vi1);
    device atomic_uint* g2 = (device atomic_uint*)(gradPos + 4 * vi2);
    atomicAddFloat(&g0[0], gp0x);
    atomicAddFloat(&g0[1], gp0y);
    atomicAddFloat(&g0[3], gp0w);
    atomicAddFloat(&g1[0], gp1x);
    atomicAddFloat(&g1[1], gp1y);
    atomicAddFloat(&g1[3], gp1w);
    atomicAddFloat(&g2[0], gp2x);
    atomicAddFloat(&g2[1], gp2y);
    atomicAddFloat(&g2[3], gp2w);
}

// ============================================================================
// Section 3: Attribute interpolation and display packing.
//
// Once rasterization identifies the winning triangle per pixel, these kernels
// interpolate vertex attributes across the image and optionally repack the
// result into a display-friendly BGRA texture for the demo app.
// ============================================================================

// Forward interpolation.
// For each pixel in the rast buffer, looks up the triangle, fetches the three
// vertex attributes, and blends via barycentrics (b0, b1, 1-b0-b1).
// ============================================================================

kernel void interpolateKernel(
    device const int3*             triangles  [[buffer(0)]],
    device const float*            attributes [[buffer(1)]],   // [V * A]
    device const float4*           rastOut    [[buffer(2)]],   // [H*W]
    constant InterpolateParams&    params     [[buffer(3)]],
    device float*                  output     [[buffer(4)]],   // [H*W*A]
    device float*                  baryCoords [[buffer(5)]],   // [H*W*3]
    uint2 gid [[thread_position_in_grid]]
) {
    int px = int(gid.x);
    int py = int(gid.y);
    if (px >= params.width || py >= params.height) return;

    int pidx = px + params.width * py;
    int A = params.numAttr;
    int outBase  = pidx * A;
    int baryBase = pidx * 3;

    float4 r = rastOut[pidx];
    int triIdx = float_to_triidx(r.w) - 1;

    if (triIdx < 0 || triIdx >= params.numTriangles) {
        for (int i = 0; i < A; i++) output[outBase + i] = 0.0f;
        baryCoords[baryBase + 0] = 0.0f;
        baryCoords[baryBase + 1] = 0.0f;
        baryCoords[baryBase + 2] = 0.0f;
        return;
    }

    int3 tri = triangles[triIdx];
    int vi0 = tri.x, vi1 = tri.y, vi2 = tri.z;
    if (vi0 < 0 || vi0 >= params.numVertices ||
        vi1 < 0 || vi1 >= params.numVertices ||
        vi2 < 0 || vi2 >= params.numVertices) {
        for (int i = 0; i < A; i++) output[outBase + i] = 0.0f;
        baryCoords[baryBase + 0] = 0.0f;
        baryCoords[baryBase + 1] = 0.0f;
        baryCoords[baryBase + 2] = 0.0f;
        return;
    }

    float b0 = r.x;
    float b1 = r.y;
    float b2 = 1.0f - b0 - b1;

    device const float* a0 = attributes + vi0 * A;
    device const float* a1 = attributes + vi1 * A;
    device const float* a2 = attributes + vi2 * A;

    for (int i = 0; i < A; i++) {
        output[outBase + i] = b0 * a0[i] + b1 * a1[i] + b2 * a2[i];
    }

    baryCoords[baryBase + 0] = b0;
    baryCoords[baryBase + 1] = b1;
    baryCoords[baryBase + 2] = b2;
}

// ============================================================================
// Display packing. Converts a float colour buffer into a BGRA texture without
// readback. The float buffers use the rasterizer's bottom-left origin, while the
// demo texture is written top-left for presentation.
// ============================================================================

kernel void packColorTextureKernel(
    device const float*                 color      [[buffer(0)]], // [H*W*C]
    device const float4*                rastOut    [[buffer(1)]], // [H*W]
    constant PackColorTextureParams&    params     [[buffer(2)]],
    texture2d<float, access::write>     outTexture [[texture(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int px = int(gid.x);
    int py = int(gid.y);
    if (px >= params.width || py >= params.height) return;

    int sourceY = params.height - 1 - py;
    int sourceIndex = sourceY * params.width + px;
    int outBase = sourceIndex * params.channels;

    float r = params.channels > 0 ? color[outBase + 0] : 0.0f;
    float g = params.channels > 1 ? color[outBase + 1] : r;
    float b = params.channels > 2 ? color[outBase + 2] : r;
    float maxColor = max(r, max(g, b));

    int triIdx = float_to_triidx(rastOut[sourceIndex].w) - 1;
    if (triIdx < 0 && maxColor <= 0.001f) {
        outTexture.write(float4(0.0f, 0.0f, 0.0f, 1.0f), uint2(px, py));
        return;
    }

    outTexture.write(float4(saturate(r), saturate(g), saturate(b), 1.0f), uint2(px, py));
}

// ============================================================================
// Backward interpolation: gradients on attributes + barycentrics.
// dy is [H*W*A] gradient on output. Writes:
//   gradAttr[V*A] (atomic accum)
//   gradRast[H*W*4] (only .xy used — db0, db1)
// ============================================================================

kernel void interpolateBackwardKernel(
    device const int3*             triangles  [[buffer(0)]],
    device const float*            attributes [[buffer(1)]],
    device const float4*           rastOut    [[buffer(2)]],
    device const float*            dy         [[buffer(3)]],   // [H*W*A]
    device float*                  gradAttr   [[buffer(4)]],   // [V*A] atomic
    device float4*                 gradRast   [[buffer(5)]],   // [H*W]
    constant InterpolateParams&    params     [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int px = int(gid.x);
    int py = int(gid.y);
    if (px >= params.width || py >= params.height) return;

    int pidx = px + params.width * py;
    int A = params.numAttr;

    float4 r = rastOut[pidx];
    int triIdx = float_to_triidx(r.w) - 1;
    if (triIdx < 0 || triIdx >= params.numTriangles) {
        gradRast[pidx] = float4(0.0f);
        return;
    }

    int3 tri = triangles[triIdx];
    int vi0 = tri.x, vi1 = tri.y, vi2 = tri.z;
    if (vi0 < 0 || vi0 >= params.numVertices ||
        vi1 < 0 || vi1 >= params.numVertices ||
        vi2 < 0 || vi2 >= params.numVertices) return;

    device const float* a0  = attributes + vi0 * A;
    device const float* a1  = attributes + vi1 * A;
    device const float* a2  = attributes + vi2 * A;
    device const float* pdy = dy + pidx * A;

    float b0 = r.x;
    float b1 = r.y;
    float b2 = 1.0f - b0 - b1;
    float gb0 = 0.0f, gb1 = 0.0f;

    for (int i = 0; i < A; i++) {
        float y  = pdy[i];
        float s0 = a0[i];
        float s1 = a1[i];
        float s2 = a2[i];
        gb0 += y * (s0 - s2);
        gb1 += y * (s1 - s2);
        atomicAddFloat((device atomic_uint*)(gradAttr + vi0 * A + i), b0 * y);
        atomicAddFloat((device atomic_uint*)(gradAttr + vi1 * A + i), b1 * y);
        atomicAddFloat((device atomic_uint*)(gradAttr + vi2 * A + i), b2 * y);
    }

    gradRast[pidx] = float4(gb0, gb1, 0.0f, 0.0f);
}

// ============================================================================
// Section 4: Antialiasing support.
//
// This block smooths silhouette discontinuities after shading. The copy kernel
// seeds a writable float buffer, and the AA kernel applies differentiable edge
// corrections by atomically blending across triangle-ID boundaries.
// ============================================================================

// Antialias forward — simple silhouette-edge filter.
//
// This is a stripped-down version of nvdiffrast's silhouette AA: when two
// adjacent pixels belong to different triangles, blend their colours along the
// edge crossing the pixel boundary. The result is a continuous function of
// vertex positions, which is what makes it differentiable.
//
// We don't build the full edge-vertex hash here (the Python version does, for
// open-mesh edges); this single-pass version handles the common closed-mesh
// case where any tri-id discontinuity is a silhouette.
// ============================================================================

// NOTE: callers MUST initialise `output` to a per-pixel copy of `color` before
// dispatching this kernel. The kernel only writes silhouette corrections via
// atomic adds — the base colour pass-through is the caller's responsibility.
// (Doing the init copy inside the kernel races with neighbour threads' atomic
// adds targeting the same pixel, since Metal compute has no global barrier.)
kernel void antialiasKernel(
    device const float*            color      [[buffer(0)]],   // [H*W*C]
    device const float4*           rastOut    [[buffer(1)]],
    device const int3*             triangles  [[buffer(2)]],
    device const float4*           positions  [[buffer(3)]],
    device float*                  output     [[buffer(4)]],   // [H*W*C], pre-filled with color
    constant AntialiasParams&      params     [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int px = int(gid.x);
    int py = int(gid.y);
    if (px >= params.width || py >= params.height) return;

    int pidx0 = px + params.width * py;
    int C = params.channels;

    // Look right.
    int tri0 = float_to_triidx(rastOut[pidx0].w) - 1;
    if (px < params.width - 1) {
        int pidx1 = pidx0 + 1;
        int tri1 = float_to_triidx(rastOut[pidx1].w) - 1;
        if (tri0 != tri1) {
            bool skipEdge = false;
            if (tri0 >= 0 && tri1 >= 0) {
                int3 a = triangles[tri0];
                int3 b = triangles[tri1];
                int shared = 0;
                shared += (a.x == b.x || a.x == b.y || a.x == b.z) ? 1 : 0;
                shared += (a.y == b.x || a.y == b.y || a.y == b.z) ? 1 : 0;
                shared += (a.z == b.x || a.z == b.y || a.z == b.z) ? 1 : 0;
                skipEdge = shared >= 2;
            }
            // Pick the foreground triangle by depth.
            int triSel = tri0 >= 0 ? tri0 : tri1;
            if (tri0 >= 0 && tri1 >= 0) {
                triSel = (rastOut[pidx0].z < rastOut[pidx1].z) ? tri0 : tri1;
            }
            if (!skipEdge && triSel >= 0 && triSel < params.numTriangles) {
                int3 tri = triangles[triSel];
                float4 p0 = positions[tri.x];
                float4 p1 = positions[tri.y];
                float4 p2 = positions[tri.z];

                // Project to pixel-space (centre).
                float w0 = 1.0f / p0.w;
                float w1 = 1.0f / p1.w;
                float w2 = 1.0f / p2.w;
                float fx = float(px) + 0.5f - params.xh;
                float fy = float(py) + 0.5f - params.yh;
                float x0 = p0.x * w0 * params.xh - fx;
                float y0 = p0.y * w0 * params.yh - fy;
                float x1 = p1.x * w1 * params.xh - fx;
                float y1 = p1.y * w1 * params.yh - fy;
                float x2 = p2.x * w2 * params.xh - fx;
                float y2 = p2.y * w2 * params.yh - fy;

                // Find closest edge crossing in x (vertical edge between pixels).
                float bb = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
                float ds = (triSel == tri0) ? 1.0f : -1.0f;

                float dx0 = x2 - x1, dx1 = x0 - x2, dx2 = x1 - x0;
                float dy0 = y2 - y1, dy1 = y0 - y2, dy2 = y1 - y0;
                float d0 = ds * (x1 * dy0 - y1 * dx0);
                float d1 = ds * (x2 * dy1 - y2 * dx1);
                float d2 = ds * (x0 * dy2 - y0 * dx2);

                float dc = -1.0e30f;
                float bestDy = 1.0f;
                if (fabs(dy0) >= fabs(dx0) && (dy0 != 0.0f) && (bb * dy0) >= 0.0f) { dc = d0 / dy0; bestDy = dy0; }
                if (fabs(dy1) >= fabs(dx1) && (dy1 != 0.0f) && (bb * dy1) >= 0.0f) {
                    float c = d1 / dy1;
                    if (c > dc) { dc = c; bestDy = dy1; }
                }
                if (fabs(dy2) >= fabs(dx2) && (dy2 != 0.0f) && (bb * dy2) >= 0.0f) {
                    float c = d2 / dy2;
                    if (c > dc) { dc = c; bestDy = dy2; }
                }
                (void)bestDy;

                float eps = 0.0625f;
                if (dc > -eps && dc < 1.0f + eps) {
                    dc = clamp(dc, 0.0f, 1.0f);
                    float alpha = ds * (0.5f - dc);
                    int dst = (alpha > 0.0f) ? pidx0 : pidx1;
                    for (int i = 0; i < C; i++) {
                        float c0 = color[pidx0 * C + i];
                        float c1 = color[pidx1 * C + i];
                        atomicAddFloat((device atomic_uint*)(output + dst * C + i),
                                       alpha * (c1 - c0));
                    }
                }
            }
        }
    }

    // Look down.
    if (py < params.height - 1) {
        int pidx1 = pidx0 + params.width;
        int tri1 = float_to_triidx(rastOut[pidx1].w) - 1;
        if (tri0 != tri1) {
            bool skipEdge = false;
            if (tri0 >= 0 && tri1 >= 0) {
                int3 a = triangles[tri0];
                int3 b = triangles[tri1];
                int shared = 0;
                shared += (a.x == b.x || a.x == b.y || a.x == b.z) ? 1 : 0;
                shared += (a.y == b.x || a.y == b.y || a.y == b.z) ? 1 : 0;
                shared += (a.z == b.x || a.z == b.y || a.z == b.z) ? 1 : 0;
                skipEdge = shared >= 2;
            }
            int triSel = tri0 >= 0 ? tri0 : tri1;
            if (tri0 >= 0 && tri1 >= 0) {
                triSel = (rastOut[pidx0].z < rastOut[pidx1].z) ? tri0 : tri1;
            }
            if (!skipEdge && triSel >= 0 && triSel < params.numTriangles) {
                int3 tri = triangles[triSel];
                float4 p0 = positions[tri.x];
                float4 p1 = positions[tri.y];
                float4 p2 = positions[tri.z];
                float w0 = 1.0f / p0.w;
                float w1 = 1.0f / p1.w;
                float w2 = 1.0f / p2.w;
                float fx = float(px) + 0.5f - params.xh;
                float fy = float(py) + 0.5f - params.yh;
                // For horizontal edges we swap x and y (matches Python flip path).
                float x0 = p0.y * w0 * params.yh - fy;
                float y0 = p0.x * w0 * params.xh - fx;
                float x1 = p1.y * w1 * params.yh - fy;
                float y1 = p1.x * w1 * params.xh - fx;
                float x2 = p2.y * w2 * params.yh - fy;
                float y2 = p2.x * w2 * params.xh - fx;

                float bb = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
                float ds = (triSel == tri0) ? 1.0f : -1.0f;

                float dx0 = x2 - x1, dx1 = x0 - x2, dx2 = x1 - x0;
                float dy0 = y2 - y1, dy1 = y0 - y2, dy2 = y1 - y0;
                float d0 = ds * (x1 * dy0 - y1 * dx0);
                float d1 = ds * (x2 * dy1 - y2 * dx1);
                float d2 = ds * (x0 * dy2 - y0 * dx2);

                float dc = -1.0e30f;
                if (fabs(dy0) >= fabs(dx0) && (dy0 != 0.0f) && (bb * dy0) >= 0.0f) dc = max(dc, d0 / dy0);
                if (fabs(dy1) >= fabs(dx1) && (dy1 != 0.0f) && (bb * dy1) >= 0.0f) dc = max(dc, d1 / dy1);
                if (fabs(dy2) >= fabs(dx2) && (dy2 != 0.0f) && (bb * dy2) >= 0.0f) dc = max(dc, d2 / dy2);

                float eps = 0.0625f;
                if (dc > -eps && dc < 1.0f + eps) {
                    dc = clamp(dc, 0.0f, 1.0f);
                    float alpha = ds * (0.5f - dc);
                    int dst = (alpha > 0.0f) ? pidx0 : pidx1;
                    for (int i = 0; i < C; i++) {
                        float c0 = color[pidx0 * C + i];
                        float c1 = color[pidx1 * C + i];
                        atomicAddFloat((device atomic_uint*)(output + dst * C + i),
                                       alpha * (c1 - c0));
                    }
                }
            }
        }
    }
}

// ============================================================================
// Float-buffer copy — used to seed the antialias output buffer with a copy of
// the colour buffer when the caller can't blit before the AA dispatch (e.g.
// `rasterizeColorTexture`, where everything runs in a single compute encoder).
// `count` is total float elements (H * W * channels).
// ============================================================================

struct CopyParams {
    int count;
};

kernel void copyFloatBufferKernel(
    device const float*       src   [[buffer(0)]],
    device float*             dst   [[buffer(1)]],
    constant CopyParams&      p     [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    int i = int(gid);
    if (i < p.count) dst[i] = src[i];
}

// ============================================================================
// Section 5: Texture sampling.
//
// These helpers and kernels implement nearest/bilinear texture reads with
// wrap, clamp, or zero boundary behavior. The forward path produces sampled
// texels per pixel, while the backward path propagates gradients to both texels
// and UVs when linear filtering is used.
// ============================================================================

// Texture sampling — nearest + bilinear, wrap/clamp/zero boundary.
// uv layout: [H*W*2], output: [H*W*C].
// ============================================================================

inline int2 wrapTexel(int iu, int iv, int w, int h, int boundaryMode, thread bool& outOfBounds) {
    outOfBounds = false;
    if (boundaryMode == 0) {       // wrap
        iu = ((iu % w) + w) % w;
        iv = ((iv % h) + h) % h;
    } else if (boundaryMode == 1) { // clamp
        iu = clamp(iu, 0, w - 1);
        iv = clamp(iv, 0, h - 1);
    } else {                       // zero
        if (iu < 0 || iu >= w || iv < 0 || iv >= h) {
            outOfBounds = true;
        }
    }
    return int2(iu, iv);
}

kernel void textureKernel(
    device const float*            texture     [[buffer(0)]],   // [TexH*TexW*C]
    device const float2*           uv          [[buffer(1)]],   // [H*W]
    constant TextureParams&        params      [[buffer(2)]],
    device float*                  output      [[buffer(3)]],   // [H*W*C]
    uint2 gid [[thread_position_in_grid]]
) {
    int px = int(gid.x);
    int py = int(gid.y);
    if (px >= params.imgWidth || py >= params.imgHeight) return;

    int pidx = px + params.imgWidth * py;
    int C = params.channels;
    int W = params.texWidth;
    int H = params.texHeight;
    device float* pOut = output + pidx * C;

    float2 st = uv[pidx];
    float u = st.x;
    float v = st.y;

    if (params.boundaryMode == 0) { // wrap: bring into [0,1)
        u = u - floor(u);
        v = v - floor(v);
    }

    if (params.filterMode == 0) {
        // Nearest.
        int iu = int(floor(u * float(W)));
        int iv = int(floor(v * float(H)));
        bool oob = false;
        int2 tc = wrapTexel(iu, iv, W, H, params.boundaryMode, oob);
        if (oob) {
            for (int i = 0; i < C; i++) pOut[i] = 0.0f;
            return;
        }
        int base = (tc.x + W * tc.y) * C;
        for (int i = 0; i < C; i++) pOut[i] = texture[base + i];
        return;
    }

    // Bilinear.
    float fu = u * float(W) - 0.5f;
    float fv = v * float(H) - 0.5f;
    int iu0 = int(floor(fu));
    int iv0 = int(floor(fv));
    int iu1 = iu0 + 1;
    int iv1 = iv0 + 1;
    float du = fu - float(iu0);
    float dv = fv - float(iv0);

    bool oob00, oob10, oob01, oob11;
    int2 tc00 = wrapTexel(iu0, iv0, W, H, params.boundaryMode, oob00);
    int2 tc10 = wrapTexel(iu1, iv0, W, H, params.boundaryMode, oob10);
    int2 tc01 = wrapTexel(iu0, iv1, W, H, params.boundaryMode, oob01);
    int2 tc11 = wrapTexel(iu1, iv1, W, H, params.boundaryMode, oob11);

    int b00 = (tc00.x + W * tc00.y) * C;
    int b10 = (tc10.x + W * tc10.y) * C;
    int b01 = (tc01.x + W * tc01.y) * C;
    int b11 = (tc11.x + W * tc11.y) * C;

    for (int i = 0; i < C; i++) {
        float v00 = oob00 ? 0.0f : texture[b00 + i];
        float v10 = oob10 ? 0.0f : texture[b10 + i];
        float v01 = oob01 ? 0.0f : texture[b01 + i];
        float v11 = oob11 ? 0.0f : texture[b11 + i];
        float a = v00 + (v10 - v00) * du;
        float b = v01 + (v11 - v01) * du;
        pOut[i] = a + (b - a) * dv;
    }
}

kernel void textureBackwardKernel(
    device const float*            texture     [[buffer(0)]],   // [TexH*TexW*C]
    device const float2*           uv          [[buffer(1)]],   // [H*W]
    device const float*            dy          [[buffer(2)]],   // [H*W*C]
    constant TextureParams&        params      [[buffer(3)]],
    device float*                  gradTexture [[buffer(4)]],   // [TexH*TexW*C]
    device float2*                 gradUV      [[buffer(5)]],   // [H*W]
    uint2 gid [[thread_position_in_grid]]
) {
    int px = int(gid.x);
    int py = int(gid.y);
    if (px >= params.imgWidth || py >= params.imgHeight) return;

    int pidx = px + params.imgWidth * py;
    int C = params.channels;
    int W = params.texWidth;
    int H = params.texHeight;

    float2 st = uv[pidx];
    float u = st.x;
    float v = st.y;
    if (params.boundaryMode == 0) {
        u = u - floor(u);
        v = v - floor(v);
    }

    float gu = 0.0f;
    float gv = 0.0f;

    if (params.filterMode == 0) {
        int iu = int(floor(u * float(W)));
        int iv = int(floor(v * float(H)));
        bool oob = false;
        int2 tc = wrapTexel(iu, iv, W, H, params.boundaryMode, oob);
        if (!oob) {
            int base = (tc.x + W * tc.y) * C;
            for (int i = 0; i < C; i++) {
                atomicAddFloat((device atomic_uint*)(gradTexture + base + i), dy[pidx * C + i]);
            }
        }
        gradUV[pidx] = float2(0.0f);
        return;
    }

    float fu = u * float(W) - 0.5f;
    float fv = v * float(H) - 0.5f;
    int iu0 = int(floor(fu));
    int iv0 = int(floor(fv));
    int iu1 = iu0 + 1;
    int iv1 = iv0 + 1;
    float du = fu - float(iu0);
    float dv = fv - float(iv0);

    bool oob00, oob10, oob01, oob11;
    int2 tc00 = wrapTexel(iu0, iv0, W, H, params.boundaryMode, oob00);
    int2 tc10 = wrapTexel(iu1, iv0, W, H, params.boundaryMode, oob10);
    int2 tc01 = wrapTexel(iu0, iv1, W, H, params.boundaryMode, oob01);
    int2 tc11 = wrapTexel(iu1, iv1, W, H, params.boundaryMode, oob11);

    int b00 = (tc00.x + W * tc00.y) * C;
    int b10 = (tc10.x + W * tc10.y) * C;
    int b01 = (tc01.x + W * tc01.y) * C;
    int b11 = (tc11.x + W * tc11.y) * C;

    float w00 = (1.0f - du) * (1.0f - dv);
    float w10 = du * (1.0f - dv);
    float w01 = (1.0f - du) * dv;
    float w11 = du * dv;

    for (int i = 0; i < C; i++) {
        float y = dy[pidx * C + i];
        float v00 = oob00 ? 0.0f : texture[b00 + i];
        float v10 = oob10 ? 0.0f : texture[b10 + i];
        float v01 = oob01 ? 0.0f : texture[b01 + i];
        float v11 = oob11 ? 0.0f : texture[b11 + i];

        if (!oob00) atomicAddFloat((device atomic_uint*)(gradTexture + b00 + i), w00 * y);
        if (!oob10) atomicAddFloat((device atomic_uint*)(gradTexture + b10 + i), w10 * y);
        if (!oob01) atomicAddFloat((device atomic_uint*)(gradTexture + b01 + i), w01 * y);
        if (!oob11) atomicAddFloat((device atomic_uint*)(gradTexture + b11 + i), w11 * y);

        gu += y * ((v10 - v00) * (1.0f - dv) + (v11 - v01) * dv) * float(W);
        gv += y * ((v01 - v00) * (1.0f - du) + (v11 - v10) * du) * float(H);
    }

    gradUV[pidx] = float2(gu, gv);
}
