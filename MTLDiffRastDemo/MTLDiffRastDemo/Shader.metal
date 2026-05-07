//
//  Shader.metal
//  MTLDiffRastDemo
//
//  Metal shader for hardware-accelerated rasterization
//

#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float2 position [[attribute(0)]];
    float3 color [[attribute(1)]];
    float depth [[attribute(2)]];
};

struct VertexOut {
    float4 position [[position]];
    float3 color;
    float depth;
};

vertex VertexOut vertexShader(const VertexIn in [[stage_in]],
                              constant float4x4 &modelViewProjection [[buffer(1)]]) {
    VertexOut out;
    out.position = modelViewProjection * float4(in.position, in.depth, 1.0);
    out.color = in.color;
    out.depth = in.depth;
    return out;
}

fragment float4 fragmentShader(const VertexOut in [[stage_in]]) {
    return float4(in.color, 1.0);
}
