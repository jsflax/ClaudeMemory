#include <metal_stdlib>
#include <RealityKit/RealityKit.h>
using namespace metal;

/// Edge surface shader — per-instance color, transparency, selection highlight.
/// Vertex color RGB = edge color, A = alpha (selection/search state).

[[visible]]
void edgeSurface(realitykit::surface_parameters params)
{
    float4 vertColor = params.geometry().color();
    float3 color = vertColor.rgb;
    float alpha = vertColor.a;

    // Slight emission for selected edges (alpha > 0.7)
    float3 emission = float3(0.0);
    if (alpha > 0.7) {
        emission = color * 0.3;
    }

    // Distance fog using view-space depth
    float3 viewPos = (params.uniforms().model_to_view() * float4(params.geometry().model_position(), 1.0)).xyz;
    float dist = length(viewPos);
    float fogFactor = clamp((dist - 100.0) / 1500.0, 0.0, 0.9);
    float3 fogColor = float3(0.051, 0.067, 0.09);
    color = mix(color, fogColor, fogFactor);
    emission *= (1.0 - fogFactor);
    alpha *= (1.0 - fogFactor * 0.5);

    params.surface().set_base_color(half3(color));
    params.surface().set_emissive_color(half3(emission));
    params.surface().set_metallic(half(0.1));
    params.surface().set_roughness(half(0.8));
    params.surface().set_opacity(half(clamp(alpha, 0.0f, 1.0f)));
}
