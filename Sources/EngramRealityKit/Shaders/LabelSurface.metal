#include <metal_stdlib>
#include <RealityKit/RealityKit.h>
using namespace metal;

/// Label surface shader — samples atlas texture for per-pixel alpha,
/// uses vertex color for tint and opacity.
///
/// Atlas texture bound via custom.texture to avoid RealityKit's internal
/// UV transforms on baseColor. Vertex color RGB = project tint, A = opacity.

[[visible]]
void labelSurface(realitykit::surface_parameters params)
{
    // Sample atlas from custom texture slot (avoids baseColor UV transforms)
    constexpr sampler texSampler(filter::linear, address::clamp_to_edge);
    float2 uv = params.geometry().uv0();
    half4 texColor = params.textures().custom().sample(texSampler, uv);

    // Text alpha from atlas — 0 for background, >0 for text pixels
    half textAlpha = texColor.a;

    // Vertex color carries tint (RGB) and opacity (A)
    float4 vertColor = params.geometry().color();

    // Use emissive so labels aren't affected by scene lighting
    params.surface().set_base_color(half3(0.0));
    params.surface().set_emissive_color(half3(vertColor.rgb) * textAlpha);
    params.surface().set_metallic(half(0.0));
    params.surface().set_roughness(half(1.0));

    // Combine texture alpha with vertex color alpha
    params.surface().set_opacity(textAlpha * half(vertColor.a));
}
