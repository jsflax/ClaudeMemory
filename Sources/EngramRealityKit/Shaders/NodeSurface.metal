#include <metal_stdlib>
#include <RealityKit/RealityKit.h>
using namespace metal;

/// Node surface shader — decodes packed visual state from vertex color alpha,
/// applies glow emission, fog attenuation, and search dimming.
///
/// Packed state format (in vertex color alpha channel):
///   stateType + intensity * 0.01  (+ 10.0 if search-dimmed)
///   stateType: 0=normal, 0.25=selected, 0.5=recall, 0.75=arrival, 1.0=search match
///   Intensity occupies 0.00–0.01 within each 0.25-wide state band.

[[visible]]
void nodeSurface(realitykit::surface_parameters params)
{
    // Read per-vertex color (project tint in RGB, packed state in alpha)
    float4 vertColor = params.geometry().color();
    float3 baseColor = vertColor.rgb;
    float packedState = vertColor.a;

    // Decode packed state — compare directly against thresholds (no floor/fract)
    bool searchDimmed = packedState >= 10.0;
    if (searchDimmed) packedState -= 10.0;

    // Base PBR properties
    float3 color = baseColor;
    float3 emission = float3(0.0);
    float metallic = 0.3;
    float roughness = 0.6;
    float opacity = 1.0;

    // Apply visual state — thresholds match 0.25-spaced state types
    if (packedState >= 0.9) {
        // Search match — cyan tint + emission
        emission = float3(0.0, 0.8, 1.0) * 0.5;
        color = mix(baseColor, float3(0.0, 0.8, 1.0), 0.3);
    } else if (packedState >= 0.7) {
        // Arrival glow — warm white emission
        float intensity = clamp((packedState - 0.75) * 100.0, 0.0, 1.0);
        emission = float3(1.0, 0.9, 0.7) * intensity * 0.8;
        color = mix(baseColor, float3(1.0, 0.95, 0.85), intensity * 0.3);
    } else if (packedState >= 0.4) {
        // Recall glow — project color emission
        float intensity = clamp((packedState - 0.5) * 100.0, 0.0, 1.0);
        emission = baseColor * intensity * 1.2;
        color = mix(baseColor, float3(1.0), intensity * 0.2);
    } else if (packedState >= 0.2) {
        // Selected — pulse glow
        float intensity = clamp((packedState - 0.25) * 100.0, 0.0, 1.0);
        emission = baseColor * intensity * 0.6;
        roughness = mix(0.6, 0.3, intensity);
    }

    // Search dimming — reduce brightness significantly
    if (searchDimmed) {
        color *= 0.15;
        emission *= 0.1;
        opacity = 0.4;
    }

    // Distance fog using view-space depth (avoids needing camera world position)
    float3 viewPos = (params.uniforms().model_to_view() * float4(params.geometry().model_position(), 1.0)).xyz;
    float dist = length(viewPos);
    float fogStart = 200.0;
    float fogEnd = 2000.0;
    float fogFactor = clamp((dist - fogStart) / (fogEnd - fogStart), 0.0, 0.85);

    // Search matches and glowing nodes are fog-proof
    if (packedState >= 0.4) {
        fogFactor *= 0.3;
    }

    float3 fogColor = float3(0.051, 0.067, 0.09);  // scene background
    color = mix(color, fogColor, fogFactor);
    emission *= (1.0 - fogFactor);

    // Output
    params.surface().set_base_color(half3(color));
    params.surface().set_emissive_color(half3(emission));
    params.surface().set_metallic(half(metallic));
    params.surface().set_roughness(half(roughness));
    params.surface().set_opacity(half(opacity));
}
