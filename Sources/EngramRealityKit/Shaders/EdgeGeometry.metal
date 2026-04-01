#include <metal_stdlib>
#include <RealityKit/RealityKit.h>
using namespace metal;

/// Edge geometry modifier — macOS 26+ MeshInstancesComponent path only.
///
/// MeshInstancesComponent handles cylinder transform (translate + rotate + scale).
/// This modifier reads per-instance visual data (color + alpha) from a custom
/// texture using instance_id(), and passes it to the surface shader via set_color().

[[visible]]
void edgeGeometry(realitykit::geometry_parameters params) RK_AVAILABILITY_MACOS_16
{
    uint idx = params.geometry().instance_id();
    float texWidth = params.uniforms().custom_parameter().y;
    constexpr sampler s(filter::nearest, address::clamp_to_edge);
    float2 uv = float2((float(idx) + 0.5) / texWidth, 0.5);
    half4 data = params.textures().custom().sample(s, uv);
    params.geometry().set_color(float4(data));
}
