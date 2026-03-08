#include <metal_stdlib>
#include "../Metal/SharedTypes.h"

using namespace metal;

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Common
// ──────────────────────────────────────────────────────────────────────────────

inline float fogFactor(float dist, float fogNear, float fogFar) {
    return saturate((dist - fogNear) / (fogFar - fogNear));
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Node Shaders (opaque, Blinn-Phong lit)
// ──────────────────────────────────────────────────────────────────────────────

struct NodeVertexOut {
    float4 position [[position]];
    float3 worldPos;
    float3 normal;
    float2 uv;
    float4 color;
};

vertex NodeVertexOut node_vertex(
    const device BatchVertex* vertices [[buffer(0)]],
    constant FrameUniforms&   frame    [[buffer(1)]],
    uint vid [[vertex_id]])
{
    BatchVertex v = vertices[vid];
    NodeVertexOut out;

    // Unpack packed state for scale pulse
    float packed   = v.color.w;
    float rawState = floor(packed);
    float intensity = clamp(fract(packed) * 100.0, 0.0, 1.0);
    float stateType = rawState >= 9.5 ? rawState - 10.0 : rawState;

    // Scale pulse (same as node_batch_geometry)
    float scalePulse = 1.0;
    if (stateType > 1.5 && stateType < 2.5) {
        scalePulse = 1.0 + sin(frame.time * 4.0) * 0.2 * intensity;   // recalled
    } else if (stateType > 2.5 && stateType < 3.5) {
        scalePulse = 1.0 + sin(frame.time * 3.0) * 0.2 * intensity;   // arriving
    } else if (stateType > 3.5 && stateType < 4.5) {
        scalePulse = 1.0 + sin(frame.time * 4.0) * 0.12;              // search matched
    } else if (stateType > 4.5 && stateType < 5.5) {
        scalePulse = 1.0 + sin(frame.time * 2.0) * 0.08 * intensity;  // mascot inspecting
    } else if (stateType > 5.5 && stateType < 6.5) {
        scalePulse = intensity;                                         // birth: scale 0→1
    } else if (stateType > 6.5 && stateType < 7.5) {
        scalePulse = 1.0 - intensity;                                   // death: scale 1→0
    }

    float3 worldPos = v.position;
    if (scalePulse != 1.0) {
        worldPos += v.normal * 0.008 * (scalePulse - 1.0);
    }

    out.position = frame.viewProjectionMatrix * float4(worldPos, 1.0);
    out.worldPos = worldPos;
    out.normal = v.normal;
    out.uv = v.uv;
    out.color = v.color;
    return out;
}

fragment float4 node_fragment(
    NodeVertexOut            in       [[stage_in]],
    constant FrameUniforms&  frame    [[buffer(0)]],
    constant LightingUniforms& lights [[buffer(1)]])
{
    float3 N = normalize(in.normal);
    float3 V = normalize(frame.cameraPosition - in.worldPos);

    // Unpack state from color.a
    float packed   = in.color.a;
    float rawState = floor(packed);
    float intensity = clamp(fract(packed) * 100.0, 0.0, 1.0);
    bool  dimmed   = rawState >= 9.5;
    float stateType = dimmed ? rawState - 10.0 : rawState;

    float3 baseColor = in.color.rgb;

    // Camera distance and fog
    float dist = length(in.worldPos - frame.cameraPosition);
    float fogT = fogFactor(dist, frame.fogNear, frame.fogFar);
    float depthFade = max(0.0, 1.0 - fogT * 0.95);

    // Fog opacity
    bool isSearchMatched = (stateType > 3.5 && stateType < 4.5);
    bool isInspecting = (stateType > 4.5 && stateType < 5.5);
    float fogOpacity;
    if (stateType > 0.5 && stateType < 1.5) {
        fogOpacity = 1.0;                                     // selected
    } else if (isSearchMatched) {
        fogOpacity = 1.0;                                     // search match
    } else if (isInspecting) {
        fogOpacity = max(0.5, 1.0 - fogT * 0.5);             // inspecting — reduced fog
    } else {
        fogOpacity = max(0.08, 1.0 - fogT * 0.92);
    }
    if (dimmed) {
        fogOpacity = min(fogOpacity, 0.12);
    }

    // Emissive
    float3 emissiveColor = float3(0);
    float  emissiveIntensity = 0;

    if (stateType > 0.5 && stateType < 1.5) {
        emissiveColor = float3(1.0);
        emissiveIntensity = 1.0;
    } else if (stateType > 1.5 && stateType < 2.5) {
        emissiveColor = float3(0.7, 0.9, 1.0);
        float pulse = 1.0 + sin(frame.time * 5.0) * 0.3;
        emissiveIntensity = 2.5 * intensity * pulse * depthFade;
    } else if (stateType > 2.5 && stateType < 3.5) {
        emissiveColor = float3(1.0, 0.85, 0.4);
        float pulse = 1.0 + sin(frame.time * 3.5) * 0.3;
        emissiveIntensity = 2.5 * intensity * pulse * depthFade;
    } else if (isSearchMatched) {
        emissiveColor = float3(0.0, 0.9, 1.0);
        float searchFade = max(0.3, depthFade);
        emissiveIntensity = 2.5 * (1.0 + sin(frame.time * 4.0) * 0.3) * searchFade;
    } else if (stateType > 4.5 && stateType < 5.5) {
        // Mascot inspecting — subtle cyan glow
        emissiveColor = float3(0.2, 0.6, 0.9);
        float pulse = 0.8 + 0.2 * sin(frame.time * 2.0);
        emissiveIntensity = 1.5 * intensity * pulse * depthFade;
    } else if (stateType > 5.5 && stateType < 6.5) {
        // Birth: warm glow that fades as node solidifies
        emissiveColor = float3(1.0, 0.9, 0.5);
        float pulse = 0.8 + 0.2 * sin(frame.time * 3.5);
        emissiveIntensity = 3.0 * (1.0 - intensity) * pulse * depthFade;
    } else if (stateType > 6.5 && stateType < 7.5) {
        // Death: red-orange dissolve
        emissiveColor = float3(1.0, 0.4, 0.1);
        float pulse = 0.8 + 0.2 * sin(frame.time * 3.5);
        emissiveIntensity = 3.0 * intensity * pulse * depthFade;
    } else {
        emissiveColor = baseColor;
        emissiveIntensity = 0.25;
    }

    // Blinn-Phong lighting
    float roughness = 0.15;
    float specPower = max(1.0, 2.0 / (roughness * roughness) - 2.0);
    float3 diffuseAccum = float3(0);
    float3 specAccum = float3(0);

    // Directional lights
    for (uint i = 0; i < lights.directionalLightCount; i++) {
        float3 L = normalize(lights.directionalLights[i].direction);
        float NdotL = max(0.0, dot(N, L));
        float3 H = normalize(L + V);
        float NdotH = max(0.0, dot(N, H));
        float spec = pow(NdotH, specPower);

        float3 lightColor = lights.directionalLights[i].color * lights.directionalLights[i].intensity;
        diffuseAccum += lightColor * NdotL;
        specAccum += lightColor * spec * 0.3;
    }

    // Point lights
    for (uint i = 0; i < lights.pointLightCount; i++) {
        float3 lightPos = lights.pointLights[i].position;
        float3 L = lightPos - in.worldPos;
        float lightDist = length(L);
        L /= max(lightDist, 0.001);
        float atten = max(0.0, 1.0 - lightDist / lights.pointLights[i].attenuationRadius);
        atten *= atten;

        float NdotL = max(0.0, dot(N, L));
        float3 H = normalize(L + V);
        float NdotH = max(0.0, dot(N, H));
        float spec = pow(NdotH, specPower);

        float3 lightColor = lights.pointLights[i].color * lights.pointLights[i].intensity * atten;
        diffuseAccum += lightColor * NdotL;
        specAccum += lightColor * spec * 0.3;
    }

    float3 ambient = baseColor * lights.ambientIntensity;
    // Maintenance atmosphere: subtle purple-blue ambient boost
    ambient += float3(0.3, 0.2, 0.5) * frame.maintenancePulse * 0.1;
    float3 lit = baseColor * diffuseAccum + specAccum + ambient;
    lit += emissiveColor * emissiveIntensity;

    // Fog bake into base color (opaque — no alpha)
    float3 foggedColor = mix(frame.bgColor, lit, fogOpacity);

    return float4(foggedColor, 1.0);
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Edge Shaders (transparent, unlit)
// ──────────────────────────────────────────────────────────────────────────────

struct EdgeVertexOut {
    float4 position [[position]];
    float3 worldPos;
    float  edgeState;
    float3 vertexColor;
};

vertex EdgeVertexOut edge_vertex(
    const device BatchVertex* vertices [[buffer(0)]],
    constant FrameUniforms&   frame    [[buffer(1)]],
    uint vid [[vertex_id]])
{
    BatchVertex v = vertices[vid];
    EdgeVertexOut out;
    out.position = frame.viewProjectionMatrix * float4(v.position, 1.0);
    out.worldPos = v.position;
    out.edgeState = v.uv.x;
    out.vertexColor = v.color.xyz;
    return out;
}

fragment float4 edge_fragment(
    EdgeVertexOut            in    [[stage_in]],
    constant FrameUniforms&  frame [[buffer(0)]])
{
    float edgeState = in.edgeState;
    float dist = length(in.worldPos - frame.cameraPosition);
    float fogT = fogFactor(dist, frame.fogNear, frame.fogFar);
    float depthFade = max(0.0, 1.0 - fogT * 0.95);

    // Spatial pulse — traveling wave
    float phase = (in.worldPos.x + in.worldPos.y + in.worldPos.z) * 8.0;
    float pulse = (sin(frame.time * 3.0 + phase) + 1.0) * 0.5;

    float opacity;
    if (edgeState > 1.5 && edgeState < 2.5) {
        opacity = 0.03 * depthFade;                              // searchDimmed
    } else if (edgeState > 0.5 && edgeState < 1.5) {
        opacity = (0.35 + pulse * 0.15) * max(0.12, depthFade);  // connected
    } else if (edgeState > 2.5) {
        opacity = (0.04 + pulse * 0.04) * depthFade;             // semanticMode
    } else {
        opacity = (0.08 + pulse * 0.06) * depthFade;             // normal
    }

    if (dist > frame.fogFar && edgeState < 0.5) {
        opacity = 0.0;
    }

    if (opacity < 0.005) {
        discard_fragment();
    }

    return float4(in.vertexColor, opacity);
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Label Shaders (transparent, textured billboard)
// ──────────────────────────────────────────────────────────────────────────────

struct LabelVertexOut {
    float4 position [[position]];
    float3 worldPos;
    float2 uv;
    float4 color;
};

vertex LabelVertexOut label_vertex(
    const device BatchVertex* vertices [[buffer(0)]],
    constant FrameUniforms&   frame    [[buffer(1)]],
    uint vid [[vertex_id]])
{
    BatchVertex v = vertices[vid];

    // Corner offset stored in normal field by stamp_label_quads kernel
    float3 corner = v.normal;  // (cx*halfW, cy*halfH, forwardBias)

    // Camera basis from view matrix (transpose of upper-left 3x3)
    float3 camRight = float3(frame.viewMatrix[0][0], frame.viewMatrix[1][0], frame.viewMatrix[2][0]);
    float3 camUp    = float3(frame.viewMatrix[0][1], frame.viewMatrix[1][1], frame.viewMatrix[2][1]);

    // Distance-based scaling (sqrt curve)
    float3 anchor = v.position;
    float dist = length(anchor - frame.cameraPosition);
    float distScale = sqrt(0.5 / max(dist, 0.05));

    float3 worldPos = anchor + (camRight * corner.x + camUp * corner.y) * distScale;

    // Forward bias for project labels
    if (corner.z > 0.0) {
        float3 camForward = -float3(frame.viewMatrix[0][2], frame.viewMatrix[1][2], frame.viewMatrix[2][2]);
        worldPos += camForward * corner.z;
    }

    LabelVertexOut out;
    out.position = frame.viewProjectionMatrix * float4(worldPos, 1.0);
    out.worldPos = worldPos;
    out.uv = v.uv;
    out.color = v.color;
    return out;
}

fragment float4 label_fragment(
    LabelVertexOut           in       [[stage_in]],
    constant FrameUniforms&  frame    [[buffer(0)]],
    texture2d<half>          atlas    [[texture(0)]],
    sampler                  s        [[sampler(0)]])
{
    half4 texColor = atlas.sample(s, in.uv);

    float vertexOpacity = in.color.a;

    // Labels already have depth-based fading from the compute kernel (maxVisible,
    // depthFade, fadeT).  Scene fog is NOT applied here because fogFar is tuned for
    // the primary galaxy; multi-galaxy scenes push far-galaxy labels well past fogFar,
    // resulting in alpha ≈ 0.03 — right at the discard edge, which causes hard binary
    // flickering as the camera moves.
    float alpha = float(texColor.a) * vertexOpacity;
    if (alpha < 0.01) {
        discard_fragment();
    }

    float3 tint = in.color.rgb;
    return float4(tint, alpha);  // straight alpha — blender applies sourceAlpha
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Nebula Shaders (transparent, fBm noise billboard)
// ──────────────────────────────────────────────────────────────────────────────

// 3D simplex noise helpers
inline float3 mod289(float3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
inline float4 mod289(float4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
inline float4 permute(float4 x) { return mod289(((x * 34.0) + 1.0) * x); }
inline float4 taylorInvSqrt(float4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float snoise3D(float3 v) {
    const float2 C = float2(1.0 / 6.0, 1.0 / 3.0);
    const float4 D = float4(0.0, 0.5, 1.0, 2.0);

    float3 i = floor(v + dot(v, float3(C.y)));
    float3 x0 = v - i + dot(i, float3(C.x));

    float3 g = step(x0.yzx, x0.xyz);
    float3 l = 1.0 - g;
    float3 i1 = min(g.xyz, l.zxy);
    float3 i2 = max(g.xyz, l.zxy);

    float3 x1 = x0 - i1 + C.x;
    float3 x2 = x0 - i2 + C.y;
    float3 x3 = x0 - D.yyy;

    i = mod289(i);
    float4 p = permute(permute(permute(
        i.z + float4(0.0, i1.z, i2.z, 1.0))
      + i.y + float4(0.0, i1.y, i2.y, 1.0))
      + i.x + float4(0.0, i1.x, i2.x, 1.0));

    float n_ = 0.142857142857;
    float3 ns = n_ * D.wyz - D.xzx;
    float4 j = p - 49.0 * floor(p * ns.z * ns.z);
    float4 x_ = floor(j * ns.z);
    float4 y_ = floor(j - 7.0 * x_);
    float4 x2_ = x_ * ns.x + float4(ns.y);
    float4 y2_ = y_ * ns.x + float4(ns.y);
    float4 h = 1.0 - abs(x2_) - abs(y2_);
    float4 b0 = float4(x2_.xy, y2_.xy);
    float4 b1 = float4(x2_.zw, y2_.zw);
    float4 s0 = floor(b0) * 2.0 + 1.0;
    float4 s1 = floor(b1) * 2.0 + 1.0;
    float4 sh = -step(h, float4(0.0));
    float4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    float4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
    float3 p0 = float3(a0.xy, h.x);
    float3 p1 = float3(a0.zw, h.y);
    float3 p2 = float3(a1.xy, h.z);
    float3 p3 = float3(a1.zw, h.w);
    float4 norm = taylorInvSqrt(float4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
    p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
    float4 m = max(0.6 - float4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot(m*m, float4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}

float fbm3D(float3 p, int octaves) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    for (int i = 0; i < octaves; i++) {
        value += amplitude * snoise3D(p * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    return value;
}

struct NebulaVertexOut {
    float4 position [[position]];
    float3 worldPos;
    float2 uv;
    float4 color;
    float  noisePhase;
    float  radius;
};

vertex NebulaVertexOut nebula_vertex(
    const device NebulaQuadVertex* vertices [[buffer(0)]],
    constant FrameUniforms&        frame    [[buffer(1)]],
    uint vid [[vertex_id]])
{
    NebulaQuadVertex v = vertices[vid];

    // Billboard expansion using camera basis
    float3 camRight = float3(frame.viewMatrix[0][0], frame.viewMatrix[1][0], frame.viewMatrix[2][0]);
    float3 camUp    = float3(frame.viewMatrix[0][1], frame.viewMatrix[1][1], frame.viewMatrix[2][1]);

    float3 worldPos = v.position + camRight * v.cornerOffset.x + camUp * v.cornerOffset.y;

    // Depth offset for parallax
    float3 camForward = -float3(frame.viewMatrix[0][2], frame.viewMatrix[1][2], frame.viewMatrix[2][2]);
    worldPos += camForward * v.cornerOffset.z;

    NebulaVertexOut out;
    out.position = frame.viewProjectionMatrix * float4(worldPos, 1.0);
    out.worldPos = worldPos;
    out.uv = v.uv;
    out.color = v.color;
    out.noisePhase = v.noisePhase;
    out.radius = v.radius;
    return out;
}

fragment float4 nebula_fragment(
    NebulaVertexOut          in    [[stage_in]],
    constant FrameUniforms&  frame [[buffer(0)]])
{
    // UV-based radial distance from center (0 at center, 1 at edge)
    float2 centered = in.uv * 2.0 - 1.0;
    float r = length(centered);

    // Quartic radial falloff: (1 - r²)²
    float radialFalloff = max(0.0, 1.0 - r * r);
    radialFalloff *= radialFalloff;

    // fBm noise — 3D coordinate scrolled by time
    float3 noiseCoord = float3(
        centered.x * 2.0 + in.noisePhase,
        centered.y * 2.0 + in.noisePhase * 0.7,
        frame.time * 0.03 + in.noisePhase * 0.3
    );
    float noise = fbm3D(noiseCoord, 4);
    noise = noise * 0.5 + 0.5;  // remap to 0..1

    float alpha = radialFalloff * noise * in.color.a;

    // Fog fade
    float dist = length(in.worldPos - frame.cameraPosition);
    float fogFade = max(0.0, 1.0 - fogFactor(dist, frame.fogNear, frame.fogFar) * 0.8);
    alpha *= fogFade;

    if (alpha < 0.002) {
        discard_fragment();
    }

    // Maintenance atmosphere tint
    float3 nebulaColor = in.color.rgb;
    if (frame.maintenancePulse > 0.001) {
        float3 maintenanceTint = float3(0.4, 0.3, 0.8);
        nebulaColor = mix(nebulaColor, maintenanceTint, frame.maintenancePulse * 0.3);
        alpha *= 1.0 + sin(frame.time * 1.5) * 0.15 * frame.maintenancePulse;
    }

    return float4(nebulaColor, alpha);  // straight alpha — blender applies sourceAlpha
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Compute: Stamp Node Spheres
//
// Each thread handles one vertex of one node's sphere instance.
// Reads SphereTemplateVertex + NodeInstance, writes BatchVertex.
// ──────────────────────────────────────────────────────────────────────────────

kernel void stamp_node_spheres(
    device const SphereTemplateVertex* templateVerts [[buffer(0)]],
    device const NodeInstance*         instances      [[buffer(1)]],
    constant     StampParams&          params         [[buffer(2)]],
    device       BatchVertex*          output         [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint totalVerts = params.nodeCount * params.vertsPerNode;
    if (tid >= totalVerts) return;

    uint nodeIdx = tid / params.vertsPerNode;
    uint vertIdx = tid % params.vertsPerNode;

    NodeInstance inst = instances[nodeIdx];
    SphereTemplateVertex sv = templateVerts[vertIdx];

    float3 worldPos = inst.position + sv.position * inst.scale;
    float3 normal   = sv.normal;

    BatchVertex bv;
    bv.position = worldPos;
    bv.normal = normal;
    bv.uv = float2(0);
    bv.color = inst.color;
    output[tid] = bv;
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Compute: Stamp Label Quads
//
// Each thread handles one label, writing 4 BatchVertex (billboard quad corners).
// ──────────────────────────────────────────────────────────────────────────────

kernel void stamp_label_quads(
    device const LabelInstance*    instances [[buffer(0)]],
    constant     LabelStampParams& params    [[buffer(1)]],
    device       BatchVertex*      output    [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.labelCount) return;

    LabelInstance inst = instances[tid];
    float3 anchor = inst.anchor;
    float3 camPos = params.cameraPos;
    float  depth  = length(anchor - camPos);

    bool isSelected    = (inst.flags & 1u) != 0;
    bool isSearchMatch = (inst.flags & 2u) != 0;
    bool searchDimmed  = (inst.flags & 4u) != 0;
    bool alwaysVisible = isSelected || isSearchMatch;

    uint base = tid * 4;

    if (!alwaysVisible && depth > inst.maxVisible) {
        BatchVertex zero;
        zero.position = float3(0); zero.normal = float3(0);
        zero.uv = float2(0); zero.color = float4(0);
        output[base] = zero; output[base+1] = zero;
        output[base+2] = zero; output[base+3] = zero;
        return;
    }

    float depthNorm = saturate((depth - params.minDepth) / max(params.depthRange, 0.001f));
    float depthFade = isSelected ? 1.0 : (1.0 - depthNorm * 0.7);
    float fadeRange = inst.maxVisible * 0.3;
    float fadeT     = isSelected ? 1.0 : saturate((inst.maxVisible - depth) / fadeRange);
    float searchFade = searchDimmed ? 0.15 : 1.0;
    float opacity = inst.color.w * fadeT * depthFade * searchFade;

    if (opacity < 0.02) {
        BatchVertex zero;
        zero.position = float3(0); zero.normal = float3(0);
        zero.uv = float2(0); zero.color = float4(0);
        output[base] = zero; output[base+1] = zero;
        output[base+2] = zero; output[base+3] = zero;
        return;
    }

    float halfH = inst.halfH;
    float halfW = halfH * inst.textAspect;
    float nz    = inst.forwardBias;

    float u0 = inst.uvRect.x, v0 = inst.uvRect.y;
    float u1 = inst.uvRect.z, v1 = inst.uvRect.w;
    float cr = inst.color.x, cg = inst.color.y, cb = inst.color.z;

    BatchVertex bv;
    bv.position = anchor;
    bv.color = float4(cr, cg, cb, opacity);

    // TL
    bv.normal = float3(-halfW, halfH, nz);
    bv.uv = float2(u0, v0);
    output[base] = bv;

    // TR
    bv.normal = float3(halfW, halfH, nz);
    bv.uv = float2(u1, v0);
    output[base + 1] = bv;

    // BL
    bv.normal = float3(-halfW, 0, nz);
    bv.uv = float2(u0, v1);
    output[base + 2] = bv;

    // BR
    bv.normal = float3(halfW, 0, nz);
    bv.uv = float2(u1, v1);
    output[base + 3] = bv;
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Compute: Pack Edge Instances (GPU-side position lookup + inset)
//
// Reads EdgeDescriptor[] + node positions[] → writes EdgeInstance[].
// Replaces per-frame O(edges) UUID dictionary lookups on CPU.
// ──────────────────────────────────────────────────────────────────────────────

kernel void pack_edge_instances(
    device const EdgeDescriptor*  descriptors [[buffer(0)]],
    device const float3*          positions   [[buffer(1)]],  // SIMD3<Float> from Swift (16-byte stride)
    constant     PackEdgeParams&  params      [[buffer(2)]],
    device       EdgeInstance*    output      [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.edgeCount) return;

    EdgeDescriptor desc = descriptors[tid];
    float3 from = positions[desc.sourceIdx] * params.scaleFactor;
    float3 to   = positions[desc.targetIdx] * params.scaleFactor;
    float3 delta = to - from;
    float  len = length(delta);

    // Degenerate or overlapping — write zero instance (stamp kernel collapses it)
    if (len <= desc.sourceRadius + desc.targetRadius) {
        EdgeInstance inst;
        inst.sourcePos = float3(0);
        inst.radius = 0;
        inst.targetPos = float3(0);
        inst.state = 0;
        inst.color = float4(0);
        output[tid] = inst;
        return;
    }

    float3 dir = delta / len;

    EdgeInstance inst;
    inst.sourcePos = from + dir * desc.sourceRadius;
    inst.radius    = desc.baseRadius;
    inst.targetPos = to   - dir * desc.targetRadius;
    inst.state     = desc.state;
    inst.color     = desc.color;
    output[tid] = inst;
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Compute: Stamp Edge Cylinders
//
// Each thread handles one vertex of one edge's cylinder (12 verts per edge).
// Reads EdgeInstance, writes BatchVertex with hexagonal cylinder geometry.
// ──────────────────────────────────────────────────────────────────────────────

constant float kCylCos[6] = { 1.0, 0.5, -0.5, -1.0, -0.5, 0.5 };
constant float kCylSin[6] = { 0.0, 0.866025403784, 0.866025403784, 0.0, -0.866025403784, -0.866025403784 };

kernel void stamp_edge_cylinders(
    device const EdgeInstance*  instances [[buffer(0)]],
    constant     EdgeStampParams& params  [[buffer(1)]],
    device       BatchVertex*   output    [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    uint totalVerts = params.edgeCount * params.vertsPerEdge;
    if (tid >= totalVerts) return;

    uint edgeIdx = tid / params.vertsPerEdge;
    uint vertIdx = tid % params.vertsPerEdge;

    EdgeInstance inst = instances[edgeIdx];
    float3 p1 = inst.sourcePos;  // already inset by CPU
    float3 p2 = inst.targetPos;  // already inset by CPU
    float3 delta = p2 - p1;
    float  len = length(delta);

    // Degenerate edge — collapse to zero
    if (len < 0.0001) {
        BatchVertex bv;
        bv.position = float3(0); bv.normal = float3(0, 1, 0);
        bv.uv = float2(0); bv.color = float4(0);
        output[tid] = bv;
        return;
    }

    float3 dir = delta / len;

    // Basis vectors for the cylinder cross-section
    float3 basisUp = abs(dir.y) < 0.99 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 basisRight   = normalize(cross(dir, basisUp));
    float3 basisForward = normalize(cross(basisRight, dir));

    // Which segment (0-5) and which ring (bottom=0, top=1)
    uint seg  = vertIdx % 6;
    uint ring = vertIdx / 6;  // 0 = bottom ring, 1 = top ring

    float c = kCylCos[seg];
    float s = kCylSin[seg];
    float3 offset = (basisRight * c + basisForward * s) * inst.radius;
    float3 normal = normalize(basisRight * c + basisForward * s);
    float3 pos = (ring == 0) ? (p1 + offset) : (p2 + offset);

    BatchVertex bv;
    bv.position = pos;
    bv.normal = normal;
    bv.uv = float2(inst.state, float(ring));
    bv.color = inst.color;
    output[tid] = bv;
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Flow Particle Shaders (transparent, soft circle billboard)
// ──────────────────────────────────────────────────────────────────────────────

struct FlowVertexOut {
    float4 position [[position]];
    float2 uv;
    float4 color;
};

vertex FlowVertexOut flow_vertex(
    const device FlowParticleVertex* vertices [[buffer(0)]],
    constant FrameUniforms&          frame    [[buffer(1)]],
    uint vid [[vertex_id]])
{
    // Each particle is 4 vertices (quad). Determine which corner.
    uint particleIdx = vid / 4;
    uint cornerIdx = vid % 4;

    FlowParticleVertex v = vertices[particleIdx];

    float3 camRight = float3(frame.viewMatrix[0][0], frame.viewMatrix[1][0], frame.viewMatrix[2][0]);
    float3 camUp    = float3(frame.viewMatrix[0][1], frame.viewMatrix[1][1], frame.viewMatrix[2][1]);

    // Corner offsets: TL, TR, BL, BR
    float2 corners[4] = { float2(-1, 1), float2(1, 1), float2(-1, -1), float2(1, -1) };
    float2 c = corners[cornerIdx];
    float2 uvs[4] = { float2(0, 0), float2(1, 0), float2(0, 1), float2(1, 1) };

    float3 worldPos = v.position + (camRight * c.x + camUp * c.y) * v.size;

    FlowVertexOut out;
    out.position = frame.viewProjectionMatrix * float4(worldPos, 1.0);
    out.uv = uvs[cornerIdx];
    out.color = v.color;
    return out;
}

fragment float4 flow_fragment(
    FlowVertexOut in [[stage_in]])
{
    // Soft circle
    float2 centered = in.uv * 2.0 - 1.0;
    float r = length(centered);
    float alpha = max(0.0, 1.0 - r * r);
    alpha *= alpha;  // quartic falloff
    alpha *= in.color.a;

    if (alpha < 0.01) {
        discard_fragment();
    }

    return float4(in.color.rgb, alpha);  // straight alpha
}
