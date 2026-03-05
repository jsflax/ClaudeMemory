#include <metal_stdlib>
#include "../Metal/SharedTypes.h"

using namespace metal;

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Mascot Shaders (opaque, PBR-textured, Blinn-Phong lit)
// ──────────────────────────────────────────────────────────────────────────────

inline float mascotFogFactor(float dist, float fogNear, float fogFar) {
    return saturate((dist - fogNear) / (fogFar - fogNear));
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Arcane Circle (procedural spinning glyphs, transparent billboard)
// ──────────────────────────────────────────────────────────────────────────────

inline float arcaneHash(float n) {
    return fract(sin(n) * 43758.5453123);
}

struct ArcaneVertexOut {
    float4 position [[position]];
    float2 uv;
    float4 color;
};

vertex ArcaneVertexOut arcane_vertex(
    constant ArcaneCircleUniforms& circle [[buffer(0)]],
    constant FrameUniforms&        frame  [[buffer(1)]],
    uint vid [[vertex_id]])
{
    // Expand 1 center point into 4 oriented corners on the GPU.
    // Uses the mascot's right/up vectors instead of camera vectors
    // so the circle faces the mascot's forward direction.
    uint cornerIdx = vid % 4;

    float2 corners[4] = { float2(-1, 1), float2(1, 1), float2(-1, -1), float2(1, -1) };
    float2 uvs[4]     = { float2(0, 0), float2(1, 0), float2(0, 1), float2(1, 1) };
    float2 c = corners[cornerIdx];

    float3 worldPos = circle.center
                    + circle.right * c.x * circle.size
                    + circle.up    * c.y * circle.size;

    ArcaneVertexOut out;
    out.position = frame.viewProjectionMatrix * float4(worldPos, 1.0);
    out.uv    = uvs[cornerIdx];
    // Use per-project tint color (defaults to cyan if zero)
    float3 tint = circle.tintColor;
    if (length(tint) < 0.01) tint = float3(0.0, 0.9, 1.0);
    out.color = float4(tint, circle.opacity);
    return out;
}

fragment float4 arcane_circle_fragment(
    ArcaneVertexOut          in    [[stage_in]],
    constant FrameUniforms&  frame [[buffer(0)]])
{
    float2 uv = in.uv * 2.0 - 1.0;
    float r     = length(uv);
    float theta = atan2(uv.y, uv.x);
    float time  = frame.time;

    float totalAlpha = 0.0;
    constexpr float PI  = 3.14159265;
    constexpr float TAU = 6.28318530;

    // ── Outer ring (r ≈ 0.88): thin line + radial tick marks ──
    {
        float ringR = 0.88;
        float ringLine = smoothstep(0.012, 0.003, abs(r - ringR));

        float rot = theta + time * 0.3;
        float segAngle = TAU / 24.0;
        float segIdx  = floor((rot + PI) / segAngle);
        float segFrac = fract((rot + PI) / segAngle);

        float h = arcaneHash(segIdx * 13.37);
        float tick = 0.0;
        if (h > 0.42) {
            // Radial tick pointing outward
            tick = smoothstep(0.44, 0.47, segFrac) * smoothstep(0.56, 0.53, segFrac);
            tick *= smoothstep(ringR + 0.07, ringR + 0.01, r) * step(ringR - 0.005, r);
        }
        totalAlpha = max(totalAlpha, max(ringLine * 0.7, tick * 0.8));
    }

    // ── Middle ring (r ≈ 0.58): glyph bars + accent dots ──
    {
        float ringR = 0.58;
        float ringLine = smoothstep(0.016, 0.004, abs(r - ringR));

        float rot = theta - time * 0.5;      // opposite spin
        float segAngle = TAU / 16.0;
        float segIdx  = floor((rot + PI) / segAngle);
        float segFrac = fract((rot + PI) / segAngle);

        float h = arcaneHash(segIdx * 7.77 + 42.0);

        // Radial bars — varying width + height → rune-like strokes
        float barHW = 0.06 + h * 0.04;
        float bar = smoothstep(0.5 - barHW - 0.02, 0.5 - barHW, segFrac)
                  * smoothstep(0.5 + barHW + 0.02, 0.5 + barHW, segFrac);
        float barH = 0.04 + h * 0.06;
        float radial = smoothstep(ringR + barH, ringR + barH * 0.2, r)
                     * smoothstep(ringR - barH, ringR - barH * 0.2, r);
        float glyph = bar * radial;

        // Accent dot beside each bar
        if (h > 0.3) {
            float dotT = abs(segFrac - 0.22);
            float dotR = abs(r - ringR);
            float dot = smoothstep(0.04, 0.02, dotT) * smoothstep(0.022, 0.008, dotR);
            glyph = max(glyph, dot * 0.8);
        }

        // Second smaller dot on the other side
        if (h > 0.6) {
            float dotT = abs(segFrac - 0.78);
            float dotR = abs(r - ringR);
            float dot = smoothstep(0.03, 0.015, dotT) * smoothstep(0.018, 0.006, dotR);
            glyph = max(glyph, dot * 0.6);
        }

        totalAlpha = max(totalAlpha, max(ringLine * 0.8, glyph * 0.9));
    }

    // ── Inner ring (r ≈ 0.32): longer radial marks ──
    {
        float ringR = 0.32;
        float ringLine = smoothstep(0.010, 0.003, abs(r - ringR));

        float rot = theta + time * 0.8;
        float segAngle = TAU / 8.0;
        float segIdx  = floor((rot + PI) / segAngle);
        float segFrac = fract((rot + PI) / segAngle);

        float h = arcaneHash(segIdx * 3.14 + 17.0);
        float mark = smoothstep(0.32, 0.38, segFrac) * smoothstep(0.68, 0.62, segFrac);
        float markH = 0.06 + h * 0.10;
        float radial = smoothstep(ringR + markH, ringR + markH * 0.25, r)
                     * smoothstep(ringR - markH, ringR - markH * 0.25, r);
        totalAlpha = max(totalAlpha, max(ringLine * 0.85, mark * radial * 0.75));
    }

    // ── Faint spoke lines connecting rings ──
    {
        float rot = theta + time * 0.15;
        float spokeAngle = TAU / 6.0;
        float spokeFrac = fract((rot + PI) / spokeAngle);
        float spoke = smoothstep(0.012, 0.002, abs(spokeFrac - 0.5) * max(r, 0.1));
        float spokeR = smoothstep(0.28, 0.34, r) * smoothstep(0.92, 0.86, r);
        totalAlpha += spoke * spokeR * 0.10;
    }

    // ── Center glow ──
    totalAlpha += exp(-r * r * 14.0) * 0.20;

    // Pulse synced with eye emissive
    float pulse = 0.7 + 0.3 * sin(time * 3.0);
    totalAlpha *= pulse;

    // Soft circular edge fade
    totalAlpha *= smoothstep(1.0, 0.92, r);

    // Vertex-level opacity (fade in/out with hover state)
    totalAlpha *= in.color.a;

    if (totalAlpha < 0.004) discard_fragment();

    // Color: tint at center → deeper blue toward edges
    float3 tint = in.color.rgb;
    float3 color = mix(tint, float3(0.25, 0.45, 1.0), r * 0.7);

    return float4(color, totalAlpha);   // straight alpha
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Holographic Info Screen (camera-facing billboard, text + scan lines)
// ──────────────────────────────────────────────────────────────────────────────

struct HoloVertexOut {
    float4 position [[position]];
    float2 uv;
    float  opacity;
    float  revealProgress;
};

vertex HoloVertexOut holo_vertex(
    constant HoloScreenUniforms& holo  [[buffer(0)]],
    constant FrameUniforms&      frame [[buffer(1)]],
    uint vid [[vertex_id]])
{
    // Expand 4 vertices into a billboard oriented to the mascot's facing direction.
    // Uses the mascot's right/up vectors (same approach as the arcane circle).
    uint cornerIdx = vid % 4;
    float2 corners[4] = { float2(-1, 1), float2(1, 1), float2(-1, -1), float2(1, -1) };
    float2 uvs[4]     = { float2(0, 0), float2(1, 0), float2(0, 1), float2(1, 1) };
    float2 c = corners[cornerIdx];

    float3 worldPos = holo.center
                    + holo.right * c.x * holo.width
                    + holo.up    * c.y * holo.height;

    HoloVertexOut out;
    out.position       = frame.viewProjectionMatrix * float4(worldPos, 1.0);
    out.uv             = uvs[cornerIdx];
    out.opacity        = holo.opacity;
    out.revealProgress = holo.revealProgress;
    return out;
}

fragment float4 holo_fragment(
    HoloVertexOut            in    [[stage_in]],
    constant FrameUniforms&  frame [[buffer(0)]],
    texture2d<half>          tex   [[texture(0)]],
    sampler                  s     [[sampler(0)]])
{
    float2 uv = in.uv;
    float time = frame.time;

    // Sample the text texture (premultiplied BGRA from CoreText)
    half4 textSample = tex.sample(s, uv);
    float textAlpha = float(textSample.a);
    // Un-premultiply to get straight RGB for compositing
    float3 textColor = textAlpha > 0.001 ? float3(textSample.rgb) / textAlpha : float3(0);

    // ── Typewriter reveal: sweep top-to-bottom, left-to-right per line ──
    // Map UV to a linear reveal coordinate: each row reveals left→right,
    // rows reveal top→bottom. revealProgress 0→1 covers the full texture.
    float revealCoord = uv.y + uv.x * 0.15;  // slight diagonal for natural feel
    float revealEdge = in.revealProgress * 1.15;  // overshoot so full text shows at progress=1
    float revealMask = smoothstep(revealEdge - 0.04, revealEdge, revealCoord);
    textAlpha *= (1.0 - revealMask);  // revealed = visible, unrevealed = hidden
    // Typing cursor glow at the reveal edge
    float cursorGlow = smoothstep(0.03, 0.0, abs(revealCoord - revealEdge)) * step(0.01, in.revealProgress) * step(in.revealProgress, 0.99);
    float3 cursorColor = float3(0.4, 0.9, 1.0) * cursorGlow * 0.6;

    // ── Semi-transparent background panel ──
    // Rounded rect mask — soften edges
    float2 d = abs(uv - 0.5) * 2.0;   // 0 at center, 1 at edge
    float edgeSoft = smoothstep(1.0, 0.92, max(d.x, d.y));
    // Glowing border frame
    float borderDist = max(d.x, d.y);
    float border = smoothstep(0.88, 0.92, borderDist) * smoothstep(1.0, 0.96, borderDist);
    float3 borderColor = float3(0.2, 0.7, 1.0);

    float panelAlpha = 0.30 * edgeSoft;

    // ── Holographic scan lines (horizontal) ──
    float scanLine = sin(uv.y * 120.0 + time * 4.0) * 0.08;
    panelAlpha += scanLine * edgeSoft;

    // ── Subtle flicker ──
    float flicker = 0.95 + 0.05 * sin(time * 17.0);

    // ── Compose: panel background + text + cursor + border ──
    float3 panelColor = float3(0.05, 0.15, 0.25);  // dark blue-teal
    // Text over panel: text covers panel where it exists
    float3 color = mix(panelColor, textColor, textAlpha);
    float alpha = max(panelAlpha, textAlpha);
    // Add typing cursor
    color += cursorColor;
    alpha = max(alpha, cursorGlow * 0.5);
    // Add border glow on top
    color += borderColor * border * 0.8;
    alpha += border * 0.6;

    alpha *= flicker * in.opacity * edgeSoft;

    if (alpha < 0.004) discard_fragment();

    // Final blue-cyan tint boost
    color = mix(color, color * float3(0.7, 0.9, 1.2), 0.15);

    return float4(color * alpha, alpha);   // output premultiplied for blending
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Node Inspection Rings (3D concentric rings orbiting inspected node)
// ──────────────────────────────────────────────────────────────────────────────

struct NodeRingVertexOut {
    float4 position [[position]];
    float2 uv;
    float  opacity;
    float3 tint;
};

vertex NodeRingVertexOut node_ring_vertex(
    constant ArcaneCircleUniforms& ring  [[buffer(0)]],
    constant FrameUniforms&        frame [[buffer(1)]],
    uint vid [[vertex_id]])
{
    uint cornerIdx = vid % 4;
    float2 corners[4] = { float2(-1, 1), float2(1, 1), float2(-1, -1), float2(1, -1) };
    float2 uvs[4]     = { float2(0, 0), float2(1, 0), float2(0, 1), float2(1, 1) };
    float2 c = corners[cornerIdx];

    float3 worldPos = ring.center
                    + ring.right * c.x * ring.size
                    + ring.up    * c.y * ring.size;

    NodeRingVertexOut out;
    out.position = frame.viewProjectionMatrix * float4(worldPos, 1.0);
    out.uv       = uvs[cornerIdx];
    out.opacity  = ring.opacity;
    float3 tint  = ring.tintColor;
    out.tint     = length(tint) < 0.01 ? float3(0.15, 0.7, 1.0) : tint;
    return out;
}

fragment float4 node_ring_fragment(
    NodeRingVertexOut        in    [[stage_in]],
    constant FrameUniforms&  frame [[buffer(0)]])
{
    float2 uv = in.uv * 2.0 - 1.0;
    float r = length(uv);
    float time = frame.time;

    float totalAlpha = 0.0;

    // ── Ring 1: inner ring pulsing outward ──
    {
        float ringR = 0.40 + 0.05 * sin(time * 1.5);
        float ring = smoothstep(0.020, 0.006, abs(r - ringR));
        totalAlpha += ring * 0.85;
    }

    // ── Ring 2: outer ring pulsing outward (opposite phase) ──
    {
        float ringR = 0.70 + 0.04 * sin(time * 1.5 + 2.094);
        float ring = smoothstep(0.016, 0.004, abs(r - ringR));
        totalAlpha += ring * 0.7;
    }

    // ── Faint glow band between rings ──
    {
        float bandCenter = 0.55;
        float band = exp(-(r - bandCenter) * (r - bandCenter) * 20.0);
        totalAlpha += band * 0.08;
    }

    // Soft circular edge fade
    totalAlpha *= smoothstep(1.0, 0.85, r);

    // Pulse synced with the mascot's hover
    float pulse = 0.8 + 0.2 * sin(time * 2.0);
    totalAlpha *= pulse;

    // Apply vertex-level opacity (fade in/out with hover state)
    totalAlpha *= in.opacity;

    if (totalAlpha < 0.004) discard_fragment();

    // Tinted color matching the inspecting node emissive
    float3 color = mix(in.tint, float3(0.3, 0.5, 1.0), r * 0.6);

    return float4(color, totalAlpha);   // straight alpha — additive blend in pipeline
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Mascot Mesh Shaders
// ──────────────────────────────────────────────────────────────────────────────

struct MascotVertexOut {
    float4 position [[position]];
    float3 worldPos;
    float3 worldNormal;
    float2 uv;
    uint   partIndex;
    uint   instanceId [[flat]];
};

vertex MascotVertexOut mascot_vertex(
    const device float*              rawVerts  [[buffer(0)]],
    constant     FrameUniforms&      frame     [[buffer(1)]],
    constant     MascotPartUniforms& part      [[buffer(2)]],
    constant     uint&               partIndex [[buffer(3)]],
    uint vid [[vertex_id]])
{
    // Read 36-byte packed vertex: 9 floats per vertex
    // position(3) + normal(3) + uv(2) + skinWeight(1)
    uint base = vid * 9;
    float3 position = float3(rawVerts[base], rawVerts[base+1], rawVerts[base+2]);
    float3 normal   = float3(rawVerts[base+3], rawVerts[base+4], rawVerts[base+5]);
    float2 uv       = float2(rawVerts[base+6], rawVerts[base+7]);
    float skinWeight = rawVerts[base+8];

    // Per-vertex skeletal skinning: blend between parent (body) and part transform.
    // skinWeight is precomputed in the mesh: 0.0 at shoulder → 1.0 at hand.
    // Vertices at the joint seam have weight ≈ 0, matching the body exactly → no gap.
    float4 worldPos4;
    float3 worldNormal;
    if (skinWeight > 0.001 && part.blendRadius > 0.001) {
        float4 posBody = part.parentMatrix * float4(position, 1.0);
        float4 posArm  = part.modelMatrix * float4(position, 1.0);
        worldPos4 = mix(posBody, posArm, skinWeight);
        float3 nrmBody = (part.parentMatrix * float4(normal, 0.0)).xyz;
        float3 nrmArm  = (part.modelMatrix * float4(normal, 0.0)).xyz;
        worldNormal = normalize(mix(nrmBody, nrmArm, skinWeight));
    } else {
        float4x4 mat = (part.blendRadius > 0.001) ? part.parentMatrix : part.modelMatrix;
        worldPos4 = mat * float4(position, 1.0);
        worldNormal = normalize((mat * float4(normal, 0.0)).xyz);
    }

    MascotVertexOut out;
    out.position    = frame.viewProjectionMatrix * worldPos4;
    out.worldPos    = worldPos4.xyz;
    out.worldNormal = worldNormal;
    out.uv          = uv;
    out.partIndex   = partIndex;
    out.instanceId  = 0;
    return out;
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Instanced Mascot Vertex Shader
// ──────────────────────────────────────────────────────────────────────────────

vertex MascotVertexOut mascot_vertex_instanced(
    const device float*                rawVerts   [[buffer(0)]],
    constant     FrameUniforms&        frame      [[buffer(1)]],
    const device MascotInstanceData*   instances  [[buffer(2)]],
    constant     uint&                 partIndex  [[buffer(3)]],
    uint vid [[vertex_id]],
    uint iid [[instance_id]])
{
    // Read 36-byte packed vertex: 9 floats per vertex
    uint base = vid * 9;
    float3 position = float3(rawVerts[base], rawVerts[base+1], rawVerts[base+2]);
    float3 normal   = float3(rawVerts[base+3], rawVerts[base+4], rawVerts[base+5]);
    float2 uv       = float2(rawVerts[base+6], rawVerts[base+7]);
    float skinWeight = rawVerts[base+8];

    MascotPartUniforms part = instances[iid].parts[partIndex];

    float4 worldPos4;
    float3 worldNormal;
    if (skinWeight > 0.001 && part.blendRadius > 0.001) {
        float4 posBody = part.parentMatrix * float4(position, 1.0);
        float4 posArm  = part.modelMatrix * float4(position, 1.0);
        worldPos4 = mix(posBody, posArm, skinWeight);
        float3 nrmBody = (part.parentMatrix * float4(normal, 0.0)).xyz;
        float3 nrmArm  = (part.modelMatrix * float4(normal, 0.0)).xyz;
        worldNormal = normalize(mix(nrmBody, nrmArm, skinWeight));
    } else {
        float4x4 mat = (part.blendRadius > 0.001) ? part.parentMatrix : part.modelMatrix;
        worldPos4 = mat * float4(position, 1.0);
        worldNormal = normalize((mat * float4(normal, 0.0)).xyz);
    }

    MascotVertexOut out;
    out.position    = frame.viewProjectionMatrix * worldPos4;
    out.worldPos    = worldPos4.xyz;
    out.worldNormal = worldNormal;
    out.uv          = uv;
    out.partIndex   = partIndex;
    out.instanceId  = iid;
    return out;
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Instanced Mascot Fragment Shader
// ──────────────────────────────────────────────────────────────────────────────

fragment float4 mascot_fragment_instanced(
    MascotVertexOut                      in         [[stage_in]],
    constant FrameUniforms&              frame      [[buffer(0)]],
    constant LightingUniforms&           lights     [[buffer(1)]],
    const device MascotInstanceData*     instances  [[buffer(2)]],
    texture2d<half>                      baseColorTex   [[texture(0)]],
    texture2d<half>                      metalRoughTex  [[texture(1)]],
    sampler                              s              [[sampler(0)]])
{
    MascotPartUniforms part = instances[in.instanceId].parts[in.partIndex];

    half4 baseColorSample = baseColorTex.sample(s, in.uv);
    float3 baseColor = float3(baseColorSample.rgb);

    half4 mrSample = metalRoughTex.sample(s, in.uv);
    float metallic  = float(mrSample.b);
    float roughness = max(float(mrSample.g), 0.08);

    float3 N = normalize(in.worldNormal);
    float3 V = normalize(frame.cameraPosition - in.worldPos);

    float specPower = max(1.0, 2.0 / (roughness * roughness) - 2.0);
    float3 diffuseAccum = float3(0);
    float3 specAccum    = float3(0);

    for (uint i = 0; i < lights.directionalLightCount; i++) {
        float3 L = normalize(lights.directionalLights[i].direction);
        float NdotL = max(0.0, dot(N, L));
        float3 H = normalize(L + V);
        float NdotH = max(0.0, dot(N, H));
        float spec = pow(NdotH, specPower);
        float3 lightColor = lights.directionalLights[i].color * lights.directionalLights[i].intensity;
        diffuseAccum += lightColor * NdotL;
        specAccum    += lightColor * spec * (0.3 + metallic * 0.5);
    }

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
        specAccum    += lightColor * spec * (0.3 + metallic * 0.5);
    }

    float3 diffColor = baseColor * (1.0 - metallic * 0.6);
    float3 ambient = diffColor * max(lights.ambientIntensity, 0.35);
    float3 lit = diffColor * diffuseAccum + specAccum + ambient;

    float fresnel = pow(1.0 - max(0.0, dot(N, V)), 3.0);
    lit += float3(0.3, 0.5, 0.8) * fresnel * 0.6;

    float3 emissive = part.emissive.rgb * part.emissive.a;
    lit += emissive;

    float dist = length(in.worldPos - frame.cameraPosition);
    float fogT = mascotFogFactor(dist, frame.fogNear, frame.fogFar);
    float3 foggedColor = mix(lit, frame.bgColor, fogT);

    return float4(foggedColor, 1.0);
}

fragment float4 mascot_fragment(
    MascotVertexOut              in       [[stage_in]],
    constant FrameUniforms&      frame    [[buffer(0)]],
    constant LightingUniforms&   lights   [[buffer(1)]],
    constant MascotPartUniforms& part     [[buffer(2)]],
    texture2d<half>              baseColorTex   [[texture(0)]],
    texture2d<half>              metalRoughTex  [[texture(1)]],
    sampler                      s              [[sampler(0)]])
{
    // Sample textures
    half4 baseColorSample = baseColorTex.sample(s, in.uv);
    float3 baseColor = float3(baseColorSample.rgb);

    half4 mrSample = metalRoughTex.sample(s, in.uv);
    float metallic  = float(mrSample.b);   // glTF convention: B = metallic
    float roughness = float(mrSample.g);   // glTF convention: G = roughness
    roughness = max(roughness, 0.08);

    float3 N = normalize(in.worldNormal);
    float3 V = normalize(frame.cameraPosition - in.worldPos);

    // Blinn-Phong lighting (same 3-light setup as node_fragment)
    float specPower = max(1.0, 2.0 / (roughness * roughness) - 2.0);
    float3 diffuseAccum = float3(0);
    float3 specAccum    = float3(0);

    // Directional lights
    for (uint i = 0; i < lights.directionalLightCount; i++) {
        float3 L = normalize(lights.directionalLights[i].direction);
        float NdotL = max(0.0, dot(N, L));
        float3 H = normalize(L + V);
        float NdotH = max(0.0, dot(N, H));
        float spec = pow(NdotH, specPower);

        float3 lightColor = lights.directionalLights[i].color * lights.directionalLights[i].intensity;
        diffuseAccum += lightColor * NdotL;
        specAccum    += lightColor * spec * (0.3 + metallic * 0.5);
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
        specAccum    += lightColor * spec * (0.3 + metallic * 0.5);
    }

    // Fresnel-ish metallic blend
    float3 diffColor = baseColor * (1.0 - metallic * 0.6);
    float3 ambient = diffColor * max(lights.ambientIntensity, 0.35);  // higher ambient floor for mascot
    float3 lit = diffColor * diffuseAccum + specAccum + ambient;

    // Fresnel rim glow — makes silhouette pop against dark background
    float fresnel = pow(1.0 - max(0.0, dot(N, V)), 3.0);
    lit += float3(0.3, 0.5, 0.8) * fresnel * 0.6;

    // Emissive from per-part uniforms
    float3 emissive = part.emissive.rgb * part.emissive.a;
    lit += emissive;

    // Distance fog — same as scene nodes so mascot blends into the environment.
    float dist = length(in.worldPos - frame.cameraPosition);
    float fogT = mascotFogFactor(dist, frame.fogNear, frame.fogFar);
    float3 foggedColor = mix(lit, frame.bgColor, fogT);

    return float4(foggedColor, 1.0);
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - GPU Mascot Matrix Compute Kernel
// ──────────────────────────────────────────────────────────────────────────────

// Inline matrix helpers (replicate MascotSystem's Swift matrix helpers)
inline float4x4 gpu_translation(float3 t) {
    return float4x4(float4(1,0,0,0), float4(0,1,0,0), float4(0,0,1,0), float4(t.x,t.y,t.z,1));
}

inline float4x4 gpu_scale(float s) {
    return float4x4(float4(s,0,0,0), float4(0,s,0,0), float4(0,0,s,0), float4(0,0,0,1));
}

inline float4x4 gpu_rotationX(float a) {
    float c = cos(a), s = sin(a);
    return float4x4(float4(1,0,0,0), float4(0,c,s,0), float4(0,-s,c,0), float4(0,0,0,1));
}

inline float4x4 gpu_rotationY(float a) {
    float c = cos(a), s = sin(a);
    return float4x4(float4(c,0,-s,0), float4(0,1,0,0), float4(s,0,c,0), float4(0,0,0,1));
}

inline float4x4 gpu_rotationZ(float a) {
    float c = cos(a), s = sin(a);
    return float4x4(float4(c,s,0,0), float4(-s,c,0,0), float4(0,0,1,0), float4(0,0,0,1));
}

// Compute matrices for one mascot → writes directly to MascotInstanceData.
kernel void mascot_compute_matrices(
    const device MascotAnimState*   animStates [[buffer(0)]],
    device MascotInstanceData*      instances  [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    MascotAnimState a = animStates[tid];

    float bobY = sin(a.time * a.bobSpeed) * a.dynamicScale * a.bobAmplitude;
    float leanZ = sin(a.time * 1.3) * 0.08 + a.flyLean;
    float leanX = sin(a.time * 0.9) * 0.06;

    float4x4 bodyTranslation = gpu_translation(a.currentPosition + float3(0, bobY, 0));
    float4x4 bodyYaw = gpu_rotationY(a.currentYaw);
    float4x4 bodyLeanZ = gpu_rotationZ(leanZ);
    float4x4 bodyLeanX = gpu_rotationX(leanX);
    float4x4 bodyScale = gpu_scale(a.dynamicScale);
    float4x4 bodyRot = bodyYaw * bodyLeanZ * bodyLeanX;
    float4x4 bodyRotScale = bodyRot * bodyScale;
    float4x4 bodyMatrix = bodyTranslation * bodyRotScale;

    float3 zp = float3(0, 0, 0);

    // Left arm: pivot rotation in model space
    float3 leftPivot = a.leftPivot;
    float4x4 toPivotL = gpu_translation(-leftPivot);
    float4x4 fromPivotL = gpu_translation(leftPivot);
    float4x4 leftArmRot = gpu_rotationX(a.leftSwing);
    float4x4 leftArmLocal = fromPivotL * leftArmRot * toPivotL;
    float4x4 leftArmMatrix = bodyTranslation * bodyRot * bodyScale * leftArmLocal;

    // Right arm: pivot rotation in model space
    float3 rightPivot = a.rightPivot;
    float4x4 toPivotR = gpu_translation(-rightPivot);
    float4x4 fromPivotR = gpu_translation(rightPivot);
    float4x4 rightArmRot = gpu_rotationX(a.rightSwing);
    float4x4 rightArmLocal = fromPivotR * rightArmRot * toPivotR;
    float4x4 rightArmMatrix = bodyTranslation * bodyRot * bodyScale * rightArmLocal;

    // Eye emissive
    float eyePulse = a.eyePulseMax * (0.6 + 0.4 * sin(a.time * a.eyeFreq));

    // Body (part 0)
    instances[tid].parts[0].modelMatrix  = bodyMatrix;
    instances[tid].parts[0].parentMatrix = bodyMatrix;
    instances[tid].parts[0].emissive     = float4(0, 0, 0, 0);
    instances[tid].parts[0].blendPivot   = zp;
    instances[tid].parts[0].blendRadius  = 0;

    // Left arm (part 1) — skinned
    instances[tid].parts[1].modelMatrix  = leftArmMatrix;
    instances[tid].parts[1].parentMatrix = bodyMatrix;
    instances[tid].parts[1].emissive     = float4(0, 0, 0, 0);
    instances[tid].parts[1].blendPivot   = leftPivot;
    instances[tid].parts[1].blendRadius  = 1.0;

    // Right arm (part 2) — skinned
    instances[tid].parts[2].modelMatrix  = rightArmMatrix;
    instances[tid].parts[2].parentMatrix = bodyMatrix;
    instances[tid].parts[2].emissive     = float4(0, 0, 0, 0);
    instances[tid].parts[2].blendPivot   = rightPivot;
    instances[tid].parts[2].blendRadius  = 1.0;

    // Eye (part 3)
    instances[tid].parts[3].modelMatrix  = bodyMatrix;
    instances[tid].parts[3].parentMatrix = bodyMatrix;
    instances[tid].parts[3].emissive     = float4(a.projectTint, eyePulse);
    instances[tid].parts[3].blendPivot   = zp;
    instances[tid].parts[3].blendRadius  = 0;

    // Bottom (part 4)
    float bottomPulse = 0.4 + 0.3 * sin(a.time * 4.0);
    instances[tid].parts[4].modelMatrix  = bodyMatrix;
    instances[tid].parts[4].parentMatrix = bodyMatrix;
    instances[tid].parts[4].emissive     = float4(0.2, 0.4, 1.0, bottomPulse);
    instances[tid].parts[4].blendPivot   = zp;
    instances[tid].parts[4].blendRadius  = 0;

    instances[tid].projectTint = a.projectTint;
    instances[tid]._instancePad = 0;
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - GPU Thruster Particle Compute Kernels
// ──────────────────────────────────────────────────────────────────────────────

// Kernel 1: Advance particle life, position, velocity. Mark dead particles.
kernel void thruster_advect(
    device ThrusterParticleGPU*     particles [[buffer(0)]],
    constant ThrusterComputeParams& params    [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.totalParticles) return;

    device ThrusterParticleGPU& p = particles[tid];
    if (p.alive == 0) return;

    p.life += params.dt;
    if (p.life >= p.maxLife) {
        p.alive = 0;
        return;
    }

    p.position += p.velocity * params.dt;
    p.velocity *= (1.0 - params.dt * params.damping);
}

// Kernel 2: Write spawn requests into dead particle slots (ring buffer).
kernel void thruster_spawn(
    device ThrusterParticleGPU*         particles [[buffer(0)]],
    const device ThrusterSpawnRequest*  spawns    [[buffer(1)]],
    constant ThrusterComputeParams&     params    [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.spawnCount) return;

    // Ring buffer write: wrap around total capacity
    uint slot = (params.spawnOffset + tid) % params.totalParticles;

    device ThrusterParticleGPU& p = particles[slot];
    ThrusterSpawnRequest s = spawns[tid];

    p.position = s.position;
    p.velocity = s.velocity;
    p.life     = 0.0;
    p.maxLife  = s.maxLife;
    p.size     = s.size;
    p.color    = s.color;
    p.alive    = 1;
}

// Kernel 3: Pack alive particles into FlowParticleVertex output buffer.
// Uses atomic counter to compact alive particles into contiguous draw buffer.
kernel void thruster_pack_vertices(
    device ThrusterParticleGPU*     particles   [[buffer(0)]],
    device FlowParticleVertex*      output      [[buffer(1)]],
    device atomic_uint&             drawCount   [[buffer(2)]],
    constant ThrusterComputeParams& params      [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.totalParticles) return;

    device ThrusterParticleGPU& p = particles[tid];
    if (p.alive == 0) return;

    uint outIdx = atomic_fetch_add_explicit(&drawCount, 1, memory_order_relaxed);

    float t = p.life / p.maxLife;
    float opacity;
    if (t < 0.05) { opacity = t / 0.05; }
    else { opacity = (1.0 - t) / 0.95; }
    opacity *= 0.85;

    output[outIdx].position = p.position;
    output[outIdx].uv       = float2(0, 0);
    output[outIdx].color    = float4(p.color, opacity);
    output[outIdx].size     = p.size * (1.0 - t * 0.5);
    output[outIdx]._pad0    = 0;
    output[outIdx]._pad1    = 0;
    output[outIdx]._pad2    = 0;
}
