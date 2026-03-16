#include <metal_stdlib>
#include <RealityKit/RealityKit.h>
#include "SharedTypes.h"

using namespace metal;

// Fog parameters in world (RealityKit scaled) coordinates.
// Internal fogNear=100, fogFar=1200 → world: 0.5, 6.0 (scaleFactor = 1/200).
constant float kFogNear = 0.5;
constant float kFogFar  = 6.0;

/// Compute distance from fragment to camera using the world-to-view matrix.
/// In view space the camera sits at the origin, so length(viewPos) = camera distance.
inline float cameraDistance(float3 worldPos, float4x4 worldToView)
{
    float3 viewPos = (worldToView * float4(worldPos, 1.0)).xyz;
    return length(viewPos);
}

/// Standard fog ramp shared by nodes and edges.
inline float fogFactor(float dist)
{
    return saturate((dist - kFogNear) / (kFogFar - kFogNear));
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Node Surface Shader (lit)
//
// custom_parameter layout: float4(stateType, effectIntensity, searchDimmed, 0)
//   stateType:       0 = normal, 1 = selected, 2 = recalled, 3 = arriving,
//                    4 = searchMatched
//   effectIntensity: glow / arrival intensity (0–1)
//   searchDimmed:    1.0 when dimmed by search spotlight
// ──────────────────────────────────────────────────────────────────────────────

[[visible]]
void node_surface(realitykit::surface_parameters params)
{
    float4 custom       = params.uniforms().custom_parameter();
    float  stateType    = custom.x;
    float  intensity    = custom.y;
    float  searchDimmed = custom.z;

    float  time     = params.uniforms().time();
    float3 worldPos = params.geometry().world_position();
    float  dist     = cameraDistance(worldPos, params.uniforms().world_to_view());

    float  fogT      = fogFactor(dist);
    float  depthFade = max(0.0f, 1.0f - fogT * 0.95f);

    // ── Fog opacity ──────────────────────────────────────────────────────
    float fogOpacity;
    if (stateType > 0.5 && stateType < 1.5) {
        fogOpacity = 1.0;                                     // selected → opaque
    } else if (stateType > 3.5 && stateType < 4.5) {
        // search matched → gentle fade, floor at 0.35 so always visible
        fogOpacity = max(0.35f, 1.0f - fogT * 0.5f);
    } else {
        fogOpacity = max(0.08f, 1.0f - fogT * 0.92f);
    }
    if (searchDimmed > 0.5) {
        fogOpacity = min(fogOpacity, 0.12f);                  // search spotlight dim
    }

    // ── Base color from material tint ────────────────────────────────────
    float3 tint = params.material_constants().base_color_tint();

    // ── Emissive based on state ──────────────────────────────────────────
    half3  emissiveColor     = half3(0);
    float  emissiveIntensity = 0;

    if (stateType > 0.5 && stateType < 1.5) {
        // Selected — bright white
        emissiveColor     = half3(1.0);
        emissiveIntensity = 1.0;
    } else if (stateType > 1.5 && stateType < 2.5) {
        // Recalled — blue-white pulsing
        emissiveColor     = half3(0.7, 0.9, 1.0);
        float pulse       = 1.0 + sin(time * 5.0) * 0.3;
        emissiveIntensity = 8.0 * intensity * pulse * depthFade;
    } else if (stateType > 2.5 && stateType < 3.5) {
        // Arriving — golden pulsing
        emissiveColor     = half3(1.0, 0.85, 0.4);
        float pulse       = 1.0 + sin(time * 3.5) * 0.3;
        emissiveIntensity = 8.0 * intensity * pulse * depthFade;
    } else if (stateType > 3.5 && stateType < 4.5) {
        // Search matched — bright cyan, gentle depth fade (floor 0.3) for depth cue
        emissiveColor     = half3(0.0, 0.9, 1.0);
        float searchFade  = max(0.3f, depthFade);
        emissiveIntensity = 8.0 * (1.0 + sin(time * 4.0) * 0.3) * searchFade;
    } else {
        // Normal — subtle self-glow matching base color
        emissiveColor     = half3(tint);
        emissiveIntensity = 0.15;
    }

    // ── Output ───────────────────────────────────────────────────────────
    params.surface().set_base_color(half3(tint));
    params.surface().set_emissive_color(emissiveColor * half(emissiveIntensity));
    params.surface().set_opacity(half(fogOpacity));
    params.surface().set_roughness(half(0.15));
    params.surface().set_metallic(half(0.05));
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Node Geometry Modifier
//
// Same custom_parameter layout as node_surface.
// Applies a scale-pulse for recalled / arriving / search-matched nodes so the
// sphere visibly breathes without touching CPU.
// ──────────────────────────────────────────────────────────────────────────────

[[visible]]
void node_geometry(realitykit::geometry_parameters params)
{
    float4 custom    = params.uniforms().custom_parameter();
    float  stateType = custom.x;
    float  intensity = custom.y;
    float  time      = params.uniforms().time();

    float scalePulse = 1.0;

    if (stateType > 1.5 && stateType < 2.5) {
        scalePulse = 1.0 + sin(time * 4.0) * 0.2 * intensity;   // recalled
    } else if (stateType > 2.5 && stateType < 3.5) {
        scalePulse = 1.0 + sin(time * 3.0) * 0.2 * intensity;   // arriving
    } else if (stateType > 3.5 && stateType < 4.5) {
        scalePulse = 1.0 + sin(time * 4.0) * 0.12;              // search matched
    }

    if (scalePulse != 1.0) {
        float3 pos = params.geometry().model_position();
        params.geometry().set_model_position_offset(pos * (scalePulse - 1.0));
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Edge Surface Shader (unlit)
//
// Per-edge state encoded in vertex attributes (for LowLevelMesh batching):
//   uv0.x = edgeState: 0=normal, 1=connected, 2=searchDimmed, 3=semanticMode
//   color = per-edge tint (project color or white for connected)
//
// Computes spatial sin() pulse from world_position (traveling wave along edge)
// and fog from camera distance — eliminates all per-edge CPU animation.
// ──────────────────────────────────────────────────────────────────────────────

[[visible]]
void edge_surface(realitykit::surface_parameters params)
{
    // Per-edge state from vertex UV (batched mesh)
    float edgeState = params.geometry().uv0().x;

    float  time     = params.uniforms().time();
    float3 worldPos = params.geometry().world_position();
    float  dist     = cameraDistance(worldPos, params.uniforms().world_to_view());

    float  fogT      = fogFactor(dist);
    float  depthFade = max(0.0f, 1.0f - fogT * 0.95f);

    // Spatial pulse — traveling wave along the edge
    float phase = (worldPos.x + worldPos.y + worldPos.z) * 8.0;
    float pulse = (sin(time * 3.0 + phase) + 1.0) * 0.5;  // 0..1

    // ── Opacity ──────────────────────────────────────────────────────────
    float opacity;
    if (edgeState > 1.5 && edgeState < 2.5) {
        opacity = 0.03 * depthFade;                            // searchDimmed
    } else if (edgeState > 0.5 && edgeState < 1.5) {
        opacity = (0.35 + pulse * 0.15) * max(0.12f, depthFade); // connected: visible but fades
    } else if (edgeState > 2.5) {
        opacity = (0.04 + pulse * 0.04) * depthFade;           // semanticMode: very subtle
    } else {
        opacity = (0.08 + pulse * 0.06) * depthFade;           // normal: subtle, distance-faded
    }

    // Cull fragments entirely beyond fog range (non-connected only)
    if (dist > kFogFar && edgeState < 0.5) {
        opacity = 0.0;
    }

    // TRANSPARENT — real alpha blending. Edges properly disappear against
    // any background (dark scene, nebulae, etc.) instead of becoming dark lines.
    half3 vertexColor = half3(params.geometry().color().rgb);

    params.surface().set_emissive_color(vertexColor);
    params.surface().set_opacity(half(opacity));
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Node Batch Surface Shader (lit, real sphere geometry)
//
// All alive nodes rendered as a single LowLevelMesh containing real sphere
// triangles (no billboards). A Metal compute kernel stamps template sphere
// vertices at each node's position/scale with per-vertex color + packed state.
//
// Vertex layout (48 bytes, same as edges):
//   position (float3)  = sphere vertex in world space (stamped by compute)
//   normal   (float3)  = sphere normal (from template, unit sphere)
//   uv0      (float2)  = unused
//   color    (float4)  = (baseColor.r, baseColor.g, baseColor.b, packedState)
//
// packedState encoding in color.a:
//   floor(packed) = rawState → stateType + (searchDimmed ? 10 : 0)
//   fract(packed) * 100 = effectIntensity (0–1)
// ──────────────────────────────────────────────────────────────────────────────

[[visible]]
void node_batch_lit_surface(realitykit::surface_parameters params)
{
    // Per-node state from vertex color
    float4 vertColor   = params.geometry().color();
    float3 baseColor   = vertColor.rgb;
    float  packed      = vertColor.a;
    float  rawState    = floor(packed);
    float  intensity   = clamp(fract(packed) * 100.0, 0.0, 1.0);
    bool   dimmed      = rawState >= 9.5;
    float  stateType   = dimmed ? rawState - 10.0 : rawState;

    float  time     = params.uniforms().time();
    float3 worldPos = params.geometry().world_position();
    float  dist     = cameraDistance(worldPos, params.uniforms().world_to_view());

    float  fogT      = fogFactor(dist);
    float  depthFade = max(0.0f, 1.0f - fogT * 0.95f);

    // ── Fog opacity ────────────────────────────────────────────────────
    bool isSearchMatched = (stateType > 3.5 && stateType < 4.5);
    float fogOpacity;
    if (stateType > 0.5 && stateType < 1.5) {
        fogOpacity = 1.0;                                     // selected → opaque
    } else if (isSearchMatched) {
        fogOpacity = 1.0;                                     // search match → fog-proof
    } else {
        fogOpacity = max(0.08f, 1.0f - fogT * 0.92f);
    }
    if (dimmed) {
        fogOpacity = min(fogOpacity, 0.12f);                  // search spotlight dim
    }

    // ── Emissive based on state ────────────────────────────────────────
    half3  emissiveColor     = half3(0);
    float  emissiveIntensity = 0;

    if (stateType > 0.5 && stateType < 1.5) {
        emissiveColor     = half3(1.0);
        emissiveIntensity = 1.0;
    } else if (stateType > 1.5 && stateType < 2.5) {
        emissiveColor     = half3(0.7, 0.9, 1.0);
        float pulse       = 1.0 + sin(time * 5.0) * 0.3;
        emissiveIntensity = 8.0 * intensity * pulse * depthFade;
    } else if (stateType > 2.5 && stateType < 3.5) {
        emissiveColor     = half3(1.0, 0.85, 0.4);
        float pulse       = 1.0 + sin(time * 3.5) * 0.3;
        emissiveIntensity = 8.0 * intensity * pulse * depthFade;
    } else if (isSearchMatched) {
        emissiveColor     = half3(0.0, 0.9, 1.0);
        emissiveIntensity = 8.0 * (1.0 + sin(time * 4.0) * 0.3);  // no depthFade — visible at any distance
    } else {
        emissiveColor     = half3(baseColor);
        emissiveIntensity = 0.15;
    }

    // ── Output (OPAQUE — fog baked into base color, no alpha) ─────────
    // Background color matches the SwiftUI background (0.051, 0.067, 0.09).
    // Fog fades base color toward background instead of using transparency.
    // This enables early-Z rejection → massive performance win.
    const half3 kBgColor = half3(0.051, 0.067, 0.09);
    half3 foggedBase = mix(kBgColor, half3(baseColor), half(fogOpacity));
    params.surface().set_base_color(foggedBase);
    params.surface().set_emissive_color(emissiveColor * half(emissiveIntensity) * half(fogOpacity));
    params.surface().set_roughness(half(0.15));
    params.surface().set_metallic(half(0.05));
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Node Batch Geometry Modifier
//
// Scale-pulse for recalled/arriving/search-matched nodes on GPU.
// Reads packed state from vertex color.a (same encoding as surface shader).
// ──────────────────────────────────────────────────────────────────────────────

[[visible]]
void node_batch_geometry(realitykit::geometry_parameters params)
{
    float4 vertColor = params.geometry().color();
    float  packed    = vertColor.a;
    float  rawState  = floor(packed);
    float  intensity = clamp(fract(packed) * 100.0, 0.0, 1.0);
    float  stateType = rawState >= 9.5 ? rawState - 10.0 : rawState;
    float  time      = params.uniforms().time();

    float scalePulse = 1.0;

    if (stateType > 1.5 && stateType < 2.5) {
        scalePulse = 1.0 + sin(time * 4.0) * 0.2 * intensity;   // recalled
    } else if (stateType > 2.5 && stateType < 3.5) {
        scalePulse = 1.0 + sin(time * 3.0) * 0.2 * intensity;   // arriving
    } else if (stateType > 3.5 && stateType < 4.5) {
        scalePulse = 1.0 + sin(time * 4.0) * 0.12;              // search matched
    }

    if (scalePulse != 1.0) {
        // Pulse along vertex normal — fixed amplitude (~20% of base nodeRadius 0.04).
        // Using normal avoids the world-origin scaling bug (old: pos * (scalePulse-1)).
        // Fixed amplitude avoids writing to uv0 which causes .lit pipeline artifacts.
        float3 normal = params.geometry().normal();
        params.geometry().set_model_position_offset(normal * 0.008 * (scalePulse - 1.0));
    }
}

// Old stamp_node_spheres and stamp_label_quads compute kernels removed.
// New versions using SharedTypes.h struct layouts live in RenderShaders.metal.

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Label Billboard Geometry Modifier
//
// Billboard quad: vertex position is the anchor (node world pos), normal carries
// pre-multiplied corner offset (cx*halfW, cy*halfH, 0). The geometry modifier
// expands corners along camera right/up vectors so quads always face the camera.
// Distance-based scaling (sqrt) keeps labels readable at varying distances.
// ──────────────────────────────────────────────────────────────────────────────

[[visible]]
void label_geometry(realitykit::geometry_parameters params)
{
    float3 corner = params.geometry().normal();  // (cx*halfW, cy*halfH, 0)

    // Camera basis from model_to_view matrix (entity has identity transform)
    float4x4 mv = params.uniforms().model_to_view();
    float3 camRight = float3(mv[0][0], mv[1][0], mv[2][0]);
    float3 camUp    = float3(mv[0][1], mv[1][1], mv[2][1]);

    // Distance-based partial perspective scaling (sqrt = same curve as old overlay)
    float3 anchor = params.geometry().model_position();
    float3 viewT = float3(mv[3][0], mv[3][1], mv[3][2]);
    float3x3 viewR = float3x3(mv[0].xyz, mv[1].xyz, mv[2].xyz);
    float3 camPos = -(transpose(viewR) * viewT);
    float dist = length(anchor - camPos);
    float distScale = sqrt(0.5 / max(dist, 0.05));

    float3 offset = (camRight * corner.x + camUp * corner.y) * distScale;

    // corner.z (from vertex nz) encodes forward bias toward camera.
    // Project labels use nz > 0 to render in front of edges.
    if (corner.z > 0.0) {
        float3 camForward = -float3(mv[0][2], mv[1][2], mv[2][2]);
        offset += camForward * corner.z;
    }

    params.geometry().set_model_position_offset(offset);
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Label Billboard Surface Shader (unlit, textured)
//
// Samples the label texture atlas. Per-vertex opacity in color.a, tint from color.rgb.
// Fog applied from camera distance. Alpha from texture × vertex × fog.
// ──────────────────────────────────────────────────────────────────────────────

[[visible]]
void label_surface(realitykit::surface_parameters params)
{
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    half4 texColor = params.textures().custom().sample(s, params.geometry().uv0());

    float vertexOpacity = params.geometry().color().a;
    float3 worldPos = params.geometry().world_position();
    float dist = cameraDistance(worldPos, params.uniforms().world_to_view());
    float fogFade = max(0.0f, 1.0f - fogFactor(dist) * 0.95f);

    half alpha = texColor.a * half(vertexOpacity) * half(fogFade);
    if (alpha < 0.01h) { params.surface().set_opacity(0.0h); return; }

    // Vertex color RGB: (1,1,1) for node labels (white), project color for project labels
    half3 tint = half3(params.geometry().color().rgb);
    params.surface().set_emissive_color(tint);
    params.surface().set_opacity(alpha);
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Force Compute Kernel
//
// Each of N threads computes charge (Coulomb) repulsion from all other N nodes.
// Reads ForceNodeFull directly — velocity fields are unused (integration on CPU).
// ──────────────────────────────────────────────────────────────────────────────

struct ForceParams {
    float chargeStrength;
    float crossChargeMultiplier;
    float sameTopicChargeScale;
    float sameProjectChargeScale;
    float cutoffSq;
    uint  nodeCount;
};

kernel void compute_charge_forces(
    device const ForceNodeFull* nodes   [[buffer(0)]],
    device       float3*    forces  [[buffer(1)]],
    constant     ForceParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.nodeCount) return;

    float3 pos_i = float3(nodes[tid].px, nodes[tid].py, nodes[tid].pz);
    int    pg_i  = nodes[tid].projectGroup;
    int    tg_i  = nodes[tid].topicGroup;

    float3 totalForce = float3(0);

    for (uint j = 0; j < params.nodeCount; j++) {
        if (j == tid) continue;

        float3 delta  = pos_i - float3(nodes[j].px, nodes[j].py, nodes[j].pz);
        float  distSq = dot(delta, delta);
        if (distSq > params.cutoffSq) continue;
        if (distSq < 1.0) distSq = 1.0;

        float charge;
        if (nodes[j].projectGroup != pg_i) {
            charge = params.chargeStrength * params.crossChargeMultiplier;
        } else if (nodes[j].topicGroup == tg_i) {
            charge = params.chargeStrength * params.sameTopicChargeScale;
        } else {
            charge = params.chargeStrength * params.sameProjectChargeScale;  // same project, different topic
        }

        float dist     = sqrt(distSq);
        float forceMag = charge / distSq;
        totalForce    += (delta / dist) * forceMag;
    }

    forces[tid] = totalForce;
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Barnes-Hut Charge Force Kernel (O(n log n))
//
// Each thread walks the octree for its node using a stack-based DFS.
// CPU builds the tree and uploads the flattened node array.
// GPU parallelizes the per-node tree traversal.
// ──────────────────────────────────────────────────────────────────────────────

kernel void compute_charge_forces_bh(
    device const ForceNodeFull*  nodes  [[buffer(0)]],
    device       float3*         forces [[buffer(1)]],
    device const BHOctreeNode*   tree   [[buffer(2)]],
    constant     BHChargeParams& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.nodeCount) return;

    float3 pos_i = float3(nodes[tid].px, nodes[tid].py, nodes[tid].pz);
    int pg_i = nodes[tid].projectGroup;
    int tg_i = nodes[tid].topicGroup;

    float3 totalForce = float3(0);

    // Stack-based DFS traversal of octree (no recursion on GPU).
    // Max stack depth: ~8 * tree_depth. For 20K nodes, depth ~15, stack ~120.
    int stack[128];
    int stackTop = 0;
    stack[0] = 0;  // root
    stackTop = 1;

    int maxIter = min((int)params.treeNodeCount * 4, 65536);
    int iter = 0;
    while (stackTop > 0 && iter < maxIter) {
        iter++;
        int nIdx = stack[--stackTop];
        if (nIdx < 0 || nIdx >= (int)params.treeNodeCount) continue;
        BHOctreeNode cell = tree[nIdx];

        if (cell.mass == 0) continue;

        // Leaf with single body — exact pairwise
        if (cell.bodyIndex >= 0) {
            int j = cell.bodyIndex;
            if ((uint)j >= params.nodeCount) continue;
            if ((uint)j == tid) continue;

            float3 delta = pos_i - float3(nodes[j].px, nodes[j].py, nodes[j].pz);
            float distSq = dot(delta, delta);
            if (distSq > params.cutoffSq) continue;
            if (distSq < 1.0) distSq = 1.0;

            float charge;
            if (nodes[j].projectGroup != pg_i) {
                charge = params.chargeStrength * params.crossChargeMultiplier;
            } else if (nodes[j].topicGroup == tg_i) {
                charge = params.chargeStrength * params.sameTopicChargeScale;
            } else {
                charge = params.chargeStrength * params.sameProjectChargeScale;
            }

            float dist = sqrt(distSq);
            totalForce += (delta / dist) * (charge / distSq);
            continue;
        }

        // Internal node — check opening angle: s²/d² < θ²
        float3 delta = pos_i - float3(cell.comX, cell.comY, cell.comZ);
        float distSq = dot(delta, delta);
        if (distSq < 1.0) distSq = 1.0;

        float cellSize = cell.halfSize * 2.0;
        float sSq = cellSize * cellSize;

        // Force monopole when cell is tiny (co-located nodes at max depth).
        // Without this, degenerate 40-deep chains make every thread walk O(n).
        if (sSq < 0.01 || sSq < distSq * params.thetaSq) {
            // Treat cell as point mass at center of mass
            if (distSq > params.cutoffSq) continue;

            float dist = sqrt(distSq);
            float forceMag = params.chargeStrength * cell.mass / distSq;
            totalForce += (delta / dist) * forceMag;
            continue;
        }

        // Too close — check if nearest point of cell is beyond cutoff
        float nearX = clamp(pos_i.x, cell.cx - cell.halfSize, cell.cx + cell.halfSize);
        float nearY = clamp(pos_i.y, cell.cy - cell.halfSize, cell.cy + cell.halfSize);
        float nearZ = clamp(pos_i.z, cell.cz - cell.halfSize, cell.cz + cell.halfSize);
        float3 nearDelta = pos_i - float3(nearX, nearY, nearZ);
        if (dot(nearDelta, nearDelta) > params.cutoffSq) continue;

        // Push children onto stack (bounds-checked to prevent GPU hang)
        for (int oct = 0; oct < 8; oct++) {
            int c = cell.children[oct];
            if (c > 0 && c < (int)params.treeNodeCount && stackTop < 128) {
                stack[stackTop++] = c;
            }
        }
    }

    forces[tid] = totalForce;
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Spring Force Kernel (gather via edge CSR)
//
// One thread per NODE. Iterates adjacency list, accumulates spring forces locally.
// Zero atomics — only tid writes to forces[tid].
// ──────────────────────────────────────────────────────────────────────────────

kernel void compute_spring_forces(
    device const uint*           adjOffsets   [[buffer(0)]],
    device const uint*           adjNeighbors [[buffer(1)]],
    device const ForceNodeFull*  nodes        [[buffer(2)]],
    device       float3*         forces       [[buffer(3)]],
    constant     ForceSimParams& params       [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.nodeCount) return;
    float3 myPos = float3(nodes[tid].px, nodes[tid].py, nodes[tid].pz);
    int myPG = nodes[tid].projectGroup;
    int myGG = nodes[tid].galaxyGroup;
    float3 total = float3(0);
    for (uint e = adjOffsets[tid]; e < adjOffsets[tid + 1]; e++) {
        uint j = adjNeighbors[e];
        // Skip cross-galaxy edges — no spring force between galaxies
        if (nodes[j].galaxyGroup != myGG) continue;
        float3 delta = float3(nodes[j].px, nodes[j].py, nodes[j].pz) - myPos;
        float d = max(length(delta), 1.0f);
        bool cross = nodes[j].projectGroup != myPG;
        float rest = cross ? params.crossProjectSpringLength : params.springLength;
        total += (delta / d) * params.springStrength * (d - rest);
    }
    forces[tid] += total;
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Centroid Accumulation Kernel (gather via group membership CSR)
//
// One thread per GROUP. Sequentially sums its members' positions.
// Zero atomics — only tid writes to centroids[tid].
// ──────────────────────────────────────────────────────────────────────────────

kernel void compute_group_centroids(
    device const uint*           memberOffsets [[buffer(0)]],
    device const uint*           members       [[buffer(1)]],
    device const ForceNodeFull*  nodes         [[buffer(2)]],
    device       GroupCentroid*  centroids     [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint start = memberOffsets[tid];
    uint end   = memberOffsets[tid + 1];
    uint count = end - start;
    if (count == 0) { centroids[tid].count = 0; return; }
    float3 sum = float3(0);
    for (uint i = start; i < end; i++) {
        uint n = members[i];
        sum += float3(nodes[n].px, nodes[n].py, nodes[n].pz);
    }
    centroids[tid].sumX = sum.x;
    centroids[tid].sumY = sum.y;
    centroids[tid].sumZ = sum.z;
    centroids[tid].count = int(count);
}

// Atomic float add helper using compare-and-swap on int reinterpretation.
// Used by centroid repulsion kernel (low-contention with ~30 groups).
inline void atomic_add_float(device float* addr, float val) {
    uint expected = as_type<uint>(*addr);
    while (true) {
        uint desired = as_type<uint>(as_type<float>(expected) + val);
        if (atomic_compare_exchange_weak_explicit(
                (device atomic_uint*)addr, &expected, desired,
                memory_order_relaxed, memory_order_relaxed))
            break;
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Centroid Repulsion Kernel
//
// Pass B: One thread per group pair. O(groups²) pairwise repulsion.
// Each thread handles one (g1, g2) pair using triangular indexing.
// ──────────────────────────────────────────────────────────────────────────────

kernel void compute_centroid_repulsion(
    device       GroupCentroid*  centroids  [[buffer(0)]],
    device       float3*        groupForce [[buffer(1)]],  // per-group accumulated force
    constant     float&         repulsion  [[buffer(2)]],
    constant     uint&          groupCount [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    // Triangular iteration: tid maps to (g1, g2) pair where g1 < g2
    uint totalPairs = groupCount * (groupCount - 1) / 2;
    if (tid >= totalPairs) return;

    // Decode triangular index: tid → (g1, g2)
    // g1 = groupCount - 1 - floor((sqrt(8*(totalPairs-1-tid)+1) - 1) / 2)
    // This is the standard inverse triangular number formula
    uint t = totalPairs - 1 - tid;
    uint g2 = groupCount - 1 - uint(floor((sqrt(float(8 * t + 1)) - 1.0) * 0.5));
    uint g1 = tid - (g2 * (g2 - 1)) / 2;
    if (g1 >= g2 || g2 >= groupCount) return;

    if (centroids[g1].count == 0 || centroids[g2].count == 0) return;

    float3 c1 = float3(centroids[g1].sumX, centroids[g1].sumY, centroids[g1].sumZ) / float(centroids[g1].count);
    float3 c2 = float3(centroids[g2].sumX, centroids[g2].sumY, centroids[g2].sumZ) / float(centroids[g2].count);

    float3 delta = c1 - c2;
    float dist = length(delta);
    if (dist < 1.0) dist = 1.0;

    float force = repulsion / (dist * dist);
    float3 fVec = (delta / dist) * force;

    float f1 = 1.0 / float(centroids[g1].count);
    float f2 = 1.0 / float(centroids[g2].count);

    atomic_add_float(&((device float*)groupForce)[g1 * 3 + 0], fVec.x * f1);
    atomic_add_float(&((device float*)groupForce)[g1 * 3 + 1], fVec.y * f1);
    atomic_add_float(&((device float*)groupForce)[g1 * 3 + 2], fVec.z * f1);
    atomic_add_float(&((device float*)groupForce)[g2 * 3 + 0], -fVec.x * f2);
    atomic_add_float(&((device float*)groupForce)[g2 * 3 + 1], -fVec.y * f2);
    atomic_add_float(&((device float*)groupForce)[g2 * 3 + 2], -fVec.z * f2);
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Apply Cohesion + Centroid Forces Kernel
//
// Pass C: One thread per node. Applies per-group cohesion pull and
// distributes per-group centroid repulsion forces.
// ──────────────────────────────────────────────────────────────────────────────

kernel void apply_cohesion_forces(
    device const ForceNodeFull*  nodes      [[buffer(0)]],
    device const GroupCentroid*  centroids  [[buffer(1)]],
    device const float3*         groupForce [[buffer(2)]],
    device       float3*         forces     [[buffer(3)]],
    constant     ForceSimParams& params     [[buffer(4)]],
    constant     uint&           groupType  [[buffer(5)]],  // 0 = project, 1 = topic
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.nodeCount) return;

    int group = (groupType == 0) ? nodes[tid].projectGroup : nodes[tid].topicGroup;
    if (group < 0 || centroids[group].count < 2) return;

    float3 centroid = float3(centroids[group].sumX, centroids[group].sumY, centroids[group].sumZ) / float(centroids[group].count);
    float3 pos = float3(nodes[tid].px, nodes[tid].py, nodes[tid].pz);
    float3 delta = centroid - pos;

    float cohStr = (groupType == 0) ? params.cohesionStrength : params.topicCohesionStrength;

    // For project groups, apply non-linear cohesion (quadratic beyond reference radius)
    if (groupType == 0) {
        float dist = length(delta);
        float refR = 30.0;  // simplified from 75th percentile — GPU-friendly
        float ratio = max(1.0, dist / refR);
        cohStr *= ratio * ratio;
    }

    float3 cohForce = delta * cohStr;

    // Add per-group centroid repulsion force
    float3 groupF = groupForce[group];

    forces[tid] += cohForce + groupF;
}

