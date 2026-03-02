#ifndef SharedTypes_h
#define SharedTypes_h

#include <simd/simd.h>

// ──────────────────────────────────────────────────────────────────────────────
// Shared C structs for Metal ↔ Swift interop.
// Both sides include this header (Metal via #include, Swift via bridging).
// ──────────────────────────────────────────────────────────────────────────────

// MARK: - Frame Uniforms

struct FrameUniforms {
    simd_float4x4 viewMatrix;
    simd_float4x4 projectionMatrix;
    simd_float4x4 viewProjectionMatrix;
    simd_float3   cameraPosition;     // world space
    float         time;
    simd_float3   bgColor;            // for fog baking (0.051, 0.067, 0.09)
    float         fogNear;            // world units
    float         fogFar;             // world units
    float         maintenancePulse;   // 0..1 lerp for maintenance atmosphere
    float         _pad1;
    float         _pad2;
};

// MARK: - Lighting

struct DirectionalLight {
    simd_float3 direction;   // normalized, world space (toward light)
    float       intensity;
    simd_float3 color;       // linear RGB
    float       _pad;
};

struct PointLightData {
    simd_float3 position;    // world space
    float       intensity;
    simd_float3 color;       // linear RGB
    float       attenuationRadius;
};

#define MAX_DIRECTIONAL_LIGHTS 3
#define MAX_POINT_LIGHTS 16

struct LightingUniforms {
    struct DirectionalLight directionalLights[MAX_DIRECTIONAL_LIGHTS];
    struct PointLightData   pointLights[MAX_POINT_LIGHTS];
    unsigned int            directionalLightCount;
    unsigned int            pointLightCount;
    float                   ambientIntensity;
    float                   _pad;
};

// MARK: - Vertex Layouts

// 48-byte vertex layout shared by nodes and edges.
// Matches the existing BatchVertex from Graph3DScene.
struct BatchVertex {
    simd_float3 position;   // 12 bytes
    simd_float3 normal;     // 12 bytes
    simd_float2 uv;         // 8 bytes
    simd_float4 color;      // 16 bytes
};                          // Total: 48 bytes

// Per-node instance data for stamp_node_spheres compute kernel (32 bytes).
struct NodeInstance {
    simd_float3 position;   // world position
    float       scale;      // sphere radius
    simd_float4 color;      // (r, g, b, packedState)
};

// Sphere template vertex for stamp kernel (24 bytes).
struct SphereTemplateVertex {
    simd_float3 position;
    simd_float3 normal;
};

// Stamp kernel dispatch parameters.
struct StampParams {
    unsigned int vertsPerNode;
    unsigned int nodeCount;
};

// Per-label instance for stamp_label_quads kernel (64 bytes).
struct LabelInstance {
    simd_float3  anchor;       // 12
    float        halfH;        // 4
    simd_float4  uvRect;       // 16 (u0, v0, u1, v1)
    simd_float4  color;        // 16 (r, g, b, baseOpacity)
    float        textAspect;   // 4
    float        maxVisible;   // 4
    float        forwardBias;  // 4
    unsigned int flags;        // 4
};                             // Total: 64 bytes

// Label stamp kernel dispatch parameters.
struct LabelStampParams {
    simd_float3  cameraPos;
    float        minDepth;
    float        depthRange;
    unsigned int labelCount;
    float        _pad0;
    float        _pad1;
};

// Per-edge instance data for stamp_edge_cylinders compute kernel (48 bytes).
struct EdgeInstance {
    simd_float3 sourcePos;   // 12  world-space endpoint
    float       radius;      //  4  cylinder radius
    simd_float3 targetPos;   // 12  world-space endpoint
    float       state;       //  4  edge visual state (0=normal, 1=connected, 2=dimmed, 3=semantic)
    simd_float4 color;       // 16  (r, g, b, 1)
};                           // Total: 48 bytes

// Edge stamp kernel dispatch parameters.
struct EdgeStampParams {
    unsigned int vertsPerEdge;   // always 12 (6-sided hexagonal cylinder)
    unsigned int edgeCount;
};

// Nebula billboard vertex (used by NebulaFogSystem).
struct NebulaQuadVertex {
    simd_float3 position;    // billboard center (world space)
    simd_float3 cornerOffset; // expansion offset (will be applied in vertex shader)
    simd_float2 uv;
    simd_float4 color;       // (r, g, b, alpha) — per-cluster tint
    float       noisePhase;  // per-quad seed for parallax
    float       radius;      // cluster radius for radial falloff
    float       _pad0;
    float       _pad1;
};

// Flow particle vertex.
struct FlowParticleVertex {
    simd_float3 position;    // world space center
    simd_float2 uv;
    simd_float4 color;       // (r, g, b, opacity)
    float       size;        // billboard expansion radius
    float       _pad0;
    float       _pad1;
    float       _pad2;
};

// MARK: - Mascot

// 32-byte vertex for mascot mesh parts.
struct MascotVertex {
    simd_float3 position;    // 12 bytes
    simd_float3 normal;      // 12 bytes
    simd_float2 uv;          // 8 bytes
};                           // Total: 32 bytes

// Per-part transform + emissive + skinning data for mascot.
struct MascotPartUniforms {
    simd_float4x4 modelMatrix;     // primary transform for this part
    simd_float4x4 parentMatrix;    // parent (body) transform for skinning blend
    simd_float4   emissive;        // (r, g, b, intensity)
    simd_float3   blendPivot;      // pivot in model space for distance-based blend
    float         blendRadius;     // distance over which blend transitions (0 = no blend)
};

// Uniforms for all 5 mascot parts.
#define MASCOT_PART_COUNT 5
struct MascotUniforms {
    struct MascotPartUniforms parts[MASCOT_PART_COUNT];
};

// Arcane circle orientation (passed to vertex shader for oriented billboard).
struct ArcaneCircleUniforms {
    simd_float3 center;
    float       size;
    simd_float3 right;      // mascot's local X axis in world space
    float       opacity;
    simd_float3 up;          // mascot's local Y axis in world space
    float       _pad0;
    simd_float3 tintColor;   // per-project tint for arcane circle / rings
    float       _pad1;
};

// Holographic info screen billboard (oriented to mascot's facing direction).
struct HoloScreenUniforms {
    simd_float3 center;          // world-space billboard center
    float       width;           // billboard half-width
    simd_float3 right;           // mascot's local X axis in world space
    float       height;          // billboard half-height
    simd_float3 up;              // mascot's local Y axis in world space
    float       opacity;         // fade in/out (0..1)
    float       revealProgress;  // typewriter reveal (0..1)
    float       _pad0;
    float       _pad1;
    float       _pad2;
};

#endif /* SharedTypes_h */
