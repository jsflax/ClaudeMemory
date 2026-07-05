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

// Per-edge descriptor for GPU edge packing (rebuilt on topology/selection change).
// The pack_edge_instances kernel reads these + a position buffer → outputs EdgeInstance.
struct EdgeDescriptor {
    unsigned int sourceIdx;      // index into position buffer
    unsigned int targetIdx;      // index into position buffer
    float        sourceRadius;   // node radius (for inset calculation)
    float        targetRadius;   // node radius (for inset calculation)
    simd_float4  color;          // (r, g, b, 1)
    float        baseRadius;     // cylinder radius
    float        state;          // 0=normal, 1=connected, 2=dimmed, 3=semantic
    float        _pad0;
    float        _pad1;
};                               // Total: 48 bytes

// Dispatch parameters for pack_edge_instances compute kernel.
struct PackEdgeParams {
    unsigned int edgeCount;
    float        scaleFactor;    // world-space scale (1/200)
    float        _pad0;
    float        _pad1;
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

// Per-mascot instance data for instanced drawing (all 5 part transforms + tint).
#define MAX_FLEET_MASCOTS 10
struct MascotInstanceData {
    struct MascotPartUniforms parts[MASCOT_PART_COUNT];
    simd_float3 projectTint;
    float       _instancePad;
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

// Conjure orb billboard (camera-facing energy sphere between mascot's hands).
struct ConjureOrbUniforms {
    simd_float3 center;          // current orb world position (hands → node lerp)
    float       size;            // billboard half-extent
    simd_float3 right;           // camera right (orb faces camera)
    float       opacity;         // overall fade (0..1)
    simd_float3 up;              // camera up
    float       orbScale;        // grow factor (0→1 during charge, 1 during flight)
    simd_float3 tintColor;       // per-project tint
    float       glowIntensity;   // inner glow brightness (ramps during charge)
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

// MARK: - GPU Node Packing

// Per-node input for pack_node_instances compute kernel.
// CPU writes this contiguously; GPU reads position from separate buffer(7).
// Position field is written by CPU but ignored by shader (reads buffer(7) instead).
struct NodePackInput {
    simd_float3 position;       // sim-space position (legacy — shader reads buffer(7))
    float       baseRadius;     // pre-computed node radius (hub/importance adjusted)
    simd_float3 baseColor;      // project color (float3)
    float       packedState;    // encoded state (stateType + searchDimmed + intensity)
};

// Per-project centroid output from GPU node packing (atomic accumulation).
#define MAX_PROJECTS 64
struct ProjectCentroidGPU {
    simd_float3 sum;            // accumulated position sum
    int         count;          // number of nodes in this project
    float       maxY;           // max Y position for label placement
    float       _pad0;
    float       _pad1;
    float       _pad2;
};

// Parameters for pack_node_instances kernel.
struct NodePackParams {
    unsigned int nodeCount;
    float        scaleFactor;   // world-space scale (1/200)
    float        nodeRadius;    // base node radius
    float        animationTime; // for pulse effects
    unsigned int projectCount;  // number of active projects
    float        _pad0;
    float        _pad1;
    float        _pad2;
};

// Point light output from GPU node packing (atomic counter for compaction).
struct PointLightEntry {
    simd_float3 position;       // world-space position
    float       intensity;
    simd_float3 color;          // light color
    float       attenuation;
};

// MARK: - GPU Nebula Packing

// Per-group input for pack_nebula_vertices compute kernel.
struct NebulaGroupInput {
    simd_float3 centroid;       // world-space centroid (already scaled)
    float       radius;         // cluster radius (scaled)
    simd_float4 color;          // (r, g, b, alpha) — cached from NSColor
    float       noisePhase;     // per-group noise seed
    float       _pad0;
    float       _pad1;
    float       _pad2;
};

// Parameters for pack_nebula_vertices kernel.
struct NebulaPackParams {
    unsigned int groupCount;
    unsigned int quadsPerGroup;  // always 3
    float        _pad0;
    float        _pad1;
};

// MARK: - GPU Label Packing

// Per-node label metadata for pack_label_instances compute kernel.
struct LabelMetadata {
    simd_float4  uvRect;        // (u0, v0, u1, v1) from atlas
    float        halfH;         // label height
    float        textAspect;    // width/height ratio
    float        maxVisible;    // max visibility distance
    float        forwardBias;   // z-offset for project labels
    float        baseOpacity;   // base alpha
    unsigned int flags;         // bit 0: selected, bit 1: searchMatch, bit 2: searchDimmed
    float        _pad0;
    float        _pad1;
};

// Parameters for pack_label_instances kernel.
struct LabelPackParams {
    simd_float3  cameraPos;     // world-space camera position (scaled)
    float        minDepth;      // min camera distance
    float        depthRange;    // max - min camera distance
    float        scaleFactor;   // 1/200
    float        nodeRadius;    // for anchor offset
    unsigned int nodeCount;     // number of node labels
};

// MARK: - GPU Force Integration

// Parameters for integrate_positions compute kernel.
struct IntegrateParams {
    unsigned int nodeCount;
    float        damping;
    float        maxSpeed;
    float        _pad;
};

// MARK: - GPU Force Simulation

// Per-node data for GPU force computation.
// Used by charge and spring+structural kernels.
struct ForceNodeFull {
    float px, py, pz;          // position
    float vx, vy, vz;          // velocity (unused by new kernels, kept for layout compat)
    int   projectGroup;
    int   topicGroup;
    int   galaxyGroup;         // galaxy index for per-galaxy center force + cross-galaxy spring skip
    int   _pad;
};

// Parameters for brute-force O(n²) charge kernel (matches JS force-charge.wgsl).
struct ChargeParams {
    unsigned int nodeCount;
    float        chargeStrength;
    float        crossChargeMultiplier;
    float        sameProjectChargeScale;
    float        sameTopicChargeScale;
    float        cutoffSq;
    float        _pad0;
    float        _pad1;
};

// Parameters for spring + structural forces kernel (matches JS force-spring.wgsl).
struct SpringParams {
    unsigned int nodeCount;
    unsigned int edgeCount;
    float        alpha;
    float        springLength;
    float        crossProjectSpringLength;
    float        springStrength;
    float        centerStrength;
    float        cohesionStrength;
    float        topicCohesionStrength;
    float        centroidRepulsion;
    float        topicCentroidRepulsion;
    unsigned int numProjects;
    unsigned int numTopics;
    float        minRefRadius;
    unsigned int numGalaxies;
    float        _pad0;
};

// Barnes-Hut octree node for GPU charge computation (O(n log n)).
// Built on CPU, uploaded to GPU buffer, walked per-thread.
struct BHOctreeNode {
    float cx, cy, cz;          // cell geometric center
    float halfSize;             // half-width of cubic cell
    float comX, comY, comZ;     // center of mass
    float mass;                 // body count (as float for division)
    int   children[8];          // child indices (-1 = empty)
    int   bodyIndex;            // >= 0: leaf with single body; -1: internal node
    int   _pad;
};

// Parameters for Barnes-Hut charge kernel.
struct BHChargeParams {
    float chargeStrength;
    float crossChargeMultiplier;
    float sameTopicChargeScale;
    float sameProjectChargeScale;
    float cutoffSq;
    float thetaSq;              // opening angle threshold squared (0.7² = 0.49)
    unsigned int nodeCount;
    unsigned int treeNodeCount;
    unsigned int galaxyGroupCount;
    unsigned int _bhpad;
};

// MARK: - GPU Mascot Matrix Compute

// Scalar animation state (CPU → GPU) for mascot_compute_matrices kernel.
struct MascotAnimState {
    simd_float3 currentPosition;  // world position
    float       currentYaw;       // radians
    float       time;             // animation time
    float       dynamicScale;     // render-space scale
    float       bobSpeed;
    float       bobAmplitude;
    float       flyLean;          // forward tilt (0 when hovering, 0.2 when flying)
    float       leftSwing;        // left arm rotation angle
    float       rightSwing;       // right arm rotation angle
    float       eyePulseMax;      // eye emissive max intensity
    float       eyeFreq;          // eye pulse frequency
    float       bottomPulse;      // bottom thruster glow intensity
    simd_float3 projectTint;      // per-project tint color
    simd_float3 leftPivot;        // left arm pivot point in model space
    simd_float3 rightPivot;       // right arm pivot point in model space
    float       _pad0;
    float       _pad1;
};

// MARK: - GPU Thruster Particles

// Persistent GPU particle state (advected on GPU each frame).
struct ThrusterParticleGPU {
    simd_float3 position;    // world space
    simd_float3 velocity;    // world space
    float       life;        // 0..maxLife (advanced by dt each frame)
    float       maxLife;
    float       size;        // billboard radius
    simd_float3 color;       // RGB
    unsigned int alive;      // 1 = alive, 0 = dead/available
    float       _pad0;
};

// CPU → GPU spawn request (one per newly spawned particle).
struct ThrusterSpawnRequest {
    simd_float3 position;
    simd_float3 velocity;
    float       maxLife;
    float       size;
    simd_float3 color;
    float       _pad0;
};

// Compute kernel params for particle system.
struct ThrusterComputeParams {
    float        dt;
    float        damping;        // velocity damping per second (e.g., 1.5)
    unsigned int totalParticles; // max capacity of particle buffer
    unsigned int spawnCount;     // number of spawn requests this frame
    unsigned int spawnOffset;    // ring buffer write offset into particle buffer
    float        _pad0;
    float        _pad1;
    float        _pad2;
};

#endif /* SharedTypes_h */
