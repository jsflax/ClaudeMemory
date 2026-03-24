#include <metal_stdlib>
using namespace metal;

#include "../../CEngramSceneTypes/include/SharedTypes.h"

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Brute-Force Charge Kernel (O(n²))
// ──────────────────────────────────────────────────────────────────────────────

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
    int    gg_i  = nodes[tid].galaxyGroup;
    bool   hasGalaxies = params.galaxyGroupCount > 1;

    float3 totalForce = float3(0);

    for (uint j = 0; j < params.nodeCount; j++) {
        if (j == tid) continue;
        // Skip charge between nodes in different galaxies
        if (hasGalaxies && nodes[j].galaxyGroup != gg_i) continue;

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
            charge = params.chargeStrength * params.sameProjectChargeScale;
        }

        float dist     = sqrt(distSq);
        float forceMag = charge / distSq;
        totalForce    += (delta / dist) * forceMag;
    }

    forces[tid] = totalForce;
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Barnes-Hut Charge Force Kernel (O(n log n))
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
    int gg_i = nodes[tid].galaxyGroup;
    bool hasGalaxies = params.galaxyGroupCount > 1;

    float3 totalForce = float3(0);

    // Stack-based DFS traversal of octree (no recursion on GPU).
    int stack[128];
    int stackTop = 0;
    stack[0] = 0;
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
            // Skip charge between nodes in different galaxies
            if (hasGalaxies && nodes[j].galaxyGroup != gg_i) continue;

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
        if (sSq < 0.01 || sSq < distSq * params.thetaSq) {
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

        // Push children onto stack
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
        float str = cross ? params.springStrength * params.crossProjectSpringScale : params.springStrength;
        total += (delta / d) * str * (d - rest);
    }
    forces[tid] += total;
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Centroid Accumulation Kernel (gather via group membership CSR)
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
        float3 p = float3(nodes[n].px, nodes[n].py, nodes[n].pz);
        sum += p;
    }
    centroids[tid].sumX = sum.x;
    centroids[tid].sumY = sum.y;
    centroids[tid].sumZ = sum.z;
    centroids[tid].count = int(count);
}

// Atomic float add helper using compare-and-swap on int reinterpretation.
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
// ──────────────────────────────────────────────────────────────────────────────

kernel void compute_centroid_repulsion(
    device       GroupCentroid*  centroids  [[buffer(0)]],
    device       float3*        groupForce [[buffer(1)]],
    constant     ForceSimParams& params    [[buffer(2)]],
    constant     uint&          groupCount [[buffer(3)]],
    constant     uint&          groupType  [[buffer(4)]],  // 0 = project, 1 = topic
    constant     float&         repulsion  [[buffer(5)]],
    device const int*           groupGalaxy [[buffer(6)]],  // per-group galaxy index (-1 = no filter)
    uint tid [[thread_position_in_grid]])
{
    // Triangular iteration: tid maps to (g1, g2) pair where g1 < g2
    uint totalPairs = groupCount * (groupCount - 1) / 2;
    if (tid >= totalPairs) return;

    uint g2 = uint(floor((sqrt(float(8 * tid + 1)) + 1.0) * 0.5));
    uint g1 = tid - g2 * (g2 - 1) / 2;
    if (g1 >= g2 || g2 >= groupCount) return;

    // Only repel projects within the same galaxy
    if (groupGalaxy[g1] >= 0 && groupGalaxy[g1] != groupGalaxy[g2]) return;

    if (centroids[g1].count == 0 || centroids[g2].count == 0) return;

    float3 c1 = float3(centroids[g1].sumX, centroids[g1].sumY, centroids[g1].sumZ) / float(centroids[g1].count);
    float3 c2 = float3(centroids[g2].sumX, centroids[g2].sumY, centroids[g2].sumZ) / float(centroids[g2].count);

    float3 delta = c1 - c2;
    float dist = length(delta);
    if (dist < 1.0) dist = 1.0;

    float force = repulsion / (dist * dist);
    float3 fVec = (delta / dist) * force;

    float f1 = 1.0 / sqrt(float(centroids[g1].count));
    float f2 = 1.0 / sqrt(float(centroids[g2].count));

    atomic_add_float(&((device float*)groupForce)[g1 * 3 + 0], fVec.x * f1);
    atomic_add_float(&((device float*)groupForce)[g1 * 3 + 1], fVec.y * f1);
    atomic_add_float(&((device float*)groupForce)[g1 * 3 + 2], fVec.z * f1);
    atomic_add_float(&((device float*)groupForce)[g2 * 3 + 0], -fVec.x * f2);
    atomic_add_float(&((device float*)groupForce)[g2 * 3 + 1], -fVec.y * f2);
    atomic_add_float(&((device float*)groupForce)[g2 * 3 + 2], -fVec.z * f2);
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Topic Centroid Repulsion Kernel (same-project filter)
// ──────────────────────────────────────────────────────────────────────────────

kernel void compute_topic_centroid_repulsion(
    device       GroupCentroid*  centroids        [[buffer(0)]],
    device       float3*        groupForce       [[buffer(1)]],
    constant     ForceSimParams& params           [[buffer(2)]],
    constant     uint&          groupCount       [[buffer(3)]],
    constant     float&         repulsion        [[buffer(4)]],
    device const int*           topicProjectGroup [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    // Triangular iteration: tid maps to (g1, g2) pair where g1 < g2
    uint totalPairs = groupCount * (groupCount - 1) / 2;
    if (tid >= totalPairs) return;

    // Decode triangular index
    uint g2 = uint(floor((sqrt(float(8 * tid + 1)) + 1.0) * 0.5));
    uint g1 = tid - g2 * (g2 - 1) / 2;
    if (g1 >= g2 || g2 >= groupCount) return;

    // Only repel topics within the same project
    if (topicProjectGroup[g1] != topicProjectGroup[g2]) return;

    if (centroids[g1].count == 0 || centroids[g2].count == 0) return;

    float3 c1 = float3(centroids[g1].sumX, centroids[g1].sumY, centroids[g1].sumZ) / float(centroids[g1].count);
    float3 c2 = float3(centroids[g2].sumX, centroids[g2].sumY, centroids[g2].sumZ) / float(centroids[g2].count);

    float3 delta = c1 - c2;
    float dist = length(delta);
    if (dist < 1.0) dist = 1.0;

    float force = repulsion / (dist * dist);
    float3 fVec = (delta / dist) * force;

    float f1 = 1.0 / sqrt(float(centroids[g1].count));
    float f2 = 1.0 / sqrt(float(centroids[g2].count));

    atomic_add_float(&((device float*)groupForce)[g1 * 3 + 0], fVec.x * f1);
    atomic_add_float(&((device float*)groupForce)[g1 * 3 + 1], fVec.y * f1);
    atomic_add_float(&((device float*)groupForce)[g1 * 3 + 2], fVec.z * f1);
    atomic_add_float(&((device float*)groupForce)[g2 * 3 + 0], -fVec.x * f2);
    atomic_add_float(&((device float*)groupForce)[g2 * 3 + 1], -fVec.y * f2);
    atomic_add_float(&((device float*)groupForce)[g2 * 3 + 2], -fVec.z * f2);
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Project Reference Radius Kernel (75th percentile)
// ──────────────────────────────────────────────────────────────────────────────

kernel void compute_project_refR(
    device const uint*           memberOffsets [[buffer(0)]],
    device const uint*           members       [[buffer(1)]],
    device const ForceNodeFull*  nodes         [[buffer(2)]],
    device const GroupCentroid*  centroids     [[buffer(3)]],
    device       float*          refR          [[buffer(4)]],
    device       float*          scratch       [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    // tid = project group index
    if (centroids[tid].count < 2) { refR[tid] = 30.0; return; }

    float3 centroid = float3(centroids[tid].sumX, centroids[tid].sumY, centroids[tid].sumZ) / float(centroids[tid].count);

    uint start = memberOffsets[tid];
    uint end   = memberOffsets[tid + 1];
    uint count = end - start;
    if (count < 2) { refR[tid] = 30.0; return; }

    // Compute distances from centroid and find min/max
    float minDist = 1e20, maxDist = 0;
    for (uint i = start; i < end; i++) {
        uint n = members[i];
        float3 p = float3(nodes[n].px, nodes[n].py, nodes[n].pz);
        float d = length(p - centroid);
        scratch[i] = d;  // CSR indices are non-overlapping per group
        if (d < minDist) minDist = d;
        if (d > maxDist) maxDist = d;
    }

    if (maxDist - minDist < 0.001) { refR[tid] = max(30.0f, maxDist); return; }

    // Histogram with 256 bins spanning [minDist, maxDist]
    float binWidth = (maxDist - minDist) / 256.0;
    int bins[256];
    for (int b = 0; b < 256; b++) bins[b] = 0;

    for (uint i = start; i < end; i++) {
        int b = int((scratch[i] - minDist) / binWidth);
        if (b >= 256) b = 255;
        if (b < 0) b = 0;
        bins[b]++;
    }

    // Scan bins to find 75th percentile
    uint target = count * 3 / 4;  // index of P75 element
    if (target == 0) target = 1;
    uint cumulative = 0;
    float p75 = maxDist;
    for (int b = 0; b < 256; b++) {
        cumulative += bins[b];
        if (cumulative >= target) {
            p75 = minDist + (float(b) + 0.5) * binWidth;
            break;
        }
    }

    refR[tid] = max(30.0f, p75);
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Apply Cohesion + Centroid Forces Kernel
// ──────────────────────────────────────────────────────────────────────────────

kernel void apply_cohesion_forces(
    device const uint*           memberOffsets    [[buffer(0)]],
    device const uint*           members          [[buffer(1)]],
    device const ForceNodeFull*  nodes            [[buffer(2)]],
    device const GroupCentroid*  centroids        [[buffer(3)]],
    device const float3*         groupForce       [[buffer(4)]],
    device       float3*         forces           [[buffer(5)]],
    constant     ForceSimParams& params           [[buffer(6)]],
    constant     uint&           groupType        [[buffer(7)]],  // 0 = project, 1 = topic
    device const float*          refRBuffer       [[buffer(8)]],
    device const GroupCentroid*  projectCentroids [[buffer(9)]],   // project centroids (for topic leash)
    device const int*            topicProjectGroup [[buffer(10)]],  // topic group → project group mapping
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.nodeCount) return;

    int group = (groupType == 0) ? nodes[tid].projectGroup : nodes[tid].topicGroup;
    if (group < 0 || centroids[group].count < 2) return;

    float3 centroid = float3(centroids[group].sumX, centroids[group].sumY, centroids[group].sumZ) / float(centroids[group].count);
    float3 pos = float3(nodes[tid].px, nodes[tid].py, nodes[tid].pz);
    float3 delta = centroid - pos;

    float cohStr = (groupType == 0) ? params.cohesionStrength : params.topicCohesionStrength;

    // For project groups, apply non-linear cohesion (quadratic beyond reference radius).
    // refR = 75th-percentile distance from centroid, pre-computed by compute_project_refR kernel.
    if (groupType == 0) {
        float dist = length(delta);
        float r = refRBuffer[group];
        float ratio = max(1.0f, dist / r);
        cohStr *= ratio * ratio;
    }

    // Structural forces must stay constant throughout convergence.
    // Pre-multiply by 1/alpha so integration's `* alpha` cancels out: net = force * 1.
    float alphaInv = 1.0 / max(params.alpha, 0.001);
    float3 cohForce = delta * cohStr * alphaInv;
    float3 groupF = groupForce[group] * alphaInv;

    forces[tid] += cohForce + groupF;

    // Topic leash: prevent topic centroids from drifting too far from parent project centroid.
    // Only applies to topic groups (groupType==1). Pulls nodes back when their topic centroid
    // exceeds maxDrift (1.5× project reference radius) from the project centroid.
    // Force is proportional to excess drift only (normalized direction), avoiding oscillation.
    if (groupType == 1 && params.topicLeashStrength > 0) {
        int projGroup = topicProjectGroup[group];
        if (projGroup >= 0 && projectCentroids[projGroup].count > 0) {
            float3 projCentroid = float3(projectCentroids[projGroup].sumX,
                                         projectCentroids[projGroup].sumY,
                                         projectCentroids[projGroup].sumZ) / float(projectCentroids[projGroup].count);
            float drift = length(centroid - projCentroid);
            float maxDrift = refRBuffer[projGroup] * 1.5;
            if (drift > maxDrift) {
                float excess = drift - maxDrift;
                float3 leashDir = (projCentroid - centroid) / max(drift, 1.0f);
                float3 leashForce = leashDir * excess * params.topicLeashStrength * alphaInv;
                forces[tid] += leashForce;
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Center Gravity Kernel
// ──────────────────────────────────────────────────────────────────────────────

kernel void apply_center_gravity(
    device const ForceNodeFull*  nodes          [[buffer(0)]],
    device       float3*         forces         [[buffer(1)]],
    constant     ForceSimParams& params         [[buffer(2)]],
    device const float3*         galaxyCenters  [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.nodeCount) return;

    float3 pos = float3(nodes[tid].px, nodes[tid].py, nodes[tid].pz);
    float3 center;
    if (params.galaxyGroupCount > 1) {
        int gg = nodes[tid].galaxyGroup;
        center = galaxyCenters[gg];
    } else {
        center = params.center;
    }
    float3 delta = center - pos;

    forces[tid] += delta * params.centerStrength;
}
