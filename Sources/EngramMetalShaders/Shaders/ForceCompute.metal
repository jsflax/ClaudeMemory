#include <metal_stdlib>
using namespace metal;

#include "../../CEngramSceneTypes/include/SharedTypes.h"

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Brute-Force Charge Kernel (O(n²)) — matches JS force-charge.wgsl
// ──────────────────────────────────────────────────────────────────────────────

kernel void compute_charge_forces(
    device const float3*      positions [[buffer(0)]],
    device       float3*      forces    [[buffer(1)]],
    device const int4*        groups    [[buffer(2)]],   // (projId, topicId, galaxyId, 0)
    constant     ChargeParams& params   [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.nodeCount) return;

    float3 pos_i = positions[tid];
    int4   group_i = groups[tid];
    float3 totalForce = float3(0);

    for (uint j = 0; j < params.nodeCount; j++) {
        if (j == tid) continue;

        float3 pos_j = positions[j];
        int4   group_j = groups[j];
        float3 delta = pos_i - pos_j;
        float  distSq = dot(delta, delta);

        // Hard cutoff
        if (distSq > params.cutoffSq) continue;

        // Distance floor
        if (distSq < 1.0) distSq = 1.0;

        // Group-based charge scaling (matches JS exactly)
        float charge = params.chargeStrength;
        if (group_i.z != group_j.z) {
            // Different galaxy — base charge, no amplification
            charge *= 1.0;
        } else if (group_i.x != group_j.x) {
            // Same galaxy, different project
            charge *= params.crossChargeMultiplier;
        } else if (group_i.y == group_j.y) {
            // Same project, same topic
            charge *= params.sameTopicChargeScale;
        } else {
            // Same project, different topic
            charge *= params.sameProjectChargeScale;
        }

        // Coulomb: F = charge / distSq, direction = unit(delta)
        float dist = sqrt(distSq);
        float forceMag = charge / distSq;
        totalForce += (delta / dist) * forceMag;
    }

    // Charge is the first force pass — write directly (clears previous frame)
    forces[tid] = totalForce;
}

// ──────────────────────────────────────────────────────────────────────────────
// MARK: - Barnes-Hut Charge Force Kernel (O(n log n))
// ──────────────────────────────────────────────────────────────────────────────

kernel void compute_charge_forces_bh(
    device const float3*         positions [[buffer(0)]],
    device       float3*         forces    [[buffer(1)]],
    device const int4*           groups    [[buffer(2)]],
    device const BHOctreeNode*   tree      [[buffer(3)]],
    constant     BHChargeParams& params    [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.nodeCount) return;

    float3 pos_i = positions[tid];
    int4   group_i = groups[tid];
    bool   hasGalaxies = params.galaxyGroupCount > 1;

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
            int4 group_j = groups[j];
            if (hasGalaxies && group_j.z != group_i.z) continue;

            float3 delta = pos_i - positions[j];
            float distSq = dot(delta, delta);
            if (distSq > params.cutoffSq) continue;
            if (distSq < 1.0) distSq = 1.0;

            float charge = params.chargeStrength;
            if (group_i.z != group_j.z) {
                charge *= 1.0;
            } else if (group_i.x != group_j.x) {
                charge *= params.crossChargeMultiplier;
            } else if (group_i.y == group_j.y) {
                charge *= params.sameTopicChargeScale;
            } else {
                charge *= params.sameProjectChargeScale;
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
// MARK: - Spring + Structural Forces — matches JS force-spring.wgsl
// Single kernel combining springs, center gravity, project cohesion,
// topic cohesion, project centroid repulsion, topic centroid repulsion.
// Accumulates on top of charge forces in the force buffer.
// ──────────────────────────────────────────────────────────────────────────────

// groupMetadata layout: [0..numTopics) = topicToProject, [numTopics..numTopics+numProjects) = projectToGalaxy
// But we use JS convention: [0..128) = topicToProject, [128..256) = projectToGalaxy
// Actually, we just need topicToProject[t] at index t and projectToGalaxy[p] at METADATA_GALAXY_OFFSET + p

#define METADATA_GALAXY_OFFSET 128

kernel void compute_spring_and_structural(
    device const float3*        positions      [[buffer(0)]],
    device       float3*        forces         [[buffer(1)]],
    device const int4*          groups         [[buffer(2)]],   // (projId, topicId, galaxyId, 0)
    device const uint2*         edges          [[buffer(3)]],   // (src, tgt) per edge
    device const uint2*         edgeRanges     [[buffer(4)]],   // (start, count) per node
    device const float4*        centroids      [[buffer(5)]],   // [projects | topics | galaxies]
    device const uint2*         centroidMapping [[buffer(6)]],  // (projCentIdx, topicCentIdx) per node
    constant     SpringParams&  params         [[buffer(7)]],
    device const uint*          groupMetadata  [[buffer(8)]],   // topicToProject + projectToGalaxy
    device const float*         refRadius      [[buffer(9)]],   // 75th-pct radius per project
    uint tid [[thread_position_in_grid]])
{
    if (tid >= params.nodeCount) return;

    float3 pos_i = positions[tid];
    int4   group_i = groups[tid];
    uint   myGalaxyId = (uint)group_i.z;
    float3 force = forces[tid]; // accumulate on top of charge forces

    // === SPRING FORCES ===
    uint2 edgeRange = edgeRanges[tid];
    for (uint e = 0; e < edgeRange.y; e++) {
        uint2 edge = edges[edgeRange.x + e];
        // Determine the other node
        uint j = edge.y;
        if (edge.y == tid) { j = edge.x; }

        float3 pos_j = positions[j];
        int4   group_j = groups[j];

        // Skip cross-galaxy edges
        if ((uint)group_j.z != myGalaxyId) continue;

        float3 delta = pos_j - pos_i;
        float dist = length(delta);
        if (dist < 1.0) dist = 1.0;

        // Rest length depends on whether cross-project
        float restLength = params.springLength;
        if (group_i.x != group_j.x) {
            restLength = params.crossProjectSpringLength;
        }

        float displacement = dist - restLength;
        float3 springForce = (delta / dist) * params.springStrength * displacement;
        force += springForce;
    }

    // === CENTER FORCE (toward galaxy center, not origin) ===
    uint galaxyCentroidIdx = params.numProjects + params.numTopics + myGalaxyId;
    float3 galaxyCenter = centroids[galaxyCentroidIdx].xyz;
    float3 centerForce = (galaxyCenter - pos_i) * params.alpha * params.centerStrength;
    force += centerForce;

    // === PROJECT COHESION (with quadratic ramp) ===
    uint projIdx = centroidMapping[tid].x;
    float4 projCentroid = centroids[projIdx];
    if (projCentroid.w > 1.0) {
        float3 centroidPos = projCentroid.xyz;
        float3 toCentroid = centroidPos - pos_i;
        float distFromCentroid = length(toCentroid);
        float rr = refRadius[projIdx];
        float ratio = max(1.0f, distFromCentroid / max(rr, params.minRefRadius));
        force += toCentroid * params.cohesionStrength * ratio * ratio;
    }

    // === TOPIC COHESION (linear) ===
    uint topicIdx = centroidMapping[tid].y;
    float4 topicCentroid = centroids[topicIdx];
    if (topicCentroid.w > 1.0) {
        float3 toCentroid = topicCentroid.xyz - pos_i;
        force += toCentroid * params.topicCohesionStrength;
    }

    // === PROJECT CENTROID REPULSION (within same galaxy only) ===
    if (projCentroid.w > 0.0) {
        float3 myProjPos = projCentroid.xyz;
        float myProjCount = projCentroid.w;
        for (uint p = 0; p < params.numProjects; p++) {
            if (p == projIdx) continue;
            // Only repel projects within the same galaxy
            uint otherGalaxy = groupMetadata[METADATA_GALAXY_OFFSET + p];
            if (otherGalaxy != myGalaxyId) continue;

            float4 otherCentroid = centroids[p];
            if (otherCentroid.w < 1.0) continue;

            float3 delta = myProjPos - otherCentroid.xyz;
            float distSq = dot(delta, delta);
            if (distSq < 1.0) distSq = 1.0;
            float dist = sqrt(distSq);
            float forceMag = params.centroidRepulsion / distSq;
            force += (delta / dist) * forceMag / myProjCount;
        }
    }

    // === TOPIC CENTROID REPULSION (within same project only) ===
    uint maxProjects = params.numProjects;
    if (topicCentroid.w > 0.0) {
        float3 myTopicPos = topicCentroid.xyz;
        float myTopicCount = topicCentroid.w;
        uint myTopicProject = groupMetadata[topicIdx - maxProjects];

        for (uint t = 0; t < params.numTopics; t++) {
            uint tIdx = maxProjects + t;
            if (tIdx == topicIdx) continue;
            // Only repel topics within same project
            if (groupMetadata[t] != myTopicProject) continue;

            float4 otherCentroid = centroids[tIdx];
            if (otherCentroid.w < 1.0) continue;

            float3 delta = myTopicPos - otherCentroid.xyz;
            float distSq = dot(delta, delta);
            if (distSq < 1.0) distSq = 1.0;
            float dist = sqrt(distSq);
            float forceMag = params.topicCentroidRepulsion / distSq;
            force += (delta / dist) * forceMag / myTopicCount;
        }
    }

    forces[tid] = force;
}
