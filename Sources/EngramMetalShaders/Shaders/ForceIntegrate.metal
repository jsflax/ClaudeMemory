#include <metal_stdlib>
using namespace metal;

#include "../../CEngramSceneTypes/include/SharedTypes.h"

/// GPU kernel: apply accumulated forces to positions and velocities.
/// Matches JS force-integrate.wgsl exactly:
///   vel = (vel + force) * damping;
///   if (length(vel) > maxSpeed) vel *= maxSpeed / length(vel);
///   pos += vel;
/// Alpha is NOT used in integration (only in center gravity, which is in the spring kernel).
kernel void integrate_positions(
    device float3*            positions  [[buffer(0)]],
    device float3*            velocities [[buffer(1)]],
    device const float3*      forces     [[buffer(2)]],
    constant IntegrateParams& params     [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.nodeCount) return;

    float3 f = forces[gid];
    float3 v = (velocities[gid] + f) * params.damping;

    float speed = length(v);
    if (speed > params.maxSpeed) {
        v *= params.maxSpeed / speed;
    }

    velocities[gid] = v;
    positions[gid] += v;
}
