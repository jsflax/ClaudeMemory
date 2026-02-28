#!/usr/bin/env python3
"""
Convert a GLB robot mascot model into Engram's custom binary mesh format.

Parses GLB binary directly (no external dependencies), splits the mesh into
5 animatable parts via connected-component analysis and centroid classification,
extracts PBR textures, and outputs:
  - mascot_mesh.bin   (header + per-part vertex/index data)
  - mascot_basecolor.jpg
  - mascot_metalrough.png

Binary format:
  uint32 partCount (5)
  Per part (5x): uint32 vertexCount, uint32 indexCount, float3 pivot  (20 bytes each)
  Then per part: [MascotVertex...] then [uint32 indices...]
  MascotVertex = float3 position + float3 normal + float2 uv  (32 bytes)

Usage:
  python3 scripts/convert_mascot_glb.py ~/Downloads/model-2.glb
"""

import json
import struct
import sys
import os
from collections import defaultdict


# ── GLB Parsing ──────────────────────────────────────────────────────────────

def parse_glb(path):
    """Parse a GLB file into JSON chunk and binary chunk."""
    with open(path, "rb") as f:
        data = f.read()

    magic, version, length = struct.unpack_from("<III", data, 0)
    assert magic == 0x46546C67, f"Not a GLB file (magic={magic:#x})"
    assert version == 2, f"Unsupported GLB version {version}"

    offset = 12
    json_data = None
    bin_data = None

    while offset < length:
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        chunk_body = data[offset + 8 : offset + 8 + chunk_len]
        if chunk_type == 0x4E4F534A:  # JSON
            json_data = json.loads(chunk_body)
        elif chunk_type == 0x004E4942:  # BIN
            bin_data = chunk_body
        offset += 8 + chunk_len

    assert json_data is not None, "No JSON chunk in GLB"
    assert bin_data is not None, "No BIN chunk in GLB"
    return json_data, bin_data


def read_accessor(gltf, bin_data, accessor_idx):
    """Read a glTF accessor into a list of tuples."""
    accessor = gltf["accessors"][accessor_idx]
    bv = gltf["bufferViews"][accessor["bufferView"]]
    offset = bv.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    count = accessor["count"]
    stride = bv.get("byteStride", 0)

    comp_type = accessor["componentType"]
    acc_type = accessor["type"]

    # Component size and format
    comp_fmt = {5120: "b", 5121: "B", 5122: "h", 5123: "H",
                5125: "I", 5126: "f"}[comp_type]
    comp_size = struct.calcsize(comp_fmt)

    # Number of components
    n_comp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
              "MAT2": 4, "MAT3": 9, "MAT4": 16}[acc_type]

    elem_size = comp_size * n_comp
    if stride == 0:
        stride = elem_size

    result = []
    for i in range(count):
        o = offset + i * stride
        vals = struct.unpack_from(f"<{n_comp}{comp_fmt}", bin_data, o)
        result.append(vals if n_comp > 1 else vals[0])
    return result


def extract_image(gltf, bin_data, image_idx):
    """Extract an image from the GLB binary chunk."""
    image = gltf["images"][image_idx]
    bv = gltf["bufferViews"][image["bufferView"]]
    offset = bv.get("byteOffset", 0)
    length = bv["byteLength"]
    mime = image.get("mimeType", "")
    data = bin_data[offset : offset + length]
    return data, mime


# ── Union-Find ───────────────────────────────────────────────────────────────

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


# ── Part Classification ──────────────────────────────────────────────────────

PART_NAMES = ["body", "left_arm", "right_arm", "eye", "bottom"]


def classify_component(centroid):
    """Classify a connected component into one of 5 parts by its centroid."""
    x, y, z = centroid

    # Eye: front-facing elements near center-top
    if abs(x) < 0.55 and z > 0.5 and -0.5 < y < 0.5:
        return 3  # eye

    # Bottom thruster area
    if y < -0.6 and abs(x) < 0.5:
        return 4  # bottom

    # Left arm (negative X in model space) — only the arm below the ball joint
    if x < -0.55 and y < -0.35:
        return 1  # left_arm

    # Right arm (positive X in model space) — only the arm below the ball joint
    if x > 0.55 and y < -0.35:
        return 2  # right_arm

    return 0  # body (default)


# ── Main Conversion ──────────────────────────────────────────────────────────

def convert(glb_path, output_dir):
    print(f"Parsing {glb_path}...")
    gltf, bin_data = parse_glb(glb_path)

    # Find the mesh (assume first mesh, first primitive)
    mesh = gltf["meshes"][0]
    prim = mesh["primitives"][0]

    # Read vertex attributes
    positions = read_accessor(gltf, bin_data, prim["attributes"]["POSITION"])
    normals = read_accessor(gltf, bin_data, prim["attributes"]["NORMAL"])
    texcoords = read_accessor(gltf, bin_data, prim["attributes"].get("TEXCOORD_0", prim["attributes"]["POSITION"]))
    indices = read_accessor(gltf, bin_data, prim["indices"])

    n_verts = len(positions)
    n_tris = len(indices) // 3
    print(f"  {n_verts} vertices, {n_tris} triangles")

    # Build connected components via union-find on triangle adjacency
    uf = UnionFind(n_verts)
    for i in range(n_tris):
        i0, i1, i2 = indices[i * 3], indices[i * 3 + 1], indices[i * 3 + 2]
        uf.union(i0, i1)
        uf.union(i1, i2)

    # Group vertices by component root
    components = defaultdict(list)
    for v in range(n_verts):
        components[uf.find(v)].append(v)
    print(f"  {len(components)} connected components")

    # Compute component centroids and classify into parts
    part_verts = [set() for _ in range(5)]   # vertex indices per part
    part_tris = [[] for _ in range(5)]        # (i0,i1,i2) per part

    comp_part = {}  # component root -> part index
    for root, verts in components.items():
        cx = sum(positions[v][0] for v in verts) / len(verts)
        cy = sum(positions[v][1] for v in verts) / len(verts)
        cz = sum(positions[v][2] for v in verts) / len(verts)
        part_idx = classify_component((cx, cy, cz))
        comp_part[root] = part_idx
        for v in verts:
            part_verts[part_idx].add(v)

    # Assign triangles to parts (by first vertex's component)
    for i in range(n_tris):
        i0 = indices[i * 3]
        root = uf.find(i0)
        part_idx = comp_part[root]
        part_tris[part_idx].append((
            indices[i * 3], indices[i * 3 + 1], indices[i * 3 + 2]
        ))

    # Print part stats
    for i, name in enumerate(PART_NAMES):
        print(f"  {name}: {len(part_verts[i])} verts, {len(part_tris[i])} tris")

    # Compute pivots
    pivots = []
    for i in range(5):
        verts = part_verts[i]
        if not verts:
            pivots.append((0.0, 0.0, 0.0))
            continue
        if i == 1:  # left_arm pivot at ball joint center
            pivots.append((-0.68, -0.21, 0.10))
        elif i == 2:  # right_arm pivot at ball joint center
            pivots.append((0.67, -0.21, 0.10))
        else:
            # Centroid
            cx = sum(positions[v][0] for v in verts) / len(verts)
            cy = sum(positions[v][1] for v in verts) / len(verts)
            cz = sum(positions[v][2] for v in verts) / len(verts)
            pivots.append((cx, cy, cz))

    # Compute per-vertex skin weights for arm parts.
    # Weight = smoothstep(distance_from_pivot / blend_radius)
    # 0.0 at shoulder → vertex stays with body; 1.0 at hand → full arm swing.
    arm_blend_radius = {}  # part_idx -> blend radius
    for arm_idx in (1, 2):
        pivot = pivots[arm_idx]
        max_dist = 0
        for v in part_verts[arm_idx]:
            dx = positions[v][0] - pivot[0]
            dy = positions[v][1] - pivot[1]
            dz = positions[v][2] - pivot[2]
            d = (dx*dx + dy*dy + dz*dz) ** 0.5
            if d > max_dist:
                max_dist = d
        arm_blend_radius[arm_idx] = max_dist
        print(f"  {PART_NAMES[arm_idx]} blend radius: {max_dist:.3f}")

    def compute_skin_weight(part_idx, pos):
        """Compute skinning weight for a vertex. 0=body, 1=full part transform."""
        if part_idx not in (1, 2):
            return 0.0  # non-arm parts don't blend
        pivot = pivots[part_idx]
        dx = pos[0] - pivot[0]
        dy = pos[1] - pivot[1]
        dz = pos[2] - pivot[2]
        d = (dx*dx + dy*dy + dz*dz) ** 0.5
        radius = arm_blend_radius[part_idx]
        if radius < 0.001:
            return 1.0
        t = min(d / radius, 1.0)
        # Smoothstep for natural deformation
        return t * t * (3.0 - 2.0 * t)

    # Build per-part remapped vertex/index data
    part_data = []
    for i in range(5):
        # Remap global vertex indices to zero-based per-part
        sorted_verts = sorted(part_verts[i])
        remap = {global_idx: local_idx for local_idx, global_idx in enumerate(sorted_verts)}

        local_positions = []
        local_normals = []
        local_uvs = []
        local_weights = []
        for global_idx in sorted_verts:
            local_positions.append(positions[global_idx])
            local_normals.append(normals[global_idx])
            if isinstance(texcoords[global_idx], (tuple, list)):
                local_uvs.append(texcoords[global_idx])
            else:
                local_uvs.append((0.0, 0.0))
            local_weights.append(compute_skin_weight(i, positions[global_idx]))

        local_indices = []
        for i0, i1, i2 in part_tris[i]:
            local_indices.extend([remap[i0], remap[i1], remap[i2]])

        part_data.append({
            "positions": local_positions,
            "normals": local_normals,
            "uvs": local_uvs,
            "weights": local_weights,
            "indices": local_indices,
        })

    # ── Write binary mesh ────────────────────────────────────────────────────

    os.makedirs(output_dir, exist_ok=True)
    bin_path = os.path.join(output_dir, "mascot_mesh.bin")
    with open(bin_path, "wb") as f:
        # Header: part count
        f.write(struct.pack("<I", 5))

        # Per-part header: vertexCount, indexCount, pivot (float3)
        for i in range(5):
            vc = len(part_data[i]["positions"])
            ic = len(part_data[i]["indices"])
            px, py, pz = pivots[i]
            f.write(struct.pack("<IIfff", vc, ic, px, py, pz))

        # Per-part vertex + index data
        for i in range(5):
            pd = part_data[i]
            # Vertices: position(float3) + normal(float3) + uv(float2) + skinWeight(float) = 36 bytes
            for j in range(len(pd["positions"])):
                px, py, pz = pd["positions"][j]
                nx, ny, nz = pd["normals"][j]
                u, v = pd["uvs"][j]
                w = pd["weights"][j]
                f.write(struct.pack("<fffffffff", px, py, pz, nx, ny, nz, u, v, w))
            # Indices: uint32
            for idx in pd["indices"]:
                f.write(struct.pack("<I", idx))

    print(f"  Wrote {bin_path} ({os.path.getsize(bin_path)} bytes)")

    # ── Extract textures ─────────────────────────────────────────────────────

    # Find material textures
    if "materials" in gltf and gltf["materials"]:
        mat = gltf["materials"][0]
        pbr = mat.get("pbrMetallicRoughness", {})

        # Base color texture
        if "baseColorTexture" in pbr:
            tex_idx = pbr["baseColorTexture"]["index"]
            tex = gltf["textures"][tex_idx]
            img_idx = tex["source"]
            img_data, mime = extract_image(gltf, bin_data, img_idx)
            ext = ".jpg" if "jpeg" in mime or "jpg" in mime else ".png"
            out_path = os.path.join(output_dir, f"mascot_basecolor{ext}")
            with open(out_path, "wb") as f:
                f.write(img_data)
            print(f"  Wrote {out_path} ({len(img_data)} bytes)")

        # Metallic-roughness texture
        if "metallicRoughnessTexture" in pbr:
            tex_idx = pbr["metallicRoughnessTexture"]["index"]
            tex = gltf["textures"][tex_idx]
            img_idx = tex["source"]
            img_data, mime = extract_image(gltf, bin_data, img_idx)
            ext = ".png" if "png" in mime else ".jpg"
            out_path = os.path.join(output_dir, f"mascot_metalrough{ext}")
            with open(out_path, "wb") as f:
                f.write(img_data)
            print(f"  Wrote {out_path} ({len(img_data)} bytes)")

    print("Done!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path/to/model.glb> [output_dir]")
        sys.exit(1)

    glb_path = os.path.expanduser(sys.argv[1])
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "Sources", "EngramVisualizer", "Resources"
    )
    convert(glb_path, output_dir)
