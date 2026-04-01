#!/usr/bin/env python3
"""
Build a USDZ mascot model with UsdSkel skeleton from the existing mascot_mesh.bin.

Reads the custom binary mesh format (5 parts with positions, normals, UVs, skin weights),
creates a proper USD skeleton with joints, binds the mesh, applies PBR textures,
and outputs a .usdz file ready for RealityKit.

Joint hierarchy:
  root (body)
  ├── left_arm   (pivot: -0.68, -0.21, 0.10)
  ├── right_arm  (pivot:  0.67, -0.21, 0.10)
  ├── eye        (pivot: centroid)
  └── bottom     (pivot: centroid)

Usage:
  python3 scripts/build_mascot_usdz.py [output_dir]
"""

import os
import struct
import shutil
import tempfile
from pathlib import Path

from pxr import Usd, UsdGeom, UsdSkel, UsdShade, Sdf, Vt, Gf


PART_NAMES = ["body", "left_arm", "right_arm", "eye", "bottom"]
JOINT_PATHS = [
    "root",
    "root/left_arm",
    "root/right_arm",
    "root/eye",
    "root/bottom",
]


def read_mesh_bin(path):
    """Read mascot_mesh.bin and return part data."""
    with open(path, "rb") as f:
        data = f.read()

    offset = 0
    (part_count,) = struct.unpack_from("<I", data, offset)
    offset += 4
    assert part_count == 5, f"Expected 5 parts, got {part_count}"

    # Read per-part headers
    headers = []
    for _ in range(part_count):
        vc, ic, px, py, pz = struct.unpack_from("<IIfff", data, offset)
        offset += 20
        headers.append((vc, ic, (px, py, pz)))

    # Read per-part vertex + index data
    parts = []
    for i in range(part_count):
        vc, ic, pivot = headers[i]

        positions = []
        normals = []
        uvs = []
        weights = []
        for _ in range(vc):
            vals = struct.unpack_from("<fffffffff", data, offset)
            offset += 36  # 9 floats * 4 bytes
            positions.append((vals[0], vals[1], vals[2]))
            normals.append((vals[3], vals[4], vals[5]))
            uvs.append((vals[6], vals[7]))
            weights.append(vals[8])

        indices = []
        for _ in range(ic):
            (idx,) = struct.unpack_from("<I", data, offset)
            offset += 4
            indices.append(idx)

        parts.append({
            "name": PART_NAMES[i],
            "pivot": pivot,
            "positions": positions,
            "normals": normals,
            "uvs": uvs,
            "weights": weights,
            "indices": indices,
        })

    return parts


def build_usd(parts, basecolor_path, metalrough_path, output_path):
    """Build a USD stage with skeleton-bound mesh."""
    stage = Usd.Stage.CreateNew(str(output_path))
    stage.SetMetadata("metersPerUnit", 1.0)
    stage.SetMetadata("upAxis", "Y")

    # Root xform
    root_xform = UsdGeom.Xform.Define(stage, "/Mascot")
    stage.SetDefaultPrim(root_xform.GetPrim())

    # ── Skeleton ──────────────────────────────────────────────────────────────

    skel_root = UsdSkel.Root.Define(stage, "/Mascot/SkelRoot")

    skeleton = UsdSkel.Skeleton.Define(stage, "/Mascot/SkelRoot/Skeleton")

    # Joint topology
    joints_attr = skeleton.CreateJointsAttr()
    joints_attr.Set(Vt.TokenArray(JOINT_PATHS))

    # Bind transforms (world-space transform of each joint at bind time)
    bind_transforms = []
    rest_transforms = []

    for i, part in enumerate(parts):
        px, py, pz = part["pivot"]
        if i == 0:
            # Root/body: identity (at origin)
            bind_transforms.append(Gf.Matrix4d(1.0))
            rest_transforms.append(Gf.Matrix4d(1.0))
        else:
            # Child joints: translated to pivot position
            m = Gf.Matrix4d(1.0)
            m.SetTranslateOnly(Gf.Vec3d(px, py, pz))
            bind_transforms.append(m)

            # Rest transform is relative to parent (root)
            rest_transforms.append(m)

    skeleton.CreateBindTransformsAttr().Set(
        Vt.Matrix4dArray(bind_transforms)
    )
    skeleton.CreateRestTransformsAttr().Set(
        Vt.Matrix4dArray(rest_transforms)
    )

    # ── Merge all parts into one mesh ─────────────────────────────────────────

    all_positions = []
    all_normals = []
    all_uvs = []
    all_indices = []
    all_joint_indices = []  # per-vertex: which joint influences this vert
    all_joint_weights = []  # per-vertex: weight of that joint
    vertex_offset = 0

    for part_idx, part in enumerate(parts):
        for j in range(len(part["positions"])):
            px, py, pz = part["positions"][j]
            all_positions.append(Gf.Vec3f(px, py, pz))

            nx, ny, nz = part["normals"][j]
            all_normals.append(Gf.Vec3f(nx, ny, nz))

            u, v = part["uvs"][j]
            # USD uses V=0 at bottom; mesh binary uses V=0 at top (OpenGL/Blender convention)
            all_uvs.append(Gf.Vec2f(u, 1.0 - v))

            skin_w = part["weights"][j]

            if part_idx == 0:
                # Body vertices: 100% root joint
                all_joint_indices.append(0)  # root
                all_joint_indices.append(0)  # padding (2 influences per vert)
                all_joint_weights.append(1.0)
                all_joint_weights.append(0.0)
            else:
                # Limb vertices: blend between root (body) and limb joint
                all_joint_indices.append(0)          # root
                all_joint_indices.append(part_idx)    # this limb's joint
                all_joint_weights.append(1.0 - skin_w)
                all_joint_weights.append(skin_w)

        for idx in part["indices"]:
            all_indices.append(idx + vertex_offset)

        vertex_offset += len(part["positions"])

    # Face vertex counts (all triangles)
    face_counts = [3] * (len(all_indices) // 3)

    # ── Mesh ──────────────────────────────────────────────────────────────────

    mesh = UsdGeom.Mesh.Define(stage, "/Mascot/SkelRoot/Mesh")
    mesh.CreatePointsAttr().Set(Vt.Vec3fArray(all_positions))
    mesh.CreateNormalsAttr().Set(Vt.Vec3fArray(all_normals))
    mesh.SetNormalsInterpolation("vertex")
    mesh.CreateFaceVertexCountsAttr().Set(Vt.IntArray(face_counts))
    mesh.CreateFaceVertexIndicesAttr().Set(Vt.IntArray(all_indices))
    mesh.CreateSubdivisionSchemeAttr().Set("none")

    # UVs as primvar
    uv_primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    uv_primvar.Set(Vt.Vec2fArray(all_uvs))

    # ── Skeleton Binding ──────────────────────────────────────────────────────

    binding = UsdSkel.BindingAPI.Apply(mesh.GetPrim())
    binding.CreateSkeletonRel().SetTargets(["/Mascot/SkelRoot/Skeleton"])

    # Joint indices: 2 influences per vertex, flat array
    ji_primvar = binding.CreateJointIndicesPrimvar(False, elementSize=2)
    ji_primvar.Set(Vt.IntArray(all_joint_indices))

    # Joint weights: 2 influences per vertex, flat array
    jw_primvar = binding.CreateJointWeightsPrimvar(False, elementSize=2)
    jw_primvar.Set(Vt.FloatArray(all_joint_weights))

    # Geometry bind transform (mesh space → skeleton space, identity here)
    binding.CreateGeomBindTransformAttr().Set(Gf.Matrix4d(1.0))

    # ── PBR Material ──────────────────────────────────────────────────────────

    material = UsdShade.Material.Define(stage, "/Mascot/SkelRoot/Material")

    # PBR shader
    shader = UsdShade.Shader.Define(stage, "/Mascot/SkelRoot/Material/PBRShader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface"
    )

    # Base color texture
    if basecolor_path and os.path.exists(basecolor_path):
        bc_reader = UsdShade.Shader.Define(
            stage, "/Mascot/SkelRoot/Material/BaseColorTex"
        )
        bc_reader.CreateIdAttr("UsdUVTexture")
        bc_reader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            os.path.basename(basecolor_path)
        )
        bc_reader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            UsdShade.Shader.Define(
                stage, "/Mascot/SkelRoot/Material/UVReader"
            ).ConnectableAPI(), "result"
        )
        bc_reader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            bc_reader.ConnectableAPI(), "rgb"
        )

        # UV reader
        uv_reader = UsdShade.Shader.Get(
            stage, "/Mascot/SkelRoot/Material/UVReader"
        )
        uv_reader.CreateIdAttr("UsdPrimvarReader_float2")
        uv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        uv_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    # Metallic-roughness texture
    if metalrough_path and os.path.exists(metalrough_path):
        mr_reader = UsdShade.Shader.Define(
            stage, "/Mascot/SkelRoot/Material/MetalRoughTex"
        )
        mr_reader.CreateIdAttr("UsdUVTexture")
        mr_reader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            os.path.basename(metalrough_path)
        )
        mr_reader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            UsdShade.Shader.Get(
                stage, "/Mascot/SkelRoot/Material/UVReader"
            ).ConnectableAPI(), "result"
        )
        # Blue channel = metallic, Green channel = roughness (glTF convention)
        mr_reader.CreateOutput("b", Sdf.ValueTypeNames.Float)
        mr_reader.CreateOutput("g", Sdf.ValueTypeNames.Float)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).ConnectToSource(
            mr_reader.ConnectableAPI(), "b"
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).ConnectToSource(
            mr_reader.ConnectableAPI(), "g"
        )

    # Bind material to mesh
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    stage.GetRootLayer().Save()
    print(f"  Wrote {output_path}")
    return str(output_path)


def package_usdz(usda_path, usdz_path, texture_paths):
    """Package .usda + textures into .usdz using usdzip."""
    import subprocess

    # Collect all files to package
    files = [usda_path] + [p for p in texture_paths if p and os.path.exists(p)]

    # Try usdzip first (from OpenUSD), then xcrun usdz_converter
    try:
        result = subprocess.run(
            ["python3", "-m", "pxr.usdzip", usdz_path] + files,
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  Packaged {usdz_path}")
            return
    except Exception:
        pass

    # Fallback: use UsdUtils to create the package
    from pxr import UsdUtils
    success = UsdUtils.CreateNewUsdzPackage(
        Sdf.AssetPath(usda_path), usdz_path
    )
    if success:
        print(f"  Packaged {usdz_path}")
    else:
        print(f"  WARNING: Could not package .usdz, use .usda directly")


def main():
    import sys

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    resources_dir = os.path.join(project_root, "Sources", "EngramVisualizer", "Resources")

    output_dir = sys.argv[1] if len(sys.argv) > 1 else resources_dir

    mesh_bin = os.path.join(resources_dir, "mascot_mesh.bin")
    basecolor = os.path.join(resources_dir, "mascot_basecolor.jpg")
    metalrough = os.path.join(resources_dir, "mascot_metalrough.png")

    if not os.path.exists(mesh_bin):
        print(f"Error: {mesh_bin} not found")
        sys.exit(1)

    print("Reading mascot_mesh.bin...")
    parts = read_mesh_bin(mesh_bin)
    for p in parts:
        print(f"  {p['name']}: {len(p['positions'])} verts, "
              f"{len(p['indices'])//3} tris, pivot={p['pivot']}")

    # Build in a temp directory so textures are co-located for packaging
    with tempfile.TemporaryDirectory() as tmpdir:
        usda_path = os.path.join(tmpdir, "mascot.usda")

        # Copy textures next to the .usda for proper relative paths
        tex_paths = []
        for src in [basecolor, metalrough]:
            if os.path.exists(src):
                dst = os.path.join(tmpdir, os.path.basename(src))
                shutil.copy2(src, dst)
                tex_paths.append(dst)
            else:
                tex_paths.append(None)

        print("Building USD stage...")
        build_usd(parts, tex_paths[0], tex_paths[1], usda_path)

        # Package as .usdz
        usdz_path = os.path.join(output_dir, "mascot.usdz")
        os.makedirs(output_dir, exist_ok=True)

        print("Packaging .usdz...")
        package_usdz(usda_path, usdz_path, tex_paths)

        # Also save the .usda for inspection
        usda_out = os.path.join(output_dir, "mascot.usda")
        shutil.copy2(usda_path, usda_out)
        print(f"  Saved {usda_out} (for inspection)")

    print(f"\nDone! Output: {usdz_path}")
    print("\nIn RealityKit:")
    print('  let mascot = try await ModelEntity.load(named: "mascot")')
    print('  // Animate joints:')
    print('  mascot.jointTransforms["root/left_arm"] = ...')


if __name__ == "__main__":
    main()
