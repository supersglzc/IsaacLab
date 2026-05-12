"""Generate a hollow-frustum stack-cup mesh and convert it to USD with collision.

Output: <repo>/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack_cup/assets/cup.usd

Geometry (m):
    outer profile (lathed):  r=0.025 at z=0.000  →  r=0.0375 at z=0.080
    wall thickness:          5 mm everywhere; floor thickness 5 mm
    inner cavity (lathed):   r=0.020781 at z=0.005  →  r=0.030 at z=0.080  (open top)

The mesh is a single watertight surface: outer side + bottom disc + top annulus
(rim) + inner cavity wall + cavity floor. Wall thickness chosen so cup-(i+1)'s
outer-bottom (r=0.025) clears cup-(i)'s inner-top (r=0.030) by 5 mm radially —
giving a stable nesting fit during stacking.

Run inside IsaacLab `.venv` (Kit boots once for the converter):

    .venv/bin/python <repo>/nautilus/task-creation/triton-franka-stackcup/make_stack_cup_usd.py

This script is idempotent — re-runs overwrite the prior USD.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args([])
args.headless = True
args.enable_cameras = False
sim_app = AppLauncher(args).app


# ----- Cup geometry -----

OUTER_BOTTOM_R = 0.025
OUTER_TOP_R    = 0.0375
HEIGHT         = 0.080
WALL_THICK     = 0.005
FLOOR_THICK    = 0.005
N_SEGMENTS     = 64

# Inner profile: same taper as the outer, offset by WALL_THICK in r,
# starts at z=FLOOR_THICK (closed bottom below).
INNER_CAVITY_BOTTOM_R = OUTER_BOTTOM_R + (OUTER_TOP_R - OUTER_BOTTOM_R) * (FLOOR_THICK / HEIGHT) - WALL_THICK
INNER_TOP_R           = OUTER_TOP_R - WALL_THICK


def _ring(r: float, z: float, n: int) -> np.ndarray:
    """Return n points evenly spaced on a circle of radius r at height z."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t), np.full(n, z)], axis=-1)


def _quad_strip_faces(top_offset: int, bot_offset: int, n: int, *, flip: bool = False) -> list[tuple[int, int, int]]:
    """Triangulate two parallel rings of n vertices each, indices [offset .. offset+n).

    `flip=False`: outward-facing normals when the top ring is above the bottom and we go around CCW.
    `flip=True`:  inverts the winding (used for the inner cavity, where outward normals point inward).
    """
    faces: list[tuple[int, int, int]] = []
    for i in range(n):
        i_next = (i + 1) % n
        a = bot_offset + i
        b = bot_offset + i_next
        c = top_offset + i_next
        d = top_offset + i
        if flip:
            faces.append((a, c, b))
            faces.append((a, d, c))
        else:
            faces.append((a, b, c))
            faces.append((a, c, d))
    return faces


def build_cup_mesh() -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    rings = {
        "outer_bottom": _ring(OUTER_BOTTOM_R,        0.0,        N_SEGMENTS),
        "outer_top":    _ring(OUTER_TOP_R,           HEIGHT,     N_SEGMENTS),
        "inner_top":    _ring(INNER_TOP_R,           HEIGHT,     N_SEGMENTS),
        "inner_floor":  _ring(INNER_CAVITY_BOTTOM_R, FLOOR_THICK, N_SEGMENTS),
    }
    centers = np.array(
        [[0.0, 0.0, 0.0],            # bottom_center (idx after rings)
         [0.0, 0.0, FLOOR_THICK]],   # floor_center
        dtype=np.float64,
    )

    verts = np.concatenate(
        [rings["outer_bottom"], rings["outer_top"], rings["inner_top"], rings["inner_floor"], centers],
        axis=0,
    )
    o_bot   = 0
    o_top   = N_SEGMENTS
    i_top   = 2 * N_SEGMENTS
    i_floor = 3 * N_SEGMENTS
    bot_c   = 4 * N_SEGMENTS
    flo_c   = 4 * N_SEGMENTS + 1

    faces: list[tuple[int, int, int]] = []

    # Outer side (outer_bottom → outer_top)
    faces += _quad_strip_faces(top_offset=o_top, bot_offset=o_bot, n=N_SEGMENTS, flip=False)
    # Top annulus (rim): outer_top → inner_top, normals up
    faces += _quad_strip_faces(top_offset=i_top, bot_offset=o_top, n=N_SEGMENTS, flip=True)
    # Inner cavity wall: inner_top → inner_floor, normals point INTO the cavity (inward radially)
    faces += _quad_strip_faces(top_offset=i_top, bot_offset=i_floor, n=N_SEGMENTS, flip=True)
    # Cavity floor disc: triangle fan from floor_center, normals up
    for i in range(N_SEGMENTS):
        i_next = (i + 1) % N_SEGMENTS
        faces.append((flo_c, i_floor + i, i_floor + i_next))
    # Bottom disc (closed bottom of cup): triangle fan from bottom_center, normals down
    for i in range(N_SEGMENTS):
        i_next = (i + 1) % N_SEGMENTS
        faces.append((bot_c, o_bot + i_next, o_bot + i))

    return verts, faces


def write_obj(path: str, verts: np.ndarray, faces: list[tuple[int, int, int]]) -> None:
    with open(path, "w") as f:
        f.write("# stack-cup hollow frustum (auto-generated)\n")
        for x, y, z in verts:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in faces:
            # OBJ uses 1-indexed vertices
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")


def main() -> int:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    asset_dir = os.path.join(
        repo_root,
        "source", "isaaclab_tasks", "isaaclab_tasks",
        "manager_based", "manipulation", "stack_cup", "assets",
    )
    os.makedirs(asset_dir, exist_ok=True)
    usd_path = os.path.join(asset_dir, "cup.usd")

    # Generate OBJ in a temp dir
    with tempfile.TemporaryDirectory() as tmp:
        obj_path = os.path.join(tmp, "cup.obj")
        verts, faces = build_cup_mesh()
        write_obj(obj_path, verts, faces)
        print(f"[mesh] wrote {obj_path}: {len(verts)} verts, {len(faces)} tris")
        print(f"[mesh] inner_top_r={INNER_TOP_R:.4f}, inner_floor_r={INNER_CAVITY_BOTTOM_R:.4f}, "
              f"wall={WALL_THICK*1000:.1f}mm, floor={FLOOR_THICK*1000:.1f}mm")

        # Convert to USD with collision via IsaacLab's MeshConverter
        from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
        from isaaclab.sim.schemas import schemas_cfg

        # Wipe any prior conversion artifacts
        if os.path.exists(usd_path):
            os.remove(usd_path)

        mesh_cfg = MeshConverterCfg(
            mass_props=schemas_cfg.MassPropertiesCfg(mass=0.05),         # 50 g per cup
            rigid_props=schemas_cfg.RigidBodyPropertiesCfg(),
            collision_props=schemas_cfg.CollisionPropertiesCfg(),
            mesh_collision_props=schemas_cfg.ConvexDecompositionPropertiesCfg(),
            asset_path=obj_path,
            force_usd_conversion=True,
            usd_dir=asset_dir,
            usd_file_name="cup.usd",
            make_instanceable=True,
        )
        MeshConverter(mesh_cfg)
        print(f"[usd ] wrote {usd_path}")

    sim_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
