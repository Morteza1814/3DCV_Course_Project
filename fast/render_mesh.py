#!/usr/bin/env python3
import numpy as np
import trimesh
import pyrender
import imageio.v2 as imageio
from pathlib import Path
import argparse

def render_mesh(mesh_path: str, out_path: str = "render.png",
                width: int = 800, height: int = 800):
    mesh_path = Path(mesh_path)
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)

    # Load mesh (GLB/OBJ/etc.)
    tm = trimesh.load(mesh_path, force='mesh')

    # Center + normalize a bit so it fits nicely in view
    tm.apply_translation(-tm.centroid)
    scale = 1.0 / max(tm.extents)
    tm.apply_scale(scale)

    # Wrap into pyrender mesh + scene
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.2, 0.2, 0.2, 1.0])
    mesh = pyrender.Mesh.from_trimesh(tm, smooth=True)
    scene.add(mesh)

    # Camera (slightly above and in front)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_pose = np.array([
        [1.0, 0.0,  0.0, 0.0],
        [0.0, 0.866, 0.5, 0.7],
        [0.0,-0.5,  0.866, 1.0],
        [0.0, 0.0,  0.0, 1.0],
    ])
    scene.add(camera, pose=cam_pose)

    # Simple light
    light = pyrender.DirectionalLight(intensity=3.0)
    scene.add(light, pose=cam_pose)

    # Offscreen render
    r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    color, depth = r.render(scene)
    r.delete()

    imageio.imwrite(out_path, color)
    print(f"Saved render to {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, help="Path to .glb/.obj mesh")
    ap.add_argument("--out", default="render.png", help="Output PNG file")
    args = ap.parse_args()
    render_mesh(args.mesh, args.out)

if __name__ == "__main__":
    main()
