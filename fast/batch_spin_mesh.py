#!/usr/bin/env python3
import numpy as np
import trimesh
import pyrender
import imageio.v2 as imageio
from pathlib import Path
import argparse

def make_scene_for_mesh(tm, width=800, height=800):
    """
    Build a pyrender scene with a fixed camera + light
    and add the given mesh. NO renderer here.
    """
    scene = pyrender.Scene(
        bg_color=[0, 0, 0, 0],
        ambient_light=[0.2, 0.2, 0.2, 1.0],
    )

    mesh = pyrender.Mesh.from_trimesh(tm, smooth=True)
    scene.add(mesh)

    # Camera slightly above and in front
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_pose = np.array([
        [1.0, 0.0,  0.0, 0.0],
        [0.0, 0.866, 0.5, 0.7],
        [0.0,-0.5,  0.866, 1.5],
        [0.0, 0.0,  0.0, 1.0],
    ])
    scene.add(camera, pose=cam_pose)

    # Light from same direction as camera
    light = pyrender.DirectionalLight(intensity=3.0)
    scene.add(light, pose=cam_pose)

    return scene

def spin_and_render_single(mesh_path, out_dir, renderer, n_frames=60,
                           width=800, height=800, axis="y"):
    """
    Spin a single mesh and save n_frames images into out_dir.
    Axis can be 'x', 'y', or 'z'.
    Reuses the given renderer (no new GL contexts).
    """
    mesh_path = Path(mesh_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load mesh
    base_tm = trimesh.load(mesh_path, force="mesh")

    # Center + scale to unit size so rotation looks nice
    base_tm.apply_translation(-base_tm.centroid)
    scale = 1.0 / max(base_tm.extents)
    base_tm.apply_scale(scale)

    # Choose rotation axis
    if axis == "x":
        rot_axis = [1, 0, 0]
    elif axis == "z":
        rot_axis = [0, 0, 1]
    else:
        rot_axis = [0, 1, 0]  # default y-axis

    stem = mesh_path.stem  # e.g. "frame_000000"

    for i in range(n_frames):
        angle = 2.0 * np.pi * i / n_frames  # 0 -> 2π over full spin

        tm = base_tm.copy()
        R = trimesh.transformations.rotation_matrix(angle, rot_axis)
        tm.apply_transform(R)

        scene = make_scene_for_mesh(tm, width, height)
        color, depth = renderer.render(scene)

        img_name = f"{stem}_{i:03d}.png"
        img_path = out_dir / img_name
        imageio.imwrite(img_path, color)
        print(f"{mesh_path.name}: saved {img_path}")

def batch_spin(glb_dir, out_root, n_frames=60, width=800, height=800, axis="y"):
    glb_dir = Path(glb_dir)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    glb_paths = sorted(glb_dir.glob("*.glb"))
    if not glb_paths:
        raise RuntimeError(f"No .glb files found in {glb_dir}")

    print(f"Found {len(glb_paths)} GLBs in {glb_dir}")

    # Create ONE renderer and reuse it for everything
    renderer = pyrender.OffscreenRenderer(width, height)

    try:
        for idx, glb_path in enumerate(glb_paths):
            stem = glb_path.stem  # e.g. "frame_000000"
            sub_out = out_root / stem

            print(f"\n[{idx+1}/{len(glb_paths)}] Spinning {glb_path} -> {sub_out}")
            spin_and_render_single(
                mesh_path=glb_path,
                out_dir=sub_out,
                renderer=renderer,
                n_frames=n_frames,
                width=width,
                height=height,
                axis=axis,
            )
    finally:
        # Ensure context is released
        renderer.delete()
        print("Renderer deleted, GL context cleaned up.")

def main():
    ap = argparse.ArgumentParser(
        description="For each GLB in a folder, create a subfolder and render a spin with the same number of frames."
    )
    ap.add_argument(
        "--glb-dir",
        required=True,
        help="Directory containing .glb files (e.g., from video_to_glbs.py)",
    )
    ap.add_argument(
        "--out-dir",
        default="spins",
        help="Output root directory where subfolders will be created",
    )
    ap.add_argument(
        "--frames",
        type=int,
        default=60,
        help="Number of spin frames per GLB (same for all)",
    )
    ap.add_argument(
        "--width",
        type=int,
        default=800,
        help="Render width",
    )
    ap.add_argument(
        "--height",
        type=int,
        default=800,
        help="Render height",
    )
    ap.add_argument(
        "--axis",
        type=str,
        default="y",
        choices=["x", "y", "z"],
        help="Axis to spin around (default: y)",
    )
    args = ap.parse_args()

    batch_spin(
        glb_dir=args.glb_dir,
        out_root=args.out_dir,
        n_frames=args.frames,
        width=args.width,
        height=args.height,
        axis=args.axis,
    )

if __name__ == "__main__":
    main()
