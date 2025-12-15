#!/usr/bin/env python3
import numpy as np
import trimesh
import pyrender
import imageio.v2 as imageio
from pathlib import Path
import argparse

def make_scene_for_mesh(tm, width=800, height=800):
    # Build a pyrender scene with camera + light
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
        [0.0,-0.5,  0.866, 1.0],
        [0.0, 0.0,  0.0, 1.0],
    ])
    scene.add(camera, pose=cam_pose)

    # Light from same direction as camera
    light = pyrender.DirectionalLight(intensity=3.0)
    scene.add(light, pose=cam_pose)

    renderer = pyrender.OffscreenRenderer(width, height)
    return scene, renderer

def spin_and_render(
    mesh_path,
    out_dir="spin_frames",
    n_frames=60,
    width=800,
    height=800,
    make_gif=True,
    frames_to_save=None,
):
    """
    Spin the mesh for n_frames around the y-axis.
    If frames_to_save is not None, it should be an iterable of indices in [0, n_frames-1]
    and only those frames will be rendered/saved.
    """
    mesh_path = Path(mesh_path)
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Normalize frames_to_save
    if frames_to_save is not None:
        frames_to_save = sorted(set(int(i) for i in frames_to_save))
        # Basic sanity check
        bad = [i for i in frames_to_save if i < 0 or i >= n_frames]
        if bad:
            raise ValueError(
                f"frames_to_save indices out of range [0, {n_frames-1}]: {bad}"
            )
        print(f"Will save only frames: {frames_to_save}")
    else:
        print(f"Will save all {n_frames} frames.")

    # Load mesh
    base_tm = trimesh.load(mesh_path, force="mesh")

    # Center + scale to unit size so rotation looks nice
    base_tm.apply_translation(-base_tm.centroid)
    scale = 1.0 / max(base_tm.extents)
    base_tm.apply_scale(scale)

    # Prepare renderer (we create a new scene per frame but reuse renderer)
    _, renderer = make_scene_for_mesh(base_tm, width, height)

    frame_paths = []

    for i in range(n_frames):
        # If user provided specific frames, skip others
        if frames_to_save is not None and i not in frames_to_save:
            continue

        angle = 2.0 * np.pi * i / n_frames  # 0 -> 2π over full spin

        # Copy base mesh and apply rotation around y-axis
        tm = base_tm.copy()
        Ry = trimesh.transformations.rotation_matrix(
            angle, [0, 1, 0]  # axis = y
        )
        tm.apply_transform(Ry)

        # New scene per frame (simpler than trying to reuse nodes)
        scene, _ = make_scene_for_mesh(tm, width, height)

        color, depth = renderer.render(scene)
        frame_path = out_dir / f"frame_{i:03d}.png"
        imageio.imwrite(frame_path, color)
        frame_paths.append(frame_path)
        print(f"Saved frame {i+1}/{n_frames}: {frame_path}")

    renderer.delete()

    if make_gif and frame_paths:
        gif_path = out_dir / "spin.gif"
        images = [imageio.imread(p) for p in frame_paths]
        imageio.mimsave(gif_path, images, duration=0.05)  # 20 fps
        print(f"Saved GIF: {gif_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, help="Path to .glb/.obj mesh")
    ap.add_argument(
        "--out-dir",
        default="spin_frames",
        help="Output folder for frames/GIF",
    )
    ap.add_argument(
        "--frames",
        type=int,
        default=60,
        help="Number of frames in one full turn",
    )
    ap.add_argument(
        "--save-frames",
        type=str,
        default=None,
        help=(
            "Comma-separated list of frame indices to save, "
            "e.g. '0,15,30'. Indices are in [0, frames-1]. "
            "If omitted, all frames are saved."
        ),
    )
    args = ap.parse_args()

    if args.save_frames is not None:
        frames_to_save = [
            int(x) for x in args.save_frames.split(",") if x.strip() != ""
        ]
    else:
        frames_to_save = None

    spin_and_render(
        mesh_path=args.mesh,
        out_dir=args.out_dir,
        n_frames=args.frames,
        frames_to_save=frames_to_save,
    )

if __name__ == "__main__":
    main()
