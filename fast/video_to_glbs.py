#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2
from gradio_client import Client, handle_file
import shutil
import os

def extract_and_convert(
    video_path,
    space_id="tencent/Hunyuan3D-2",
    frames_dir="frames",
    glb_dir="glbs",
    steps=30,
    guidance=5.0,
    octree_resolution=256,
    seed=1234,
    randomize_seed=True,
    max_frames=None,
):
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    frames_dir = Path(frames_dir)
    glb_dir = Path(glb_dir)
    tmp_download_dir = glb_dir / "_hf_tmp"  # where HF will drop tokenID folders

    frames_dir.mkdir(parents=True, exist_ok=True)
    glb_dir.mkdir(parents=True, exist_ok=True)
    tmp_download_dir.mkdir(parents=True, exist_ok=True)

    # 1) Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video metadata says ~{total_meta} frames")

    # 2) Create HF Space client (once)
    client = Client(space_id, download_files=str(tmp_download_dir))
    print(f"Loaded Hunyuan3D-2 Space as API: {space_id}")

    frame_idx = 0
    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break  # end of video

        frame_name = f"frame_{frame_idx:06d}.png"
        frame_path = frames_dir / frame_name

        # Save frame as PNG
        cv2.imwrite(str(frame_path), frame)

        # 3) Call Space: frame -> GLB
        print(f"\n[Frame {frame_idx}] Sending {frame_path} to Hunyuan3D-2...")
        result = client.predict(
            # caption
            None,
            # image (single-view)
            handle_file(str(frame_path)),
            # mv_image_front, mv_image_back, mv_image_left, mv_image_right
            None,
            None,
            None,
            None,
            # num_steps
            steps,
            # cfg_scale (guidance_scale)
            guidance,
            # seed
            seed,
            # octree_resolution
            octree_resolution,
            # check_box_rembg (remove background)
            True,
            # num_chunks (controls memory/speed tradeoff in decoding)
            8000,
            # randomize_seed
            randomize_seed,
            api_name="/shape_generation",
        )

        out0 = result[0]
        # out0 is usually a dict like {'value': '/path/to/tokenID/white_mesh.glb', '__type__': 'update'}
        if isinstance(out0, dict):
            mesh_path = out0.get("path") or out0.get("name") or out0.get("value")
        else:
            mesh_path = out0

        if not mesh_path:
            raise RuntimeError(f"Could not find mesh path in result: {out0}")

        mesh_file = Path(mesh_path)
        print(f"[Frame {frame_idx}] Raw mesh file from HF: {mesh_file}")

        # 4) Rename/move GLB to ordered name: glbs/frame_000000.glb, etc.
        target_glb = glb_dir / f"frame_{frame_idx:06d}.glb"
        shutil.move(str(mesh_file), str(target_glb))
        print(f"[Frame {frame_idx}] Saved GLB as: {target_glb}")

        # Optional: clean up empty tokenID folder if possible
        try:
            token_dir = mesh_file.parent
            if token_dir.exists() and token_dir != tmp_download_dir:
                # remove dir if empty
                if not any(token_dir.iterdir()):
                    token_dir.rmdir()
        except Exception:
            pass

        frame_idx += 1

    cap.release()
    print(f"\nDone. Processed {frame_idx} frames.")
    print(f"Frames saved in: {frames_dir.resolve()}")
    print(f"GLBs saved in:   {glb_dir.resolve()}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from an MP4 and convert each frame to a GLB via Hunyuan3D-2."
    )
    parser.add_argument("--video", "-v", required=True, help="Path to input video (e.g., input.mp4)")
    parser.add_argument("--space", default="tencent/Hunyuan3D-2", help="Hugging Face Space ID")
    parser.add_argument("--frames-dir", default="frames", help="Directory to save extracted frames")
    parser.add_argument("--glb-dir", default="glbs", help="Directory to save ordered GLBs")
    parser.add_argument("--steps", type=int, default=30, help="Shape diffusion steps (default: 30)")
    parser.add_argument("--guidance", type=float, default=5.0, help="Guidance scale (default: 5.0)")
    parser.add_argument("--octree-resolution", type=int, default=256,
                        help="Octree resolution (default: 256)")
    parser.add_argument("--seed", type=int, default=1234, help="Base seed")
    parser.add_argument("--no-randomize-seed", action="store_true",
                        help="Disable random seed; use fixed --seed instead.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Optional: limit number of frames to process (for testing).")

    args = parser.parse_args()

    randomize_seed = not args.no_randomize_seed

    extract_and_convert(
        video_path=args.video,
        space_id=args.space,
        frames_dir=args.frames_dir,
        glb_dir=args.glb_dir,
        steps=args.steps,
        guidance=args.guidance,
        octree_resolution=args.octree_resolution,
        seed=args.seed,
        randomize_seed=randomize_seed,
        max_frames=args.max_frames,
    )

if __name__ == "__main__":
    main()
