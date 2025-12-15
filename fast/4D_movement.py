#!/usr/bin/env python3
import argparse
from pathlib import Path
import imageio.v2 as imageio
import re
import numpy as np

def make_diagonal_walking_spin(
    root_dir,
    prefix="frame",
    duration=0.05,
    out_gif="walking_spin.gif",
    frames_per_subfolder=1,
    shift_per_frame=0,
):
    """
    Build a GIF using frames where folder index == spin index:
        frame_000000_000.png,
        frame_000001_001.png, ...

    frames_per_subfolder:
        How many times to repeat the matching frame from each subfolder.

    shift_per_frame:
        Horizontal pixel shift per frame in the output GIF.
        0 = no shift (walk in place).
        >0 = baby walks to the right over time.
    """

    root = Path(root_dir)
    if not root.is_dir():
        raise NotADirectoryError(root)

    # Find subfolders like frame_000123
    folder_re = re.compile(rf"^{re.escape(prefix)}_(\d+)$")

    folders = []
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        m = folder_re.match(sub.name)
        if not m:
            continue
        idx = int(m.group(1))
        folders.append((idx, sub))

    if not folders:
        print(f"No '{prefix}_NNNNNN' style folders found in {root}")
        return

    # Sort by numeric frame index
    folders.sort(key=lambda x: x[0])

    frames = []
    unique_used = 0
    skipped = 0

    for idx, sub in folders:
        stem = sub.name  # e.g. "frame_000042"
        spin_idx_str = f"{idx:03d}"            # e.g. "042"
        png_name = f"{stem}_{spin_idx_str}.png"  # frame_000042_042.png
        png_path = sub / png_name

        if png_path.is_file():
            for _ in range(frames_per_subfolder):
                frames.append(png_path)
            unique_used += 1
        else:
            skipped += 1
            print(f"Warning: missing {png_path}, skipping this frame.")

    if not frames:
        print("No matching diagonal frames (frame_xxx_xxx) found. Nothing to do.")
        return

    total_frames = len(frames)
    print(
        f"Building GIF from {total_frames} frames "
        f"({unique_used} unique subfolders used, {skipped} skipped)."
    )

    # Load first frame to get size
    first_img = imageio.imread(frames[0])
    H, W = first_img.shape[:2]

    images = []

    if shift_per_frame <= 0:
        # Original behavior: no walking, just in-place animation
        images = [imageio.imread(p) for p in frames]
    else:
        # Make canvas wide enough for the full walk
        max_shift = shift_per_frame * (total_frames - 1)
        canvas_w = W + max_shift

        for frame_idx, p in enumerate(frames):
            img = imageio.imread(p)
            # Create white/transparent background matching input dtype
            if img.ndim == 2:  # grayscale
                canvas = np.ones((H, canvas_w), dtype=img.dtype) * 255
            else:              # RGB or RGBA
                canvas = np.ones((H, canvas_w, img.shape[2]), dtype=img.dtype) * 255

            x = shift_per_frame * frame_idx
            x = int(x)
            # Paste original image into canvas at offset x
            canvas[:, x:x+W] = img
            images.append(canvas)

    out_path = root / out_gif
    imageio.mimsave(out_path, images, duration=duration)
    print(f"Saved walking-spin GIF: {out_path}")

def main():
    ap = argparse.ArgumentParser(
        description="Merge frame_xxx_xxx across all spin subfolders into a single walking-spin GIF."
    )
    ap.add_argument(
        "--root-dir",
        required=True,
        help="Root directory containing spin subfolders (e.g., 'spins').",
    )
    ap.add_argument(
        "--duration",
        type=float,
        default=0.05,
        help="Frame duration in seconds (default: 0.05 = 20 FPS).",
    )
    ap.add_argument(
        "--prefix",
        type=str,
        default="frame",
        help="Folder/file prefix (default: 'frame').",
    )
    ap.add_argument(
        "--out-gif",
        type=str,
        default="walking_spin.gif",
        help="Output GIF file name (saved inside root-dir).",
    )
    ap.add_argument(
        "--frames-per-subfolder",
        type=int,
        default=1,
        help="How many times to repeat each matching frame_xxx_xxx (default: 1).",
    )
    ap.add_argument(
        "--shift-per-frame",
        type=int,
        default=0,
        help="Horizontal pixel shift per frame (0 = in place, >0 = walk right).",
    )
    args = ap.parse_args()

    make_diagonal_walking_spin(
        root_dir=args.root_dir,
        prefix=args.prefix,
        duration=args.duration,
        out_gif=args.out_gif,
        frames_per_subfolder=args.frames_per_subfolder,
        shift_per_frame=args.shift_per_frame,
    )

if __name__ == "__main__":
    main()
