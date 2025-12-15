#!/usr/bin/env python3
import argparse
from pathlib import Path
import imageio.v2 as imageio
import re

def make_diagonal_walking_spin(
    root_dir,
    prefix="frame",
    duration=0.05,
    out_gif="walking_spin.gif",
    frames_per_subfolder=1,
):
    """
    root_dir: folder containing subfolders like:
        root_dir/
          frame_000000/
            frame_000000_000.png
            ...
          frame_000001/
            frame_000001_001.png
            ...
    We build a single GIF using frames where the numeric indices match:
        frame_000000_000.png,
        frame_000001_001.png,
        frame_000002_002.png, ...

    Only subfolders starting with `prefix` and having matching PNGs are used.

    frames_per_subfolder:
        How many times to repeat the matching frame from each subfolder
        in the final GIF (default: 1).
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
        spin_idx_str = f"{idx:03d}"          # e.g. "042"
        png_name = f"{stem}_{spin_idx_str}.png"  # frame_000042_042.png
        png_path = sub / png_name

        if png_path.is_file():
            # Repeat the same frame frames_per_subfolder times
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

    images = [imageio.imread(p) for p in frames]
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
    args = ap.parse_args()

    make_diagonal_walking_spin(
        root_dir=args.root_dir,
        prefix=args.prefix,
        duration=args.duration,
        out_gif=args.out_gif,
        frames_per_subfolder=args.frames_per_subfolder,
    )

if __name__ == "__main__":
    main()
