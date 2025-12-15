#!/usr/bin/env python3
import cv2
import os
import argparse
from pathlib import Path

def extract_frames(video_path, output_dir, prefix="frame"):
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Try to read number of frames from metadata (may be approximate)
    total_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # no more frames

        # Format: frame_000000.png, frame_000001.png, ...
        filename = f"{prefix}_{frame_idx:06d}.png"
        out_path = output_dir / filename

        # Write frame as PNG
        cv2.imwrite(str(out_path), frame)

        frame_idx += 1

        if frame_idx % 50 == 0:
            print(f"Saved {frame_idx} frames...", end="\r")

    cap.release()

    print()  # newline after progress
    print(f"Done. Extracted {frame_idx} frames.")

    # Show metadata vs actual, just for info
    if total_frames_meta > 0:
        print(f"Metadata frame count: {total_frames_meta}")
    print(f"Actual frames written: {frame_idx}")

    return frame_idx

def main():
    parser = argparse.ArgumentParser(
        description="Extract all frames from an MP4 video."
    )
    parser.add_argument(
        "--video",
        "-v",
        required=True,
        help="Path to input video file (e.g., input.mp4)",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        default="frames",
        help="Directory to save extracted frames (default: ./frames)",
    )
    parser.add_argument(
        "--prefix",
        "-p",
        default="frame",
        help="Prefix for frame filenames (default: 'frame')",
    )

    args = parser.parse_args()

    n = extract_frames(args.video, args.out_dir, prefix=args.prefix)
    print(f"Total number of frames: {n}")

if __name__ == "__main__":
    main()
