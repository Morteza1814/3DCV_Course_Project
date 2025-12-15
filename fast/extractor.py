#!/usr/bin/env python3
import argparse
from pathlib import Path

from gradio_client import Client, handle_file
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Call Hunyuan3D-2 Hugging Face Space to turn an image into a 3D mesh."
    )
    parser.add_argument("--image", required=True, help="Path to input image (PNG/JPG).")
    parser.add_argument(
        "--space",
        default="tencent/Hunyuan3D-2",
        help="Hugging Face Space ID (default: tencent/Hunyuan3D-2)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of shape diffusion steps (lower = faster, default: 30).",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=5.0,
        help="Guidance scale (default: 5.0).",
    )
    parser.add_argument(
        "--octree_resolution",
        type=int,
        default=256,
        help="Octree resolution for decoding (lower = faster / less detail, default: 256).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed (ignored when --randomize-seed is set).",
    )
    parser.add_argument(
        "--no-randomize-seed",
        action="store_true",
        help="Disable random seed; use fixed --seed instead.",
    )
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.is_file():
        raise SystemExit(f"Input image not found: {img_path}")

    # Create the Space client
    client = Client(args.space, download_files="outputs/")

    # IMPORTANT: find the correct API endpoint name once
    # This prints all callable endpoints with their arguments; do this in a REPL if needed.
    # print(client.view_api())

    # The Gradio app wires this function to the “Gen Shape” button with inputs:
    # [caption, image, mv_image_front, mv_image_back, mv_image_left, mv_image_right,
    #  num_steps, cfg_scale, seed, octree_resolution, check_box_rembg, num_chunks, randomize_seed]
    # and output: [file_out, html_gen_mesh, stats, seed]. :contentReference[oaicite:1]{index=1}

    randomize_seed = not args.no_randomize_seed

    # Call the “shape_generation” endpoint: image → bare (white) mesh
    result = client.predict(
        # caption
        None,
        # image (single-view)
        handle_file(str(img_path)),
        # mv_image_front, mv_image_back, mv_image_left, mv_image_right
        None,
        None,
        None,
        None,
        # num_steps
        args.steps,
        # cfg_scale (guidance_scale)
        args.guidance,
        # seed
        args.seed,
        # octree_resolution
        args.octree_resolution,
        # check_box_rembg (remove background)
        True,
        # num_chunks (controls memory/speed tradeoff in decoding)
        8000,
        # randomize_seed
        randomize_seed,
        api_name="/shape_generation",  # if this ever breaks, run client.view_api() to see the current name

    )

    out0 = result[0]
    print("Raw file_out:", out0)  # optional debug

    if isinstance(out0, dict):
        # Try all known patterns: 'path', 'name', then 'value' (what you're seeing)
        mesh_path = out0.get("path") or out0.get("name") or out0.get("value")
    else:
        mesh_path = out0  # old behavior: directly a string

    if not mesh_path:
        raise RuntimeError(f"Could not find mesh path in result: {out0}")

    mesh_file = Path(mesh_path)
    print("Saved mesh at:", mesh_file.resolve())

    print("Mesh file:", mesh_file)  # first output is the generated mesh file (.glb by default) :contentReference[oaicite:2]{index=2}

if __name__ == "__main__":
    main()
