#!/usr/bin/env python3
# coding: utf-8

"""
Audio + face photo -> natural talking-head video.

Pipeline:
  1. SadTalker:    audio -> driving video (realistic speech motion)
  2. LivePortrait: driving video + your face -> final video

Usage:
    python run.py --audio speech.wav
    python run.py --audio speech.wav --image /path/to/face.jpg
    python run.py --audio speech.wav --image face.jpg --output output/result.mp4
"""

import argparse
import os
import os.path as osp
import subprocess
import sys
import tempfile

DEFAULT_FACE      = osp.join(osp.dirname(osp.realpath(__file__)), "assets", "my_face.jpg")
SADTALKER_DIR     = os.environ.get("SADTALKER_DIR", "/Users/narayanyerrabachu/facefusion-pipeline/SadTalker")
LIVEPORTRAIT_DIR  = os.environ.get("LIVEPORTRAIT_DIR", "/Users/narayanyerrabachu/git/LivePortrait")


def parse_args():
    p = argparse.ArgumentParser(description="Audio-driven portrait animation")
    p.add_argument("--image", "-i", default=DEFAULT_FACE, help="Portrait image path")
    p.add_argument("--audio", "-a", default=None, help="Audio file (wav/mp3/m4a)")
    p.add_argument("--output", "-o", default="output/result.mp4", help="Output video path")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--skip-sadtalker", action="store_true",
                   help="Skip SadTalker, provide --driving directly")
    p.add_argument("--driving", "-d", default=None,
                   help="Driving video (skip SadTalker if provided)")
    p.add_argument("--with-liveportrait", action="store_true",
                   help="Add LivePortrait stage after SadTalker (experimental)")
    return p.parse_args()


def run_sadtalker(image_path: str, audio_path: str, out_dir: str) -> str:
    """Run SadTalker to generate a driving video from audio."""
    print("\n[Stage 1/2] SadTalker: audio → driving video...")
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        sys.executable,
        osp.join(SADTALKER_DIR, "inference.py"),
        "--driven_audio", osp.abspath(audio_path),
        "--source_image", osp.abspath(image_path),
        "--result_dir", osp.abspath(out_dir),
        "--expression_scale", "1.5",
        "--preprocess", "crop",
    ]
    result = subprocess.run(cmd, cwd=SADTALKER_DIR, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError("SadTalker failed. Check output above.")

    # SadTalker saves into a timestamped subfolder
    for root, dirs, files in os.walk(out_dir):
        for f in files:
            if f.endswith(".mp4"):
                return osp.join(root, f)
    raise RuntimeError(f"SadTalker produced no .mp4 under {out_dir}")


def run_liveportrait(source_image: str, driving_video: str, output_path: str, fps: float):
    """Run LivePortrait to transfer driving motion onto source face."""
    print("\n[Stage 2/2] LivePortrait: motion transfer → final video...")
    os.makedirs(osp.dirname(osp.abspath(output_path)), exist_ok=True)

    out_dir = osp.dirname(osp.abspath(output_path))

    cmd = [
        sys.executable,
        osp.join(LIVEPORTRAIT_DIR, "inference.py"),
        "--source", osp.abspath(source_image),
        "--driving", osp.abspath(driving_video),
        "--output_dir", out_dir,
        "--flag_pasteback",
        "--flag_stitching",
        "--animation_region", "all",
        "--driving_option", "expression-friendly",
    ]
    result = subprocess.run(cmd, cwd=LIVEPORTRAIT_DIR, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError("LivePortrait failed. Check output above.")

    # find and rename output
    src_base = osp.splitext(osp.basename(source_image))[0]
    drv_base = osp.splitext(osp.basename(driving_video))[0]
    expected = osp.join(out_dir, f"{src_base}--{drv_base}.mp4")
    if osp.exists(expected):
        os.rename(expected, output_path)
        print(f"\nDone → {output_path}")
    else:
        # find any new mp4
        for f in sorted(os.listdir(out_dir)):
            if f.endswith(".mp4") and f != osp.basename(output_path):
                os.rename(osp.join(out_dir, f), output_path)
                print(f"\nDone → {output_path}")
                return
        print(f"\nOutput saved in: {out_dir}")


def main():
    args = parse_args()

    if not osp.exists(args.image):
        raise FileNotFoundError(
            f"Portrait not found: {args.image}\n"
            f"Put your face photo at: {DEFAULT_FACE}"
        )
    if args.audio and not osp.exists(args.audio):
        raise FileNotFoundError(f"Audio not found: {args.audio}")
    if not args.driving and not args.audio:
        raise ValueError("Either --audio or --driving must be provided")

    # Stage 1: SadTalker
    if args.driving:
        driving_video = args.driving
        print(f"[Stage 1/2] Skipped — using provided driving video: {driving_video}")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            driving_video = run_sadtalker(args.image, args.audio, tmpdir)
            if args.with_liveportrait:
                run_liveportrait(args.image, driving_video, args.output, args.fps)
            else:
                import shutil
                os.makedirs(osp.dirname(osp.abspath(args.output)), exist_ok=True)
                shutil.copy(driving_video, args.output)
                print(f"\nDone → {args.output}")
        return

    # Stage 2 only (if --driving was provided)
    if args.with_liveportrait:
        run_liveportrait(args.image, driving_video, args.output, args.fps)
    else:
        import shutil
        os.makedirs(osp.dirname(osp.abspath(args.output)), exist_ok=True)
        shutil.copy(driving_video, args.output)
        print(f"\nDone → {args.output}")


if __name__ == "__main__":
    main()
