"""
Extract frames from the remaining DJI videos at a fixed time interval.

Time-based sampling (1 frame every N seconds) instead of every-Nth-video-frame
avoids the near-duplicate flood: camera motion between 2-second samples is
large, so frames are genuinely different.

Only Wide-lens (_W) variants are taken from multi-lens flights — they match
the horizontal Brio rig's field of view; tele/zoom variants are redundant.

Usage:
    python extract_frames.py \
        --videos "/run/media/dvaghani/Expansion/Yolo/02_Video" \
        --dst    "/run/media/dvaghani/Expansion/Yolo/frames_round2" \
        --every 2.0 \
        --skip DJI_0006 DJI_0007 DJI_0008 DJI_0010 DJI_0011 DJI_0012
"""
import argparse
import subprocess
from pathlib import Path


def video_list(root: Path, skip: set):
    vids = []
    # top-level videos (all lenses — these are single-lens captures)
    for v in sorted(root.glob("*.MP4")):
        if v.stem in skip:
            continue
        vids.append(v)
    # multi-lens flights: Wide only
    for v in sorted(root.rglob("*_W.MP4")):
        vids.append(v)
    return vids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", required=True)
    p.add_argument("--dst",    required=True)
    p.add_argument("--every",  type=float, default=2.0,
                   help="Seconds between extracted frames.")
    p.add_argument("--skip",   nargs="*", default=[])
    args = p.parse_args()

    dst = Path(args.dst); dst.mkdir(parents=True, exist_ok=True)
    fps = 1.0 / args.every
    vids = video_list(Path(args.videos), set(args.skip))
    print(f"{len(vids)} videos, 1 frame / {args.every}s\n")

    total = 0
    for v in vids:
        # unique short tag: parent flight id (if any) + stem
        tag = v.stem
        out_pat = str(dst / f"{tag}_%04d.jpg")
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
               "-i", str(v), "-vf", f"fps={fps}", "-q:v", "3", out_pat]
        subprocess.run(cmd, check=True)
        n = len(list(dst.glob(f"{tag}_*.jpg")))
        print(f"  {v.name:45s} -> {n} frames")
        total += n

    print(f"\nTotal extracted: {total} frames in {dst}")


if __name__ == "__main__":
    main()
