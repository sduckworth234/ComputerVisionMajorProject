import os
import cv2 as cv
import argparse
from pathlib import Path
import json
import sys
import random

# Utility: ensure_dir
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# =========================================================================
# Simple frame sampler (every Nth frame, no scoring, no cuts)
# =========================================================================

def sample_every_nth_frame(video_path: str, out_dir: str, stride: int = 100) -> int:
    """Save every Nth frame from a single video into out_dir as PNG.

    Returns the number of frames saved.
    """
    ensure_dir(out_dir)
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[sample] Could not open: {video_path}")
        return 0
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            out_path = os.path.join(out_dir, f"frame_{idx:06d}.png")
            cv.imwrite(out_path, frame)
            saved += 1
        idx += 1
    cap.release()
    print(f"[sample] {Path(video_path).name}: frames={total} stride={stride} saved={saved} -> {out_dir}")
    return saved

def sample_random_frames(video_path: str, out_dir: str, k: int = 200, seed: int | None = None) -> int:
    """Randomly sample K unique frames from a single video and save as PNGs.

    Returns the number of frames saved.
    """
    ensure_dir(out_dir)
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[sample] Could not open: {video_path}")
        return 0
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        print(f"[sample] No frames reported for {video_path}")
        cap.release()
        return 0

    kk = min(int(k), total)
    rng = random.Random(seed) if seed is not None else random
    indices = sorted(rng.sample(range(total), kk))

    saved = 0
    for idx in indices:
        cap.set(cv.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        out_path = os.path.join(out_dir, f"frame_{idx:06d}.png")
        cv.imwrite(out_path, frame)
        saved += 1

    # Write the list of actual saved indices for traceability
    try:
        with open(os.path.join(out_dir, "sampled_indices.txt"), "w") as f:
            for idx in indices:
                f.write(f"{idx}\n")
    except Exception:
        pass

    cap.release()
    print(f"[sample] {Path(video_path).name}: total={total} random_k={kk} saved={saved} -> {out_dir}")
    return saved


def sample_videos_in_dir(input_dir: str, out_root: str | None = None, stride: int = 100, pattern: str = "*.mp4", random_k: int | None = None, seed: int | None = None) -> None:
    """Scan input_dir for videos (pattern) and sample each into per-video subfolders.

    Subfolder name matches video stem (without extension).
    """
    in_p = Path(input_dir)
    if not in_p.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    vids = sorted(in_p.glob(pattern))
    if not vids:
        print(f"[sample] No videos found in {input_dir} matching {pattern}")
        return
    out_root_p = Path(out_root) if out_root else (in_p / "SAMPLES")
    out_root_p.mkdir(parents=True, exist_ok=True)

    grand_total = 0
    for v in vids:
        sub_out = out_root_p / v.stem
        if random_k is not None:
            saved = sample_random_frames(str(v), str(sub_out), k=int(random_k), seed=seed)
        else:
            saved = sample_every_nth_frame(str(v), str(sub_out), stride=stride)
        grand_total += saved
    print(f"[sample] Done. Videos processed: {len(vids)}. Total frames saved: {grand_total}. Root: {out_root_p}")

def build_argparser():
    p = argparse.ArgumentParser(description="Frame tools: simple sampling (every Nth frame)")
    # Simple sampling mode (every Nth frame)
    p.add_argument("--sample-dir", type=str, default=None, help="Directory containing videos to sample (every Nth frame)")
    p.add_argument("--sample-out", type=str, default=None, help="Output root for sampled frames (defaults to <sample-dir>/SAMPLES)")
    p.add_argument("--sample-stride", type=int, default=100, help="Save every Nth frame (default 100)")
    p.add_argument("--sample-pattern", type=str, default="*.mp4", help="Glob pattern for videos (default *.mp4)")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    # Optional non-sampling placeholders (not required)
    p.add_argument("--input", type=str, default=None, help="Input video (non-sampling mode)")
    p.add_argument("--output", type=str, default=None, help="Output directory (non-sampling mode)")
    p.add_argument("--random-k", type=int, default=None, help="If set, randomly sample K frames per video instead of fixed stride")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (optional)")
    return p

def main():
    args = build_argparser().parse_args()

    # Sampling-only mode: extract every Nth frame from all videos in a directory
    if args.sample_dir:
        if args.verbose:
            print(f"[sample] Scanning: {args.sample_dir}  pattern={args.sample_pattern}  stride={args.sample_stride}")
        sample_videos_in_dir(
            input_dir=args.sample_dir,
            out_root=args.sample_out,
            stride=args.sample_stride,
            pattern=args.sample_pattern,
            random_k=args.random_k,
            seed=args.seed,
        )
        return

    # Non-sampling path (only if both provided)
    if args.input and args.output:
        # Lazy import to avoid NameError if extraction code is not in this file
        try:
            from BALL_TRACKING.vid2png import extract_and_filter_frames  # type: ignore
        except Exception:
            print("[error] Non-sampling mode not available in this build. Use --sample-dir.")
            sys.exit(2)
        extract_and_filter_frames(args.input, args.output)
        return

    # Nothing to do
    print("Usage: --sample-dir <dir> [--sample-out <dir>] [--sample-stride 100] [--sample-pattern '*.mp4']")
    sys.exit(1)


# Ensure module executes
if __name__ == "__main__":
    main()
