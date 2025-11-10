"""
Court validity classifier (classical CV features + small ML classifier)

Dataset layout (hand-labelled):
  /Users/owendyne/Documents/GitHub/ComputerVisionMajorProject/data/DATA/LABLED_FRAMES/
      AO_set1/VALID/*.png
      AO_set1/INVALID/*.png
      USO_set2/VALID/*.png
      USO_set2/INVALID/*.png
      ... (8 folders total; each has VALID and INVALID)

Goal: learn to classify frames as VALID (all 4 court corners visible) vs INVALID
(crowd/close-ups/other cameras), invariant to surface colour (blue/green/clay/grass).

We exploit geometry/appearance that is stable across tournaments:
  - Strong, long, high-contrast *white* court lines in multiple orientations
  - Presence of a near-horizontal *net band* roughly across the middle
  - Court layout imposes characteristic *orientation diversity* and line-length stats
  - Texture is *smoother* than crowds; edge density patterns differ

We extract only classical CV features (OpenCV/NumPy). For this small, fixed dataset we use a tiny
custom KNN (no sklearn) for evaluation/prediction. No deep nets.

CLI examples:
  # Default: run with built-in paths & settings (no args)
  python BALL_TRACKING/court_detector.py

  # Or override pieces when needed
  python BALL_TRACKING/court_detector.py \
      --data-root \
      "/Users/owendyne/Documents/GitHub/ComputerVisionMajorProject/data/DATA/LABLED_FRAMES" \
      --model-out \
      "/Users/owendyne/Documents/GitHub/ComputerVisionMajorProject/data/models/court_validity_knn.npz" \
      --test-split 0.2 --k 7 --weights distance

  # Predict on a directory of frames and write CSV
  python BALL_TRACKING/court_detector.py \
      --predict \
      --frames-dir \
      "/Users/owendyne/Documents/GitHub/ComputerVisionMajorProject/data/owen_samples_random200/SomeMatch" \
      --model-out \
      "/Users/owendyne/Documents/GitHub/ComputerVisionMajorProject/data/models/court_validity_knn.npz" \
      --pred-csv \
      "/Users/owendyne/Documents/GitHub/ComputerVisionMajorProject/data/owen_samples_random200/SomeMatch/preds.csv"
"""

from __future__ import annotations

import sys
import math
import csv
import random
from pathlib import Path
from typing import List, Tuple, Dict
from typing import Sequence

import numpy as np
import cv2
import pickle

# --------------------
# Thread limiting and import guards for reproducibility and avoiding OpenCV stalls
import os
import time
try:
    cv2.setNumThreads(1)
except Exception:
    pass
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# --------------------
# GLOBAL HOLD-OUT SPLIT
# Set TEST_SPLIT to a value in (0,1] to use a single train/test split instead of K-fold cv2.
# Set to 0.0 to disable hold-out and use K-fold cv2.
TEST_SPLIT: float = 0.01

# Use only a random fraction of the dataset for feature extraction + training/testing
DATA_FRACTION: float = 1  # e.g., 0.1 to use 10% of available (balanced) data

IMAGE_SIZE: int = 1080
# --------------------
# Built-in defaults so the script can run with no args
DATA_ROOT: str = "/Users/owendyne/Documents/GitHub/ComputerVisionMajorProject/data/DATA/LABLED_FRAMES"
MODEL_OUT: str = "/Users/owendyne/Documents/GitHub/ComputerVisionMajorProject/data/models/court_validity_knn.npz"
K_VALUE: int = 7
WEIGHTS: str = "distance"  # or "uniform"
# Default pickle model path (same folder as this file)
# Default pickle model path (same folder as this file)
MODEL_PKL: str = str(Path(__file__).with_name("court_validity_knn.pkl"))
# Debug: save intermediate visualization images for one random VALID frame per group
DEBUG_STEPS: bool = False
# --------------------------------------------------------------------------------------
# Feature extraction utilities
# --------------------------------------------------------------------------------------

# --- Global tuning constants (court line detection etc) ---
WHITE_S_MAX: int = 85
WHITE_V_MIN: int = 170
TOPHAT_PCT: float = 80.0
EDGE_PCT: float = 60.0
DILATE_ITERS: int = 3

# --- Debug/visualization helpers ---
def _to_u8(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype=np.uint8)
    y = (x - mn) / (mx - mn)
    y = (y * 255.0).clip(0, 255)
    return y.astype(np.uint8)

def _compute_intermediate_images(img_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    """Return key intermediate images for debugging/visualization."""
    img = img_bgr  # native 1920x1080 (no resizing)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    # White mask
    white = _white_line_mask(gray, hsv, mag)

    # Sobel Y magnitude visualization
    sobel_y = gy
    abs_sy = np.abs(sobel_y)
    sobel_y_u8 = _to_u8(abs_sy)

    # LSD lines overlay
    lines = _lsd_lines(gray, min_len_frac=0.05)
    lsd_vis = img.copy()
    for (x1, y1, x2, y2) in lines:
        cv2.line(lsd_vis, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)

    # Middle horizontal band overlay for the "net" proxy
    band_vis = img.copy()
    mid0, mid1 = int(0.3 * h), int(0.7 * h)
    cv2.rectangle(band_vis, (0, mid0), (w - 1, mid1 - 1), (0, 255, 255), 2)

    # Compose outputs
    outs: Dict[str, np.ndarray] = {
        "orig": img,
        "gray": gray,
        "white_mask": white,
        "sobel_y": sobel_y_u8,
        "lsd_overlay": lsd_vis,
        "midband_overlay": band_vis,
    }
    return outs

def _save_intermediates_for_groups(data_root: str, out_dir: Path) -> int:
    """For each dataset subfolder (group), pick a random VALID frame and save intermediates.
    Returns the number of groups visualized.
    """
    root = Path(data_root)
    subdirs = sorted([p for p in root.iterdir() if p.is_dir()])
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random
    n_done = 0
    for s in subdirs:
        group = s.name
        vdir = s / "VALID"
        frames = sorted(vdir.glob("*.png"))
        if not frames:
            print(f"[debug] skip group '{group}': no VALID/*.png found")
            continue
        p = rng.choice(frames)
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[debug] skip '{group}': failed to read {p}")
            continue
        ims = _compute_intermediate_images(img)
        for key, im in ims.items():
            # ensure 3-channel for saving masks/gray consistently
            if im.ndim == 2:
                tosave = im
            else:
                tosave = im
            fn = out_dir / f"{group}_{key}.png"
            cv2.imwrite(str(fn), tosave)
        print(f"[debug] wrote intermediates for group '{group}' -> {out_dir}")
        n_done += 1
    return n_done

def _resize_max(img: np.ndarray, max_w: int = 720) -> np.ndarray:
    if img.shape[1] <= max_w:
        return img
    scale = max_w / img.shape[1]
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _white_line_mask(gray: np.ndarray, hsv: np.ndarray, mag: np.ndarray | None = None) -> np.ndarray:
    """
    More permissive white mask for clay courts: relaxed HSV gates, keeps green exclusion.
    Expects precomputed GRAY, HSV, and optional gradient magnitude MAG for efficiency.
    Returns a single-channel uint8 mask in {0,255}.
    """
    # Use precomputed HSV channels
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # --- Adaptive HSV thresholds (more permissive for dusty/low-contrast lines) ---
    v_p80 = np.percentile(v, 80)
    s_p30 = np.percentile(s, 30)
    v_thresh = max(WHITE_V_MIN, min(235, v_p80))
    s_thresh = min(WHITE_S_MAX, max(20, s_p30))

    val_ok = (v >= v_thresh)
    sat_ok = (s <= s_thresh)

    # Exclude *green* hue range unless very gray or extremely bright (keep clay/orange unaffected)
    green_hue = (h >= 30) & (h <= 100)
    not_green = (~green_hue) | (s <= 35) | (v >= 235)

    mask_hsv = (val_ok & sat_ok & not_green)

    # White tophat to emphasize thin bright lines on textured/clay surfaces (on precomputed gray)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)
    th_tophat = np.percentile(tophat, TOPHAT_PCT)
    mask_tophat = (tophat >= th_tophat)

    # Edge magnitude gate: use precomputed magnitude if available, else compute once here
    if mag is None:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
    th_edge = np.percentile(mag, EDGE_PCT)
    mask_edge = (mag >= th_edge)

    # Combine: HSV white AND (tophat OR edge)
    m = (mask_hsv & (mask_tophat | mask_edge)).astype(np.uint8) * 255

    # Morphological tidy-up; small dilation helps reconnect faint segments
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    if DILATE_ITERS > 0:
        m = cv2.dilate(m, k, iterations=int(DILATE_ITERS))
    return m



# --- LSD-based line segment detector ---
def _lsd_lines(gray_or_edges: np.ndarray, min_len_frac: float = 0.10) -> List[Tuple[int,int,int,int]]:
    """Detect line segments using OpenCV's LSD. Returns segments longer than a fraction of min(w,h).
    Compatible with multiple OpenCV versions (positional args; fallback to no-arg ctor).
    """
    if len(gray_or_edges.shape) == 3:
        gray = cv2.cvtColor(gray_or_edges, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_or_edges
    h, w = gray.shape[:2]
    min_len = int(min(w, h) * float(min_len_frac))

    # Create detector (some builds don't accept keyword args; some need no args)
    refine = getattr(cv2, 'LSD_REFINE_STD', 1)

    lsd = cv2.createLineSegmentDetector(refine)

    res = lsd.detect(gray)
    lines = res[0] if isinstance(res, (tuple, list)) else res
    if lines is None:
        return []

    out: List[Tuple[int,int,int,int]] = []
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        if _line_len(x1, y1, x2, y2) >= min_len:
            out.append((int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))))
    return out


def _line_len(x1, y1, x2, y2) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def _line_angle_deg(x1, y1, x2, y2) -> float:
    ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
    if ang < 0:
        ang += 180.0
    return ang



def _orientation_hist(lines: List[Tuple[int,int,int,int]], nbins: int = 12) -> Tuple[np.ndarray, float, float]:
    """Return (hist, entropy, top2_sep_deg) for line orientations in [0,180).
    hist is L1-normalized over nbins; entropy is Shannon on the normalized hist; top2_sep_deg is
    the absolute angular separation (degrees) between the two largest bins, mapped to [0,90].
    """
    if not lines:
        h = np.zeros(nbins, dtype=np.float32)
        return h, 0.0, 0.0
    angs = []
    for x1, y1, x2, y2 in lines:
        a = _line_angle_deg(x1, y1, x2, y2)
        if a >= 180.0:
            a -= 180.0
        angs.append(a)
    angs = np.asarray(angs, dtype=np.float32)
    hist, _ = np.histogram(angs, bins=nbins, range=(0.0, 180.0))
    hist = hist.astype(np.float32)
    if hist.sum() > 0:
        hist /= hist.sum()
    # Shannon entropy
    eps = 1e-8
    entropy = float(-(hist * np.log(hist + eps)).sum())
    # Top-2 separation (wrap-aware, folded to [0,90])
    if hist.size >= 2 and hist.max() > 0:
        top2 = np.argsort(hist)[-2:]
        centers = (np.arange(nbins) + 0.5) * (180.0 / nbins)
        s = abs(centers[top2[1]] - centers[top2[0]])
        if s > 90.0:
            s = 180.0 - s
        top2_sep = float(s)
    else:
        top2_sep = 0.0
    return hist, entropy, top2_sep


def extract_features(img_bgr: np.ndarray) -> Dict[str, float]:
    """Compute geometry/texture features that characterise a full-court broadcast view.

    Returns a dict of scalar features (stable across surfaces).
    """
    img = _resize_max(img_bgr, IMAGE_SIZE)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    # 1) Net proxy (horiz band energy in central 40%)
    sobel_y = gy
    abs_sy = np.abs(sobel_y)
    total_sy = float(np.sum(abs_sy)) + 1e-6
    mid0, mid1 = int(0.3*h), int(0.7*h)
    mid_sy = float(np.sum(abs_sy[mid0:mid1, :]))
    horiz_band_energy = mid_sy / total_sy

    # 2) White-line mask & fraction (colour-invariant-ish)
    white = _white_line_mask(gray, hsv, mag)
    white_frac = float(cv2.countNonZero(white)) / float(white.size)

    # 3) Line segments (LSD only — simpler & robust)
    # Downscale gray for LSD for speed
    gray_lsd = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    lines = _lsd_lines(gray_lsd, min_len_frac=0.05)
    lengths = [float(np.hypot(x2 - x1, y2 - y1)) for (x1, y1, x2, y2) in lines]
    long_thresh = 0.18 * min(w, h)
    n_long = int(sum(1 for L in lengths if L >= long_thresh))
    len_mean = float(np.mean(lengths)) if lengths else 0.0
    len_std = float(np.std(lengths)) if lengths else 0.0

    # 4) Orientation structure: entropy + coarse pooled bins (0,45,135)
    hist_full, orient_entropy, _sep = _orientation_hist(lines, nbins=12)
    hist0 = float(hist_full[0:2].sum())
    hist45 = float(hist_full[3:5].sum())
    hist135 = float(hist_full[9:11].sum())

    # 5) Texture sharpness (variance of Laplacian)
    grayf = gray.astype(np.float32) / 255.0
    lap_var = float(np.var(cv2.Laplacian(grayf, cv2.CV_32F, ksize=3)))

    feats: Dict[str, float] = {
        "orient_entropy": float(orient_entropy),
        "hist0": hist0,
        "hist45": hist45,
        "hist135": hist135,
        "white_frac": white_frac,
        "horiz_band_energy": horiz_band_energy,
        "n_long": float(n_long),
        "len_mean": len_mean,
        "len_std": len_std,
        "lap_var": lap_var,
    }
    return feats



# --------------------------------------------------------------------------------------
# Balanced hold-out split (no sklearn): even per folder and per class
# --------------------------------------------------------------------------------------

def balanced_train_test_indices(groups: Sequence[str], y: np.ndarray, test_fraction: float = 0.2) -> Tuple[List[int], List[int], dict]:
    """Return train_idx, test_idx ensuring each folder contributes the same number of
    VALID and INVALID examples, limited by the smallest folder per-class size.

    Strategy:
      1) For each group (folder), collect indices for VALID (y=1) and INVALID (y=0).
      2) Compute min_valid = min_g len(valid[g]); min_invalid = min_g len(invalid[g]).
      3) For each group, randomly sample exactly min_valid VALID and min_invalid INVALID indices.
      4) Within each group & class, send ceil(sampled * test_fraction) to test, rest to train.

    This yields an even number of frames from each folder and an even number of
    valid/invalid across folders.
    """
    from collections import defaultdict
    rng = random  # use module-level RNG (no fixed seed)

    # Build per-group, per-class index lists for the *current X/y order*
    g_valid = defaultdict(list)
    g_invalid = defaultdict(list)
    for i, (g, yi) in enumerate(zip(groups, y)):
        if yi == 1:
            g_valid[g].append(i)
        else:
            g_invalid[g].append(i)

    # Drop any groups missing a class entirely
    groups_used = [g for g in set(groups) if (len(g_valid[g]) > 0 and len(g_invalid[g]) > 0)]

    min_valid = min(len(g_valid[g]) for g in groups_used)
    min_invalid = min(len(g_invalid[g]) for g in groups_used)

    # Sample equal counts from each group for each class
    selected_by_group = {}
    for g in groups_used:
        v_idx = g_valid[g][:]
        i_idx = g_invalid[g][:]
        rng.shuffle(v_idx)
        rng.shuffle(i_idx)
        v_sel = v_idx[:min_valid]
        i_sel = i_idx[:min_invalid]
        selected_by_group[g] = {1: v_sel, 0: i_sel}

    # Within each group/class, split into test/train with same fraction
    train_idx: List[int] = []
    test_idx: List[int] = []
    for g in groups_used:
        for cls in (1, 0):
            block = selected_by_group[g][cls]
            rng.shuffle(block)
            n = len(block)
            n_test = int(round(n * float(test_fraction)))
            n_test = max(0, min(n, n_test))
            test_idx.extend(block[:n_test])
            train_idx.extend(block[n_test:])

    # Summary stats
    stats = {
        "groups_used": sorted(groups_used),
        "min_valid_per_group": int(min_valid),
        "min_invalid_per_group": int(min_invalid),
        "total_valid_selected": int(min_valid * len(groups_used)),
        "total_invalid_selected": int(min_invalid * len(groups_used)),
        "test_fraction": float(test_fraction),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
    }
    return train_idx, test_idx, stats


# --------------------------------------------------------------------------------------
# Optional: downselect to an equal number of images per folder (and per class)
# --------------------------------------------------------------------------------------
def filter_pairs_equal_per_group(pairs: Sequence[Tuple[str, int, str]], per_class: bool = True, shuffle: bool = True) -> List[Tuple[str, int, str]]:
    """
    Return a subset of (path, label, group) where every group contributes the same number
    of samples. By default, enforce the equality *per class* (VALID/INVALID) so that each
    group contributes the same number of VALID and the same number of INVALID frames.

    Strategy (per_class=True):
      - Keep only groups that have at least one VALID and one INVALID.
      - For each remaining group g, compute counts of VALID (y=1) and INVALID (y=0).
      - Let n_valid = min_g count_valid[g], n_invalid = min_g count_invalid[g].
      - From each group, randomly take exactly n_valid VALID and n_invalid INVALID items.

    If per_class=False:
      - Compute n_min = min_g total_count[g] and take exactly n_min total items per group.

    Uses the module-level RNG from `random` (no fixed seed) for selection.
    """
    from collections import defaultdict
    rng = random

    # Organize by group (and by class if requested)
    by_group = defaultdict(list)
    by_group_class = defaultdict(lambda: {0: [], 1: []})
    for pth, lbl, grp in pairs:
        by_group[grp].append((pth, lbl, grp))
        by_group_class[grp][lbl].append((pth, lbl, grp))

    groups_all = sorted(by_group.keys())
    selected: List[Tuple[str, int, str]] = []

    if per_class:
        # Only keep groups that have both classes represented
        groups_used = [g for g in groups_all if len(by_group_class[g][0]) > 0 and len(by_group_class[g][1]) > 0]
        if not groups_used:
            # Nothing to do; return original list
            return list(pairs)

        # Determine equal counts per class
        n_valid = min(len(by_group_class[g][1]) for g in groups_used)
        n_invalid = min(len(by_group_class[g][0]) for g in groups_used)

        for g in groups_used:
            vals = by_group_class[g][1][:]
            invs = by_group_class[g][0][:]
            if shuffle:
                rng.shuffle(vals)
                rng.shuffle(invs)
            selected.extend(vals[:n_valid])
            selected.extend(invs[:n_invalid])

        if shuffle:
            rng.shuffle(selected)

        # Report
        print(f"[balance] groups_used={groups_used}")
        print(f"[balance] per-class per-group: VALID={n_valid}, INVALID={n_invalid}")
        print(f"[balance] total selected = {len(selected)} (={len(groups_used)}*(valid+invalid))")
        return selected
    else:
        # Equalize only total samples per group
        groups_used = groups_all
        n_min = min(len(by_group[g]) for g in groups_used)
        for g in groups_used:
            items = by_group[g][:]
            if shuffle:
                rng.shuffle(items)
            selected.extend(items[:n_min])
        if shuffle:
            rng.shuffle(selected)
        print(f"[balance] groups_used={groups_used}")
        print(f"[balance] per-group total={n_min}, total selected={len(selected)}")
        return selected

#
# --------------------------------------------------------------------------------------
# Dataset loader and training
# --------------------------------------------------------------------------------------


# --- Fast, safer image reader to avoid cv2.imread stalls ---
def _imread_fast(path: str) -> np.ndarray | None:
    """Read image using numpy+imdecode to avoid occasional cv2.imread stalls.
    Returns BGR image or None on failure."""
    try:
        # Quick sanity on path and size
        if not os.path.isfile(path):
            return None
        # Using fromfile tends to be faster and avoids locale-related hangs
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size == 0:
            return None
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

# --------------------------------------------------------------------------------------
# Optional: take a fixed fraction of the already-balanced pairs (preserve per-group/class)
# --------------------------------------------------------------------------------------
def downselect_fraction_per_group_class(pairs: Sequence[Tuple[str, int, str]],
                                        frac: float,
                                        shuffle: bool = True) -> List[Tuple[str, int, str]]:
    """
    From a list of (path,label,group) that is *already balanced per group/class*,
    take the same fraction from each (group, class) bucket, rounding down but keeping at least
    one item if the bucket was non-empty.
    """
    from collections import defaultdict
    rng = random
    buckets = defaultdict(lambda: {0: [], 1: []})
    for pth, lbl, grp in pairs:
        buckets[grp][lbl].append((pth, lbl, grp))

    selected: List[Tuple[str, int, str]] = []
    for grp, by_cls in buckets.items():
        for cls in (0, 1):
            items = by_cls[cls][:]
            if not items:
                continue
            if shuffle:
                rng.shuffle(items)
            k = max(1, int(math.floor(len(items) * float(frac))))
            selected.extend(items[:k])

    if shuffle:
        rng.shuffle(selected)
    return selected

def _iter_labelled_images(data_root: str) -> List[Tuple[str, int, str]]:
    """Return list of (image_path, label, group) where label=1 for VALID, 0 for INVALID.
    group is the immediate subfolder name under data_root (e.g., AO_set1).
    """
    triples: List[Tuple[str, int, str]] = []
    root = Path(data_root)
    subdirs = sorted([p for p in root.iterdir() if p.is_dir()])
    for s in subdirs:
        group = s.name
        vdir = s / "VALID"
        idir = s / "INVALID"
        for p in sorted(vdir.glob("*.png")):
            triples.append((str(p), 1, group))
        for p in sorted(idir.glob("*.png")):
            triples.append((str(p), 0, group))
    return triples


def build_feature_matrix(pairs: Sequence[Tuple[str, int, str]]) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Compute features for all images -> (X, y, feature_names, groups)."""
    feature_names: List[str] = []
    X_list: List[List[float]] = []
    y_list: List[int] = []
    groups: List[str] = []

    t0 = time.perf_counter()
    total = len(pairs)
    print(f"[feat] starting feature extraction for {total} images")
    sys.stdout.flush()

    from pathlib import Path
    for i, (path, label, group) in enumerate(pairs):
        img = _imread_fast(path)
        if img is None:
            print(f"[warn] could not read {path}")
            sys.stdout.flush()
            continue
        if i < 10:
            dt = time.perf_counter() - t0
            print(f"[feat] read {i+1}/{total}  +{dt:.2f}s  path={Path(path).name}")
            sys.stdout.flush()
        try:
            feats = extract_features(img)
        except Exception as e:
            print(f"[feat][skip] {Path(path).name} error={e}")
            sys.stdout.flush()
            continue
        if not feature_names:
            feature_names = list(feats.keys())
        X_list.append([feats[k] for k in feature_names])
        y_list.append(label)
        groups.append(group)
        if (i < 50 and (i+1) % 5 == 0) or ((i+1) % 25 == 0):
            dt = time.perf_counter() - t0
            ips = (i+1) / max(1e-6, dt)
            eta = (total - (i+1)) / max(1e-6, ips)
            print(f"[feat] {i+1}/{total}  avg {ips:.2f} img/s  ETA {eta/60:.1f} min")
            sys.stdout.flush()

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int32)
    return X, y, feature_names, groups



# --------------------------------------------------------------------------------------
# Simple KNN (no sklearn) + standardization + metrics + model I/O
# --------------------------------------------------------------------------------------

def _std_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return mu, sigma


def _std_apply(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (X - mu) / sigma


def _knn_predict_batch(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, k: int = 7, weights: str = "distance") -> np.ndarray:
    # Pairwise squared distances (n_te, n_tr)
    d2 = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(axis=2)
    if k >= Xtr.shape[0]:
        k = Xtr.shape[0]
    # Get k nearest neighbor indices (unsorted OK for summations)
    knn_idx = np.argpartition(d2, kth=k-1, axis=1)[:, :k]
    # Gather neighbor labels
    nn_labels = ytr[knn_idx]  # (n_te, k)
    if weights == "uniform":
        w = np.ones_like(nn_labels, dtype=np.float32)
    else:
        # distance weights: 1/(sqrt(d2)+eps)
        eps = 1e-8
        nn_d2 = np.take_along_axis(d2, knn_idx, axis=1)
        w = 1.0 / (np.sqrt(nn_d2) + eps)
    # Weighted vote for class 1 vs class 0
    w = w.astype(np.float32)
    sum1 = (w * nn_labels).sum(axis=1)
    sum0 = (w * (1 - nn_labels)).sum(axis=1)
    pred = (sum1 >= sum0).astype(np.int32)
    return pred


def _metrics_binary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = (2 * prec * rec) / max(1e-8, (prec + rec))
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def save_knn_model(path: str, feature_names: list[str], mu: np.ndarray, sigma: np.ndarray, Xtr: np.ndarray, ytr: np.ndarray, k: int, weights: str) -> None:
    np.savez(path, feature_names=np.array(feature_names, dtype=object), mu=mu, sigma=sigma, Xtr=Xtr, ytr=ytr, k=np.array(k), weights=np.array(weights))
    print(f"[save] model -> {path}")


def load_knn_model(path: str):
    d = np.load(path, allow_pickle=True)
    return {
        "feature_names": list(d["feature_names"].tolist()),
        "mu": d["mu"],
        "sigma": d["sigma"],
        "Xtr": d["Xtr"],
        "ytr": d["ytr"].astype(np.int32),
        "k": int(d["k"]),
        "weights": str(d["weights"]) if not isinstance(d["weights"], np.ndarray) else str(d["weights"].item()),
    }

# --------------------------------------------------------------------------------------
# Pickle model helpers and convenience prediction function
# --------------------------------------------------------------------------------------

def save_knn_model_pickle(path: str, feature_names: list[str], mu: np.ndarray, sigma: np.ndarray, Xtr: np.ndarray, ytr: np.ndarray, k: int, weights: str) -> None:
    payload = {
        "feature_names": list(feature_names),
        "mu": mu,
        "sigma": sigma,
        "Xtr": Xtr,
        "ytr": ytr.astype(np.int32),
        "k": int(k),
        "weights": str(weights),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[save] pickle model -> {path}")

def load_knn_model_pickle(path: str):
    with open(path, "rb") as f:
        d = pickle.load(f)
    # Ensure types are correct
    d["feature_names"] = list(d["feature_names"])
    d["ytr"] = np.asarray(d["ytr"], dtype=np.int32)
    d["k"] = int(d["k"])
    d["weights"] = str(d["weights"])
    return d

def predict_validity_from_bgr_with_pickle(img_bgr: np.ndarray, pkl_path: str = MODEL_PKL) -> int:
    """Classify a single BGR image as VALID(1) or INVALID(0) using the saved pickle model."""
    mdl = load_knn_model_pickle(pkl_path)
    feat_order = mdl["feature_names"]
    mu, sigma = mdl["mu"], mdl["sigma"]
    Xtr, ytr = mdl["Xtr"], mdl["ytr"]
    k, weights = mdl["k"], mdl["weights"]

    feats = extract_features(img_bgr)
    x = np.asarray([[feats[kf] for kf in feat_order]], dtype=np.float32)
    x_std = _std_apply(x, mu, sigma)
    yhat = int(_knn_predict_batch(Xtr, ytr, x_std, k=k, weights=weights)[0])
    return yhat

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Train/predict court validity (CV features + custom KNN; no sklearn)")
    ap.add_argument("--data-root", type=str, default=None, help="Override dataset root (else uses DATA_ROOT)")
    ap.add_argument("--model-out", type=str, default=None, help="Override model path (.npz); else uses MODEL_OUT")
    ap.add_argument("--predict", action="store_true", help="Prediction mode on a folder of frames")
    ap.add_argument("--frames-dir", type=str, default=None, help="Directory of frames (.png) to classify (prediction mode)")
    ap.add_argument("--pred-csv", type=str, default=None, help="Write predictions CSV (prediction mode)")
    ap.add_argument("--test-split", type=float, default=None, help="Override TEST_SPLIT for this run")
    ap.add_argument("--k", type=int, default=None, help="Override K_VALUE for this run")
    ap.add_argument("--weights", type=str, default=None, choices=["uniform","distance"], help="Override WEIGHTS for this run")
    ap.add_argument("--data-fraction", type=float, default=None, help="Override DATA_FRACTION for this run (0<frac<=1)")
    ap.add_argument("--debug-steps", action="store_true",
                    help="Save intermediate visualizations for a random VALID frame from each dataset subfolder, then exit")
    args = ap.parse_args()

    # Fallbacks so the script runs fine with no CLI flags
    data_root = args.data_root or DATA_ROOT
    model_out = args.model_out or MODEL_OUT
    holdout = TEST_SPLIT if (args.test_split is None) else float(args.test_split)
    k_val = int(args.k) if (args.k is not None) else int(K_VALUE)
    weights_val = (args.weights if args.weights is not None else WEIGHTS)

    data_fraction = float(DATA_FRACTION if args.data_fraction is None else args.data_fraction)
    if not (0.0 < data_fraction <= 1.0):
        print(f"[ERR] --data-fraction (or DATA_FRACTION) must be in (0,1]; got {data_fraction}")
        sys.exit(2)

    debug_steps = bool(args.debug_steps or DEBUG_STEPS)

    if args.data_root is None:
        print(f"[DEFAULT] DATA_ROOT -> {data_root}")
    if args.model_out is None and not args.predict:
        print(f"[DEFAULT] MODEL_OUT -> {model_out}")
    if args.k is None:
        print(f"[DEFAULT] K_VALUE -> {k_val}")
    if args.weights is None:
        print(f"[DEFAULT] WEIGHTS -> {weights_val}")
    if args.test_split is None:
        print(f"[DEFAULT] TEST_SPLIT -> {holdout}")
    if args.data_fraction is None:
        print(f"[DEFAULT] DATA_FRACTION -> {data_fraction}")
    if not args.debug_steps and DEBUG_STEPS:
        print(f"[DEFAULT] DEBUG_STEPS -> {debug_steps}")

    if args.predict:
        if not args.frames_dir or not model_out:
            print("[ERR] --predict needs --frames-dir and --model-out (path to .npz model)")
            sys.exit(2)
        use_pickle = str(model_out).lower().endswith(".pkl")
        mdl = load_knn_model_pickle(model_out) if use_pickle else load_knn_model(model_out)
        feat_order = mdl["feature_names"]
        mu, sigma = mdl["mu"], mdl["sigma"]
        Xtr, ytr = mdl["Xtr"], mdl["ytr"]
        k, weights = mdl["k"], mdl["weights"]
        frames = sorted(Path(args.frames_dir).glob("*.png"))
        rows = []
        for p in frames:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            feats = extract_features(img)
            x = np.asarray([[feats[kf] for kf in feat_order]], dtype=np.float32)
            x_std = _std_apply(x, mu, sigma)
            yhat = int(_knn_predict_batch(Xtr, ytr, x_std, k=k, weights=weights)[0])
            rows.append((p.name, yhat))
        if args.pred_csv:
            with open(args.pred_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["filename", "pred_valid"])
                for r in rows:
                    w.writerow(r)
            print(f"[pred] wrote {len(rows)} rows -> {args.pred_csv}")
        else:
            print("filename,pred_valid")
            for r in rows:
                print(f"{r[0]},{r[1]}")
        return

    # Train/eval path
    if not data_root:
        print("[ERR] --data-root is required for training")
        sys.exit(2)

    # Debug visualization path: save intermediates for one random VALID frame per group and exit
    if debug_steps:
        base_out = Path(model_out).parent if model_out else Path(data_root)
        out_dir = base_out / "debug_intermediates"
        n = _save_intermediates_for_groups(data_root, out_dir)
        if n == 0:
            print("[debug] no groups visualized (no VALID frames found).")
        else:
            print(f"[debug] saved intermediates for {n} group(s) -> {out_dir}")
        return

    pairs = _iter_labelled_images(data_root)
    # Enforce equal contribution per folder and per class (VALID/INVALID)
    pairs = filter_pairs_equal_per_group(pairs, per_class=True, shuffle=True)

    # If requested, further downselect to a fixed fraction to speed up feature extraction/training
    if data_fraction < 1.0:
        before = len(pairs)
        pairs = downselect_fraction_per_group_class(pairs, frac=data_fraction, shuffle=True)
        print(f"[fraction] using {len(pairs)}/{before} images (~{data_fraction*100:.1f}%)")

    if not pairs:
        print(f"[ERR] no labelled images found under {data_root}")
        sys.exit(2)
    print(f"[data] found {len(pairs)} labelled images")

    X, y, feat_names, groups = build_feature_matrix(pairs)
    print(f"[feat] X={X.shape}, positives(VALID)={int(y.sum())}, negatives(INVALID)={int((1-y).sum())}")

    if holdout and holdout > 0.0:
        print(f"[HOLD-OUT] Balanced per-folder & per-class split (test_size={holdout:.2f})")
        train_idx, test_idx, st = balanced_train_test_indices(groups, y, test_fraction=holdout)
        print("[HOLD-OUT] stats:", st)
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
    else:
        print("[TRAIN-ONLY] Using all data for training; reporting training-set metrics")
        Xtr, ytr = X, y
        Xte, yte = X, y

    mu, sigma = _std_fit(Xtr)
    Xtr_std = _std_apply(Xtr, mu, sigma)
    Xte_std = _std_apply(Xte, mu, sigma)

    ypred = _knn_predict_batch(Xtr_std, ytr, Xte_std, k=int(k_val), weights=str(weights_val))
    m = _metrics_binary(yte, ypred)
    print(f"[METRICS] acc={m['acc']*100:.2f}%  prec={m['prec']*100:.2f}%  rec={m['rec']*100:.2f}%  F1={m['f1']*100:.2f}%  N={len(yte)}")

    if model_out:
        Path(model_out).parent.mkdir(parents=True, exist_ok=True)
        save_knn_model(model_out, feat_names, mu, sigma, Xtr_std, ytr.astype(np.int32), int(k_val), str(weights_val))
        # Also save a pickle version in the same directory as this file for easy reuse across the project
        save_knn_model_pickle(MODEL_PKL, feat_names, mu, sigma, Xtr_std, ytr.astype(np.int32), int(k_val), str(weights_val))


if __name__ == "__main__":
    main()