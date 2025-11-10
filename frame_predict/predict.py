import numpy as np
import cv2
import pickle
from pathlib import Path

# ------------------------------
# Self-contained helpers (model I/O, standardisation, KNN, feature extraction)
# ------------------------------
import math
from typing import Dict, List, Tuple

MODEL_PKL = "frame_predict/court_validity_knn.pkl"
IMAGE_PATH = "data/sample_frames/sci.png"

# Load KNN model from pickle file
def load_knn_model_pickle(path: str) -> Dict:
    with open(path, "rb") as f:
        d = pickle.load(f)
    # Normalize types
    d["feature_names"] = list(d.get("feature_names", []))
    d["mu"] = np.asarray(d["mu"], dtype=np.float32)
    d["sigma"] = np.asarray(d["sigma"], dtype=np.float32)
    d["Xtr"] = np.asarray(d["Xtr"], dtype=np.float32)
    d["ytr"] = np.asarray(d["ytr"], dtype=np.int32)
    d["k"] = int(d.get("k", 7))
    d["weights"] = str(d.get("weights", "distance"))
    return d

# Apply standardization using mean and std dev
def _std_apply(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32, copy=False)
    mu = mu.astype(np.float32, copy=False)
    sigma = sigma.astype(np.float32, copy=False)
    # Avoid divide-by-zero
    safe_sigma = np.where(np.abs(sigma) < 1e-8, 1.0, sigma)
    return (X - mu) / safe_sigma

# KNN prediction with Euclidean distance and weighted voting
def _knn_predict_batch(Xtr: np.ndarray, ytr: np.ndarray, X: np.ndarray, k: int = 7, weights: str = "distance") -> np.ndarray:
    # Vectorized KNN for batch prediction
    # Compute pairwise squared distances: (x - xtr)^2 = x^2 + xtr^2 - 2 x·xtr
    x2 = np.sum(X * X, axis=1, keepdims=True)              # [N,1]
    xt2 = np.sum(Xtr * Xtr, axis=1, keepdims=True).T       # [1,Ntr]
    d2 = x2 + xt2 - 2.0 * (X @ Xtr.T)                      # [N,Ntr]
    d2 = np.maximum(d2, 0.0)
    d = np.sqrt(d2 + 1e-12)                                # [N,Ntr]

    # Indices of k nearest neighbours
    k = int(k)
    k = max(1, min(k, Xtr.shape[0]))
    idx = np.argpartition(d, kth=k-1, axis=1)[:, :k]       # [N,k]
    # Gather distances and labels
    d_nn = np.take_along_axis(d, idx, axis=1)              # [N,k]
    y_nn = ytr[idx]                                        # [N,k]

    if str(weights).lower() == "distance":
        w = 1.0 / (d_nn + 1e-8)
    else:
        w = np.ones_like(d_nn, dtype=np.float32)

    # Weighted vote for binary classes {0,1}
    # Score for class 1
    score1 = np.sum(w * (y_nn == 1), axis=1)
    score0 = np.sum(w * (y_nn == 0), axis=1)
    yhat = (score1 >= score0).astype(np.int32)
    return yhat

_FEATURE_MISSING_WARNED = False

def extract_features(img_bgr: np.ndarray) -> Dict[str, float]:
    # Extract geometry and texture features for court validity classification
    img = _resize_max(img_bgr, 960)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) Net proxy (horiz band energy in central 40%)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    abs_sy = np.abs(sobel_y)
    total_sy = float(np.sum(abs_sy)) + 1e-6
    mid0, mid1 = int(0.3*h), int(0.7*h)
    mid_sy = float(np.sum(abs_sy[mid0:mid1, :]))
    horiz_band_energy = mid_sy / total_sy

    # 2) White-line mask & fraction (colour-invariant-ish)
    white = _white_line_mask(img)
    white_frac = float(cv2.countNonZero(white)) / float(white.size)

    # 3) Line segments (LSD only — simpler & robust)
    lines = _lsd_lines(gray, min_len_frac=0.10)
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


# Resize image to max width while maintaining aspect ratio
def _resize_max(img: np.ndarray, max_w: int = 960) -> np.ndarray:
    if img.shape[1] <= max_w:
        return img
    scale = max_w / img.shape[1]
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _white_line_mask(bgr: np.ndarray) -> np.ndarray:
    # Create binary mask for white court lines using HLS color space
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)
    # lightness high, saturation low (tolerate highlights)
    mask_l = (l >= 200).astype(np.uint8)
    mask_s = (s <= 80).astype(np.uint8)
    m = (mask_l & mask_s) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    return m



# Detect line segments using LSD
def _lsd_lines(gray_or_edges: np.ndarray, min_len_frac: float = 0.10) -> List[Tuple[int,int,int,int]]:
    # Extract line segments longer than min_len_frac of frame dimension
    if len(gray_or_edges.shape) == 3:
        gray = cv2.cvtColor(gray_or_edges, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_or_edges
    h, w = gray.shape[:2]
    min_len = int(min(w, h) * float(min_len_frac))

    # Create detector (some builds don't accept keyword args; some need no args)
    refine = getattr(cv2, 'LSD_REFINE_STD', 1)
    try:
        lsd = cv2.createLineSegmentDetector(refine)
    except TypeError:
        # Older OpenCV: no-arg constructor
        lsd = cv2.createLineSegmentDetector()

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
    # Compute orientation histogram and entropy for line segments
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

def _vectorise_features_from_names(feats: Dict[str, float], feature_names: List[str]) -> np.ndarray:
    # Convert feature dictionary to numpy array in specified order
    global _FEATURE_MISSING_WARNED
    x = np.zeros((1, len(feature_names)), dtype=np.float32)
    missing = []
    for i, k in enumerate(feature_names):
        if k in feats:
            x[0, i] = float(feats[k])
        else:
            x[0, i] = 0.0
            missing.append(k)
    if missing and not _FEATURE_MISSING_WARNED:
        print(f"[warn] extractor missing {len(missing)} feature(s) expected by model; filling zeros: {missing[:8]}{'...' if len(missing)>8 else ''}")
        _FEATURE_MISSING_WARNED = True
    return x


def predict_image_file_with_pickle(image_path: str, pkl_path: str = MODEL_PKL) -> bool:
    # Predict if image shows valid court view using KNN model
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    mdl = load_knn_model_pickle(pkl_path)
    feat_order = mdl["feature_names"]
    mu, sigma = mdl["mu"], mdl["sigma"]
    Xtr, ytr = mdl["Xtr"], mdl["ytr"]
    k, weights = mdl["k"], mdl["weights"]
    feats = extract_features(img)
    x = np.asarray([[feats[kf] for kf in feat_order]], dtype=np.float32)
    x_std = _std_apply(x, mu, sigma)
    yhat = int(_knn_predict_batch(Xtr, ytr, x_std, k=k, weights=weights)[0])
    return bool(yhat == 1)

if __name__ == "__main__":
    result = predict_image_file_with_pickle(IMAGE_PATH, MODEL_PKL)
    print("True (VALID)" if result else "False (INVALID)")