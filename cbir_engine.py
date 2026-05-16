"""
CBIR Engine — Content-Based Image Retrieval
============================================
Extracts visual features (HSV color histogram + ORB texture descriptors)
and finds the most similar images using cosine similarity.

Author : Mounir Bekkar
"""

import cv2
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# ── CONFIGURATION ─────────────────────────────────────────────────────────────
HSV_BINS    = [8, 8, 8]          # bins per HSV channel → 512-dim histogram
ORB_FEATURES = 100               # max keypoints for ORB
INDEX_FILE  = "cbir_index.pkl"   # serialized index path


# ── FEATURE EXTRACTION ────────────────────────────────────────────────────────

def extract_hsv_histogram(image: np.ndarray) -> np.ndarray:
    """
    Compute a normalized HSV color histogram.
    Shape: (8 * 8 * 8,) = 512-dimensional vector.
    """
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [img_hsv],
        [0, 1, 2],
        None,
        HSV_BINS,
        [0, 180, 0, 256, 0, 256]
    )
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)


def extract_orb_descriptor(image: np.ndarray) -> np.ndarray:
    """
    Compute the mean ORB descriptor across detected keypoints.
    Shape: (32,) — mean of all keypoint descriptors.
    Returns a zero vector if no keypoints found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb  = cv2.ORB_create(nfeatures=ORB_FEATURES)
    kp, des = orb.detectAndCompute(gray, None)

    if des is not None and len(des) > 0:
        return des.mean(axis=0).astype(np.float32)  # shape: (32,)
    else:
        return np.zeros(32, dtype=np.float32)


def extract_features(img_path: str) -> np.ndarray:
    """
    Full feature vector for one image.
    Concatenates HSV histogram (512-dim) + ORB mean descriptor (32-dim).
    Total: 544-dimensional vector.
    """
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Cannot read image: {img_path}")

    hsv_feat = extract_hsv_histogram(image)   # 512-dim
    orb_feat = extract_orb_descriptor(image)  # 32-dim

    return np.concatenate([hsv_feat, orb_feat])  # 544-dim


# ── INDEX BUILDING ─────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def build_index(image_dir: str, save_path: str = INDEX_FILE) -> dict:
    """
    Build a feature index for all images in the given directory.

    Args:
        image_dir: path to folder containing images
        save_path: where to save the serialized index

    Returns:
        dict {image_path: feature_vector}
    """
    index    = {}
    img_dir  = Path(image_dir)
    img_list = [
        p for p in img_dir.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if len(img_list) == 0:
        raise FileNotFoundError(f"No images found in: {image_dir}")

    print(f"[INDEX] Found {len(img_list)} images in '{image_dir}'")

    for img_path in tqdm(img_list, desc="Extracting features"):
        try:
            feat = extract_features(str(img_path))
            index[str(img_path)] = feat
        except Exception as e:
            print(f"  [SKIP] {img_path.name}: {e}")

    # Save to disk
    with open(save_path, "wb") as f:
        pickle.dump(index, f)

    print(f"[INDEX] Saved {len(index)} entries → {save_path}")
    return index


def load_index(index_path: str = INDEX_FILE) -> dict:
    """Load a pre-built index from disk."""
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Index not found: {index_path}\n"
            "Run build_index() first."
        )
    with open(index_path, "rb") as f:
        index = pickle.load(f)
    print(f"[INDEX] Loaded {len(index)} entries from {index_path}")
    return index


# ── SEARCH ────────────────────────────────────────────────────────────────────

def find_similar(
    query_path: str,
    index: dict,
    top_k: int = 5,
    exclude_self: bool = True
) -> list[tuple[str, float]]:
    """
    Find the top-k most similar images to the query.

    Args:
        query_path:   path to the query image
        index:        {image_path: feature_vector} dict
        top_k:        number of results to return
        exclude_self: exclude the query image itself from results

    Returns:
        List of (image_path, similarity_score) sorted descending.
    """
    query_feat = extract_features(query_path).reshape(1, -1)

    paths  = list(index.keys())
    feats  = np.array(list(index.values()))

    # Cosine similarity: 1.0 = identical, 0.0 = completely different
    scores = cosine_similarity(query_feat, feats)[0]

    # Sort descending
    ranked = np.argsort(scores)[::-1]

    results = []
    for idx in ranked:
        path  = paths[idx]
        score = float(scores[idx])

        if exclude_self and os.path.abspath(path) == os.path.abspath(query_path):
            continue

        results.append((path, round(score, 4)))

        if len(results) >= top_k:
            break

    return results


# ── BATCH EVALUATION ──────────────────────────────────────────────────────────

def evaluate_precision_at_k(
    index: dict,
    k: int = 5,
    label_fn=None
) -> float:
    """
    Evaluate Precision@k assuming images share labels encoded in filenames.
    Example: cat_001.jpg, cat_002.jpg → same label 'cat'.

    Args:
        index:    feature index
        k:        number of results to evaluate
        label_fn: function(path) -> label string
                  defaults to: first part of filename before '_'

    Returns:
        Mean Precision@k across all queries.
    """
    if label_fn is None:
        label_fn = lambda p: Path(p).stem.split("_")[0]

    precisions = []

    for query_path in index:
        query_label = label_fn(query_path)
        results     = find_similar(query_path, index, top_k=k, exclude_self=True)

        relevant = sum(
            1 for path, _ in results
            if label_fn(path) == query_label
        )
        precisions.append(relevant / k)

    mean_p = float(np.mean(precisions)) if precisions else 0.0
    return mean_p
