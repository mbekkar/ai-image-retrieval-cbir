"""
Unit Tests — CBIR Engine
=========================
Run with: pytest tests.py -v

Author : Mounir Bekkar
"""

import os
import cv2
import numpy as np
import tempfile
import pickle
import pytest
from pathlib import Path

from cbir_engine import (
    extract_hsv_histogram,
    extract_orb_descriptor,
    extract_features,
    build_index,
    load_index,
    find_similar,
    INDEX_FILE,
)


# ── FIXTURES ──────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_image(tmp_path):
    """Create a 128×128 red JPEG image."""
    img  = np.zeros((128, 128, 3), dtype=np.uint8)
    img[:, :, 2] = 200  # red channel
    path = str(tmp_path / "red.jpg")
    cv2.imwrite(path, img)
    return path


@pytest.fixture
def dummy_dataset(tmp_path):
    """Create 3 color categories × 5 images."""
    colors = {"red": (0,0,200), "green": (0,200,0), "blue": (200,0,0)}
    for cat, bgr in colors.items():
        cat_dir = tmp_path / cat
        cat_dir.mkdir()
        for i in range(5):
            img  = np.full((64, 64, 3), bgr, dtype=np.uint8)
            noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
            img  = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(str(cat_dir / f"{cat}_{i}.jpg"), img)
    return str(tmp_path)


@pytest.fixture
def dummy_index(dummy_dataset, tmp_path):
    """Build index from dummy dataset."""
    idx_path = str(tmp_path / "test_index.pkl")
    idx = build_index(dummy_dataset, save_path=idx_path)
    return idx, idx_path


# ── FEATURE EXTRACTION TESTS ─────────────────────────────────────────────────

class TestHSVHistogram:
    def test_shape(self, dummy_image):
        img  = cv2.imread(dummy_image)
        hist = extract_hsv_histogram(img)
        assert hist.shape == (512,), f"Expected (512,), got {hist.shape}"

    def test_normalized(self, dummy_image):
        img  = cv2.imread(dummy_image)
        hist = extract_hsv_histogram(img)
        assert 0.0 <= hist.max() <= 1.0, "Histogram should be normalized [0,1]"

    def test_dtype(self, dummy_image):
        img  = cv2.imread(dummy_image)
        hist = extract_hsv_histogram(img)
        assert hist.dtype == np.float32

    def test_different_images_differ(self):
        red   = np.full((64, 64, 3), [0, 0, 200],   dtype=np.uint8)
        blue  = np.full((64, 64, 3), [200, 0, 0],   dtype=np.uint8)
        h_red  = extract_hsv_histogram(red)
        h_blue = extract_hsv_histogram(blue)
        # They should not be equal
        assert not np.allclose(h_red, h_blue), "Different color images should produce different histograms"


class TestORBDescriptor:
    def test_shape(self, dummy_image):
        img = cv2.imread(dummy_image)
        des = extract_orb_descriptor(img)
        assert des.shape == (32,), f"Expected (32,), got {des.shape}"

    def test_dtype(self, dummy_image):
        img = cv2.imread(dummy_image)
        des = extract_orb_descriptor(img)
        assert des.dtype == np.float32

    def test_uniform_image_returns_zeros(self):
        """A perfectly uniform image has no keypoints → zero descriptor."""
        img = np.full((64, 64, 3), [128, 128, 128], dtype=np.uint8)
        des = extract_orb_descriptor(img)
        assert des.shape == (32,)


class TestExtractFeatures:
    def test_shape(self, dummy_image):
        feat = extract_features(dummy_image)
        assert feat.shape == (544,), f"Expected (544,), got {feat.shape}"

    def test_invalid_path_raises(self):
        with pytest.raises(ValueError, match="Cannot read image"):
            extract_features("/nonexistent/path/image.jpg")

    def test_dtype(self, dummy_image):
        feat = extract_features(dummy_image)
        assert feat.dtype == np.float32


# ── INDEX TESTS ───────────────────────────────────────────────────────────────

class TestIndex:
    def test_build_returns_dict(self, dummy_dataset, tmp_path):
        idx = build_index(dummy_dataset, save_path=str(tmp_path / "idx.pkl"))
        assert isinstance(idx, dict)

    def test_build_count(self, dummy_dataset, tmp_path):
        idx = build_index(dummy_dataset, save_path=str(tmp_path / "idx.pkl"))
        assert len(idx) == 15, f"Expected 15 images (3×5), got {len(idx)}"

    def test_build_saves_file(self, dummy_dataset, tmp_path):
        save_path = str(tmp_path / "idx.pkl")
        build_index(dummy_dataset, save_path=save_path)
        assert os.path.exists(save_path)

    def test_load_matches_build(self, dummy_index):
        idx, idx_path = dummy_index
        loaded = load_index(idx_path)
        assert set(loaded.keys()) == set(idx.keys())

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_index(str(tmp_path / "missing.pkl"))

    def test_feature_vectors_shape(self, dummy_index):
        idx, _ = dummy_index
        for path, feat in idx.items():
            assert feat.shape == (544,), f"Bad shape for {path}: {feat.shape}"

    def test_empty_dir_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            build_index(str(empty), save_path=str(tmp_path / "idx.pkl"))


# ── SEARCH TESTS ──────────────────────────────────────────────────────────────

class TestSearch:
    def test_returns_list(self, dummy_image, dummy_index):
        idx, _ = dummy_index
        results = find_similar(dummy_image, idx, top_k=3)
        assert isinstance(results, list)

    def test_top_k_respected(self, dummy_image, dummy_index):
        idx, _ = dummy_index
        for k in [1, 3, 5]:
            results = find_similar(dummy_image, idx, top_k=k)
            assert len(results) <= k

    def test_scores_descending(self, dummy_image, dummy_index):
        idx, _ = dummy_index
        results = find_similar(dummy_image, idx, top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True), "Results should be sorted descending"

    def test_scores_in_range(self, dummy_image, dummy_index):
        idx, _ = dummy_index
        results = find_similar(dummy_image, idx, top_k=5)
        for _, score in results:
            assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

    def test_self_excluded(self, dummy_index):
        idx, _ = dummy_index
        query = list(idx.keys())[0]
        results = find_similar(query, idx, top_k=5, exclude_self=True)
        paths = [p for p, _ in results]
        assert os.path.abspath(query) not in [os.path.abspath(p) for p in paths]

    def test_same_color_ranks_first(self, dummy_dataset, dummy_index):
        """
        A red query image should have red images at the top.
        """
        idx, _ = dummy_index
        # Find a red image in the index
        red_path = next(p for p in idx.keys() if "red" in p.lower())
        results  = find_similar(red_path, idx, top_k=5, exclude_self=True)
        top_path = results[0][0]
        assert "red" in top_path.lower(), f"Expected red image at top, got: {top_path}"


# ── INTEGRATION TEST ──────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline(self, dummy_dataset, tmp_path):
        """End-to-end: build → save → load → search."""
        idx_path = str(tmp_path / "integration.pkl")

        # Build
        idx = build_index(dummy_dataset, save_path=idx_path)
        assert len(idx) == 15

        # Load
        loaded = load_index(idx_path)
        assert len(loaded) == 15

        # Search
        query = list(loaded.keys())[0]
        results = find_similar(query, loaded, top_k=3, exclude_self=True)
        assert len(results) == 3
        assert all(0.0 <= s <= 1.0 for _, s in results)
        assert results[0][1] >= results[-1][1]  # sorted descending


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
