"""
Generate Sample Dataset
========================
Creates a small test dataset with synthetic colored images
so you can test the CBIR system without downloading a real dataset.

Creates 5 categories × 10 images = 50 images total.
Categories: red, green, blue, yellow, purple

Author : Mounir Bekkar
"""

import os
import cv2
import numpy as np
from pathlib import Path


CATEGORIES = {
    "red":    (0,   0,   200),
    "green":  (0,   180, 0  ),
    "blue":   (200, 0,   0  ),
    "yellow": (0,   200, 200),
    "purple": (180, 0,   180),
}

IMAGES_PER_CLASS = 10
IMG_SIZE = (128, 128)
OUTPUT_DIR = Path("dataset")


def make_synthetic_image(base_color: tuple, noise_level: int = 40) -> np.ndarray:
    """
    Create a 128×128 BGR image with a base color + random noise + geometric shapes.
    Makes each image slightly different to be realistic.
    """
    b, g, r = base_color

    # Base colored background
    img = np.full((*IMG_SIZE, 3), [b, g, r], dtype=np.uint8)

    # Add noise
    noise = np.random.randint(-noise_level, noise_level, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Random circles
    for _ in range(np.random.randint(2, 6)):
        cx = np.random.randint(10, IMG_SIZE[1] - 10)
        cy = np.random.randint(10, IMG_SIZE[0] - 10)
        rad = np.random.randint(8, 30)
        color = (
            np.clip(b + np.random.randint(-60, 60), 0, 255),
            np.clip(g + np.random.randint(-60, 60), 0, 255),
            np.clip(r + np.random.randint(-60, 60), 0, 255),
        )
        cv2.circle(img, (cx, cy), rad, color, -1)

    # Random rectangle
    x1, y1 = np.random.randint(0, 60, size=2)
    x2, y2 = x1 + np.random.randint(20, 60), y1 + np.random.randint(20, 60)
    rect_color = (
        np.clip(b - 40, 0, 255),
        np.clip(g - 40, 0, 255),
        np.clip(r - 40, 0, 255),
    )
    cv2.rectangle(img, (x1, y1), (x2, y2), rect_color, -1)

    return img


def generate_dataset(output_dir: Path = OUTPUT_DIR, n: int = IMAGES_PER_CLASS):
    """Generate synthetic images for all categories."""
    output_dir.mkdir(exist_ok=True)
    total = 0

    for category, base_color in CATEGORIES.items():
        cat_dir = output_dir / category
        cat_dir.mkdir(exist_ok=True)

        for i in range(n):
            img  = make_synthetic_image(base_color, noise_level=45)
            path = cat_dir / f"{category}_{i+1:03d}.jpg"
            cv2.imwrite(str(path), img)
            total += 1

        print(f"  ✓ {category:<10} — {n} images generated → {cat_dir}")

    print(f"\n[DONE] {total} images generated in '{output_dir}/'")
    print(f"       Structure: {len(CATEGORIES)} categories × {n} images\n")
    print("Now build the index:")
    print("  python search.py index --dir ./dataset\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic CBIR dataset")
    parser.add_argument("--out", default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--n",   type=int, default=IMAGES_PER_CLASS, help="Images per category")
    args = parser.parse_args()

    print(f"\n[DATASET] Generating synthetic images in '{args.out}' ...")
    generate_dataset(Path(args.out), args.n)
