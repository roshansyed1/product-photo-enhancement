"""
utils.py
Utility functions for the product photo enhancement pipeline.
"""

import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


def load_metadata(meta_csv_path):
    meta = pd.read_csv(meta_csv_path)
    print(f"Metadata loaded: {len(meta)} rows")
    return meta


def get_image_paths(meta, clean_dir, n=None, product_types=None):
    if product_types:
        meta = meta[meta['product_type'].isin(product_types)]
    paths = []
    for _, row in meta.iterrows():
        fname = os.path.basename(row['image_path'])
        full_path = os.path.join(clean_dir, fname)
        if os.path.exists(full_path):
            paths.append(full_path)
        if n and len(paths) >= n:
            break
    print(f"Found {len(paths)} valid image paths")
    return paths


def load_image(path, size=(512, 512)):
    img = Image.open(path).convert("RGB")
    img = img.resize(size, Image.LANCZOS)
    return img


def visualize_pairs(clean_imgs, degraded_imgs, n=4, save_path=None):
    fig, axes = plt.subplots(2, n, figsize=(5*n, 10))
    for i in range(n):
        axes[0][i].imshow(clean_imgs[i])
        axes[0][i].set_title(f"Clean {i+1}")
        axes[0][i].axis("off")
        axes[1][i].imshow(degraded_imgs[i])
        axes[1][i].set_title(f"Degraded {i+1}")
        axes[1][i].axis("off")
    plt.suptitle("Clean vs Degraded Pairs", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    plt.show()


def save_image(img_pil, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img_pil.save(path, quality=95)
