"""
prepare_dataset.py
Creates clean/degraded training pairs from ABO dataset.
"""

import os
import sys
import random
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.append('/content/project/src')
from degradation import degrade_image
from utils import load_metadata, load_image

META_CSV   = "/content/drive/MyDrive/product_enhancement/data/meta.csv"
CLEAN_DIR  = "/content/drive/MyDrive/product_enhancement/data/clean/"
OUT_CLEAN  = "/content/drive/MyDrive/product_enhancement/data/pairs/clean/"
OUT_DEGRAD = "/content/drive/MyDrive/product_enhancement/data/pairs/degraded/"
N_SAMPLES  = 10000
IMG_SIZE   = (512, 512)
SEVERITY   = "medium"

os.makedirs(OUT_CLEAN, exist_ok=True)
os.makedirs(OUT_DEGRAD, exist_ok=True)

meta = load_metadata(META_CSV)
all_fnames = [os.path.basename(p) for p in meta['image_path'].tolist()]
random.shuffle(all_fnames)

print(f"Processing {N_SAMPLES} pairs...")
count, skipped = 0, 0

for fname in tqdm(all_fnames):
    if count >= N_SAMPLES:
        break
    clean_path = os.path.join(CLEAN_DIR, fname)
    if not os.path.exists(clean_path):
        skipped += 1
        continue
    try:
        img = load_image(clean_path, size=IMG_SIZE)
        degraded = degrade_image(img, severity=SEVERITY)
        stem = os.path.splitext(fname)[0]
        img.save(os.path.join(OUT_CLEAN, f"{stem}.jpg"), quality=95)
        degraded.save(os.path.join(OUT_DEGRAD, f"{stem}.jpg"), quality=95)
        count += 1
    except Exception:
        skipped += 1

print(f"Done: {count} pairs saved, {skipped} skipped")
