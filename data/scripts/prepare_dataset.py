"""
prepare_dataset.py
Creates clean/degraded training pairs from ABO dataset.
Saves pairs to Google Drive for training.
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

# Paths
META_CSV   = "/content/drive/MyDrive/product_enhancement/data/meta.csv"
LOCAL_DIR  = "/content/abo_local/images/images/"
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
    local_path = os.path.join(LOCAL_DIR, fname)
    if not os.path.exists(local_path):
        skipped += 1
        continue
    try:
        img = load_image(local_path, size=IMG_SIZE)
        degraded = degrade_image(img, severity=SEVERITY)
        stem = os.path.splitext(fname)[0]
        img.save(os.path.join(OUT_CLEAN, f"{stem}.jpg"), quality=95)
        degraded.save(os.path.join(OUT_DEGRAD, f"{stem}.jpg"), quality=95)
        count += 1
    except Exception as e:
        skipped += 1

print(f"Done: {count} pairs saved, {skipped} skipped")
