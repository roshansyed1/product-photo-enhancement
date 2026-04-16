"""
degradation.py
Synthetic degradation pipeline for product photo enhancement.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO
import random


def add_gaussian_noise(img_np, severity=0.05):
    noise = np.random.normal(0, severity * 255, img_np.shape).astype(np.float32)
    return np.clip(img_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def add_blur(img_np, radius=None):
    if radius is None:
        radius = random.uniform(0.5, 1.5)
    return np.array(Image.fromarray(img_np).filter(ImageFilter.GaussianBlur(radius=radius)))


def add_jpeg_compression(img_np, quality=None):
    if quality is None:
        quality = random.randint(50, 75)
    img_pil = Image.fromarray(img_np)
    buffer = BytesIO()
    img_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.array(Image.open(buffer).copy())


def adjust_brightness(img_np, factor=None):
    if factor is None:
        factor = random.uniform(0.6, 1.4)
    return np.array(ImageEnhance.Brightness(Image.fromarray(img_np)).enhance(factor))


def adjust_color(img_np, factor=None):
    if factor is None:
        factor = random.uniform(0.6, 1.4)
    return np.array(ImageEnhance.Color(Image.fromarray(img_np)).enhance(factor))


def add_uneven_lighting(img_np):
    """Gradient lighting across the image — realistic, not a harsh stripe."""
    h, w = img_np.shape[:2]
    direction = random.choice(['left', 'right', 'top', 'bottom'])
    gradient = np.ones((h, w), dtype=np.float32)

    strength = random.uniform(0.4, 0.75)

    if direction == 'left':
        for col in range(w):
            gradient[:, col] = 1.0 - (1.0 - strength) * (col / w)
    elif direction == 'right':
        for col in range(w):
            gradient[:, col] = strength + (1.0 - strength) * (col / w)
    elif direction == 'top':
        for row in range(h):
            gradient[row, :] = 1.0 - (1.0 - strength) * (row / h)
    else:
        for row in range(h):
            gradient[row, :] = strength + (1.0 - strength) * (row / h)

    gradient = np.stack([gradient]*3, axis=-1)
    result = (img_np.astype(np.float32) * gradient)
    return np.clip(result, 0, 255).astype(np.uint8)


def add_background_clutter(img_np):
    h, w = img_np.shape[:2]
    cluttered = img_np.copy()
    for _ in range(random.randint(1, 3)):
        x1 = random.randint(0, w-1)
        y1 = random.randint(0, h-1)
        x2 = random.randint(x1, min(x1 + w//6, w-1))
        y2 = random.randint(y1, min(y1 + h//6, h-1))
        color = [random.randint(100, 200) for _ in range(3)]
        overlay = cluttered.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cluttered = cv2.addWeighted(overlay, 0.25, cluttered, 0.75, 0)
    return cluttered


def degrade_image(img_pil, severity="medium"):
    """
    Apply realistic degradations to simulate amateur product photos.
    severity: 'low', 'medium', 'high'
    """
    img_np = np.array(img_pil.convert("RGB"))

    if severity == "low":
        fns = random.sample([
            lambda x: add_gaussian_noise(x, 0.03),
            lambda x: add_blur(x, random.uniform(0.3, 0.8)),
            lambda x: adjust_brightness(x, random.uniform(0.8, 1.2)),
            lambda x: add_jpeg_compression(x, random.randint(70, 85)),
        ], k=random.randint(1, 2))

    elif severity == "medium":
        fns = random.sample([
            lambda x: add_gaussian_noise(x, 0.06),
            lambda x: add_blur(x, random.uniform(0.8, 1.5)),
            lambda x: add_jpeg_compression(x, random.randint(55, 70)),
            lambda x: adjust_brightness(x, random.uniform(0.65, 1.35)),
            lambda x: adjust_color(x, random.uniform(0.65, 1.35)),
            lambda x: add_uneven_lighting(x),
        ], k=random.randint(2, 3))

    else:  # high
        fns = [
            lambda x: add_gaussian_noise(x, 0.1),
            lambda x: add_blur(x, random.uniform(1.5, 2.5)),
            lambda x: add_jpeg_compression(x, random.randint(35, 55)),
            lambda x: add_uneven_lighting(x),
            lambda x: add_background_clutter(x),
        ]

    for fn in fns:
        img_np = fn(img_np)

    return Image.fromarray(img_np)
