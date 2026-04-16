"""
degradation.py
Synthetic degradation pipeline for product photo enhancement.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO
import random
import os


def add_gaussian_noise(img_np, severity=0.1):
    noise = np.random.normal(0, severity * 255, img_np.shape).astype(np.float32)
    return np.clip(img_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def add_blur(img_np, radius=None):
    if radius is None:
        radius = random.uniform(0.5, 2.5)
    img_pil = Image.fromarray(img_np)
    return np.array(img_pil.filter(ImageFilter.GaussianBlur(radius=radius)))


def add_jpeg_compression(img_np, quality=None):
    if quality is None:
        quality = random.randint(30, 65)
    img_pil = Image.fromarray(img_np)
    buffer = BytesIO()
    img_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.array(Image.open(buffer).copy())


def adjust_brightness(img_np, factor=None):
    if factor is None:
        factor = random.uniform(0.4, 1.6)
    img_pil = Image.fromarray(img_np)
    return np.array(ImageEnhance.Brightness(img_pil).enhance(factor))


def adjust_color(img_np, factor=None):
    if factor is None:
        factor = random.uniform(0.5, 1.5)
    img_pil = Image.fromarray(img_np)
    return np.array(ImageEnhance.Color(img_pil).enhance(factor))


def add_shadow(img_np):
    h, w = img_np.shape[:2]
    shadow = img_np.copy().astype(np.float32)
    x1 = random.randint(0, w // 2)
    x2 = random.randint(w // 2, w)
    shadow[:, x1:x2] *= random.uniform(0.3, 0.6)
    return np.clip(shadow, 0, 255).astype(np.uint8)


def add_background_clutter(img_np):
    h, w = img_np.shape[:2]
    cluttered = img_np.copy()
    for _ in range(random.randint(2, 6)):
        x1, y1 = random.randint(0, w-1), random.randint(0, h-1)
        x2 = random.randint(x1, min(x1 + w//4, w-1))
        y2 = random.randint(y1, min(y1 + h//4, h-1))
        color = [random.randint(0, 255) for _ in range(3)]
        alpha = random.uniform(0.2, 0.5)
        overlay = cluttered.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cluttered = cv2.addWeighted(overlay, alpha, cluttered, 1-alpha, 0)
    return cluttered


def degrade_image(img_pil, severity="medium"):
    img_np = np.array(img_pil.convert("RGB"))

    if severity == "low":
        fns = random.sample([
            lambda x: add_gaussian_noise(x, 0.05),
            lambda x: add_blur(x, random.uniform(0.3, 1.0)),
            lambda x: adjust_brightness(x, random.uniform(0.7, 1.3)),
        ], k=random.randint(1, 2))

    elif severity == "medium":
        fns = random.sample([
            lambda x: add_gaussian_noise(x, 0.1),
            lambda x: add_blur(x),
            lambda x: add_jpeg_compression(x),
            lambda x: adjust_brightness(x),
            lambda x: adjust_color(x),
            lambda x: add_shadow(x),
        ], k=random.randint(2, 4))

    else:
        fns = [
            lambda x: add_gaussian_noise(x, 0.15),
            lambda x: add_blur(x, random.uniform(2.0, 4.0)),
            lambda x: add_jpeg_compression(x, random.randint(20, 45)),
            lambda x: adjust_brightness(x, random.uniform(0.3, 0.5)),
            lambda x: add_background_clutter(x),
        ]

    for fn in fns:
        img_np = fn(img_np)

    return Image.fromarray(img_np)
