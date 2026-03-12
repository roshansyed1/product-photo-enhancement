# Product Photo Enhancement using Deep Learning

## Overview
This project develops an automated pipeline for enhancing amateur product photographs into professional-quality images using generative deep learning techniques. The system addresses challenges faced by small businesses lacking access to professional photography equipment and expertise.

## Problem Statement
Amateur product photos suffer from poor lighting, cluttered backgrounds, inconsistent colors, and low resolution. This pipeline automates the enhancement process end-to-end.

## Technical Approach
Multi-stage enhancement pipeline:
1. Product segmentation and background removal (RMBG-2.0 / BiRefNet)
2. Lighting and color correction
3. Resolution enhancement (Real-ESRGAN)
4. Professional background generation (Stable Diffusion + ControlNet)

## Models
- Stable Diffusion v1.5 (base generation)
- ControlNet with edge/depth conditioning
- RMBG-2.0 (background removal)
- Real-ESRGAN (super resolution)

## Datasets
- InstructPix2Pix (timbrooks/instructpix2pix-clip-filtered) - 1K-5K subset
- Amazon Berkeley Objects (ABO) - 147K professional product images
- Products-10K - diverse product categories

## Evaluation Metrics
- FID (Frechet Inception Distance)
- CLIP Similarity Score
- LPIPS (Learned Perceptual Image Patch Similarity)
- PSNR / SSIM
- User study (50+ participants)

## Repository Structure
```
product-photo-enhancement/
├── data/scripts/        # Data download and preprocessing scripts
├── notebooks/           # Colab notebooks for experiments
├── src/                 # Source code modules
├── models/checkpoints/  # Saved model weights
├── experiments/         # Experiment configs and logs
├── results/             # Metrics and visualizations
├── docs/papers/         # Reference papers
└── demo/                # Gradio demo application
```

## Setup
```bash
pip install -r requirements.txt
```

## Course
Generative Deep Learning — Final Project (2026)

## Team
- Roshan Syed
- Abhigna
- (+ 2-3 team members)
