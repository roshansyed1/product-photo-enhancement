"""
train.py
Fine-tune InstructPix2Pix on product photo enhancement pairs using LoRA.
Optimized for T4 GPU (15GB VRAM).
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer
import torch.nn.functional as F
from tqdm import tqdm

# Paths
CLEAN_DIR  = "/content/drive/MyDrive/product_enhancement/data/pairs/clean/"
DEGRAD_DIR = "/content/drive/MyDrive/product_enhancement/data/pairs/degraded/"
OUTPUT_DIR = "/content/drive/MyDrive/product_enhancement/models/"
LOG_DIR    = "/content/project/experiments/"

# Hyperparameters
MODEL_ID      = "timbrooks/instruct-pix2pix"
LEARNING_RATE = 1e-4
BATCH_SIZE    = 1       # T4 safe
GRAD_ACCUM    = 4       # effective batch = 4
NUM_EPOCHS    = 3
IMG_SIZE      = 256     # 256 for T4, use 512 if A100
MAX_SAMPLES   = 5000    # subset for initial training
LORA_RANK     = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


class ProductPairDataset(Dataset):
    def __init__(self, clean_dir, degraded_dir, max_samples=None, img_size=256):
        self.clean_dir = clean_dir
        self.degraded_dir = degraded_dir
        self.img_size = img_size

        fnames = sorted(os.listdir(clean_dir))
        if max_samples:
            fnames = fnames[:max_samples]
        self.fnames = fnames

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        clean = Image.open(os.path.join(self.clean_dir, fname)).convert("RGB")
        degraded = Image.open(os.path.join(self.degraded_dir, fname)).convert("RGB")
        return {
            "clean": self.transform(clean),
            "degraded": self.transform(degraded),
            "prompt": "enhance this product photo to professional quality"
        }


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load pipeline components
    print("Loading model...")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        safety_checker=None
    )

    unet = pipe.unet
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    noise_scheduler = pipe.scheduler

    # Apply LoRA to UNet
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.1
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    unet = unet.to(device)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)

    # Dataset
    dataset = ProductPairDataset(CLEAN_DIR, DEGRAD_DIR,
                                  max_samples=MAX_SAMPLES, img_size=IMG_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=2)
    print(f"Dataset: {len(dataset)} pairs")

    optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    unet.train()
    global_step = 0
    losses = []

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            clean = batch["clean"].to(device, dtype=torch.float16)
            degraded = batch["degraded"].to(device, dtype=torch.float16)

            # Encode images to latents
            with torch.no_grad():
                clean_latents = vae.encode(clean).latent_dist.sample() * 0.18215
                degraded_latents = vae.encode(degraded).latent_dist.sample() * 0.18215

            # Add noise
            noise = torch.randn_like(clean_latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                      (clean_latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(clean_latents, noise, timesteps)

            # Encode prompt
            with torch.no_grad():
                prompt = batch["prompt"]
                tokens = tokenizer(prompt, return_tensors="pt",
                                   padding="max_length", max_length=77,
                                   truncation=True).input_ids.to(device)
                encoder_hidden_states = text_encoder(tokens)[0]

            # Concatenate degraded latents as conditioning
            concat_latents = torch.cat([noisy_latents, degraded_latents], dim=1)

            # Forward pass
            with torch.cuda.amp.autocast():
                noise_pred = unet(concat_latents, timesteps,
                                  encoder_hidden_states=encoder_hidden_states).sample
                loss = F.mse_loss(noise_pred, noise) / GRAD_ACCUM

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * GRAD_ACCUM
            losses.append(loss.item() * GRAD_ACCUM)

            if (step + 1) % GRAD_ACCUM == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

            if global_step % 100 == 0 and global_step > 0:
                print(f"Step {global_step} | Loss: {np.mean(losses[-100:]):.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(OUTPUT_DIR, f"lora_epoch_{epoch+1}")
        unet.save_pretrained(ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

    print("Training complete")
    return losses


if __name__ == "__main__":
    losses = train()
