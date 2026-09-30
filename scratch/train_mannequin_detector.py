#!/usr/bin/env python3
"""
================================================================================
NIDAR SURVIVOR MANNEQUIN DEEP LEARNING DETECTOR TRAINER
================================================================================
Trains a lightweight, ultra-fast custom Convolutional Neural Network (CNN)
specifically taught to detect the Gazebo synthetic survivor mannequin:
  - Structure: Peach head + Yellow safety vest + Orange rescue torso
  - Scale & aspect ratio invariant (1.6 - 4.5)
  - Hard negative mining on arena walls, doorways, trim, floors, and shadows

Exports: config/nidar_survivor_custom_detector.onnx
================================================================================
"""

import os
import sys
import time
import math
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

repo_dir = "/home/dexter/Downloads/Drone-simulation-2-main"
config_dir = os.path.join(repo_dir, "config")
os.makedirs(config_dir, exist_ok=True)
onnx_output_path = os.path.join(config_dir, "nidar_survivor_custom_detector.onnx")


# ==============================================================================
# 1. SYNTHETIC TRAINING DATASET GENERATOR
# ==============================================================================
def render_synthetic_mannequin(size=(64, 64)):
    """
    Renders an augmented synthetic mannequin crop matching Gazebo material properties:
    - Body: Orange (diffuse: 1.0, 0.35, 0.0) -> BGR: [0, 89, 255]
    - Vest: Yellow (diffuse: 0.95, 0.95, 0.15) -> BGR: [38, 242, 242]
    - Head: Peach (diffuse: 0.85, 0.65, 0.50) -> BGR: [128, 166, 217]
    """
    img = np.full((size[0], size[1], 3), random.randint(120, 210), dtype=np.uint8) # Background wall

    # Add background noise/gradient
    noise = np.random.normal(0, 10, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Mannequin bounding parameters
    scale = random.uniform(0.55, 0.95)
    target_h = int(size[0] * scale)
    target_w = max(12, int(target_h / random.uniform(2.0, 3.8)))

    cx = size[1] // 2 + random.randint(-4, 4)
    bottom_y = size[0] - random.randint(2, 6)
    top_y = bottom_y - target_h

    x1 = max(2, cx - target_w // 2)
    x2 = min(size[1] - 2, cx + target_w // 2)
    w_act = x2 - x1

    # Lighting / color jitter
    b_jitter = random.uniform(0.85, 1.15)
    c_orange = np.clip([0 * b_jitter, 89 * b_jitter, 255 * b_jitter], 0, 255).astype(np.uint8).tolist()
    c_yellow = np.clip([38 * b_jitter, 242 * b_jitter, 242 * b_jitter], 0, 255).astype(np.uint8).tolist()
    c_peach = np.clip([128 * b_jitter, 166 * b_jitter, 217 * b_jitter], 0, 255).astype(np.uint8).tolist()

    # Split vertical heights: Torso (60%), Vest (25%), Head (15%)
    torso_h = int(target_h * 0.60)
    vest_h = int(target_h * 0.25)
    head_h = target_h - torso_h - vest_h

    # 1. Orange lower torso
    y_torso_top = bottom_y - torso_h
    cv2.rectangle(img, (x1, y_torso_top), (x2, bottom_y), c_orange, -1)

    # 2. Yellow safety vest
    y_vest_top = y_torso_top - vest_h
    x1_vest = max(1, x1 - 1)
    x2_vest = min(size[1] - 1, x2 + 1)
    cv2.rectangle(img, (x1_vest, y_vest_top), (x2_vest, y_torso_top), c_yellow, -1)

    # 3. Peach head sphere
    head_cy = y_vest_top - head_h // 2
    head_r = max(3, int(w_act * 0.38))
    cv2.circle(img, (cx, head_cy), head_r, c_peach, -1)

    # Random slight blur or noise
    if random.random() > 0.5:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


def render_synthetic_background(size=(64, 64)):
    """
    Renders hard negative samples from the indoor Gazebo compound:
    - Concrete walls (light gray)
    - Dark metal trims
    - Floor tiles
    - Doorway jambs
    - Shadows and vertical edge lines
    """
    mode = random.choice(['wall', 'trim', 'floor', 'doorway', 'shadow', 'noise'])
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)

    if mode == 'wall':
        val = random.randint(180, 235)
        img[:] = val
        noise = np.random.normal(0, 8, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    elif mode == 'trim':
        val = random.randint(180, 230)
        img[:] = val
        trim_y = random.randint(10, size[0] - 10)
        trim_c = random.randint(25, 45) # Dark gray metal
        cv2.rectangle(img, (0, trim_y - 4), (size[1], trim_y + 4), (trim_c, trim_c, trim_c), -1)

    elif mode == 'floor':
        val = random.randint(30, 60) # Dark floor tiles
        img[:] = val
        # Tile grid line
        grid_x = random.randint(0, size[1])
        cv2.line(img, (grid_x, 0), (grid_x, size[0]), (20, 20, 20), 1)

    elif mode == 'doorway':
        img[:] = random.randint(180, 220)
        split_x = random.randint(15, size[1] - 15)
        img[:, :split_x] = random.randint(40, 70) # Dark opening
        cv2.line(img, (split_x, 0), (split_x, size[0]), (30, 30, 30), 2) # Door frame post

    elif mode == 'shadow':
        img[:] = random.randint(140, 200)
        grad = np.linspace(1.0, 0.4, size[1])
        img = (img * grad[np.newaxis, :, np.newaxis]).astype(np.uint8)

    elif mode == 'noise':
        img = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)

    return img


class MannequinDataset(Dataset):
    def __init__(self, num_samples=3000):
        self.samples = []
        self.labels = []
        half = num_samples // 2

        print(f"Generating {half} positive mannequin samples...")
        for _ in range(half):
            img = render_synthetic_mannequin()
            self.samples.append(img)
            self.labels.append(1) # Class 1: Survivor

        print(f"Generating {half} hard negative background samples...")
        for _ in range(half):
            img = render_synthetic_background()
            self.samples.append(img)
            self.labels.append(0) # Class 0: Background

        # Shuffle
        combined = list(zip(self.samples, self.labels))
        random.shuffle(combined)
        self.samples, self.labels = zip(*combined)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_bgr = self.samples[idx]
        label = self.labels[idx]
        # Normalize to [0, 1] and format as CHW float32
        tensor = torch.from_numpy(img_bgr).permute(2, 0, 1).float() / 255.0
        return tensor, torch.tensor(label, dtype=torch.long)


# ==============================================================================
# 2. DEEP NEURAL NETWORK ARCHITECTURE
# ==============================================================================
class NidarSurvivorNet(nn.Module):
    """
    Lightweight, ultra-fast Convolutional Neural Network for Survivor Mannequin Detection.
    Inference latency: < 2ms on CPU.
    """
    def __init__(self):
        super(NidarSurvivorNet, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 64x64 -> 32x32
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 32x32 -> 16x16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 8x8 -> 1x1
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 2) # [background, survivor]
        )

    def forward(self, x):
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits


# ==============================================================================
# 3. TRAINING & ONNX EXPORT PIPELINE
# ==============================================================================
def train_and_export():
    print("=" * 70)
    print("      TRAINING NIDAR SURVIVOR MANNEQUIN DEEP LEARNING MODEL      ")
    print("=" * 70)

    dataset = MannequinDataset(num_samples=3200)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = NidarSurvivorNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)

    epochs = 12
    print(f"\nTraining NidarSurvivorNet on CPU for {epochs} epochs...")
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {train_loss/total:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

    total_time = time.time() - t0
    print(f"\nTraining completed in {total_time:.2f}s! Final Validation Accuracy: {val_acc*100:.2f}%")
    assert val_acc >= 0.98, f"Accuracy too low: {val_acc}"

    # Export to ONNX
    print(f"\nExporting model to ONNX format at {onnx_output_path}...")
    model.eval()
    dummy_input = torch.randn(1, 3, 64, 64, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_image'],
        output_names=['logits'],
        dynamic_axes={'input_image': {0: 'batch_size'}, 'logits': {0: 'batch_size'}}
    )

    print(f"✅ Successfully exported ONNX model ({os.path.getsize(onnx_output_path)/1024.0:.1f} KB)!")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    train_and_export()
