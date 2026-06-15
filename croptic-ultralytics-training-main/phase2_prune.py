# ================================
# phase2_prune.py — ModelOpt Structured Pruning
# ================================
# Jalankan: python phase2_prune.py
# Input   : (edit WEIGHT_PATH di bawah)
# Output  : runs/modelopt_pruned/modelopt_pruned_model.pth
# ================================

# ---- CONFIG ----

WEIGHT_PATH      = "runs/yolov8m_tune_weighted/exp/weights/best.pt"  # <-- ganti sesuai output Phase 1
IMG_SIZE         = 416
PRUNE_FLOPS      = 0.70
PRUNE_SAVE_DIR   = "runs/modelopt_pruned"

# ---- IMPORTS ----

import os
from pathlib import Path

import torch
from ultralytics import YOLO

# ---- MAIN ----

print("\n" + "=" * 80)
print("Phase 2 - NVIDIA ModelOpt Structured Pruning")
print("=" * 80 + "\n")

Path(PRUNE_SAVE_DIR).mkdir(parents=True, exist_ok=True)

print(f"Loading model: {WEIGHT_PATH}")
model = YOLO(WEIGHT_PATH)

import modelopt.torch.prune as mtp
import modelopt.torch.opt as mto

pytorch_model = model.model
pytorch_model.eval()

device = next(pytorch_model.parameters()).device
dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)

print(f"Model device       : {device}")
print(f"Dummy input shape  : {dummy_input.shape}")
print(f"Target FLOPs ratio : {PRUNE_FLOPS}")

pruned_model, prune_result = mtp.prune(
    model=pytorch_model,
    mode="fastnas",
    constraints={"flops": PRUNE_FLOPS},
    dummy_input=dummy_input,
)

model.model = pruned_model

save_path = os.path.join(PRUNE_SAVE_DIR, "modelopt_pruned_model.pth")
mto.save(pruned_model, save_path)

print(f"\nPruning completed.")
print(f"Output: {save_path}")

if prune_result is not None:
    print("Pruning result:")
    print(prune_result)
