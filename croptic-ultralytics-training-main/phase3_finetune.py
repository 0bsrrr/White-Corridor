# ================================
# phase3_finetune.py — Post-Pruning Fine-Tuning
# ================================
# Jalankan: python phase3_finetune.py
# Input   : (edit BASE_MODEL + PRUNED_PTH di bawah)
# Output  : runs/yolov8m_pruned_finetuned/finetune/weights/best.pt
# ================================

# ---- CONFIG ----

BASE_MODEL         = "yolov8m.pt"                                        # model arsitektur untuk restore pruned
PRUNED_PTH         = "runs/modelopt_pruned/modelopt_pruned_model.pth"    # <-- output Phase 2
DATA               = "dataset/indrapuri_yolo/data.yaml"
POST_PRUNE_EPOCHS  = 20
IMG_SIZE           = 416
FINETUNE_PROJECT   = "runs/yolov8m_pruned_finetuned"
FINETUNE_NAME      = "finetune"

# ---- IMPORTS ----

import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
import ultralytics.data.build as build

# ---- DEVICE ----

if torch.cuda.is_available():
    DEVICE = "0"
    print(f"CUDA detected: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"
    print("CUDA not available. Using CPU.")

# ---- WEIGHTED DATASET ----

class YOLOWeightedDataset(YOLODataset):
    def __init__(self, *args, mode="train", **kwargs):
        super().__init__(*args, **kwargs)
        self.train_mode = "train" in self.prefix
        self.count_instances()
        class_weights = np.sum(self.counts) / self.counts
        self.class_weights = np.array(class_weights)
        self.agg_func = np.mean
        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()
        print("\nClass counts:", self.counts)
        print("Class weights:", self.class_weights)

    def count_instances(self):
        self.counts = [0 for _ in range(len(self.data["names"]))]
        for label in self.labels:
            cls = label["cls"].reshape(-1).astype(int)
            for idx in cls:
                self.counts[idx] += 1
        self.counts = np.array(self.counts)
        self.counts = np.where(self.counts == 0, 1, self.counts)

    def calculate_weights(self):
        weights = []
        for label in self.labels:
            cls = label["cls"].reshape(-1).astype(int)
            if cls.size == 0:
                weights.append(1)
                continue
            weights.append(self.agg_func(self.class_weights[cls]))
        return weights

    def calculate_probabilities(self):
        total = sum(self.weights)
        return [w / total for w in self.weights]

    def __getitem__(self, index):
        if not self.train_mode:
            return self.transforms(self.get_image_and_label(index))
        index = np.random.choice(len(self.labels), p=self.probabilities)
        return self.transforms(self.get_image_and_label(index))

build.YOLODataset = YOLOWeightedDataset

# ---- MAIN ----

print("\n" + "=" * 80)
print("Phase 3 - Post-Pruning Fine-Tuning")
print("=" * 80 + "\n")

print(f"Restoring pruned model from: {PRUNED_PTH}")

import modelopt.torch.opt as mto

base = YOLO(BASE_MODEL)
restored = mto.restore(base.model, PRUNED_PTH)
base.model = restored
model = base

print("Pruned model restored.")

model.train(
    data=DATA,
    epochs=POST_PRUNE_EPOCHS,
    imgsz=IMG_SIZE,
    device=DEVICE,
    project=FINETUNE_PROJECT,
    name=FINETUNE_NAME,
    exist_ok=True,
    seed=42,
)

print("\nFine-tuning completed.")
print(f"Output: {FINETUNE_PROJECT}/{FINETUNE_NAME}/weights/best.pt")
