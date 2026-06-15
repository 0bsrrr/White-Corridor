# ================================
# phase1_tune.py — Hyperparameter Tuning
# ================================
# Jalankan: python phase1_tune.py
# Output  : runs/yolov8m_tune_weighted/exp/weights/best.pt
# ================================

# ---- CONFIG ----

DATA       = "dataset/indrapuri_yolo/data.yaml"
MODEL      = "yolov8m.pt"
EPOCHS     = 500
ITERATIONS = 50
IMG_SIZE   = 416
TUNE_PROJECT = "runs/yolov8m_tune_weighted"
TUNE_NAME    = "exp"
SEARCH_SPACE = {
    "lr0":      (1e-6, 1e-2),
    "lrf":      (0.001, 0.3),
    "momentum": (0.6, 0.98),
}

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
print("Phase 1 - Genetic Algorithm Hyperparameter Tuning")
print("=" * 80 + "\n")

model = YOLO(MODEL)
print(f"Base model loaded: {MODEL}")

results = model.tune(
    data=DATA,
    epochs=EPOCHS,
    iterations=ITERATIONS,
    device=DEVICE,
    space=SEARCH_SPACE,
    project=TUNE_PROJECT,
    name=TUNE_NAME,
    exist_ok=True,
    seed=42,
)

print("\nTuning completed.")
print(results)
print(f"\nOutput: {TUNE_PROJECT}/{TUNE_NAME}/weights/best.pt")
