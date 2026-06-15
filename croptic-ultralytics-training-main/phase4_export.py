# ================================
# phase4_export.py — Validasi dan Export
# ================================
# Jalankan: python phase4_export.py
# Input   : (edit WEIGHT_PATH di bawah)
# Output  : file .onnx / .engine / .tflite di samping WEIGHT_PATH
# ================================

# ---- CONFIG ----

WEIGHT_PATH     = "runs/yolov8m_pruned_finetuned/finetune/weights/best.pt"  # <-- output Phase 3
DATA            = "dataset/indrapuri_yolo/data.yaml"
IMG_SIZE        = 416
EXPORT_FORMATS  = ["onnx", "engine", "tflite"]   # hapus format yang tidak dibutuhkan

# ---- IMPORTS ----

import torch
from ultralytics import YOLO

# ---- DEVICE ----

if torch.cuda.is_available():
    DEVICE = "0"
    print(f"CUDA detected: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"
    print("CUDA not available. Using CPU.")

# ---- MAIN ----

print("\n" + "=" * 80)
print("Phase 4 - Validasi dan Model Export")
print("=" * 80 + "\n")

print(f"Loading model: {WEIGHT_PATH}")
model = YOLO(WEIGHT_PATH)

# ---- VALIDASI ----

print("\n" + "-" * 40)
print("Validasi")
print("-" * 40 + "\n")

metrics = model.val(
    data=DATA,
    imgsz=IMG_SIZE,
    device=DEVICE,
)

print("Validasi selesai.")
print(metrics)

# ---- EXPORT ----

print("\n" + "-" * 40)
print("Export")
print("-" * 40)

for fmt in EXPORT_FORMATS:
    print(f"\nExporting: {fmt.upper()}")

    if fmt == "onnx":
        output = model.export(
            format="onnx",
            imgsz=IMG_SIZE,
            simplify=True,
            opset=12,
        )

    elif fmt == "engine":
        output = model.export(
            format="engine",
            imgsz=IMG_SIZE,
            int8=True,
            data=DATA,
            simplify=True,
            device=0,
        )

    elif fmt == "tflite":
        output = model.export(
            format="tflite",
            imgsz=IMG_SIZE,
            int8=True,
            data=DATA,
        )

    else:
        output = model.export(format=fmt, imgsz=IMG_SIZE)

    print(f"Sukses: {output}")

print("\nPhase 4 selesai.")
