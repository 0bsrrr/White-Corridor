from __future__ import annotations

import os
import glob
from pathlib import Path

import torch
import numpy as np

from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
import ultralytics.data.build as build


# ================================
# Weighted Dataset
# ================================

class YOLOWeightedDataset(YOLODataset):
    """
    Weighted sampling dataset for handling class imbalance.
    """

    def __init__(self, *args, mode="train", **kwargs):
        super(YOLOWeightedDataset, self).__init__(*args, **kwargs)

        self.train_mode = "train" in self.prefix

        # Count class instances
        self.count_instances()

        # Inverse frequency weighting
        class_weights = np.sum(self.counts) / self.counts
        self.class_weights = np.array(class_weights)

        # Aggregation function
        self.agg_func = np.mean

        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()

        print("\nClass counts:", self.counts)
        print("Class weights:", self.class_weights)

    # ----------------------------

    def count_instances(self):
        self.counts = [0 for _ in range(len(self.data["names"]))]

        for label in self.labels:
            cls = label["cls"].reshape(-1).astype(int)
            for idx in cls:
                self.counts[idx] += 1

        self.counts = np.array(self.counts)

        # Avoid division by zero
        self.counts = np.where(self.counts == 0, 1, self.counts)

    # ----------------------------

    def calculate_weights(self):
        weights = []

        for label in self.labels:
            cls = label["cls"].reshape(-1).astype(int)

            if cls.size == 0:
                weights.append(1)
                continue

            weight = self.agg_func(self.class_weights[cls])
            weights.append(weight)

        return weights

    # ----------------------------

    def calculate_probabilities(self):
        total_weight = sum(self.weights)
        return [w / total_weight for w in self.weights]

    # ----------------------------

    def __getitem__(self, index):
        if not self.train_mode:
            return self.transforms(self.get_image_and_label(index))

        index = np.random.choice(len(self.labels), p=self.probabilities)

        return self.transforms(self.get_image_and_label(index))


# ================================
# Monkey Patch Dataset
# ================================
# Ultralytics akan memakai dataset custom ini saat training/tuning.

build.YOLODataset = YOLOWeightedDataset


# ================================
# NVIDIA ModelOpt Pruning
# ================================

def prune_yolo_model(
    yolo_model: YOLO,
    target_flops: float = 0.70,
    imgsz: int = 416,
    save_dir: str = "runs/modelopt_pruned"
) -> YOLO:
    """
    Structured pruning menggunakan NVIDIA ModelOpt.

    Pruning ini dipakai untuk mengurangi FLOPs model agar inference
    menjadi lebih efisien.
    """

    print("\n" + "=" * 80)
    print("Phase 2 - NVIDIA ModelOpt Structured Pruning")
    print("=" * 80 + "\n")

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    try:
        import modelopt.torch.prune as mtp
        import modelopt.torch.opt as mto

    except ImportError as error:
        print("NVIDIA ModelOpt is not installed or cannot be imported.")
        print(f"Import error: {error}")
        print("Pruning skipped. Original model will be used.")
        return yolo_model

    try:
        pytorch_model = yolo_model.model
        pytorch_model.eval()

        device = next(pytorch_model.parameters()).device
        dummy_input = torch.randn(1, 3, imgsz, imgsz).to(device)

        print(f"Model device       : {device}")
        print(f"Dummy input shape  : {dummy_input.shape}")
        print(f"Target FLOPs ratio : {target_flops}")

        # FastNAS dipakai untuk structured pruning berbasis constraint FLOPs.
        pruned_model, prune_result = mtp.prune(
            model=pytorch_model,
            mode="fastnas",
            constraints={"flops": target_flops},
            dummy_input=dummy_input,
        )

        # Pasang kembali model hasil pruning ke wrapper YOLO.
        yolo_model.model = pruned_model

        # Simpan model hasil pruning versi ModelOpt.
        modelopt_save_path = os.path.join(save_dir, "modelopt_pruned_model.pth")
        mto.save(pruned_model, modelopt_save_path)

        print("ModelOpt pruning completed.")
        print(f"ModelOpt pruned model saved to: {modelopt_save_path}")

        if prune_result is not None:
            print("Pruning result:")
            print(prune_result)

        return yolo_model

    except Exception as error:
        print("ModelOpt pruning failed during runtime.")
        print(f"Error detail: {error}")
        print("Pruning skipped. Original model will be used.")
        return yolo_model


# ================================
# Quantization and Export
# ================================

def export_quantized_model(
    yolo_model: YOLO,
    data_yaml: str,
    imgsz: int = 416,
    formats: list[str] | None = None,
):
    """
    Export dan quantization model menggunakan Ultralytics.

    Format:
    - ONNX      : format umum untuk deployment lintas framework.
    - TensorRT : engine untuk inference cepat di NVIDIA GPU.
    - TFLite   : format untuk edge/mobile deployment.
    """

    print("\n" + "=" * 80)
    print("Phase 4 - Quantization and Model Export")
    print("=" * 80 + "\n")

    if formats is None:
        formats = ["onnx", "engine", "tflite"]

    for fmt in formats:
        print(f"\nExporting format: {fmt}")

        try:
            if fmt == "onnx":
                # ONNX dibuat sebagai format portable.
                # Quantization ONNX bisa dilakukan downstream jika dibutuhkan.
                output_file = yolo_model.export(
                    format="onnx",
                    imgsz=imgsz,
                    simplify=True,
                    opset=12,
                )

            elif fmt == "engine":
                # TensorRT engine untuk NVIDIA GPU.
                # INT8 aktif agar inference lebih cepat dan model lebih ringan.
                output_file = yolo_model.export(
                    format="engine",
                    imgsz=imgsz,
                    int8=True,
                    data=data_yaml,
                    simplify=True,
                    device=0,
                )

            elif fmt == "tflite":
                # TFLite INT8 untuk deployment edge/mobile.
                output_file = yolo_model.export(
                    format="tflite",
                    imgsz=imgsz,
                    int8=True,
                    data=data_yaml,
                )

            else:
                output_file = yolo_model.export(
                    format=fmt,
                    imgsz=imgsz,
                )

            print(f"Export success: {output_file}")

        except Exception as error:
            print(f"Export failed for format: {fmt}")
            print(f"Error detail: {error}")


# ================================
# MAIN
# ================================

if __name__ == "__main__":

    print("\n" + "=" * 80)
    print("Start YOLO Hyperparameter Tuning with Weighted Sampling, ModelOpt Pruning, and Quantization")
    print("=" * 80 + "\n")

    # ====================
    # CONFIG
    # ====================

    DATA = "dataset/indrapuri_yolo/data.yaml"
    MODEL = "yolov8m.pt"

    EPOCHS = 500
    ITERATIONS = 50
    IMG_SIZE = 416

    PRUNE_FLOPS_TARGET = 0.70
    POST_PRUNE_EPOCHS = 20

    ENABLE_TUNING = True
    ENABLE_PRUNING = True
    ENABLE_POST_PRUNE_FINETUNE = True
    ENABLE_EXPORT = True

    EXPORT_FORMATS = ["onnx", "engine", "tflite"]

    if torch.cuda.is_available():
        DEVICE = "0"
        print(f"CUDA detected: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = "cpu"
        print("CUDA not available. Using CPU.")

    # ====================
    # SEARCH SPACE
    # ====================

    search_space = {
        "lr0": (1e-6, 1e-2),
        "lrf": (0.001, 0.3),
        "momentum": (0.6, 0.98),
    }

    # ====================
    # MODEL
    # ====================

    print("\n" + "=" * 80)
    print("Load Base YOLO Model")
    print("=" * 80 + "\n")

    model = YOLO(MODEL)

    # ====================
    # PHASE 1: TUNING
    # ====================

    tune_project = "runs/yolov8m_tune_weighted" # gonta ganti
    tune_name = "exp"

    if ENABLE_TUNING:
        print("\n" + "=" * 80)
        print("Phase 1 - Genetic Algorithm Hyperparameter Tuning")
        print("=" * 80 + "\n")

        try:
            results = model.tune(
                data=DATA,
                epochs=EPOCHS,
                iterations=ITERATIONS,
                device=DEVICE,
                space=search_space,
                project=tune_project,
                name=tune_name,
                exist_ok=True,
                seed=42,
            )

            print("\nTuning completed.")
            print(results)

        except Exception as error:
            print("Tuning failed.")
            print(f"Error detail: {error}")
            print("Pipeline will continue using the current model.")

    else:
        print("Tuning skipped by config.")

    # ====================
    # LOAD BEST TUNED MODEL
    # ====================

    print("\n" + "=" * 80)
    print("Load Best Tuned Model")
    print("=" * 80 + "\n")

    pattern = os.path.join(tune_project, "**", "weights", "best.pt")
    best_tuned_candidates = glob.glob(pattern, recursive=True)

    if len(best_tuned_candidates) > 0:
        best_tuned_candidates = sorted(
            best_tuned_candidates,
            key=os.path.getmtime,
            reverse=True
        )
        best_tuned_weight = best_tuned_candidates[0]

        print(f"Best tuned weight found: {best_tuned_weight}")
        model = YOLO(best_tuned_weight)

    else:
        print("No tuned best.pt found. Using current model.")

    # ====================
    # PHASE 2: PRUNING
    # ====================

    if ENABLE_PRUNING:
        model = prune_yolo_model(
            yolo_model=model,
            target_flops=PRUNE_FLOPS_TARGET,
            imgsz=IMG_SIZE,
            save_dir="runs/modelopt_pruned",
        )
    else:
        print("Pruning skipped by config.")

    # ====================
    # PHASE 3: POST-PRUNING FINE-TUNING
    # ====================

    finetune_project = "runs/yolov8m_pruned_finetuned"
    finetune_name = "finetune"

    if ENABLE_POST_PRUNE_FINETUNE:
        print("\n" + "=" * 80)
        print("Phase 3 - Post-Pruning Fine-Tuning")
        print("=" * 80 + "\n")

        try:
            model.train(
                data=DATA,
                epochs=POST_PRUNE_EPOCHS,
                imgsz=IMG_SIZE,
                device=DEVICE,
                project=finetune_project,
                name=finetune_name,
                exist_ok=True,
                seed=42,
            )

            print("Post-pruning fine-tuning completed.")

        except Exception as error:
            print("Post-pruning fine-tuning failed.")
            print(f"Error detail: {error}")
            print("Pipeline will continue using the latest available model.")

    else:
        print("Post-pruning fine-tuning skipped by config.")

    # ====================
    # LOAD FINAL MODEL
    # ====================

    print("\n" + "=" * 80)
    print("Load Final Fine-Tuned Model")
    print("=" * 80 + "\n")

    pattern = os.path.join(finetune_project, "**", "weights", "best.pt")
    final_candidates = glob.glob(pattern, recursive=True)

    if len(final_candidates) > 0:
        final_candidates = sorted(
            final_candidates,
            key=os.path.getmtime,
            reverse=True
        )
        best_final_weight = final_candidates[0]

        print(f"Final best weight found: {best_final_weight}")
        model = YOLO(best_final_weight)

    else:
        print("No final best.pt found. Using current model.")

    # ====================
    # VALIDATION BEFORE EXPORT
    # ====================

    print("\n" + "=" * 80)
    print("Validation Before Export")
    print("=" * 80 + "\n")

    try:
        metrics = model.val(
            data=DATA,
            imgsz=IMG_SIZE,
            device=DEVICE,
        )

        print("Validation completed.")
        print(metrics)

    except Exception as error:
        print("Validation failed.")
        print(f"Error detail: {error}")
        print("Export will still be attempted.")

    # ====================
    # PHASE 4: QUANTIZATION / EXPORT
    # ====================

    if ENABLE_EXPORT:
        export_quantized_model(
            yolo_model=model,
            data_yaml=DATA,
            imgsz=IMG_SIZE,
            formats=EXPORT_FORMATS,
        )
    else:
        print("Export skipped by config.")

    print("\n" + "=" * 80)
    print("Pipeline Finished")
    print("=" * 80 + "\n")

    print("All enabled stages have finished.")